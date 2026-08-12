/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TBROADCAST<GridGroup>: a true concurrent MPSC
// broadcast collective.
//
// Payload and notification solve different collision problems:
//
//   * Every source owns the rank-indexed slot in each receiver's shared ring.
//     BcastSlotCount >= GroupMax makes simultaneous payload writes disjoint.
//   * Notification uses GridPipe's dedicated ready/free/close SPR triplet at
//     CollectiveChan, exactly like TPUSH.  Because every edge is MPSC, producers
//     atomically add ready and close at each receiver, and receivers atomically
//     add free at the producer.  Ordinary absolute sync_hscb stores are forbidden
//     here: concurrent last-writer-wins stores would lose credits and deadlock.
//   * One aggregate ready/close counter does not identify which source arrived.
//     A receiver therefore waits for the complete declared producer set before
//     draining any rank-indexed slot.  SetBcastExpectedProducerCount(K) declares
//     that set size (K=groupSize-1 for all-gather, K=1 for a single source).
//
// A producer waits until every receiver returned the prior round's free credit
// before reusing its rank-indexed slot.  Thus one payload slot per possible source
// is sufficient across any number of rounds, with no static L1 signal lanes.

#ifndef PTO_A2A3_GRID_TBROADCAST_HPP
#define PTO_A2A3_GRID_TBROADCAST_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by the demo's gridpipe_payload_inl.hpp (same
// pluggable payload hook contract as GridTPush.hpp / GridTPop.hpp).  Kept
// out-of-line so GridPipe is not tied to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride);
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot2D(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes,
    uint32_t rowCount, uint32_t producerStride, uint32_t dstStride, uint32_t tileStride);
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes);
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// ===========================================================================
// TBROADCAST send: broadcast THIS core's `tile` to every OTHER member of its
// group, writing into each receiver's shared ring at THIS source's prefix-offset
// slot, then atomically ringing each receiver's shared ready/close SPRs.  Safe to
// call from every group member concurrently -- that is the whole point.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TBROADCAST requires a GridPipe opted into broadcast (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TBROADCAST requires BcastSlotCount > 0");
    static_assert(
        Pipe::CollectiveChan < kGridChanCount,
        "TBROADCAST requires a dedicated GridPipe ready/free/close SPR triplet beyond the unicast channels");

    const int myRank = pto::RankInGroup(Group, pipe.coord, pipe.groupRect); // prefix-offset base, count_k = 1
    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(myRank); // this source's single global index
    constexpr int kCollectiveChan = Pipe::CollectiveChan;
    __gm__ uint32_t* localReady = pipe.readyScb[kCollectiveChan];

    // Payload sub-window inside the ring slot (see GridTPush.hpp).  Disabled =
    // whole slot, which is what this path always did.
    const GridPayloadWindow win = pipe.bcastWindow;
    const uint32_t payloadBytes = GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride));
    if (myRank < 0 || groupSize <= 0 || groupSize > Pipe::GroupMax ||
        payloadBytes > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    const uint32_t slotOff =
        (gidx % static_cast<uint32_t>(Pipe::BcastSlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) +
        win.entryOffset;

    // Before reusing this source's rank-indexed slot for another round, wait for
    // one free credit from every remote receiver of all prior rounds.
    const uint32_t freeThreshold = pipe.prodIndex[kCollectiveChan];
    __gm__ uint32_t* localFree = pipe.freeScb[kCollectiveChan];
    if (!wait_ipc_scb_sim(
            localFree, freeThreshold, static_cast<uint32_t>(kGridChanCount + kCollectiveChan), maxSpins)) {
        __gm__ uint32_t* freeFault = localFree ? localFree + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
        return false;
    }

    // Materialise the broadcast source in this pipe's isolated producer L1
    // range before addressing any receiver ring.  Real WSE has unified L1 SRAM,
    // not a separate vector UB mapping; the tile pointer below is retained only
    // as the A3 mock's DMA scratch.
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Phase 1: payload fan-out.  Each peer receives the tile directly in its OWN
    // shared ring at slot gidx%SC -- the disjoint prefix-offset assignment
    // guarantees no two sources write the same slot of the same receiver, so the
    // payloads never collide.
    //
    // The fan-out collapses to ONE copy_l1_to_group intrinsic (the COPY mode of
    // the group-collective CCE instruction in grid_cce_intrinsic.hpp).  That
    // instruction names the group by its two CORNER BLOCK IDS and addresses member
    // b's copy of the ring slot as myRingSlot + (b - selfBlockId)*blockStride, so a
    // multi-row SUBRECT is no longer a special case: the row-boundary jump is a jump
    // in block id, which the instruction does itself.  What it does need is that the
    // windows the resolver hands back be AFFINE in the block id.  They are in this
    // mock (the host lays the per-cell windows out contiguously), but measure it
    // rather than assume it -- a layout that is not affine takes the per-member
    // copy_l1_to_neighbor_l1 loop below.
    __gm__ uint8_t* myRingSlot = pipe.bcastRingBase + slotOff; // offset is identical in every window
    const pto::GridBlockRect group = pto::GridBlockRectOfGroup(Group, pipe.coord, pipe.shape, pipe.groupRect);
    const uint32_t selfBlockId = static_cast<uint32_t>(pto::BlockIdFromCoord(pipe.coord, pipe.shape));
    uint32_t blockStride = 0;
    bool uniformArena = (groupSize > 1);
    for (int m = 0; m < groupSize && uniformArena; ++m) {
        const uint32_t memberBlockId = pto::GridBlockRectMember(group, static_cast<uint32_t>(m));
        if (memberBlockId == selfBlockId) {
            continue; // our own copy is myRingSlot itself -- it measures nothing
        }
        __gm__ uint8_t* slotM =
            a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, static_cast<int>(memberBlockId));
        const int64_t gap = static_cast<int64_t>(reinterpret_cast<uint64_t>(slotM)) -
                            static_cast<int64_t>(reinterpret_cast<uint64_t>(myRingSlot));
        const int64_t hops = static_cast<int64_t>(memberBlockId) - static_cast<int64_t>(selfBlockId);
        const int64_t perBlock = (gap % hops == 0) ? (gap / hops) : 0;
        if (perBlock <= 0 || perBlock > static_cast<int64_t>(0xFFFFFFFFu)) {
            // Not affine in the block id, or a stride the uint32 operand cannot carry.
            uniformArena = false;
        } else if (blockStride == 0) {
            blockStride = static_cast<uint32_t>(perBlock);
        } else if (static_cast<int64_t>(blockStride) != perBlock) {
            uniformArena = false;
        }
    }
    uniformArena = uniformArena && (blockStride != 0);
    __ubuf__ void* transferScratch = a2a3_grid_payload::TileUbPtr<TileProd>(tile);
    auto* scratchBytes = reinterpret_cast<__ubuf__ uint8_t*>(transferScratch);
    // Normalised window: a disabled window is one row of the whole slot, so the
    // loops below cover both cases without branching on rowCount per row.
    const uint32_t rowCount = (win.rowCount == 0) ? 1u : win.rowCount;
    const uint32_t rowBytes = (win.rowCount == 0) ? static_cast<uint32_t>(Pipe::SlotStride) : win.rowBytes;
    const uint32_t tileRowStride = (win.rowCount == 0) ? 0u : GridPayloadTileStride(win);
    const uint32_t slotRowStride = (win.rowCount == 0) ? 0u : GridPayloadSlotStride(win);
    if (uniformArena) {
        // One intrinsic fans one ROW out to every member's copy of the slot.  This
        // includes this source's OWN copy (the member whose block id is ours); that
        // write is harmless -- a receiver never drains its own shard (srcRank ==
        // myRank is skipped), and the bytes written are this source's own shard
        // anyway.  copy_l1_to_group selects the broadcast (replicate-fan-out) NoC
        // mode, and selfBlockId names this core as the collective's SOURCE the same
        // way a combine names the caller as its sink.  The group operation moves
        // a CONTIGUOUS run per member, so a 2-D window costs one intrinsic per row --
        // still a single batched doorbell pass below.
        for (uint32_t r = 0; r < rowCount; ++r) {
            pto::copy_l1_to_group(
                reinterpret_cast<__gm__ const void*>(localProducerSlot + r * slotRowStride),
                reinterpret_cast<__gm__ void*>(myRingSlot + r * slotRowStride),
                reinterpret_cast<__ubuf__ void*>(scratchBytes + r * tileRowStride), rowBytes, blockStride, group,
                selfBlockId);
        }
    } else {
        // Fallback for a window layout the group instruction cannot address (not
        // affine in the block id) or a one-member group: one
        // copy_l1_to_neighbor_l1 per peer.
        for (int m = 0; m < groupSize; ++m) {
            if (m == myRank) {
                continue; // do not send to self; this core's own shard stays local.
            }
            const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
            __gm__ uint8_t* peerSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, peerBlockId);
            if (win.rowCount == 0) {
                a2a3_grid_payload::CopyProducerSramToNeighborSlot<TileProd>(
                    peerSlot, localProducerSlot, tile, Pipe::SlotStride);
            } else {
                a2a3_grid_payload::CopyProducerSramToNeighborSlot2D<TileProd>(
                    peerSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
                    GridPayloadSlotStride(win), GridPayloadTileStride(win));
            }
        }
    }

    // Single publish fence (data-before-ready, design doc C2) for the ENTIRE
    // multicast: every per-target MTE3 burst above must commit to the peers'
    // windows before any ready doorbell fires below.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Phase 2: MPSC ready fan-out.  Every concurrent source targets the same
    // ready SPR in a receiver, so each notification MUST be an atomic increment.
    for (int m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
        __gm__ uint32_t* peerReady = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, localReady, peerBlockId);
        atom_add_hscb(peerReady, 1);
    }

    // CLOSE is a separate aggregate proof that each producer completed its
    // round.  Keep it after READY even on a backend whose two atomic DMAs can be
    // independently scheduled.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    __gm__ uint32_t* localClose = pipe.closeScb[kCollectiveChan];
    for (int m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
        __gm__ uint32_t* peerClose = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, localClose, peerBlockId);
        atom_add_hscb(peerClose, 1);
    }

    pipe.prodIndex[kCollectiveChan] += static_cast<uint32_t>(groupSize - 1);
    pipe.PersistProdIndex(kCollectiveChan);
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, 0);
}

// ===========================================================================
// TBROADCAST receive (TPOP<GridGroup>): drain the shard that source `srcRank`
// broadcast into THIS core's shared ring.  The first drain of a round waits until
// the receiver's aggregate ready AND close SPRs contain the full declared source
// set, after which every rank-indexed slot in that set is safe to read.  Each
// drain atomically returns one free credit directly to that source.
//
// Callers must drain exactly `bcastExpectedProducerCount` distinct remote source
// ranks per round.  Source order is unrestricted because payload slots are
// disjoint; the FFN all-gather uses ascending rank for deterministic assembly.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(
    Pipe& pipe, TileCons& tile, int srcRank, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TPOP<GridGroup> requires a broadcast GridPipe (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TPOP<GridGroup> requires BcastSlotCount > 0");
    static_assert(
        Pipe::CollectiveChan < kGridChanCount,
        "TPOP<GridGroup> requires a dedicated GridPipe ready/free/close SPR triplet");

    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const int myRank = pto::RankInGroup(Group, pipe.coord, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(srcRank);
    constexpr int kCollectiveChan = Pipe::CollectiveChan;
    __gm__ uint32_t* localReady = pipe.readyScb[kCollectiveChan];
    const uint32_t producerCount = pipe.bcastExpectedProducerCount;
    if (groupSize <= 0 || groupSize > Pipe::GroupMax || srcRank < 0 || srcRank >= groupSize || srcRank == myRank ||
        producerCount == 0 || producerCount > static_cast<uint32_t>(groupSize - 1)) {
        __gm__ uint32_t* protocolFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(protocolFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    // consIndex counts drained remote shards.  All drains in the same round wait
    // for the same aggregate barrier; after the final drain, integer division
    // advances the threshold to the next round.
    const uint32_t roundBase = (pipe.consIndex[kCollectiveChan] / producerCount) * producerCount;
    const uint32_t readyThreshold = roundBase + producerCount;
    if (!wait_ipc_scb_sim(localReady, readyThreshold, static_cast<uint32_t>(kCollectiveChan), maxSpins)) {
        __gm__ uint32_t* readyFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    __gm__ uint32_t* localClose = pipe.closeScb[kCollectiveChan];
    if (!wait_ipc_scb_sim(
            localClose, readyThreshold, static_cast<uint32_t>(2 * kGridChanCount + kCollectiveChan), maxSpins)) {
        __gm__ uint32_t* closeFault = localClose ? localClose + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(closeFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Local read of this receiver's own ring slot (design doc: TPOP reads only
    // local SRAM -- the payload was pushed here, never read cross-core).  Same
    // payload window as the send half -- in a group collective both sides move
    // the same geometry, so one window describes both.
    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    const uint32_t slotOff =
        (gidx % static_cast<uint32_t>(Pipe::BcastSlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) +
        win.entryOffset;
    __gm__ uint8_t* localSlot = pipe.bcastRingBase + slotOff;
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyLocalSlotToTile2D<TileCons>(
            tile, localSlot, win.rowBytes, win.rowCount, GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // consume-before-free fence (design doc C3): the local read above must
    // complete before we tell the next occupant the slot is free.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Return one credit to the source whose slot was consumed.  Every receiver
    // targets that producer's same free SPR, so this edge is MPSC as well.
    const int producerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, srcRank, pipe.groupRect);
    __gm__ uint32_t* peerFree =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[kCollectiveChan], producerBlockId);
    atom_add_hscb(peerFree, 1);
    pipe.consIndex[kCollectiveChan] += 1;
    pipe.PersistConsIndex(kCollectiveChan);
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe& pipe, TileCons& tile, int srcRank)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcRank, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP

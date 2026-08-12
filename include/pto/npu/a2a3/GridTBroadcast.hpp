/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TBROADCAST<GridGroup> -- the 真·同时 MPSC
// broadcast collective (Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移).
//
// Problem the single-source TPUSH<GridSpan> broadcast could NOT solve: an
// AllGather where EVERY core broadcasts its own shard at once.  K concurrent
// senders all writing one shared per-receiver ready_scb would clobber its
// monotone absolute count (last-writer-wins loses K-1 updates) and deadlock.
// TBROADCAST breaks that fan-in by reusing the §4.2 scheme:
//
//   * SHARED RING per receiver, addressed by GLOBAL index gidx (slot = gidx%SC).
//   * PREFIX-OFFSET assignment: each member k owns a disjoint global-index
//     interval.  count_k is statically known (= 1 shard in the AllGather demo),
//     so base_k = k is computed locally under SPMD -- variant a, ZERO atomic,
//     no reservation round-trip.
//   * PER-SOURCE ready lanes (variant B): source k overwrites ONLY lane k with
//     gidx+1.  One writer per lane ⟹ every lane is SPSC ⟹ K concurrent senders
//     are correct.  (Variant A -- one shared atomic-add ready counter -- is the
//     documented alternative; it needs HW-DEP-A and cannot gate per-tile, so we
//     implement variant B.)
//   * FREE direction is X→K scatter: the single consumer X is the sole writer
//     of each free lane, so the free edge is SPSC per lane and the BROADCAST
//     min-credit tree (方案④ A1) is NOT needed (design doc §7.4).
//
// High-efficiency DIRECTED free notification: when X consumes global index c it
// need not wake every still-pending producer -- only the ONE producer that will
// next reuse the freed physical slot c%SC, namely owner(c + SC).  X computes
// that owner from the (statically known) interval layout and notifies exactly
// that one core, taking the free bandwidth O(K) → O(1) per consumed tile.
//
// Sender-side free backpressure: a producer about to write global index gidx
// waits free_lane[owner(gidx)] ≥ gidx - SC + 1 only when gidx ≥ SC (slot reuse).
// The single-shot AllGather (count_k = 1, SC ≥ group size) never reuses a slot,
// so both the producer free-wait and the consumer directed-notify are dormant
// there; the machinery is general and is exercised once SC < group size.

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
// slot, then ringing each receiver's per-source ready lane.  Safe to call from
// every group member concurrently -- that is the whole point.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TBROADCAST requires a GridPipe opted into broadcast (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TBROADCAST requires BcastSlotCount > 0");

    const int myRank = pto::RankInGroup(Group, pipe.coord, pipe.groupRect); // prefix-offset base, count_k = 1
    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(myRank); // this source's single global index

    // Payload sub-window inside the ring slot (see GridTPush.hpp).  Disabled =
    // whole slot, which is what this path always did.
    const GridPayloadWindow win = pipe.bcastWindow;
    const uint32_t payloadBytes = GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride));
    if (payloadBytes > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault =
            pipe.bcastReadyLanes ?
                pipe.bcastReadyLanes + myRank * grid_mock::kBcastLaneStrideU32 + grid_mock::kFaultFlagWordOffset :
                nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    const uint32_t slotOff =
        (gidx % static_cast<uint32_t>(Pipe::BcastSlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) +
        win.entryOffset;

    // Producer-side free backpressure (slot reuse only).  Dormant for the
    // single-shot AllGather (SC >= group size ⟹ threshold <= 0).  The producer
    // waits on its OWN free lane (indexed by its rank) for the consumer to have
    // freed the previous occupant of slot gidx%SC.
    if (gidx >= static_cast<uint32_t>(Pipe::BcastSlotCount)) {
        const uint32_t freeThreshold = gidx + 1 - static_cast<uint32_t>(Pipe::BcastSlotCount);
        __gm__ uint32_t* myFreeLane = pipe.bcastFreeLanes + myRank * grid_mock::kBcastLaneStrideU32;
        if (!wait_ipc_scb_sim(myFreeLane, freeThreshold, /*slot=*/0, maxSpins)) {
            __gm__ uint32_t* freeFault = myFreeLane ? myFreeLane + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
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

    // Phase 2: batched ready doorbells.  Variant B -- each receiver's ready lane
    // for THIS source (lane[myRank]) is overwritten with gidx+1 by exactly one
    // writer (us), so the per-lane edge stays SPSC regardless of how many other
    // sources are broadcasting concurrently.
    __gm__ uint32_t* myReadyLane = pipe.bcastReadyLanes + myRank * grid_mock::kBcastLaneStrideU32;
    const uint32_t readyValue = gidx + 1;
    for (int m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
        __gm__ uint32_t* peerReady = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, myReadyLane, peerBlockId);
        sync_hscb(peerReady, readyValue);
    }
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, 0);
}

// ===========================================================================
// TBROADCAST receive (TPOP<GridGroup>): drain the shard that source `srcRank`
// broadcast into THIS core's shared ring.  Waits this receiver's per-source ready
// lane[srcRank], copies slot srcRank%SC out, then issues the DIRECTED free
// notification -- the single consumer of this ring tells exactly the one
// producer that will next reuse the freed slot (owner(srcRank + SC)) that it may
// proceed, instead of broadcasting the free credit to every pending source.
//
// Callers MUST drain in ascending srcRank order so the directed-notification
// chain (free for index c unlocks the writer of index c + SC) advances in lock
// step with consumption.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(
    Pipe& pipe, TileCons& tile, int srcRank, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TPOP<GridGroup> requires a broadcast GridPipe (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TPOP<GridGroup> requires BcastSlotCount > 0");

    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(srcRank);
    const uint32_t readyThreshold = gidx + 1;

    // Wait for source srcRank's shard to land.  Per-source lane (variant B): the
    // only writer is the source of that rank, so this single wait is SPSC.
    __gm__ uint32_t* readyLane = pipe.bcastReadyLanes + srcRank * grid_mock::kBcastLaneStrideU32;
    if (!wait_ipc_scb_sim(readyLane, readyThreshold, /*slot=*/0, maxSpins)) {
        __gm__ uint32_t* readyFault = readyLane ? readyLane + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Local read of this receiver's own ring slot (design doc: TPOP reads only
    // local SRAM -- the payload was pushed here, never read cross-core).  Same
    // payload window as the send half -- in a group collective both sides move
    // the same geometry, so one window describes both.
    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = readyLane ? readyLane + grid_mock::kFaultFlagWordOffset : nullptr;
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

    // DIRECTED free notification.  The freed physical slot srcRank%SC will next
    // be reused by the producer that owns global index srcRank + SC.  Notify
    // exactly that one core (bandwidth O(1)/tile, not O(group)); dormant when
    // srcRank + SC >= groupSize (no reuse -- the single-shot case).
    const uint32_t nextGidx = gidx + static_cast<uint32_t>(Pipe::BcastSlotCount);
    if (nextGidx < static_cast<uint32_t>(groupSize)) {
        const int nextOwner = pto::GroupOwnerOfIndex(static_cast<int>(nextGidx));
        const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, nextOwner, pipe.groupRect);
        // The producer's free lane lives in ITS window at lane[nextOwner]; this
        // core is the sole writer of that lane (single consumer of this ring).
        __gm__ uint32_t* producerFreeLane = pipe.bcastFreeLanes + nextOwner * grid_mock::kBcastLaneStrideU32;
        __gm__ uint32_t* peerFree = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, producerFreeLane, peerBlockId);
        sync_hscb(peerFree, readyThreshold); // value = srcRank + 1 = nextOwner's free threshold
    }
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe& pipe, TileCons& tile, int srcRank)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcRank, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP

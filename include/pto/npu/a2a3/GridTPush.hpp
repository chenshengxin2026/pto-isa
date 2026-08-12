/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPUSH.
//
// TPUSH names the CONSUMER it is writing to -- an ordinary mesh rank -- and the
// channel comes from the pipe's binding table, not from a compass point in the
// template arguments.  Every transfer is exactly one hop; there is no direction and
// no distance operand anywhere in the family.
//
// Producer-side expansion calls the V8 CCE facades directly (V8 section 3.5.3
// TPUSH), with no intermediate PTO wrapper:
//   - wait_ipc_scb                 (WAIT_SPR on the local free_scb; read+block in one
//                                   instruction, no MOV_SPR2X peek -- V8)
//   - copy_l1_to_neighbor_l1       (COPY_L1_TO_NBR payload write, via the hook)
//   - sync_hscb                    (SYNC_HSCB store prod_idx -> peer ready_scb)
// Peer address resolution (ResolvePeerSlotAddr / RemoteScbPtr) is a plain runtime
// helper in the demo's gridpipe_payload_inl.hpp, not an intrinsic.
//
// payload transfer is intentionally pluggable: the tile->producer-L1 staging and
// producer-L1->neighbor-L1 adapters live alongside the demo kernel (they need the
// Tile type), while the generic handshake sequence stays here.

#ifndef PTO_A2A3_GRID_TPUSH_HPP
#define PTO_A2A3_GRID_TPUSH_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by demo's gridpipe_runtime adaptor.
// At the demo level we inject a concrete implementation that knows how to
// move a specific tile type to/from a mock SRAM slot via TSTORE/TLOAD. Keeping
// the hook out-of-line avoids tying GridPipe to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

// Resolve a local GM slot address to the same byte offset in peerBlockId's window
// (mock: the GM window standing in for peerBlockId's SRAM; native: mesh geometry).
AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);

// Resolve a local scoreboard word to peerBlockId's scoreboard word (sync_hscb dst).
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);

// Stage a tile in the pipe's isolated local producer L1 slot.
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride);

// Copy the staged local L1 source into the resolved peer receive slot.  `tile`
// is scratch only in the A3 GM mock; it is not the architectural source address.
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot2D(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes,
    uint32_t rowCount, uint32_t producerStride, uint32_t dstStride, uint32_t tileStride);

// Drain this core's local GM slot into the tile (V7 TPOP local read: the existing
// local copy; deliberately no cross-core read of payload).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes);

// 2-D form of the drain (slot is the source, tile the destination).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile2D(
    TileT& tile, __gm__ uint8_t* localSlot, uint32_t rowBytes, uint32_t rowCount, uint32_t slotStride,
    uint32_t tileStride);

// Mock-only read-locality guard: true iff [localSlot, +bytes) is inside
// callerBlockId's own GmSramArena segment (native: always local by construction).
AICORE bool PopSlotIsLocal(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, uint32_t bytes, int callerBlockId);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

namespace grid_detail {

// Fault sentinel for a scoreboard, null-safe: nullptr + offset is UB and would slip
// a non-null (but invalid) pointer past MockSetFault's null guard.
AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

// A binding whose downstream-consumer history overflowed has already lost the
// state required to reopen that consumer safely.  The flag is sticky, so whichever
// op runs first reports it.
template <typename Pipe>
AICORE inline bool ReportPendingBindFault(Pipe& pipe)
{
    if (pipe.consHistFull) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return true;
    }
    return false;
}

AICORE inline void GridPublishFence()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

// L1 has no WAIT_SPR equivalent.  Bind request/response payloads therefore use
// a bounded scalar poll in the mock and a normal blocking poll when maxSpins is
// zero.  The remote side writes a payload word first and a commit word last; the
// caller always polls the commit word.
AICORE inline bool WaitL1WordNotEqual(__gm__ uint32_t* word, uint32_t pending, uint32_t maxSpins, uint32_t& value)
{
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        value = mov_x_to_gpr(word);
        if (value != pending) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename Pipe>
AICORE inline int WaitBindableProducerChannel(Pipe& pipe, uint32_t consId, uint32_t maxSpins)
{
    int prodChan = kGridInvalidChan;
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (prodChan == kGridInvalidChan) {
        prodChan = pipe.PickBindableProducerChannel(consId);
        if (prodChan != kGridInvalidChan) {
            return prodChan;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(FaultWord(pipe.freeScb[0]), grid_mock::kFaultWaitProducerChannelTimeout);
            return kGridInvalidChan;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
    return prodChan;
}

// Producer chooses a local producer channel first.  The consumer returns an
// independent receive channel and commits the response by writing the dedicated
// completion L1 word last.
template <typename Pipe>
AICORE inline int EnsureOutgoingConsumerBinding(Pipe& pipe, uint32_t consId, uint32_t maxSpins)
{
    if (pipe.consumers.StateOf(consId) == GridConsumerState::ACTIVE) {
        const int prodChan = pipe.consumers.ProducerChannelOf(consId);
        const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
        if (prodChan >= 0 && prodChan < Pipe::ChanCount && peerConsChan >= 0 && peerConsChan < Pipe::ChanCount &&
            pipe.prodChanConsId[prodChan] == consId &&
            pipe.prodChanState[prodChan] == GridProducerChannelState::ACTIVE) {
            pipe.curProdChan = prodChan;
            pipe.consumers.curConsId = consId;
            return prodChan;
        }
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }

    if (pipe.consumers.FindOrAlloc(consId) < 0) {
        pipe.consHistFull = true;
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }

    const int prodChan = WaitBindableProducerChannel(pipe, consId, maxSpins);
    if (prodChan == kGridInvalidChan) {
        return prodChan;
    }

    grid_cce_detail::write_local_word(pipe.bindResponseReadyL1, kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseConsChanL1, kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseCompleteL1, kGridBindPending);
    GridPublishFence();

    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    __gm__ uint32_t* peerRequestProdChan =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestProdChanL1, static_cast<int>(consId));
    __gm__ uint32_t* peerRequestProdId =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestProdIdL1, static_cast<int>(consId));
    sync_hscb(peerRequestProdChan, static_cast<uint32_t>(prodChan) + 1u);
    GridPublishFence();
    sync_hscb(peerRequestProdId, GridRecPackId(static_cast<uint32_t>(selfBlockId)));

    uint32_t completeWord = kGridBindPending;
    if (!WaitL1WordNotEqual(pipe.bindResponseCompleteL1, kGridBindPending, maxSpins, completeWord)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
        return kGridInvalidChan;
    }
    if (completeWord != kGridBindHandshakeComplete) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }

    const uint32_t peerConsChanWord = mov_x_to_gpr(pipe.bindResponseConsChanL1);
    if (peerConsChanWord == 0 || peerConsChanWord > static_cast<uint32_t>(Pipe::ChanCount)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    const int peerConsChan = static_cast<int>(peerConsChanWord - 1u);

    pipe.prodIndex[prodChan] = mov_x_to_gpr(pipe.bindResponseReadyL1);
    pipe.PersistProdIndex(prodChan);
    if (!pipe.ActivateConsumer(consId, prodChan, peerConsChan)) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    return prodChan;
}

// Consumer half for one committed [producer id, producer channel] request.
template <typename Pipe>
AICORE inline int AcceptIncomingProducerBinding(Pipe& pipe, uint32_t prodId, int peerProdChan, uint32_t maxSpins)
{
    int consChan = pipe.ConsumerChannelOfProducer(prodId);
    if (consChan != kGridInvalidChan && !pipe.ConsumerChannelIsReusable(consChan)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[consChan]), grid_mock::kFaultBindChannelBusy);
        return kGridInvalidChan;
    }
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (consChan == kGridInvalidChan) {
        consChan = pipe.PickBindableConsumerChannel();
        if (consChan != kGridInvalidChan) {
            break;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultWaitBindableChannelTimeout);
            return kGridInvalidChan;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }

    const uint32_t readyBase = pipe.ReadConsumerReadyCount(consChan);
    const uint32_t freeBase = pipe.consIndex[consChan];
    pipe.consChanProdId[consChan] = prodId;
    pipe.consChanBindCnt[consChan] += 1;
    pipe.consChanCloseBase[consChan] = readyBase;
    pipe.consChanPeerProdChan[consChan] = peerProdChan;
    pipe.curConsChan = consChan;
    pipe.prevProdId = prodId;
    pipe.StoreRecord();

    grid_cce_detail::write_local_word(pipe.bindRequestProdChanL1, kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestProdIdL1, kGridBindPending);

    __gm__ uint32_t* peerReadyResponse =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseReadyL1, static_cast<int>(prodId));
    __gm__ uint32_t* peerFree =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerProdChan], static_cast<int>(prodId));
    __gm__ uint32_t* peerConsumerChannelResponse =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseConsChanL1, static_cast<int>(prodId));
    __gm__ uint32_t* peerCompleteResponse =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseCompleteL1, static_cast<int>(prodId));
    sync_hscb(peerReadyResponse, readyBase);
    sync_hscb(peerFree, freeBase);
    sync_hscb(peerConsumerChannelResponse, static_cast<uint32_t>(consChan) + 1u);
    GridPublishFence();
    sync_hscb(peerCompleteResponse, kGridBindHandshakeComplete);
    return consChan;
}

// TPOP calls this before resolving prodId.  Producer id is the request commit
// word; producer channel is payload written before it.
template <typename Pipe>
AICORE inline int EnsureIncomingProducerBinding(Pipe& pipe, uint32_t prodId, uint32_t maxSpins)
{
    while (true) {
        // Finish the expected producer's current turn before accepting a possibly
        // early request for the next turn.  CLOSE is sent after the final READY;
        // while consIndex is still below that final count, this TPOP must retain
        // the old peer/channel mapping so it can drain the last item correctly.
        const int existing = pipe.ConsumerChannelOfProducer(prodId);
        if (existing != kGridInvalidChan) {
            const uint32_t closeCount = pipe.ReadConsumerCloseCount(existing);
            if (closeCount <= pipe.consChanCloseBase[existing] || pipe.consIndex[existing] < closeCount) {
                return existing;
            }
        }

        const uint32_t requestWord = mov_x_to_gpr(pipe.bindRequestProdIdL1);
        if (requestWord != kGridBindPending) {
            const uint32_t requestProdId = GridRecUnpackId(requestWord);
            if (!GridBlockIdValid(requestProdId, pipe.shape)) {
                grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
                return kGridInvalidChan;
            }
            const uint32_t peerProdChanWord = mov_x_to_gpr(pipe.bindRequestProdChanL1);
            if (peerProdChanWord == 0 || peerProdChanWord > static_cast<uint32_t>(Pipe::ChanCount)) {
                grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
                return kGridInvalidChan;
            }
            const int accepted =
                AcceptIncomingProducerBinding(pipe, requestProdId, static_cast<int>(peerProdChanWord - 1u), maxSpins);
            if (accepted == kGridInvalidChan) {
                return accepted;
            }
            if (requestProdId == prodId) {
                return accepted;
            }
            continue;
        }

        uint32_t observed = kGridBindPending;
        if (!WaitL1WordNotEqual(pipe.bindRequestProdIdL1, kGridBindPending, maxSpins, observed)) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindRequestTimeout);
            return kGridInvalidChan;
        }
        // Re-read through the top of the loop so validation and acceptance stay
        // in one place.
    }
}

} // namespace grid_detail

// Push `tile` to the core whose LOGICAL BLOCK ID is `consId`.  The producer and
// consumer sides use independently negotiated channel indices.
//
// The call site derives `consId` from topology (GridPeerBlockIdForPush) or its
// schedule.  The first transfer to an UNBOUND/CLOSED consumer negotiates a channel;
// later ACTIVE transfers use that state-machine entry directly.
template <typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(
    Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false,
    uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "GridPipe TPUSH needs a pipe with at least one channel (ChanCount > 0)");

    if (grid_detail::ReportPendingBindFault(pipe)) {
        return false;
    }

    // Boundary check.  A boundary cell has no downstream, and the call site says so
    // by passing kGridNoPeer; anything else outside the mesh is a mis-derived rank.
    if (!GridBlockIdValid(consId, pipe.shape)) {
        grid_mock::MockBoundaryFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultPushOutOfMesh);
        return false;
    }

    // ACTIVE is the steady-state fast path.  UNBOUND/CLOSED negotiates independent
    // local producer and remote consumer channels.
    const int prodChan = grid_detail::EnsureOutgoingConsumerBinding(pipe, consId, maxSpins);
    if (prodChan == kGridInvalidChan) {
        return false;
    }
    const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
    if (peerConsChan < 0 || peerConsChan >= Pipe::ChanCount) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return false;
    }

    // Step 1 (V8 P1): wait for a free slot.  free threshold = prod_idx-SlotCount+1;
    //   WAIT_SPR alone reads the local free_scb and blocks (read+block in one
    //   instruction; no MOV_SPR2X peek -- V8).  The `prodIndex >= SlotCount` guard is
    //   exactly threshold > 0, so the first SlotCount pushes skip the wait (startup
    //   zero-block, V8 R6).  free_scb of channel c occupies IPC_SCB slot
    //   kGridChanCount+c.
    const uint32_t idx = pipe.prodIndex[prodChan];
    const uint32_t freeSlot = static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(prodChan);
    if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
        const uint32_t freeThreshold = idx + 1 - Pipe::SlotCount;
        if (!wait_ipc_scb_sim(pipe.freeScb[prodChan], freeThreshold, freeSlot, maxSpins)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2 (V7 P2): compute the local slot address from the producer GPR
    //   (slot_off = (prod_idx % SlotCount) * SlotStride); pure local scalar math.
    //   SlotStride addresses the ring; the payload window says what part of the
    //   slot this push actually moves (a5 TPipe: entryBase + entryOffset, with the
    //   length coming from the transfer descriptor rather than the slot size).
    const GridPayloadWindow win = pipe.pushWindow[prodChan];

    // Range guard (differs from a5: there the length is implied by the tile /
    // GlobalTensor descriptors and cannot exceed the slot; here it is a runtime
    // number, and an overrun writes into the PEER's window).
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultPushPayloadRange);
        return false;
    }

    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotStride + win.entryOffset;
    // `slotBase[peerConsChan]` is used only as an address template; resolving it
    // maps to that channel's receive ring in the peer's window.
    __gm__ uint8_t* localSlot = pipe.slotBase[peerConsChan] + slotOff;
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;

    // Step 2.5: materialise the outbound payload in the dedicated producer L1
    // range.  Source formula (one synchronous producer slot per pipe):
    //   producer = producerSlotBase + entryOffset + row*slotStride
    // Destination formula remains the selected receive-ring slot above.  The two
    // bases are disjoint by the window layout in grid_pipe_runtime.hpp.
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
    grid_detail::GridPublishFence();

    // Step 3 (V7 P3): payload transfer into the CONSUMER's SRAM/L1 slot region.
    //   The runtime helper resolves that rank's slot -- same byte offset, other
    //   window -- while the payload hook reads the isolated producer L1 slot and
    //   calls the copy_l1_to_neighbor_l1 CCE facade (COPY_L1_TO_NBR).
    const int peerBlockId = static_cast<int>(consId);
    __gm__ uint8_t* neighborSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, peerBlockId);
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyProducerSramToNeighborSlot<TileProd>(
            neighborSlot, localProducerSlot, tile, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyProducerSramToNeighborSlot2D<TileProd>(
            neighborSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
            GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // Publish fence (V7 P4, data-before-ready / R5). Orders the payload write
    // (MTE3 into the peer window) before the ready sync_hscb store below.  V7's
    // preferred form issues SYNC_HSCB(READY) from the payload's async pipe so it
    // *naturally* orders after the payload DMA (no explicit fence); this A2/A3
    // mock instead uses the conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
    // fallback (V7 3.4.1 grid_publish_fence).  Without it the scalar-pipe store
    // can become visible on the peer before the MTE3 slot bytes commit to DDR,
    // causing the consumer's read to pick up pre-publish (zero) data.
    grid_detail::GridPublishFence();

    // Step 4 (V7 P5): announce readiness -- sync_hscb (SYNC_HSCB) store of
    //   prod_idx (= idx+1) into the consumer's ready_scb for THIS CHANNEL INDEX
    //   (overwrite store of a monotone absolute count; single external writer per
    //   SPSC).  ready_scb of channel c occupies IPC_SCB slot c.
    //
    __gm__ uint32_t* neighborReady =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[peerConsChan], peerBlockId);
    sync_hscb(neighborReady, idx + 1);

    // Step 5 (V7 P5): bump the local producer GPR (drives slot addr / free
    //   threshold / the absolute count published to the consumer).
    pipe.prodIndex[prodChan] = idx + 1;
    pipe.PersistProdIndex(prodChan);

    if (isLastTransfer) {
        // CLOSE is ordered after payload + READY and carries the same final
        // absolute count.  The consumer compares it with the ready baseline it
        // captured at bind time, so no local SCB clear/reset is required.
        grid_detail::GridPublishFence();
        __gm__ uint32_t* neighborClose =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.closeScb[peerConsChan], peerBlockId);
        sync_hscb(neighborClose, idx + 1);
        if (!pipe.CloseConsumer(consId)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[peerConsChan]), grid_mock::kFaultBindProtocol);
            return false;
        }
    }
    return true;
}

template <typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false)
{
    (void)GRID_TRY_TPUSH_IMPL<Pipe, TileProd>(pipe, tile, consId, isLastTransfer, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP

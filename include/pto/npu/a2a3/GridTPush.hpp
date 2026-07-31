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
// The direction and hop distance are the PIPE's (Pipe::Dir / Pipe::Dist), not
// the call's: a pipe is bound to one (producer, consumer) pair, and pushing
// elsewhere means using another pipe (grid_intrinsic.hpp section 1).  Dist == 1
// is the original nearest-neighbor behavior.  Scheme A: a K-hop unicast keeps
// the receiver's per-channel slot/flag state at fan-in 1, so distance only
// changes the resolved target rank and the doorbell reach -- the window layout,
// slot ring, flag count and the TPOP read-locality guard are all unchanged.  See
// RankForPushK/CanPushK in grid_intrinsic.hpp and the design analysis 2026-06-02.
//
// Producer-side expansion calls the V8 CCE facades directly (V8 section 3.5.3
// TPUSH), with no intermediate PTO wrapper:
//   - wait_ipc_scb                 (WAIT_SPR on the local free_scb; read+block in one
//                                   instruction, no MOV_SPR2X peek -- V8)
//   - copy_ubuf_to_neighbor_ubuf   (COPY_UBUF_TO_NBR payload write, via the hook)
//   - sync_hscb                    (SYNC_HSCB store prod_idx -> peer ready_scb)
// Peer address resolution (ResolvePeerSlotAddr / RemoteScbPtr) is a plain runtime
// helper in the demo's gridpipe_payload_inl.hpp, not an intrinsic.
//
// payload transfer is intentionally pluggable: the tile->UB adapter that feeds
// copy_ubuf_to_neighbor_ubuf lives alongside the demo kernel (it needs the Tile
// type), while the generic handshake sequence stays here.

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

// Resolve a local GM slot address to the same byte offset in peerRank's window
// (mock: the GM window standing in for peerRank's SRAM; native: mesh geometry).
AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerRank);

// Resolve a local scoreboard word to peerRank's scoreboard word (sync_hscb dst).
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerRank);

// Push a tile into the resolved neighbor slot: extract the tile UB pointer, then
// call the copy_ubuf_to_neighbor_ubuf CCE facade (V7 COPY_UBUF_TO_NBR).
template <typename TileT>
__tf__ AICORE void CopyTileToNeighborSramSlot(__gm__ uint8_t* dstNeighborSlot, TileT& tile, int slotBytes);

// 2-D form, used when the caller set a GridPayloadWindow with rowCount > 0.
template <typename TileT>
__tf__ AICORE void CopyTileToNeighborSramSlot2D(
    __gm__ uint8_t* dstNeighborSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t slotStride);

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
// callerRank's own GmSramArena segment (native: always local by construction).
AICORE bool PopSlotIsLocal(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, uint32_t bytes, int callerRank);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// The channel is the pipe's: Dir/Dist come from Pipe, so a TPUSH always targets
// the consumer peer the pipe was declared for.  A pipe bound to SOURCE cannot
// push (SOURCE is a TPOP-only injection channel) and is rejected below; the
// static_assert in pto_instr.hpp catches it at the call site first.
template <typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(Pipe& pipe, TileProd& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    constexpr GridDirection Dir = Pipe::Dir;
    static_assert(Dir != GridDirection::SOURCE, "GridPipe TPUSH on a SOURCE-bound pipe is illegal (design doc 4.3)");

    // Boundary check. In production builds the compiler folds CanPushK() against
    // constexpr coord/Dist; here we keep the runtime check so dynamic
    // coordinates still trap.  Dist == 1 is the original nearest-neighbor path.
    if (!pipe.HasConsumer()) {
        grid_mock::MockBoundaryFault(pipe.cons.readyScb, grid_mock::PushFaultCode(Dir));
        return false;
    }

    // Step 1 (V8 P1): wait for a free slot.  free threshold = prod_idx-SlotCount+1;
    //   WAIT_SPR alone reads the local free_scb and blocks (read+block in one
    //   instruction; no MOV_SPR2X peek -- V8).  The `prodIndex >= SlotCount` guard is
    //   exactly threshold > 0, so the first SlotCount pushes skip the wait (startup
    //   zero-block, V8 R6).  This pipe's free_scb is IPC_SCB slot Pipe::FreeScbSlot.
    const uint32_t idx = pipe.prod.prodIndex;
    if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
        const uint32_t freeThreshold = idx + 1 - Pipe::SlotCount;
        if (!wait_ipc_scb_sim(pipe.prod.freeScb, freeThreshold, Pipe::FreeScbSlot, maxSpins)) {
            // Offset the fault-flag word only when the base scb pointer is real: nullptr + offset
            // is UB and would slip a non-null (but invalid) pointer past MockSetFault's null guard.
            __gm__ uint32_t* freeFault =
                pipe.prod.freeScb ? pipe.prod.freeScb + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2 (V7 P2): compute the local slot address from the producer GPR
    //   (slot_off = (prod_idx % SlotCount) * SlotStride); pure local scalar math.
    //   SlotStride addresses the ring; the producer's payload window says what part
    //   of the slot this push actually moves (a5 TPipe: entryBase + entryOffset,
    //   with the length coming from the transfer descriptor, not the slot size).
    const GridPayloadWindow win = pipe.prod.window;

    // Range guard (differs from a5: there the length is implied by the tile /
    // GlobalTensor descriptors and cannot exceed the slot; here it is a runtime
    // number, and an overrun writes into the PEER's window).
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = pipe.prod.freeScb ? pipe.prod.freeScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultPushPayloadRange);
        return false;
    }

    __gm__ uint8_t* localSlot = pipe.slots.Slot(idx) + win.entryOffset;

    // Step 3 (V7 P3): payload transfer to the *target's* SRAM/L1 slot region.
    //   For Dist > 1 this is the rank Dist hops away along Dir (a routed write);
    //   the data is delivered directly and does not land in intermediate cores.
    //   The runtime helper resolves the target rank's slot; the payload hook then
    //   extracts the tile UB pointer and calls the copy_ubuf_to_neighbor_ubuf CCE
    //   facade (V7 COPY_UBUF_TO_NBR).
    const int peerRank = pipe.ConsumerRank();
    __gm__ uint8_t* neighborSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.ctx.runtimeCtx, localSlot, peerRank);
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyTileToNeighborSramSlot<TileProd>(neighborSlot, tile, Pipe::SlotStride);
    } else {
        // push: tile is the source, slot the destination.
        a2a3_grid_payload::CopyTileToNeighborSramSlot2D<TileProd>(
            neighborSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win), GridPayloadSlotStride(win));
    }

    // Publish fence (V7 P4, data-before-ready / R5). Orders the payload write
    // (MTE3 into the peer window) before the ready sync_hscb store below.  V7's
    // preferred form issues SYNC_HSCB(READY) from the payload's async pipe so it
    // *naturally* orders after the payload DMA (no explicit fence); this A2/A3
    // mock instead uses the conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
    // fallback (V7 3.4.1 grid_publish_fence).  Without it the scalar-pipe store
    // can become visible on the peer before the MTE3 slot bytes commit to DDR,
    // causing the consumer's read to pick up pre-publish (zero) data.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Step 4 (V7 P5): announce readiness -- sync_hscb (SYNC_HSCB) store of
    //   prod_idx (= idx+1) into the consumer peer's ready_scb IPC_SCB (overwrite
    //   store of a monotone absolute count; single external writer per SPSC).  The
    //   peer declared the same pipe type at the same window offset, so its ready
    //   scb is ours resolved to its rank; natively it is IPC_SCB slot
    //   Pipe::ReadyScbSlot.
    //
    // Doorbell reach (Dist > 1): the A2/A3 mock routes the HSCB store to any rank
    // via RemoteScbPtr(peerRank), so K-hop works here as-is.  Native lowering
    // resolves the peer IPC_SCB address from (dir,dist) (V7 HW-DEP-1).
    __gm__ uint32_t* neighborReady = a2a3_grid_payload::RemoteScbPtr(pipe.ctx.runtimeCtx, pipe.cons.readyScb, peerRank);
    sync_hscb(neighborReady, idx + 1);

    // Step 5 (V7 P5): bump the local producer GPR (drives slot addr / free
    //   threshold / the absolute count published to the consumer peer).
    pipe.prod.prodIndex = idx + 1;
    return true;
}

template <typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe& pipe, TileProd& tile)
{
    (void)GRID_TRY_TPUSH_IMPL<Pipe, TileProd>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP

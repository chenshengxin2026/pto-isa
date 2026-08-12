/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TREDUCE<Op>: a fused "receive-combine-forward"
// reduce hop.  Builds on GridTPush.hpp / GridTPop.hpp -- see the V7 design spec
// section 5 (worked ReduceSum example), which frames the row reduce as the SAME
// single-hop SPSC handshake as AllGather, differing ONLY in the per-hop middle
// operation: AllGather relays the tile, ReduceSum folds it in with a combine
// before forwarding.  TREDUCE is that fused hop.
//
// Semantics per cell.  A hop names its two peers -- the producer it drains and the
// consumer it forwards to -- and kGridNoPeer for either half is what makes a cell a
// source or a sink.  No explicit "am I root" flag, and no direction: the caller
// derived both ids from the topology (GridPeerBlockIdForPop / GridPeerBlockIdForPush) and the
// mesh boundary already turned into a kGridNoPeer there.
//   * interior/sink (prodId names a core):
//         recv  <- TPOP(prodId)       (drain the transiting partial)
//         acc   <- combine(acc, recv) (fold in this cell's local contribution)
//   * source/interior (consId names a core):
//         TPUSH(acc, consId)          (forward the running reduction one hop)
//   * sink (consId == kGridNoPeer): acc holds the COMPLETE reduction; the caller
//     stores it.
//
// Along-the-path / on-transit compute (随路/过路计算): a fabric that can combine
// in the router collapses this whole receive-combine-forward body into ONE
// routed reduce-forward instruction -- `recv` and the in-core combine disappear
// into the transfer.  On A3 there is no such fabric and the adder is core-local,
// so TREDUCE lowers to exactly the local TPOP + combine + TPUSH sequence below.
// Nothing in the (Op, pipe, acc, recv, prodId, consId) signature changes between
// the two lowerings; only the body does.

#ifndef PTO_A2A3_GRID_TREDUCE_HPP
#define PTO_A2A3_GRID_TREDUCE_HPP

#include <cstdint>

#include <pto/comm/comm_types.hpp>         // pto::comm::ReduceOp (Sum/Max/Min) -- shared with the collective TREDUCE
#include <pto/npu/a2a3/GridTPop.hpp>       // GRID_TPOP_IMPL (receive half)
#include <pto/npu/a2a3/GridTPush.hpp>      // GRID_TPUSH_IMPL (forward half) + payload hooks
#include <pto/npu/a2a3/grid_intrinsic.hpp> // GridBlockIdValid
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

// Per-hop combine: fold the transiting partial `recv` into the accumulator `acc`
// with the reduce operator.  On A3 the adder lives inside the core, so this is an
// ordinary in-UB Vec op (TADD/TMAX/TMIN, resolved via ADL on the tile type); a
// future along-the-path-compute lowering performs this in the fabric during the
// transfer and drops the call entirely.  `Op` is a compile-time constant, so the
// branch folds away and only the selected instruction is instantiated.
template <pto::comm::ReduceOp Op, typename TileAcc, typename TileRecv>
AICORE inline void GridReduceCombine(TileAcc& acc, TileRecv& recv)
{
    if constexpr (Op == pto::comm::ReduceOp::Sum) {
        TADD(acc, acc, recv);
    } else if constexpr (Op == pto::comm::ReduceOp::Max) {
        TMAX(acc, acc, recv);
    } else {
        static_assert(Op == pto::comm::ReduceOp::Min, "GridPipe TREDUCE supports ReduceOp Sum/Max/Min only");
        TMIN(acc, acc, recv);
    }
}

// GridPipe TREDUCE<Op>: one fused reduce hop (see the file header).
// `acc` is in/out -- on entry the cell's local contribution, on return the
// running reduction up to and including this cell (at the sink, the complete
// result).  `recv` is the landing tile for the transiting partial (mandatory on
// A3's in-core adder; unused by a fabric that combines on transit).  Both tiles
// must share the reduce dtype/shape.  `prodId` / `consId` are the two peers of this
// hop, each kGridNoPeer where the chain ends.
//
// Fences use the same conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR) publish
// form as GridTPush.hpp (parse-safe on every target profile).  A full barrier
// subsumes the fine-grained MTE2->V->MTE3 crossings the hand-written kernel used:
//   * after the pop, before the combine: drain the MTE2 slot->recv copy (and the
//     caller's MTE2 producer of acc) so the Vec combine reads settled UB;
//   * after the combine / before the push: drain the V combine so the MTE3
//     payload copy reads the settled accumulator.
// The push half additionally carries its own data-before-ready publish fence
// inside GRID_TPUSH_IMPL.  Each half is gated on its peer id being a real core, so
// a boundary cell never enters the out-of-mesh fault path.
template <pto::comm::ReduceOp Op, typename Pipe, typename TileAcc, typename TileRecv>
AICORE void GRID_TREDUCE_IMPL(Pipe& pipe, TileAcc& acc, TileRecv& recv, uint32_t prodId, uint32_t consId)
{
    // Receive-and-combine half.  A source cell (no producer on this hop) has nothing
    // to drain and forwards its own contribution unchanged.
    if (GridBlockIdValid(prodId, pipe.shape)) {
        GRID_TPOP_IMPL<Pipe, TileRecv>(pipe, recv, prodId);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GridReduceCombine<Op, TileAcc, TileRecv>(acc, recv);
    }

    // Forward half.  A sink cell (no consumer on this hop) keeps the complete
    // reduction in `acc` for the caller to store.
    if (GridBlockIdValid(consId, pipe.shape)) {
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GRID_TPUSH_IMPL<Pipe, TileAcc>(pipe, acc, consId);
    }
}

// Forward declaration: TileUbPtr is provided by the demo's
// gridpipe_payload_inl.hpp (same pluggable payload-hook contract as the other
// a2a3_grid_payload helpers).  Kept out-of-line so this group-reduce facade
// stays tile-agnostic (it hands the mov_ubuf_group intrinsic raw UB ptrs).
namespace a2a3_grid_payload {
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);
} // namespace a2a3_grid_payload

// ===========================================================================
// GRID_TREDUCE_GROUP_IMPL: SPR-notified N->1 group fan-in.
//
// EVERY member calls this function with the same group and sinkBlockId.  A
// contributor's payload already lives at its symmetric `groupSlot`; it waits for
// the prior free credit, publishes that memory, then atomically increments the
// sink's ready and close SPRs.  The sink waits for all N-1 increments, performs
// one mov_ubuf_group reduction, and atomically returns one free credit to every
// contributor.  The dedicated triplet is
// readyScb/freeScb/closeScb[Pipe::CollectiveChan], i.e. the first fixed GridPipe
// SPR triplet beyond its active unicast channel pool.
//
// Atomic accumulation is essential at the sink: N-1 producers target the same
// ready/close words concurrently.  Backpressure prevents any contributor from
// publishing round r+1 before the sink has consumed its round-r contribution.
// The payload itself is symmetric per-block storage and therefore has one writer
// per block; only the notification path is MPSC.
// ===========================================================================
template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE bool GRID_TRY_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(
        Pipe::CollectiveChan < kGridChanCount,
        "group TREDUCE requires a dedicated GridPipe ready/free/close SPR triplet beyond the unicast channels");
    constexpr int kCollectiveChan = Pipe::CollectiveChan;
    const uint32_t selfBlockId = static_cast<uint32_t>(pto::BlockIdFromCoord(pipe.coord, pipe.shape));
    const uint32_t memberCount = pto::GridBlockRectSize(group);
    __gm__ uint32_t* localReady = pipe.readyScb[kCollectiveChan];

    if (memberCount == 0 || !pto::GridBlockRectContains(group, selfBlockId) ||
        !pto::GridBlockRectContains(group, sinkBlockId)) {
        __gm__ uint32_t* protocolFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(protocolFault, grid_mock::kFaultBindProtocol);
        return false;
    }

    if (selfBlockId != sinkBlockId) {
        // Contributor: one free credit per completed prior round.  Round zero
        // waits for threshold zero and therefore proceeds immediately.
        const uint32_t round = pipe.prodIndex[kCollectiveChan];
        __gm__ uint32_t* localFree = pipe.freeScb[kCollectiveChan];
        if (!wait_ipc_scb_sim(localFree, round, static_cast<uint32_t>(kGridChanCount + kCollectiveChan), maxSpins)) {
            __gm__ uint32_t* freeFault = localFree ? localFree + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }

        // The contribution was written before this call.  Publish it before the
        // ready increment so the sink cannot observe a doorbell before payload.
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        __gm__ uint32_t* sinkReady =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, localReady, static_cast<int>(sinkBlockId));
        atom_add_hscb(sinkReady, 1);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        __gm__ uint32_t* sinkClose = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.closeScb[kCollectiveChan], static_cast<int>(sinkBlockId));
        atom_add_hscb(sinkClose, 1);
        pipe.prodIndex[kCollectiveChan] = round + 1;
        pipe.PersistProdIndex(kCollectiveChan);
        return true;
    }

    // Sink: ready and close are aggregate counters.  Waiting for N-1 new values
    // on both proves that every contributor published and ended this round.
    const uint32_t peerCount = memberCount - 1;
    const uint32_t threshold = pipe.consIndex[kCollectiveChan] + peerCount;
    if (!wait_ipc_scb_sim(localReady, threshold, static_cast<uint32_t>(kCollectiveChan), maxSpins)) {
        __gm__ uint32_t* readyFault = localReady ? localReady + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    __gm__ uint32_t* localClose = pipe.closeScb[kCollectiveChan];
    if (!wait_ipc_scb_sim(
            localClose, threshold, static_cast<uint32_t>(2 * kGridChanCount + kCollectiveChan), maxSpins)) {
        __gm__ uint32_t* closeFault = localClose ? localClose + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(closeFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileAcc>(acc));
    __ubuf__ T* scr = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileScratch>(scratch));
    pto::mov_ubuf_group(
        reinterpret_cast<__ubuf__ void*>(dst), reinterpret_cast<__gm__ void*>(const_cast<__gm__ T*>(groupSlot)), bytes,
        blockStride, static_cast<pto::GridCollOp>(static_cast<uint32_t>(Op) + 1), static_cast<uint32_t>(sizeof(T)),
        group, sinkBlockId, reinterpret_cast<__ubuf__ void*>(scr));

    // Do not let a contributor overwrite its symmetric slot until the group read
    // above has retired.  Each target free SPR can itself have multiple writers
    // across overlapping collectives, so retain atomic-add semantics here too.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    for (uint32_t r = 0; r < memberCount; ++r) {
        const uint32_t peerBlockId = pto::GridBlockRectMember(group, r);
        if (peerBlockId == selfBlockId) {
            continue;
        }
        __gm__ uint32_t* peerFree = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.freeScb[kCollectiveChan], static_cast<int>(peerBlockId));
        atom_add_hscb(peerFree, 1);
    }
    pipe.consIndex[kCollectiveChan] = threshold;
    pipe.PersistConsIndex(kCollectiveChan);
    return true;
}

template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE void GRID_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0)
{
    (void)GRID_TRY_TREDUCE_GROUP_IMPL<Op, T, Pipe, TileAcc, TileScratch>(
        pipe, acc, scratch, groupSlot, bytes, group, sinkBlockId, blockStride, grid_mock::kDefaultWfeMaxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TREDUCE_HPP

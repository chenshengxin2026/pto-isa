/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe 接力计数 producer-handoff smoke kernel (THANDOFF).
//
// Pure data movement, Vector-only (no Cube / matmul).  Every cell owns ONE
// window -- one ring, one scoreboard pair -- and that single physical channel is
// driven by two different producers in turn:
//
//   phase 1   EAST  pipe: producer = west neighbour.  Each cell pushes
//             HANDOFF_P1_TILES tiles east; each receiver pops only
//             HANDOFF_P1_POPS of them, deliberately leaving the ring undrained.
//   THANDOFF  the consumer's producer changes west -> north.  Counters relay
//             (or rebase to 0 where the ring is provably drained -- column 0 has
//             no phase-1 producer at all, so it always rebases while columns to
//             its right relay whenever P1_POPS < P1_TILES).
//   phase 2   SOUTH pipe: producer = north neighbour.  Each receiver first drains
//             the phase-1 leftovers -- payload written by the RETIRED producer,
//             popped through the NEW pipe -- and then the new tiles, which the
//             relayed producer wrote at slot E % SlotCount, right behind them.
//
// The host verifies every popped tile by stamp, so a mis-relayed baseline shows
// up as the wrong cell's data rather than as a hang.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "gridpipe_payload_inl.hpp"
#include "handoff_smoke_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using SmokeTile = Tile<TileType::Vec, float, HANDOFF_T, HANDOFF_W, BLayout::RowMajor>;

// ONE physical channel, two producer bindings.  Both pipes must name the SAME
// IPC_SCB base -- the ready/free pair they share is a physical register pair, and
// the 2*Dir default would hand them different ones.  THANDOFF static_asserts it.
constexpr int kHandoffScb = 0;
using Phase1Pipe = GridPipe<SmokeTile, GridDirection::EAST, HANDOFF_SLOT_BYTES, HANDOFF_SLOT_COUNT, 1, kHandoffScb>;
using Phase2Pipe = GridPipe<SmokeTile, GridDirection::SOUTH, HANDOFF_SLOT_BYTES, HANDOFF_SLOT_COUNT, 1, kHandoffScb>;

// The phase-2 push must not block on free credit before the pop loop below it
// runs, so the whole two-phase run has to fit inside one ring pass.
static_assert(
    HANDOFF_P1_TILES + HANDOFF_P2_TILES <= HANDOFF_SLOT_COUNT,
    "ring must hold both phases without wrapping, or the phase-2 push blocks ahead of the phase-2 pop");

using ShapeTW = Shape<1, 1, 1, HANDOFF_T, HANDOFF_W>;
using StrideTW = Stride<HANDOFF_T * HANDOFF_W, HANDOFF_T * HANDOFF_W, HANDOFF_T * HANDOFF_W, HANDOFF_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
#endif

__global__ AICORE void HandoffSmokeKernel(
    __gm__ uint8_t* fftsAddr, __gm__ uint8_t* windows, __gm__ uint8_t* inBuf, __gm__ uint8_t* outBuf,
    __gm__ uint8_t* hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }

    if constexpr (DAV_VEC) {
        SmokeTile sendTile;
        SmokeTile recvTile;
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);

        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t* window = windows + blockIdx * HANDOFF_WINDOW_BYTES;

        // Both pipes are wired to the SAME window: the handoff hands over a
        // channel, it does not create a second one.
        Phase1Pipe eastPipe;
        Phase2Pipe southPipe;
        a2a3_grid::InitGridPipeFromWindow(
            eastPipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw), /*pipeId=*/0);
        a2a3_grid::InitGridPipeFromWindow(
            southPipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw), /*pipeId=*/1);

        __gm__ float* inBase =
            reinterpret_cast<__gm__ float*>(inBuf) + blockIdx * HANDOFF_IN_TILES * HANDOFF_TILE_ELEMS;
        __gm__ float* outBase =
            reinterpret_cast<__gm__ float*>(outBuf) + blockIdx * HANDOFF_OUT_TILES * HANDOFF_TILE_ELEMS;
        int popSlot = 0;

        // ------------------------------------------------------------------
        // Phase 1: push east, then pop only part of what arrived.
        // ------------------------------------------------------------------
        if (eastPipe.HasConsumer()) {
            for (int i = 0; i < HANDOFF_P1_TILES; ++i) {
                GSmoke inG(inBase + i * HANDOFF_TILE_ELEMS);
                TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
                TPUSH(eastPipe, sendTile);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }
        }

        const int p1Pops = eastPipe.HasProducer() ? HANDOFF_P1_POPS : 0;
        for (int i = 0; i < p1Pops; ++i) {
            TPOP(eastPipe, recvTile);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(outBase + popSlot * HANDOFF_TILE_ELEMS);
            TSTORE(outG, recvTile);
            ++popSlot;
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }

        // ------------------------------------------------------------------
        // 接力计数: hand the channel's producer role from the EAST binding to the
        // SOUTH one.  Every cell calls it; the halves it runs come from the
        // topology.  `retiredEnd` = the phase-1 tile count, which is what lets the
        // consumer PROVE the retiring producer is finished before reading E off
        // its ready_scb (cells without a phase-1 producer skip that proof).
        // ------------------------------------------------------------------
        THANDOFF(eastPipe, southPipe, static_cast<uint32_t>(HANDOFF_SEQ), static_cast<uint32_t>(HANDOFF_P1_TILES));
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // ------------------------------------------------------------------
        // Phase 2: push south from the relayed baseline, then drain the phase-1
        // leftovers followed by the new tiles -- both through the SOUTH pipe.
        // ------------------------------------------------------------------
        if (southPipe.HasConsumer()) {
            for (int i = 0; i < HANDOFF_P2_TILES; ++i) {
                GSmoke inG(inBase + (HANDOFF_P1_TILES + i) * HANDOFF_TILE_ELEMS);
                TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
                TPUSH(southPipe, sendTile);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }
        }

        // Leftovers first: they sit at ring indices [cons_idx, E), which is exactly
        // where the relayed cons_idx points, so an ordinary TPOP picks them up.
        const int leftovers =
            (southPipe.HasProducer() && eastPipe.HasProducer()) ? (HANDOFF_P1_TILES - HANDOFF_P1_POPS) : 0;
        const int p2Pops = southPipe.HasProducer() ? HANDOFF_P2_TILES : 0;
        for (int i = 0; i < leftovers + p2Pops; ++i) {
            TPOP(southPipe, recvTile);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(outBase + popSlot * HANDOFF_TILE_ELEMS);
            TSTORE(outG, recvTile);
            ++popSlot;
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
    }
#else
    (void)fftsAddr;
    (void)windows;
    (void)inBuf;
    (void)outBuf;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchHandoffSmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, int gridRows, int gridCols,
    void* stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    HandoffSmokeKernel<<<totalBlocks, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx, gridRows, gridCols);
}

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// Single-device multi-block FFN, ReduceSum combine via the pto::comm collective
// API (pto::comm::TREDUCE) instead of the GridPipe mock.
//
// Two kernels, launched back-to-back with a host stream-sync barrier between
// them (the collective is root-only and assumes every rank's window slot is
// already written; there is no cheap in-kernel cross-block barrier):
//
//   1. Compute kernel (gridRows*gridCols blocks, mixed Cube/Vec): each cell
//      computes its model-parallel FFN shard partial downPartial[T,H] (fp32)
//      and stores it into its own HCCL window slot.
//   2. Collective kernel (gridRows blocks, Vec): block `row` is the root of the
//      row's column group.  It builds a ParallelGroup over the gridCols window
//      slots of its row and runs pto::comm::TREDUCE(Sum) -> y_row[T,H].
//
// The single-device mock uses a fake HcclDeviceContext where
//   windowsIn[i] = base + i * FFN_GRID_WINDOW_BYTES  and rankId = 0.
// HcclRemotePtr(ctx, win0, pe) then resolves to base + pe*FFN_GRID_WINDOW_BYTES,
// so the per-cell window slots are reduced across the same device.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/fifo.hpp>
// pto::comm::TREDUCE/TGATHER are pulled in transitively by <pto/pto-inst.hpp>
// (pto_instr.hpp includes pto_comm_inst.hpp); only the comm types are included
// directly here.  Including pto_comm_inst.hpp again re-processes event.hpp out
// of order and breaks PIPE_FIX resolution.
#include <pto/comm/comm_types.hpp>

#include "common.hpp"
#include "ffn_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using DownF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_TILE, BLayout::RowMajor>;

// TREDUCE staging tiles: dynamic valid dims, mirroring the comm ST test usage.
using ReduceTile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_TILE, BLayout::RowMajor, -1, -1>;

using ShapeTH = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_TILE>;
using StrideTH = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                        FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHF32 = GlobalTensor<float, ShapeTH, StrideTH, Layout::ND>;

constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = 0x1000;
constexpr int kUbHiddenF32 = 0x2000;
constexpr int kUbHiddenF16 = 0x3000;
constexpr int kUbDownF32 = 0x4000;

constexpr int kL1X = 0x00000;
constexpr int kL1Hidden = 0x10000;
constexpr int kL1WGate = 0x20000;
constexpr int kL1WUp = 0x24000;
constexpr int kL1WDown = 0x28000;

// TREDUCE acc/recv UB addresses (collective kernel).  Distinct, non-overlapping.
constexpr int kUbReduceAcc = 0x0000;
constexpr int kUbReduceRecv = 0x10000;

#endif

// ===========================================================================
// Phase 1: compute kernel.  Each cell computes downPartial[T,H] fp32 and stores
// it into its own window slot.  Cube does the three GEMMs; Vec does activation.
// ===========================================================================
__global__ AICORE void DistributedFfnGridCommReduceSumComputeKernel(
    __gm__ uint8_t *fftsAddr, __gm__ uint8_t *windows, __gm__ uint8_t *x, __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
    __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial, __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
    __gm__ uint8_t *downPartial, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }

    constexpr int validM = FFN_TOKEN_TILE; // 16
    constexpr int validK = FFN_MODEL_TILE; // 64
    constexpr int validN = FFN_FFN_TILE;   // 64
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;

    using GX = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
                            Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GW = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
                            Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TL = TileLeft<half, M, K, validM, validK>;
    using TR = TileRight<half, K, N, validK, validN>;
    using TC = TileAcc<float, M, N, validM, validN>;

    using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_GATE_PARTIAL_BYTES, 1>;
    using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_UP_PARTIAL_BYTES, 1>;
    using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_HIDDEN_BYTES, 1>;
    using DownPipe = TPipe<6, Direction::DIR_C2V, FFN_DOWN_PARTIAL_BYTES, 1>;

    TileA xMat;
    TileA hiddenMat;
    TileB wGateMat;
    TileB wUpMat;
    TileB wDownMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(hiddenMat, kL1Hidden);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);
    TASSIGN(wDownMat, kL1WDown);

    TL aT;
    TR bT;
    TC cT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);

    constexpr int xTileBytes = FFN_X_BYTES;
    constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
    constexpr int wUpTileBytes = FFN_W_UP_BYTES;
    constexpr int wDownTileBytes = FFN_W_DOWN_BYTES;
    constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
    constexpr int hiddenTileBytes = FFN_HIDDEN_BYTES;
    constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
    __gm__ uint8_t *xBlock = x + blockIdx * xTileBytes;
    __gm__ uint8_t *wGateBlock = wGate + blockIdx * wGateTileBytes;
    __gm__ uint8_t *wUpBlock = wUp + blockIdx * wUpTileBytes;
    __gm__ uint8_t *wDownBlock = wDown + blockIdx * wDownTileBytes;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *hiddenBlock = hiddenIn + blockIdx * hiddenTileBytes;
    __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;
    // Final per-cell partial lives in this cell's window slot (intra-offset 0).
    __gm__ uint8_t *winSlot = windows + blockIdx * FFN_GRID_WINDOW_BYTES;

    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);
    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);

    if constexpr (DAV_CUBE) {
        GX xG(reinterpret_cast<__gm__ half *>(xBlock));
        GW wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
        GW wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));
        GW wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));

        TLOAD(xMat, xG);
        TLOAD(wGateMat, wGateG);
        TLOAD(wUpMat, wUpG);
        TLOAD(wDownMat, wDownG);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        // -------- gate: gatePartial = x @ W_gate --------
        TMOV(aT, xMat);
        TMOV(bT, wGateMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TPUSH<GatePipe, TC, TileSplitAxis::TILE_NO_SPLIT>(gatePipe, cT);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // -------- up: upPartial = x @ W_up --------
        TMOV(aT, xMat);
        TMOV(bT, wUpMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TPUSH<UpPipe, TC, TileSplitAxis::TILE_NO_SPLIT>(upPipe, cT);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        TPOP<HiddenPipe, TileA, TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenMat);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // -------- down: downPartial = hidden @ W_down --------
        TMOV(aT, hiddenMat);
        TMOV(bT, wDownMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TPUSH<DownPipe, TC, TileSplitAxis::TILE_NO_SPLIT>(downPipe, cT);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }

    if constexpr (DAV_VEC) {
        GateF32Tile gateF32;
        UpF32Tile upF32;
        HiddenF32Tile hiddenF32;
        HiddenF16Tile hiddenF16;
        DownF32Tile downF32;
        TASSIGN(gateF32, kUbGateF32);
        TASSIGN(upF32, kUbUpF32);
        TASSIGN(hiddenF32, kUbHiddenF32);
        TASSIGN(hiddenF16, kUbHiddenF16);
        TASSIGN(downF32, kUbDownF32);

        TPOP<GatePipe, GateF32Tile, TileSplitAxis::TILE_NO_SPLIT>(gatePipe, gateF32);
        TPOP<UpPipe, UpF32Tile, TileSplitAxis::TILE_NO_SPLIT>(upPipe, upF32);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        TLRELU(gateF32, gateF32, FFN_PRELU_ALPHA);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        TMUL(hiddenF32, gateF32, upF32);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

        TPUSH<HiddenPipe, HiddenF16Tile, TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenF16);

        TPOP<DownPipe, DownF32Tile, TileSplitAxis::TILE_NO_SPLIT>(downPipe, downF32);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif

        // Publish this cell's [T,H] fp32 partial into its window slot.  The
        // phase-2 collective reduces these across the row's columns.
        GTHF32 winG(reinterpret_cast<__gm__ float *>(winSlot));
        TSTORE(winG, downF32);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
#else
    (void)fftsAddr;
    (void)windows;
    (void)x;
    (void)wGate;
    (void)wUp;
    (void)wDown;
    (void)gatePartial;
    (void)upPartial;
    (void)hiddenIn;
    (void)downPartial;
    (void)gridRows;
    (void)gridCols;
#endif
}

// ===========================================================================
// Phase 2: collective kernel.  One block per row; the block is the root of its
// row's column group and runs pto::comm::TREDUCE(Sum) over the gridCols window
// slots into y_row[T,H].
// ===========================================================================
__global__ AICORE void DistributedFfnGridCommReduceSumCollectiveKernel(__gm__ uint8_t *fftsAddr,
                                                                       __gm__ uint8_t *windowsBase,
                                                                       __gm__ uint8_t *yOutput,
                                                                       __gm__ uint8_t *hcclCtxRaw, int gridRows,
                                                                       int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int row = get_block_idx();
    if (row < 0 || row >= gridRows) {
        return;
    }

    if constexpr (DAV_VEC) {
        auto *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtxRaw);
        // win0 lies in window 0 (the arena base == windowsIn[0]); HcclRemotePtr
        // with rankId==0 maps win0 + pe*FFN_GRID_WINDOW_BYTES, so folding the
        // row offset into `pe = row*gridCols + c` selects this row's column c.
        auto *win0 = reinterpret_cast<__gm__ float *>(windowsBase);

        GTHF32 tensors[FFN_GRID_COLS];
        for (int c = 0; c < gridCols; ++c) {
            int peCell = row * gridCols + c;
            __gm__ float *remote = HcclRemotePtr(ctx, win0, peCell);
            tensors[c] = GTHF32(remote);
        }
        comm::ParallelGroup<GTHF32> pg(tensors, gridCols, /*rootIdx=*/0);

        __gm__ uint8_t *yBlock = yOutput + row * FFN_Y_OUTPUT_BYTES;
        GTHF32 yG(reinterpret_cast<__gm__ float *>(yBlock));

        ReduceTile accTile(FFN_TOKEN_TILE, FFN_MODEL_TILE);
        ReduceTile recvTile(FFN_TOKEN_TILE, FFN_MODEL_TILE);
        TASSIGN(accTile, kUbReduceAcc);
        TASSIGN(recvTile, kUbReduceRecv);

        comm::TREDUCE(pg, yG, accTile, recvTile, comm::ReduceOp::Sum);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
    }
#else
    (void)fftsAddr;
    (void)windowsBase;
    (void)yOutput;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchFfnCommReduceSumComputeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *x, uint8_t *wGate, uint8_t *wUp,
                                         uint8_t *wDown, uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
                                         uint8_t *downPartial, int gridRows, int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridCommReduceSumComputeKernel<<<totalBlocks, nullptr, stream>>>(
        ffts, windows, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, gridRows, gridCols);
}

void launchFfnCommReduceSumCollectiveKernel(uint8_t *ffts, uint8_t *windowsBase, uint8_t *yOutput, uint8_t *hcclCtx,
                                            int gridRows, int gridCols, void *stream)
{
    if (gridRows <= 0) {
        return;
    }
    DistributedFfnGridCommReduceSumCollectiveKernel<<<gridRows, nullptr, stream>>>(ffts, windowsBase, yOutput, hcclCtx,
                                                                                   gridRows, gridCols);
}

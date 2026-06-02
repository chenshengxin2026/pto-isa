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

// Single-device multi-block FFN, AllGather combine via the pto::comm collective
// API (pto::comm::TGATHER) instead of the GridPipe mock.
//
// Two kernels, launched back-to-back with a host stream-sync barrier between
// them:
//
//   1. Compute kernel (gridRows*gridCols blocks, mixed Cube/Vec): each cell
//      computes its hidden shard hidden[T,Fi] (fp16) and stores it into its own
//      HCCL window slot.  (No down projection here.)
//   2. Collective kernel (gridRows*gridCols blocks, mixed Cube/Vec): every cell
//      is the root of its row group.  Vec runs pto::comm::TGATHER to collect the
//      row's gridCols hidden shards into a GM scratch laid out [gridCols*T, Fi]
//      (TGATHER stacks along DIM_3 / rows), re-lays-out to hidden_full[T,F] via
//      per-column TLOAD + TINSERT, and hands it to Cube; Cube does the down GEMM
//      hidden_full @ W_down[:,Hc] -> y[:,Hc]; Vec stores the cell's H output
//      shard.  Per-cell gather emulates an all-gather.
//
// The single-device mock uses a fake HcclDeviceContext where
//   windowsIn[i] = base + i * FFN_GRID_WINDOW_BYTES  and rankId = 0.
// HcclRemotePtr(ctx, win0, pe) resolves to base + pe*FFN_GRID_WINDOW_BYTES, so
// folding the row offset into `pe = row*gridCols + c` selects the row's columns.

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

constexpr int AlignUp(int value, int align)
{
    return ((value + align - 1) / align) * align;
}

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using HiddenFullF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TOTAL_TILE, BLayout::RowMajor>;
using DownF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE, BLayout::RowMajor>;
// TGATHER staging tile: dynamic valid dims, mirroring the comm ST test usage.
using GatherStageTile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor, -1, -1>;

// Hidden shard view [T, Fi] (window slot in phase 1; per-column gather source /
// scratch-block re-layout source in phase 2).
using ShapeTFi = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_FFN_TILE>;
using StrideTFi = Stride<FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_TOKEN_TILE * FFN_FFN_TILE,
                         FFN_FFN_TILE, 1>;
using GTFiF16 = GlobalTensor<half, ShapeTFi, StrideTFi, Layout::ND>;

// TGATHER destination: row-stacked [gridCols*T, Fi].
using ShapeGatherDst = Shape<1, 1, 1, FFN_GRID_COLS * FFN_TOKEN_TILE, FFN_FFN_TILE>;
using StrideGatherDst = Stride<FFN_GRID_COLS * FFN_TOKEN_TILE * FFN_FFN_TILE,
                               FFN_GRID_COLS * FFN_TOKEN_TILE * FFN_FFN_TILE,
                               FFN_GRID_COLS * FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_FFN_TILE, 1>;
using GGatherDst = GlobalTensor<half, ShapeGatherDst, StrideGatherDst, Layout::ND>;

// Output H shard [T, Hc] embedded in the row's [T, H] output (row stride = H).
using ShapeTHShard = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE>;
using StrideTHShard = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                             FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHShardF32 = GlobalTensor<float, ShapeTHShard, StrideTHShard, Layout::ND>;

constexpr int kUbAlignBytes = 0x1000;
constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = AlignUp(kUbGateF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF32 = AlignUp(kUbUpF32 + FFN_UP_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF16 = AlignUp(kUbHiddenF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);

// Phase-2 UB map (gather + relayout + down store).
constexpr int kUbGatherStage = 0x0000;
constexpr int kUbShardF16 = AlignUp(kUbGatherStage + FFN_HIDDEN_BYTES, kUbAlignBytes);
constexpr int kUbHiddenFullF16 = AlignUp(kUbShardF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
constexpr int kUbDownF32 = AlignUp(kUbHiddenFullF16 + FFN_HIDDEN_FULL_BYTES, kUbAlignBytes);

constexpr int kL1X = 0x00000;
constexpr int kL1Hidden = 0x10000;
constexpr int kL1WGate = 0x20000;
constexpr int kL1WUp = 0x24000;
constexpr int kL1WDown = 0x28000;

#endif

// ===========================================================================
// Phase 1: compute kernel.  Each cell computes hidden[T,Fi] fp16 and stores it
// into its own window slot.  Cube does gate/up GEMMs; Vec does activation.
// ===========================================================================
__global__ AICORE void DistributedFfnGridCommAllGatherComputeKernel(__gm__ uint8_t *fftsAddr, __gm__ uint8_t *windows,
                                                                    __gm__ uint8_t *x, __gm__ uint8_t *wGate,
                                                                    __gm__ uint8_t *wUp, __gm__ uint8_t *gatePartial,
                                                                    __gm__ uint8_t *upPartial, int gridRows,
                                                                    int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }

    constexpr int validM = FFN_TOKEN_TILE; // T
    constexpr int validK = FFN_MODEL_TILE; // H
    constexpr int validN = FFN_FFN_TILE;   // Fi
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

    TileA xMat;
    TileB wGateMat;
    TileB wUpMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);

    TL aT;
    TR bT;
    TC cT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);

    constexpr int xTileBytes = FFN_X_BYTES;
    constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
    constexpr int wUpTileBytes = FFN_W_UP_BYTES;
    constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
    __gm__ uint8_t *xBlock = x + blockIdx * xTileBytes;
    __gm__ uint8_t *wGateBlock = wGate + blockIdx * wGateTileBytes;
    __gm__ uint8_t *wUpBlock = wUp + blockIdx * wUpTileBytes;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
    // hidden[T,Fi] partial lives in this cell's window slot (intra-offset 0).
    __gm__ uint8_t *winSlot = windows + blockIdx * FFN_GRID_WINDOW_BYTES;

    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);

    if constexpr (DAV_CUBE) {
        GX xG(reinterpret_cast<__gm__ half *>(xBlock));
        GW wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
        GW wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));

        TLOAD(xMat, xG);
        TLOAD(wGateMat, wGateG);
        TLOAD(wUpMat, wUpG);

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
    }

    if constexpr (DAV_VEC) {
        GateF32Tile gateF32;
        UpF32Tile upF32;
        HiddenF32Tile hiddenF32;
        HiddenF16Tile hiddenF16;
        TASSIGN(gateF32, kUbGateF32);
        TASSIGN(upF32, kUbUpF32);
        TASSIGN(hiddenF32, kUbHiddenF32);
        TASSIGN(hiddenF16, kUbHiddenF16);

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

        // Publish this cell's [T,Fi] fp16 hidden shard into its window slot.
        GTFiF16 winG(reinterpret_cast<__gm__ half *>(winSlot));
        TSTORE(winG, hiddenF16);

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
    (void)gatePartial;
    (void)upPartial;
    (void)gridRows;
    (void)gridCols;
#endif
}

// ===========================================================================
// Phase 2: collective kernel.  Every cell gathers its row's hidden shards via
// pto::comm::TGATHER, re-lays-out to hidden_full[T,F], down-projects with its
// W_down[:,Hc] shard, and stores its [T,Hc] output shard.
// ===========================================================================
__global__ AICORE void DistributedFfnGridCommAllGatherCollectiveKernel(
    __gm__ uint8_t *fftsAddr, __gm__ uint8_t *windowsBase, __gm__ uint8_t *gatherScratch, __gm__ uint8_t *hiddenScratch,
    __gm__ uint8_t *downScratch, __gm__ uint8_t *wDown, __gm__ uint8_t *yOutput, __gm__ uint8_t *hcclCtxRaw,
    int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    static_assert((FFN_MODEL_TILE % FFN_GRID_COLS) == 0, "AllGather split requires H divisible by gridCols.");

    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }
    int row = blockIdx / gridCols;
    int col = blockIdx - row * gridCols;

    constexpr int validM = FFN_TOKEN_TILE;     // T
    constexpr int validF = FFN_FFN_TOTAL_TILE; // F = Fi * cols
    constexpr int validHShard = FFN_MODEL_SHARD_TILE;
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int KDown = ((validF + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int NDown = ((validHShard + blockAlign - 1) / blockAlign) * blockAlign;

    using GWDown =
        GlobalTensor<half, Shape<1, 1, 1, validF, validHShard>,
                     Stride<validF * validHShard, validF * validHShard, validF * validHShard, validHShard, 1>>;
    using HiddenFullMat =
        Tile<TileType::Mat, half, M, KDown, BLayout::ColMajor, validM, validF, SLayout::RowMajor, 512>;
    using WDownMat =
        Tile<TileType::Mat, half, KDown, NDown, BLayout::ColMajor, validF, validHShard, SLayout::RowMajor, 512>;
    using TLDown = TileLeft<half, M, KDown, validM, validF>;
    using TRDown = TileRight<half, KDown, NDown, validF, validHShard>;
    using TCDown = TileAcc<float, M, NDown, validM, validHShard>;

    using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_HIDDEN_FULL_BYTES, 1>;
    using DownPipe = TPipe<6, Direction::DIR_C2V, FFN_DOWN_PARTIAL_BYTES, 1>;

    __gm__ uint8_t *wDownBlock = wDown + blockIdx * FFN_W_DOWN_BYTES;
    __gm__ uint8_t *gatherBlock = gatherScratch + blockIdx * FFN_HIDDEN_FULL_BYTES;
    __gm__ uint8_t *hiddenBlock = hiddenScratch + blockIdx * FFN_HIDDEN_FULL_BYTES;
    __gm__ uint8_t *downBlock = downScratch + blockIdx * FFN_DOWN_PARTIAL_BYTES;
    __gm__ uint8_t *yBlock =
        yOutput + row * FFN_Y_OUTPUT_BYTES + col * FFN_MODEL_SHARD_TILE * static_cast<int>(sizeof(float));

    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);

    if constexpr (DAV_CUBE) {
        HiddenFullMat hiddenMat;
        WDownMat wDownMat;
        TASSIGN(hiddenMat, kL1Hidden);
        TASSIGN(wDownMat, kL1WDown);

        TLDown aDownT;
        TRDown bDownT;
        TCDown cDownT;
        TASSIGN(aDownT, 0x0);
        TASSIGN(bDownT, 0x0);
        TASSIGN(cDownT, 0x0);

        GWDown wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));
        TLOAD(wDownMat, wDownG);

        // Cube receives hidden_full[T,F] from Vec.
        TPOP<HiddenPipe, HiddenFullMat, TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenMat);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // -------- down: yShard = hidden_full @ W_down[:, Hc] --------
        TMOV(aDownT, hiddenMat);
        TMOV(bDownT, wDownMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cDownT, aDownT, bDownT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TPUSH<DownPipe, TCDown, TileSplitAxis::TILE_NO_SPLIT>(downPipe, cDownT);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }

    if constexpr (DAV_VEC) {
        auto *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtxRaw);
        auto *win0 = reinterpret_cast<__gm__ half *>(windowsBase);

        // (a) TGATHER: collect the row's gridCols hidden shards [T,Fi] into a GM
        //     scratch laid out [gridCols*T, Fi] (stacked along rows).
        GTFiF16 tensors[FFN_GRID_COLS];
        for (int c = 0; c < gridCols; ++c) {
            int peCell = row * gridCols + c;
            __gm__ half *remote = HcclRemotePtr(ctx, win0, peCell);
            tensors[c] = GTFiF16(remote);
        }
        comm::ParallelGroup<GTFiF16> pg(tensors, gridCols, /*rootIdx=*/0);

        GGatherDst gatherG(reinterpret_cast<__gm__ half *>(gatherBlock));
        GatherStageTile stage(FFN_TOKEN_TILE, FFN_FFN_TILE);
        TASSIGN(stage, kUbGatherStage);
        comm::TGATHER(pg, gatherG, stage);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // (b) Re-layout [gridCols*T, Fi] -> hidden_full[T, F]: shard c (scratch
        //     rows [c*T,(c+1)*T)) is the hidden of column c, which belongs at
        //     feature columns [c*Fi,(c+1)*Fi).
        HiddenF16Tile shardF16;
        HiddenFullF16Tile hiddenFullF16;
        TASSIGN(shardF16, kUbShardF16);
        TASSIGN(hiddenFullF16, kUbHiddenFullF16);

        for (int c = 0; c < gridCols; ++c) {
            GTFiF16 blockView(reinterpret_cast<__gm__ half *>(gatherBlock) + c * FFN_TOKEN_TILE * FFN_FFN_TILE);
            TLOAD(shardF16, blockView);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
            TINSERT(hiddenFullF16, shardF16, 0, static_cast<uint16_t>(c * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TPUSH<HiddenPipe, HiddenFullF16Tile, TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenFullF16);

        // (c) Receive the down-projected output shard from Cube and store it.
        DownF32Tile downF32;
        TASSIGN(downF32, kUbDownF32);
        TPOP<DownPipe, DownF32Tile, TileSplitAxis::TILE_NO_SPLIT>(downPipe, downF32);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        GTHShardF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
        TSTORE(yG, downF32);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
#else
    (void)fftsAddr;
    (void)windowsBase;
    (void)gatherScratch;
    (void)hiddenScratch;
    (void)downScratch;
    (void)wDown;
    (void)yOutput;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchFfnCommAllGatherComputeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *x, uint8_t *wGate, uint8_t *wUp,
                                         uint8_t *gatePartial, uint8_t *upPartial, int gridRows, int gridCols,
                                         void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridCommAllGatherComputeKernel<<<totalBlocks, nullptr, stream>>>(ffts, windows, x, wGate, wUp,
                                                                                   gatePartial, upPartial, gridRows,
                                                                                   gridCols);
}

void launchFfnCommAllGatherCollectiveKernel(uint8_t *ffts, uint8_t *windowsBase, uint8_t *gatherScratch,
                                            uint8_t *hiddenScratch, uint8_t *downScratch, uint8_t *wDown,
                                            uint8_t *yOutput, uint8_t *hcclCtx, int gridRows, int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridCommAllGatherCollectiveKernel<<<totalBlocks, nullptr, stream>>>(
        ffts, windowsBase, gatherScratch, hiddenScratch, downScratch, wDown, yOutput, hcclCtx, gridRows, gridCols);
}

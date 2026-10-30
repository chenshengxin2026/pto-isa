/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

using Source = Tile<TileType::Vec, float4_e2m1x2_t, 19, 128>;
using NDSource = Tile<TileType::Vec, float8_e4m3_t, 19, 128>;
using Padded = Tile<
    TileType::Vec, float8_e4m3_t, 64, 128, BLayout::ColMajor, 19, 128, SLayout::RowMajor, 512, PadValue::Null,
    CompactMode::RowPlusOne>;
using Matrix = Tile<TileType::Mat, float8_e4m3_t, 64, 128, BLayout::ColMajor, 19, 128, SLayout::RowMajor>;
using Pipe = TPipe<3, Direction::DIR_V2C, 65536, 2>;
static_assert(Padded::Rows == 64 && Padded::Cols == 128 && Padded::Numel == 8192);
static_assert(Padded::StorageNumel == 65 * 128 && Padded::StorageBytes == 65 * 128);
static_assert((Padded::ValidRow + 15) / 16 * 16 + 1 == 33 && Padded::Rows + 1 == 65);

using LegacySource = Tile<TileType::Vec, float4_e2m1x2_t, 32, 128>;
using LegacyPadded = Tile<
    TileType::Vec, float8_e4m3_t, 33, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null,
    CompactMode::RowPlusOne>;
using LegacyMatrix = Tile<TileType::Mat, float8_e4m3_t, 64, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor>;
static_assert(LegacyPadded::kRowPlusOneRowsPadded);
static_assert(LegacyPadded::StorageNumel == 33 * 128 && LegacyPadded::StorageBytes == 33 * 128);
static_assert((LegacyPadded::ValidRow + 15) / 16 * 16 + 1 == LegacyPadded::Rows);

__tf__ PTO_INTERNAL void exportLastLegacyStorageByte(LegacyPadded::TileDType __in__ padded, __gm__ uint8_t* output)
{
    output[0] = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(padded))[LegacyPadded::StorageBytes - 1];
}

__tf__ PTO_INTERNAL void exportLastStorageByte(Padded::TileDType __in__ padded, __gm__ uint8_t* output)
{
    output[0] = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(padded))[Padded::StorageBytes - 1];
}

// Compile compatibility only, in explicit and __PTO_AUTO__ modes. These kernels
// establish allocation/type acceptance, without a numerical DMA/FIFO oracle.
extern "C" __global__ AICORE void tcvtFp4RowPlusOneCompile(
    __gm__ float4_e2m1x2_t* input, __gm__ uint8_t* output, __gm__ void* fifo, bool push)
{
    Source source;
    Padded padded;
    Matrix matrix;
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(padded, Source::StorageBytes);
    TASSIGN(matrix, 0);
#endif
    GlobalTensor<float4_e2m1x2_t, Shape<1, 1, 1, 19, 128>, pto::Stride<2432, 2432, 2432, 128, 1>> global(input);
    TLOAD(source, global);
    TCVT(padded, source, RoundMode::CAST_RINT);
    TINSERT(matrix, padded, 0, 0);
    TINSERT<TInsertMode::SPLIT2>(matrix, padded, 0, 0);
    TINSERT<TInsertMode::SPLIT4>(matrix, padded, 0, 0);
    if (push) {
        Pipe pipe(fifo, 0, 0);
        TPUSH<Pipe, Padded, TileSplitAxis::TILE_NO_SPLIT>(pipe, padded);
    }
    // Force retention of the final physical byte in the tile_size allocation.
    exportLastStorageByte(padded.data(), output);
}

extern "C" __global__ AICORE void tcvtFp4RowPlusOneMoveCompile(__gm__ float8_e4m3_t* input, __gm__ uint8_t* output)
{
    NDSource source;
    Padded padded;
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(padded, NDSource::StorageBytes);
#endif
    GlobalTensor<float8_e4m3_t, Shape<1, 1, 1, 19, 128>, pto::Stride<2432, 2432, 2432, 128, 1>> global(input);
    TLOAD(source, global);
    TMOV(padded, source);
    exportLastStorageByte(padded.data(), output);
}

extern "C" __global__ AICORE void tcvtFp4LegacyRowPlusOneCompile(__gm__ float4_e2m1x2_t* input, __gm__ uint8_t* output)
{
    LegacySource source;
    LegacyPadded padded;
    LegacyMatrix matrix;
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(padded, LegacySource::StorageBytes);
    TASSIGN(matrix, 0);
#endif
    GlobalTensor<float4_e2m1x2_t, Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>> global(input);
    TLOAD(source, global);
    TCVT(padded, source, RoundMode::CAST_RINT);
    TINSERT(matrix, padded, 0, 0);
    TINSERT<TInsertMode::SPLIT2>(matrix, padded, 0, 0);
    TINSERT<TInsertMode::SPLIT4>(matrix, padded, 0, 0);
    exportLastLegacyStorageByte(padded.data(), output);
}

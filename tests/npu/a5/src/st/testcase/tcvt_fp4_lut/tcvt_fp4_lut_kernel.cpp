/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "cases.h"

using namespace pto;

namespace {
constexpr int kGuardBytes = BLOCK_BYTE_SIZE;

template <typename T, bool Dynamic>
PTO_INTERNAL T makeTile(unsigned rows, unsigned cols)
{
    if constexpr (Dynamic) {
        return T(rows, cols);
    } else {
        return T();
    }
}

template <int SR, int SC, int DR, int DC, int VR, int VC, bool NZ, int Compact, bool Dynamic>
__global__ AICORE void runFp4Lut(
    __gm__ uint8_t* dst, __gm__ uint8_t* src, unsigned rows, unsigned cols, int round, int api)
{
    constexpr int srcBytes = SR * SC / 2;
    using Source =
        Tile<TileType::Vec, float4_e2m1x2_t, SR, SC, BLayout::RowMajor, Dynamic ? DYNAMIC : VR, Dynamic ? DYNAMIC : VC>;
    using Destination = Tile<
        TileType::Vec, float8_e4m3_t, DR, DC, NZ ? BLayout::ColMajor : BLayout::RowMajor, Dynamic ? DYNAMIC : VR,
        Dynamic ? DYNAMIC : VC, NZ ? SLayout::RowMajor : SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Null,
        Compact == 2 ? CompactMode::RowPlusOne : (Compact == 1 ? CompactMode::Normal : CompactMode::Null)>;
    constexpr int dstBytes = Destination::StorageBytes + 2 * kGuardBytes;
    auto source = makeTile<Source, Dynamic>(rows, cols);
    auto destination = makeTile<Destination, Dynamic>(rows, cols);
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(source, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, srcBytes>, pto::Stride<1, 1, 1, srcBytes, 1>> input(src);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, dstBytes>, pto::Stride<1, 1, 1, dstBytes, 1>> output(dst);
    // Return a raw view of all physical padding and guards, independently of TSTORE NZ.
    TLOAD(rawInput, input);
    TLOAD(rawOutput, output);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode rounding = static_cast<RoundMode>(round);
    Tile<TileType::Vec, int32_t, 1, 8> unusedTemporary;
    switch (api) {
        case 0:
            TCVT(destination, source, rounding);
            break;
        case 1:
            TCVT(destination, source, rounding, SaturationMode::ON);
            break;
        case 2:
            TCVT(destination, source, rounding, SaturationMode::OFF);
            break;
        case 3:
            TCVT(destination, source, unusedTemporary, rounding);
            break;
        case 4:
            TCVT(destination, source, unusedTemporary, rounding, SaturationMode::OFF);
            break;
        default:
            TCVT<false>(destination, source, rounding, SaturationMode::ON);
            break;
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, rawOutput);
}

// Supported pairs exercise the original generic dispatch with bit-exact goldens.
template <bool Fp4Source>
__global__ AICORE void runRegression(__gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    using Src = std::conditional_t<Fp4Source, float4_e2m1x2_t, float>;
    using Dst = std::conditional_t<Fp4Source, bfloat16_t, float8_e4m3_t>;
    constexpr int columns = 128;
    constexpr int srcBytes = Fp4Source ? columns / 2 : columns * sizeof(Src);
    constexpr int dstBytes = columns * sizeof(Dst);
    Tile<TileType::Vec, Src, 1, columns> source;
    Tile<TileType::Vec, Dst, 1, columns> destination;
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes + 2 * kGuardBytes> rawOutput;
    TASSIGN(source, 0);
    TASSIGN(rawInput, 0);
    TASSIGN(rawOutput, srcBytes);
    TASSIGN(destination, srcBytes + kGuardBytes);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, srcBytes>, pto::Stride<1, 1, 1, srcBytes, 1>> input(src);
    GlobalTensor<
        uint8_t, Shape<1, 1, 1, 1, dstBytes + 2 * kGuardBytes>, pto::Stride<1, 1, 1, dstBytes + 2 * kGuardBytes, 1>>
        output(dst);
    TLOAD(rawInput, input);
    TLOAD(rawOutput, output);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(destination, source, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, rawOutput);
}

__global__ AICORE void runBulkPrimitive(__gm__ uint8_t* dst, __gm__ uint8_t* src, unsigned sourceOffset, bool rotation)
{
    constexpr int srcBytes = 768;
    constexpr int registerBytes = 256;
    constexpr int dstBytes = registerBytes + 2 * kGuardBytes;
    Tile<TileType::Vec, uint8_t, 1, srcBytes> rawInput;
    Tile<TileType::Vec, uint8_t, 1, dstBytes> rawOutput;
    TASSIGN(rawInput, 0);
    TASSIGN(rawOutput, srcBytes);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, srcBytes>, pto::Stride<1, 1, 1, srcBytes, 1>> input(src);
    GlobalTensor<uint8_t, Shape<1, 1, 1, 1, dstBytes>, pto::Stride<1, 1, 1, dstBytes, 1>> output(dst);
    TLOAD(rawInput, input);
    TLOAD(rawOutput, output);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    __ubuf__ uint8_t* srcPtr = rawInput.data();
    __ubuf__ uint8_t* dstPtr = rawOutput.data();
    __VEC_SCOPE__
    {
        MaskReg fullMask = pset_b8(PAT_ALL);
        RegTensor<uint8_t> sourceRegister, index, result;
        vlds(sourceRegister, srcPtr, sourceOffset, NORM);
        vci((RegTensor<int8_t>&)index, static_cast<int8_t>(rotation ? -128 : 0), INC_ORDER);
        vselr(result, sourceRegister, index);
        vsts(result, dstPtr, kGuardBytes, NORM_B8, fullMask);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, rawOutput);
}
template <CompactMode compactMode>
__global__ AICORE void runFp4LutStore(__gm__ uint8_t* dst, __gm__ uint8_t* src)
{
    constexpr int sourceRows = kFp4LutStoreSourceRows;
    constexpr int columns = kFp4LutStoreColumns;
    constexpr int storeRows = kFp4LutStoreRows;
    constexpr int panelColumns = kFp4LutStorePanelColumns;
    constexpr int sourceBytes = sourceRows * columns / 2;
    using Source = Tile<TileType::Vec, float4_e2m1x2_t, sourceRows, columns, BLayout::RowMajor, sourceRows, columns>;
    using CompactTile = Tile<
        TileType::Vec, float8_e4m3_t, 64, columns, BLayout::ColMajor, DYNAMIC, columns, SLayout::RowMajor,
        TileConfig::fractalABSize, PadValue::Null, compactMode>;
    using OutShape = Shape<1, columns / panelColumns, storeRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, panelColumns>;
    using OutStride =
        pto::Stride<columns * storeRows, storeRows * panelColumns, FRACTAL_NZ_ROW * panelColumns, panelColumns, 1>;
    using OutGlobal = GlobalTensor<float8_e4m3_t, OutShape, OutStride, Layout::NZ>;

    static_assert(compactMode == CompactMode::Normal || compactMode == CompactMode::RowPlusOne);
    constexpr int storageRows = compactMode == CompactMode::RowPlusOne ? 65 : 64;
    static_assert(storeRows == (sourceRows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW);
    static_assert(CompactTile::StorageBytes == storageRows * columns && CompactTile::Rows == 64);

    Source source;
    CompactTile compact(sourceRows);
    TASSIGN(source, 0x0);
    TASSIGN(compact, 0x10000);
    OutGlobal output(reinterpret_cast<__gm__ float8_e4m3_t*>(dst + kGuardBytes));

    constexpr uint16_t sourceBurstLen = sourceBytes / BLOCK_BYTE_SIZE;
    copy_gm_to_ubuf((__ubuf__ void*)source.data(), (__gm__ void*)src, 0, 1, sourceBurstLen, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    __ubuf__ uint8_t* compactPtr = (__ubuf__ uint8_t*)compact.data();
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(uint8_t);
    constexpr uint16_t fillRepeats = (CompactTile::StorageBytes + elementsPerRepeat - 1) / elementsPerRepeat;
    __VEC_SCOPE__
    {
        RegTensor<uint8_t> fill;
        for (uint16_t repeat = 0; repeat < fillRepeats; ++repeat) {
            uint32_t offset = repeat * elementsPerRepeat;
            uint32_t remaining = CompactTile::StorageBytes - offset;
            uint32_t count = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg mask = CreatePredicate<uint8_t>(count);
            vdup(fill, static_cast<uint8_t>(0x7f), mask, MODE_ZEROING);
            vsts(fill, compactPtr, offset, NORM_B8, mask);
        }
    }

    TCVT(compact, source, RoundMode::CAST_RINT);
    // Preserve the producer pitch while exposing the complete aligned region.
    compact.SetValidRow(storeRows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, compact);
}
} // namespace

void launchFp4Lut(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream)
{
    switch (static_cast<Fp4LutCaseId>(caseId)) {
#define FP4_LUT_LAUNCH(Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic) \
    case Fp4LutCaseId::Name:                                               \
        runFp4Lut<SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic>            \
            <<<1, nullptr, stream>>>(dst, src, rows, cols, round, api);    \
        break;
        FP4_LUT_CASES(FP4_LUT_LAUNCH)
#undef FP4_LUT_LAUNCH
    }
}

void launchFp4LutRegression(int pair, unsigned char* dst, unsigned char* src, void* stream)
{
    if (pair == 0) {
        runRegression<true><<<1, nullptr, stream>>>(dst, src);
    } else {
        runRegression<false><<<1, nullptr, stream>>>(dst, src);
    }
}

void launchFp4LutBulkPrimitive(unsigned char* dst, unsigned char* src, int sourceOffset, bool rotation, void* stream)
{
    runBulkPrimitive<<<1, nullptr, stream>>>(dst, src, sourceOffset, rotation);
}

void launchFp4LutStore(unsigned char* dst, unsigned char* src, int compactMode, void* stream)
{
    if (compactMode == 1) {
        runFp4LutStore<CompactMode::Normal><<<1, nullptr, stream>>>(dst, src);
    } else if (compactMode == 2) {
        runFp4LutStore<CompactMode::RowPlusOne><<<1, nullptr, stream>>>(dst, src);
    }
}

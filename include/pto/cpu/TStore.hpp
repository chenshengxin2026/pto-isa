/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

#include <pto/common/constants.hpp>
#include <cassert>
#include "pto/cpu/parallel.hpp"
#include "common.hpp"
#include "nz_utils.hpp"

namespace pto {

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TStoreInstrL12Gm(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t nBurst, uint16_t lenBurst,
    uint16_t gmGap, uint16_t l1Gap)
{
    const uint32_t blockSize = C0_SIZE_BYTE;
    uint16_t dstStride = (static_cast<size_t>(lenBurst) + gmGap) * blockSize / sizeof(typename TileData::DType);
    uint16_t srcStride = (static_cast<size_t>(lenBurst) + l1Gap) * blockSize / sizeof(typename TileData::DType);
    uint8_t elemNum = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    for (uint16_t i = 0; i < nBurst; i++) {
        for (size_t j = 0; j < lenBurst * elemNum; j++) {
            SetProperDataPart(dst, dstStride * i + j, src[srcStride * i + j]);
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TStore5HD(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int srcC1, int srcH, int srcW,
    int gStrideC1, int gStrideH, int gStrideW, int dstC1, int dstH, int dstW)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    uint16_t nBurst = srcH;
    uint16_t lenBurst = srcW;

    // GM gap is encoded in 32-byte blocks between stored rows.
    uint16_t gmGap = ((gStrideH - srcW * c0ElemCount) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint16_t l1Gap = 0;

    for (uint32_t j = 0; j < srcC1; j++) {
        typename GlobalData::DType* dstAddrP = dst + j * gStrideC1;
        __cbuf__ typename TileData::DType* srcAddrP = src + j * dstH * dstW * c0ElemCount;

        TStoreInstrL12Gm<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TStore6HD(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int dstN, int dstD, int dstC1,
    int dstH, int dstW, int gStrideN, int gStrideD, int gStrideC1, int gStrideH, int gStrideW, int srcN, int srcD,
    int srcC1, int srcH, int srcW)
{
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    for (uint32_t n = 0; n < srcN; n++) {
        for (uint32_t d = 0; d < srcD; d++) {
            int64_t offsetDst = n * gStrideN + d * gStrideD;
            int64_t offsetSrc = (n * srcD * srcH * srcW * srcC1 + d * srcH * srcW * srcC1) * c0ElemCount;

            TStore5HD<TileData, GlobalData>(
                dst + offsetDst, src + offsetSrc, srcC1, srcH, srcW, gStrideC1, gStrideH, gStrideW, srcC1, srcH, srcW);
        }
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void TStoreAcc(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars)
{
    const size_t validRow = src.GetValidRow();
    const size_t validCol = src.GetValidCol();

    const size_t shape_0 = dst.GetShape(GlobalTensorDim::DIM_0);
    const size_t shape_1 = dst.GetShape(GlobalTensorDim::DIM_1);
    const size_t shape_2 = dst.GetShape(GlobalTensorDim::DIM_2);
    const size_t shape_3 = dst.GetShape(GlobalTensorDim::DIM_3);
    const size_t shape_4 = dst.GetShape(GlobalTensorDim::DIM_4);

    const size_t stride_0 = dst.GetStride(GlobalTensorDim::DIM_0);
    const size_t stride_1 = dst.GetStride(GlobalTensorDim::DIM_1);
    const size_t stride_2 = dst.GetStride(GlobalTensorDim::DIM_2);
    const size_t stride_3 = dst.GetStride(GlobalTensorDim::DIM_3);
    const size_t stride_4 = dst.GetStride(GlobalTensorDim::DIM_4);

    uint64_t scalar = 0;
    for (int s0 = 0; s0 < shape_0; ++s0) {
        for (int s2 = 0; s2 < shape_2; ++s2) {
            for (int s3 = 0; s3 < shape_3; ++s3) {
                size_t row = s0 * shape_2 * shape_3 + s2 * shape_3 + s3;
                for (int s1 = 0; s1 < shape_1; ++s1) {
                    for (int s4 = 0; s4 < shape_4; ++s4) {
                        typename TileData::DType val;
                        size_t col = s1 * shape_4 + s4;
                        if (row < validRow && col < validCol) {
                            if constexpr (quantMode != QuantMode_t::NoQuant) {
                                scalar = scalars[TileData::isRowMajor ? col : row];
                            }
                            val = src.GetElement(row, col);
                        }
                        const auto converted = ConvertStoreValue<
                            typename GlobalData::DType, typename TileData::DType, quantMode, applyRelu>(val, scalar);
                        if constexpr (atomicType == AtomicType::AtomicAdd) {
                            dst.AddToElement(s0, s1, s2, s3, s4, converted);
                        } else {
                            dst.SetElement(s0, s1, s2, s3, s4, converted);
                        }
                    }
                }
            }
        }
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void StorePlainGT(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars)
{
    for (int64_t i = 0; i < dst.GetShape(GlobalTensorDim::DIM_0); i++) {
        const int64_t tileHighRankOffset0 = i * dst.GetShape(GlobalTensorDim::DIM_1);
        for (int64_t j = 0; j < dst.GetShape(GlobalTensorDim::DIM_1); j++) {
            const int64_t tileHighRankOffset1 = (tileHighRankOffset0 + j) * dst.GetShape(GlobalTensorDim::DIM_2);
            for (int64_t k = 0; k < dst.GetShape(GlobalTensorDim::DIM_2); k++) {
                const int64_t tileHighRankOffset2 =
                    (tileHighRankOffset1 + k) *
                    dst.GetShape(
                        TileData::BFractal == BLayout::RowMajor ? GlobalTensorDim::DIM_3 : GlobalTensorDim::DIM_4);
                cpu::parallel_for_1d(
                    0, dst.GetShape(GlobalTensorDim::DIM_3),
                    static_cast<std::size_t>(dst.GetShape(GlobalTensorDim::DIM_3)) *
                        dst.GetShape(GlobalTensorDim::DIM_4),
                    [&](std::size_t r) {
                        const auto cols = dst.GetShape(GlobalTensorDim::DIM_4);
                        PTO_CPU_VECTORIZE_LOOP
                        for (int64_t c = 0; c < cols; c++) {
                            typename TileData::DType val;
                            if constexpr (TileData::BFractal == BLayout::RowMajor) {
                                val = src.GetElement(tileHighRankOffset2 + r, c);
                            } else {
                                val = src.GetElement(r, tileHighRankOffset2 + c);
                            }

                            uint64_t scalar = 0;
                            if constexpr (quantMode != QuantMode_t::NoQuant) {
                                scalar = scalars[TileData::isRowMajor ? c : r];
                            }

                            const auto converted = ConvertStoreValue<
                                typename GlobalData::DType, typename TileData::DType, quantMode, applyRelu>(
                                val, scalar);
                            if constexpr (atomicType == AtomicType::AtomicAdd) {
                                dst.AddToElement(i, j, k, r, c, converted);
                            } else {
                                dst.SetElement(i, j, k, r, c, converted);
                            }
                        }
                    });
            }
        }
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
__tf__ PTO_INLINE void TStore(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars)
{
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        assert(
            src.GetValidRow() == dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) &&
            src.GetValidCol() == dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                                     dst.GetShape(GlobalTensorDim::DIM_4));

    } else {
        assert(
            dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                dst.GetShape(GlobalTensorDim::DIM_4) >=
            src.GetValidRow() * src.GetValidCol());
    }
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        using D = typename GlobalData::DType;
        using S = typename TileData::DType;
        ForEachNZElement<TileData>(
            src.GetValidRow(), src.GetValidCol(), dst.GetShape(GlobalTensorDim::DIM_1),
            dst.GetShape(GlobalTensorDim::DIM_3), dst.GetShape(GlobalTensorDim::DIM_4),
            dst.GetStride(GlobalTensorDim::DIM_0), dst.GetStride(GlobalTensorDim::DIM_1),
            dst.GetStride(GlobalTensorDim::DIM_2), dst.GetStride(GlobalTensorDim::DIM_3),
            dst.GetStride(GlobalTensorDim::DIM_4), [&](size_t r, size_t c, size_t tile_idx, size_t gd_idx) {
                StoreElement<D, S, TileData, quantMode, applyRelu, atomicType>(
                    dst.data(), gd_idx, src.data()[tile_idx], r, c, scalars);
            });
    } else if (TileData::Loc == TileType::Acc) {
        TStoreAcc<GlobalData, TileData, quantMode, applyRelu, atomicType>(dst, src, scalars);
    } else {
        StorePlainGT<GlobalData, TileData, quantMode, applyRelu, atomicType>(dst, src, scalars);
    }
}

template <
    typename TileData, typename GlobalData, QuantMode_t quantMode, bool applyRelu,
    AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, const std::vector<uint64_t>& scalars = {})
{
    static_assert(
        GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
            GlobalData::layout == pto::Layout::NZ || GlobalData::layout == pto::Layout::NDC1HWC0 ||
            GlobalData::layout == pto::Layout::NC1HWC0,
        "Only ND, DN, NZ, NC1HWC0 and NDC1HWC0 GLobal Tensors are currently supported");
    if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0 && is_conv_tile_v<TileData>) {
        TStore6HD<TileData, GlobalData>(
            dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4),
            dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3), dst.GetStride(4), src.GetShape(0),
            src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4));
    } else {
        TStore<GlobalData, TileData, quantMode, applyRelu, atomicType>(dst, src, scalars);
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, false, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    (void)Phase;
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    TSTORE_IMPL<TileData, GlobalData, QuantMode_t::NoQuant, useRelu, atomicType>(dst, src);
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    size_t vector_size = 0;
    if constexpr (TileData::isRowMajor) {
        vector_size = src.GetValidCol();
    } else {
        vector_size = src.GetValidRow();
    }
    std::vector<uint64_t> scalars(vector_size, preQuantScalar);
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu, atomicType>(dst, src, scalars);
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
__aicore__ void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu, atomicType>(dst, src, scalars);
}
} // namespace pto
#endif

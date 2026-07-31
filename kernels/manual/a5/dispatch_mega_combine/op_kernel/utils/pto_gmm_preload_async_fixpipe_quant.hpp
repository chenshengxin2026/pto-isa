/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP
#define DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP

#include <type_traits>

#include "const_args.hpp"
#include "common_helpers.hpp"
#include "pto_sync_substrate.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * 16;
constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t STRIDE_LIMIT = 65536;

template <typename TileAcc, typename TileLeft, typename TileRight>
AICORE inline void LaunchPtoMatmul(TileAcc& cTile, TileLeft& aTile, TileRight& bTile, bool initC)
{
    if (initC) {
        pto::TMATMUL(cTile, aTile, bTile);
    } else {
        pto::TMATMUL_ACC(cTile, aTile, bTile);
    }
}

template <typename Element, int Rows, int Cols>
AICORE inline void PtoLoadNdGmToNzL1(
    uint64_t dstL1Offset, __gm__ Element* src, uint32_t rows, uint32_t cols, uint32_t leadingDim)
{
    using L1Tile = pto::Tile<
        pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::ND>;

    if (leadingDim < STRIDE_LIMIT) {
        SrcShape srcShape(rows, cols);
        SrcStride srcStride(
            static_cast<int64_t>(rows) * leadingDim, static_cast<int64_t>(rows) * leadingDim,
            static_cast<int64_t>(rows) * leadingDim, leadingDim);
        SrcGlobal srcGlobal(src, srcShape, srcStride);
        L1Tile dstTile(rows, cols);
        pto::TASSIGN(dstTile, dstL1Offset);
        pto::TLOAD(dstTile, srcGlobal);
    } else {
        for (uint32_t row = 0; row < rows; ++row) {
            SrcShape srcShape(1, cols);
            SrcStride srcStride(cols, cols, cols, cols);
            SrcGlobal srcGlobal(src + static_cast<uint64_t>(row) * leadingDim, srcShape, srcStride);
            L1Tile dstTile(1, cols);
            pto::TASSIGN(dstTile, dstL1Offset + static_cast<uint64_t>(row) * BYTE_PER_C0);
            pto::TLOAD(dstTile, srcGlobal);
        }
    }
}

template <typename Element, int Rows, int Cols>
AICORE inline void PtoLoadPackedWeightGmToL1(
    uint64_t dstL1Offset, __gm__ Element* src, uint32_t validRows, uint32_t validCols, uint32_t fullRows)
{
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using L1Tile = pto::Tile<
        pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, pto::DYNAMIC, pto::DYNAMIC, C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, ELE_NUM_PER_C0, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::NZ>;

    const uint32_t rowBlocks = ceilDiv<C0_NUM_PER_FRACTAL>(validRows);
    const uint32_t colBlocks = ceilDiv<ELE_NUM_PER_C0>(validCols);
    const uint32_t validRowsAligned = rowBlocks * C0_NUM_PER_FRACTAL;
    const uint32_t validColsAligned = colBlocks * ELE_NUM_PER_C0;
    const uint32_t fullRowsAligned = roundUp<C0_NUM_PER_FRACTAL>(fullRows);
    const uint32_t srcColBlockStride = fullRowsAligned * ELE_NUM_PER_C0;
    const uint32_t dstColBlockStride = Rows * ELE_NUM_PER_C0;
    const uint32_t rowBlockStride = BYTE_PER_FRACTAL / sizeof(Element);

    if (srcColBlockStride / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
        SrcShape srcShape(colBlocks, rowBlocks);
        SrcStride srcStride(static_cast<int64_t>(srcColBlockStride) * colBlocks, srcColBlockStride, rowBlockStride);
        SrcGlobal srcGlobal(src, srcShape, srcStride);
        L1Tile dstTile(validRowsAligned, validColsAligned);
        pto::TASSIGN(dstTile, dstL1Offset);
        pto::TLOAD(dstTile, srcGlobal);
    } else {
        for (uint32_t colBlock = 0; colBlock < colBlocks; ++colBlock) {
            SrcShape srcShape(1, rowBlocks);
            SrcStride srcStride(
                static_cast<int64_t>(rowBlockStride) * rowBlocks, rowBlockStride * rowBlocks, rowBlockStride);
            SrcGlobal srcGlobal(src + static_cast<uint64_t>(colBlock) * srcColBlockStride, srcShape, srcStride);
            L1Tile dstTile(validRowsAligned, ELE_NUM_PER_C0);
            pto::TASSIGN(dstTile, dstL1Offset + static_cast<uint64_t>(colBlock) * dstColBlockStride * sizeof(Element));
            pto::TLOAD(dstTile, srcGlobal);
        }
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
AICORE inline void PtoStoreAccToGm(
    __gm__ ElementDst* dst, uint64_t accOffset, uint64_t scaleOffset, uint32_t validRow, uint32_t validCol,
    uint32_t leadingDim)
{
    using DstShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using DstStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using GlobalDataOut = pto::GlobalTensor<ElementDst, DstShape, DstStride, pto::Layout::ND>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
    using ScalingTile = pto::Tile<
        pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;

    DstShape dstShape(validRow, validCol);
    DstStride dstStride(
        static_cast<int64_t>(validRow) * leadingDim, static_cast<int64_t>(validRow) * leadingDim,
        static_cast<int64_t>(validRow) * leadingDim, leadingDim);
    GlobalDataOut dstGlobal(dst, dstShape, dstStride);
    AccTile accTile(validRow, validCol);
    ScalingTile scalingTile(validCol);

    pto::TASSIGN(accTile, accOffset);
    pto::TASSIGN(scalingTile, scaleOffset);

    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, pto::AtomicType::AtomicNone, reluMode>(
            dstGlobal, accTile, scalingTile);
    } else {
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, accTile, scalingTile);
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
AICORE inline void PtoStoreAccToGm(
    __gm__ ElementDst* dst, uint64_t accOffset, uint32_t validRow, uint32_t validCol, uint32_t leadingDim)
{
    using DstShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using DstStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using GlobalDataOut = pto::GlobalTensor<ElementDst, DstShape, DstStride, pto::Layout::ND>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

    DstShape dstShape(validRow, validCol);
    DstStride dstStride(
        static_cast<int64_t>(validRow) * leadingDim, static_cast<int64_t>(validRow) * leadingDim,
        static_cast<int64_t>(validRow) * leadingDim, leadingDim);
    GlobalDataOut dstGlobal(dst, dstShape, dstStride);
    AccTile accTile(validRow, validCol);

    pto::TASSIGN(accTile, accOffset);
    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        pto::TSTORE<AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(dstGlobal, accTile);
    } else {
        pto::TSTORE(dstGlobal, accTile);
    }
}

template <int Cols>
AICORE inline void StagePerChannelScale(
    uint64_t l1SOffset, uint64_t fixpipeOffset, __gm__ uint64_t* gmBlockS, uint32_t cols, event_t eventId)
{
    using ScaleMatTile = pto::Tile<
        pto::TileType::Mat, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;
    using ScalingTile = pto::Tile<
        pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;

    uint32_t validCols = cols;
    using ScaleShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using ScaleStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<uint64_t, ScaleShape, ScaleStride, pto::Layout::ND>;
    ScaleShape scaleShape(validCols);
    ScaleStride scaleStride(validCols, validCols, validCols, validCols);
    ScaleGlobal gmBlockSGlobal(gmBlockS, scaleShape, scaleStride);
    ScaleMatTile scaleMatTile(validCols);
    ScalingTile scalingTile(validCols);

    pto::TASSIGN(scaleMatTile, l1SOffset);
    pto::TASSIGN(scalingTile, fixpipeOffset);
    pto::TLOAD(scaleMatTile, gmBlockSGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, eventId);
    wait_flag(PIPE_MTE2, PIPE_FIX, eventId);
    pto::TMOV(scalingTile, scaleMatTile);
}

template <typename ElementA, typename ElementC, typename ElementAccumulator, int Rows, int Cols>
AICORE inline void StoreAccumulator(
    __gm__ ElementC* dst, uint64_t accOffset, uint64_t scaleOffset, uint32_t validRow, uint32_t validCol,
    uint32_t leadingDim)
{
    if constexpr (std::is_same_v<ElementA, int8_t>) {
        PtoStoreAccToGm<ElementC, ElementAccumulator, Rows, Cols>(
            dst, accOffset, scaleOffset, validRow, validCol, leadingDim);
    } else if constexpr (std::is_same_v<ElementA, half>) {
        PtoStoreAccToGm<ElementC, ElementAccumulator, Rows, Cols>(dst, accOffset, validRow, validCol, leadingDim);
    }
}

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_SHUFFLE_K_, uint32_t L1_M_, uint32_t L1_N_, uint32_t L1_K_, uint32_t L0_M_, uint32_t L0_N_,
    uint32_t L0_K_, typename ElementA_, typename ElementB_, typename ElementC_>
struct PtoGmmPreloadAsyncFixpipe {
public:
    // Type Aliases
    using ArchTag = AtlasA5;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator =
        std::conditional_t<std::is_same_v<ElementA, int8_t> && std::is_same_v<ElementB, int8_t>, int32_t, float>;

    static constexpr bool ASYNC = true;
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;

    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr uint32_t L1_M = L1_M_;
    static constexpr uint32_t L1_N = L1_N_;
    static constexpr uint32_t L1_K = L1_K_;
    static constexpr uint32_t L0_M = L0_M_;
    static constexpr uint32_t L0_N = L0_N_;
    static constexpr uint32_t L0_K = L0_K_;

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_N * L1_K * sizeof(ElementB);
    static constexpr uint32_t L1S_TILE_SIZE = L1_N * sizeof(int64_t);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_M * L0_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_K * L0_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_M * L1_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L1_M_ALIGN = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t L1_N_ALIGN = BYTE_PER_C0 / sizeof(ElementB);

    static_assert(
        (std::is_same_v<ElementA, int8_t> && std::is_same_v<ElementB, int8_t>) ||
            (std::is_same_v<ElementA, half> && std::is_same_v<ElementB, half>) ||
            (std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>) ||
            (std::is_same_v<ElementA, bfloat16_t> && std::is_same_v<ElementB, bfloat16_t>),
        "Unsupported GMM matmul element combination.");
    // Check L1 tile
    static_assert(
        (std::is_same_v<ElementA, int8_t> ?
             (L1A_TILE_SIZE + L1B_TILE_SIZE) * L1_STAGES + L1S_TILE_SIZE * L0C_STAGES <= ArchTag::L1_SIZE :
             (L1A_TILE_SIZE + L1B_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE),
        "L1 tile exceeding the L1 space for the given data type");

    // Check L0 tile
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0 tile exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0 tile exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0 tile exceeding the L0C space!");
    static_assert(L1S_TILE_SIZE * L0C_STAGES <= ArchTag::FIXBUF_SIZE, "Scale tiles exceed the fixpipe buffer space!");

    static_assert(
        L1_M == L0_M && L1_N == L0_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    AICORE inline PtoGmmPreloadAsyncFixpipe(uint32_t l1BufAddrStart = 0, uint32_t FpAddrStart = 0)
    {
        InitL1(l1BufAddrStart);
        InitFpBuf(FpAddrStart);
        InitL0A();
        InitL0B();
        InitL0C();
    }

    AICORE inline ~PtoGmmPreloadAsyncFixpipe()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[i]);
            wait_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            wait_flag(PIPE_M, PIPE_MTE1, l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            wait_flag(PIPE_M, PIPE_MTE1, l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            wait_flag(PIPE_FIX, PIPE_M, l0CEventList[i]);
        }
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                wait_flag(PIPE_FIX, PIPE_MTE2, scaleEventList[i]);
            }
        }
    }

    AICORE inline void operator()(
        __gm__ ElementA* gmBlockAPtr, __gm__ ElementB* gmBlockBPtr, __gm__ ElementC* gmBlockCPtr,
        __gm__ uint64_t* gmBlockSPtr, uint32_t actualM, uint32_t actualN, uint32_t actualK, uint32_t aLeadingDim,
        uint32_t bFullRows, uint32_t bColStart, uint32_t cLeadingDim)
    {
        const KTilePipelineParams params{gmBlockAPtr, gmBlockBPtr, gmBlockCPtr, gmBlockSPtr, actualM,    actualN,
                                         actualK,     aLeadingDim, bFullRows,   bColStart,   cLeadingDim};
        RunKTilePipeline<false>(params, 0);
    }

    template <typename CvDirect>
    AICORE inline void RunDirect(
        __gm__ ElementA* gmBlockAPtr, __gm__ ElementB* gmBlockBPtr, __gm__ uint64_t* gmBlockSPtr, uint32_t actualM,
        uint32_t actualN, uint32_t actualK, uint32_t aLeadingDim, uint32_t bFullRows, uint32_t bColStart,
        const CvDirect& cvDirect)
    {
        const KTilePipelineParams params{gmBlockAPtr, gmBlockBPtr, nullptr,   gmBlockSPtr, actualM, actualN,
                                         actualK,     aLeadingDim, bFullRows, bColStart,   0U};
        RunKTilePipeline<true>(params, cvDirect);
    }

    AICORE inline void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileGmm(l1TileGmmParamsList[l1TileGmmParamsId]);
            l1TileGmmParamsId = (l1TileGmmParamsId + 1 < PRELOAD_STAGES) ? (l1TileGmmParamsId + 1) : 0;
            --preloadCount;
        }
    }

    template <typename CvDirect>
    AICORE inline void SynchronizeBlockDirect(const CvDirect& cvDirect)
    {
        while (preloadCount > 0) {
            L1TileGmmDirect(l1TileGmmParamsList[l1TileGmmParamsId], cvDirect);
            l1TileGmmParamsId = (l1TileGmmParamsId + 1 < PRELOAD_STAGES) ? (l1TileGmmParamsId + 1) : 0;
            --preloadCount;
        }
    }

private:
    struct KTilePipelineParams {
        __gm__ ElementA* gmBlockA;
        __gm__ ElementB* gmBlockB;
        __gm__ ElementC* gmBlockC;
        __gm__ uint64_t* gmBlockS;
        uint32_t actualM;
        uint32_t actualN;
        uint32_t actualK;
        uint32_t aLeadingDim;
        uint32_t bFullRows;
        uint32_t bColStart;
        uint32_t cLeadingDim;
    };

    struct L1TileGmmParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        uint32_t actualM;
        uint32_t actualN;
        uint32_t cLeadingDim = 0;
        bool isKLoopFirst;
        bool isKLoopLast;
        __gm__ ElementC* gmBlockC;
        __gm__ uint64_t* gmBlockS;
        AICORE inline L1TileGmmParams() = default;
    };

    template <bool Direct, typename CvDirect>
    AICORE inline void RunKTilePipeline(const KTilePipelineParams& input, const CvDirect& cvDirect)
    {
        const uint32_t kTileCount = ceilDiv<L1_K>(input.actualK);
        const uint32_t mRound = roundUp<L1_M_ALIGN>(input.actualM);
        const uint32_t nRound = roundUp<L1_N_ALIGN>(input.actualN);

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = get_block_idx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            const uint32_t candidateTileIdx = startTileIdx + kLoopIdx;
            const uint32_t kTileIdx = candidateTileIdx < kTileCount ? candidateTileIdx : candidateTileIdx - kTileCount;
            const uint32_t kActual = kTileIdx < kTileCount - 1U ? L1_K : input.actualK - kTileIdx * L1_K;
            const uint32_t kStart = kTileIdx * L1_K;
            __gm__ ElementA* gmTileA = input.gmBlockA + kStart;
            constexpr uint32_t bElemsPerC0 = BYTE_PER_C0 / sizeof(ElementB);
            const uint32_t bRowsAligned = roundUp<C0_NUM_PER_FRACTAL>(input.bFullRows);
            const uint64_t gmTileBOffset =
                static_cast<uint64_t>(kStart / C0_NUM_PER_FRACTAL) * (BYTE_PER_FRACTAL / sizeof(ElementB)) +
                static_cast<uint64_t>(input.bColStart / bElemsPerC0) * bRowsAligned * bElemsPerC0 +
                static_cast<uint64_t>(kStart % C0_NUM_PER_FRACTAL) * bElemsPerC0 + input.bColStart % bElemsPerC0;
            __gm__ ElementB* gmTileB = input.gmBlockB + gmTileBOffset;

            wait_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[l1ListId]);
            PtoLoadNdGmToNzL1<ElementA, L1_M, L1_K>(
                l1AOffsetList[l1ListId], gmTileA, input.actualM, kActual, input.aLeadingDim);
            set_flag(PIPE_MTE2, PIPE_MTE1, l1AEventList[l1ListId]);
            wait_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[l1ListId]);
            PtoLoadPackedWeightGmToL1<ElementB, L1_K, L1_N>(
                l1BOffsetList[l1ListId], gmTileB, kActual, input.actualN, input.bFullRows);
            set_flag(PIPE_MTE2, PIPE_MTE1, l1BEventList[l1ListId]);

            if (preloadCount == PRELOAD_STAGES) {
                if constexpr (Direct) {
                    L1TileGmmDirect(l1TileGmmParamsList[l1TileGmmParamsId], cvDirect);
                } else {
                    L1TileGmm(l1TileGmmParamsList[l1TileGmmParamsId]);
                }
            }

            const uint32_t preloadParamsId = l1TileGmmParamsId + preloadCount < PRELOAD_STAGES ?
                                                 l1TileGmmParamsId + preloadCount :
                                                 l1TileGmmParamsId + preloadCount - PRELOAD_STAGES;
            auto& params = l1TileGmmParamsList[preloadParamsId];
            params.l1ListId = l1ListId;
            params.mRound = mRound;
            params.nRound = nRound;
            params.kActual = kActual;
            params.actualM = input.actualM;
            params.actualN = input.actualN;
            params.isKLoopFirst = kLoopIdx == 0U;
            params.isKLoopLast = kLoopIdx == kTileCount - 1U;
            if constexpr (!Direct) {
                params.cLeadingDim = input.cLeadingDim;
            }
            if (params.isKLoopLast) {
                params.gmBlockS = input.gmBlockS;
                if constexpr (!Direct) {
                    params.gmBlockC = input.gmBlockC;
                }
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileGmmParamsId = l1TileGmmParamsId + 1U < PRELOAD_STAGES ? l1TileGmmParamsId + 1U : 0U;
            }
            l1ListId = l1ListId + 1U < L1_STAGES ? l1ListId + 1U : 0U;
        }
    }

    AICORE inline void InitL1(uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;

        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1AOffsetList[i] = l1AOffset + L1A_TILE_SIZE * i;
            l1BOffsetList[i] = l1BOffset + L1B_TILE_SIZE * i;
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            set_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[i]);
            set_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[i]);
        }
        uint32_t l1SOffset = l1BOffset + L1B_TILE_SIZE * L1_STAGES;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                l1SOffsetList[i] = l1SOffset + L1S_TILE_SIZE * i;
                scaleEventList[i] = i;
                set_flag(PIPE_FIX, PIPE_MTE2, scaleEventList[i]);
            }
        }
    }

    AICORE inline void InitFpBuf(uint32_t FpAddrStart)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            fixpipeOffsetList[i] = FpAddrStart + L1S_TILE_SIZE * i;
        }
    }

    AICORE inline void InitL0A()
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0AOffsetList[i] = L0A_TILE_SIZE * i;
            l0AEventList[i] = i;
            set_flag(PIPE_M, PIPE_MTE1, l0AEventList[i]);
        }
    }

    AICORE inline void InitL0B()
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BOffsetList[i] = L0B_TILE_SIZE * i;
            l0BEventList[i] = i + L0A_STAGES;
            set_flag(PIPE_M, PIPE_MTE1, l0BEventList[i]);
        }
    }

    AICORE inline void InitL0C()
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0COffsetList[i] = L0C_TILE_SIZE * i;
            l0CEventList[i] = i;
            set_flag(PIPE_FIX, PIPE_M, l0CEventList[i]);
        }
    }

    AICORE inline void L1TileGmm(L1TileGmmParams const& params)
    {
        if (!PrepareL1TileGmmResult(params)) {
            return;
        }
        StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1_M, L1_N>(
            params.gmBlockC, l0COffsetList[l0CListId], fixpipeOffsetList[l0CListId], params.actualM, params.actualN,
            params.cLeadingDim);
        ReleaseL1TileGmmResult();
    }

    template <typename CvDirect>
    AICORE inline void L1TileGmmDirect(L1TileGmmParams const& params, const CvDirect& cvDirect)
    {
        if (!PrepareL1TileGmmResult(params)) {
            return;
        }
        cvDirect.template Store<ElementA, ElementC, ElementAccumulator, L1_M, L1_N>(
            l0COffsetList[l0CListId], fixpipeOffsetList[l0CListId], params.actualM, params.actualN);
        ReleaseL1TileGmmResult();
    }

    AICORE inline bool PrepareL1TileGmmResult(L1TileGmmParams const& params)
    {
        uint32_t mPartLoop = ceilDiv<L0_M>(params.mRound);
        uint32_t nPartLoop = ceilDiv<L0_N>(params.nRound);
        uint32_t kPartLoop = ceilDiv<L0_K>(params.kActual);
        using APanelTile = pto::Tile<
            pto::TileType::Mat, ElementA, L1_M, L1_K, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
            pto::SLayout::RowMajor>;
        using BPanelTile = pto::Tile<
            pto::TileType::Mat, ElementB, L1_K, L1_N, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
            pto::SLayout::RowMajor>;
        using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, pto::DYNAMIC>;
        using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        using AccPanelTile = pto::TileAccCompact<ElementAccumulator, L1_M, L1_N, pto::DYNAMIC, pto::DYNAMIC>;
        using AccTile = pto::TileAccCompact<ElementAccumulator, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        APanelTile aPanel(params.mRound, params.kActual);
        BPanelTile bPanel(params.kActual, params.nRound);
        AccPanelTile accPanel(params.mRound, params.nRound);
        pto::TASSIGN(aPanel, l1AOffsetList[params.l1ListId]); // 双缓冲
        pto::TASSIGN(bPanel, l1BOffsetList[params.l1ListId]);
        pto::TASSIGN(accPanel, l0COffsetList[l0CListId]);

        if (params.isKLoopFirst) {
            wait_flag(PIPE_FIX, PIPE_M, l0CEventList[l0CListId]);
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0_M : (params.mRound - mPartIdx * L0_M);

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) { // K 维继续切分
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0_K : (params.kActual - kPartIdx * L0_K);

                LeftTile aTile(mPartActual, kPartActual);
                pto::TASSIGN(aTile, l0AOffsetList[l0AListId]);

                wait_flag(PIPE_M, PIPE_MTE1, l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    wait_flag(PIPE_MTE2, PIPE_MTE1, l1AEventList[params.l1ListId]);
                }
                pto::TEXTRACT(aTile, aPanel, mPartIdx * L0_M, kPartIdx * L0_K);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0_N : (params.nRound - nPartIdx * L0_N);

                    RightTile bTile(kPartActual, nPartActual);
                    pto::TASSIGN(bTile, l0BOffsetList[l0BListId]);

                    wait_flag(PIPE_M, PIPE_MTE1, l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, l1BEventList[params.l1ListId]);
                    }
                    pto::TEXTRACT(bTile, bPanel, kPartIdx * L0_K, nPartIdx * L0_N);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[params.l1ListId]);
                    }

                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    AccTile cTile(mPartActual, nPartActual);
                    pto::TSUBVIEW(cTile, accPanel, mPartIdx * L0_M, nPartIdx * L0_N);
                    LaunchPtoMatmul(cTile, aTile, bTile, initC);

                    constexpr uint32_t kPipeBarrierThreshold = 10;
                    constexpr uint32_t kFractalEdge = 16;
                    if ((mPartActual / kFractalEdge) * (nPartActual / kFractalEdge) < kPipeBarrierThreshold) {
                        pipe_barrier(PIPE_M);
                    }

                    set_flag(PIPE_M, PIPE_MTE1, l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                set_flag(PIPE_M, PIPE_MTE1, l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (!params.isKLoopLast) {
            return false;
        }
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            const event_t scaleEvent = static_cast<event_t>(scaleEventList[l0CListId]);
            wait_flag(PIPE_FIX, PIPE_MTE2, scaleEvent);
            StagePerChannelScale<L1_N>(
                l1SOffsetList[l0CListId], fixpipeOffsetList[l0CListId], params.gmBlockS, params.actualN, scaleEvent);
            if constexpr (L0C_STAGES == 1U) {
                set_flag(PIPE_MTE2, PIPE_FIX, scaleEvent);
                wait_flag(PIPE_MTE2, PIPE_FIX, scaleEvent);
                pipe_barrier(PIPE_FIX);
            }
        }
        set_flag(PIPE_M, PIPE_FIX, l0CEventList[l0CListId]);
        wait_flag(PIPE_M, PIPE_FIX, l0CEventList[l0CListId]);
        return true;
    }

    AICORE inline void ReleaseL1TileGmmResult()
    {
        set_flag(PIPE_FIX, PIPE_M, l0CEventList[l0CListId]);
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            set_flag(PIPE_FIX, PIPE_MTE2, scaleEventList[l0CListId]);
        }
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

    uint64_t fixpipeOffsetList[L0C_STAGES];

    uint64_t l1AOffsetList[L1_STAGES];
    uint64_t l1BOffsetList[L1_STAGES];
    uint64_t l1SOffsetList[L0C_STAGES];
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    uint64_t l0AOffsetList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    uint64_t l0BOffsetList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    uint64_t l0COffsetList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    int32_t scaleEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileGmmParams l1TileGmmParamsList[PRELOAD_STAGES];
    uint32_t l1TileGmmParamsId{0};
    uint32_t preloadCount{0};
};

#endif // DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_COMMON_H
#define DISPATCH_MEGA_COMBINE_GMM_COMMON_H

#include "kernel_operator.h"

#include "utils/common_helpers.hpp"
#include "utils/pto_gmm_preload_async_fixpipe_quant.hpp"

constexpr uint32_t kGmmCommonSwizzleOffset = 9U;

template <uint32_t L0CStages>
using GmmCommonPipelineWithL0CStages =
    PtoGmmPreloadAsyncFixpipe<1, 2, 2, 2, L0CStages, true, 128, 256, 512, 128, 256, 128, int8_t, int8_t, half>;

using GmmCommonPipeline = GmmCommonPipelineWithL0CStages<1>;
using GmmCommonL0CDoubleBufferPipeline = GmmCommonPipelineWithL0CStages<2>;

struct GmmCommonTileInfo {
    uint32_t actualM = 0;
    uint32_t actualN = 0;
    uint32_t blockRowStart = 0;
    uint32_t blockColStart = 0;
};

AICORE inline uint32_t GmmCommonTileM(uint32_t currentM, uint32_t l1TileM)
{
    return static_cast<uint32_t>(ceilDiv(currentM, l1TileM));
}

AICORE inline uint32_t GmmCommonTileN(uint32_t problemN, uint32_t l1TileN)
{
    return static_cast<uint32_t>(ceilDiv(problemN, l1TileN));
}

AICORE inline uint32_t GmmCommonCoreLoops(uint32_t currentM, uint32_t problemN, uint32_t l1TileM, uint32_t l1TileN)
{
    return GmmCommonTileM(currentM, l1TileM) * GmmCommonTileN(problemN, l1TileN);
}

AICORE inline uint32_t GmmCommonStartLoopIdx(uint32_t coreIdx, uint32_t coreNum, uint32_t startCoreIdx)
{
    return ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
}

AICORE inline void GmmCommonGetBlockCoordMN(
    uint32_t loopIdx, uint32_t tileM, uint32_t tileN, uint32_t& blockM, uint32_t& blockN)
{
    const uint32_t tileBlockLoop = static_cast<uint32_t>(ceilDiv(tileN, kGmmCommonSwizzleOffset));
    const uint32_t tileBlockIdx = loopIdx / (kGmmCommonSwizzleOffset * tileM);
    const uint32_t inTileBlockIdx = loopIdx % (kGmmCommonSwizzleOffset * tileM);
    uint32_t nCol = kGmmCommonSwizzleOffset;
    if (tileBlockIdx + 1U == tileBlockLoop) {
        nCol = tileN - kGmmCommonSwizzleOffset * tileBlockIdx;
    }
    blockM = inTileBlockIdx / nCol;
    blockN = tileBlockIdx * kGmmCommonSwizzleOffset + inTileBlockIdx % nCol;
    if ((tileBlockIdx & 1U) != 0U) {
        blockM = tileM - blockM - 1U;
    }
}

AICORE inline void GmmCommonGetActualBlockShapeMN(
    uint32_t blockM, uint32_t blockN, uint32_t tileM, uint32_t tileN, uint32_t currentM, uint32_t problemN,
    uint32_t l1TileM, uint32_t l1TileN, uint32_t& actualM, uint32_t& actualN)
{
    actualM = (blockM + 1U == tileM) ? (currentM - blockM * l1TileM) : l1TileM;
    actualN = (blockN + 1U == tileN) ? (problemN - blockN * l1TileN) : l1TileN;
}

AICORE inline GmmCommonTileInfo GmmCommonBuildTileInfo(
    uint32_t currentM, uint32_t problemN, uint32_t l1TileM, uint32_t l1TileN, uint32_t loopIdx)
{
    GmmCommonTileInfo info;
    const uint32_t tileM = GmmCommonTileM(currentM, l1TileM);
    const uint32_t tileN = GmmCommonTileN(problemN, l1TileN);
    uint32_t blockM = 0;
    uint32_t blockN = 0;
    GmmCommonGetBlockCoordMN(loopIdx, tileM, tileN, blockM, blockN);
    GmmCommonGetActualBlockShapeMN(
        blockM, blockN, tileM, tileN, currentM, problemN, l1TileM, l1TileN, info.actualM, info.actualN);
    info.blockRowStart = blockM * l1TileM;
    info.blockColStart = blockN * l1TileN;
    return info;
}

AICORE inline uint64_t GmmCommonPackedWeightExpertStride(uint32_t kRows, uint32_t nCols)
{
    constexpr uint32_t kKAlign = C0_NUM_PER_FRACTAL;
    constexpr uint32_t kNAlign = BYTE_PER_C0 / sizeof(int8_t);
    return static_cast<uint64_t>(roundUp<kKAlign>(kRows)) * roundUp<kNAlign>(nCols);
}

AICORE inline uint32_t MoeCurrentMRaw(
    __gm__ int32_t* cumsumMMPtr, uint32_t rankSize, uint32_t expertPerRank, uint32_t groupIdx)
{
    return static_cast<uint32_t>(cumsumMMPtr[static_cast<uint64_t>(rankSize - 1U) * expertPerRank + groupIdx]);
}

AICORE inline uint32_t MoeClipCurrentM(uint32_t currentMRaw, uint32_t groupBase, uint32_t maxOutputSize)
{
    if (groupBase >= maxOutputSize) {
        return 0U;
    }
    const uint32_t remaining = maxOutputSize - groupBase;
    return currentMRaw > remaining ? remaining : currentMRaw;
}

template <typename GmmPipeline>
AICORE inline void GmmCommonRunTile(
    GmmPipeline& gmmPipeline, __gm__ int8_t* gmAPtr, __gm__ int8_t* gmWeightPtr, __gm__ half* gmCPtr,
    __gm__ uint64_t* gmScalePtr, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, uint32_t loopIdx,
    uint32_t problemN, uint32_t actualK, uint32_t aLeadingDim, uint32_t bFullRows, uint32_t cLeadingDim,
    uint32_t scaleGroupStride, uint32_t l1TileM, uint32_t l1TileN)
{
    const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemN, l1TileM, l1TileN, loopIdx);

    const uint64_t gmOffsetA = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * aLeadingDim;
    const uint64_t gmOffsetB = static_cast<uint64_t>(groupIdx) * GmmCommonPackedWeightExpertStride(bFullRows, problemN);
    const uint64_t gmOffsetC = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * cLeadingDim +
                               static_cast<uint64_t>(tileInfo.blockColStart);
    const uint64_t gmOffsetS = static_cast<uint64_t>(groupIdx) * scaleGroupStride + tileInfo.blockColStart;

    gmmPipeline(
        gmAPtr + gmOffsetA, gmWeightPtr + gmOffsetB, gmCPtr + gmOffsetC, gmScalePtr + gmOffsetS, tileInfo.actualM,
        tileInfo.actualN, actualK, aLeadingDim, bFullRows, tileInfo.blockColStart, cLeadingDim);
}

template <typename GmmPipeline, typename CvDirect>
AICORE inline void GmmCommonRunTileDirect(
    GmmPipeline& gmmPipeline, __gm__ int8_t* gmAPtr, __gm__ int8_t* gmWeightPtr, __gm__ uint64_t* gmScalePtr,
    uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, uint32_t loopIdx, uint32_t problemN, uint32_t actualK,
    uint32_t aLeadingDim, uint32_t bFullRows, uint32_t scaleGroupStride, uint32_t l1TileM, uint32_t l1TileN,
    const CvDirect& cvDirect)
{
    const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemN, l1TileM, l1TileN, loopIdx);

    const uint64_t gmOffsetA = static_cast<uint64_t>(groupBase + tileInfo.blockRowStart) * aLeadingDim;
    const uint64_t gmOffsetB = static_cast<uint64_t>(groupIdx) * GmmCommonPackedWeightExpertStride(bFullRows, problemN);
    const uint64_t gmOffsetS = static_cast<uint64_t>(groupIdx) * scaleGroupStride + tileInfo.blockColStart;

    gmmPipeline.RunDirect(
        gmAPtr + gmOffsetA, gmWeightPtr + gmOffsetB, gmScalePtr + gmOffsetS, tileInfo.actualM, tileInfo.actualN,
        actualK, aLeadingDim, bFullRows, tileInfo.blockColStart, cvDirect);
}

#endif // DISPATCH_MEGA_COMBINE_GMM_COMMON_H

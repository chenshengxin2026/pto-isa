/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_H
#define DISPATCH_MEGA_COMBINE_GMM2_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_sync_substrate.hpp"

using Gmm2Pipeline = GmmCommonL0CDoubleBufferPipeline;

class Gmm2 {
public:
    AICORE inline void Init(
        GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void Process();

private:
    AICORE inline void WaitV2CReady() const
    {
        CrossCoreWaitFlag<0x2>(MEGA_MOE_V2C_HARD_FLAG_BASE);
        pipe_barrier(PIPE_ALL);
    }
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    struct Gmm2CvDirectStore {
        Gmm2CombineCvProducer& producer;

        AICORE inline explicit Gmm2CvDirectStore(Gmm2CombineCvProducer& cvProducer) : producer(cvProducer) {}

        template <typename ElementA, typename ElementC, typename ElementAccumulator, int Rows, int Cols>
        AICORE inline void Store(uint64_t accOffset, uint64_t scaleOffset, uint32_t validRow, uint32_t validCol) const
        {
            static_assert(std::is_same_v<ElementA, int8_t>);
            static_assert(std::is_same_v<ElementC, half>);
            static_assert(std::is_same_v<ElementAccumulator, int32_t>);
            using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
            using CvTile = pto::Tile<
                pto::TileType::Vec, ElementC, Rows, Cols, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
                pto::SLayout::NoneBox>;
            using ScalingTile = pto::Tile<
                pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC,
                pto::SLayout::NoneBox>;

            const uint32_t lane = producer.NextLane();
            producer.WaitLaneFree();

            AccTile accTile(validRow, validCol);
            CvTile cvTile(validRow, validCol);
            ScalingTile scalingTile(validCol);
            pto::TASSIGN(accTile, accOffset);
            pto::TASSIGN(cvTile, MEGA_MOE_GMM2_COMBINE_CV_SLOT_OFFSET);
            pto::TASSIGN(scalingTile, scaleOffset);

            if (lane == 0U) {
                pto::TMOV<CvTile, AccTile, ScalingTile, pto::AccToVecMode::SingleModeVec0>(
                    cvTile, accTile, scalingTile);
            } else {
                pto::TMOV<CvTile, AccTile, ScalingTile, pto::AccToVecMode::SingleModeVec1>(
                    cvTile, accTile, scalingTile);
            }
            producer.RecordLaneReady();
        }
    };
    AICORE inline void RunGmmTileDirect(
        Gmm2Pipeline& gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, uint32_t loopIdx,
        const Gmm2CvDirectStore& cvDirect) const
    {
        GmmCommonRunTileDirect(
            gmmPipeline, gmPermutedTokenPtr_, weight2Ptr_, scale2Ptr_, groupIdx, groupBase, currentM, loopIdx, outputN_,
            inputK_, inputK_, inputK_, outputN_, tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN,
            cvDirect);
    }
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    __gm__ int8_t* gmPermutedTokenPtr_ = nullptr;
    __gm__ int8_t* weight2Ptr_ = nullptr;
    __gm__ uint64_t* scale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;

    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

AICORE inline void Gmm2::Init(
    GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    tilingData_ = tilingData;

    inputK_ = tilingData_->megaMoeInfo.N / 2U;
    outputN_ = tilingData_->megaMoeInfo.K;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->swigluTiling.gmPermutedTokenOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ int8_t*>(weight2GM);
    scale2Ptr_ = reinterpret_cast<__gm__ uint64_t*>(scale2GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}
AICORE inline void Gmm2::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    Gmm2Pipeline gmmPipeline;
    Gmm2CombineCvProducer cvProducer;
    Gmm2CvDirectStore cvDirect(cvProducer);
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t segmentNum = MoeSwigluSegmentNum(expertPerRank_);
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        WaitV2CReady();

        const uint32_t segmentStartExpert = MoeSwigluSegmentStartExpert(expertPerRank_, segmentIdx);
        const uint32_t segmentEndExpert = MoeSwigluSegmentEndExpert(expertPerRank_, segmentIdx);

        for (uint32_t groupIdx = segmentStartExpert; groupIdx < segmentEndExpert; ++groupIdx) {
            const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
            const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                RunGmmTileDirect(gmmPipeline, groupIdx, groupBase, currentM, loopIdx, cvDirect);
            }
            gmmPipeline.SynchronizeBlockDirect(cvDirect);
            groupBase += currentM;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum_;
        }
    }
    cvProducer.Drain();
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_H

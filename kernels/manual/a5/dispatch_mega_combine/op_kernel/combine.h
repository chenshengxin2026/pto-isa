/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_H
#define DISPATCH_MEGA_COMBINE_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/peer_memory_layout.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kCombineBufferNum = 2U;
constexpr uint32_t kCombineSmallTokenSubtileRows = 16U;
constexpr uint32_t kCombineSmallTokenSubtileCols = 256U;
constexpr uint32_t kCombineSmallMaxElems = 8192U;
static_assert(MEGA_MOE_GMM2_COMBINE_CV_SCALE_CACHE_ELEMS % MEGA_MOE_GMM2_COMBINE_CV_TILE_M == 0U);
constexpr uint32_t kCombineCvScaleMaxRowBlocks =
    MEGA_MOE_GMM2_COMBINE_CV_SCALE_CACHE_ELEMS / MEGA_MOE_GMM2_COMBINE_CV_TILE_M;
static_assert(kCombineCvScaleMaxRowBlocks <= 64U);
static_assert(GmmCommonPipeline::L1_M == MEGA_MOE_GMM2_COMBINE_CV_TILE_M);

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void Process();

private:
    static_assert(
        std::is_same_v<OutputElement, half> || std::is_same_v<OutputElement, bfloat16_t>,
        "combine output must be half or bfloat16");

    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<float, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DBlockGlobal = pto::GlobalTensor<OutputElement, BlockShape, BlockStride, pto::Layout::ND>;
    using SmallTileC = pto::Tile<
        pto::TileType::Vec, half, kCombineSmallTokenSubtileRows, kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor,
        -1, -1>;
    using SmallTileFp32 = pto::Tile<
        pto::TileType::Vec, float, kCombineSmallTokenSubtileRows, kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor,
        -1, -1>;
    using SmallTileD = pto::Tile<
        pto::TileType::Vec, OutputElement, kCombineSmallTokenSubtileRows, kCombineSmallTokenSubtileCols,
        pto::BLayout::RowMajor, -1, -1>;
    using CvTile = pto::Tile<
        pto::TileType::Vec, half, MEGA_MOE_GMM2_COMBINE_CV_TILE_M, MEGA_MOE_GMM2_COMBINE_CV_TILE_N,
        pto::BLayout::RowMajor, -1, -1>;
    using CvScaleTile = pto::Tile<
        pto::TileType::Vec, float, 1, MEGA_MOE_GMM2_COMBINE_CV_TILE_M, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
    struct CvScaleBlockInfo {
        uint32_t tileRowOffset;
        uint32_t srcRowOffset;
        uint32_t rowNum;
        uint32_t cacheRowOffset;
    };

    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId + 2U); }
    AICORE inline event_t CvScaleReadyEvent() const { return EVENT_ID4; }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline uint32_t TokenPerExpertResetElems() const;
    AICORE inline bool ResetTokenPerExpert(uint32_t elems) const;
    AICORE inline void ProcessFinalBoundary();
    AICORE inline void PopCvTile(Gmm2CombineCvPipe& cvPipe, uint32_t actualM, uint32_t actualN) const;
    AICORE inline void IssueCvScaleBlockLoad(uint32_t srcRowOffset, uint32_t rowNum, uint32_t cacheRowOffset) const;
    AICORE inline bool BuildCvAssignedScaleBlocks(
        CvScaleBlockInfo* scaleBlocks, uint32_t& scaleBlockCount, uint32_t groupBase, uint32_t currentM,
        uint32_t startLoopIdx, uint32_t coreLoops, uint32_t aicCoreNum, uint32_t l1TileM, uint32_t l1TileN,
        uint32_t firstCvTileSequence, uint32_t aivSubCoreIdx) const;
    AICORE inline uint32_t FindCvScaleCacheRowOffset(
        const CvScaleBlockInfo* scaleBlocks, uint32_t scaleBlockCount, uint32_t tileRowOffset) const;
    AICORE inline void IssueCvScaleBlockLoads(const CvScaleBlockInfo* scaleBlocks, uint32_t scaleBlockCount) const;
    AICORE inline void WaitCvScaleBlockLoads() const;
    AICORE inline void PrepareCvSmallSubtile(
        uint32_t bufferId, uint32_t cvRowInTile, uint32_t scaleRowOffset, uint32_t rowNum, uint32_t colNum) const;
    AICORE inline void StoreSmallSubtileIntersection(
        uint32_t bufferId, __gm__ OutputElement* dstBase, uint32_t dstRowOffset, uint32_t ubRowOffset, uint32_t rowNum,
        uint32_t colBegin, uint32_t colNum);
    AICORE inline void StoreSmallSubtileToRanks(
        uint32_t groupIdx, const GmmCommonTileInfo& tileInfo, uint32_t tileRowBegin, uint32_t rows, uint32_t bufferId);
    AICORE inline void ProcessDirectTokenPath();

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    MegaMoePeerRowMapper rowMapper_;
    __gm__ float* perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t* tokenPerExpertPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t l1TileM_ = 0;
    uint32_t l1TileN_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t pingpongId_ = 0;
    uint32_t cvTileSequence_ = 0;
    uint64_t ubFp32Offset_[kCombineBufferNum] = {0, 0};
    uint64_t ubDOffset_[kCombineBufferNum] = {0, 0};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    pingpongId_ = 0;
    cvTileSequence_ = 0;

    problemK_ = tilingData->megaMoeInfo.K;
    maxOutputSize_ = tilingData->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData->megaMoeInfo.expertPerRank;
    expertNumAligned_ = tilingData->frontReorderTiling.expertNumAligned;
    rankSize_ = tilingData->runtimeInfo.rankSize;
    l1TileM_ = tilingData->gmm2Tiling.l1TileM;
    l1TileN_ = tilingData->gmm2Tiling.l1TileN;
    coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
    coreNum_ = get_block_num() * get_subblockdim();

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_);

    perTokenScale2Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tilingData->swigluTiling.perTokenScale2Offset);
    __gm__ int32_t* cumsumMM =
        reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
    __gm__ int32_t* preSumBeforeRank =
        reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.preSumBeforeRankOffset);
    tokenPerExpertPtr_ =
        reinterpret_cast<__gm__ int32_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    rowMapper_.Init(
        cumsumMM, preSumBeforeRank, tokenPerExpertPtr_, tilingData->runtimeInfo.rank, rankSize_, expertPerRank_,
        tilingData->frontReorderTiling.expertNumAligned);

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::InitUbLayout()
{
    uint64_t ubOffset = 0;
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        ubDOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(OutputElement), UB_ALIGN);
        ubFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(float), UB_ALIGN);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::TokenPerExpertResetElems() const
{
    return rankSize_ * expertNumAligned_;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::ResetTokenPerExpert(uint32_t elems) const
{
    if (coreIdx_ != coreNum_ - 1U) {
        return false;
    }
    PtoFillUb<int32_t>(0U, 0, elems);
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<int32_t>(tokenPerExpertPtr_, 0U, elems);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFinalBoundary()
{
    FinalizeLocalPipe();
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    (void)ResetTokenPerExpert(TokenPerExpertResetElems());
    remoteWindow_.CrossRankSync();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PopCvTile(
    Gmm2CombineCvPipe& cvPipe, uint32_t actualM, uint32_t actualN) const
{
    CvTile cvTile(actualM, actualN);
    pto::TPOP<Gmm2CombineCvPipe, CvTile, pto::TileSplitAxis::TILE_NO_SPLIT>(cvPipe, cvTile);
    pipe_barrier(PIPE_ALL);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::IssueCvScaleBlockLoad(
    uint32_t srcRowOffset, uint32_t rowNum, uint32_t cacheRowOffset) const
{
    CvScaleTile scaleTile(rowNum);
    pto::TASSIGN(
        scaleTile, MEGA_MOE_GMM2_COMBINE_CV_TILE_SCALE_OFFSET + static_cast<uint64_t>(cacheRowOffset) * sizeof(float));
    VectorShape scaleShape(rowNum);
    VectorStride scaleStride(rowNum, rowNum, rowNum, rowNum);
    ScaleGlobal scaleGlobal(perTokenScale2Ptr_ + srcRowOffset, scaleShape, scaleStride);
    pto::TLOAD(scaleTile, scaleGlobal);
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::BuildCvAssignedScaleBlocks(
    CvScaleBlockInfo* scaleBlocks, uint32_t& scaleBlockCount, uint32_t groupBase, uint32_t currentM,
    uint32_t startLoopIdx, uint32_t coreLoops, uint32_t aicCoreNum, uint32_t l1TileM, uint32_t l1TileN,
    uint32_t firstCvTileSequence, uint32_t aivSubCoreIdx) const
{
    scaleBlockCount = 0U;
    uint32_t cvTileSequence = firstCvTileSequence;
    for (uint32_t scanLoopIdx = startLoopIdx; scanLoopIdx < coreLoops; scanLoopIdx += aicCoreNum) {
        const uint32_t tileLane = Gmm2CombineCvLane(cvTileSequence);
        ++cvTileSequence;
        if (tileLane != aivSubCoreIdx) {
            continue;
        }
        const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemK_, l1TileM, l1TileN, scanLoopIdx);
        bool alreadyLoaded = false;
        for (uint32_t blockIdx = 0U; blockIdx < scaleBlockCount; ++blockIdx) {
            if (scaleBlocks[blockIdx].tileRowOffset == tileInfo.blockRowStart) {
                alreadyLoaded = true;
                break;
            }
        }
        if (alreadyLoaded) {
            continue;
        }
        if (scaleBlockCount >= kCombineCvScaleMaxRowBlocks) {
            return false;
        }
        scaleBlocks[scaleBlockCount].tileRowOffset = tileInfo.blockRowStart;
        scaleBlocks[scaleBlockCount].srcRowOffset = groupBase + tileInfo.blockRowStart;
        scaleBlocks[scaleBlockCount].rowNum = tileInfo.actualM;
        scaleBlocks[scaleBlockCount].cacheRowOffset = scaleBlockCount * MEGA_MOE_GMM2_COMBINE_CV_TILE_M;
        ++scaleBlockCount;
    }
    return true;
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::FindCvScaleCacheRowOffset(
    const CvScaleBlockInfo* scaleBlocks, uint32_t scaleBlockCount, uint32_t tileRowOffset) const
{
    for (uint32_t blockIdx = 0U; blockIdx < scaleBlockCount; ++blockIdx) {
        if (scaleBlocks[blockIdx].tileRowOffset == tileRowOffset) {
            return scaleBlocks[blockIdx].cacheRowOffset;
        }
    }
    return 0U;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::IssueCvScaleBlockLoads(
    const CvScaleBlockInfo* scaleBlocks, uint32_t scaleBlockCount) const
{
    for (uint32_t blockIdx = 0; blockIdx < scaleBlockCount; ++blockIdx) {
        IssueCvScaleBlockLoad(
            scaleBlocks[blockIdx].srcRowOffset, scaleBlocks[blockIdx].rowNum, scaleBlocks[blockIdx].cacheRowOffset);
    }
    set_flag(PIPE_MTE2, PIPE_S, CvScaleReadyEvent());
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::WaitCvScaleBlockLoads() const
{
    wait_flag(PIPE_MTE2, PIPE_S, CvScaleReadyEvent());
    pipe_barrier(PIPE_ALL);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrepareCvSmallSubtile(
    uint32_t bufferId, uint32_t cvRowInTile, uint32_t scaleRowOffset, uint32_t rowNum, uint32_t colNum) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    SmallTileFp32 fp32Tile(rowNum, colNum);
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
    pto::TASSIGN(
        cTile, MEGA_MOE_GMM2_COMBINE_CV_SLOT_OFFSET +
                   static_cast<uint64_t>(cvRowInTile) * kCombineSmallTokenSubtileCols * sizeof(half));
    pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    // A5 V queue is ordered; this V->MTE2 event releases the buffer after TCVT without a global barrier.
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    CvScaleTile scaleTile(rowNum);
    pto::TASSIGN(
        scaleTile, MEGA_MOE_GMM2_COMBINE_CV_TILE_SCALE_OFFSET + static_cast<uint64_t>(scaleRowOffset) * sizeof(float));
    for (uint32_t row = 0; row < rowNum; ++row) {
        const float scale = scaleTile.GetValue(row);
        SmallTileFp32 rowTile(1, colNum);
        pto::TASSIGN(
            rowTile,
            ubFp32Offset_[bufferId] + static_cast<uint64_t>(row) * kCombineSmallTokenSubtileCols * sizeof(float));
        pto::TMULS(rowTile, rowTile, scale);
    }
    pipe_barrier(PIPE_ALL);

    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId]);
    pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreSmallSubtileIntersection(
    uint32_t bufferId, __gm__ OutputElement* dstBase, uint32_t dstRowOffset, uint32_t ubRowOffset, uint32_t rowNum,
    uint32_t colBegin, uint32_t colNum)
{
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(
        dTile, ubDOffset_[bufferId] +
                   static_cast<uint64_t>(ubRowOffset) * kCombineSmallTokenSubtileCols * sizeof(OutputElement));
    BlockShape dShape(rowNum, colNum);
    BlockStride dStride(
        static_cast<int64_t>(rowNum) * problemK_, static_cast<int64_t>(rowNum) * problemK_,
        static_cast<int64_t>(rowNum) * problemK_, problemK_);
    DBlockGlobal dGlobal(dstBase + static_cast<uint64_t>(dstRowOffset) * problemK_ + colBegin, dShape, dStride);
    pto::TSTORE(dGlobal, dTile);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreSmallSubtileToRanks(
    uint32_t groupIdx, const GmmCommonTileInfo& tileInfo, uint32_t tileRowBegin, uint32_t rows, uint32_t bufferId)
{
    const uint32_t tileEnd = tileRowBegin + rows;
    uint32_t rankRowBegin = 0U;
    uint32_t tileOffset = 0U;
    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        const uint32_t rankRows = rowMapper_.RowsRaw(srcRank, groupIdx);
        const uint32_t rankRowEnd = rankRowBegin + rankRows;
        if (rankRowBegin >= tileEnd) {
            break;
        }
        if (rankRowEnd <= tileRowBegin) {
            rankRowBegin = rankRowEnd;
            continue;
        }
        const uint32_t intersectionBegin = rankRowBegin > tileRowBegin ? rankRowBegin : tileRowBegin;
        const uint32_t intersectionEnd = rankRowEnd < tileEnd ? rankRowEnd : tileEnd;
        const uint32_t intersectionRows = intersectionEnd - intersectionBegin;
        const uint32_t dstRow = rowMapper_.DstRowOffset(srcRank, groupIdx) + intersectionBegin - rankRowBegin;
        __gm__ OutputElement* dstBase = reinterpret_cast<__gm__ OutputElement*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.offsetD, static_cast<int32_t>(srcRank)));
        if (dstBase != nullptr) {
            StoreSmallSubtileIntersection(
                bufferId, dstBase, dstRow, tileOffset, intersectionRows, tileInfo.blockColStart, tileInfo.actualN);
        }
        tileOffset += intersectionRows;
        rankRowBegin = rankRowEnd;
    }
    set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectTokenPath()
{
    Gmm2CombineCvPipe cvPipe(nullptr, MEGA_MOE_GMM2_COMBINE_CV_SLOT_OFFSET, 0U);
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t aicCoreIdx = get_block_idx();
    const uint32_t aicCoreNum = get_block_num();
    const uint32_t aivSubCoreIdx = get_subblockid();
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        const uint32_t currentMRaw = rowMapper_.CurrentM(groupIdx);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        const uint32_t coreLoops = GmmCommonCoreLoops(currentM, problemK_, l1TileM_, l1TileN_); // 按照L1的size做切tile
        const uint32_t startLoopIdx = GmmCommonStartLoopIdx(aicCoreIdx, aicCoreNum, startCoreIdx);
        CvScaleBlockInfo scaleBlocks[kCombineCvScaleMaxRowBlocks];
        uint32_t scaleBlockCount = 0U;
        const bool scaleCacheEnabled = BuildCvAssignedScaleBlocks(
            scaleBlocks, scaleBlockCount, groupBase, currentM, startLoopIdx, coreLoops, aicCoreNum, l1TileM_, l1TileN_,
            cvTileSequence_, aivSubCoreIdx);
        if (scaleCacheEnabled && scaleBlockCount != 0U) {
            IssueCvScaleBlockLoads(scaleBlocks, scaleBlockCount);
        }
        bool scaleLoadsPending = scaleCacheEnabled && scaleBlockCount != 0U;
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
            const uint32_t tileLane = Gmm2CombineCvLane(cvTileSequence_);
            ++cvTileSequence_;
            if (tileLane != aivSubCoreIdx) {
                continue;
            }
            const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemK_, l1TileM_, l1TileN_, loopIdx);
            const uint32_t subtileCount =
                static_cast<uint32_t>(ceilDiv(tileInfo.actualM, kCombineSmallTokenSubtileRows));
            if (!scaleCacheEnabled) {
                const CvScaleBlockInfo tileScaleBlock = {
                    tileInfo.blockRowStart, groupBase + tileInfo.blockRowStart, tileInfo.actualM, 0U};
                IssueCvScaleBlockLoads(&tileScaleBlock, 1U);
                scaleLoadsPending = true;
            }
            PopCvTile(cvPipe, tileInfo.actualM, tileInfo.actualN);
            if (scaleLoadsPending) {
                WaitCvScaleBlockLoads();
                scaleLoadsPending = false;
            }
            const uint32_t scaleRowBase =
                scaleCacheEnabled ? FindCvScaleCacheRowOffset(scaleBlocks, scaleBlockCount, tileInfo.blockRowStart) :
                                    0U;
            for (uint32_t subtile = 0; subtile < subtileCount; ++subtile) {
                const uint32_t rowInTile = subtile * kCombineSmallTokenSubtileRows;
                const uint32_t rows = (tileInfo.actualM - rowInTile > kCombineSmallTokenSubtileRows) ?
                                          kCombineSmallTokenSubtileRows :
                                          (tileInfo.actualM - rowInTile);
                const uint32_t tileRowBegin = tileInfo.blockRowStart + rowInTile;
                const uint32_t bufferId = pingpongId_;
                pingpongId_ = (pingpongId_ + 1U) % kCombineBufferNum;
                PrepareCvSmallSubtile(bufferId, rowInTile, scaleRowBase + rowInTile, rows, tileInfo.actualN);
                StoreSmallSubtileToRanks(groupIdx, tileInfo, tileRowBegin, rows, bufferId);
            }
            wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent((pingpongId_ + kCombineBufferNum - 1U) % kCombineBufferNum));
            set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent((pingpongId_ + kCombineBufferNum - 1U) % kCombineBufferNum));
            pipe_barrier(PIPE_ALL);
            pto::TFREE<Gmm2CombineCvPipe, pto::TileSplitAxis::TILE_NO_SPLIT>(cvPipe);
        }
        startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        groupBase += currentM;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    SetInitialFlags();
    ProcessDirectTokenPath();
    ProcessFinalBoundary();
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DISPATCH_H
#define DISPATCH_MEGA_COMBINE_DISPATCH_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/peer_memory_layout.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDispatchGatherBufferNum = 2U;
constexpr uint32_t kDispatchGatherRowsPerBatch = 8U;
constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchGatherPackedWordTileCols = kDispatchGatherPackedTileCols / sizeof(uint32_t);
constexpr uint32_t kDispatchGatherPackedScaleTileCols = kDispatchGatherPackedTileCols / sizeof(float);
constexpr uint64_t kDispatchGatherPongUbOffset = 96U * 1024U;
constexpr uint32_t kDispatchGatherPayloadStoreCols = 4064U;

static_assert(kDispatchGatherPackedTileCols % sizeof(uint32_t) == 0U);
static_assert(kDispatchGatherPackedTileCols % sizeof(float) == 0U);
static_assert(kDispatchGatherPackedWordTileCols <= 4095U);
static_assert(kDispatchGatherPayloadStoreCols % UB_ALIGN == 0U);
static_assert(kDispatchGatherPayloadStoreCols <= 4095U);

class DispatchGather {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        const auto& info = tilingData->megaMoeInfo;
        problemK_ = info.K;
        maxOutputSize_ = info.maxOutputSize;
        expertPerRank_ = info.expertPerRank;
        const uint32_t rank = tilingData->runtimeInfo.rank;
        rankSize_ = tilingData->runtimeInfo.rankSize;

        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();

        gmAPtr_ = reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData->dispatchTiling.gmAOffset);
        perTokenScalePtr_ =
            reinterpret_cast<__gm__ float*>(workspaceGM + tilingData->dispatchTiling.perTokenScaleOffset);
        __gm__ int32_t* cumsumMM =
            reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
        __gm__ int32_t* preSumBeforeRank =
            reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.preSumBeforeRankOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(remoteWindow_);
        __gm__ int32_t* tokenPerExpert =
            reinterpret_cast<__gm__ int32_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
        rowMapper_.Init(
            cumsumMM, preSumBeforeRank, tokenPerExpert, rank, rankSize_, expertPerRank_,
            tilingData->frontReorderTiling.expertNumAligned);
    }

    AICORE inline void Process() const
    {
        if ASCEND_IS_AIV {
            ProcessRankSplitCopy();
        }
    }

private:
    struct DispatchGatherChunk {
        uint32_t bufferId = 0U;
        uint32_t rowOffset = 0U;
        uint32_t rowNum = 0U;
    };

    AICORE inline uint32_t PackedRowStride() const { return problemK_ + static_cast<uint32_t>(UB_ALIGN); }

    AICORE inline bool PayloadWordTStoreSupported() const
    {
        return problemK_ % sizeof(uint32_t) == 0U && problemK_ / sizeof(uint32_t) <= kDispatchGatherPackedWordTileCols;
    }

    AICORE inline uint32_t CopyCumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        return rowMapper_.CumsumBeforeSource(srcRank, groupIdx);
    }

    AICORE inline uint64_t PackedUbOffset(uint32_t bufferId) const
    {
        return bufferId == 0U ? 0U : kDispatchGatherPongUbOffset;
    }

    AICORE inline event_t BufferEvent(uint32_t bufferId) const { return bufferId == 0U ? EVENT_ID2 : EVENT_ID3; }

    AICORE inline void PrepareCopyEvents() const
    {
        set_flag(PIPE_MTE3, PIPE_MTE2, BufferEvent(0U));
        set_flag(PIPE_MTE3, PIPE_MTE2, BufferEvent(1U));
    }

    AICORE inline void WaitCopyEvents() const
    {
        wait_flag(PIPE_MTE3, PIPE_MTE2, BufferEvent(0U));
        wait_flag(PIPE_MTE3, PIPE_MTE2, BufferEvent(1U));
    }

    AICORE inline void SetGmm1GroupReady(uint32_t groupIdx) const
    {
        pipe_barrier(PIPE_ALL);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(MegaMoeD2CHardFlagId(groupIdx));
    }

    AICORE inline void StoreTokenRows(
        __gm__ int8_t* dst, uint64_t ubOffsetBytes, uint32_t outputOffset, uint32_t rowNum) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        if (PayloadWordTStoreSupported()) {
            using PayloadWordGlobal = pto::GlobalTensor<uint32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
            using PayloadWordTile = pto::Tile<
                pto::TileType::Vec, uint32_t, kDispatchGatherRowsPerBatch, kDispatchGatherPackedWordTileCols,
                pto::BLayout::RowMajor, -1, -1>;

            const uint32_t wordCols = problemK_ / sizeof(uint32_t);
            ShapeDyn payloadShape(1, 1, 1, rowNum, wordCols);
            StrideDyn payloadStride(rowNum * wordCols, rowNum * wordCols, rowNum * wordCols, wordCols, 1);
            PayloadWordGlobal payloadDst(
                reinterpret_cast<__gm__ uint32_t*>(dst + outputOffset), payloadShape, payloadStride);
            PayloadWordTile payloadTile(rowNum, wordCols);
            pto::TASSIGN(payloadTile, ubOffsetBytes);
            pto::TSTORE(payloadDst, payloadTile);
            return;
        }

        using PayloadGlobal = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PayloadTile = pto::Tile<
            pto::TileType::Vec, int8_t, kDispatchGatherRowsPerBatch, kDispatchGatherPackedTileCols,
            pto::BLayout::RowMajor, -1, -1>;
        for (uint32_t colOffset = 0U; colOffset < problemK_; colOffset += kDispatchGatherPayloadStoreCols) {
            const uint32_t curCols = problemK_ - colOffset > kDispatchGatherPayloadStoreCols ?
                                         kDispatchGatherPayloadStoreCols :
                                         problemK_ - colOffset;
            ShapeDyn payloadShape(1, 1, 1, rowNum, curCols);
            StrideDyn payloadStride(rowNum * problemK_, rowNum * problemK_, rowNum * problemK_, problemK_, 1);
            PayloadGlobal payloadDst(dst + outputOffset + colOffset, payloadShape, payloadStride);
            PayloadTile payloadTile(rowNum, curCols);
            pto::TASSIGN(payloadTile, ubOffsetBytes + colOffset);
            pto::TSTORE(payloadDst, payloadTile);
        }
    }

    AICORE inline void StoreTokenScales(
        __gm__ float* dst, uint64_t ubOffsetBytes, uint32_t outputOffset, uint32_t rowNum) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using ScaleGlobal = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using ScaleTile = pto::Tile<
            pto::TileType::Vec, float, kDispatchGatherRowsPerBatch, kDispatchGatherPackedScaleTileCols,
            pto::BLayout::RowMajor, -1, -1>;

        ShapeDyn scaleShape(1, 1, 1, rowNum, 1);
        StrideDyn scaleStride(rowNum, rowNum, rowNum, 1, 1);
        ScaleGlobal scaleDst(dst + outputOffset, scaleShape, scaleStride);
        ScaleTile scaleTile(rowNum, 1);
        pto::TASSIGN(scaleTile, ubOffsetBytes + problemK_);
        pto::TSTORE(scaleDst, scaleTile);
    }

    AICORE inline void StorePendingChunk(
        __gm__ int8_t* dst, __gm__ float* dstScale, const DispatchGatherChunk& chunk) const
    {
        const event_t event = BufferEvent(chunk.bufferId);
        const uint64_t ubOffsetBytes = PackedUbOffset(chunk.bufferId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        StoreTokenRows(dst, ubOffsetBytes, chunk.rowOffset * problemK_, chunk.rowNum);
        StoreTokenScales(dstScale, ubOffsetBytes, chunk.rowOffset, chunk.rowNum);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
    }

    AICORE inline void FetchRemoteRows(
        __gm__ int8_t* dst, __gm__ float* dstScale, __gm__ int8_t* src, uint32_t rows, uint32_t& pingpongId) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using PackedGlobal = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PackedTile = pto::Tile<
            pto::TileType::Vec, int8_t, kDispatchGatherRowsPerBatch, kDispatchGatherPackedTileCols,
            pto::BLayout::RowMajor, -1, -1>;

        const uint32_t packedStride = PackedRowStride();
        const uint32_t processCount = static_cast<uint32_t>(ceilDiv(rows, kDispatchGatherRowsPerBatch));
        for (uint32_t processIdx = 0U; processIdx < processCount; ++processIdx) {
            pingpongId = (pingpongId + 1U) % kDispatchGatherBufferNum;
            const uint32_t bufferId = pingpongId;
            const event_t event = BufferEvent(bufferId);
            const uint64_t ubOffsetBytes = PackedUbOffset(bufferId);
            const uint32_t rowOffset = processIdx * kDispatchGatherRowsPerBatch;
            const uint32_t rowNum =
                rows - rowOffset > kDispatchGatherRowsPerBatch ? kDispatchGatherRowsPerBatch : rows - rowOffset;

            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            ShapeDyn packedShape(1, 1, 1, rowNum, packedStride);
            StrideDyn packedStrideShape(
                rowNum * packedStride, rowNum * packedStride, rowNum * packedStride, packedStride, 1);
            PackedGlobal remotePacked(
                src + static_cast<uint64_t>(rowOffset) * packedStride, packedShape, packedStrideShape);
            PackedTile packedTile(rowNum, packedStride);
            pto::TASSIGN(packedTile, ubOffsetBytes);
            pto::TLOAD(packedTile, remotePacked);
            set_flag(PIPE_MTE2, PIPE_MTE3, event);

            StorePendingChunk(dst, dstScale, {bufferId, rowOffset, rowNum});
        }
    }

    AICORE inline void FetchRankGroupRows(
        uint32_t srcRank, uint32_t groupIdx, uint32_t groupBase, uint32_t& pingpongId) const
    {
        const uint32_t rawRows = rowMapper_.RowsRaw(srcRank, groupIdx);
        const uint32_t dstRowBase = groupBase + CopyCumsumBeforeSource(srcRank, groupIdx);
        if (dstRowBase >= maxOutputSize_) {
            return;
        }
        const uint32_t rows = rawRows > maxOutputSize_ - dstRowBase ? maxOutputSize_ - dstRowBase : rawRows;
        if (rows == 0U) {
            return;
        }

        __gm__ int8_t* remoteRows = reinterpret_cast<__gm__ int8_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.offsetA, static_cast<int32_t>(srcRank)));
        const uint32_t srcRowBase = rowMapper_.DstRowOffset(srcRank, groupIdx);
        __gm__ int8_t* remoteSrc = remoteRows + static_cast<uint64_t>(srcRowBase) * PackedRowStride();
        FetchRemoteRows(
            gmAPtr_ + static_cast<uint64_t>(dstRowBase) * problemK_, perTokenScalePtr_ + dstRowBase, remoteSrc, rows,
            pingpongId);
    }

    AICORE inline void ProcessRankSplitCopy() const
    {
        uint32_t groupBase = 0U;
        uint32_t pingpongId = 0U;
        for (uint32_t groupIdx = 0U; groupIdx < expertPerRank_; ++groupIdx) {
            uint32_t currentM = 0U;
            if (coreIdx_ < rankSize_) {
                PrepareCopyEvents();
                currentM = rowMapper_.CurrentM(groupIdx);
                for (uint32_t srcRank = coreIdx_; srcRank < rankSize_; srcRank += coreNum_) {
                    FetchRankGroupRows(srcRank, groupIdx, groupBase, pingpongId);
                }
                groupBase += currentM;
                WaitCopyEvents();
            }
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
            SetGmm1GroupReady(groupIdx);
        }
    }

    __gm__ int8_t* gmAPtr_ = nullptr;
    __gm__ float* perTokenScalePtr_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    MegaMoePeerRowMapper rowMapper_;

    uint32_t problemK_ = 0U;
    uint32_t maxOutputSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t coreIdx_ = 0U;
    uint32_t coreNum_ = 1U;
};

#endif // DISPATCH_MEGA_COMBINE_DISPATCH_H

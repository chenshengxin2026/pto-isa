/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_SWIGLU_H
#define DISPATCH_MEGA_COMBINE_SWIGLU_H

#include "kernel_operator.h"

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_vector.hpp"
#include "utils/pto_sync_substrate.hpp"

constexpr uint32_t kSwigluVecTileElems = 1024U;
constexpr uint32_t kSwigluFullRowIoBlockChunks = 4U;
constexpr uint32_t kSwigluUbBufferNum = 2U;
constexpr uint32_t kSwigluScaleTileElems = 128U;
constexpr uint32_t kSwigluScaleChunkBuffers = 2U;
constexpr uint32_t kSwigluScalarScratchBytes = 32U;
constexpr float kSwigluDynamicQuantEps = 1.0e-6f;

class Swiglu {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void Process();

private:
    AICORE inline void WaitC2VReady() const
    {
        CrossCoreWaitFlag<0x2>(MEGA_MOE_C2V_HARD_FLAG_BASE);
        pipe_barrier(PIPE_ALL);
    }
    AICORE inline void BuildSegmentMetadata(uint32_t segmentIdx, uint32_t& segmentRowBase, uint32_t& segmentRows) const;
    AICORE inline void WriteSharedSegmentMetadata(uint32_t segmentIdx) const;
    AICORE inline void ReadSharedSegmentMetadata(
        uint32_t segmentIdx, uint32_t& segmentRowBase, uint32_t& rowSplitBase, uint32_t& rowSplitRem) const;
    AICORE inline void SetV2CReady() const
    {
        pipe_barrier(PIPE_ALL);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(MEGA_MOE_V2C_HARD_FLAG_BASE);
    }
    AICORE inline uint64_t AlignUbBytes(uint64_t value) const { return (value + 31U) / 32U * 32U; }
    AICORE inline uint64_t SwigluMaxScratchBytes() const
    {
        const uint64_t bytes = static_cast<uint64_t>(outputN_ / 2U) * sizeof(float);
        return bytes < kSwigluScalarScratchBytes ? kSwigluScalarScratchBytes : AlignUbBytes(bytes);
    }
    AICORE inline uint64_t SwigluBufferBytes() const;
    AICORE inline uint64_t SwigluCOffset(uint32_t bufferId) const
    {
        if (bufferId == 0U) {
            return 0;
        }
        return static_cast<uint64_t>(bufferId) * SwigluBufferBytes();
    }
    AICORE inline uint64_t SwigluDOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCOffset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(half));
    }
    AICORE inline uint64_t SwigluCFp32Offset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluDOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(int8_t));
    }
    AICORE inline uint64_t SwigluWorkOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluCFp32Offset(bufferId) + static_cast<uint64_t>(problemN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluAbsOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluWorkOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluMaxOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluAbsOffset(bufferId) + static_cast<uint64_t>(outputN_) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluMaxOffset(bufferId) + SwigluMaxScratchBytes());
    }
    AICORE inline uint64_t SwigluScaleOutputBytes() const
    {
        return AlignUbBytes(static_cast<uint64_t>(kSwigluScaleTileElems) * sizeof(float));
    }
    AICORE inline uint64_t SwigluScaleOutputOffset(uint32_t bufferId) const
    {
        return AlignUbBytes(SwigluBufferBytes() * kSwigluUbBufferNum) +
               static_cast<uint64_t>(bufferId) * SwigluScaleOutputBytes();
    }
    AICORE inline bool FullRowUbFits() const
    {
        return SwigluScaleOutputOffset(kSwigluScaleChunkBuffers - 1U) + SwigluScaleOutputBytes() <= AtlasA5::UB_SIZE;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t StoreDoneEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t ScaleStoreEvent(uint32_t bufferId) const { return bufferId == 0U ? EVENT_ID2 : EVENT_ID3; }
    AICORE inline void InitFullRowPipeline() const;
    AICORE inline void FinalizeFullRowPipeline() const;
    AICORE inline void RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const;
    AICORE inline void IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const;
    AICORE inline float ComputeAndStorePreparedFullRow(uint32_t rowIdx, uint32_t bufferId, float perTokenScale) const;
    AICORE inline void IssueStoreScale2Chunk(uint32_t rowStart, uint32_t rowCount, uint32_t scaleBufferId) const;

    __gm__ MegaMoeSwigluSegmentRuntimeMeta* segmentMetaPtr_ = nullptr;
    __gm__ half* gmCPtr_ = nullptr;
    __gm__ float* perTokenScalePtr_ = nullptr;
    __gm__ int8_t* gmPermutedTokenPtr_ = nullptr;
    __gm__ float* perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    uint32_t problemN_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

AICORE inline void Swiglu::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    problemN_ = tilingData->megaMoeInfo.N;
    outputN_ = problemN_ / 2U;
    maxOutputSize_ = tilingData->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData->runtimeInfo.rankSize;

    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    gmCPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM + tilingData->gmm1Tiling.gmCOffset);
    perTokenScalePtr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tilingData->dispatchTiling.perTokenScaleOffset);
    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData->swigluTiling.gmPermutedTokenOffset);
    perTokenScale2Ptr_ = reinterpret_cast<__gm__ float*>(workspaceGM + tilingData->swigluTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
    segmentMetaPtr_ = reinterpret_cast<__gm__ MegaMoeSwigluSegmentRuntimeMeta*>(
        workspaceGM + tilingData->swigluTiling.swigluSegmentMetaOffset);
}
AICORE inline void Swiglu::BuildSegmentMetadata(
    uint32_t segmentIdx, uint32_t& segmentRowBase, uint32_t& segmentRows) const
{
    const uint32_t segmentStartExpert = MoeSwigluSegmentStartExpert(expertPerRank_, segmentIdx);
    const uint32_t segmentEndExpert = MoeSwigluSegmentEndExpert(expertPerRank_, segmentIdx);
    segmentRowBase = 0;
    segmentRows = 0;

    uint32_t groupBase = 0;
    for (uint32_t groupIdx = 0; groupIdx < segmentEndExpert; ++groupIdx) {
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        if (groupIdx == segmentStartExpert) {
            segmentRowBase = groupBase;
        }
        if (groupIdx >= segmentStartExpert) {
            segmentRows += currentM;
        }
        groupBase += currentM;
    }
}

AICORE inline void Swiglu::WriteSharedSegmentMetadata(uint32_t segmentIdx) const
{
    if (coreIdx_ != 0U) {
        return;
    }

    uint32_t segmentRowBase = 0;
    uint32_t segmentRows = 0;
    BuildSegmentMetadata(segmentIdx, segmentRowBase, segmentRows);
    const uint32_t rowSplitBase = segmentRows / coreNum_;
    const uint32_t rowSplitRem = segmentRows - rowSplitBase * coreNum_;

    volatile __gm__ MegaMoeSwigluSegmentRuntimeMeta* entry = segmentMetaPtr_ + segmentIdx;
    entry->segmentRowBase = segmentRowBase;
    entry->rowSplitBase = rowSplitBase;
    entry->rowSplitRem = rowSplitRem;
    pipe_barrier(PIPE_ALL);
    MegaMoeDcciGmRange(
        reinterpret_cast<__gm__ void*>(segmentMetaPtr_ + segmentIdx), sizeof(MegaMoeSwigluSegmentRuntimeMeta));
}

AICORE inline void Swiglu::ReadSharedSegmentMetadata(
    uint32_t segmentIdx, uint32_t& segmentRowBase, uint32_t& rowSplitBase, uint32_t& rowSplitRem) const
{
    volatile __gm__ MegaMoeSwigluSegmentRuntimeMeta* entry = segmentMetaPtr_ + segmentIdx;
    segmentRowBase = entry->segmentRowBase;
    rowSplitBase = entry->rowSplitBase;
    rowSplitRem = entry->rowSplitRem;
}
AICORE inline uint64_t Swiglu::SwigluBufferBytes() const
{
    return AlignUbBytes(SwigluScaleOffset(0) + kSwigluScalarScratchBytes);
}

AICORE inline void Swiglu::InitFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbBufferNum; ++bufferId) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

AICORE inline void Swiglu::FinalizeFullRowPipeline() const
{
    for (uint32_t bufferId = 0; bufferId < kSwigluUbBufferNum; ++bufferId) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        wait_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    }
    for (uint32_t bufferId = 0; bufferId < kSwigluScaleChunkBuffers; ++bufferId) {
        wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(bufferId));
    }
}

AICORE inline void Swiglu::RunFullRowEpilogue(uint32_t localRowStart, uint32_t localRows) const
{
    if (localRows == 0U || outputN_ == 0U || !FullRowUbFits()) {
        return;
    }
    InitFullRowPipeline();
    uint32_t scaleChunkStart = 0;
    uint32_t scaleChunkCount = 0;
    uint32_t scaleBufferId = 0;
    IssueFullRowLoad(localRowStart, 0U);
    for (uint32_t localRow = 0; localRow < localRows; ++localRow) {
        const uint32_t bufferId = localRow % kSwigluUbBufferNum;
        const uint32_t rowIdx = localRowStart + localRow;
        const float perTokenScale =
            PrepareFullRowCompute(rowIdx, bufferId); //  用来反量化 GMM1 结果，进入 SwiGLU fp32 计算
        if (localRow + 1U < localRows) {
            IssueFullRowLoad(localRowStart + localRow + 1U, (localRow + 1U) % kSwigluUbBufferNum);
        }
        // SwiGLU 输出重新量化后的 scale，给后续 GMM2 反量化/scale 使用
        const float scale2 = ComputeAndStorePreparedFullRow(rowIdx, bufferId, perTokenScale);
        if (scaleChunkCount == 0U) {
            wait_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
        }
        PtoSetValue<float, kSwigluScaleTileElems>(SwigluScaleOutputOffset(scaleBufferId), scaleChunkCount, scale2);
        ++scaleChunkCount;
        if (scaleChunkCount == kSwigluScaleTileElems) { // 满128行，一起写回GM
            IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
            scaleChunkStart += scaleChunkCount;
            scaleChunkCount = 0;
            scaleBufferId = scaleBufferId + 1U == kSwigluScaleChunkBuffers ? 0U : scaleBufferId + 1U;
        }
    }
    if (scaleChunkCount > 0U) {
        IssueStoreScale2Chunk(localRowStart + scaleChunkStart, scaleChunkCount, scaleBufferId);
    }
    FinalizeFullRowPipeline();
}

AICORE inline void Swiglu::IssueStoreScale2Chunk(uint32_t rowStart, uint32_t rowCount, uint32_t scaleBufferId) const
{
    if (rowCount == 0U) {
        return;
    }
    set_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    wait_flag(PIPE_S, PIPE_MTE3, ScaleStoreEvent(scaleBufferId));
    PtoStoreVector<float, kSwigluScaleTileElems>(
        perTokenScale2Ptr_ + rowStart, SwigluScaleOutputOffset(scaleBufferId), rowCount);
    set_flag(PIPE_MTE3, PIPE_S, ScaleStoreEvent(scaleBufferId));
}

AICORE inline void Swiglu::IssueFullRowLoad(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CGlobal = pto::GlobalTensor<half, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileC = pto::Tile<
        pto::TileType::Vec, half, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    __gm__ half* gmCRow = gmCPtr_ + static_cast<uint64_t>(rowIdx) * problemN_;

    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    uint32_t offset = 0;
    while (problemN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (problemN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileC cTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        BlockShape cShape(chunkRows, kSwigluVecTileElems);
        BlockStride cStride(
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        CBlockGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < problemN_) {
        const uint32_t cur = problemN_ - offset;
        TileC cTile(1, cur);
        pto::TASSIGN(cTile, ubCOffset + static_cast<uint64_t>(offset) * sizeof(half));
        VectorShape cShape(cur);
        VectorStride cStride(cur, cur, cur, cur);
        CGlobal cGlobal(gmCRow + offset, cShape, cStride);
        pto::TLOAD(cTile, cGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
}

AICORE inline float Swiglu::PrepareFullRowCompute(uint32_t rowIdx, uint32_t bufferId) const
{
    using TileC = PtoVecTile<half, kSwigluVecTileElems>;
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;

    const uint64_t ubCOffset = SwigluCOffset(bufferId);
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);

    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 fp32Tile(1, cur);
        TileC cTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(fp32Tile, ubCFp32Offset + elemOffset * sizeof(float));
        pto::TASSIGN(cTile, ubCOffset + elemOffset * sizeof(half));
        pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    return perTokenScalePtr_[rowIdx];
}

AICORE inline float Swiglu::ComputeAndStorePreparedFullRow(
    uint32_t rowIdx, uint32_t bufferId, float perTokenScale) const
{
    using TileFp32 = PtoVecTile<float, kSwigluVecTileElems>;
    using TileHalf = PtoVecTile<half, kSwigluVecTileElems>;
    using TileD = PtoVecTile<int8_t, kSwigluVecTileElems>;
    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<int8_t, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockTileD = pto::Tile<
        pto::TileType::Vec, int8_t, kSwigluFullRowIoBlockChunks, kSwigluVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DBlockGlobal = pto::GlobalTensor<int8_t, BlockShape, BlockStride, pto::Layout::ND>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;

    const uint64_t ubDOffset = SwigluDOffset(bufferId);
    const uint64_t ubCFp32Offset = SwigluCFp32Offset(bufferId);
    const uint64_t ubWorkOffset = SwigluWorkOffset(bufferId);
    const uint64_t ubAbsOffset = SwigluAbsOffset(bufferId);
    const uint64_t ubMaxOffset = SwigluMaxOffset(bufferId);
    __gm__ int8_t* gmDRow = gmPermutedTokenPtr_ + static_cast<uint64_t>(rowIdx) * outputN_;

    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < problemN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = problemN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : problemN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(srcTile, ubCFp32Offset + elemOffset);
        pto::TMULS(dstTile, srcTile, perTokenScale);
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        pto::TASSIGN(dstTile, ubWorkOffset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(srcTile, ubCFp32Offset + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TMULS(dstTile, srcTile, -1.0f);
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TEXP(dstTile, srcTile);
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TADDS(dstTile, srcTile, 1.0f);
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 xTile(1, cur);
        TileFp32 denomTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(xTile, ubCFp32Offset + elemOffset);
        pto::TASSIGN(denomTile, ubWorkOffset + elemOffset);
        pto::TDIV(dstTile, xTile, denomTile);
    }
    pipe_barrier(PIPE_ALL);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 siluTile(1, cur);
        TileFp32 gateTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(siluTile, ubWorkOffset + elemOffset);
        pto::TASSIGN(gateTile, ubCFp32Offset + static_cast<uint64_t>(outputN_ + offset) * sizeof(float));
        pto::TMUL(dstTile, siluTile, gateTile);
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TABS(absTile, srcTile);
    }
    pipe_barrier(PIPE_ALL);

    bool firstReduceChunk = true;
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 absTile(1, cur);
        TileFp32 tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(absTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(tmpTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(rowMaxTile, firstReduceChunk ? ubMaxOffset : ubAbsOffset);
        pto::TROWMAX(rowMaxTile, absTile, tmpTile);
        pipe_barrier(PIPE_ALL);
        if (!firstReduceChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, ubMaxOffset);
            pto::TASSIGN(newTile, ubAbsOffset);
            pto::TASSIGN(dstTile, ubMaxOffset);
            pto::TMAX(dstTile, accTile, newTile);
            pipe_barrier(PIPE_ALL);
        }
        firstReduceChunk = false;
    }
    pipe_barrier(PIPE_ALL);

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    TileFp32 maxScalarTile(1, 1);
    pto::TASSIGN(maxScalarTile, ubMaxOffset);
    const float maxAbs = maxScalarTile.GetValue(0);
    const float scale2 = maxAbs > 0.0f ? maxAbs / 127.0f : kSwigluDynamicQuantEps / 127.0f;
    const float quantScale = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;

    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileFp32 dstTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        pto::TASSIGN(dstTile, ubAbsOffset + elemOffset);
        pto::TASSIGN(srcTile, ubWorkOffset + elemOffset);
        pto::TMULS(dstTile, srcTile, quantScale);
    }
    pipe_barrier(PIPE_ALL);

    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileHalf halfTile(1, cur);
        TileFp32 srcTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(halfTile, ubWorkOffset + elemOffset * sizeof(half));
        pto::TASSIGN(srcTile, ubAbsOffset + elemOffset * sizeof(float));
        pto::TCVT(halfTile, srcTile, pto::RoundMode::CAST_NONE);
    }
    pipe_barrier(PIPE_ALL);

    wait_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    for (uint32_t offset = 0; offset < outputN_; offset += kSwigluVecTileElems) {
        const uint32_t cur = outputN_ - offset > kSwigluVecTileElems ? kSwigluVecTileElems : outputN_ - offset;
        TileD dTile(1, cur);
        TileHalf halfTile(1, cur);
        const uint64_t elemOffset = static_cast<uint64_t>(offset);
        pto::TASSIGN(dTile, ubDOffset + elemOffset * sizeof(int8_t));
        pto::TASSIGN(halfTile, ubWorkOffset + elemOffset * sizeof(half));
        pto::TCVT(dTile, halfTile, pto::RoundMode::CAST_RINT);
    }

    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    uint32_t offset = 0;
    while (outputN_ - offset >= kSwigluVecTileElems) {
        const uint32_t fullChunks = (outputN_ - offset) / kSwigluVecTileElems;
        const uint32_t chunkRows = fullChunks > kSwigluFullRowIoBlockChunks ? kSwigluFullRowIoBlockChunks : fullChunks;
        BlockTileD dTile(chunkRows, kSwigluVecTileElems);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        BlockShape dShape(chunkRows, kSwigluVecTileElems);
        BlockStride dStride(
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems,
            static_cast<int64_t>(chunkRows) * kSwigluVecTileElems, kSwigluVecTileElems);
        DBlockGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
        offset += chunkRows * kSwigluVecTileElems;
    }
    if (offset < outputN_) {
        const uint32_t cur = outputN_ - offset;
        TileD dTile(1, cur);
        pto::TASSIGN(dTile, ubDOffset + static_cast<uint64_t>(offset) * sizeof(int8_t));
        VectorShape dShape(cur);
        VectorStride dStride(cur, cur, cur, cur);
        DGlobal dGlobal(gmDRow + offset, dShape, dStride);
        pto::TSTORE(dGlobal, dTile);
    }

    set_flag(PIPE_MTE3, PIPE_V, StoreDoneEvent(bufferId));
    return scale2;
}

AICORE inline void Swiglu::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    const uint32_t segmentNum = MoeSwigluSegmentNum(expertPerRank_);
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        WaitC2VReady();
        WriteSharedSegmentMetadata(segmentIdx);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();

        uint32_t segmentRowBase = 0;
        uint32_t rowSplitBase = 0;
        uint32_t rowSplitRem = 0;
        ReadSharedSegmentMetadata(segmentIdx, segmentRowBase, rowSplitBase, rowSplitRem);
        const uint32_t localRows = rowSplitBase + (coreIdx_ < rowSplitRem ? 1U : 0U);
        const uint32_t prefixRows = coreIdx_ * rowSplitBase + (coreIdx_ < rowSplitRem ? coreIdx_ : rowSplitRem);
        const uint32_t localRowStart = segmentRowBase + prefixRows;
        RunFullRowEpilogue(localRowStart, localRows);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        SetV2CReady();
    }
}

#endif // DISPATCH_MEGA_COMBINE_SWIGLU_H

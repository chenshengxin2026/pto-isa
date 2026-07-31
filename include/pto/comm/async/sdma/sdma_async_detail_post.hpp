/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP

#include "pto/comm/async/sdma/sdma_async_detail_basic.hpp"

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

PTO_INTERNAL void UpdateCachedPostDoneIds(uint64_t postId, uint64_t queueMask, SdmaRuntimeContext& runtimeCtx)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) != 0ULL && runtimeCtx.postDoneId[queue] < postId) {
            runtimeCtx.postDoneId[queue] = postId;
        }
        queueMask >>= 1U;
        ++queue;
    }
}

PTO_INTERNAL bool CheckPostDoneIds(
    __gm__ uint8_t* postDoneBase, uint64_t postId, uint64_t queueMask, UbTmpBuf& tmpBuf, bool blocking)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) == 0ULL) {
            queueMask >>= 1U;
            ++queue;
            continue;
        }
        uint32_t attempts = 0U;
        while (GetValue<uint64_t>(GetPostDoneRecordAddr(postDoneBase, queue), tmpBuf) < postId) {
            if (!blocking || ++attempts >= kPostPollLimit) {
                return false;
            }
        }
        queueMask >>= 1U;
        ++queue;
    }
    return true;
}

PTO_INTERNAL bool InitializeRuntimeCtx(const SdmaSession& session)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    const SdmaExecContext& execCtx = session.execCtx;
    if (execCtx.contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    runtimeCtx.postDoneBase = ResolvePostDoneBase(execCtx);
    if (runtimeCtx.postDoneBase == nullptr) {
        return false;
    }
    for (uint32_t queue = 0U; queue < execCtx.baseConfig.queue_num; ++queue) {
        SetValue<uint64_t>(GetPostDoneRecordAddr(runtimeCtx.postDoneBase, queue), tmpBuf, execCtx.syncId, 0ULL);
    }
    pipe_barrier(PIPE_ALL);

    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo* channels = channelBase + execCtx.channelGroupIdx * execCtx.baseConfig.queue_num;
    // ChannelInfo is populated outside AI Core and read through scalar loads during Post.
    // Invalidate stale metadata once at session build time to keep DCCI out of the Post hot path.
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < execCtx.baseConfig.queue_num; ++queue) {
        __gm__ BatchWriteChannelInfo* channel = channels + queue;
        const uint64_t packedHeadTail = GetValue<uint64_t>((__gm__ uint8_t*)channel, tmpBuf);
        runtimeCtx.sqHead[queue] = static_cast<uint32_t>(packedHeadTail);
        runtimeCtx.sqTail[queue] = static_cast<uint32_t>(packedHeadTail >> 32U);
    }
    return true;
}

PTO_INTERNAL bool StoreFlagPayload(
    uint64_t postId, uint32_t queueCount, __gm__ uint8_t* flagPayload, const SdmaSession& session, UbTmpBuf& tmpBuf)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (postId > kFlagPayloadDepth) {
        // Post IDs are contiguous, so this slot was last used by postId - depth.
        // Wait for that flag source to be consumed before overwriting the slot.
        const uint64_t oldPostId = postId - kFlagPayloadDepth;
        const uint32_t slot = static_cast<uint32_t>(postId % kFlagPayloadDepth);
        const uint64_t oldQueueMask = QueueCountToMask(runtimeCtx.flagPayloadQueueCount[slot]);
        bool knownComplete = true;
        uint64_t remaining = oldQueueMask;
        uint32_t queue = 0U;
        while (remaining != 0ULL) {
            if ((remaining & 1ULL) != 0ULL && runtimeCtx.postDoneId[queue] < oldPostId) {
                knownComplete = false;
                break;
            }
            remaining >>= 1U;
            ++queue;
        }
        if (!knownComplete) {
            if (!CheckPostDoneIds(runtimeCtx.postDoneBase, oldPostId, oldQueueMask, tmpBuf, true)) {
                return false;
            }
            UpdateCachedPostDoneIds(oldPostId, oldQueueMask, runtimeCtx);
        }
    }

    *reinterpret_cast<volatile __gm__ uint64_t*>(flagPayload) = postId;
    __asm__ __volatile__("" ::: "memory");
    runtimeCtx.flagPayloadQueueCount[postId % kFlagPayloadDepth] = static_cast<uint8_t>(queueCount);
    return true;
}

PTO_INTERNAL bool ValidateSinglePostSqCapacity(
    __gm__ BatchWriteChannelInfo* channels, const SdmaConfig& config, uint32_t dataQueueCount, uint32_t postQueueCount)
{
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        uint32_t dataSqes = 0U;
        if (queue < dataQueueCount) {
            dataSqes = (config.iter_num - 1U - queue) / config.queue_num + 1U;
        }
        const uint32_t sqesPerPost = dataSqes + 1U;
        const uint32_t sqDepth = channels[queue].sq_depth;
        if (sqDepth == 0U || sqesPerPost > sqDepth) {
            return false;
        }
    }
    return true;
}

PTO_INTERNAL void SubmitDataTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode,
    const SdmaConfig& config, SdmaRuntimeContext& runtimeCtx)
{
    for (uint32_t index = 0U; index < config.iter_num; ++index) {
        const uint32_t queue = index % config.queue_num;
        uint32_t transferBytes = static_cast<uint32_t>(config.block_bytes);
        if (index + 1U == config.iter_num) {
            transferBytes = static_cast<uint32_t>(config.per_core_bytes - index * config.block_bytes);
        }
        __gm__ uint8_t* src = sendBuffer + config.comm_block_offset + index * config.block_bytes;
        __gm__ uint8_t* dst = recvBuffer + config.comm_block_offset + index * config.block_bytes;
        AddOneMemcpySqe(
            channels + queue, src, dst, opcode, transferBytes, runtimeCtx.sqTail[queue],
            runtimeCtx.sqTail[queue] - runtimeCtx.sqHead[queue]);
        runtimeCtx.sqTail[queue] = (runtimeCtx.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void RingDoorbell(
    __gm__ BatchWriteChannelInfo* channel, uint32_t sqTail, UbTmpBuf& tmpBuf, uint32_t syncId)
{
#ifdef PTO_NPU_ARCH_A5
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base, tmpBuf, syncId, sqTail);
#else
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base + 8U, tmpBuf, syncId, sqTail);
#endif
}

PTO_INTERNAL void PublishDataTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t dataQueueCount, const uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels->sq_base, ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < dataQueueCount; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void SubmitFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* flagPayload, uint32_t postQueueCount,
    SdmaRuntimeContext& runtimeCtx)
{
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        AddOneMemcpySqe(
            channels + queue, flagPayload, GetPostDoneRecordAddr(runtimeCtx.postDoneBase, queue), 0U, kPostIdFlagBytes,
            runtimeCtx.sqTail[queue], runtimeCtx.sqTail[queue] - runtimeCtx.sqHead[queue]);
        runtimeCtx.sqTail[queue] = (runtimeCtx.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PersistSqTails(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, const SdmaRuntimeContext& runtimeCtx, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        const uint64_t packed = (static_cast<uint64_t>(runtimeCtx.sqTail[queue]) << 32U) | runtimeCtx.sqHead[queue];
        SetValue<uint64_t>((__gm__ uint8_t*)(channels + queue), tmpBuf, syncId, packed);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PublishFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t beginQueue, uint32_t endQueue, const uint32_t* sqTail,
    UbTmpBuf& tmpBuf, uint32_t syncId)
{
    for (uint32_t queue = beginQueue; queue < endQueue; ++queue) {
        const uint32_t flagSqeIndex = (sqTail[queue] + channels[queue].sq_depth - 1U) % channels[queue].sq_depth;
        __gm__ BatchWriteItem* sqRing = reinterpret_cast<__gm__ BatchWriteItem*>(channels[queue].sq_base);
        __gm__ BatchWriteItem* flagSqe = sqRing + flagSqeIndex;
        __asm__ __volatile__("");
        dcci((__gm__ void*)flagSqe, SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
    for (uint32_t queue = beginQueue; queue < endQueue; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
    pipe_barrier(PIPE_ALL);
}

struct SdmaPostState {
    __gm__ BatchWriteChannelInfo* channels;
    __gm__ uint8_t* flagPayload;
    UbTmpBuf tmpBuf;
    uint32_t dataQueueCount;
    uint32_t postQueueCount;
    uint64_t eventHandle;
};

PTO_INTERNAL bool BeginSdmaPost(
    uint64_t messageLen, const SdmaSession& session, SdmaConfig& config, SdmaPostState& state)
{
    const SdmaExecContext& execCtx = session.execCtx;
    if (!session.valid || !BuildTransferConfig(execCtx.baseConfig, messageLen, config) || config.iter_num == 0U ||
        config.queue_num == 0U || config.queue_num > kPostMaxQueues ||
        execCtx.channelGroupIdx >= kSdmaMaxChannel / config.queue_num) {
        return false;
    }
    state.dataQueueCount = config.iter_num < config.queue_num ? config.iter_num : config.queue_num;
    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    state.channels = channelBase + execCtx.channelGroupIdx * config.queue_num;
    state.tmpBuf = execCtx.tmpBuf;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    state.postQueueCount =
        runtimeCtx.usedQueueCount > state.dataQueueCount ? runtimeCtx.usedQueueCount : state.dataQueueCount;
    if (!ValidateSinglePostSqCapacity(state.channels, config, state.dataQueueCount, state.postQueueCount) ||
        runtimeCtx.nextPostId >= kSdmaHandlePostIdMask) {
        return false;
    }
    const uint64_t postId = runtimeCtx.nextPostId + 1ULL;
    state.flagPayload = GetFlagPayloadAddr(ResolveFlagPayloadBase(execCtx), postId);
    if (!StoreFlagPayload(postId, state.postQueueCount, state.flagPayload, session, state.tmpBuf) ||
        !EncodeSdmaEventHandle(postId, state.postQueueCount, state.eventHandle)) {
        return false;
    }
    runtimeCtx.nextPostId = postId;
    runtimeCtx.usedQueueCount = state.postQueueCount;
    return true;
}

PTO_INTERNAL AsyncEvent FinishSdmaPost(const SdmaConfig& config, const SdmaPostState& state, const SdmaSession& session)
{
    const SdmaExecContext& execCtx = session.execCtx;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    UbTmpBuf tmpBuf = state.tmpBuf;
    PublishDataTransferSqes(state.channels, state.dataQueueCount, runtimeCtx.sqTail, tmpBuf, execCtx.syncId);
    SubmitFlagTransferSqes(state.channels, state.flagPayload, state.postQueueCount, runtimeCtx);
    PersistSqTails(state.channels, config.queue_num, runtimeCtx, tmpBuf, execCtx.syncId);
    PublishFlagTransferSqes(state.channels, 0U, state.dataQueueCount, runtimeCtx.sqTail, tmpBuf, execCtx.syncId);
    if (state.dataQueueCount < state.postQueueCount) {
        PublishFlagTransferSqes(
            state.channels, state.dataQueueCount, state.postQueueCount, runtimeCtx.sqTail, tmpBuf, execCtx.syncId);
    }
    return AsyncEvent(state.eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent SdmaPostAsync(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode, uint64_t messageLen,
    const SdmaSession& session)
{
    if (recvBuffer == nullptr || sendBuffer == nullptr) {
        return {};
    }
    SdmaConfig config{};
    SdmaPostState state{};
    if (!BeginSdmaPost(messageLen, session, config, state)) {
        return {};
    }
    SubmitDataTransferSqes(state.channels, recvBuffer, sendBuffer, opcode, config, session.runtimeCtx);
    return FinishSdmaPost(config, state, session);
}

PTO_INTERNAL bool SdmaEventCheck(uint64_t postId, uint64_t queueMask, const SdmaSession& session, bool blocking)
{
    if (!session.valid || postId == 0ULL || queueMask == 0ULL) {
        return false;
    }
    UbTmpBuf tmpBuf = session.eventCtx.tmpBuf;
    if (!CheckPostDoneIds(session.runtimeCtx.postDoneBase, postId, queueMask, tmpBuf, blocking)) {
        return false;
    }
    UpdateCachedPostDoneIds(postId, queueMask, session.runtimeCtx);
    return true;
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t postId = 0ULL;
    uint32_t queueCount = 0U;
    if (!DecodeSdmaEventHandle(handle, postId, queueCount)) {
        return false;
    }
    const uint64_t queueMask = QueueCountToMask(queueCount);
    return SdmaEventCheck(postId, queueMask, session, false);
}

PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t postId = 0ULL;
    uint32_t queueCount = 0U;
    if (!DecodeSdmaEventHandle(handle, postId, queueCount)) {
        return false;
    }
    const uint64_t queueMask = QueueCountToMask(queueCount);
    return SdmaEventCheck(postId, queueMask, session, true);
}

} // namespace detail
} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP

/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

// GridPipe's A2/A3 headers normally inherit these target spellings from the
// compiler runtime.  This control-path-only CPU test supplies the minimal host
// equivalents instead of pulling in every tile instruction implementation.
#define __gm__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __biasbuf__
#define __tf__
using pipe_t = int;
inline constexpr pipe_t PIPE_ALL = 0;
inline constexpr int SINGLE_CACHE_LINE = 0;
inline constexpr int DSB_DDR = 0;
inline void pipe_barrier(pipe_t) {}
inline void dcci(const volatile void*, int) {}
inline void dsb(int) {}
inline uint32_t get_block_idx() { return 0; }

// The queue control path does not use collective tensor descriptors.  Supplying
// the one enum referenced by the CCE facade keeps this unit test independent of
// the CPU simulator's full tile implementation.
#define PTO_COMM_COMM_TYPES_HPP
namespace pto {
namespace comm {
enum class ReduceOp : uint8_t {
    Sum = 0,
    Max = 1,
    Min = 2,
};
} // namespace comm
} // namespace pto

#include <pto/npu/a2a3/GridTPop.hpp>

namespace {

struct BindQueueTestContext {
    uint8_t* windowBase = nullptr;
    uint32_t windowBytes = 0;
    uint32_t windowCount = 0;
};

struct BindQueueTile {};

using BindQueuePipe = pto::GridPipe<BindQueueTile, 64, 2>;

constexpr uint32_t kProducerCount = 4;
constexpr uint32_t kConsumerId = kProducerCount;
constexpr uint32_t kPeerCount = kProducerCount + 1;

} // namespace

namespace pto {
namespace a2a3_grid_payload {

AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId)
{
    auto* ctx = reinterpret_cast<BindQueueTestContext*>(runtimeCtx);
    auto* local = reinterpret_cast<uint8_t*>(localScb);
    if (ctx == nullptr || local == nullptr || peerBlockId < 0 || peerBlockId >= static_cast<int>(ctx->windowCount)) {
        return nullptr;
    }
    for (uint32_t window = 0; window < ctx->windowCount; ++window) {
        uint8_t* base = ctx->windowBase + window * ctx->windowBytes;
        if (local >= base && local < base + ctx->windowBytes) {
            const uint32_t offset = static_cast<uint32_t>(local - base);
            return reinterpret_cast<uint32_t*>(ctx->windowBase + peerBlockId * ctx->windowBytes + offset);
        }
    }
    return nullptr;
}

} // namespace a2a3_grid_payload
} // namespace pto

namespace {

TEST(GridBindQueueTest, ConcurrentProducersAreDequeuedOneAtATime)
{
    constexpr uint32_t windowBytes = pto::a2a3_grid::WindowBytes<BindQueuePipe>();
    static_assert(windowBytes % pto::grid_mock::kScbLineStride == 0);

    std::vector<uint8_t> rawWindows(windowBytes * kPeerCount + pto::grid_mock::kScbLineStride - 1, 0);
    const uintptr_t rawAddress = reinterpret_cast<uintptr_t>(rawWindows.data());
    const uintptr_t alignedAddress =
        (rawAddress + pto::grid_mock::kScbLineStride - 1) & ~(pto::grid_mock::kScbLineStride - 1);
    auto* windows = reinterpret_cast<uint8_t*>(alignedAddress);
    BindQueueTestContext context{windows, windowBytes, kPeerCount};

    std::array<BindQueuePipe, kProducerCount> producers{};
    BindQueuePipe consumer{};
    const pto::GridShape shape{1, static_cast<int>(kPeerCount)};
    for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
        pto::a2a3_grid::InitGridPipeFromWindow(
            producers[producer], shape, pto::GridCoord{0, static_cast<int>(producer)}, windows + producer * windowBytes,
            &context, 0);
    }
    pto::a2a3_grid::InitGridPipeFromWindow(
        consumer, shape, pto::GridCoord{0, static_cast<int>(kConsumerId)}, windows + kConsumerId * windowBytes,
        &context, 0);

    std::atomic<uint32_t> startCount{0};
    std::atomic<bool> start{false};
    std::array<std::atomic<int>, kProducerCount> bindResults{};
    std::array<std::thread, kProducerCount> producerThreads;
    for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
        bindResults[producer].store(pto::kGridInvalidChan, std::memory_order_relaxed);
        producerThreads[producer] = std::thread([&, producer] {
            startCount.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const int channel =
                pto::grid_detail::EnsureOutgoingConsumerBinding(producers[producer], kConsumerId, 10000000U);
            bindResults[producer].store(channel, std::memory_order_release);
        });
    }
    while (startCount.load(std::memory_order_acquire) != kProducerCount) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    // Wait until all four remote writers have committed distinct request lines.
    // None can return yet because the consumer has not serviced a request.
    uint32_t committed = 0;
    for (uint32_t spin = 0; spin < 10000000U && committed != kProducerCount; ++spin) {
        committed = 0;
        for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
            const uint32_t value = pto::grid_cce_detail::read_local_word(
                consumer.BindRequestQueueEntry(producer) + pto::kGridBindQueueCommitWord);
            committed += value != pto::kGridBindPending ? 1U : 0U;
        }
        if (committed != kProducerCount) {
            std::this_thread::yield();
        }
    }
    EXPECT_EQ(committed, kProducerCount);

    for (uint32_t serviced = 0; serviced < kProducerCount; ++serviced) {
        int acceptedChannel = pto::kGridInvalidChan;
        const auto result =
            pto::grid_detail::ServiceOneDynamicBindQueueRequest(consumer, pto::kGridNoPeer, acceptedChannel);
        EXPECT_EQ(result, pto::grid_detail::DynamicBindQueueServiceResult::ACCEPTED);
        if (result != pto::grid_detail::DynamicBindQueueServiceResult::ACCEPTED ||
            acceptedChannel == pto::kGridInvalidChan) {
            break;
        }
        EXPECT_EQ(consumer.consChanProdId[acceptedChannel], serviced);

        uint32_t remainingRequests = 0;
        uint32_t completedResponses = 0;
        for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
            const uint32_t requestCommit = pto::grid_cce_detail::read_local_word(
                consumer.BindRequestQueueEntry(producer) + pto::kGridBindQueueCommitWord);
            remainingRequests += requestCommit != pto::kGridBindPending ? 1U : 0U;
            const uint32_t responseCommit = pto::grid_cce_detail::read_local_word(
                producers[producer].BindResponseQueueEntry(kConsumerId) + pto::kGridBindQueueResponseCommitWord);
            completedResponses += responseCommit != pto::kGridBindPending ? 1U : 0U;
        }
        EXPECT_EQ(remainingRequests, kProducerCount - serviced - 1);
        EXPECT_EQ(completedResponses, serviced + 1);
    }

    for (auto& thread : producerThreads) {
        thread.join();
    }
    for (uint32_t producer = 0; producer < kProducerCount; ++producer) {
        EXPECT_EQ(bindResults[producer].load(std::memory_order_acquire), BindQueuePipe::ChanCount - 1);
        EXPECT_EQ(producers[producer].consumers.StateOf(kConsumerId), pto::GridConsumerState::ACTIVE);
        EXPECT_EQ(
            pto::grid_cce_detail::read_local_word(
                producers[producer].BindResponseQueueEntry(kConsumerId) + pto::kGridBindQueueResponseTokenWord),
            producers[producer].bindRequestToken);
    }
}

} // namespace

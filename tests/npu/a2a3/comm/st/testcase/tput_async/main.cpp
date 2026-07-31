/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>

#include "tput_async_kernel.h"
#include "../comm_mpi.h"

namespace {

int FirstDeviceId()
{
    const char* value = std::getenv("PTO_COMM_ST_FIRST_DEVICE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && parsed >= 0 ? static_cast<int>(parsed) : 0;
}

class TPutAsyncPostStability2Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 2;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 2 MPI ranks";
        }
        if (!IsTPutAsyncPostStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

class TPutAsyncPostStability3Ranks : public ::testing::Test {
protected:
    void SetUp() override
    {
        constexpr int kRanks = 3;
        if (CommMpiSize() != kRanks) {
            GTEST_SKIP() << "Requires exactly 3 MPI ranks";
        }
        if (!IsTPutAsyncPostStabilityDeviceRangeAvailable(kRanks, FirstDeviceId())) {
            GTEST_SKIP() << "Requested device range is unavailable";
        }
    }
};

} // namespace

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall_4Ranks) { ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0))); }
TEST(TPutAsync, Vec_Int32Large) { ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TPutAsync, Vec_Uint8Small_8Ranks) { ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(8, 8, 0, 0))); }

// ============================================================================
// Configurable SdmaBaseConfig Tests
// ============================================================================
TEST(TPutAsync, Vec_Int32_QueueNum2) { ASSERT_TRUE((RunPutAsyncWithConfig<int32_t, 4096>(2, 2, 0, 0, 4096, 0, 2))); }
TEST(TPutAsync, Vec_Float_SmallBlockBytes)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 4096>(2, 2, 0, 0, 4096, 0, 1)));
}
TEST(TPutAsync, Vec_Float_LargeBlockBytes)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 4096>(2, 2, 0, 0, 2 * 1024 * 1024, 0, 1)));
}
TEST(TPutAsync, Vec_Float_CommOffset)
{
    ASSERT_TRUE((RunPutAsyncWithConfig<float, 2048>(2, 2, 0, 0, 1024 * 1024, 1024 * sizeof(float), 1)));
}

// ============================================================================
// Multi-Core Tests (blockDim > 1)
// ============================================================================
TEST(TPutAsync, Vec_Float_MultiCoreSplit) { ASSERT_TRUE((RunPutAsyncMultiCore<float, 2048>(2, 2, 0, 0, 2, 0))); }
TEST(TPutAsync, Vec_Float_MultiCoreIndep) { ASSERT_TRUE((RunPutAsyncMultiCore<float, 256>(2, 2, 0, 0, 2, 1))); }

// ============================================================================
// Concurrent Per-Rank Scatter Tests (every rank: nranks cores, distinct channels)
// Mirrors the tget_async ConcurrentRank pattern with TPUT_ASYNC.
// ============================================================================
// iters=1: single TPUT+Wait per core session (baseline concurrency).
TEST(TPutAsync, ConcurrentRank_Float_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 1, 0)));
}
TEST(TPutAsync, ConcurrentRank_Int32_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<int32_t, 8192>(8, 8, 0, 0, 1, 0)));
}
// iters=16 reusing one session per core.
TEST(TPutAsync, ConcurrentRank_FloatIter16Reuse_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 0)));
}
// iters=16 rebuilding a fresh session each round.
TEST(TPutAsync, ConcurrentRank_FloatIter16Fresh_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncConcurrentRank<float, 8192>(8, 8, 0, 0, 16, 1)));
}

// Issue four Q1 Posts between two ranks, waiting for and remotely consuming each result immediately.
TEST_F(TPutAsyncPostStability2Ranks, Immediate_P4_Q1)
{
    ASSERT_TRUE(RunTPutAsyncImmediatePostWait(2, 2, 0, FirstDeviceId(), 4, 1, 1));
}

// Issue 16 Q4 Posts before waiting for each saved event and validating remote consumption.
TEST_F(TPutAsyncPostStability2Ranks, Deferred_P16_Q4)
{
    ASSERT_TRUE(RunTPutAsyncConsecutivePostsWaitEach(2, 2, 0, FirstDeviceId(), 16, 1, 4));
}

// Run 512 immediate-wait Q4 Posts to exercise repeated reuse of the 64-slot flag payload ring.
TEST_F(TPutAsyncPostStability2Ranks, Immediate_512Posts_Q4)
{
    ASSERT_TRUE(RunTPutAsyncImmediatePostWait(2, 2, 0, FirstDeviceId(), 8, 64, 4));
}

// Use separate destinations for a Q4 Post and a smaller Q1 Post; waiting only for the latter must complete both.
TEST_F(TPutAsyncPostStability2Ranks, UsedQueueCount_Q4ThenQ1_LastWaitOnly)
{
    ASSERT_TRUE(RunTPutAsyncPostsWaitFinal(2, 2, 0, FirstDeviceId(), 2, 1, 4));
}

// Issue eight deferred Q4 Posts from the root to each of two peers and verify remote consumption.
TEST_F(TPutAsyncPostStability3Ranks, Deferred_P8_Q4_3Ranks)
{
    ASSERT_TRUE(RunTPutAsyncConsecutivePostsWaitEach(3, 3, 0, FirstDeviceId(), 8, 1, 4));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "probe.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <gtest/gtest.h>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace {
void CheckProbe(const TargetProbe& result, int arch, bool fixed)
{
    EXPECT_EQ(result.architecture, arch);
    EXPECT_EQ(result.ubBytes, (arch == 5 ? 256 : 192) * 1024u);
    EXPECT_EQ(result.accBytes, (arch == 5 ? 256 : 128) * 1024u);
    EXPECT_TRUE(result.convertsBfloat16);
    TargetProbe legacy{};
    if (arch == 2) {
        // The legacy compatibility loop permits compiler-specific vectorization.
        probe_legacy(&legacy, 2);
    }
    for (std::size_t row = 0; row < 8; ++row) {
        EXPECT_EQ(result.rowSums[row], arch == 5 ? 0.0f : legacy.rowSums[row]);
#if defined(__GNUC__) && !defined(__clang__)
        if (arch == 2) {
            EXPECT_EQ(result.rowSums[row], 1.0f);
        }
#endif
    }
    EXPECT_EQ(std::isinf(result.loadPadding), arch == 5);
    if (arch != 5) {
        EXPECT_EQ(result.loadPadding, 0.0f);
    } else {
        EXPECT_TRUE(result.a5HighAddress);
    }
    if (fixed) {
        EXPECT_TRUE(result.rejectsInitialize);
        EXPECT_TRUE(result.rejectsDefault);
        EXPECT_TRUE(result.rejectsMemoryInit);
        EXPECT_EQ(result.capabilityArch, arch == 5 ? 1 : 0);
        EXPECT_EQ(result.supportsFp8, arch == 5);
    }
}

void CheckInterleaved(ProbeFunction first, int firstArch, ProbeFunction second, int secondArch)
{
    std::array<TargetProbe, 3> results{};
    first(&results[0], 0);
    second(&results[1], 0);
    first(&results[2], 0);
    CheckProbe(results[0], firstArch, true);
    CheckProbe(results[1], secondArch, true);
    CheckProbe(results[2], firstArch, true);
    EXPECT_NE(results[0].memoryModel, results[1].memoryModel);
    EXPECT_NE(results[0].ubStorage, results[1].ubStorage);
    EXPECT_EQ(results[0].memoryModel, results[2].memoryModel);
    EXPECT_EQ(results[0].ubStorage, results[2].ubStorage);
}
} // namespace

TEST(CpuTarget, LinkedTranslationUnitsKeepDistinctTargets)
{
    CheckInterleaved(probe_a2a3, 2, probe_a5, 5);
    CheckInterleaved(probe_a5, 5, probe_a2a3, 2);
    std::thread worker([] { CheckInterleaved(probe_a5, 5, probe_a2a3, 2); });
    worker.join();
}

TEST(CpuTarget, FixedTargetsIgnoreConflictingEnvironment)
{
    for (const char* arch : {"a5", "a2a3", "invalid"}) {
        ASSERT_EQ(setenv("PTO_CPU_SIM_ARCH", arch, 1), 0);
        std::thread worker([] { CheckInterleaved(probe_a5, 5, probe_a2a3, 2); });
        worker.join();
    }
    ASSERT_EQ(unsetenv("PTO_CPU_SIM_ARCH"), 0);
}

TEST(CpuTarget, LegacyCanSwitchWithoutChangingFixedTargets)
{
    std::thread worker([] {
        TargetProbe a2a3{}, a5{};
        probe_legacy(&a2a3, 2);
        probe_a5(&a5, 0);
        CheckProbe(a2a3, 2, false);
        EXPECT_NE(a2a3.memoryModel, a5.memoryModel);
        EXPECT_NE(a2a3.ubStorage, a5.ubStorage);
        probe_legacy(&a5, 5);
        CheckProbe(a5, 5, false);
        EXPECT_EQ(a2a3.memoryModel, a5.memoryModel);
        CheckInterleaved(probe_a2a3, 2, probe_a5, 5);
        probe_legacy(&a2a3, 2);
        CheckProbe(a2a3, 2, false);
    });
    worker.join();
}

#if defined(__unix__) || defined(__APPLE__)
void CheckDynamicLibraries(int first)
{
    const std::array<const char*, 2> paths{PTO_TARGET_LIB_A2A3, PTO_TARGET_LIB_A5};
    {
        std::array<void*, 2> libraries{};
        for (int index : {first, 1 - first}) {
            libraries[index] = dlopen(paths[index], RTLD_NOW | RTLD_GLOBAL);
            ASSERT_NE(libraries[index], nullptr) << dlerror();
        }
        auto a2a3 = reinterpret_cast<ProbeFunction>(dlsym(libraries[0], "probe_dynamic_a2a3"));
        auto a5 = reinterpret_cast<ProbeFunction>(dlsym(libraries[1], "probe_dynamic_a5"));
        ASSERT_NE(a2a3, nullptr);
        ASSERT_NE(a5, nullptr);
        std::thread worker([=] {
            if (first == 0) {
                CheckInterleaved(a2a3, 2, a5, 5);
            } else {
                CheckInterleaved(a5, 5, a2a3, 2);
            }
        });
        worker.join();
        ASSERT_EQ(dlclose(libraries[1 - first]), 0);
        ASSERT_EQ(dlclose(libraries[first]), 0);
    }
}
TEST(CpuTarget, GloballyLoadedLibrariesA2A3First)
{
    ASSERT_EXIT(
        {
            CheckDynamicLibraries(0);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(CpuTarget, GloballyLoadedLibrariesA5First)
{
    ASSERT_EXIT(
        {
            CheckDynamicLibraries(1);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), "");
}
#endif

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstddef>
#include <cstdint>

// Single-card host runners for public TPREFETCH_ASYNC correctness and L2
// access-latency tests. Defined in tprefetch_async_kernel.cpp.
template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(
    int deviceId, uint32_t postCount = 1U, bool waitEachEvent = true, bool useExternalSession = false);

bool RunPrefetchAsyncL2Benefit(int deviceId, double& coldAverageUs, double& prefetchedAverageUs);

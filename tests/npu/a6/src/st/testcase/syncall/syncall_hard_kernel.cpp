/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIV-only hard (FFTS) SYNCALL ST, dav-920r1-vec single-chevron launch. The
// AIV-only path is a plain ffts_cross_core_sync + wait_flag_dev pair that never
// touches the intra-block AIC/AIV channel, so it is unaffected by the CV ratio.

#include "syncall_hard_common.hpp"

constexpr int32_t kAivHardBlockCount = 18;

extern "C" __global__ AICORE void RunHardSyncAll(__gm__ int32_t __out__* out, __gm__ int32_t __out__* flags)
{
    RunHardSyncAllBody<kAivHardBlockCount, SyncCoreType::AIVOnly>(out, flags, static_cast<int32_t>(get_block_idx()));
}

void LaunchHardSyncAll(int32_t* out, int32_t* flags, void* stream)
{
    RunHardSyncAll<<<kAivHardBlockCount, nullptr, stream>>>(out, flags);
}

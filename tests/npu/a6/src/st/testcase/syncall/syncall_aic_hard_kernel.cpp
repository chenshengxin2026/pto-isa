/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only hard (FFTS) SYNCALL ST, dav-920r1-cube single-chevron launch. Like the
// AIV-only case this never reaches the intra-block handshake, so it serves as the
// control for the MIX case: if this passes and MIX hangs, the fault is in the
// AIC/AIV pairing rather than in FFTS itself.

#include "syncall_hard_common.hpp"

constexpr int32_t kAicHardBlockCount = 18;

extern "C" __global__ AICORE void RunHardSyncAllAIC(__gm__ int32_t __out__* out, __gm__ int32_t __out__* flags)
{
    RunHardSyncAllBody<kAicHardBlockCount, SyncCoreType::AICOnly>(out, flags, static_cast<int32_t>(get_block_idx()));
}

void LaunchHardSyncAllAIC(int32_t* out, int32_t* flags, void* stream)
{
    RunHardSyncAllAIC<<<kAicHardBlockCount, nullptr, stream>>>(out, flags);
}

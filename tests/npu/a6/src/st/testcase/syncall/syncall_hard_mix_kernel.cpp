/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MIX hard (FFTS) SYNCALL ST for A6, bare dav-920r1 auto-split single-chevron
// launch. One __global__ symbol is compiled for both cube and vector; A6 pairs one
// AIC with one AIV, so the chevron spawns 1 cube + 1 vector per block and 18 blocks
// give 18 + 18 = 36 participants.
//
// This is the case that regresses when SYNCALL<Mix>() assumes A5's 1:2 ratio: the
// cube side would wait on the subblock-1 intra-block flag that no core sets. The
// three barriers in RunHardSyncAllBody also cover the second symptom, where a spare
// cube-side credit lets a later barrier through without the AIV having arrived.

#include "syncall_hard_common.hpp"

constexpr int32_t kMixAicBlocks = 18;
constexpr int32_t kMixParticipants = kMixAicBlocks * 2;

// Cube blocks take the low slots, their paired vector cores the high ones.
PTO_INTERNAL int32_t GetMixLogicalIdx()
{
#if defined(__DAV_VEC__)
    return kMixAicBlocks + static_cast<int32_t>(get_block_idx());
#else
    return static_cast<int32_t>(get_block_idx());
#endif
}

extern "C" __global__ AICORE void RunHardSyncAllMix(__gm__ int32_t __out__* out, __gm__ int32_t __out__* flags)
{
    RunHardSyncAllBody<kMixParticipants, SyncCoreType::Mix>(out, flags, GetMixLogicalIdx());
}

void LaunchHardSyncAllMix(int32_t* out, int32_t* flags, void* stream)
{
    RunHardSyncAllMix<<<kMixAicBlocks, nullptr, stream>>>(out, flags);
}

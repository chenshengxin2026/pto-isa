/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN " AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6_SYNCALL_HPP
#define PTO_NPU_A6_SYNCALL_HPP

#include <pto/common/arch_macro.hpp>
#include <pto/npu/a6/TSync.hpp>
#include <pto/common/syncall_soft.hpp>

namespace pto {

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8

PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return (
        (baseConst & FFTS_BASE_COUNT_WIDTH) + ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
        ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
#if defined(PTO_COMPILE_VEC)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL);
#endif
        return;
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#if defined(PTO_COMPILE_CUBE)
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
#endif
        return;
    }

#if defined(PTO_COMPILE_CUBE)
    // A6 pairs one AIC with a single AIV, so the subblock-1 channel that A5 uses
    // (flag id + SYNC_FLAG_ID_MAX) has no core on the other end: waiting on it would
    // hang and setting it would leave a credit that releases the next barrier early.
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#elif defined(PTO_COMPILE_VEC)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
#endif
}
} // namespace pto
#endif

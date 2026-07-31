/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP
#define DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP

#include "kernel_operator.h"

constexpr uint16_t kA5VecSubblockFlagOffset = 16;

template <uint8_t ModeId, pipe_t Pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
{
    (void)ModeId;
    static_assert(
        Pipe == PIPE_MTE3 || Pipe == PIPE_FIX,
        "CrossCoreSetFlag only supports AIV producer PIPE_MTE3 or AIC producer PIPE_FIX.");
    if constexpr (Pipe == PIPE_MTE3) {
        if ASCEND_IS_AIV {
            set_intra_block(Pipe, flagId);
        }
    }
    if constexpr (Pipe == PIPE_FIX) {
        if ASCEND_IS_AIC {
            set_intra_block(Pipe, flagId);
            set_intra_block(Pipe, static_cast<uint16_t>(flagId + kA5VecSubblockFlagOffset));
        }
    }
}

template <uint8_t ModeId>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
{
    (void)ModeId;
    if ASCEND_IS_AIV {
        // Producers publish GM completion on their output pipe; consumers wait before loading through MTE2.
        wait_intra_block(PIPE_MTE2, flagId);
    }
    if ASCEND_IS_AIC {
        wait_intra_block(PIPE_MTE2, flagId);
        wait_intra_block(PIPE_MTE2, static_cast<uint16_t>(flagId + kA5VecSubblockFlagOffset));
    }
}

#endif // DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP

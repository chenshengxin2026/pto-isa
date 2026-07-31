/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H
#define DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H

#include <pto/pto-inst.hpp>

#include "utils/const_args.hpp"

using Gmm2CombineCvPipe = pto::TPipe<
    MEGA_MOE_GMM2_COMBINE_CV_READY_HARD_FLAG, pto::Direction::DIR_C2V, MEGA_MOE_GMM2_COMBINE_CV_SLOT_BYTES,
    MEGA_MOE_GMM2_COMBINE_CV_DEFAULT_FIFO_DEPTH, MEGA_MOE_GMM2_COMBINE_CV_DEFAULT_FIFO_DEPTH, true>;

static_assert(MEGA_MOE_GMM2_COMBINE_CV_MAX_AIV_PER_AIC == 2U);

AICORE inline uint32_t Gmm2CombineCvLane(uint32_t tileIndex) { return tileIndex & 1U; }

class Gmm2CombineCvProducer {
public:
    AICORE inline uint32_t NextLane() const { return Gmm2CombineCvLane(tileIndex_); }

    AICORE inline void WaitLaneFree() const
    {
        if (tileIndex_ < MEGA_MOE_GMM2_COMBINE_CV_MAX_AIV_PER_AIC) {
            return;
        }
#ifdef __DAV_CUBE__
        wait_intra_block(PIPE_FIX, PhysicalFlag(MEGA_MOE_GMM2_COMBINE_CV_FREE_HARD_FLAG, NextLane()));
#endif
    }

    AICORE inline void RecordLaneReady()
    {
#ifdef __DAV_CUBE__
        set_intra_block(PIPE_FIX, PhysicalFlag(MEGA_MOE_GMM2_COMBINE_CV_READY_HARD_FLAG, NextLane()));
#endif
        ++tileIndex_;
    }

    AICORE inline void Drain() const
    {
#ifdef __DAV_CUBE__
        if (tileIndex_ != 0U) {
            wait_intra_block(PIPE_FIX, PhysicalFlag(MEGA_MOE_GMM2_COMBINE_CV_FREE_HARD_FLAG, 0U));
        }
        if (tileIndex_ > 1U) {
            wait_intra_block(PIPE_FIX, PhysicalFlag(MEGA_MOE_GMM2_COMBINE_CV_FREE_HARD_FLAG, 1U));
        }
#endif
    }

private:
    AICORE inline static uint16_t PhysicalFlag(uint16_t logicalFlag, uint32_t lane)
    {
        return static_cast<uint16_t>(logicalFlag + lane * Gmm2CombineCvPipe::VEC_CORE_ID_OFFSET);
    }

    uint32_t tileIndex_ = 0U;
};

#endif // DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>

#ifndef PTO_NPU_ARCH_A5
#define PTO_NPU_ARCH_A5
#endif

#ifndef __CCE_AICORE__
#define __CCE_AICORE__
#endif

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine.h"
#include "kernel_launch.hpp"
#include "op_kernel/utils/hccl_window.hpp"

extern "C" __global__ __aicore__ void dispatch_mega_combine_kernel(
    GM_ADDR fftsAddr, GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2,
    GM_ADDR probs, GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR workspaceGM, GM_ADDR tilingGM, uint32_t startSync)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    if (workspaceGM != nullptr && tilingGM != nullptr) {
        const __gm__ MegaMoeTilingData* tilingData = reinterpret_cast<__gm__ MegaMoeTilingData*>(tilingGM);
        if (startSync != 0U) {
            PtoRemoteWindow remoteWindow;
            remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext));
            remoteWindow.CrossRankStartSyncAiv();
            remoteWindow.CrossRankStartSyncAic();
            pto::SYNCALL<pto::SyncCoreType::Mix>();
        }
        MegaMoe<bfloat16_t> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingData);
        op.Process();
    }

    pipe_barrier(PIPE_ALL);
}

void launchMegaMoe(const MegaMoeLaunchArgs& args, void* stream)
{
    dispatch_mega_combine_kernel<<<args.block_dim, nullptr, stream>>>(
        (GM_ADDR)args.ffts, (GM_ADDR)args.x, (GM_ADDR)args.weight1, (GM_ADDR)args.weight2, (GM_ADDR)args.expert_idx,
        (GM_ADDR)args.scale1, (GM_ADDR)args.scale2, (GM_ADDR)args.probs, (GM_ADDR)args.out,
        (GM_ADDR)args.expert_token_nums, (GM_ADDR)args.workspace, (GM_ADDR)args.tiling, args.start_sync);
}

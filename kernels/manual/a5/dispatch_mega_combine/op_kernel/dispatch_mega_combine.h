/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_H
#define DISPATCH_MEGA_COMBINE_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#if defined(__DAV_VEC__)
#include "combine.h"
#include "dispatch.h"
#include "front_reorder.h"
#include "front_fullload_sort.h"
#include "front_vms_sort.h"
#endif
#if defined(__DAV_CUBE__)
#include "gmm1.h"
#include "gmm2.h"
#endif
#if defined(__DAV_VEC__)
#include "swiglu.h"
#include "unpermute.h"
#endif

#if defined(__DAV_VEC__)
template <typename InputElement>
AICORE inline void FrontRunVmsSort(FrontReorderVmsSort<InputElement>& path)
{
    if ASCEND_IS_AIV {
        path.RunSort();
        FrontRunPostSortPipeline<InputElement>(path);
    }
}

template <typename InputElement>
AICORE inline void FrontRunFullLoad(FrontReorderFullLoad<InputElement>& path)
{
    if ASCEND_IS_AIV {
        path.RunFullLoadSort();
        path.StoreExpandedRowIdxToGm();
        path.BuildLocalTokenPerExpertFromSort();
        path.QuantAndScatterPackedRows();
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        FrontRunEndPostprocess(path);
    }
}

template <typename InputElement>
AICORE inline void FrontReorderProcess(
    GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    const uint32_t frontCase = tilingData->frontReorderTiling.frontCase;
    if (frontCase == kFrontCaseFullLoadDynamic) {
        FrontReorderFullLoad<InputElement> path;
        path.Init(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        FrontRunFullLoad(path);
        return;
    }
    if (frontCase == kFrontCaseOneCoreDynamic || frontCase == kFrontCaseMultiCoreDynamic) {
        FrontReorderVmsSort<InputElement> path;
        path.Init(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        FrontRunVmsSort(path);
    }
}
#endif

template <typename CType_>
class MegaMoe {
public:
    __aicore__ inline void Init(
        GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
        GM_ADDR probs, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GM_ADDR xGM_ = nullptr;
    GM_ADDR weight1GM_ = nullptr;
    GM_ADDR weight2GM_ = nullptr;
    GM_ADDR scale1GM_ = nullptr;
    GM_ADDR scale2GM_ = nullptr;
    GM_ADDR expertIdGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR probsGM_ = nullptr;
    GM_ADDR outGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
};

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::Init(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    expertIdGM_ = expertIdGM;
    expertTokenNumsGM_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
    probsGM_ = probs;
    outGM_ = outGM;
    tilingData_ = tilingData;
}

template <typename CType_>
__aicore__ inline void MegaMoe<CType_>::Process()
{
    using OutputElement = half;
#if defined(__DAV_VEC__)
    FrontReorderProcess<CType_>(xGM_, expertIdGM_, expertTokenNumsGM_, workspaceGM_, tilingData_);

    DispatchGather dispatchGather;
    dispatchGather.Init(workspaceGM_, tilingData_);
    dispatchGather.Process();
#endif
#if defined(__DAV_CUBE__)
    Gmm1 gmm1;
    gmm1.Init(weight1GM_, scale1GM_, workspaceGM_, tilingData_);
    gmm1.Process();
#endif
#if defined(__DAV_VEC__)
    Swiglu swiglu;
    swiglu.Init(workspaceGM_, tilingData_);
    swiglu.Process();
#endif
#if defined(__DAV_CUBE__)
    Gmm2 gmm2;
    gmm2.Init(weight2GM_, scale2GM_, workspaceGM_, tilingData_);
    gmm2.Process();
#endif
#if defined(__DAV_VEC__)
    Combine<OutputElement> combine;
    combine.Init(workspaceGM_, tilingData_);
    combine.Process();

    Unpermute<OutputElement> unpermute;
    unpermute.Init(workspaceGM_, probsGM_, outGM_, tilingData_);
    unpermute.Process();
#endif
}

#endif // DISPATCH_MEGA_COMBINE_H

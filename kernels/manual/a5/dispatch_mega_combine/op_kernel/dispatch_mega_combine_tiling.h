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

#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kFrontCaseFullLoadDynamic = 21000U;
constexpr uint32_t kFrontCaseOneCoreDynamic = 11000U;
constexpr uint32_t kFrontCaseMultiCoreDynamic = 11010U;

struct MegaMoeInfo {
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    uint32_t expertPerRank = 0;
    uint32_t maxOutputSize = 0;
    uint32_t topK = 0;
};

struct MegaMoeRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

struct MegaMoeFrontReorderTiling {
    uint32_t routeElems = 0;
    uint16_t frontCase = 0;
    uint32_t expertNum = 0;
    uint32_t expertNumAligned = 0;
    uint32_t sortLoopMaxElement = 0;
    uint32_t sortLastCorePerLoopElems = 0;
    uint32_t expandedRowIdxOffset = 0;
    uint32_t localTokenPerExpertOffset = 0;
    uint32_t cumsumMMOffset = 0;
    uint32_t preSumBeforeRankOffset = 0;
    uint32_t alignedRouteElems = 0;
    uint32_t sortPerCoreElems = 0;
    uint32_t sortLastCoreElems = 0;
    uint32_t sortPerCoreLoops = 0;
    uint32_t sortPerCorePerLoopElems = 0;
    uint32_t sortPerCoreLastLoopElems = 0;
    uint32_t sortLastCoreLoops = 0;
    uint32_t sortLastCoreLastLoopElems = 0;
    uint16_t sortNeedCoreNum = 0;
    uint16_t sortOutLoopMaxElems = 0;
    uint32_t frontExpandedExpertOffset = 0;
    uint32_t frontExpandDstToSrcOffset = 0;
    uint32_t frontSortWs0Offset = 0;
    uint32_t frontSortWs1Offset = 0;
    uint32_t frontWorkspaceBytes = 0;
};

struct MegaMoeDispatchTiling {
    uint64_t gmAOffset = 0;
    uint64_t perTokenScaleOffset = 0;
};

struct MegaMoeGmm1Tiling {
    uint64_t gmCOffset = 0;
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
};

struct MegaMoeSwigluTiling {
    uint64_t gmPermutedTokenOffset = 0;
    uint64_t perTokenScale2Offset = 0;
    uint64_t swigluSegmentMetaOffset = 0;
};

struct alignas(64) MegaMoeSwigluSegmentRuntimeMeta {
    uint32_t segmentRowBase = 0;
    uint32_t rowSplitBase = 0;
    uint32_t rowSplitRem = 0;
};

struct MegaMoeGmm2Tiling {
    uint32_t l1TileM = 128;
    uint32_t l1TileN = 256;
};

struct MegaMoeUnpermuteTiling {
    uint32_t unpermuteTileCols = 1024;
    uint32_t unpermuteTokenBatch = 256;
};

static_assert(sizeof(MegaMoeSwigluSegmentRuntimeMeta) == 64);

struct MegaMoeTilingData {
    MegaMoeInfo megaMoeInfo;
    MegaMoeRuntimeInfo runtimeInfo;
    MegaMoeFrontReorderTiling frontReorderTiling;
    MegaMoeDispatchTiling dispatchTiling;
    MegaMoeGmm1Tiling gmm1Tiling;
    MegaMoeSwigluTiling swigluTiling;
    MegaMoeGmm2Tiling gmm2Tiling;
    MegaMoeUnpermuteTiling unpermuteTiling;
};

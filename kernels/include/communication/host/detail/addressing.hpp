/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/addressing.hpp is host-only"
#endif

#include <cstddef>
#include <iostream>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"
#include "communication/host/detail/hccl_tiling.hpp"
#include "communication/host/detail/mpi_helpers.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);

#ifndef HCCL_RANK_GRAPH_H
using CommTopo = uint32_t;
inline constexpr uint32_t kCommTopoMesh = 0b1u;
#else
inline constexpr CommTopo kCommTopoMesh = COMM_TOPO_1DMESH;
#endif
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t isSetDevice);
inline constexpr uint32_t kCommIsNotSetDevice = 0;

inline bool InitMeshWindow(CommContext& ctx, void* ctxPtr)
{
    ctx.winDevCtx = reinterpret_cast<CommDeviceContext*>(ctxPtr);
    ctx.ownsWinDevCtx = false;
    aclError aRet = aclrtMemcpy(
        &ctx.winHostCtx, sizeof(ctx.winHostCtx), ctx.winDevCtx, sizeof(ctx.winHostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[PTO-DOMAIN] MESH D2H CommDeviceContext failed: " << static_cast<int>(aRet) << "\n";
        return false;
    }
    ctx.winBase = reinterpret_cast<void*>(ctx.winHostCtx.windowsIn[ctx.winHostCtx.rankId]);
    return true;
}

inline bool FillRingWindowsFromRemote(
    CommContext& ctx, const CommOpResParamHead& head, const std::vector<RemoteResPtr>& remoteResArr)
{
    memset_s(&ctx.winHostCtx, sizeof(ctx.winHostCtx), 0, sizeof(ctx.winHostCtx));
    ctx.winHostCtx.rankId = head.localUsrRankId;
    ctx.winHostCtx.rankNum = head.rankSize;
    ctx.winHostCtx.winSize = head.winSize;
    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            ctx.winHostCtx.windowsIn[i] = head.localWindowsIn;
            continue;
        }
        uint64_t devPtr = remoteResArr[i].nextDevicePtr;
        if (devPtr == 0) {
            std::cerr << "[PTO-DOMAIN] RING remoteRes[" << i << "] null\n";
            return false;
        }
        CommRankRelationResV2 remoteInfo{};
        aclError aRet = aclrtMemcpy(
            &remoteInfo, sizeof(remoteInfo), reinterpret_cast<void*>(devPtr), sizeof(remoteInfo),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[PTO-DOMAIN] RING read relation failed\n";
            return false;
        }
        ctx.winHostCtx.windowsIn[i] = remoteInfo.windowsIn;
    }
    return true;
}

inline bool UploadRingDeviceCtx(CommContext& ctx)
{
    void* newDevMem = nullptr;
    aclError aRet = aclrtMalloc(&newDevMem, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
        std::cerr << "[PTO-DOMAIN] RING aclrtMalloc deviceCtx failed\n";
        return false;
    }
    aRet = aclrtMemcpy(
        newDevMem, sizeof(CommDeviceContext), &ctx.winHostCtx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        aclrtFree(newDevMem);
        std::cerr << "[PTO-DOMAIN] RING H2D deviceCtx failed\n";
        return false;
    }
    ctx.winDevCtx = reinterpret_cast<CommDeviceContext*>(newDevMem);
    ctx.ownsWinDevCtx = true;
    ctx.winBase = reinterpret_cast<void*>(ctx.winHostCtx.windowsIn[ctx.winHostCtx.rankId]);
    return true;
}

inline bool InitRingWindow(CommContext& ctx, void* ctxPtr)
{
    auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);
    CommOpResParamHead head{};
    const size_t headOff = offsetof(CommOpResParam, localUsrRankId);
    aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS || head.rankSize == 0 || head.rankSize > kMaxRankNum) {
        std::cerr << "[PTO-DOMAIN] RING read head failed or bad rankSize\n";
        return false;
    }
    std::vector<RemoteResPtr> remoteResArr(head.rankSize);
    const size_t remoteResOff = offsetof(CommOpResParam, remoteRes);
    const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
    aRet = aclrtMemcpy(
        remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[PTO-DOMAIN] RING read remoteRes failed\n";
        return false;
    }
    uint64_t wsFields[2] = {0, 0};
    (void)aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!FillRingWindowsFromRemote(ctx, head, remoteResArr)) {
        return false;
    }
    ctx.winHostCtx.workSpace = wsFields[0];
    ctx.winHostCtx.workSpaceSize = wsFields[1];
    return UploadRingDeviceCtx(ctx);
}

inline bool AllocWindowByTiling(CommContext& ctx)
{
    char group[128] = {};
    HcclResult hret = HcclGetCommName(ctx.comm, group);
    if (hret != HCCL_SUCCESS) {
        std::cerr << "[PTO-DOMAIN] HcclGetCommName failed\n";
        return false;
    }
    CommTopo topoRet = static_cast<CommTopo>(0);
    hret = HcomGetL0TopoTypeEx(group, &topoRet, kCommIsNotSetDevice);
    if (hret != HCCL_SUCCESS) {
        std::cerr << "[PTO-DOMAIN] HcomGetL0TopoTypeEx failed\n";
        return false;
    }
    HcclComm commHandle = nullptr;
    hret = HcomGetCommHandleByGroup(group, &commHandle);
    if (hret != HCCL_SUCCESS || commHandle == nullptr) {
        std::cerr << "[PTO-DOMAIN] HcomGetCommHandleByGroup failed\n";
        return false;
    }
    if (ctx.bootstrap == Bootstrap::Mpi) {
        DomainMpiBarrier();
    }
    Mc2CommConfigV2 tiling{};
    FillDefaultMc2TilingV2(tiling, group);
    void* ctxPtr = nullptr;
    hret = HcclAllocComResourceByTiling(commHandle, ctx.stream, &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
        std::cerr << "[PTO-DOMAIN] HcclAllocComResourceByTiling failed: " << static_cast<int>(hret) << "\n";
        return false;
    }
    if (topoRet == kCommTopoMesh) {
        return InitMeshWindow(ctx, ctxPtr);
    }
    return InitRingWindow(ctx, ctxPtr);
}

#ifdef PTO_DOMAIN_URMA_HOST
inline bool SetupUrmaDevBuf(const CommConfig& cfg, CommContext& ctx)
{
    if (cfg.symBytes == 0) {
        std::cerr << "[PTO-DOMAIN] URMA requires symBytes > 0\n";
        return false;
    }
    aclError aRet = aclrtMalloc(&ctx.urmaDevBuf, cfg.symBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || ctx.urmaDevBuf == nullptr) {
        std::cerr << "[PTO-DOMAIN] URMA aclrtMalloc failed: " << static_cast<int>(aRet) << "\n";
        return false;
    }
    return true;
}
#endif

inline bool SetupAddressing(const CommConfig& cfg, CommContext& ctx)
{
    if (NeedsWindowFamily(cfg.backends)) {
        if (!AllocWindowByTiling(ctx)) {
            std::cerr << "[PTO-DOMAIN] setupAddressing(Window) failed\n";
            return false;
        }
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (NeedsUrmaFamily(cfg.backends)) {
        if (!SetupUrmaDevBuf(cfg, ctx)) {
            std::cerr << "[PTO-DOMAIN] setupAddressing(URMA) failed\n";
            return false;
        }
    }
#else
    if (NeedsUrmaFamily(cfg.backends)) {
        std::cerr << "[PTO-DOMAIN] URMA not supported on this arch build\n";
        return false;
    }
#endif
    return true;
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP

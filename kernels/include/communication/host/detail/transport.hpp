/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/transport.hpp is host-only"
#endif

#include <iostream>
#include <memory>

#include "securec.h"

#include "acl/acl.h"

#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

inline bool SetupSdmaTransport(CommContext& ctx)
{
    ctx.sdmaMgr = std::make_unique<sdma::SdmaWorkspaceManager>();
    if (!ctx.sdmaMgr->Init()) {
        std::cerr << "[PTO-DOMAIN] SdmaWorkspaceManager::Init failed\n";
        ctx.sdmaMgr.reset();
        return false;
    }
    // C-style cast: reinterpret_cast to __gm__* is ill-formed under -xcce.
    ctx.sdmaWs = (__gm__ uint8_t*)(ctx.sdmaMgr->GetWorkspaceAddr());
    if (ctx.sdmaWs == nullptr) {
        std::cerr << "[PTO-DOMAIN] SDMA workspace addr is null\n";
        ctx.sdmaMgr.reset();
        return false;
    }
    return true;
}

#ifdef PTO_DOMAIN_URMA_HOST
inline bool BackfillUrmaWindows(const CommConfig& cfg, CommContext& ctx)
{
    memset_s(&ctx.urmaHostCtx, sizeof(ctx.urmaHostCtx), 0, sizeof(ctx.urmaHostCtx));
    ctx.urmaHostCtx.rankId = static_cast<uint32_t>(cfg.rankId);
    ctx.urmaHostCtx.rankNum = static_cast<uint32_t>(cfg.rankNum);
    ctx.urmaHostCtx.winSize = cfg.symBytes;
    for (uint32_t peer = 0; peer < ctx.urmaHostCtx.rankNum && peer < kMaxRankNum; ++peer) {
        ctx.urmaHostCtx.windowsIn[peer] = ctx.urmaMgr->PeerBaseAddr(peer);
    }
    void* devCtx = nullptr;
    aclError aRet = aclrtMalloc(&devCtx, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || devCtx == nullptr) {
        std::cerr << "[PTO-DOMAIN] URMA aclrtMalloc CommDeviceContext failed\n";
        return false;
    }
    aRet = aclrtMemcpy(
        devCtx, sizeof(CommDeviceContext), &ctx.urmaHostCtx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        aclrtFree(devCtx);
        std::cerr << "[PTO-DOMAIN] URMA H2D CommDeviceContext failed\n";
        return false;
    }
    ctx.urmaDevCtx = reinterpret_cast<CommDeviceContext*>(devCtx);
    ctx.ownsUrmaDevCtx = true;
    return true;
}

inline bool SetupUrmaTransport(const CommConfig& cfg, CommContext& ctx)
{
    ctx.urmaMgr = std::make_unique<urma::UrmaWorkspaceManager>();
    if (!ctx.urmaMgr->Init(
            ctx.comm, static_cast<uint32_t>(cfg.rankId), static_cast<uint32_t>(cfg.rankNum), ctx.urmaDevBuf,
            cfg.symBytes)) {
        std::cerr << "[PTO-DOMAIN] UrmaWorkspaceManager::Init failed\n";
        ctx.urmaMgr.reset();
        return false;
    }
    // C-style cast: reinterpret_cast to __gm__* is ill-formed under -xcce.
    ctx.urmaWs = (__gm__ uint8_t*)(ctx.urmaMgr->GetWorkspaceAddr());
    if (ctx.urmaWs == nullptr) {
        std::cerr << "[PTO-DOMAIN] URMA workspace addr is null\n";
        ctx.urmaMgr.reset();
        return false;
    }
    return BackfillUrmaWindows(cfg, ctx);
}
#endif

inline bool SetupTransport(const CommConfig& cfg, CommContext& ctx)
{
    if (HasBackend(cfg.backends, CommBackend::SDMA)) {
        if (!SetupSdmaTransport(ctx)) {
            return false;
        }
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (HasBackend(cfg.backends, CommBackend::URMA)) {
        if (!SetupUrmaTransport(cfg, ctx)) {
            return false;
        }
    }
#else
    if (HasBackend(cfg.backends, CommBackend::URMA)) {
        std::cerr << "[PTO-DOMAIN] URMA transport not available in this build\n";
        return false;
    }
#endif
    return true;
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_HPP
#define PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_HPP

// ============================================================================
// Public host-side API for the PTO communication domain.
// Include this header (not detail/) from host translation units.
// ============================================================================

#if defined(__CCE_KT_TEST__)
#error "communication/host/comm_context.hpp is host-only"
#endif

#include <iostream>

#include "acl/acl.h"
#include "hccl/hccl.h"

// Host g++ does not understand CCE address-space attributes. Stub them before
// pulling AsyncSession (which embeds __ubuf__/__gm__ pointers for device use).
#if !defined(__CCE__) && !defined(__DAV__)
#ifndef __gm__
#define __gm__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef __CPU_SIM
#define __CPU_SIM
#define PTO_DOMAIN_DEFINED_CPU_SIM_FOR_HOST 1
#endif
#endif

#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/comm_types.hpp"
#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"
#include "communication/host/detail/addressing.hpp"
#include "communication/host/detail/bootstrap.hpp"
#include "communication/host/detail/mpi_helpers.hpp"
#include "communication/host/detail/transport.hpp"

#if defined(PTO_DOMAIN_DEFINED_CPU_SIM_FOR_HOST)
#undef __CPU_SIM
#undef PTO_DOMAIN_DEFINED_CPU_SIM_FOR_HOST
#endif

namespace pto {
namespace comm {
namespace domain {

inline void CommContext::Reset()
{
#ifdef PTO_DOMAIN_URMA_HOST
    if (urmaMgr) {
        urmaMgr->Finalize();
        urmaMgr.reset();
    }
    urmaWs = nullptr;
    if (ownsUrmaDevCtx && urmaDevCtx != nullptr) {
        aclrtFree(urmaDevCtx);
        urmaDevCtx = nullptr;
        ownsUrmaDevCtx = false;
    }
    if (urmaDevBuf != nullptr) {
        aclrtFree(urmaDevBuf);
        urmaDevBuf = nullptr;
    }
#endif
    if (sdmaMgr) {
        sdmaMgr->Finalize();
        sdmaMgr.reset();
    }
    sdmaWs = nullptr;
    if (ownsWinDevCtx && winDevCtx != nullptr) {
        aclrtFree(winDevCtx);
    }
    winDevCtx = nullptr;
    winBase = nullptr;
    ownsWinDevCtx = false;
    if (ownsComm && comm != nullptr) {
        HcclCommDestroy(comm);
    }
    comm = nullptr;
    ownsComm = false;
    if (stream != nullptr) {
        rtStreamDestroy(stream);
        stream = nullptr;
    }
    winHostCtx = {};
#ifdef PTO_DOMAIN_URMA_HOST
    urmaHostCtx = {};
#endif
    backends = 0;
    deviceId = -1;
}

// ============================================================================
// DestroyComm: release all resources held by a CommContext.
// Equivalent to ctx.Reset(); safe on default-constructed or already-destroyed
// contexts. After return, ctx may be reused with BuildComm.
// ============================================================================
inline void DestroyComm(CommContext& ctx) { ctx.Reset(); }

// ============================================================================
// BuildComm: build a communication domain (bootstrap + addressing + transport).
// Single host entry for HCCL, Window/URMA addressing, and optional SDMA/URMA
// workspaces according to cfg.backends. Does not build AsyncSession — call
// BuildAsyncSession separately after success.
// On failure, partially allocated resources in out are released; returns false.
// Precondition: cfg.rankNum in [1, kMaxRankNum]; cfg.backends != 0.
// ============================================================================
inline bool BuildComm(const CommConfig& cfg, CommContext& out)
{
    DestroyComm(out);
    if (cfg.backends == 0) {
        std::cerr << "[PTO-DOMAIN] BuildComm: backends is empty\n";
        return false;
    }
    if (cfg.rankNum <= 0 || static_cast<uint32_t>(cfg.rankNum) > kMaxRankNum) {
        std::cerr << "[PTO-DOMAIN] BuildComm: rankNum(" << cfg.rankNum << ") out of range [1, " << kMaxRankNum << "]\n";
        return false;
    }
    out.backends = cfg.backends;
    if (!detail::BootstrapComm(cfg, out)) {
        std::cerr << "[PTO-DOMAIN] BuildComm: bootstrapComm failed\n";
        DestroyComm(out);
        return false;
    }
    if (!detail::SetupAddressing(cfg, out)) {
        std::cerr << "[PTO-DOMAIN] BuildComm: setupAddressing failed\n";
        DestroyComm(out);
        return false;
    }
    if (!detail::SetupTransport(cfg, out)) {
        std::cerr << "[PTO-DOMAIN] BuildComm: setupTransport failed\n";
        DestroyComm(out);
        return false;
    }
    return true;
}

// ============================================================================
// GetDeviceContext: device-side CommDeviceContext for an addressing family.
// Returns nullptr if the family was not enabled / not available.
// Pass the pointer to the kernel; device code uses RemotePtr / RemoteTensor.
// ============================================================================
inline __gm__ CommDeviceContext* GetDeviceContext(const CommContext& ctx, AddrFamily family)
{
    // C-style cast: reinterpret_cast to __gm__* is ill-formed under -xcce.
    if (family == AddrFamily::Window) {
        return (__gm__ CommDeviceContext*)(ctx.winDevCtx);
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (family == AddrFamily::Urma) {
        return (__gm__ CommDeviceContext*)(ctx.urmaDevCtx);
    }
#endif
    return nullptr;
}

// ============================================================================
// GetSymmetricBase: local symmetric-memory base for an addressing family.
// Returns nullptr if the family is not available.
// First kSignalPrefixBytes are reserved for signals; app data starts after that.
// ============================================================================
inline void* GetSymmetricBase(const CommContext& ctx, AddrFamily family)
{
    if (family == AddrFamily::Window) {
        return ctx.winBase;
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (family == AddrFamily::Urma) {
        return ctx.urmaDevBuf;
    }
#endif
    return nullptr;
}

// ============================================================================
// GetEngineWorkspace: DMA engine workspace for async transport.
// Returns nullptr if the engine was not enabled.
// Usually consumed via BuildAsyncSession; exposed for advanced callers.
// ============================================================================
inline __gm__ uint8_t* GetEngineWorkspace(const CommContext& ctx, DmaEngine engine)
{
    if (engine == DmaEngine::SDMA) {
        return ctx.sdmaWs;
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (engine == DmaEngine::URMA) {
        return ctx.urmaWs;
    }
#endif
    return nullptr;
}

// ============================================================================
// HostBarrier: synchronize all ranks on the host control / data plane.
// If bootstrap is MPI, runs MPI_Barrier first; then HcclBarrier + stream sync
// when HcclComm and stream are valid.
// ============================================================================
inline void HostBarrier(const CommContext& ctx)
{
    if (ctx.bootstrap == Bootstrap::Mpi) {
        detail::DomainMpiBarrier();
    }
    if (ctx.comm != nullptr && ctx.stream != nullptr) {
        (void)HcclBarrier(ctx.comm, ctx.stream);
        (void)aclrtSynchronizeStream(ctx.stream);
    }
}

constexpr uint32_t kDefaultSdmaQueueNum = 1;

// ============================================================================
// BuildAsyncSessionSdma / BuildAsyncSessionUrma: engine-specific session prefills.
// ============================================================================
inline bool BuildAsyncSessionSdma(const CommContext& ctx, AsyncSession& out)
{
    if (ctx.sdmaWs == nullptr) {
        std::cerr << "[PTO-DOMAIN] BuildAsyncSession(SDMA): workspace missing\n";
        return false;
    }
    out = AsyncSession{};
    out.engine = DmaEngine::SDMA;
    out.valid = true;
    out.contextGm = ctx.sdmaWs;
    out.tmpBufAddr = nullptr;
    out.tmpBufSize = 0;
    out.syncId = 0;
    out.channelGroupIdx = sdma::kAutoChannelGroupIdx;
    out.blockBytes = sdma::kDefaultSdmaBlockBytes;
    out.commBlockOffset = 0;
    out.queueNum = kDefaultSdmaQueueNum;
    return true;
}

#ifdef PTO_DOMAIN_URMA_HOST
inline bool BuildAsyncSessionUrma(const CommContext& ctx, AsyncSession& out)
{
    if (ctx.urmaWs == nullptr) {
        std::cerr << "[PTO-DOMAIN] BuildAsyncSession(URMA): workspace missing\n";
        return false;
    }
    out = AsyncSession{};
    out.engine = DmaEngine::URMA;
    out.valid = true;
    out.contextGm = ctx.urmaWs;
    out.destRankId = 0; // unused by new API; kept for compatibility.
    out.qpIdx = 0;
    return true;
}
#endif

// ============================================================================
// BuildAsyncSession: prefill an AsyncSession for a DMA engine (after BuildComm).
// Fills engine-agnostic / host-known fields (workspace, defaults).
// URMA sessions typically need no device Modify; peer is passed at the call.
// Returns false if the engine is not enabled or workspace is missing.
// ============================================================================
inline bool BuildAsyncSession(const CommContext& ctx, DmaEngine engine, AsyncSession& out)
{
    if (engine == DmaEngine::SDMA) {
        if (!HasBackend(ctx.backends, CommBackend::SDMA)) {
            std::cerr << "[PTO-DOMAIN] BuildAsyncSession: SDMA not enabled\n";
            return false;
        }
        return BuildAsyncSessionSdma(ctx, out);
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (engine == DmaEngine::URMA) {
        if (!HasBackend(ctx.backends, CommBackend::URMA)) {
            std::cerr << "[PTO-DOMAIN] BuildAsyncSession: URMA not enabled\n";
            return false;
        }
        return BuildAsyncSessionUrma(ctx, out);
    }
#endif
    std::cerr << "[PTO-DOMAIN] BuildAsyncSession: unsupported engine\n";
    return false;
}

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_HPP

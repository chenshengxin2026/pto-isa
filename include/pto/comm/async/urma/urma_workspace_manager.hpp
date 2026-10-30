/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_channel_helper.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. Allocate the notify pool and register symmetric + notify memory
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    static constexpr const char* kDefaultSharedTagPrefix = "pto_urma_jetty";

    static uint32_t QueryDeviceAivCount()
    {
        int32_t deviceId = 0;
        if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
            return 0U;
        }
        int64_t queried = 0;
        const aclError err =
            aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &queried);
        if (err != ACL_SUCCESS || queried <= 0 ||
            queried > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return 0U;
        }
        return static_cast<uint32_t>(queried);
    }

    bool Init(
        HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize,
        UrmaLayout layout = UrmaLayout::PER_PEER, uint32_t aivCount = kUrmaAutoAivCount, uint32_t jettiesPerCore = 1,
        uint32_t sqDepth = 0, const char* sharedTagPrefix = kDefaultSharedTagPrefix)
    {
        if (initialized_) {
            return true;
        }
        if (rankCount < 2U || rankId >= rankCount) {
            PTO_HOST_LOG_ERROR(
                "URMA", "rankId=" << rankId << " rankCount=" << rankCount
                                  << " is not a usable group; need rankCount >= 2 and rankId < rankCount");
            return false;
        }
        if (symmetricAddr == nullptr || symmetricSize == 0U) {
            PTO_HOST_LOG_ERROR(
                "URMA", "symmetric memory must be a non-empty device region, addr=" << symmetricAddr
                                                                                    << " size=" << symmetricSize);
            return false;
        }

        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        layout_ = layout;
        sqDepth_ = sqDepth;
        sharedTagPrefix_ = (sharedTagPrefix != nullptr) ? sharedTagPrefix : kDefaultSharedTagPrefix;
        railPeerLinks_.clear();
        railCount_ = 1U;
        netLayer_ = 0U;

        if (layout == UrmaLayout::SHARED_POOL && !BuildRailPlan()) {
            return false;
        }

        uint32_t resolvedAivCount = 0;
        uint32_t resolvedJettiesPerCore = 1;
        uint32_t resolvedRowsPerPeer = 1;
        if (!ResolveLayoutConfig(
                layout, aivCount, jettiesPerCore, resolvedAivCount, resolvedJettiesPerCore, resolvedRowsPerPeer)) {
            return false;
        }
        aivCount_ = resolvedAivCount;
        jettiesPerCore_ = resolvedJettiesPerCore;
        rowsPerPeer_ = resolvedRowsPerPeer;

        if (!AllocateNotifyPool()) {
            return false;
        }
        if (!RegisterMemory()) {
            return false;
        }
        if (!BuildChannels()) {
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    // The owning communication context must stop kernels, drain QPs, and
    // destroy HCCL communication resources before calling Finalize().
    void Finalize()
    {
        FreeDeviceAddr(urmaInfoDevice_);
        FreeDeviceAddr(eidDevice_);
        FreeDeviceAddr(notifyPoolDevice_);
        channelHandles_.clear();
        peerBaseAddrs_.clear();
        sharedPeerOrder_.clear();
        railPeerLinks_.clear();
        railCount_ = 1U;
        netLayer_ = 0U;
        channelsPerJetty_ = 0;
        rowsPerPeer_ = 1;
        initialized_ = false;
    }

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    uint64_t PeerBaseAddr(uint32_t peer) const
    {
        if (peer >= peerBaseAddrs_.size()) {
            return 0;
        }
        return peerBaseAddrs_[peer];
    }

    UrmaLayout Layout() const { return layout_; }
    bool IsPerPeer() const { return layout_ == UrmaLayout::PER_PEER; }

    uint32_t JettyCount() const { return IsPerPeer() ? rankCount_ * rowsPerPeer_ : aivCount_ * jettiesPerCore_; }
    uint32_t JettiesPerCore() const { return jettiesPerCore_; }
    uint32_t RowsPerPeer() const { return IsPerPeer() ? rowsPerPeer_ : 1U; }

    uint64_t GetWorkspaceSize() const
    {
        if (!initialized_) {
            return 0;
        }
        return ComputeWorkspaceBytes();
    }

private:
    bool ResolveLayoutConfig(
        UrmaLayout layout, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t& resolvedAivCount,
        uint32_t& resolvedJettiesPerCore, uint32_t& resolvedRowsPerPeer) const
    {
        if (layout == UrmaLayout::PER_PEER) {
            resolvedAivCount = aivCount;
            if (resolvedAivCount == kUrmaAutoAivCount) {
                resolvedAivCount = QueryDeviceAivCount();
                if (resolvedAivCount == 0U) {
                    PTO_HOST_LOG_ERROR("URMA", "cannot query the device AIV count; pass aivCount explicitly");
                    return false;
                }
            } else if (resolvedAivCount == 0U) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "aivCount=0 is not a PerPeer size; omit it for the device AIV count or pass N >= 1");
                return false;
            }
            if (jettiesPerCore > kUrmaMaxJettiesPerCore) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "jettiesPerCore=" << jettiesPerCore << " must be in [1, " << kUrmaMaxJettiesPerCore
                                              << "] or " << kUrmaAutoJettiesPerCore
                                              << " for one queue per AIV per peer");
                return false;
            }
            resolvedJettiesPerCore = (jettiesPerCore == kUrmaAutoJettiesPerCore) ? 1U : jettiesPerCore;
            if (resolvedAivCount > std::numeric_limits<uint32_t>::max() / resolvedJettiesPerCore) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "aivCount=" << resolvedAivCount << " * jettiesPerCore=" << resolvedJettiesPerCore
                                        << " overflows the per-peer queue count");
                return false;
            }
            resolvedRowsPerPeer = resolvedAivCount * resolvedJettiesPerCore;
            if (resolvedRowsPerPeer == 0U || resolvedRowsPerPeer > kUrmaMaxRowsPerPeer) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "PerPeer queues per peer=" << resolvedRowsPerPeer << " must be in [1, "
                                                       << kUrmaMaxRowsPerPeer << "]");
                return false;
            }
            return true;
        }
        if (layout != UrmaLayout::SHARED_POOL) {
            PTO_HOST_LOG_ERROR("URMA", "unknown layout " << static_cast<uint32_t>(layout));
            return false;
        }

        resolvedAivCount = aivCount;
        if (resolvedAivCount == kUrmaAutoAivCount) {
            resolvedAivCount = QueryDeviceAivCount();
            if (resolvedAivCount == 0U) {
                PTO_HOST_LOG_ERROR("URMA", "cannot query the device AIV count; pass aivCount explicitly");
                return false;
            }
        } else if (resolvedAivCount == 0U) {
            PTO_HOST_LOG_ERROR(
                "URMA", "aivCount=0 is not a SharedPool size; omit it for the device AIV "
                        "count, pass N >= 1, or use UrmaLayout::PER_PEER");
            return false;
        }
        if (jettiesPerCore > kUrmaMaxJettiesPerCore) {
            PTO_HOST_LOG_ERROR(
                "URMA", "jettiesPerCore=" << jettiesPerCore << " must be in [1, " << kUrmaMaxJettiesPerCore << "] or "
                                          << kUrmaAutoJettiesPerCore << " to take the rail count");
            return false;
        }
        if (jettiesPerCore == kUrmaAutoJettiesPerCore && railCount_ > kUrmaMaxJettiesPerCore) {
            PTO_HOST_LOG_ERROR(
                "URMA", "rail plan found "
                            << railCount_ << " ports but a core may own at most " << kUrmaMaxJettiesPerCore
                            << " jetties; pass jettiesPerCore explicitly or raise kUrmaMaxJettiesPerCore (it sizes "
                               "AsyncEvent's per-jetty targets)");
            return false;
        }
        resolvedJettiesPerCore = (jettiesPerCore == kUrmaAutoJettiesPerCore) ? railCount_ : jettiesPerCore;
        if (resolvedAivCount > std::numeric_limits<uint32_t>::max() / resolvedJettiesPerCore) {
            PTO_HOST_LOG_ERROR(
                "URMA", "aivCount=" << resolvedAivCount << " * jettiesPerCore=" << resolvedJettiesPerCore
                                    << " overflows the jetty count");
            return false;
        }
        return true;
    }

    uint64_t CtxRowCount() const { return JettyCount(); }
    uint64_t TargetRowCount() const
    {
        return IsPerPeer() ? JettyCount() : static_cast<uint64_t>(JettyCount()) * rankCount_;
    }

    uint64_t ComputeWorkspaceBytes() const
    {
        return static_cast<uint64_t>(
            sizeof(UrmaInfo) + CtxRowCount() * (2U * sizeof(UrmaWQCtx) + 2U * sizeof(UrmaCqCtx)) +
            TargetRowCount() * sizeof(UrmaMemInfo));
    }

    bool AllocateNotifyPool()
    {
        const uint64_t regionCount = TargetRowCount();
        if (regionCount == 0U ||
            regionCount > std::numeric_limits<uint64_t>::max() / sizeof(UrmaNotifyResourceRegion)) {
            PTO_HOST_LOG_ERROR("URMA", "invalid notify pool region count=" << regionCount);
            return false;
        }
        notifyPoolSize_ = regionCount * sizeof(UrmaNotifyResourceRegion);
        aclError err = aclrtMalloc(&notifyPoolDevice_, notifyPoolSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "aclrtMalloc(notifyPool) failed: " << err << " size=" << notifyPoolSize_);
            return false;
        }
        err = aclrtMemset(notifyPoolDevice_, notifyPoolSize_, 0, notifyPoolSize_);
        if (err != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "aclrtMemset(notifyPool) failed: " << err);
            return false;
        }
        return true;
    }

    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclCommMemReg failed: " << static_cast<int>(ret));
            return false;
        }

        CommMem notifyMem{};
        notifyMem.type = COMM_MEM_TYPE_DEVICE;
        notifyMem.addr = notifyPoolDevice_;
        notifyMem.size = notifyPoolSize_;
        ret = HcclCommMemReg(comm_, kUrmaNotifyMemTag, &notifyMem, &notifyMemHandle_);
        if (ret != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclCommMemReg(notifyPool) failed: " << static_cast<int>(ret));
            return false;
        }
        memHandles_[0] = memHandle_;
        memHandles_[1] = notifyMemHandle_;
        return true;
    }

    // One usable UB link to a peer, resolved from the rank graph.
    struct PeerLink {
        uint32_t peer{0};
        CommProtocol proto{};
        EndpointDesc localEp{};
        EndpointDesc remoteEp{};
    };

    bool FindPeerLinkFrom(uint32_t srcRank, uint32_t dstRank, PeerLink& out) const
    {
        uint32_t linkNum = 0;
        CommLink* linkList = nullptr;
        HcclResult rc = HcclRankGraphGetLinks(comm_, 0, srcRank, dstRank, &linkList, &linkNum);
        if (rc != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR(
                "URMA",
                "HcclRankGraphGetLinks src=" << srcRank << " dst=" << dstRank << " ret=" << static_cast<int>(rc));
            return false;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            CommProtocol proto = linkList[i].linkAttr.linkProtocol;
            if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
                continue;
            }
            out = PeerLink{dstRank, proto, linkList[i].srcEndpointDesc, linkList[i].dstEndpointDesc};
            return true;
        }
        PTO_HOST_LOG_ERROR("URMA", "rank=" << srcRank << " no UBC_TP/CTP link to peer=" << dstRank);
        return false;
    }

    bool FindPeerLink(uint32_t peer, PeerLink& out) const { return FindPeerLinkFrom(rankId_, peer, out); }

    static uint32_t CanonicalPeerOf(uint32_t rank, uint32_t rankCount)
    {
        for (uint32_t peer = 0; peer < rankCount; ++peer) {
            if (peer != rank) {
                return peer;
            }
        }
        return rank;
    }

    bool FindRankCanonicalLocal(uint32_t rank, EndpointDesc& out) const
    {
        const uint32_t peer = CanonicalPeerOf(rank, rankCount_);
        if (peer == rank) {
            return false;
        }
        PeerLink link{};
        if (!FindPeerLinkFrom(rank, peer, link)) {
            return false;
        }
        out = link.localEp;
        return true;
    }

    void FillChannelDesc(HcclChannelDesc& desc, const PeerLink& link)
    {
        HcclChannelDescInit(&desc, 1);
        desc.remoteRank = link.peer;
        desc.notifyNum = 0;
        desc.channelProtocol = link.proto;
        desc.localEndpoint = link.localEp;
        desc.remoteEndpoint = link.remoteEp;
        desc.memHandles = memHandles_;
        desc.memHandleNum = kUrmaRegisteredMemCount;
        ApplySqDepth(desc);
    }

    void ApplySqDepth(HcclChannelDesc& desc) const
    {
        if (sqDepth_ == 0) {
            return;
        }
        const uint32_t depth = sqDepth_;
        (void)memcpy_s(desc.raws, sizeof(desc.raws), &depth, sizeof(depth));
    }

    static bool IsSameLocalEndpoint(const EndpointDesc& a, const EndpointDesc& b)
    {
        return a.protocol == b.protocol && a.commAddr.type == b.commAddr.type &&
               std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0 &&
               a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
    }

#include "pto/comm/async/urma/urma_workspace_channels.hpp"
#include "pto/comm/async/urma/urma_workspace_extract.hpp"

    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    static void FreeDeviceAddr(void*& addr)
    {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr const char* kUrmaSymMemTag = "pto_urma_sym";
    static constexpr const char* kUrmaNotifyMemTag = "pto_urma_notify";
    static constexpr uint32_t kUrmaRegisteredMemCount = 2U;
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};
    HcclMemHandle notifyMemHandle_{nullptr};
    HcclMemHandle memHandles_[kUrmaRegisteredMemCount]{};

    UrmaLayout layout_{UrmaLayout::PER_PEER};
    uint32_t aivCount_{0};
    uint32_t jettiesPerCore_{1};
    uint32_t rowsPerPeer_{1};
    uint32_t sqDepth_{0};
    std::string sharedTagPrefix_{kDefaultSharedTagPrefix};
    uint32_t channelsPerJetty_{0};
    std::vector<uint32_t> sharedPeerOrder_;
    uint32_t railCount_{1};
    uint32_t netLayer_{0};

    std::vector<std::vector<PeerLink>> railPeerLinks_;

    std::vector<ChannelHandle> channelHandles_;

    void* urmaInfoDevice_{nullptr};
    void* eidDevice_{nullptr};
    void* notifyPoolDevice_{nullptr};
    uint64_t notifyPoolSize_{0};
    std::vector<uint64_t> peerBaseAddrs_;

    bool initialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP
#define PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP

#include <cstdint>

namespace pto {
namespace comm {
namespace domain {

// ============================================================================
// CommBackend: capability bitmask for BuildComm (combinable).
// ============================================================================
enum CommBackend : uint32_t {
    MTE = 1u << 0,  // Sync TPUT/TGET; Window family; no transport workspace
    SDMA = 1u << 1, // Async; Window family; SDMA workspace
    URMA = 1u << 2, // Async; URMA family; UrmaInfo workspace
    // CCU = 1u << 3, // reserved
};

// ============================================================================
// AddrFamily: addressing family for GetDeviceContext / GetSymmetricBase.
// ============================================================================
enum class AddrFamily : uint32_t {
    Window = 0, // MTE + SDMA shared
    Urma = 1,   // URMA-owned (windowsIn backfilled)
};

// ============================================================================
// Bootstrap: how HcclRootInfo / barrier is exchanged.
// ============================================================================
enum class Bootstrap { Mpi, External };

// ============================================================================
// CommConfig: host-facing build config.
// Hccl handles are opaque void* so this header stays free of HCCL includes
// (cast to HcclComm / HcclRootInfo* in host-only code).
// ============================================================================
struct CommConfig {
    int rankId = -1;
    int rankNum = 0;
    int deviceId = -1; // default: rankId % nDevices + first
    uint32_t backends = CommBackend::MTE;
    uint64_t symBytes = 0;
    Bootstrap bootstrap = Bootstrap::Mpi;

    void* existingComm = nullptr;   // HcclComm when External
    const void* rootInfo = nullptr; // const HcclRootInfo* when External
};

inline constexpr uint32_t kSignalPrefixBytes = 64u * sizeof(int32_t);

inline bool HasBackend(uint32_t backends, CommBackend bit) { return (backends & static_cast<uint32_t>(bit)) != 0u; }

inline bool NeedsWindowFamily(uint32_t backends)
{
    return HasBackend(backends, CommBackend::MTE) || HasBackend(backends, CommBackend::SDMA);
}

inline bool NeedsUrmaFamily(uint32_t backends) { return HasBackend(backends, CommBackend::URMA); }

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP

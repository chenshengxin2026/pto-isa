/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP
#define PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP

// ============================================================================
// Public device-side API for the PTO communication domain.
// Requires CCE/bisheng (AICORE, __gm__, __ubuf__, get_block_idx).
// Do NOT include from host-only translation units.
// ============================================================================

#include <cstdint>

#include "pto/comm/comm_types.hpp"
#include "communication/comm_device_context.hpp"
#include "pto/common/pto_tile.hpp"

namespace pto {
namespace comm {
namespace domain {

// ============================================================================
// RemotePtr: map a local symmetric pointer to the same offset on a peer rank.
// Precondition: windowsIn[r] == windowsOut[r] for all ranks; localPtr lies in
// this rank's windowsIn[rankId] region.
// ============================================================================
template <typename T>
AICORE inline __gm__ T* RemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int peer)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peer] + offset);
}

// ============================================================================
// RemoteTensor: GlobalTensor view over a peer's symmetric buffer.
// Equivalent to RemotePtr followed by constructing GlobalT with shape/stride.
// ============================================================================
template <typename GlobalT, typename... ShapeArgs>
AICORE inline GlobalT RemoteTensor(
    __gm__ CommDeviceContext* ctx, typename GlobalT::DType* localPtr, int peer, ShapeArgs&&... shapeArgs)
{
    auto* remote = RemotePtr(ctx, localPtr, peer);
    return GlobalT(remote, static_cast<ShapeArgs&&>(shapeArgs)...);
}

// ============================================================================
// MakeParallelGroup: ParallelGroup covering all ranks for collective instructions.
// Fills tensorsStorage[0 .. rankNum-1] with RemoteTensor views of localBuf on
// each peer, then returns ParallelGroup::Create(...).
// ============================================================================
template <typename GlobalT, typename ShapeT, typename StrideT>
AICORE inline ParallelGroup<GlobalT> MakeParallelGroup(
    __gm__ CommDeviceContext* ctx, GlobalT* tensorsStorage, typename GlobalT::DType* localBuf, int root,
    const ShapeT& shape, const StrideT& stride)
{
    const int nranks = static_cast<int>(ctx->rankNum);
    for (int peer = 0; peer < nranks; ++peer) {
        tensorsStorage[peer] = RemoteTensor<GlobalT>(ctx, localBuf, peer, shape, stride);
    }
    return ParallelGroup<GlobalT>::Create(tensorsStorage, nranks, root);
}

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP

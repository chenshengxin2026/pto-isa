/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_COMM_DEVICE_CONTEXT_HPP
#define PTO_COMM_DOMAIN_COMM_DEVICE_CONTEXT_HPP

#include <cstdint>

namespace pto {
namespace comm {
namespace domain {

inline constexpr uint32_t kMaxRankNum = 64;

// ============================================================================
// CommDeviceContext: host/device shared POD.
// Layout matches the leading fields of HCCL HcclCombinOpParamA5 / test
// CommDeviceContext (windowsIn/Out addressing).
// ============================================================================
struct CommDeviceContext {
    uint64_t workSpace{0};
    uint64_t workSpaceSize{0};
    uint32_t rankId{0};
    uint32_t rankNum{0};
    uint64_t winSize{0};
    uint64_t windowsIn[kMaxRankNum]{};
    uint64_t windowsOut[kMaxRankNum]{};
};

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_COMM_DEVICE_CONTEXT_HPP

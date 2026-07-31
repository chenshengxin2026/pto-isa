/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_RT_FWD_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_RT_FWD_HPP

// Minimal forward declarations for CANN runtime types used by domain headers.
// Avoids pulling in the full runtime header while keeping extern "C" signatures
// consistent across translation units.

#include <cstdint>

namespace pto {
namespace comm {
namespace domain {

using rtStream_t = void*;
using rtError_t = int32_t;

extern "C" rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

inline constexpr int32_t kRtStreamPriorityDefault = 0;

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_RT_FWD_HPP

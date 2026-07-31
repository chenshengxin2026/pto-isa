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

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace pto::mocker::vf {

inline constexpr uint64_t kMemBarPenaltyPlaceholder = 20;
inline constexpr uint64_t kUnknownInstructionFallbackCycles = 5;

inline bool IsOneOf(std::string_view name, std::initializer_list<std::string_view> candidates)
{
    for (std::string_view candidate : candidates) {
        if (name == candidate) {
            return true;
        }
    }
    return false;
}

inline uint64_t VecCycle(std::string_view name)
{
    if (IsOneOf(name, {"plt_b8", "plt_b16", "plt_b32", "pset_b8", "pset_b16", "pset_b32"})) {
        return 2;
    }
    if (IsOneOf(name, {"vlds", "vsts", "vdup", "vmov", "vsel"})) {
        return 4;
    }
    if (IsOneOf(
            name, {"vadd",  "vsub",   "vmul", "vand", "vor",  "vxor", "vshl",  "vshr",  "vshls", "vshrs", "vabs",
                   "vrelu", "vlrelu", "vnot", "vneg", "vmin", "vmax", "vmins", "vadds", "pand",  "por",   "pnot"})) {
        return 6;
    }
    if (IsOneOf(name, {"vdiv", "vexp", "vsqrt", "vln", "vcvt", "vmadd", "vmuls", "vmula", "vtrc"}) ||
        name.rfind("vcmp", 0) == 0) {
        return 10;
    }
    if (name == "pipe_barrier") {
        return kMemBarPenaltyPlaceholder;
    }
    return kUnknownInstructionFallbackCycles;
}

// Unsupported VfSim programs use the A2/A3-style repeat-times * per-op-cycle
// calculation. Keep the unknown-op value explicit so it is visible to callers.
inline uint64_t FallbackVecCycle(std::string_view name) { return VecCycle(name); }

} // namespace pto::mocker::vf

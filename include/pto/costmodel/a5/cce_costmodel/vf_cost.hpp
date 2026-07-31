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

#include "pto/costmodel/a5/VfSim/VfSimCostModel.h"
#include "vec_cycle_generated.hpp"
#include "vf_info.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pto::mocker::vf {

namespace detail {

inline void WalkNodes(const std::vector<VfNode>& nodes, uint64_t mul, uint64_t& total)
{
    for (const VfNode& n : nodes) {
        if (IsLoop(n)) {
            const VfLoop& lp = AsLoop(n);
            WalkNodes(lp.body, mul * lp.count, total);
        } else if (IsInst(n)) {
            total += VecCycle(AsInst(n).opName) * mul;
        } else { // MEMBAR
            total += kMemBarPenaltyPlaceholder * mul;
        }
    }
}

} // namespace detail

inline uint64_t PredictVfCycles(const VfInfo& vf) { return PredictVfCyclesWithVfSim(std::vector<VfInfo>{vf}); }

inline uint64_t PredictVfCycles(const std::vector<VfInfo>& vfs) { return PredictVfCyclesWithVfSim(vfs); }

inline uint64_t LoopProduct(const std::vector<VfNode>& nodes)
{
    uint64_t prod = 1;
    for (const VfNode& n : nodes)
        if (IsLoop(n)) {
            prod *= AsLoop(n).count;
            prod *= LoopProduct(AsLoop(n).body);
        }
    return prod;
}

inline std::vector<std::string> FlattenLeafInstrs(const std::vector<VfNode>& nodes)
{
    std::vector<std::string> out;
    for (const VfNode& n : nodes) {
        if (IsLoop(n)) {
            auto sub = FlattenLeafInstrs(AsLoop(n).body);
            out.insert(out.end(), sub.begin(), sub.end());
        } else if (IsInst(n)) {
            out.push_back(AsInst(n).opName);
        }
    }
    return out;
}

inline uint64_t LoopDepth(const std::vector<VfNode>& nodes)
{
    uint64_t best = 0;
    for (const VfNode& n : nodes)
        if (IsLoop(n))
            best = std::max(best, 1 + LoopDepth(AsLoop(n).body));
    return best;
}

} // namespace pto::mocker::vf

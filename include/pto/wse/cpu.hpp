/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_WSE_CPU_HPP
#define PTO_WSE_CPU_HPP
#include <pto/wse/calendar_types.hpp>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace pto::wse {
// Segment identity and launch generation are mock transport metadata, not a
// claim about the unresolved on-wire WSE epoch format.
class CpuReceiver {
public:
    CpuReceiver(uint32_t capacity, uint32_t expected, uint32_t epoch, uint32_t launch)
        : arena(capacity), expected(expected), epoch(epoch), launch(launch)
    {
        if (expected > capacity || epoch == 0 || epoch > 255 || launch == 0)
            throw std::invalid_argument("invalid receive context");
    }

    bool Commit(
        uint32_t source, uint32_t segment, uint32_t offset, const std::vector<uint8_t>& payload, uint32_t packetEpoch,
        uint32_t packetLaunch, bool remote = true)
    {
        if (packetEpoch != epoch || packetLaunch != launch)
            throw std::invalid_argument("stale packet identity");
        if (uint64_t(offset) + payload.size() > arena.size())
            throw std::out_of_range("packet exceeds receive arena");
        auto identity = std::make_pair(source, segment);
        auto prior = seen.find(identity);
        if (prior != seen.end()) {
            if (prior->second != std::make_pair(offset, payload))
                throw std::invalid_argument("conflicting duplicate");
            return false;
        }
        for (auto [begin, end] : ranges)
            if (offset < end && begin < offset + payload.size())
                throw std::invalid_argument("overlapping segments");
        if (remote && uint64_t(committed) + payload.size() > expected)
            throw std::invalid_argument("receive count exceeds expected amount");
        seen.emplace(identity, std::make_pair(offset, payload));
        ranges.emplace_back(offset, offset + payload.size());
        std::memcpy(arena.data() + offset, payload.data(), payload.size());
        if (remote)
            committed += payload.size();
        return true;
    }

    bool Complete() const { return committed == expected; }
    uint32_t Committed() const { return committed; }
    std::vector<uint8_t> arena;

private:
    uint32_t expected;
    uint32_t epoch;
    uint32_t launch;
    uint32_t committed = 0;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, std::vector<uint8_t>>> seen;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
};
} // namespace pto::wse
#endif

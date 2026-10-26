/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_WSE_CALENDAR_CONTRACT_HPP
#define PTO_WSE_CALENDAR_CONTRACT_HPP
#include <cstdint>

namespace pto::wse {
using bsp_group_id_t = uint32_t;
using calendar_slot_id_t = uint32_t;
using epoch_id_t = uint64_t;
using route_hint_t = uint32_t;
enum calendar_red_t { CAL_RED_NONE, CAL_RED_SUM, CAL_RED_MAX, CAL_RED_MIN, CAL_RED_PROD };

enum class Error : uint32_t { OK, Binding, Epoch, Geometry, Capacity, Duplicate, Timeout, ByteCount };

struct ReceiveContract {
    uint32_t capacity;
    uint32_t expected;
    epoch_id_t epoch;
    uint32_t committed = 0;
    uint32_t sources = 0;

    Error check(
        epoch_id_t incomingEpoch, uint32_t source, uint32_t offset, uint32_t rows, uint32_t bytes,
        uint32_t stride) const
    {
        if (incomingEpoch != epoch)
            return Error::Epoch;
        if (source >= 24 || (sources & (1u << source)))
            return Error::Duplicate;
        if (!rows || !bytes || (rows > 1 && stride < bytes))
            return Error::Geometry;
        if (uint64_t(offset) + uint64_t(rows - 1) * stride + bytes > capacity)
            return Error::Capacity;
        if (uint64_t(committed) + uint64_t(rows) * bytes > expected)
            return Error::ByteCount;
        return Error::OK;
    }
    void commit(uint32_t source, uint32_t bytes)
    {
        sources |= 1u << source;
        committed += bytes;
    }
};

template <uint32_t Rows, uint32_t Cols, uint32_t Axis, uint32_t... Members>
struct AllGatherPlan {
    static_assert(Axis < 2 && sizeof...(Members) > 0 && sizeof...(Members) <= 24);
    static_assert(Rows > 0 && Cols > 0 && Cols % 16 == 0);
    static_assert(uint64_t(Rows) * Cols * 2 * sizeof...(Members) <= 128 * 1024);
    static constexpr uint32_t kMembers = sizeof...(Members);
    static constexpr uint32_t kMaxTransfers = 1;
    static constexpr uint32_t ids[kMembers] = {Members...};
    static constexpr uint32_t kRows = Rows;
    static constexpr uint32_t kRowBytes = Cols * 2;
    static constexpr uint32_t kSendBytes = Rows * kRowBytes;
    static constexpr uint32_t kRecvBytes = kMembers * kSendBytes;
    static constexpr uint32_t kDstStride = Axis == 1 ? kMembers * kRowBytes : kRowBytes;
    static constexpr uint32_t offset(uint32_t rank) { return rank * (Axis == 1 ? kRowBytes : kSendBytes); }
    static constexpr uint32_t row(uint32_t blockId)
    {
        for (uint32_t i = 0; i < kMembers; ++i)
            if (ids[i] == blockId)
                return i;
        return kMembers;
    }
};
} // namespace pto::wse
#endif

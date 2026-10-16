/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_WSE_CALENDAR_TYPES_HPP
#define PTO_WSE_CALENDAR_TYPES_HPP
#include <cstdint>
#include <cstddef>

namespace pto::wse {
enum calendar_op_t : uint32_t { CAL_ALLGATHER = 1 };
enum calendar_red_t : uint32_t { CAL_RED_NONE = 0 };

struct alignas(16) CalendarRouteEntry {
    uint8_t routeBits[10];
    uint8_t landCount;
    uint8_t flags;
    uint32_t expectedRxBytes;
};

// ABI 3 dedicates the first 96 bits to the 48-node whitepaper topology.
struct alignas(16) CalendarRouteEntry48 {
    uint8_t routeBits[12];
    uint32_t expectedRxBytes;
};
static_assert(sizeof(CalendarRouteEntry) == 16 && sizeof(CalendarRouteEntry48) == 16);
static_assert(offsetof(CalendarRouteEntry, expectedRxBytes) == 12);
static_assert(offsetof(CalendarRouteEntry48, expectedRxBytes) == 12);

template <calendar_op_t Op>
struct InstalledSlotRef {
    uint32_t bindingId;
};
struct CoreBinding {
    uint32_t logicalId;
    uint32_t physicalId;
};
struct CalendarState {
    uint32_t epochCtr[8] = {};
};
struct RecordEvent {
    uint32_t error;
    uint32_t committedBytes;
    uint32_t epoch;
};
// Scalar GM stores require separate cache lines for independent A3 workers.
constexpr uint32_t CalendarStatusWords = 16;

enum CalendarError : uint32_t {
    CAL_OK = 0,
    CAL_BAD_BINDING = 1,
    CAL_BAD_GEOMETRY = 2,
    CAL_BAD_ROUTE = 3,
    CAL_BAD_MEMBER = 4,
    CAL_INCOMPLETE = 5,
    CAL_BAD_EPOCH = 6,
    CAL_DUPLICATE = 7
};

struct ReceiveContext {
    uint32_t capacity = 0;
    uint32_t expected = 0;
    uint32_t committed = 0;
    uint32_t seen = 0;
    uint32_t epoch = 0;
    uint32_t error = CAL_OK;
    bool armed = false;
};

constexpr uint32_t PageBytes(uint32_t bytes) { return (bytes + 2047) / 2048 * 2048; }
} // namespace pto::wse
#endif

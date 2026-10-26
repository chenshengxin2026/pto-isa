/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_WSE_CALENDAR_HPP
#define PTO_WSE_CALENDAR_HPP
#include "cce_mock_a3.hpp"

#if defined(__CCE_AICORE__)
namespace pto::wse {
template <class Plan>
struct CalendarTables {
    __gm__ const route_hint_t (*routes)[Plan::kMaxTransfers];
    __gm__ const calendar_slot_id_t (*slots)[Plan::kMaxTransfers];
    bsp_group_id_t sync_group;
    CceMockA3& cce;
};

template <class Plan, class Send, class Recv, class... Deps>
AICORE RecordEvent TCalendar(
    const CalendarTables<Plan>& tables, uint32_t blockId, epoch_id_t epoch, uint32_t expVal, const Send& send,
    Recv& recv, Deps... deps)
{
    static_assert(Send::Loc == TileType::Mat && Recv::Loc == TileType::Mat);
    static_assert(Send::SFractal == SLayout::NoneBox && Recv::SFractal == SLayout::NoneBox);
    static_assert(sizeof(typename Send::DType) == 2);
    (TSYNC(deps), ...);
    auto& cce = tables.cce;
    uint32_t row = Plan::row(blockId);
    if (row >= Plan::kMembers || expVal != (Plan::kMembers - 1) * Plan::kSendBytes) {
        cce.fail(Error::Binding);
        return {};
    }
    auto src = reinterpret_cast<__cbuf__ const uint8_t*>(send.data());
    auto dst = reinterpret_cast<__cbuf__ uint8_t*>(recv.data());
    uint64_t srcOffset = reinterpret_cast<uint64_t>(src), dstOffset = reinterpret_cast<uint64_t>(dst);
    if (srcOffset < dstOffset + Plan::kRecvBytes && dstOffset < srcOffset + Plan::kSendBytes) {
        cce.fail(Error::Capacity);
        return {};
    }
    pipe_barrier(PIPE_ALL);
    cce.__sync_bsp(tables.sync_group, epoch);
    cce.wait_noc_to_l1(dst, Plan::kRecvBytes, expVal, epoch);
    cce.copy_l1_to_noc_l1_align_b8(
        dst + Plan::offset(row), src, Plan::kRows, Plan::kRowBytes, Plan::kRows == 1 ? 0 : Plan::kRowBytes,
        Plan::kRows == 1 ? 0 : Plan::kDstStride, tables.routes[row][0], tables.slots[row][0], epoch, CAL_RED_NONE, f16);
    cce.finish_receive();
    pipe_barrier(PIPE_ALL);
    return {};
}

template <uint32_t Axis, uint32_t... Members, class Send, class Recv>
AICORE void RunCalendar(uint64_t context, const Send& send, Recv& recv, uint32_t expVal)
{
    using Plan = AllGatherPlan<Send::Rows, Send::Cols, Axis, Members...>;
    static_assert(Recv::Rows == Send::Rows * (Axis == 0 ? Plan::kMembers : 1));
    static_assert(Recv::Cols == Send::Cols * (Axis == 1 ? Plan::kMembers : 1));
    auto ctx = reinterpret_cast<__gm__ uint64_t*>(context);
    uint32_t blockId = ctx[0];
    uint32_t row = Plan::row(blockId);
    CceMockA3 cce(ctx, static_cast<uint32_t>(ctx[1]));
    if (row >= Plan::kMembers || ctx[9] != Plan::kMembers || ctx[1] != row) {
        cce.fail(Error::Binding);
    } else {
        CalendarTables<Plan> tables{
            reinterpret_cast<__gm__ const route_hint_t(*)[1]>(ctx[4]),
            reinterpret_cast<__gm__ const calendar_slot_id_t(*)[1]>(ctx[5]), static_cast<uint32_t>(ctx[2]), cce};
        TCalendar<Plan>(tables, blockId, ctx[3], expVal, send, recv);
    }
    cce.finish();
}
} // namespace pto::wse
#endif
#endif

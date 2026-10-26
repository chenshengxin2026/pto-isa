/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_WSE_CCE_MOCK_A3_HPP
#define PTO_WSE_CCE_MOCK_A3_HPP
#include <pto/pto-inst.hpp>
#include "calendar_contract.hpp"

#if defined(__CCE_AICORE__)
namespace pto::wse {
// Each participant owns one 128-byte context, one 64-byte status and one signal line.
// Context pointer fields refer to launch-owned device allocations.
class CceMockA3 {
public:
    AICORE CceMockA3(__gm__ uint64_t* context, uint32_t row) : ctx(context), rank(row) {}

    AICORE void fail(Error error)
    {
        if (fault == Error::OK)
            fault = error;
    }
    AICORE bool ok() const { return fault == Error::OK; }

    AICORE void __sync_bsp(bsp_group_id_t group, epoch_id_t epoch)
    {
        if (group != ctx[2] || epoch != ctx[3] || rank >= ctx[9]) {
            fail(Error::Binding);
            return;
        }
        publish(0, epoch);
        for (uint32_t peer = 0; peer < ctx[9] && ok(); ++peer)
            await(peer, 0, epoch);
    }

    // Registration does not wait for payload: SEND must remain issuable.
    AICORE void wait_noc_to_l1(__cbuf__ void* dst, uint32_t capacityBytes, uint32_t expVal, epoch_id_t epoch)
    {
        recv = reinterpret_cast<__cbuf__ uint8_t*>(dst);
        contract = {capacityBytes, expVal, epoch, 0, 0};
        if (epoch != ctx[3])
            fail(Error::Epoch);
        if (expVal > capacityBytes)
            fail(Error::Capacity);
    }

    AICORE void copy_l1_to_noc_l1_align_b8(
        __cbuf__ void* dst, __cbuf__ const void* src, uint32_t burstCount, uint32_t burstBytes, uint32_t srcStrideBytes,
        uint32_t dstStrideBytes, route_hint_t nextHop, calendar_slot_id_t slot, epoch_id_t epoch, calendar_red_t redOp,
        vtype_t dtype)
    {
        if (!ok())
            return;
        auto routes = reinterpret_cast<__gm__ uint32_t*>(ctx[4]);
        auto slots = reinterpret_cast<__gm__ uint32_t*>(ctx[5]);
        uint32_t audience = ((1u << ctx[9]) - 1) & ~(1u << rank);
        if (nextHop != routes[rank] || nextHop != audience || slot != slots[rank] || slot != ctx[2] * 32 + rank ||
            redOp != CAL_RED_NONE || dtype != f16 || epoch != ctx[3]) {
            fail(Error::Binding);
            return;
        }
        uint32_t offset = reinterpret_cast<__cbuf__ uint8_t*>(dst) - recv;
        if (!burstCount || !burstBytes || burstBytes % 32 ||
            (burstCount > 1 && (srcStrideBytes < burstBytes || dstStrideBytes < burstBytes)) ||
            uint64_t(burstCount) * burstBytes > ctx[10]) {
            fail(Error::Geometry);
            return;
        }
        if (uint64_t(offset) + uint64_t(burstCount - 1) * dstStrideBytes + burstBytes > contract.capacity) {
            fail(Error::Capacity);
            return;
        }
        auto mailbox = reinterpret_cast<__gm__ uint8_t*>(ctx[6]) + rank * ctx[10];
        auto source = reinterpret_cast<__cbuf__ const uint8_t*>(src);
        for (uint32_t row = 0; row < burstCount; ++row)
            copy_cbuf_to_gm(mailbox + row * burstBytes, source + row * srcStrideBytes, 0, 1, burstBytes / 32, 0, 0);
        pipe_barrier(PIPE_MTE3);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        auto signal = signals() + rank * 8;
        signal[2] = offset;
        signal[3] = burstCount;
        signal[4] = burstBytes;
        signal[5] = dstStrideBytes;
        publish(1, epoch);
    }

    AICORE void finish_receive()
    {
        // Receive-side pulls model NoC commit; the buffers themselves are real L1.
        for (uint32_t peer = 0; peer < ctx[9] && ok(); ++peer) {
            await(peer, 1, contract.epoch);
            if (!ok())
                break;
            auto packet = signals() + peer * 8;
            uint32_t offset = packet[2], rows = packet[3], bytes = packet[4], stride = packet[5];
            bool remote = peer != rank;
            // Check self-copy geometry too, but never include it in the remote counter.
            ReceiveContract local{contract.capacity, static_cast<uint32_t>(ctx[10]), contract.epoch};
            Error error = remote ? contract.check(packet[1], peer, offset, rows, bytes, stride) :
                                   local.check(packet[1], peer, offset, rows, bytes, stride);
            if (error != Error::OK) {
                fail(error);
                break;
            }
            auto mailbox = reinterpret_cast<__gm__ uint8_t*>(ctx[6]) + peer * ctx[10];
            for (uint32_t row = 0; row < rows; ++row)
                copy_gm_to_cbuf(recv + offset + row * stride, mailbox + row * bytes, 0, 1, bytes / 32, 0, 0, PAD_NONE);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            if (remote)
                contract.commit(peer, rows * bytes);
        }
        if (ok() && contract.committed != contract.expected)
            fail(Error::ByteCount);
    }

    AICORE void finish()
    {
        if (rank >= ctx[9] || rank >= 24)
            return;
        auto status = reinterpret_cast<__gm__ uint64_t*>(ctx[8]) + rank * 8;
        status[0] = static_cast<uint32_t>(fault);
        status[1] = contract.committed;
        status[2] = ctx[3];
        status[3] = ctx[0];
        status[4] = ctx[2];
        dcci(status, cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

private:
    __gm__ uint64_t* ctx;
    uint32_t rank;
    Error fault = Error::OK;
    __cbuf__ uint8_t* recv = nullptr;
    ReceiveContract contract{0, 0, 0};
    AICORE __gm__ volatile uint64_t* signals() const { return reinterpret_cast<__gm__ volatile uint64_t*>(ctx[7]); }
    AICORE void publish(uint32_t field, uint64_t value)
    {
        auto signal = signals() + rank * 8;
        signal[field] = value;
        dcci(signal, cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
    AICORE void await(uint32_t peer, uint32_t field, uint64_t epoch)
    {
        auto signal = signals() + peer * 8;
        for (uint32_t poll = 0; poll < 1000000; ++poll) {
            dcci(signal, cache_line_t::SINGLE_CACHE_LINE);
            if (signal[field] == epoch)
                return;
        }
        fail(Error::Timeout);
    }
};
} // namespace pto::wse
#endif
#endif

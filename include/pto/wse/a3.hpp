/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_WSE_A3_HPP
#define PTO_WSE_A3_HPP
#include <pto/pto-inst.hpp>
#include <pto/wse/calendar_types.hpp>

namespace pto::wse {

#if defined(__CCE_AICORE__)

template <typename T, int Rows, int Cols, int RowStride = Cols>
using LocalView = GlobalTensor<T, Shape<1, 1, 1, Rows, Cols>, Stride<1, 1, 1, RowStride, 1>>;

template <typename T, int Rows, int Cols>
AICORE __gm__ T* LocalSlot(__gm__ T* base, uint32_t physical)
{
    constexpr uint32_t SlotBytes = (Rows * Cols * sizeof(T) + 2047) / 2048 * 2048;
    return base + physical * (SlotBytes / sizeof(T));
}

template <int Rows, int Cols, int Workers, int Inner>
AICORE void LocalGemm(__gm__ half* lhs, __gm__ half* rhs, __gm__ float* out, __gm__ int32_t* mapping)
{
#if defined(__DAV_C220_CUBE__)
    static_assert(Rows > 0 && Rows <= 16 && Cols % 16 == 0 && Inner % 64 == 0);
    uint32_t logical = get_block_idx();
    if (logical >= Workers)
        return;
    uint32_t physical = mapping[logical];
    auto* a = LocalSlot<half, Rows, Inner>(lhs, physical);
    auto* b = LocalSlot<half, Inner, Cols>(rhs, physical);
    auto* c = LocalSlot<float, Rows, Cols>(out, physical);
    using AMat = Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, Rows, 64, SLayout::RowMajor>;
    using BMat = Tile<TileType::Mat, half, 64, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor>;
    AMat am;
    BMat bm;
    TileLeft<half, 16, 64, Rows, 64> al;
    TileRight<half, 64, 16, 64, 16> br;
    TileAcc<float, 16, 16, Rows, 16> acc;
    TASSIGN(am, 0);
    TASSIGN(bm, 4096);
    TASSIGN(al, 0);
    TASSIGN(br, 0);
    TASSIGN(acc, 0);
    for (int n = 0; n < Cols; n += 16) {
        for (int k = 0; k < Inner; k += 64) {
            LocalView<half, Rows, 64, Inner> av(a + k);
            LocalView<half, 64, 16, Cols> bv(b + k * Cols + n);
            TLOAD(am, av);
            TLOAD(bm, bv);
            pipe_barrier(PIPE_ALL);
            TMOV(al, am);
            TMOV(br, bm);
            pipe_barrier(PIPE_ALL);
            if (k == 0)
                TMATMUL(acc, al, br);
            else
                TMATMUL_ACC(acc, acc, al, br);
            pipe_barrier(PIPE_ALL);
        }
        LocalView<float, Rows, 16, Cols> cv(c + n);
        TSTORE(cv, acc);
        pipe_barrier(PIPE_ALL);
    }
#endif
}

template <int Rows, int Cols, int Workers, int Kind>
AICORE void LocalVector(__gm__ float* lhs, __gm__ float* rhs, __gm__ float* out, __gm__ int32_t* mapping)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0 || get_block_idx() >= Workers)
        return;
    uint32_t physical = mapping[get_block_idx()];
    set_mask_norm();
    set_vector_mask(-1, -1);
    using Vec = Tile<TileType::Vec, float, Rows, Cols, BLayout::RowMajor, Rows, Cols>;
    Vec a, b, tmp, result;
    constexpr int Bytes = Rows * Cols * 4;
    static_assert(Bytes * 4 <= 192 * 1024);
    TASSIGN(a, 0);
    TASSIGN(b, Bytes);
    TASSIGN(tmp, Bytes * 2);
    TASSIGN(result, Bytes * 3);
    LocalView<float, Rows, Cols> av(LocalSlot<float, Rows, Cols>(lhs, physical));
    LocalView<float, Rows, Cols> bv(LocalSlot<float, Rows, Cols>(rhs, physical));
    LocalView<float, Rows, Cols> cv(LocalSlot<float, Rows, Cols>(out, physical));
    TLOAD(a, av);
    if constexpr (Kind == 1)
        TLOAD(b, bv);
    pipe_barrier(PIPE_ALL);
    if constexpr (Kind == 0) {
        TMULS(tmp, a, -1.0f);
        pipe_barrier(PIPE_V);
        TEXP(b, tmp);
        pipe_barrier(PIPE_V);
        TADDS(tmp, b, 1.0f);
        pipe_barrier(PIPE_V);
        TDIV(result, a, tmp);
    } else {
        TMUL(result, a, b);
    }
    pipe_barrier(PIPE_ALL);
    TSTORE(cv, result);
    pipe_barrier(PIPE_ALL);
#endif
}

template <int Rows, int Cols, int Workers>
AICORE void LocalSilu(__gm__ float* lhs, __gm__ float* rhs, __gm__ float* out, __gm__ int32_t* mapping)
{
    LocalVector<Rows, Cols, Workers, 0>(lhs, rhs, out, mapping);
}

template <int Rows, int Cols, int Workers>
AICORE void LocalMul(__gm__ float* lhs, __gm__ float* rhs, __gm__ float* out, __gm__ int32_t* mapping)
{
    LocalVector<Rows, Cols, Workers, 1>(lhs, rhs, out, mapping);
}

template <int Rows, int Cols, int Workers>
AICORE void LocalCast(__gm__ float* lhs, __gm__ float*, __gm__ half* out, __gm__ int32_t* mapping)
{
#if defined(__DAV_C220_VEC__)
    static_assert(Rows > 0 && Rows <= 16 && Cols % 16 == 0 && Rows * Cols * 6 <= 192 * 1024);
    if (get_subblockid() != 0 || get_block_idx() >= Workers)
        return;
    uint32_t physical = mapping[get_block_idx()];
    set_mask_norm();
    set_vector_mask(-1, -1);
    Tile<TileType::Vec, float, Rows, Cols, BLayout::RowMajor, Rows, Cols> src;
    Tile<TileType::Vec, half, Rows, Cols, BLayout::RowMajor, Rows, Cols> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, Rows * Cols * 4);
    LocalView<float, Rows, Cols> sv(LocalSlot<float, Rows, Cols>(lhs, physical));
    LocalView<half, Rows, Cols> dv(LocalSlot<half, Rows, Cols>(out, physical));
    TLOAD(src, sv);
    pipe_barrier(PIPE_ALL);
    TCVT(dst, src, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_ALL);
    TSTORE(dv, dst);
    pipe_barrier(PIPE_ALL);
#endif
}

template <calendar_op_t Op, calendar_red_t Red = CAL_RED_NONE>
struct CalendarTables {
    __gm__ int64_t* routes;
    InstalledSlotRef<Op> slots;
    static constexpr auto opcode = Op;
    static constexpr auto redOp = Red;
};

namespace mock {
AICORE inline void wait_noc_to_ubuf(
    ReceiveContext& ctx, uint32_t capacity, uint32_t expected, calendar_op_t opcode, uint32_t epoch)
{
    ctx.capacity = capacity;
    ctx.expected = expected;
    ctx.epoch = epoch;
    ctx.armed = expected <= capacity && opcode == CAL_ALLGATHER && epoch > 0 && epoch <= 255;
    if (!ctx.armed)
        ctx.error = CAL_BAD_GEOMETRY;
}

// A3 runs the fabric delivery at the receiving worker. Only this adapter may
// read peer-owned backing slots. TINSERT models the new UB-to-NoC-to-UB opcode.
template <class Send, class Recv>
AICORE void copy_ubuf_to_noc_ubuf_align_b8(
    ReceiveContext& ctx, Send& send, Recv& recv, uint32_t rank, uint32_t bytes, bool remote)
{
    if (!ctx.armed) {
        ctx.error = CAL_BAD_BINDING;
        return;
    }
    if (ctx.seen & (1u << rank)) {
        ctx.error = CAL_DUPLICATE;
        return;
    }
    ctx.seen |= 1u << rank;
    TINSERT(recv, send, 0, rank * Send::Cols);
    pipe_barrier(PIPE_ALL);
    if (remote)
        ctx.committed += bytes;
}
} // namespace mock

template <int Rows, int Cols, int GroupSize, int Workers, int Nodes, class Send, class Recv>
AICORE RecordEvent TCalendar(
    CalendarTables<CAL_ALLGATHER> tables, CalendarState& state, CoreBinding self, uint32_t selfOff, Send& send,
    Recv& recv, __gm__ half* transport, __gm__ int32_t* mapping, __gm__ int32_t* members)
{
    uint64_t rbLo = tables.routes[self.physicalId * 2];
    uint64_t rbHi = tables.routes[self.physicalId * 2 + 1];
    (void)rbLo;
    constexpr uint32_t Expected = Rows * Cols * 2 * (GroupSize - 1);
    uint32_t epoch = ++state.epochCtr[CAL_ALLGATHER];
    ReceiveContext ctx;
    if (uint32_t(rbHi >> 32) != Expected || selfOff >= Cols * 2 * GroupSize)
        return {CAL_BAD_GEOMETRY, 0, epoch};
    mock::wait_noc_to_ubuf(ctx, Rows * Cols * GroupSize * 2, Expected, CAL_ALLGATHER, epoch);
    for (uint32_t rank = 0; rank < GroupSize; ++rank) {
        uint32_t source = members[self.logicalId * GroupSize + rank];
        if (source >= Workers)
            return {CAL_BAD_MEMBER, ctx.committed, epoch};
        uint32_t physical = mapping[source];
        if (physical >= Nodes)
            return {CAL_BAD_MEMBER, ctx.committed, epoch};
        uint64_t lo = tables.routes[physical * 2];
        uint64_t hi = tables.routes[physical * 2 + 1];
        uint32_t pair =
            self.physicalId < 32 ? (lo >> (2 * self.physicalId)) & 3u : (hi >> (2 * (self.physicalId - 32))) & 3u;
        bool remote = source != self.logicalId;
        if ((remote && pair != 3) || (!remote && selfOff != rank * Cols * 2))
            return {CAL_BAD_ROUTE, ctx.committed, epoch};
        LocalView<half, Rows, Cols> peer(LocalSlot<half, Rows, Cols>(transport, physical));
        TLOAD(send, peer);
        pipe_barrier(PIPE_ALL);
        mock::copy_ubuf_to_noc_ubuf_align_b8(ctx, send, recv, rank, Rows * Cols * 2, remote);
        if (ctx.error)
            return {ctx.error, ctx.committed, epoch};
    }
    return {ctx.committed == Expected ? CAL_OK : CAL_INCOMPLETE, ctx.committed, epoch};
}

template <int Rows, int Cols, int GroupSize, int PlanSlot, int RecvOffset, int Workers, int Nodes>
AICORE void CalendarKernel(
    __gm__ half* sendStorage, __gm__ half* recvStorage, __gm__ int64_t* routes, __gm__ int32_t* mapping,
    __gm__ int32_t* members, __gm__ int32_t* control, __gm__ int32_t* status)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0 || get_block_idx() >= Workers)
        return;
    static_assert(RecvOffset >= Rows * Cols * 2 && RecvOffset % 32 == 0);
    static_assert(RecvOffset + Rows * Cols * GroupSize * 2 <= 192 * 1024);
    uint32_t logical = get_block_idx();
    uint32_t physical = mapping[logical];
    auto* record = status + logical * CalendarStatusWords;
    record[0] = CAL_BAD_BINDING;
    if (physical >= Nodes || control[0] <= 0 || control[1] != PlanSlot || control[2] != CAL_ALLGATHER ||
        control[3] != 1 || control[5] <= 0)
        return;
    record[0] = CAL_BAD_EPOCH;
    if (control[4] <= 0 || control[4] > 255)
        return;
    uint32_t selfOff = GroupSize * Cols * 2;
    for (uint32_t rank = 0; rank < GroupSize; ++rank)
        if (members[logical * GroupSize + rank] == logical)
            selfOff = rank * Cols * 2;
    set_mask_norm();
    set_vector_mask(-1, -1);
    Tile<TileType::Vec, half, Rows, Cols, BLayout::RowMajor, Rows, Cols> send;
    Tile<TileType::Vec, half, Rows, Cols * GroupSize, BLayout::RowMajor, Rows, Cols * GroupSize> recv;
    TASSIGN(send, 0);
    TASSIGN(recv, RecvOffset);
    CalendarState state{};
    state.epochCtr[CAL_ALLGATHER] = control[4] - 1;
    CalendarTables<CAL_ALLGATHER> tables{routes, {uint32_t(control[0])}};
    RecordEvent done = TCalendar<Rows, Cols, GroupSize, Workers, Nodes>(
        tables, state, {logical, physical}, selfOff, send, recv, sendStorage, mapping, members);
    if (done.error == CAL_OK) {
        LocalView<half, Rows, Cols * GroupSize> output(LocalSlot<half, Rows, Cols * GroupSize>(recvStorage, physical));
        TSTORE(output, recv);
        pipe_barrier(PIPE_ALL);
    }
    record[0] = done.error;
    record[1] = done.committedBytes;
    record[2] = done.epoch;
    record[3] = tables.slots.bindingId;
    record[4] = logical;
    record[5] = physical;
#endif
}
#endif
} // namespace pto::wse
#endif

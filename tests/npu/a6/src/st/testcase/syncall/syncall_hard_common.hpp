/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Shared body for the A6 hard (FFTS) SYNCALL ST cases. Every participant owns one
// 64-byte GM slot, publishes a value into it, crosses a barrier, then reads every
// peer slot back. Three barriers per run: a single barrier only proves the cores
// met once, while the 1:1 regression this guards against (a cube-side
// set_intra_block with no AIV consumer leaving a spare credit) first shows up on
// the second barrier, which would then pass straight through.

#ifndef SYNCALL_HARD_COMMON_HPP
#define SYNCALL_HARD_COMMON_HPP

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

// One full 64-byte cache line per participant slot. The cube core publishes with a
// scalar store plus dcci, which writes back the whole line, so narrower slots would
// let one core clobber its neighbour.
constexpr int32_t kInt32PerCacheLine = 16;
constexpr uint64_t kFlagUbAddr = 0x0;
constexpr uint64_t kReadUbAddr = 0x1000;
constexpr uint64_t kOutUbAddr = 0x2000;

PTO_INTERNAL void SyncAllDcci(__gm__ void* ptr)
{
    __asm__ __volatile__("" ::: "memory");
    dcci(ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
}

PTO_INTERNAL void SyncAllDcciRange(__gm__ int32_t* base, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        SyncAllDcci(static_cast<__gm__ void*>(base + i * kInt32PerCacheLine));
    }
    dsb(DSB_DDR);
    __asm__ __volatile__("" ::: "memory");
}

// Publish one int32 into this participant's GM slot.
PTO_INTERNAL void StoreParticipantLine(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr)
{
#if defined(__DAV_VEC__)
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    pto_copy_ubuf_to_gm_align_v2<int32_t>(dst, ub, 0, 1, sizeof(int32_t), 0, 0, 0);
    pipe_barrier(PIPE_ALL);
#elif defined(__DAV_CUBE__)
    // A6 AIC has no copy_cbuf_to_gm, but a scalar GM store plus dcci works.
    (void)ubAddr;
    dst[0] = value;
    SyncAllDcci(static_cast<__gm__ void*>(dst));
    dsb(DSB_DDR);
#else
    (void)dst;
    (void)value;
    (void)ubAddr;
#endif
}

// Read every participant slot and report whether slot i holds (i + 1) * multiplier.
PTO_INTERNAL int32_t CheckAllFlags(__gm__ int32_t* flags, int32_t totalParticipants, int32_t multiplier)
{
    SyncAllDcciRange(flags, totalParticipants);
    int32_t allVisible = 1;
#if defined(__DAV_VEC__)
    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(kReadUbAddr);
    const uint32_t lenBurst =
        static_cast<uint32_t>(totalParticipants) * static_cast<uint32_t>(kInt32PerCacheLine) * sizeof(int32_t);
    pto_copy_gm_to_ubuf_align_v2<int32_t>(readUb, flags, 0, 1, lenBurst, 0, 0, false, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    for (int32_t i = 0; i < totalParticipants; ++i) {
        if (readUb[i * kInt32PerCacheLine] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
#elif defined(__DAV_CUBE__)
    for (int32_t i = 0; i < totalParticipants; ++i) {
        if ((flags + i * kInt32PerCacheLine)[0] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
#else
    (void)flags;
    (void)totalParticipants;
    (void)multiplier;
#endif
    return allVisible;
}

// Shared run body. idx is the participant's logical slot: the block index for
// single-core-type launches, and cube blocks followed by vector blocks for MIX.
// out[idx] ends up 1 only if this participant saw every peer's round-1 and
// round-2 publication, which requires all three barriers to have held.
template <int32_t TotalParticipants, SyncCoreType CoreType>
PTO_INTERNAL void RunHardSyncAllBody(__gm__ int32_t* out, __gm__ int32_t* flags, int32_t idx)
{
    __gm__ int32_t* mySlot = flags + idx * kInt32PerCacheLine;

    StoreParticipantLine(mySlot, idx + 1, kFlagUbAddr);
    SYNCALL<CoreType>();
    const int32_t firstVisible = CheckAllFlags(flags, TotalParticipants, 1);

    // Second barrier: hold until every peer finished reading round 1, so the
    // round-2 overwrite below cannot race ahead of a slower reader.
    SYNCALL<CoreType>();
    StoreParticipantLine(mySlot, (idx + 1) * 2, kFlagUbAddr);
    SYNCALL<CoreType>();
    const int32_t secondVisible = CheckAllFlags(flags, TotalParticipants, 2);

    StoreParticipantLine(out + idx * kInt32PerCacheLine, firstVisible & secondVisible, kOutUbAddr);
}

#endif

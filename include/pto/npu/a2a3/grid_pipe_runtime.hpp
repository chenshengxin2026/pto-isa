/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 GridPipe runtime helpers: shmem window layout, init helpers, neighbor
// rank resolution.  See the V6 IPC_SCB scoreboard design and its A2/A3 mock in
// include/pto/npu/a2a3/grid_intrinsic.hpp.

#ifndef PTO_A2A3_GRID_PIPE_RUNTIME_HPP
#define PTO_A2A3_GRID_PIPE_RUNTIME_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

namespace pto {
namespace a2a3_grid {

// shmem window layout, in bytes.  ONE WINDOW PER PIPE PER RANK -- a pipe is one
// channel bound to one (producer, consumer) pair, so it owns one scoreboard pair
// and one slot ring.  A core that talks to several peers declares several pipes
// and gives each its own window region (the demos carve them out of one per-cell
// arena at fixed offsets, which keeps every rank's offsets identical -- required,
// because the peer resolver maps a local address to the SAME byte offset in the
// peer's window).
//
// Unicast pipe (GridPipe).  Every scoreboard owns a FULL CACHE LINE
// (grid_mock::kScbLineStride) because each has a DIFFERENT external writer and
// the write-back is line-granular -- see the rationale on kScbLineStride in
// grid_intrinsic.hpp.  Packing them (the original 4 B spacing) let one peer's
// store silently drop another peer's doorbell.
//
//   offset                        contents                       written by
//   ------------------------------------------------------------------------
//   kReadyScbOffset   (0)         ready   scb u32  (consumer sem)  producer peer
//   kFreeScbOffset    (64)        free    scb u32  (producer sem)  consumer peer
//   kInstallScbOffset (128)       install scb u32  (producer sem)  consumer peer
//                                 INSTALL_BASE doorbell = handoff generation
//   kOpenScbOffset    (192)       open    scb u32  (consumer sem)  producer peer
//                                 OPEN_ACK, same generation
//   kBatonOutOffset   (256)       baton L1 u32     (consumer sem)  THIS core
//                                 the retiring prod_idx staged for ST_HSCB
//   kBatonInOffset    (320)       baton L1 u32     (producer sem)  consumer peer
//                                 the relayed prod_idx, delivered by ST_HSCB and
//                                 lifted into the GPR by MOV_L12X
//   384 .. kFlagsBytes-1          reserved (alignment, telemetry)
//   kSlotRegionOffset (512)       slot ring [SlotCount * SlotStride]
//
// The last two are L1/SRAM words, NOT scoreboards: they carry a value to be READ,
// so they have no IPC_SCB slot number and nothing ever waits on them.  Two of them
// because under SPMD one core is both sides of a handoff at once -- it stages its
// own outgoing baton while its successor consumer delivers its incoming one, and a
// single shared word would collide.
//
// Each word's fault sentinel sits kFaultFlagWordOffset u32 words INTO ITS OWN
// line, so the sentinels stay clear of every live word.  (The sentinel write is
// local, so it shares a line with a remotely-written word; that is harmless in
// practice because a sentinel is only ever written on a run that has already
// failed.)
//
// The four handoff words are idle in steady state.
//
// Group pipe (GridGroupPipe, scheme-② 真·同时 MPSC):
//   0 .. kFlagsBytes-1            reserved (fault sentinels, alignment); the
//                                 group's semaphores are the lanes below, so
//                                 there is no scoreboard pair here
//   kSlotRegionOffset (128)       shared payload ring [SlotCount * SlotStride]
//   + SlotCount*SlotStride        per-source ready lanes [GroupMax * 64 B]
//                                 (variant B; one cache line per lane -- see
//                                 grid_mock::kBcastLaneStride)
//   + GroupMax*64                 per-source free lanes  [GroupMax * 64 B]
//                                 (this core is the sole writer of each)
//
// The pipe's header words, as u32 word indices / byte offsets into its window.
// Lines 0/1 are the steady-state ready/free scoreboard pair; lines 2/3 the 接力计数
// handoff doorbells; lines 4/5 the two baton L1 words (see the layout comment
// above).  The stride is one cache line, NOT one word -- that is the correctness
// requirement, not padding.
inline constexpr uint32_t kHeaderLineCount = 6;
inline constexpr uint32_t kReadyScbWord = 0 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kFreeScbWord = 1 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kInstallScbWord = 2 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kOpenScbWord = 3 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kBatonOutWord = 4 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kBatonInWord = 5 * grid_mock::kScbLineStrideU32;
inline constexpr uint32_t kReadyScbOffset = kReadyScbWord * sizeof(uint32_t);
inline constexpr uint32_t kFreeScbOffset = kFreeScbWord * sizeof(uint32_t);
inline constexpr uint32_t kInstallScbOffset = kInstallScbWord * sizeof(uint32_t);
inline constexpr uint32_t kOpenScbOffset = kOpenScbWord * sizeof(uint32_t);
inline constexpr uint32_t kBatonOutOffset = kBatonOutWord * sizeof(uint32_t);
inline constexpr uint32_t kBatonInOffset = kBatonInWord * sizeof(uint32_t);

// Flag header: kHeaderLineCount cache lines plus reserved headroom, rounded up so
// the slot ring stays generously aligned.  Host launchers mirror this constant
// (the *_GRID_FLAGS_BYTES in the demo configs).
inline constexpr uint32_t kFlagsBytes = 512;
static_assert(
    kFlagsBytes >= kHeaderLineCount * grid_mock::kScbLineStride, "flag header must hold one line per header word");
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

// Payload ring bytes -- identical formula for both pipe flavours (the group
// pipe's ring is the shared MPSC ring, so its SlotCount is the ring depth SC).
template <int SlotStride, int SlotCount>
inline constexpr uint32_t kSlotRegionBytes()
{
    return static_cast<uint32_t>(SlotCount) * static_cast<uint32_t>(SlotStride);
}

// One direction's worth of per-source lanes (ready or free), one cache line each.
template <int GroupMax>
inline constexpr uint32_t kLaneRegionBytes()
{
    return static_cast<uint32_t>(GroupMax) * grid_mock::kBcastLaneStride;
}

template <int SlotStride, int SlotCount>
inline constexpr uint32_t kPipeWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount>();
}

template <int SlotStride, int SlotCount, int GroupMax>
inline constexpr uint32_t kGroupPipeWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount>() + // shared payload ring
           kLaneRegionBytes<GroupMax>() +                                  // per-source ready lanes (variant B)
           kLaneRegionBytes<GroupMax>();                                   // per-source free  lanes
}

// Host-side helper: total bytes ONE pipe needs in each rank's window.
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    if constexpr (is_grid_group_pipe_v<Pipe>) {
        return kGroupPipeWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::GroupMax>();
    } else {
        return kPipeWindowBytes<Pipe::SlotStride, Pipe::SlotCount>();
    }
}

// Wire up a GridPipe / GridGroupPipe instance from a flat GM window owned by this
// rank.  The host launcher allocates WindowBytes<Pipe>() bytes per rank per pipe,
// then the kernel prologue calls this once per pipe.  `runtimeCtx` is the HCCL
// device context handle used later by GridTPush/GridTPop/GridTBroadcast to
// resolve cross-rank addresses.
//
// Offsets use the constexpr VARIABLES above plus plain arithmetic on the pipe's
// static members: CCE forbids calling a host constexpr *function* from an AICORE
// context, so the kXxx<...>() helpers are for the host mirrors only.
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(
    Pipe& pipe, GridShape shape, GridCoord coord, __gm__ uint8_t* window, __gm__ void* runtimeCtx, uint32_t pipeId)
{
    // (1) runtime-context group: identical for every pipe on this core.
    pipe.ctx.runtimeCtx = runtimeCtx;
    pipe.ctx.shape = shape;
    pipe.ctx.coord = coord;
    pipe.ctx.pipeId = pipeId;

    // (2) slot group: the payload ring follows the reserved flag header.
    pipe.slots.base = window + kSlotRegionOffset;

    // (3) semaphore group.
    if constexpr (is_grid_group_pipe_v<Pipe>) {
        // MPSC: per-source lane arrays instead of a scoreboard pair.
        const uint32_t ringBytes = static_cast<uint32_t>(Pipe::SlotCount) * static_cast<uint32_t>(Pipe::SlotStride);
        const uint32_t readyOff = kSlotRegionOffset + ringBytes;
        const uint32_t freeOff = readyOff + static_cast<uint32_t>(Pipe::GroupMax) * grid_mock::kBcastLaneStride;
        pipe.cons.readyLanes = reinterpret_cast<__gm__ uint32_t*>(window + readyOff);
        pipe.prod.freeLanes = reinterpret_cast<__gm__ uint32_t*>(window + freeOff);
    } else {
        // SPSC: this pipe's own ready/free scoreboard pair, its 接力计数 handoff
        // trio, and zeroed GPR counters.  Two pipes declared over the SAME window
        // (a time-division producer handoff) therefore land on the same physical
        // words, which is exactly what lets THANDOFF relay the counters instead of
        // starting the successor on an unrelated ring.
        __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
        pipe.cons.readyScb = scbs + kReadyScbWord;
        pipe.cons.openScb = scbs + kOpenScbWord;
        pipe.cons.batonL1 = scbs + kBatonOutWord;
        pipe.cons.consIndex = 0;
        pipe.prod.freeScb = scbs + kFreeScbWord;
        pipe.prod.installScb = scbs + kInstallScbWord;
        pipe.prod.batonL1 = scbs + kBatonInWord;
        pipe.prod.prodIndex = 0;
    }
    pipe.prod.window = GridPayloadWindow{};
    pipe.cons.window = GridPayloadWindow{};
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP

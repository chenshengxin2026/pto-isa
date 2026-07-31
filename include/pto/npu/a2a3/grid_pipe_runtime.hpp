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
// Unicast pipe (GridPipe):
//   offset                        contents
//   ----------------------------------------------------------------------
//   kReadyScbOffset (0)           ready scb u32  -- consumer semaphore, written
//                                 by the producer peer's SYNC_HSCB(READY)
//   kFreeScbOffset  (4)           free  scb u32  -- producer semaphore, written
//                                 by the consumer peer's SYNC_HSCB(FREE)
//   8 .. kFlagsBytes-1            reserved (fault sentinels, alignment, telemetry)
//   kSlotRegionOffset (128)       slot ring [SlotCount * SlotStride]
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
// Keep enough reserved words for the GridTPush/GridTPop fault sentinels:
//   readyScb + kFaultFlagWordOffset   (word 10)
//   freeScb  + kFaultFlagWordOffset   (word 11)
inline constexpr uint32_t kFlagsBytes = 128;
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

// The pipe's scoreboard pair, as u32 word indices / byte offsets into its window.
inline constexpr uint32_t kReadyScbWord = 0;
inline constexpr uint32_t kFreeScbWord = 1;
inline constexpr uint32_t kReadyScbOffset = kReadyScbWord * sizeof(uint32_t);
inline constexpr uint32_t kFreeScbOffset = kFreeScbWord * sizeof(uint32_t);

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
        // SPSC: this pipe's own ready/free scoreboard pair + zeroed GPR counters.
        __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
        pipe.cons.readyScb = scbs + kReadyScbWord;
        pipe.cons.consIndex = 0;
        pipe.prod.freeScb = scbs + kFreeScbWord;
        pipe.prod.prodIndex = 0;
    }
    pipe.prod.window = GridPayloadWindow{};
    pipe.cons.window = GridPayloadWindow{};
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP

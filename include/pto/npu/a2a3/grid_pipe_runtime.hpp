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

// shmem window layout (per rank), in bytes.  The ready/free scoreboard words
// stand in for the V6 ready_scb_<dir> / free_scb_<dir> IPC_SCB slots (each
// carries a monotone absolute count written by the peer's HSCB store):
//
//   offset                                         contents
//   ----------------------------------------------------------------------
//   0                                              ready scoreboards [kGridDirectionCount] u32
//   4 * kGridDirectionCount                        free scoreboards [kGridDirectionCount] u32
//   8 * kGridDirectionCount                        reserved (fault sentinels, alignment, telemetry)
//   kSlotRegionOffset                              slot region for the ALLOCATED directions
//     + ring(dir) * SlotCount * SlotStride         slot ring for that direction
//   kSlotRegionOffset + R*SlotCount*SlotStride     TBROADCAST region (only if GroupMax > 0):
//     + 0                                            shared payload ring [BcastSlotCount * SlotStride]
//     + BcastSlotCount*SlotStride                    per-source ready lanes [GroupMax * 64 B] (variant B;
//                                                    one cache line per lane -- see kBcastLaneStride)
//     + GroupMax*64                                   per-source free  lanes [GroupMax * 64 B] (X sole writer)
//
// R = GridDirRingCount(DirMask) and ring(dir) = GridDirRingIndex(DirMask, dir):
// only the directions a pipe actually pushes/pops get a ring, and their rings are
// packed from offset 0.  The default DirMask (kGridDirAll) gives R = 5 and
// ring(dir) == GridDirectionIndex(dir), i.e. the original layout byte for byte.
// A pure-broadcast pipe (DirMask = kGridDirNone) has R = 0 and pays no unicast
// bytes at all.
//
// The TBROADCAST region is appended only when GroupMax > 0; a unicast-only pipe
// (BcastSlotCount = GroupMax = 0) has no broadcast region and a byte-identical
// window to the pre-TBROADCAST layout.  Keep enough reserved words for
// GridTPush/GridTPop fault sentinels:
//   readyScb[dir] + kFaultFlagWordOffset
//   freeScb[dir]  + kFaultFlagWordOffset
inline constexpr uint32_t kFlagsBytes = 128;
inline constexpr uint32_t kSlotRegionOffset = kFlagsBytes;

inline constexpr uint32_t kReadyScbOffset(GridDirection d) { return static_cast<uint32_t>(d) * sizeof(uint32_t); }

inline constexpr uint32_t kFreeScbOffset(GridDirection d)
{
    return kGridDirectionCount * sizeof(uint32_t) + static_cast<uint32_t>(d) * sizeof(uint32_t);
}

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kSlotRegionBytes()
{
    return static_cast<uint32_t>(GridDirRingCount(DirMask)) * SlotCount * SlotStride;
}

// TBROADCAST (scheme-②) region offsets/sizes.  No-ops (zero) when GroupMax == 0.
template <int SlotBytes, int BcastSlotCount>
inline constexpr uint32_t kBcastRingBytes()
{
    return static_cast<uint32_t>(BcastSlotCount) * static_cast<uint32_t>(SlotBytes);
}

template <int GroupMax>
inline constexpr uint32_t kBcastLaneBytes()
{
    return static_cast<uint32_t>(GroupMax) * grid_mock::kBcastLaneStride; // one 64 B cache line per lane
}

template <int SlotStride, int SlotCount, int BcastSlotCount, int GroupMax>
inline constexpr uint32_t kBcastRegionBytes()
{
    return kBcastRingBytes<SlotStride, BcastSlotCount>() + // shared payload ring
           kBcastLaneBytes<GroupMax>() +                   // per-source ready lanes (variant B)
           kBcastLaneBytes<GroupMax>();                    // per-source free  lanes
}

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kWindowBytes()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount, DirMask>();
}

template <int SlotStride, int SlotCount, int BcastSlotCount, int GroupMax, int DirMask = kGridDirAll>
inline constexpr uint32_t kWindowBytesWithBcast()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount, DirMask>() +
           kBcastRegionBytes<SlotStride, SlotCount, BcastSlotCount, GroupMax>();
}

template <int SlotStride, int SlotCount, int DirMask = kGridDirAll>
inline constexpr uint32_t kDirSlotRegionOffset(GridDirection d)
{
    return kSlotRegionOffset + static_cast<uint32_t>(GridDirRingIndex(DirMask, d)) * SlotCount * SlotStride;
}

// Wire up a GridPipe instance from a flat GM window owned by this rank.
// The host launcher allocates WindowBytes<Pipe>() bytes per rank, then calls
// this in the kernel prologue.  `runtimeCtx` is the HCCL device context handle
// used later by GridTPush/GridTPop/GridTBroadcast to resolve cross-rank
// addresses.
//
// The unicast offsets use the constexpr variable kSlotRegionOffset + plain
// arithmetic (CCE forbids calling a host constexpr *function* from an AICORE
// context, so we do not call the kXxxOffset() helpers here even though they are
// constexpr -- only the variable + the pipe's static members are needed).
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(
    Pipe& pipe, GridShape shape, GridCoord coord, __gm__ uint8_t* window, __gm__ void* runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    // Scoreboards stay indexed by direction (all 5 always exist -- they are 4 bytes
    // each); only the slot RINGS are packed by DirMask.  `ring` walks the allocated
    // directions in order, so ring(dir) matches GridDirRingIndex(DirMask, dir)
    // without calling it from this AICORE context.
    __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
    int ring = 0;
    for (int i = 0; i < kGridDirectionCount; ++i) {
        pipe.readyScb[i] = scbs + i;
        pipe.freeScb[i] = scbs + kGridDirectionCount + i;
        if (((Pipe::DirMask >> i) & 1) != 0) {
            pipe.slotBase[i] = window + kSlotRegionOffset + ring * Pipe::SlotCount * Pipe::SlotStride;
            ++ring;
        } else {
            pipe.slotBase[i] = nullptr; // no ring allocated for this direction
        }
        pipe.prodIndex[i] = 0;
        pipe.consIndex[i] = 0;
        pipe.pushWindow[i] = GridPayloadWindow{};
        pipe.popWindow[i] = GridPayloadWindow{};
    }
    pipe.bcastWindow = GridPayloadWindow{};

    // TBROADCAST region (scheme-② 真·同时 MPSC).  Only wired when the pipe
    // opted in (GroupMax > 0); a unicast-only pipe leaves these null and pays
    // zero window bytes for broadcast.  Offsets are computed inline from the
    // constexpr variable + the pipe's static members (see the note above).
    if constexpr (Pipe::GroupMax > 0) {
        const uint32_t slotRegionBytes = static_cast<uint32_t>(Pipe::RingCount) *
                                         static_cast<uint32_t>(Pipe::SlotCount) *
                                         static_cast<uint32_t>(Pipe::SlotStride);
        const uint32_t ringOff = kSlotRegionOffset + slotRegionBytes;
        const uint32_t readyOff =
            ringOff + static_cast<uint32_t>(Pipe::BcastSlotCount) * static_cast<uint32_t>(Pipe::SlotStride);
        const uint32_t freeOff = readyOff + static_cast<uint32_t>(Pipe::GroupMax) *
                                                grid_mock::kBcastLaneStride; // ready region = GroupMax lanes * 64 B
        pipe.bcastRingBase = window + ringOff;
        pipe.bcastReadyLanes = reinterpret_cast<__gm__ uint32_t*>(window + readyOff);
        pipe.bcastFreeLanes = reinterpret_cast<__gm__ uint32_t*>(window + freeOff);
    }
}

// Host-side helper: total bytes per rank for a single GridPipe (broadcast
// region included when the pipe opted in).
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    if constexpr (Pipe::GroupMax > 0) {
        return kWindowBytesWithBcast<
            Pipe::SlotStride, Pipe::SlotCount, Pipe::BcastSlotCount, Pipe::GroupMax, Pipe::DirMask>();
    } else {
        return kWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::DirMask>();
    }
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP

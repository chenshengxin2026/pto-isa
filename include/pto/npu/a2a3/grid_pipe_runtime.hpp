/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 GridPipe runtime helpers: shmem window layout and the init helper that
// wires a pipe's concurrency array to it.  See the V6 IPC_SCB scoreboard design
// and its A2/A3 mock in include/pto/npu/a2a3/grid_intrinsic.hpp.

#ifndef PTO_A2A3_GRID_PIPE_RUNTIME_HPP
#define PTO_A2A3_GRID_PIPE_RUNTIME_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

namespace pto {
namespace a2a3_grid {

// shmem window layout (per rank), in bytes.  The ready/free/close scoreboard words stand
// in for the V6 IPC_SCB slots of each CHANNEL (each carries a monotone absolute
// count written by the peer bound to that channel via an HSCB store):
//
//   offset                                          contents
//   -----------------------------------------------------------------------
//   0                                               ready scoreboards, one per
//     + c * kScbLineStride                            channel, ONE CACHE LINE EACH
//   kGridChanCount * kScbLineStride                 free scoreboards, likewise
//     + c * kScbLineStride
//   2 * kGridChanCount * kScbLineStride             close scoreboards, likewise
//     + c * kScbLineStride
//   kScbHeaderBytes                                 bind-request L1 line
//   kScbHeaderBytes + kScbLineStride                bind-response L1 line
//   kRecordOffset                                   pipe record: bindings, consumer
//                                                     FSM/history, both channel maps,
//                                                     close bases and run counters
//   kSlotRegionOffset                               slot region, ChanCount rings
//     + c * SlotCount * SlotStride                    ring of channel c
//   kSlotRegionOffset + C*SlotCount*SlotStride      TBROADCAST region (GroupMax > 0):
//     + 0                                             shared payload ring [BcastSlotCount * SlotStride]
//     + BcastSlotCount*SlotStride                     per-source ready lanes [GroupMax * 64 B] (variant B;
//                                                     one cache line per lane -- see kBcastLaneStride)
//     + GroupMax*64                                    per-source free  lanes [GroupMax * 64 B] (X sole writer)
//   end of receive rings / broadcast lanes           producer staging [SlotStride]
//                                                     local L1 source for every outbound transfer
//
// TWO PROPERTIES THE REST OF THE SYSTEM LEANS ON.
//
// (1) The scoreboard header is a FIXED kGridChanCount triplets, whatever a pipe's
//     ChanCount is, and every scoreboard owns a whole cache line.  Fixed, because
//     the peer resolver maps a local address to the SAME byte offset in the peer's
//     window, so two pipes in one build must agree on where channel c's doorbell
//     lives.  A line each, because each has a DIFFERENT external writer and the
//     mock's write-back is line-granular -- see kScbLineStride in
//     grid_intrinsic.hpp for the lost-update this prevents.  Only the RINGS are
//     trimmed by ChanCount, so a pure-broadcast pipe (ChanCount = 0) pays the
//     header and no unicast payload bytes at all.
//
// (2) The pipe record is LOCAL-ONLY -- this core is its sole reader and writer, no
//     peer ever stores into it -- so its words may share cache lines freely, and it
//     sits OUTSIDE kFlagsBytes.  That boundary matters: the host launchers scan the
//     flag header for fault sentinels, and the record holds ordinary counters and
//     block ids that would read as sentinels if they were inside.
//
// (3) The final SlotStride bytes are a LOCAL PRODUCER STAGING SLOT, not another
//     receive-ring entry.  Real WSE hardware has one unified L1 SRAM address space
//     (there is no physically separate Vec UB).  An outbound tile is therefore
//     staged here first and the NoC maps this local L1 address to the peer's receive
//     payload ring.  Keeping the producer slot after every receive-side region makes
//     source and destination storage disjoint even when a cell relays a tile while
//     its own receive ring is live.
//
// Fault sentinels live at word kFaultFlagWordOffset of the scoreboard (or lane)
// they belong to, which is inside that scoreboard's own line and clear of every
// live word.
inline constexpr uint32_t kScbHeaderBytes = 3U * static_cast<uint32_t>(kGridChanCount) * grid_mock::kScbLineStride;

// Two remotely-written control lines used only when a producer opens/reopens a
// time-division MPSC binding.  Request = [producer id commit, producer channel];
// response = [ready baseline, consumer channel, completion commit].  Producer and
// consumer channels are independent.  Keeping request/response on separate cache
// lines prevents line-granular mock write-back from clobbering the other mailbox.
inline constexpr uint32_t kBindRequestOffset = kScbHeaderBytes;
inline constexpr uint32_t kBindResponseOffset = kBindRequestOffset + grid_mock::kScbLineStride;
inline constexpr uint32_t kControlBytes = 2U * grid_mock::kScbLineStride;

// Pipe record: bindings, consumer FSM/history, producer/consumer channel maps and
// states, close bases, and persistent prod/cons counter mirrors.  It lets a schedule span several kernel
// launches; rounded up to a cache line so the slot region stays aligned.
inline constexpr uint32_t kRecordOffset = kScbHeaderBytes + kControlBytes;
inline constexpr uint32_t kRecordBytes =
    ((static_cast<uint32_t>(kGridRecordWords) * static_cast<uint32_t>(sizeof(uint32_t)) + grid_mock::kScbLineStride -
      1U) /
     grid_mock::kScbLineStride) *
    grid_mock::kScbLineStride;

// Bytes the host scans for fault sentinels: the scoreboard header only.
inline constexpr uint32_t kFlagsBytes = kScbHeaderBytes;
inline constexpr uint32_t kSlotRegionOffset = kRecordOffset + kRecordBytes;

inline constexpr uint32_t kReadyScbOffset(int chan) { return static_cast<uint32_t>(chan) * grid_mock::kScbLineStride; }

inline constexpr uint32_t kFreeScbOffset(int chan)
{
    return (static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(chan)) * grid_mock::kScbLineStride;
}

inline constexpr uint32_t kCloseScbOffset(int chan)
{
    return (2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(chan)) * grid_mock::kScbLineStride;
}

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kSlotRegionBytes()
{
    return static_cast<uint32_t>(ChanCount) * SlotCount * SlotStride;
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

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kProducerRegionOffset()
{
    return kSlotRegionOffset + kSlotRegionBytes<SlotStride, SlotCount, ChanCount>();
}

template <int SlotStride, int SlotCount, int BcastSlotCount, int GroupMax, int ChanCount = kGridChanCount>
inline constexpr uint32_t kProducerRegionOffsetWithBcast()
{
    return kProducerRegionOffset<SlotStride, SlotCount, ChanCount>() +
           kBcastRegionBytes<SlotStride, SlotCount, BcastSlotCount, GroupMax>();
}

template <int SlotStride, int SlotCount, int ChanCount = kGridChanCount>
inline constexpr uint32_t kWindowBytes()
{
    return kProducerRegionOffset<SlotStride, SlotCount, ChanCount>() +
           static_cast<uint32_t>(SlotStride); // isolated local producer staging slot
}

template <int SlotStride, int SlotCount, int BcastSlotCount, int GroupMax, int ChanCount = kGridChanCount>
inline constexpr uint32_t kWindowBytesWithBcast()
{
    return kProducerRegionOffsetWithBcast<SlotStride, SlotCount, BcastSlotCount, GroupMax, ChanCount>() +
           static_cast<uint32_t>(SlotStride); // isolated local producer staging slot
}

template <int SlotStride, int SlotCount>
inline constexpr uint32_t kChanSlotRegionOffset(int chan)
{
    return kSlotRegionOffset + static_cast<uint32_t>(chan) * SlotCount * SlotStride;
}

// Wire up a GridPipe instance from a flat GM window owned by this rank.
// The host launcher allocates WindowBytes<Pipe>() bytes per rank, then calls
// this in the kernel prologue.  `runtimeCtx` is the HCCL device context handle
// used later by GridTPush/GridTPop/GridTBroadcast to resolve cross-rank
// addresses.
//
// It wires resources and ADOPTS the window's pipe record.  No channel is bound
// here: which producer each element serves is a runtime decision the kernel makes
// dynamically by the first TPUSH/TPOP identity handshake.
//
// Adopting rather than clearing is what makes a multi-launch schedule work: the
// scoreboards and rings in this window outlive the kernel, so the allocator's
// memory of which of them are already dirty has to as well.  A window the host has
// just memset reads back as "nothing bound" on its own.
//
// The offsets use the constexpr VARIABLES above plus plain arithmetic (CCE forbids
// calling a host constexpr *function* from an AICORE context, so the kXxxOffset()
// helpers are not called here even though they are constexpr).
template <typename Pipe>
AICORE inline void InitGridPipeFromWindow(
    Pipe& pipe, GridShape shape, GridCoord coord, __gm__ uint8_t* window, __gm__ void* runtimeCtx, uint32_t pipeId)
{
    pipe.shape = shape;
    pipe.coord = coord;
    pipe.runtimeCtx = runtimeCtx;
    pipe.pipeId = pipeId;

    // Scoreboards exist for all kGridChanCount channels (the header is fixed);
    // only the RINGS are trimmed to Pipe::ChanCount.  All three scoreboards of a
    // channel are a whole cache line apart, so step by kScbLineStrideU32 in u32 units.
    __gm__ uint32_t* scbs = reinterpret_cast<__gm__ uint32_t*>(window);
    for (int c = 0; c < kGridChanCount; ++c) {
        const uint32_t readyWord = static_cast<uint32_t>(c) * grid_mock::kScbLineStrideU32;
        const uint32_t freeWord =
            (static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(c)) * grid_mock::kScbLineStrideU32;
        const uint32_t closeWord =
            (2U * static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(c)) * grid_mock::kScbLineStrideU32;
        pipe.readyScb[c] = scbs + readyWord;
        pipe.freeScb[c] = scbs + freeWord;
        pipe.closeScb[c] = scbs + closeWord;
        if (c < Pipe::ChanCount) {
            pipe.slotBase[c] = window + kSlotRegionOffset + c * Pipe::SlotCount * Pipe::SlotStride;
        } else {
            pipe.slotBase[c] = nullptr; // no ring allocated for this channel
        }
        pipe.pushWindow[c] = GridPayloadWindow{};
        pipe.popWindow[c] = GridPayloadWindow{};
    }
    pipe.consHistFull = false;
    // Binding table + consumer history come from the window, not from zero.
    pipe.LoadRecord(scbs + kRecordOffset / sizeof(uint32_t));
    pipe.bindRequestProdIdL1 = reinterpret_cast<__gm__ uint32_t*>(window + kBindRequestOffset);
    pipe.bindRequestProdChanL1 = pipe.bindRequestProdIdL1 + 1;
    pipe.bindResponseReadyL1 = reinterpret_cast<__gm__ uint32_t*>(window + kBindResponseOffset);
    pipe.bindResponseConsChanL1 = pipe.bindResponseReadyL1 + 1;
    pipe.bindResponseCompleteL1 = pipe.bindResponseReadyL1 + 2;
    // Do not clear bindRequestProdIdL1 here.  A producer in an earlier hardware wave
    // may already have deposited a request in this not-yet-scheduled consumer's
    // window.  The host-zeroed +1 encoding arms the mailbox, and the consumer
    // clears each request after accepting it.
    pipe.bcastWindow = GridPayloadWindow{};

    const uint32_t slotRegionBytes = static_cast<uint32_t>(Pipe::ChanCount) * static_cast<uint32_t>(Pipe::SlotCount) *
                                     static_cast<uint32_t>(Pipe::SlotStride);
    uint32_t producerOff = kSlotRegionOffset + slotRegionBytes;

    // TBROADCAST region (scheme-② 真·同时 MPSC).  Only wired when the pipe
    // opted in (GroupMax > 0); a unicast-only pipe leaves these null and pays
    // zero window bytes for broadcast.  Offsets are computed inline from the
    // constexpr variable + the pipe's static members (see the note above).
    if constexpr (Pipe::GroupMax > 0) {
        const uint32_t ringOff = kSlotRegionOffset + slotRegionBytes;
        const uint32_t readyOff =
            ringOff + static_cast<uint32_t>(Pipe::BcastSlotCount) * static_cast<uint32_t>(Pipe::SlotStride);
        const uint32_t freeOff = readyOff + static_cast<uint32_t>(Pipe::GroupMax) *
                                                grid_mock::kBcastLaneStride; // ready region = GroupMax lanes * 64 B
        pipe.bcastRingBase = window + ringOff;
        pipe.bcastReadyLanes = reinterpret_cast<__gm__ uint32_t*>(window + readyOff);
        pipe.bcastFreeLanes = reinterpret_cast<__gm__ uint32_t*>(window + freeOff);
        producerOff = freeOff + static_cast<uint32_t>(Pipe::GroupMax) * grid_mock::kBcastLaneStride;
    }

    // One synchronous outbound transfer uses this slot at a time.  It is appended
    // after all receive-side rings/lanes so a producer can never alias a payload
    // that this same cell is concurrently waiting to consume.
    pipe.producerSlotBase = window + producerOff;
}

// Host-side helper: total bytes per rank for a single GridPipe (broadcast
// region included when the pipe opted in).
template <typename Pipe>
inline constexpr uint32_t WindowBytes()
{
    if constexpr (Pipe::GroupMax > 0) {
        return kWindowBytesWithBcast<
            Pipe::SlotStride, Pipe::SlotCount, Pipe::BcastSlotCount, Pipe::GroupMax, Pipe::ChanCount>();
    } else {
        return kWindowBytes<Pipe::SlotStride, Pipe::SlotCount, Pipe::ChanCount>();
    }
}

} // namespace a2a3_grid
} // namespace pto

#endif // PTO_A2A3_GRID_PIPE_RUNTIME_HPP

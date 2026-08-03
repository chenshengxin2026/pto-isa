/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Grid TPUSH/TPOP model + A2/A3 mock support (A2/A3 backend).
//
// Design_spec: Grid_TPUSH_TPOP_ISA...V8.md (the IPC_SCB scoreboard route).
//
// This header holds ONLY the data model and mock support; the CCE handshake
// intrinsics themselves live in grid_cce_intrinsic.hpp as the V8 two-name facade
// layer (copy_ubuf_to_neighbor_ubuf / sync_hscb / wait_ipc_scb -> __builtin_cce_*).
// There is deliberately NO intermediate PTO wrapper (the old sync_neighbor_scb /
// wait_local_spr / mov_local_spr / ScbOperand / neighbor_sram_addr vocabulary is
// gone, per V8 section 3.4 / section 6 point 4):
//   * Section 1: GridPipe mesh model + neighbor / K-hop resolvers.
//   * Section 2: A2/A3 GM-mock support -- boundary faults + the GmSramArena
//                address-segment model that enforces the NoC "TPOP reads local
//                SRAM only" rule for the GM-window mock.
//
// The GridPipe TPUSH/TPOP overloads in pto/common/pto_instr.hpp and the A2/A3
// backends in GridTPush.hpp / GridTPop.hpp both pull this single header in (which
// in turn pulls grid_cce_intrinsic.hpp).  Compiler-visible static constraints
// are still enforced via static_assert inside the overloads in pto_instr.hpp.

#ifndef PTO_A2A3_GRID_INTRINSIC_HPP
#define PTO_A2A3_GRID_INTRINSIC_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp> // for AICORE (callable from both host and aicore contexts)

#include <pto/npu/a2a3/grid_cce_intrinsic.hpp> // V8 CCE facade layer (the ONLY handshake-intrinsic layer)

// ===========================================================================
// Section 1: GridPipe -- neighbor-core FIFO communication primitives.
//
// This is the proposal-level abstraction described in the V8 design spec (the
// IPC_SCB scoreboard handshake route).  The per-channel FIFO state below is read
// by the GridTPush.hpp / GridTPop.hpp sequence expansions, which call the CCE
// facades in grid_cce_intrinsic.hpp: cross-core notify = sync_hscb (SYNC_HSCB /
// ST_HSCB, a monotone absolute count into the peer's IPC_SCB); local wait =
// wait_ipc_scb (WAIT_SPR, read+block in one instruction; no MOV_SPR2X peek --
// V8); payload = copy_ubuf_to_neighbor_ubuf (COPY_UBUF_TO_NBR).
// On A2/A3 there is no cross-core neighbor-IPC_SCB addressing (V8 HW-DEP-1) nor a
// UB->neighbor-UB write (V8 HW-DEP-0), so those facades run their GM mock and
// Section 2 stands in for the IPC_SCB scoreboards with HCCL shared windows and
// GM words.
//
// ---------------------------------------------------------------------------
// ONE PIPE == ONE CHANNEL == ONE (producer, consumer) PAIR.
// ---------------------------------------------------------------------------
// Everything a pipe carries is core-local REGISTER state on real silicon:
// ready_scb / free_scb are IPC_SCB slots (SPR) and prod_idx / cons_idx are GPR
// run-counters.  A register cannot be re-pointed at a different peer mid-flight,
// so the peers a pipe talks to are part of its TYPE: a unicast GridPipe is bound
// to (Dir, Dist) -- its producer is the core Dist hops UPSTREAM along Dir and its
// consumer the core Dist hops DOWNSTREAM -- and a GridGroupPipe is bound to a
// GridGroup.  Talking to a different producer or a different consumer means
// DECLARING A DIFFERENT PIPE, exactly as a5's TPipe binds its FlagID + Direction
// at the type level.  (The GM mock could physically index another peer's window,
// but designing to the mock's freedom instead of the register model is what let
// one pipe object silently multiplex five directions before.)
//
// The state splits into the three groups the pipes are assembled from:
//   ctx   (GridPipeCtx)     -- the runtime context INSTANCE plus this core's
//                              place in the mesh: everything needed to ADDRESS
//                              the peer.  The one group that is identical
//                              across every pipe on a core.
//   slots (GridSlotRing)    -- the payload ring: base address, one-slot stride,
//                              ring depth.
//   prod  (GridProducerSem) -- the producer semaphore: free_scb + prod_idx.
//   cons  (GridConsumerSem) -- the consumer semaphore: ready_scb + cons_idx.
// ===========================================================================

// Forward declaration: provided by the target backend (cpu_stub.hpp on
// CPU sim builds, CCE intrinsic / runtime header on A2/A3 NPU builds).
// GetGridCoord below uses this; we declare it here so this header can be
// included before any backend headers without triggering an undeclared name.
uint32_t get_block_idx();

namespace pto {

// ---------------------------------------------------------------------------
// 2D mesh coordinates and shape (design doc section 2).
// ---------------------------------------------------------------------------
struct GridShape {
    int gridRows = 0; // N
    int gridCols = 0; // M
};

struct GridCoord {
    int row = 0; // 0 .. gridRows-1
    int col = 0; // 0 .. gridCols-1
};

// Half-open sub-rectangle [row0, row1) x [col0, col1) describing an arbitrary
// rectangular group member set (GridGroup::SUBRECT).  ROW / COL are the special
// cases rect = {r, r+1, 0, cols} / {0, rows, c, c+1}; a general SUBRECT lets a
// single TBROADCAST reach every cell in the rectangle (any-to-any via the mock's
// logical-rank window addressing, including diagonal / far peers).
struct GridRect {
    int row0 = 0;
    int row1 = 0;
    int col0 = 0;
    int col1 = 0;
};

// ---------------------------------------------------------------------------
// Direction enum (design doc section 3.1).  Strongly-typed to avoid clashing
// with the cluster-local pto::Direction enum used by TPipe.
// ---------------------------------------------------------------------------
enum class GridDirection : uint8_t {
    SOURCE = 0, // GM/Host/Runtime injection.  Only valid for TPOP.
    NORTH = 1,  // row -> row-1
    EAST = 2,   // col -> col+1
    WEST = 3,   // col -> col-1
    SOUTH = 4,  // row -> row+1
};

inline constexpr int kGridDirectionCount = 5;

AICORE constexpr int GridDirectionIndex(GridDirection d) { return static_cast<int>(d); }

// ---------------------------------------------------------------------------
// Coordinate bootstrap (design doc 2.1).  Row-major mapping from launcher's
// block_idx to (row, col).  AICORE-qualified because it calls get_block_idx(),
// which is a device intrinsic and has no host implementation.
// ---------------------------------------------------------------------------
AICORE inline GridCoord GetGridCoord(GridShape shape)
{
    int blockIdx = static_cast<int>(get_block_idx());
    return GridCoord{blockIdx / shape.gridCols, blockIdx % shape.gridCols};
}

AICORE constexpr int RankFromCoord(GridCoord coord, GridShape shape) { return coord.row * shape.gridCols + coord.col; }

// ---------------------------------------------------------------------------
// Compile-time / run-time direction validity (design doc 2.3).
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPush(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row > 0;
        case GridDirection::EAST:
            return c.col + 1 < s.gridCols;
        case GridDirection::WEST:
            return c.col > 0;
        case GridDirection::SOUTH:
            return c.row + 1 < s.gridRows;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr bool CanPop(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + 1 < s.gridRows;
        case GridDirection::EAST:
            return c.col > 0;
        case GridDirection::WEST:
            return c.col + 1 < s.gridCols;
        case GridDirection::SOUTH:
            return c.row > 0;
        case GridDirection::SOURCE:
            return true;
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPush(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + 1};
        case GridDirection::WEST:
            return {c.row, c.col - 1};
        case GridDirection::SOUTH:
            return {c.row + 1, c.col};
        case GridDirection::SOURCE:
            return c; // Unused; static_assert blocks TPUSH<SOURCE>.
    }
    return c;
}

AICORE constexpr GridCoord NeighborForPop(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - 1};
        case GridDirection::WEST:
            return {c.row, c.col + 1};
        case GridDirection::SOUTH:
            return {c.row - 1, c.col};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

inline constexpr int kInvalidRank = -1;

AICORE constexpr int NeighborRankForPush(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPush(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPush(dir, c);
    return n.row * s.gridCols + n.col;
}

AICORE constexpr int NeighborRankForPop(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPop(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPop(dir, c);
    return n.row * s.gridCols + n.col;
}

// ---------------------------------------------------------------------------
// Multi-hop (routed K-hop unicast) generalisation of the neighbor resolvers.
//
// Scheme A: a K-hop *unicast* push keeps the receiver's per-channel slot/flag
// state at fan-in 1, so distance enters only the *target rank* (and the
// doorbell reach), never the buffer count.  A K-hop push is therefore the
// 1-hop expansion with "+1/-1" replaced by "+K/-K"; nothing in the GridPipe
// window layout changes.  The GridKHopSelfCheck() static_assert below pins
// k == 1 to the existing CanPush/NeighborRankForPush/CanPop/NeighborRankForPop
// behaviour so the default-distance (= 1) path stays byte-identical.
//
// Precondition (caller's responsibility): within one direction and phase, at
// most one (source, distance) pair targets a given receiver, i.e. fan-in <= 1.
// Concurrent multi-source receive (gather/multicast) is out of scope here and
// needs the fan-in-indexed channel layout (Scheme B).
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row - k >= 0;
        case GridDirection::EAST:
            return c.col + k < s.gridCols;
        case GridDirection::WEST:
            return c.col - k >= 0;
        case GridDirection::SOUTH:
            return c.row + k < s.gridRows;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPushK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + k};
        case GridDirection::WEST:
            return {c.row, c.col - k};
        case GridDirection::SOUTH:
            return {c.row + k, c.col};
        case GridDirection::SOURCE:
            return c; // Unused; TPUSH<SOURCE> is blocked by static_assert.
    }
    return c;
}

AICORE constexpr int RankForPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPushK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPushK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Consumer side: the producer that fed a `dir` channel sits k hops in the
// *opposite* direction (an EAST channel is fed from the WEST, etc).  Used by
// TPOP to route the free-credit doorbell back to the K-hop producer.
AICORE constexpr bool CanPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + k < s.gridRows; // upstream to the south
        case GridDirection::EAST:
            return c.col - k >= 0; // upstream to the west
        case GridDirection::WEST:
            return c.col + k < s.gridCols; // upstream to the east
        case GridDirection::SOUTH:
            return c.row - k >= 0; // upstream to the north
        case GridDirection::SOURCE:
            return true; // SOURCE pop is bound to the runtime queue, distance-free.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPopK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - k};
        case GridDirection::WEST:
            return {c.row, c.col + k};
        case GridDirection::SOUTH:
            return {c.row - k, c.col};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

AICORE constexpr int RankForPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPopK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPopK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Compile-time pin: the K-hop resolvers must collapse to the 1-hop neighbor
// resolvers at k == 1, so existing TPUSH<DIR>/TPOP<DIR> behaviour (Dist == 1)
// is preserved bit-for-bit.  A representative 2-hop case is also checked.
AICORE constexpr bool GridKHopSelfCheck()
{
    GridShape s{4, 4};
    GridCoord c{2, 2};
    bool ok = true;
    // k == 1 reproduces the 1-hop resolvers for every real direction.
    ok = ok && (CanPushK(GridDirection::NORTH, c, s, 1) == CanPush(GridDirection::NORTH, c, s));
    ok = ok && (CanPushK(GridDirection::EAST, c, s, 1) == CanPush(GridDirection::EAST, c, s));
    ok = ok && (CanPushK(GridDirection::WEST, c, s, 1) == CanPush(GridDirection::WEST, c, s));
    ok = ok && (CanPushK(GridDirection::SOUTH, c, s, 1) == CanPush(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPushK(GridDirection::NORTH, c, s, 1) == NeighborRankForPush(GridDirection::NORTH, c, s));
    ok = ok && (RankForPushK(GridDirection::EAST, c, s, 1) == NeighborRankForPush(GridDirection::EAST, c, s));
    ok = ok && (RankForPushK(GridDirection::WEST, c, s, 1) == NeighborRankForPush(GridDirection::WEST, c, s));
    ok = ok && (RankForPushK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPush(GridDirection::SOUTH, c, s));
    ok = ok && (CanPopK(GridDirection::NORTH, c, s, 1) == CanPop(GridDirection::NORTH, c, s));
    ok = ok && (CanPopK(GridDirection::EAST, c, s, 1) == CanPop(GridDirection::EAST, c, s));
    ok = ok && (CanPopK(GridDirection::WEST, c, s, 1) == CanPop(GridDirection::WEST, c, s));
    ok = ok && (CanPopK(GridDirection::SOUTH, c, s, 1) == CanPop(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPopK(GridDirection::NORTH, c, s, 1) == NeighborRankForPop(GridDirection::NORTH, c, s));
    ok = ok && (RankForPopK(GridDirection::EAST, c, s, 1) == NeighborRankForPop(GridDirection::EAST, c, s));
    ok = ok && (RankForPopK(GridDirection::WEST, c, s, 1) == NeighborRankForPop(GridDirection::WEST, c, s));
    ok = ok && (RankForPopK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPop(GridDirection::SOUTH, c, s));
    // Representative 2-hop case on a 1x4 row: col0 --EAST,2--> col2, popped at col2.
    GridShape row{1, 4};
    ok = ok && CanPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2);
    ok = ok && (RankForPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2) == 2);
    ok = ok && !CanPushK(GridDirection::EAST, GridCoord{0, 2}, row, 2); // 2+2 == 4 out of range
    ok = ok && CanPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2);
    ok = ok && (RankForPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2) == 0);
    return ok;
}
static_assert(GridKHopSelfCheck(), "GridPipe K-hop resolver self-test failed");

// ---------------------------------------------------------------------------
// Broadcast GROUP -- the participant set of a TBROADCAST collective.
//
// GROUP replaces the old single-source GridSpan "span" and, with it, the
// handshake model: a TBROADCAST is no longer one source multicasting to a
// fan-in-1 span (which forbade concurrent senders).  It is a 真·同时 MPSC
// channel (see Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移): every
// member of the GROUP may broadcast its own shard into each receiver's shared
// ring *concurrently*.  The prefix-offset assignment (each source owns a
// disjoint global-index interval) plus per-source ready lanes keep every
// physical edge SPSC, so K concurrent senders never clobber a shared counter.
//
//   GridGroup::ROW = every cell on the source's row    (the row is the group)
//   GridGroup::COL = every cell on the source's column (the column is the group)
//
// A group still decomposes into two opposite 1-D arms for topology description
// -- ROW = EAST+WEST, COL = NORTH+SOUTH (GroupArmA / GroupArmB) -- but the
// prefix-offset send addresses peers by their rank-in-group directly, not by
// arm, so a receiver drains member `srcRank` with TPOP(pipe, tile, srcRank)
// regardless of which arm it sits on.  The group is bound to the pipe TYPE
// (GridGroupPipe below), because switching groups switches every peer.
// ---------------------------------------------------------------------------
enum class GridGroup : uint8_t {
    ROW = 0,     // group = the source's row:    EAST arm + WEST arm
    COL = 1,     // group = the source's column: NORTH arm + SOUTH arm
    SUBRECT = 2, // group = an arbitrary sub-rectangle [row0,row1)x[col0,col1)
                 //          (runtime-described via the pipe's `rect`).  It subsumes
                 //          ROW/COL and needs no arm decomposition: members are
                 //          addressed by rank-in-rect directly.
};

// The two opposite GridDirections a group decomposes into (topology only).
// constexpr so they fold into non-type template arguments where useful.
AICORE constexpr GridDirection GroupArmA(GridGroup g)
{
    return g == GridGroup::ROW ? GridDirection::EAST : GridDirection::NORTH;
}

AICORE constexpr GridDirection GroupArmB(GridGroup g)
{
    return g == GridGroup::ROW ? GridDirection::WEST : GridDirection::SOUTH;
}

// ---------------------------------------------------------------------------
// Scheme-② prefix-offset helpers.  Every member contributes a statically-known
// count (1 shard for the AllGather demo), so the prefix-offset base of member k
// is just k (computed locally under SPMD -- variant a, zero atomic).  These
// helpers map between a member's rank-in-group, its grid coordinate, and the
// global index space the shared ring is addressed by (slot = gidx % SC).
// ---------------------------------------------------------------------------
// Number of members in the group that `coord` belongs to.  The trailing
// `rect` is consulted only for SUBRECT (ROW/COL ignore it); defaulting it keeps
// every existing ROW/COL call site unchanged.
AICORE constexpr int GridGroupSize(GridGroup g, GridShape s, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? s.gridCols :
           (g == GridGroup::COL) ? s.gridRows :
                                   ((rect.row1 - rect.row0) * (rect.col1 - rect.col0));
}

// This cell's rank within its group = its prefix-offset base (count_k = 1).
// ROW groups vary along the column axis; COL groups along the row axis; SUBRECT
// uses a row-major rank within [row0,row1)x[col0,col1).
AICORE constexpr int RankInGroup(GridGroup g, GridCoord c, GridRect rect = {})
{
    return (g == GridGroup::ROW) ? c.col :
           (g == GridGroup::COL) ? c.row :
                                   ((c.row - rect.row0) * (rect.col1 - rect.col0) + (c.col - rect.col0));
}

// Coordinate of the member whose rank-in-group is `rankInGroup`, given this
// cell's coordinate (the member shares this cell's fixed axis for ROW/COL).
// SUBRECT inverts the row-major rank entirely from `rect` (self-independent).
AICORE constexpr GridCoord GroupMemberCoord(GridGroup g, GridCoord self, int rankInGroup, GridRect rect = {})
{
    if (g == GridGroup::ROW) {
        return GridCoord{self.row, rankInGroup};
    }
    if (g == GridGroup::COL) {
        return GridCoord{rankInGroup, self.col};
    }
    const int colSpan = rect.col1 - rect.col0;
    return GridCoord{rect.row0 + rankInGroup / colSpan, rect.col0 + rankInGroup % colSpan};
}

AICORE constexpr int GroupMemberRank(GridGroup g, GridCoord self, GridShape s, int rankInGroup, GridRect rect = {})
{
    return RankFromCoord(GroupMemberCoord(g, self, rankInGroup, rect), s);
}

// Owner (rank-in-group) of global index `gidx`.  With count_k = 1 the prefix
// partition is the identity, so owner(gidx) = gidx; general variable-count
// partitions would replace this with a prefix-sum lookup (variant a) or an
// atomic-add reservation (variant b).  Kept as a named function so the
// directed-free path reads as the design doc states it ("owner(c + SC)").
AICORE constexpr int GroupOwnerOfIndex(int gidx) { return gidx; }

// ---------------------------------------------------------------------------
// GridPayloadWindow -- the sub-window of a slot that one TPUSH/TPOP actually
// moves.  This is the GridPipe equivalent of a5 TPipe's `entryOffset` plus the
// shape/stride pair its TSTORE/TLOAD descriptors carry (a5 TPush.hpp:78/274/289):
// the SLOT STRIDE (Pipe::SlotStride) addresses the ring, while the fields below
// describe the transfer.  Previously both were the single constant SlotBytes, so
// every push moved a whole slot even when only a prefix was valid.
//
// One window belongs to ONE semaphore side -- the producer's lives in
// GridProducerSem, the consumer's in GridConsumerSem -- mirroring a5's
// Producer::entryOffset / Consumer::entryOffset.
//
//   entryOffset  byte offset of the sub-window inside the slot
//   rowBytes     bytes moved per row
//   rowCount     number of rows; 0 DISABLES the window (whole slot, 1-D,
//                Pipe::SlotStride bytes at offset 0 -- the original behaviour)
//   tileStride   byte stride between rows in the local tile (0 => rowBytes)
//   slotStride   byte stride between rows inside the slot   (0 => rowBytes)
//
// The strides are named by WHICH BUFFER they walk, not by src/dst, because the
// two swap roles between the halves: a push reads the tile and writes the slot, a
// pop reads the slot and writes the tile.  (`slotStride` is the per-row stride
// INSIDE one slot; the ring's slot-to-slot stride is Pipe::SlotStride.)
//
// rowCount > 1 expresses a 2-D sub-block (e.g. the valid column prefix of a
// row-major tile).  The COPY_UBUF_TO_NBR machine instruction takes a single
// `bytes` operand, so the lowering emits one burst per row and ONE ready
// doorbell for the whole window -- the doorbell count per TPUSH is unchanged.
// Hardware that grows src/dst stride operands can fold the loop into one burst.
// ---------------------------------------------------------------------------
struct GridPayloadWindow {
    uint32_t entryOffset = 0;
    uint32_t rowBytes = 0;
    uint32_t rowCount = 0; // 0 => disabled: whole slot
    uint32_t tileStride = 0;
    uint32_t slotStride = 0;
};

AICORE inline uint32_t GridPayloadTileStride(const GridPayloadWindow& w)
{
    return w.tileStride != 0 ? w.tileStride : w.rowBytes;
}

AICORE inline uint32_t GridPayloadSlotStride(const GridPayloadWindow& w)
{
    return w.slotStride != 0 ? w.slotStride : w.rowBytes;
}

// Bytes spanned inside the slot, measured from the SLOT base (entryOffset
// included).  A disabled window spans the whole slot.  This is what the range
// guard compares against SlotStride.
AICORE inline uint32_t GridPayloadSlotExtent(const GridPayloadWindow& w, uint32_t slotStride)
{
    if (w.rowCount == 0) {
        return slotStride;
    }
    return w.entryOffset + (w.rowCount - 1) * GridPayloadSlotStride(w) + w.rowBytes;
}

// ===========================================================================
// The three groups of state a pipe binds (see the section header).
// ===========================================================================

// ---------------------------------------------------------------------------
// (1) Runtime-context group -- how a peer is ADDRESSED.
//
// `runtimeCtx` is the opaque global context INSTANCE the A2/A3 backend hands to
// ResolvePeerSlotAddr / RemoteScbPtr to turn a local address into the same byte
// offset in a peer's window (HCCL device context here; other targets may
// reinterpret).  `shape` / `coord` place this core in the mesh, which is what
// turns the pipe's bound (Dir, Dist) -- or its group -- into a concrete peer
// rank.  This group is the same for every pipe on a core; each pipe holds its
// own copy so no pipe operation needs a second argument.
// ---------------------------------------------------------------------------
struct GridPipeCtx {
    __gm__ void* runtimeCtx = nullptr; // global context instance (peer address resolution)
    GridShape shape{};                 // mesh shape (design doc 2.1)
    GridCoord coord{};                 // this core's cell
    uint32_t pipeId = 0;               // stable logical id (runtime telemetry)
};

// ---------------------------------------------------------------------------
// (2) Slot group -- the payload ring.
//
// `base` is the ring base inside THIS core's window; SlotStride is the ring
// ADDRESSING stride (one slot's size) and SlotCount its depth.  The transfer
// LENGTH is NOT SlotStride -- it comes from the semaphore-side
// GridPayloadWindow, or defaults to the whole slot.
// ---------------------------------------------------------------------------
template <int SlotStride_, int SlotCount_>
struct GridSlotRing {
    static_assert(SlotStride_ > 0, "GridSlotRing requires SlotStride > 0");
    static_assert(SlotCount_ > 0, "GridSlotRing requires SlotCount > 0");

    static constexpr int SlotStride = SlotStride_; // one slot's size (ring addressing stride)
    static constexpr int SlotCount = SlotCount_;   // ring depth

    __gm__ uint8_t* base = nullptr; // payload ring base [SlotCount * SlotStride]

    // Ring addressing: slot of the absolute index `idx` (prod_idx / cons_idx /
    // the group's global index).
    AICORE uint32_t SlotOffset(uint32_t idx) const
    {
        return (idx % static_cast<uint32_t>(SlotCount)) * static_cast<uint32_t>(SlotStride);
    }
    AICORE __gm__ uint8_t* Slot(uint32_t idx) const { return base + SlotOffset(idx); }
};

// ---------------------------------------------------------------------------
// (3) Semaphore group -- one struct per side of the channel.
//
// On native silicon `freeScb` / `readyScb` are IPC_SCB slots (SPR) carrying a
// monotone absolute count written by the single peer on the other side of the
// edge (an HSCB store) and read/blocked-on locally, while `prodIndex` /
// `consIndex` are GPR run-counters (slot address, wait threshold, and the
// absolute count published to the peer) that never live in an IPC_SCB.  In this
// A2/A3 mock the scoreboards are GM words standing in for those slots.
//
// Both sides live in one pipe because under SPMD one core plays both roles on a
// channel: it consumes what its upstream produced and produces for its
// downstream.  They are separate structs because they are separate registers
// bound to separate peers -- exactly a5 TPipe's `prod` / `cons` pair.
// ---------------------------------------------------------------------------
// `installScb` / `openScb` are the 接力计数 (relay-counting) handoff DOORBELLS and
// `batonL1` the relayed prod_idx itself; all three are idle in steady state and
// touched only by THANDOFF (see GridTHandoff.hpp).  The doorbells follow the SAME
// ownership rule as free/ready -- a producer-side word is written by its consumer
// peer and a consumer-side one by its producer peer, so every word keeps a single
// external writer (C1).
//
// `batonL1` is the 接力棒 -- the one value a handoff has to physically carry from
// the retiring channel to its successor: the prod_idx the incoming producer must
// start counting from.  Everything else the successor needs is already in place
// (the ring and its contents never move, and free credit is an ordinary scoreboard
// store), so this single word IS the handoff's payload.
//
// Note what it is NOT: an L1/SRAM word, not an IPC_SCB, and with no slot number.  A
// scoreboard is a thing you WAIT on (WAIT_SPR compares inside the instruction); the
// baton is a thing you MOVE -- out of the retiring channel's ready_scb by MOV_SPR2X,
// across by ST_HSCB, and into the successor's prod_idx GPR by MOV_L12X.  Keeping it
// out of the scoreboard file is also what holds the handoff's IPC_SCB budget down to
// the two doorbells.
struct GridProducerSem {
    __gm__ uint32_t* freeScb = nullptr;    // IPC_SCB (SPR): free credit, written by the consumer peer
    __gm__ uint32_t* installScb = nullptr; // IPC_SCB (SPR): INSTALL_BASE doorbell (handoff generation), ditto
    __gm__ uint32_t* batonL1 = nullptr;    // L1 word: prod_idx baseline DELIVERED here by the consumer's ST_HSCB
    uint32_t prodIndex = 0;                // GPR: absolute count of tiles pushed
    GridPayloadWindow window{};            // sub-window this side moves (a5: Producer::entryOffset)
};

struct GridConsumerSem {
    __gm__ uint32_t* readyScb = nullptr; // IPC_SCB (SPR): ready count, written by the producer peer
    __gm__ uint32_t* openScb = nullptr;  // IPC_SCB (SPR): OPEN_ACK (handoff generation), written by the producer peer
    __gm__ uint32_t* batonL1 = nullptr;  // L1 word: MOV_SPR2X drops the outgoing baton here for ST_HSCB to forward
    uint32_t consIndex = 0;              // GPR: absolute count of tiles popped
    GridPayloadWindow window{};          // sub-window this side moves (a5: Consumer::entryOffset)
};

// Group-collective (MPSC) counterparts.  The single ready/free word becomes an
// ARRAY of per-source lanes indexed by rank-in-group: variant-B ready lanes (one
// writer each -- the source of that rank -- so every lane is SPSC and K
// concurrent senders never clobber a shared counter) and free lanes (this core,
// as the single consumer of its own ring, is the sole writer of each, so the
// free direction is SPSC too -- no min-credit tree needed, design doc §7.4).
// There is no prod/cons GPR counter: with count_k = 1 the global index of a
// member IS its rank-in-group, so the index is derived, not run.
struct GridGroupProducerSem {
    __gm__ uint32_t* freeLanes = nullptr; // [GroupMax] per-source free lanes (this core writes peers')
    GridPayloadWindow window{};
};

struct GridGroupConsumerSem {
    __gm__ uint32_t* readyLanes = nullptr; // [GroupMax] per-source ready lanes (variant B)
    GridPayloadWindow window{};
};

// ---------------------------------------------------------------------------
// GridPipe<TileT, Dir, SlotStride, SlotCount, Dist = 1, ScbId = 2*Dir>
//
// One SPSC unicast channel of the mesh, bound to the peers it talks to:
//   producer peer = the core `Dist` hops UPSTREAM   along Dir (feeds cons)
//   consumer peer = the core `Dist` hops DOWNSTREAM along Dir (fed by prod)
// A TPUSH publishes into the consumer peer's ring; a TPOP drains this core's own
// ring, which its producer peer filled.  Both ends of the same logical edge
// declare the same pipe type, which is what makes the offsets line up under
// SPMD.
//
// Reusing one pipe for a second direction/distance is NOT possible by
// construction -- that would re-point core-local SPR/GPR state at another peer.
// Declare a second pipe instead (e.g. an EAST pipe and a WEST pipe for a
// bidirectional relay) and give it its own window region.
//
// `ScbId` names the FIRST of the two IPC_SCB slots the pipe occupies (ready =
// ScbId, free = ScbId+1), the same resource-allocation knob as a5 TPipe's
// FlagID.  It defaults to 2*Dir, which reproduces "one scoreboard pair per
// direction"; two pipes that share a direction on one core must be given
// distinct ids.  The A2/A3 mock reads the GM word and ignores the slot number;
// native WAIT_SPR uses it.
//
// A pipe that takes part in a 接力计数 producer handoff (THANDOFF) additionally
// occupies ScbId+2..ScbId+3 -- the install / open doorbells below.  Those slots
// are idle in steady state, so the budget is charged where it is spent: the
// static_assert here still only requires ScbId+1 < 16, and GridTHandoff.hpp
// asserts ScbId+3 < 16 for the pipes actually handed off.  The two pipes on
// either side of a handoff are one physical channel, so they must be declared
// with the SAME explicit ScbId (the 2*Dir default differs per direction).
// ---------------------------------------------------------------------------
template <
    typename TileT_, GridDirection Dir_, int SlotStride_, int SlotCount_, int Dist_ = 1,
    int ScbId_ = 2 * static_cast<int>(Dir_)>
struct GridPipe {
    static_assert(Dist_ >= 1, "GridPipe requires Dist >= 1 (routed K-hop unicast)");
    static_assert(ScbId_ >= 0 && ScbId_ + 1 < 16, "GridPipe occupies IPC_SCB slots ScbId and ScbId+1 (0..15)");

    using TileType = TileT_;
    using Ring = GridSlotRing<SlotStride_, SlotCount_>;

    // Bound channel identity.
    static constexpr GridDirection Dir = Dir_;
    static constexpr int Dist = Dist_;
    // Ring geometry, re-exported so call sites can keep saying Pipe::SlotStride.
    static constexpr int SlotStride = Ring::SlotStride;
    // Compatibility spelling of the same constant.  Reads as "one slot is this
    // many bytes"; kept so existing call sites and window mirrors keep working.
    static constexpr int SlotBytes = Ring::SlotStride;
    static constexpr int SlotCount = Ring::SlotCount;
    // IPC_SCB slot pair (native WAIT_SPR operand; ignored by the GM mock).
    static constexpr uint32_t ReadyScbSlot = static_cast<uint32_t>(ScbId_);
    static constexpr uint32_t FreeScbSlot = static_cast<uint32_t>(ScbId_) + 1;
    // 接力计数 handoff doorbells -- reserved only for pipes that take part in a
    // THANDOFF (see the ScbId note above).  The relayed prod_idx itself needs no
    // slot: it travels through L1 (GridProducerSem::batonL1).
    static constexpr uint32_t InstallScbSlot = static_cast<uint32_t>(ScbId_) + 2;
    static constexpr uint32_t OpenScbSlot = static_cast<uint32_t>(ScbId_) + 3;

    GridPipeCtx ctx{};      // (1) how to address the peer
    Ring slots{};           // (2) payload ring
    GridProducerSem prod{}; // (3) free_scb + prod_idx   (this core -> consumer peer)
    GridConsumerSem cons{}; // (3) ready_scb + cons_idx  (producer peer -> this core)

    // --- bound peers ------------------------------------------------------
    // Rank of the peer that CONSUMES what this core pushes (kInvalidRank off-mesh).
    AICORE int ConsumerRank() const { return RankForPushK(Dir, ctx.coord, ctx.shape, Dist); }
    // Rank of the peer that PRODUCES what this core pops (kInvalidRank off-mesh).
    AICORE int ProducerRank() const { return RankForPopK(Dir, ctx.coord, ctx.shape, Dist); }
    AICORE bool HasConsumer() const { return CanPushK(Dir, ctx.coord, ctx.shape, Dist); }
    AICORE bool HasProducer() const { return CanPopK(Dir, ctx.coord, ctx.shape, Dist); }
    AICORE int SelfRank() const { return RankFromCoord(ctx.coord, ctx.shape); }

    // --- payload sub-windows (a5 TPipe's prod/cons entryOffset) ------------
    // All zero = disabled = move the whole slot, which is what every call site
    // did before these existed.  Set them right before the TPUSH/TPOP they apply
    // to; they persist until reset.
    AICORE void SetPushWindow(const GridPayloadWindow& w) { prod.window = w; }
    AICORE void SetPopWindow(const GridPayloadWindow& w) { cons.window = w; }
    AICORE void ResetPushWindow() { prod.window = GridPayloadWindow{}; }
    AICORE void ResetPopWindow() { cons.window = GridPayloadWindow{}; }
};

// 接力计数 (relay counting) across a time-division producer handoff is spelled out
// in GridTHandoff.hpp; the state it needs is already here.  There is deliberately
// no baton STRUCT: the two counters it would hold are `cons.readyScb` (which
// already holds E, the retiring producer's final prod_idx, put there by its last
// SYNC_HSCB(READY)) and `cons.consIndex` -- both live values of the pipe being
// retired, so wrapping them in a side structure would only duplicate them.

// ---------------------------------------------------------------------------
// GridGroupPipe<TileT, Group, SlotStride, SlotCount, GroupMax>
//
// One MPSC group-collective channel (scheme-② 真·同时 MPSC), bound to the GROUP
// whose members are its producers and consumers.  Every member may broadcast
// concurrently, so the single ready/free scoreboard of the unicast pipe becomes
// per-source lane arrays; the ring is SHARED by the whole group and addressed by
// the GLOBAL index gidx (slot = gidx % SlotCount), each source owning a disjoint
// prefix-offset interval.
//
// Switching to another group switches every peer, so the group is part of the
// type: the two phases of a 2-D AllGather (a ROW group then a COL group) are two
// pipes, not one pipe used twice.
// ---------------------------------------------------------------------------
template <typename TileT_, GridGroup Group_, int SlotStride_, int SlotCount_, int GroupMax_>
struct GridGroupPipe {
    static_assert(GroupMax_ > 0, "GridGroupPipe requires GroupMax > 0");

    using TileType = TileT_;
    using Ring = GridSlotRing<SlotStride_, SlotCount_>;

    // Bound channel identity.
    static constexpr GridGroup Group = Group_;
    static constexpr int GroupMax = GroupMax_; // lanes reserved per source (one ready + one free each)
    static constexpr int SlotStride = Ring::SlotStride;
    static constexpr int SlotBytes = Ring::SlotStride;
    static constexpr int SlotCount = Ring::SlotCount; // shared-ring depth (SC)

    GridPipeCtx ctx{};           // (1) how to address the peer
    GridRect rect{};             // member set for SUBRECT (ROW/COL ignore it)
    Ring slots{};                // (2) shared MPSC payload ring
    GridGroupProducerSem prod{}; // (3) per-source free lanes
    GridGroupConsumerSem cons{}; // (3) per-source ready lanes

    // --- bound peers ------------------------------------------------------
    AICORE int GroupSize() const { return GridGroupSize(Group, ctx.shape, rect); }
    // This core's rank-in-group == its prefix-offset base (count_k = 1).
    AICORE int SelfGroupRank() const { return pto::RankInGroup(Group, ctx.coord, rect); }
    // Global rank of the member whose rank-in-group is `rankInGroup`.
    AICORE int MemberRank(int rankInGroup) const
    {
        return GroupMemberRank(Group, ctx.coord, ctx.shape, rankInGroup, rect);
    }
    AICORE int SelfRank() const { return RankFromCoord(ctx.coord, ctx.shape); }

    // --- payload sub-window -----------------------------------------------
    // In a group collective both halves move the same geometry (a source
    // replicates its own shard, a receiver drains another source's), so the
    // convenience setter writes both sides at once.
    AICORE void SetWindow(const GridPayloadWindow& w)
    {
        prod.window = w;
        cons.window = w;
    }
    AICORE void ResetWindow()
    {
        prod.window = GridPayloadWindow{};
        cons.window = GridPayloadWindow{};
    }
};

// ---------------------------------------------------------------------------
// SFINAE markers: let pto_instr.hpp's TPUSH/TPOP/TREDUCE/TBROADCAST grid
// overloads disambiguate against the existing TPipe overloads, and separate the
// unicast pipe from the group pipe.
// ---------------------------------------------------------------------------
template <typename T>
struct is_grid_pipe : std::false_type {};

template <typename TileT, GridDirection Dir, int SlotStride, int SlotCount, int Dist, int ScbId>
struct is_grid_pipe<GridPipe<TileT, Dir, SlotStride, SlotCount, Dist, ScbId>> : std::true_type {};

template <typename T>
inline constexpr bool is_grid_pipe_v = is_grid_pipe<std::remove_reference_t<T>>::value;

template <typename T>
struct is_grid_group_pipe : std::false_type {};

template <typename TileT, GridGroup Group, int SlotStride, int SlotCount, int GroupMax>
struct is_grid_group_pipe<GridGroupPipe<TileT, Group, SlotStride, SlotCount, GroupMax>> : std::true_type {};

template <typename T>
inline constexpr bool is_grid_group_pipe_v = is_grid_group_pipe<std::remove_reference_t<T>>::value;

// Either flavour -- used where an overload must simply step aside for any grid
// pipe (e.g. the reversed-argument TPUSH/TPOP overloads of the TPipe family).
template <typename T>
inline constexpr bool is_any_grid_pipe_v = is_grid_pipe_v<T> || is_grid_group_pipe_v<T>;

} // namespace pto

// ===========================================================================
// Section 2: A2/A3 GM-mock support -- boundary-fault sentinels.
//
// The IPC_SCB / HSCB handshake mock now lives in the CCE facades themselves
// (grid_cce_intrinsic.hpp: sync_hscb / wait_ipc_scb GM branches).
// What remains here is purely the mock's out-of-mesh fault reporting: a TPUSH /
// TPOP whose (dir,dist) target leaves the mesh writes a sentinel GM word that
// the host launcher polls after each kernel.  Real silicon raises a hardware
// fault instead; these have no V8 machine-instruction counterpart.
// ===========================================================================

namespace pto {
namespace grid_mock {

#ifndef PTO_GRID_MOCK_WFE_MAX_SPINS
#define PTO_GRID_MOCK_WFE_MAX_SPINS 100000000U
#endif

inline constexpr uint32_t kDefaultWfeMaxSpins = PTO_GRID_MOCK_WFE_MAX_SPINS;
// Fault sentinel, in u32 words from the scoreboard (or lane) it belongs to.  A
// unicast pipe's ready/free scbs are words 0/1 of its window, so the sentinels
// land on words 10/11 -- inside the reserved flag header the host scans, and
// clear of both scoreboards.  For a broadcast lane (64 B apart) it is word 10 of
// that lane's own cache line.
inline constexpr uint32_t kFaultFlagWordOffset = 10;

// ONE CACHE LINE PER INDEPENDENTLY-WRITTEN SCOREBOARD.
//
// AICORE caches are not coherent between cores and the mock's sync_hscb store
// commits through a line-granular dcci write-back, so a core that stores into
// one word of a line writes back the WHOLE line from its own (possibly stale)
// copy.  Two DIFFERENT cores storing into two words of the SAME line therefore
// lose each other's updates: the doorbell simply never appears, and the peer
// blocks forever on a threshold that was already met.
//
// So every word with its own external writer gets its own cache line.  This was
// first hit on the TBROADCAST per-source lanes (GroupMax doorbells packed into
// one 64 B line, "wait ready timeout"), and it applies verbatim to the unicast
// window's scoreboards: ready is written by the producer peer, free / base /
// install by the consumer peer, open by the (new) producer peer -- up to four
// distinct writers into what used to be a single line.
inline constexpr uint32_t kScbLineStride = 64;                                       // bytes; one scoreboard per line
inline constexpr uint32_t kScbLineStrideU32 = kScbLineStride / sizeof(uint32_t);     // == 16 (u32 step per scoreboard)
inline constexpr uint32_t kBcastLaneStride = kScbLineStride;                         // TBROADCAST lanes: same rule
inline constexpr uint32_t kBcastLaneStrideU32 = kBcastLaneStride / sizeof(uint32_t); // == 16 (u32 step per lane)

// The SYNC_HSCB / WAIT_SPR mocks that used to live here are now the GM-mock
// branches of the CCE facades in grid_cce_intrinsic.hpp (sync_hscb /
// wait_ipc_scb).  Only the boundary-fault sentinels remain here, because they
// are pure mock diagnostics (the host launcher polls them) with no V8
// machine-instruction counterpart.

inline AICORE void MockSetFault(__gm__ uint32_t* faultFlag, uint32_t faultCode)
{
    if (faultFlag != nullptr) {
        volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(faultFlag);
        __asm__ __volatile__("" ::: "memory");
        *ptr = faultCode;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
}

// MOCK: V6 out-of-mesh boundary fault (TPUSH/TPOP off the mesh edge).
//
// V6: a TPUSH/TPOP whose (dir,dist) target leaves the mesh raises a fault
// (raise_fault(kFaultPushOOB/kFaultPopOOB), V6 3.5.3 P0/C0).
//
// A2/A3 mock: explicit early-exit + sentinel write so the host can detect the
// out-of-bound attempt.  Real boards will raise a fault; here we trap softly
// by writing a sentinel and aborting the kernel branch.  The host launcher
// inspects a "fault sentinel" GM word after each kernel and fails the run.
inline AICORE void MockBoundaryFault(__gm__ uint32_t* faultSentinel, uint32_t faultCode)
{
    if (faultSentinel != nullptr) {
        *reinterpret_cast<volatile __gm__ uint32_t*>(faultSentinel) = faultCode;
    }
    // Best-effort halt of the current kernel branch.  Real silicon will fault
    // here; on A2/A3 we just stop emitting further GridPipe ops in this branch.
}

// Fault codes mirror SPR_BOUNDARY_MASK fields (design doc section 5.2).
inline constexpr uint32_t kFaultPushNorth = 0x101;
inline constexpr uint32_t kFaultPushEast = 0x102;
inline constexpr uint32_t kFaultPushWest = 0x103;
inline constexpr uint32_t kFaultPushSouth = 0x104;
inline constexpr uint32_t kFaultPushSource = 0x105; // Always illegal.
inline constexpr uint32_t kFaultPopNorth = 0x201;
inline constexpr uint32_t kFaultPopEast = 0x202;
inline constexpr uint32_t kFaultPopWest = 0x203;
inline constexpr uint32_t kFaultPopSouth = 0x204;
// TPOP tried to drain a slot outside this core's own SRAM segment.  The NoC
// fabric has no remote-read path, so this can only happen via a mis-wired mock;
// the GmSramArena guard in GRID_TRY_TPOP_IMPL traps it here (design: NoC is
// write-only, TPOP is local-only).
inline constexpr uint32_t kFaultPopNonLocal = 0x205;
inline constexpr uint32_t kFaultWaitReadyTimeout = 0x301;
inline constexpr uint32_t kFaultWaitFreeTimeout = 0x302;
// A GridPayloadWindow reaches past the end of its slot.  Once the transfer
// length stopped being the compile-time SlotStride, nothing statically bounds it
// any more, and a push whose window overruns writes into the PEER's window --
// silent cross-core corruption that is far harder to trace than a local overrun.
// So the range is checked at runtime and trapped here instead.  (a5 gets this for
// free: its lengths come from the tile/GlobalTensor descriptors, so the geometry
// is self-consistent by construction.)
inline constexpr uint32_t kFaultPushPayloadRange = 0x401;
inline constexpr uint32_t kFaultPopPayloadRange = 0x402;
inline constexpr uint32_t kFaultBcastPayloadRange = 0x403;

// 接力计数 producer handoff (THANDOFF, GridTHandoff.hpp) faults.
//
// kFaultHandoffWindowMismatch is the one that catches a genuine design error
// rather than a hang: handing off between two pipes wired to DIFFERENT windows
// relays counters that describe a ring the successor will never touch, and the
// successor then writes from a bogus baseline into a ring whose real occupancy it
// has not been told about.  The two pipes must be declared over one window (one
// physical channel, two producer bindings), so a mismatch is trapped instead of
// silently producing a corrupt relay.
inline constexpr uint32_t kFaultHandoffRetireTimeout = 0x501;  // retiring producer's last READY never landed
inline constexpr uint32_t kFaultHandoffInstallTimeout = 0x502; // INSTALL_BASE doorbell never arrived
inline constexpr uint32_t kFaultHandoffOpenTimeout = 0x503;    // OPEN_ACK never arrived (rebase branch)
inline constexpr uint32_t kFaultHandoffWindowMismatch = 0x504; // old/new pipe are not the same physical channel

// Direction-keyed fault code lookup.  Explicit switch avoids relying on the
// numeric layout of GridDirection so renumbering the enum cannot silently
// remap fault codes.
AICORE constexpr uint32_t PushFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPushNorth;
        case GridDirection::EAST:
            return kFaultPushEast;
        case GridDirection::WEST:
            return kFaultPushWest;
        case GridDirection::SOUTH:
            return kFaultPushSouth;
        case GridDirection::SOURCE:
            return kFaultPushSource;
    }
    return kFaultPushSource;
}

AICORE constexpr uint32_t PopFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPopNorth;
        case GridDirection::EAST:
            return kFaultPopEast;
        case GridDirection::WEST:
            return kFaultPopWest;
        case GridDirection::SOUTH:
            return kFaultPopSouth;
        case GridDirection::SOURCE:
            return 0; // SOURCE pop is legal; never raises a boundary fault.
    }
    return 0;
}

} // namespace grid_mock
} // namespace pto

// ===========================================================================
// Section 3: GmSramArena -- GM address-segment model of per-core SRAM (mock).
//
// The neighbor-SRAM addressing / transfer that used to live here as a
// CCE-intrinsic-style API (get_neighbor_sram_addr / copy_ubuf_to_neighbor_ubuf /
// copy_local_slot_to_ubuf / sram_pop_is_local, with neighbor_sram_addr /
// NeighborSramOperand operands and a fabricated __builtin_pto_* stub) is gone:
// per V8, payload PUSH lowers directly to the copy_ubuf_to_neighbor_ubuf CCE
// facade (grid_cce_intrinsic.hpp) and TPOP's local drain reuses the existing
// local copy (no Grid-specific intrinsic).  The peer-window / local-slot address
// resolution is now a plain runtime helper in the demo's gridpipe_payload_inl.hpp.
//
// What remains here is the GmSramArena model the TPOP guard still needs to
// enforce the NoC "TPOP reads local SRAM only" rule against the GM-window mock.
// ===========================================================================

namespace pto {

// ---------------------------------------------------------------------------
// GmSramArena: explicit GM address-segment model of future-hardware per-core
// SRAM.
//
// Real silicon gives every core a private on-chip SRAM that the NoC fabric can
// only *write* into from a neighbor (TPUSH = cross-hop write), never *read* out
// of remotely (TPOP only drains the local core's own SRAM).  Until that
// hardware exists we model the SRAM as a contiguous GM arena cut into equal,
// per-core address segments.  Core `c` owns segment `c`:
//
//   [base + c*segBytes, base + (c+1)*segBytes)
//
// The NoC contract this encodes:
//   * a core may WRITE across segments  (TPUSH pushes into a neighbor segment),
//   * a core may only READ its own segment (TPOP pops from local SRAM).
//
// The arena is the single source of truth for "which core owns this address",
// so the mock TPOP path can reject a cross-segment read instead of silently
// servicing it through the GM-backed fake window (which physically *can* read
// any address, unlike the fabric it stands in for).
struct GmSramArena {
    uint64_t base = 0;     // segment 0 base == contiguous arena base
    uint64_t segBytes = 0; // bytes per per-core segment (== HCCL winSize in the demo)
    uint32_t numSegs = 0;  // number of cores / segments

    AICORE constexpr uint64_t SegmentBase(int seg) const { return base + static_cast<uint64_t>(seg) * segBytes; }

    // Index of the segment that owns `addr`, or -1 if `addr` is outside the arena.
    AICORE constexpr int SegmentOf(uint64_t addr) const
    {
        if (numSegs == 0 || segBytes == 0 || addr < base) {
            return -1;
        }
        uint64_t idx = (addr - base) / segBytes;
        return idx < numSegs ? static_cast<int>(idx) : -1;
    }

    // True iff [addr, addr+bytes) lies entirely within segment `seg`.  This is
    // exactly the "may core `seg` read this slot?" test used by the TPOP guard.
    AICORE constexpr bool InSegment(int seg, uint64_t addr, uint64_t bytes) const
    {
        if (seg < 0 || static_cast<uint32_t>(seg) >= numSegs) {
            return false;
        }
        uint64_t lo = SegmentBase(seg);
        uint64_t hi = lo + segBytes;
        return addr >= lo && (addr + bytes) <= hi && (addr + bytes) >= addr; // last term traps wrap-around
    }
};

// Compile-time self-test of the segment classifier.  It is built into every
// A2/A3 kernel that pulls in this header (GridTPush.hpp -> pto_instr_impl.hpp),
// so a regression in the segment math fails the build rather than silently
// mis-routing a TPOP.  It also doubles as executable documentation of the rule.
AICORE constexpr bool GmSramArenaSelfCheck()
{
    GmSramArena arena{0x1000, 0x100, 4}; // 4 cores, 0x100-byte segments, based at 0x1000
    bool ok = true;
    ok = ok && (arena.SegmentOf(0x1000) == 0);    // first byte of core 0
    ok = ok && (arena.SegmentOf(0x11FF) == 1);    // last byte of core 1
    ok = ok && (arena.SegmentOf(0x1200) == 2);    // first byte of core 2
    ok = ok && (arena.SegmentOf(0x0FFF) == -1);   // below the arena
    ok = ok && (arena.SegmentOf(0x1400) == -1);   // past the arena ([0x1000,0x1400))
    ok = ok && arena.InSegment(1, 0x1100, 0x40);  // wholly inside core 1 -> local
    ok = ok && !arena.InSegment(1, 0x11F0, 0x40); // spills past core 1 -> not local
    ok = ok && !arena.InSegment(1, 0x1200, 0x10); // core 1 reading core 2 -> not local
    return ok;
}
static_assert(GmSramArenaSelfCheck(), "GmSramArena segment classifier self-test failed");

} // namespace pto

#endif // PTO_A2A3_GRID_INTRINSIC_HPP

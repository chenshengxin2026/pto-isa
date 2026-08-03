/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-time config for the GridPipe 接力计数 producer-handoff smoke kernel
// (THANDOFF, design doc Grid_TPUSH_TPOP_WSE核间握手机制选型 §1.3 / §3.2 B1).
//
// A ROWS x COLS grid runs two phases over ONE physical channel per cell:
//
//   phase 1   EAST pipe   -- each cell pushes HANDOFF_P1_TILES tiles to its east
//                            neighbour; each receiver pops only HANDOFF_P1_POPS
//                            of them.
//   THANDOFF  the consumer keeps its ring, its scoreboard pair and its cell; only
//             its PRODUCER changes (west neighbour -> north neighbour), so the
//             counters have to be handed over.
//   phase 2   SOUTH pipe  -- each cell pushes HANDOFF_P2_TILES tiles to its south
//                            neighbour; each receiver first drains whatever phase
//                            1 left in its ring (written by the RETIRED producer,
//                            popped through the NEW pipe), then the new tiles.
//
// HANDOFF_P1_POPS is the knob that selects which branch of the baton fires:
//   P1_POPS <  P1_TILES  -> ring NOT drained  -> relay  (prod_idx continues at E)
//   P1_POPS == P1_TILES  -> ring drained      -> rebase (both sides restart at 0)
// Both are exercised in one run regardless: column 0 has no phase-1 producer at
// all, so its ring is empty and it always takes the rebase branch.

#ifndef HANDOFF_SMOKE_CONFIG_HPP
#define HANDOFF_SMOKE_CONFIG_HPP

#ifndef CONFIG_HANDOFF_ROWS
#define CONFIG_HANDOFF_ROWS 3
#endif

#ifndef CONFIG_HANDOFF_COLS
#define CONFIG_HANDOFF_COLS 4
#endif

// Tiles pushed per phase-1 edge, and how many of them the receiver pops BEFORE
// the handoff.  P1_POPS < P1_TILES leaves the ring undrained on purpose.
#ifndef CONFIG_HANDOFF_P1_TILES
#define CONFIG_HANDOFF_P1_TILES 2
#endif

#ifndef CONFIG_HANDOFF_P1_POPS
#define CONFIG_HANDOFF_P1_POPS 1
#endif

#ifndef CONFIG_HANDOFF_P2_TILES
#define CONFIG_HANDOFF_P2_TILES 1
#endif

#ifndef CONFIG_HANDOFF_T
#define CONFIG_HANDOFF_T 16
#endif

#ifndef CONFIG_HANDOFF_W
#define CONFIG_HANDOFF_W 64
#endif

constexpr int HANDOFF_ROWS = CONFIG_HANDOFF_ROWS;
constexpr int HANDOFF_COLS = CONFIG_HANDOFF_COLS;
constexpr int HANDOFF_P1_TILES = CONFIG_HANDOFF_P1_TILES;
constexpr int HANDOFF_P1_POPS = CONFIG_HANDOFF_P1_POPS;
constexpr int HANDOFF_P2_TILES = CONFIG_HANDOFF_P2_TILES;
constexpr int HANDOFF_T = CONFIG_HANDOFF_T;
constexpr int HANDOFF_W = CONFIG_HANDOFF_W;

static_assert(HANDOFF_P1_POPS <= HANDOFF_P1_TILES, "cannot pop more phase-1 tiles than were pushed");

constexpr int HANDOFF_TILE_ELEMS = HANDOFF_T * HANDOFF_W;
constexpr int HANDOFF_TILE_BYTES = HANDOFF_TILE_ELEMS * 4; // fp32 payload tile

// Per-cell input tiles: the phase-1 batch followed by the phase-2 batch.
constexpr int HANDOFF_IN_TILES = HANDOFF_P1_TILES + HANDOFF_P2_TILES;
// Per-cell output slots: phase-1 pops + the phase-1 leftovers drained after the
// handoff + the phase-2 pops.  Cells that lack one of the two producers simply
// leave the tail slots at zero.
constexpr int HANDOFF_OUT_TILES = HANDOFF_P1_TILES + HANDOFF_P2_TILES;

// One slot per in-flight tile plus slack: the ring has to hold the phase-1
// leftovers AND the phase-2 tiles the relayed producer writes behind them.
constexpr int HANDOFF_SLOT_BYTES = HANDOFF_TILE_BYTES;
constexpr int HANDOFF_SLOT_COUNT = 4;

// The handoff generation THANDOFF stamps on its two doorbells.  One handoff in
// this kernel, so generation 1.
constexpr int HANDOFF_SEQ = 1;

// Host-visible mirror of pto::a2a3_grid::WindowBytes<Pipe>():
//   layout = kFlagsBytes (512) + SlotCount * SlotStride.
// The EAST and SOUTH pipes are ONE physical channel handed from one producer to
// the next, so they share this single window -- that sharing is the whole point.
// Keep in sync with include/pto/npu/a2a3/grid_pipe_runtime.hpp.
constexpr int HANDOFF_GRID_FLAGS_BYTES = 512;
constexpr int HANDOFF_WINDOW_BYTES = HANDOFF_GRID_FLAGS_BYTES + HANDOFF_SLOT_COUNT * HANDOFF_SLOT_BYTES;

// Stamp carried by cell `c`'s tile `k` (k < P1_TILES: phase 1; else phase 2).
// Distinct per (cell, tile) so a mis-relayed slot shows up as a wrong value
// rather than a plausible one.
inline float HandoffStamp(int cell, int k) { return static_cast<float>((cell + 1) * 10 + k); }

#endif // HANDOFF_SMOKE_CONFIG_HPP

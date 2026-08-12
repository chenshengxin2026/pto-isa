# Single-device Multi-block FFN GridPipe Demo

## Goal

This demo validates the three distributed-FFN GridPipe collective interfaces — **TPUSH**, **TBROADCAST**, **TREDUCE** — on a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks; each block owns one logical cell. There are **four examples**, one per (interface, FFN pattern), all on the same pure 1D N-cut 4×8 = 32-cell topology with the real DeepSeek-v4 Pro shapes (M=T=8, H=7168, I=3072):

| Example (run script / executable) | Interface verified | FFN pattern | Cross-cell collective |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | explicit `TPOP(pipe, ..., prodId)` + `TADD` + `TPUSH(pipe, ..., consId, isLastTransfer)` relay |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | nearest-neighbor `TPUSH`/`TPOP` relay gather (fan-in-1 DAG) |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC group broadcast |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | fused `TREDUCE<GridGroup, Sum>` N→1 group fan-in (`mov_ubuf_group`, op=SUM) |

Each example compares its `[T, H]` output with `golden.bin` using a `1e-3` tolerance. All four pass **bit-exact** on the NPU (`max diff = 0`, run with `-r npu`); see [Bit-exactness notes](#bit-exactness-notes).

The cross-cell collectives use the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, per-channel ready/free/close counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

Unicast names peers at runtime (`consId` on TPUSH and `prodId` on TPOP), while concurrent group broadcast keeps its `TBROADCAST<GridGroup>` / `TPOP<GridGroup>` interface. The standalone broadcast smoke test lives under `smoke/`.

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the four host executables and their mixed Cube/Vec device kernel shared libraries. |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | Set up CANN, generate data, configure CMake, build, and run the TREDUCE / TPUSH ReduceSum examples. |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | Set up CANN, generate data, configure CMake, build, and run the TBROADCAST / TPUSH AllGather examples. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, SwiGLU clamp bounds, the A3 precision-mapping table, and Batcher GM arena byte sizes. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declarations (one per example). |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host drivers: ACL setup, fake HCCL context / local GridPipe windows, working buffers, Batcher load/distribute, kernel launch, golden comparison, cleanup. |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel: the EAST+SOUTH reduce uses the fused `TREDUCE<GridGroup, Sum>` group fan-in (`mov_ubuf_group`, op=SUM) at the sink. |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel: EAST and SOUTH share one channel across launches, exercising close + relay-count rebinding with explicit TPOP + TADD + TPUSH. |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host drivers. |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel: the two gather phases use `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`. |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel: the two gather phases use a bidirectional `TPUSH`/`TPOP` relay. |
| `batcher.hpp` | Host-side GM-simulated **Batcher**: owns the full input + the full DRAM-resident weights in GM, splits them column-parallel into per-cell shards, broadcasts x, and exposes the output-collection region. |
| `tpipe_tmov_inl.hpp` | Directional `TMOV` overloads that lower Cube↔Vec C2V/V2C transfers to the existing `TPUSH`/`TPOP`, so the kernel body never spells out the handshake. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window adapter: peer-slot/SCB resolution, tile-to-producer-L1 staging, producer-L1-to-peer-ring copies, local receive-ring drains, and the TPOP locality guard. |
| `smoke/` | Standalone Vec-only GridPipe broadcast smoke test (`bcast_smoke_*` + `run_bcast_smoke.sh`). |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | Grid CCE facades: unified-L1 copies, absolute `sync_hscb`, MPSC `atom_add_hscb`, blocking `wait_ipc_scb`, and group `mov_ubuf_group`. The A3 mock represents L1 ranges with GM windows. |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 data model + mock support: per-channel ready/free/close SCBs, producer/consumer bindings, the three-state per-consumer FSM, durable relay counters, mesh/group resolvers, fault sentinels, and the `GmSramArena` TPOP locality guard. |
| `scripts/gen_data.py` | Generates the FULL fp16 X/weight tensors (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) the Batcher consumes, plus an fp32 SwiGLU `golden` reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Bit-exactness notes

Run with `-r npu` (the `sim`/`camodel` modes fail `aclrtSetDevice` 507033); on a shared host every run goes through `task-submit`. All four examples produce `max diff = 0` vs `golden.bin` — bit-exact, not merely within the `1e-3` tolerance. Two real bugs once masked that, both now fixed:

- **MPSC SPR doorbells (TBROADCAST/TREDUCE).** Broadcast payloads remain collision-free because source rank `k` owns slot `k` in every receiver. Notification no longer uses static per-source L1 lanes: each collective reserves `ready_scb/free_scb/close_scb[CollectiveChan]` from the fixed `GridPipe` SPR header. All producers atomically add READY/CLOSE at a receiver, and all receivers atomically add FREE at a producer. Replacing the old lanes with ordinary absolute `sync_hscb` stores would be incorrect—simultaneous writers would overwrite one another (last-writer-wins), lose counts, and deadlock. The A3 mock implements `atom_add_hscb` with an atomic s32 UB→GM accumulate; native `__atom_add_hscb` peer-SPR routing remains an explicit hardware dependency.
- **Phase-D output T-stride (both AllGather kernels).** The AllGather y-shard `[T, Hc]` is written into the *full* `[T, H]` output, so its row stride must be the full output width `kHfull` (= `H` = 7168). A copy-paste from the `hidden_full` store had left it at `kIfull` (= `I` = 3072), scrambling y rows 1–7 (≈50 % zero output / large drift). One-line fix `kIfull` → `kHfull` in the `GY` store of both AllGather kernels.

The `treduce` ReduceSum additionally requires its per-cell partial buffers (`partialBuf` / `rowPartialBuf`) to be laid out **segment-major** — each `[T, kHBase]` H-segment contiguous at offset `h*(T*kHBase)` — so the group fan-in reads every row-mate's segment as one contiguous byte range; only the final `yFull` keeps the strided `[T, H]` golden layout.

A 32-block launch still cannot run in one wave on 24 physical AICores — a single-wave launch oversubscription-deadlocks phase C, whose COL groups span all 4 rows (first-batch cells spin on second-batch row-3 doorbells that never get a core). The host therefore launches in waves sized from `--phys-cores` (`rowsPerWave = physCores/cols`, `colsPerWave = physCores/rows` → 2 waves each for phases B and C, 6 launches total, ~5 ms). With the stride fix the wave split is purely a scheduling concern, not a reliability problem.

## Execution Flow

1. Each `run_*.sh` parses arguments. Defaults are the real DeepSeek-v4 Pro shapes on a 4×8 = 32-cell mesh: `gridRows=4`, `gridCols=8`, `T=8` (token tile), `H=7168`, `Fi=96` (per-cell I shard; full `I = Fi * cells = 3072`), `n-ranks=1`, and `phys-cores=24`.
2. Unless `--build-only` is set, `scripts/gen_data.py --pure-ncut` generates the flat full-tensor Batcher inputs (`x_full`, `w_gate_full`, `w_up_full`, `w_down_full`) plus the SwiGLU `golden.bin`.
3. CMake builds two targets per example — a `..._mixed_kernel` `dav-c220` shared library and the matching host executable (e.g. `distributed_ffn_grid_treduce_reducesum_mixed_kernel` + `distributed_ffn_grid_treduce_reducesum`). The two AllGather kernels are additionally compiled with `-DCONFIG_FFN_GRID_ALLGATHER`.
4. The host initializes ACL on the selected device.
5. The host allocates contiguous device buffers for `gridRows * gridCols` cells.
6. The host allocates one local GridPipe SRAM window per cell, backed by GM in the mock, and builds a fake `HcclDeviceContext`:

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. The host **Batcher** (`batcher.hpp`) loads the full input + full DRAM-resident weights into GM, splits the weights column-parallel into per-col shards, and broadcasts x per-row (see [Batcher (GM-simulated)](#batcher-gm-simulated)).
8. The host obtains the FFTS base address with `rtGetC2cCtrlAddr()` and launches `DistributedFfnGridMixedKernel` once with `gridRows * gridCols` blocks.
9. Inside each block, Cube and Vec branches exchange intermediate tiles through A2/A3 `TPipe` FIFOs. The kernel issues these C2V/V2C transfers as a directional `TMOV` (`TMOV(pipe, tile)` to produce, `TMOV(tile, pipe)` to consume); the underlying `TPUSH`/`TPOP` stays implicit (see `tpipe_tmov_inl.hpp`):

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TMOV C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TMOV C2V-->

Vec:
  hidden[row,col] = fp16(SwiGLU(gatePartial) * upPartial)   # SiLU(clamp(gate)) * up
  hidden[row,col] --TMOV V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TMOV C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

The cross-cell `EAST`/`WEST` reduce and gather keep their explicit GridPipe `TPUSH`/`TPOP`; only the in-block Cube↔Vec C2V/V2C traffic is folded into `TMOV`.

10. The host synchronizes the stream, checks GridPipe fault flags, copies `yOutput` back, and compares it with `golden.bin`.

## Key Designs

### Batcher (GM-simulated), SwiGLU, and the A3 precision map

This demo is aligned to the WSE-FFN tile-level expansion (`WSE-FFN-tile级全展开图.svg`), which casts an external **Batcher** as the owner of the full input and the full DRAM-resident weights, responsible for splitting/distributing them to cores and collecting the output. A2/A3 has no such hardware, so `batcher.hpp` simulates the Batcher entirely in GM:

- **Full weights resident in GM** (`w_gate_full`/`w_up_full` `[H,F]`, `w_down_full` `[F,H]`), mirroring the SVG's `DRAM 常驻` store.
- **Distribute** slices those full weights column-parallel and writes a contiguous per-col shard (`[H,Fi]` gate/up; `[F,Hc]` AllGather or `[Fi,H]` ReduceSum for down) into a per-col GM region. Each core then TLOADs its own shard (DRAM→L1 stream), exactly like a core streaming its Batcher-delivered weight tile.
- **Broadcast** writes the full `x` into GM; every column in a row reads the same `x` (broadcast, "复制 broadcast → N 核"). This also drops the legacy per-cell duplication: `x` is per-row, weights are per-col.
- **Collect**: cores write their y shards (AllGather) / the EAST reduce writes the per-row sum (ReduceSum) straight into the Batcher `y` region of GM.

The kernel addresses Batcher storage by `(row, col)`: `x = xFull + row*…`, `w = wShards + col*…`, `y = yFull + row*… (+ col*Hc)`.

The SVG activation is **SwiGLU = SiLU(clamp(gate)) · up** ("SiLU + clamp(max=10)"). The Vec branch composes SiLU from existing intrinsics in fp32: `SiLU(g) = g / (1 + e^-g)` via `TMAXS`/`TMINS` (clamp ±10), `TMULS(-1)` → `TEXP` → `TADDS(1)` (denominator), `TDIV`, then `TMUL` with `up`. `gen_data.py` uses the identical clamp+SiLU for the golden reference.

The SVG also carries low precisions A3 does not support (FP4 weights, FP8 activations, BF16 I/O). Per the extension design every tile-graph precision is mapped to **one A3-supported dtype** (see the table in `ffn_config.hpp`): FP4/FP8/BF16 → `half`, FP32 accumulators/output stay `float`. The `act_quant` and weight-`unpack` stages therefore exist as named, zero-cost identity points in the kernel — they document where the SVG casts would live without adding any A3-unsupported conversion. The fp16/fp32 data path already *is* the mapped result.

### Mixed Cube/Vec launch

The device kernel is compiled for `dav-c220`. Cube and Vec code paths are guarded by `__DAV_CUBE__` and `__DAV_VEC__`, so both sides live in one kernel source and synchronize through regular A2/A3 `TPipe` ready/free handshakes.

### Implicit C2V/V2C `TMOV`

`tpipe_tmov_inl.hpp` adds two `TMOV` overloads so the kernel body expresses Cube↔Vec transfers as a single tile-move op instead of explicit `TPUSH`/`TPOP`:

- `TMOV(pipe, tile)` — producer side; forwards to `TPUSH` (write `tile` into the C2V/V2C FIFO).
- `TMOV(tile, pipe)` — consumer side; forwards to `TPOP` (read the next slot into `tile`).

Which physical core writes vs reads, and whether the pipe is C2V or V2C, stays encoded in the `TPipe` type and its `__DAV_CUBE__`/`__DAV_VEC__` guards, so the call site is direction-agnostic. The overloads take exactly two `(pipe, tile)`/`(tile, pipe)` arguments (no wait-event pack), which makes them strictly more specialized than the generic tile-to-tile `TMOV(dst, src, ...)`; overload resolution therefore selects them for any `TPipe`/tile pair and leaves every other `TMOV` use unchanged. This keeps the Cube↔Vec handshake implicit at the call site, the way a real WSE fabric move hides the producer/consumer split, while reusing the existing `TPUSH`/`TPOP` sync and record machinery verbatim.

### Single-device logical grid

`get_block_idx()` is the row-major cell id:

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

All cells run on one device. `gridRows` controls data-parallel token tiles, and `gridCols` controls model-parallel FFN shards.

### Local GridPipe mock

The host allocates `gridRows * gridCols` local SRAM windows, backed by GM in the mock. `TPUSH(..., consId)` resolves that consumer's SRAM slot with `ResolvePeerSlotAddr`, writes the payload, then publishes READY; `TPOP(..., prodId)` waits on local READY, loads the local slot, and returns FREE credit to that producer.

The mock uses GM flag polling and cache maintenance to emulate the intended LPU WSE `SPR` / `WFE` behavior on A2/A3.

### NoC write-only address-segment SRAM model (`GmSramArena`)

To stay close to real silicon, the mock models future-hardware per-core SRAM as an explicit **GM address-segment arena**. The single contiguous `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window buffer is cut into equal per-core segments, so segment `c` (== `windowsIn[c]`) *is* core `c`'s private SRAM:

```text
segment c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

Each segment keeps receive and producer storage distinct:

```text
control + record | unicast receive rings | optional broadcast receive ring | producer staging slot
                                                                             [SlotStride bytes]
```

The producer slot is appended after every receive-side region. `TPUSH` and `TBROADCAST` first stage the tile there, then map that local L1 source to the selected peer payload-ring address. This models real WSE, where vector data and L1 share one SRAM address space, without allowing the producer source to alias a receive ring.

`GmSramArena` (in `include/pto/npu/a2a3/grid_intrinsic.hpp`) carries `{base, segBytes, numSegs}` plus the `SegmentOf` / `InSegment` classifiers; the demo builds it on-device from the fake `HcclDeviceContext` window table (`SramArenaFromCtx`). It is the single source of truth for "which core owns this address".

This makes the NoC contract explicit and **enforced**: the fabric can only *write* across cores, never *read*.

- `TPUSH(pipe, tile, consId, ...)` reads this core's dedicated producer slot and writes the payload into the **consumer's receive ring** — a cross-segment write, exactly what the fabric does.
- `TPOP(pipe, tile, prodId)` may only drain **this core's own** segment. `GRID_TRY_TPOP_IMPL` calls the `PopSlotIsLocal` guard before the payload read; on a cross-segment read it raises `kFaultPopNonLocal` (`0x205`, "pop non-local segment") and aborts the pop. The host's `CheckGridPipeFaults` surfaces it.

On native hardware `PopSlotIsLocal` is a no-op (`true`): a TPOP read address is local by construction because the fabric has no remote-read path. The guard exists only because the A2/A3 mock backs SRAM with a GM window that *can* physically read any address, so without it a demo could silently rely on a remote read the silicon cannot perform. A compile-time `static_assert(GmSramArenaSelfCheck())` is built into every A2/A3 kernel, so a regression in the segment math fails the build rather than mis-routing a pop.

> The `pto::comm` variants (`TREDUCE` / `TGATHER`) intentionally do **not** follow this rule: they are a root-pulls-from-every-rank collective (HCCL/RDMA-style remote reads), a different memory model from the WSE NoC. Only the GridPipe `TPUSH`/`TPOP` path is constrained to write-only.

### IPC_SCB scoreboard intrinsic API

GridPipe ready/free/close synchronization follows the V8 IPC_SCB scoreboard route. Each channel owns one SCB of each kind. The handshake intrinsics live in `include/pto/npu/a2a3/grid_cce_intrinsic.hpp` as a thin CCE facade layer — each facade forwards 1:1 to a `__builtin_cce_*` under `PTO_GRID_CCE_NATIVE`, and otherwise emulates the same semantics in the A2/A3 mock with a GM word + cache maintenance (`dcci`/`dsb`):

- `copy_l1_to_neighbor_l1(dstNeighborSlot, srcProducerSlot, transferScratch, bytes)` (G1 / HW-DEP-0) writes the isolated local producer L1 slot into the resolved neighbor receive-ring slot. `transferScratch` exists only for the A3 GM-backed DMA pump; it is not an architectural source address. The operation is not self-synchronizing; the following `sync_hscb(READY)` publishes data-ready.
- `sync_hscb(peerScb, absCount)` (V8 `SYNC_HSCB`/`ST_HSCB`, G2 — reused HSCB store + neighbor IPC_SCB addressing / HW-DEP-1) stores an absolute count into the peer's `ready_scb`, `free_scb`, or `close_scb`. The target kind and peer are resolved into `peerScb` by `RemoteScbPtr`.
- `atom_add_hscb(peerScb, delta)` atomically increments a peer scoreboard for MPSC READY/FREE/CLOSE fan-in. The A3 mock lowers it to atomic s32 accumulation; the native facade maps to `__atom_add_hscb` and requires peer IPC_SCB targeting/wakeup support from the final WSE ISA.
- `wait_ipc_scb(localScb, threshold, slot)` (V8 `WAIT_SPR`, G3 — reused IPC_SCB blocking wait) reads + blocks in **one** instruction: the entry reads the local IPC_SCB and proceeds if it is already `>= threshold`, else blocks the current pipe until the peer's `sync_hscb` store raises it. V8 dropped the V7 `MOV_SPR2X` non-blocking peek — there is no separate read step. The demo calls the `wait_ipc_scb_sim(..., maxSpins)` mock wrapper, which adds a spin-timeout fault sentinel so a handshake deadlock fails the test instead of hanging; the documented hardware interface is the void `wait_ipc_scb`.

Payload destination resolution (turning a local receive-ring slot / scoreboard word into the same byte offset in a peer's GM window) is a plain runtime helper in `gridpipe_payload_inl.hpp` (`ResolvePeerSlotAddr` / `RemoteScbPtr`), not an intrinsic. The source is separately fixed at `producerSlotBase`. TPOP's local drain reuses `copy_gm_to_ubuf`; the NoC is write-only, so there is deliberately **no cross-core read** of payload. `PopSlotIsLocal` rejects a mis-wired cross-segment read.

Native lowering targets the real CCE HSCB/IPC_SCB stack. The compiler-facing copy builtin retains its historical `ubuf` spelling, but the facade always passes the dedicated unified-L1 producer address; the caller's tile address is never used as the NoC source mapping. The A3 mock represents scoreboards and both L1 ranges with GM plus cache maintenance.

`TPUSH(pipe, tile, consId)` waits the selected local producer channel's `free_scb`, stages the tile in `producerSlotBase`, copies that L1 range to the independently selected consumer channel's ring, then publishes `prod_idx`. `TPOP(pipe, tile, prodId)` waits its local consumer channel's `ready_scb`, reads only that local receive ring, then publishes `cons_idx` to the negotiated producer channel's `free_scb` at the peer. The two channel indices need not be equal.

### Time-division MPSC relay counting

The producer keeps one FSM entry per consumer (`UNBOUND`, `ACTIVE`, or `CLOSED`) plus a local producer-channel table (`UNBOUND`, `ACTIVE`, or producer-`CLOSED`). An `ACTIVE` consumer uses the normal TPUSH fast path. For an `UNBOUND` or `CLOSED` consumer, the producer first reserves an unused or producer-`CLOSED` local channel, waiting if none is available. It sends `[producer block id, local producer channel]` through the consumer's bind-request L1 line; it does not wait for the consumer to choose the producer-side resource.

The consumer independently scans its receive channels, preferring never-used entries and otherwise requiring `close_scb > closeBaseline` and `cons_idx >= close_scb` so the old producer has closed and its final item has drained. It returns `[ready_scb baseline, consumer channel, completion]` through the producer's response L1 line and writes its current `cons_idx` into `free_scb[producer channel]`. `completion` is written last; the producer polls only that explicit field, then MOVs the ready baseline into `prod_idx` and records the local-producer-channel ↔ remote-consumer-channel mapping. Counts are absolute overwrite values and continue monotonically on each consumer ring across producer turns; they are not increments and no SCB is reset.

The public A2/A3 API is `TPUSH(pipe, tile, consId, isLastTransfer, events...)`; omitting the Boolean is equivalent to `isLastTransfer == false`. Set it to `true` only on the last tile of that producer→consumer turn. The final TPUSH publishes the same absolute count to the remote consumer channel's `close_scb` after payload and READY, then transitions both that consumer FSM entry and the local producer channel to `CLOSED`. This is time-division MPSC: only one bind request may target a consumer at a time.

### Group broadcast and reduce data movement

The group COPY and reduce modes share the native group opcode, but their facades keep the local address contract explicit:

- `copy_l1_to_group(srcProducerSlot, groupSlot, transferScratch, ...)` is the broadcast path. It fans out from the isolated producer L1 slot; the A3 mock expands the operation per member, while native still emits one group COPY.
- `mov_ubuf_group(..., op=SUM/MAX/MIN, ...)` is the current group-reduce path. The sink reads each member contribution and folds members in ascending block-id order, preserving relay accumulation order.

`GRID_TBROADCAST` stages once, then uses `copy_l1_to_group` for an affine group arena or per-member `copy_l1_to_neighbor_l1` otherwise. Its aggregate SPR barrier waits for the full declared producer set before any rank slot is drained. Group `TREDUCE` is called by every member: contributors atomically publish READY/CLOSE, the sink waits for all `N-1`, lowers SUM/MAX/MIN to `mov_ubuf_group`, and returns atomic FREE credits. It is distinct from the peer-id TPOP + TADD + TPUSH relay.

### fp32 reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison. The ReduceSum reduce is H-chunked (`kHSegs` = 7): the `treduce` example reads symmetric per-cell segment storage under the dedicated aggregate-SPR handshake, while the `tpush` example relays segments hop-by-hop with peer-id TPOP + TADD + TPUSH. Phase B and C keep their collective counters monotone in the persistent window; no SCB is reset.

### Peer-id unicast

Unicast direction is derived by the schedule, not encoded in the instruction type. TPUSH receives the destination logical block id (`consId`), TPOP receives the source logical block id (`prodId`), and `RemoteScbPtr` resolves the symmetric slot/SCB offset in that peer's window. This keeps one GridPipe usable when the peer changes between phases.

### Concurrent group broadcast (TBROADCAST)

`TBROADCAST<GridGroup>` (`ROW`, `COL`, or `SUBRECT`) broadcasts a cell's tile to every other group member as one op: per-target writes into rank-indexed receiver slots are batched with no inter-target fence, the whole broadcast pays one publish fence, then aggregate ready/close SPRs are atomically incremented. It is not lowered to a per-hop `TPUSH` loop.

Unlike the old single-source `TPUSH<GridSpan>` multicast, every group member may call `TBROADCAST` simultaneously. Payload writes cannot conflict because `BcastSlotCount >= GroupMax` gives each source a disjoint rank slot. Signal writes do target the same receiver SPR, so atomic accumulation is mandatory. Before draining, a caller sets `pipe.SetBcastExpectedProducerCount(K)` (`K=groupSize-1` for AllGather, `K=1` for the smoke); every TPOP in that round waits the same complete-set READY/CLOSE barrier, reads one distinct source slot, and atomically returns FREE to that source. All participants in a group must be resident in the same hardware wave, because `WAIT_SPR` cannot make progress if a required producer has not been scheduled.

### GridPipe smoke tests

`bcast_smoke` is a Vec-only data-movement smoke test (no Cube, no matmul, no data files) on the same GM-backed mock. One source cell broadcasts a stamped fp32 `[T, W]` tile to its row, column, or sub-rectangle; each receiver drains and stores it, and the host checks `out[cell] == in[source]`.

### AllGather variant

The two AllGather examples (`run_tbroadcast_allgather.sh`, `run_tpush_allgather.sh`) share the same pure-N-cut data (`scripts/gen_data.py --pure-ncut`) as the ReduceSum examples; the kernel is compiled with `-DCONFIG_FFN_GRID_ALLGATHER`, which makes the host Batcher slice `W_down` along the output **H** (each cell gets an `[I_full, Hc]` shard, `Hc = H / cells`) and turns the cross-cell work into a two-phase gather that rebuilds the full fp16 `hidden [T, I_full]` on every cell before the down GEMM — each cell then writes one `Y[:, Hc]` output shard, so there is no post-down ReduceSum. The host stitches the shards and compares with `golden.bin`. Pure-N-cut requires `--model-tile` (H) divisible by the cell count (`grid-rows * grid-cols`) so `Hc` is an integer width, and full `I` (`ffn-tile * cells`) divisible by the cell count.

## How to Run

### Build only

```bash
bash run_treduce_reducesum.sh    --build-only
bash run_tbroadcast_allgather.sh --build-only
```

### Run on NPU

The scripts default to the DeepSeek-v4 Pro shapes (4×8 = 32 cells), so a plain invocation runs the real shape:

```bash
bash run_treduce_reducesum.sh    -r npu -v Ascend910B1 --device-id 0
bash run_tpush_reducesum.sh      -r npu -v Ascend910B1 --device-id 0
bash run_tbroadcast_allgather.sh -r npu -v Ascend910B1 --device-id 0
bash run_tpush_allgather.sh      -r npu -v Ascend910B1 --device-id 0
```

On a shared host, wrap each run in `task-submit` (e.g. `task-submit bash run_treduce_reducesum.sh -r npu --device-id 0`).

### GridPipe smoke tests

```bash
# Single-source broadcast: 1x5 row, source at col 2 (use --span-col 1 + Rx1 grid for a column broadcast)
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

The smoke script accepts `--build-only` and needs no data generation.

### Common arguments

```text
-r, --run-mode      sim or npu, default npu (sim/camodel fail aclrtSetDevice 507033)
-v, --soc-version   default Ascend910B1
-n, --n-ranks       fixed to 1
-d, --device-id     selected ACL device id; defaults to TASK_DEVICE, FFN_GRID_DEVICE_ID, ASCEND_DEVICE_ID, DEVICE_ID, then 0
--grid-rows         logical grid row count, default 4
--grid-cols         logical grid column count, default 8
--token-tile        token tile T (M) per cell, default 8
--model-tile        hidden dim H, default 7168; pure-N-cut requires H % (grid-rows*grid-cols) == 0
--ffn-tile          per-cell intermediate dim I_shard, default 96 (full I = ffn-tile*cells = 3072; must divide evenly by cells)
--phys-cores        physical AICores to emulate on, default 24 (waves are sized from this; <32 forces a multi-wave launch)
--build-only        build only; skip data generation and execution
```

The broadcast smoke script reuses `-r/-v/-d`, `--grid-rows/--grid-cols`, `--token-tile/--model-tile` (tile `[T, W]`), and `--build-only`; it adds `--src`, `--span-col`, and `--subrect` with `--rect-r0/r1/c0/c1` / `--rect-src` to scope a sub-rectangle.

## Expected Result

On success, each FFN executable prints its bit-exact verdict:

```text
[SUCCESS] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TBROADCAST AllGather PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH AllGather PASS.
```

The smoke tests print:

```text
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```

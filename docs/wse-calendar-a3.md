# WSE Calendar functional simulation on A3

The experimental headers under `include/pto/wse/` provide real A3 PTO
computation and a functional adapter for hardware mechanisms absent from A3.
They are opt-in and do not change the existing `pto-inst.hpp` instruction ABI.

`a3.hpp` implements tiled FP16/FP32 GEMM using `TLOAD`, `TMOV`, `TMATMUL`,
`TMATMUL_ACC`, and `TSTORE`; SiLU/product/cast use PTO vector instructions.
Local tensor storage is addressed through a page-padded slot selected by the
logical-to-physical mapping. The original FFN uses 6144 bytes of L1, 2048
bytes each of L0A/L0B, 1024 bytes of L0C, and a maximum UB address of 65536.

## Calendar API and route layout

`CalendarKernel` is the compiler-facing adapter. It builds
`CalendarTables<CAL_ALLGATHER>`, private `CalendarState`, and `CoreBinding`,
then invokes `TCalendar` with explicit logical rank placement. The public
templates are under `pto::wse`; supported operations are intentionally limited
to the static AllGather needed by the two-round FFN.

| Entry | Route bytes | Additional metadata | Expected count | Size/alignment |
| --- | --- | --- | --- | --- |
| `CalendarRouteEntry` (ABI 2) | 10, for 40 nodes | landing count and flags | offset 12, u32 | 16/16 bytes |
| `CalendarRouteEntry48` (ABI 3) | 12, for 48 nodes | inferred from bitmap/profile | offset 12, u32 | 16/16 bytes |

The 48-node profile is an explicit simulation adaptation for the whitepaper's
6x8 topology. It does not silently truncate the original 80-bit route bitmap.
Deployment must choose the same profile as the compiled kernel.

Each node has two route bits: transit `P` and landing `L`; landing without
transit is illegal. Host validation checks the complete induced tree. Device
code reads deployed route entries, checks destination landing bits and logical
offsets, and reports bytes actually committed by the adapter.

## Mock fabric and lifecycle

`mock::wait_noc_to_ubuf` arms the receive arena before any delivery.
`mock::copy_ubuf_to_noc_ubuf_align_b8` performs native UB insertion and tracks
remote valid bytes. The adapter pulls immutable source backing from the peer
at the receiving worker; it does not emit a nonexistent WSE instruction.
Only this adapter reads peer-owned slots. Host stage synchronization supplies
the producer visibility and full-domain drain that real WSE would implement
with fabric/Batcher mechanisms.

This receiver-driven model allows 32 logical workers on an A3 device with
fewer concurrently resident cores. No worker spins waiting for an unscheduled
peer. It models collective ordering and buffer ownership, not NoC cycles,
packet retries, link contention, or WSE throughput.

`CalendarStatusWords=16` reserves a distinct 64-byte scalar cache line for each
worker. The first six i32 words are error, committed remote bytes, epoch,
binding ID, logical ID, and physical ID. Remaining words are padding. Packed
24-byte records are unsafe because independent A3 scalar cache writebacks can
overwrite neighboring records.

`cpu.hpp` models receiver packet identity for CPU contract tests. It rejects
stale epoch/launch, conflicting duplicates, overlapping segments, out-of-bounds
writes, and byte-count overflow; exact duplicate segments are idempotent.
The A3 transport has no retransmission generator, so duplicate-packet behavior
is a CPU-model property rather than a hardware-retry claim.

## Verification

```bash
g++ -std=c++20 -O2 -Iinclude tests/cpu/wse_calendar_contract.cpp -o /tmp/wse-contract
/tmp/wse-contract
bisheng -fPIC -shared -xcce --npu-arch=dav-2201 -O2 -std=c++17 \
  -Iinclude tests/npu/wse/calendar_kernel.cpp -o /tmp/wse-calendar.so
```

The workspace's `tests/a3_calendar.py` drives the native Calendar test on a
queue-locked A3. It validates both rounds under identity/reverse mapping,
bit-exact output, all worker byte counters, and rejection of invalid controls.
`tests/npu/wse/local_gemm_kernel.cpp` isolates native tiled GEMM. The end-to-end
runner additionally executes the modified PyPTO/PTOAS-generated FFN.

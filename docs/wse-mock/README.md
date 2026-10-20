# WSE mock: PTO-ISA kernels on A3

This branch implements native A3 computation and a functional WSE Calendar
adapter. The upstream baseline is
`b75b4bfd020af051aedfd98084db28a59d3457e2`. The PR targets the fork's `wse-mock`
branch; review code is on `wse-mock-calendar-a3`.

## Changes and interfaces

| File | Role |
| ---- | ---- |
| [a3.hpp](../../include/pto/wse/a3.hpp) | Native local compute and receiver-driven mock Calendar |
| [calendar_types.hpp](../../include/pto/wse/calendar_types.hpp) | Route entries, binding/state types and status ABI |
| [cpu.hpp](../../include/pto/wse/cpu.hpp) | Receiver contract with packet identity, overlap and byte-count checks |
| [CPU contract](../../tests/cpu/wse_calendar_contract.cpp) | Stale epochs/launches, duplicate handling and receiver bounds |
| [Native probes](../../tests/npu/wse) | Calendar and local GEMM kernels |

The opt-in `pto::wse` namespace provides `LocalGemm`, `LocalSilu`, `LocalMul`,
`LocalCast`, and compiler-facing `CalendarKernel`. Existing `pto-inst.hpp`
instruction interfaces are unchanged. GEMM uses native PTO matrix operations;
activation/cast and Calendar copies use native vector/DMA operations.

`CalendarKernel` builds `CalendarTables`, `CalendarState` and `CoreBinding`
and invokes `TCalendar`. Only the transport adapter can read peer-owned private
backing slots. Receiver arming precedes delivery, and committed remote bytes
exclude the local copy. Host stage drains allow 32 logical workers to execute
on A3 without requiring 32 simultaneously resident cores.
See the [API and protocol reference](../wse-calendar-a3.md).

## Build and run

From this checkout:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude \
  tests/cpu/wse_calendar_contract.cpp -o /tmp/wse-contract
/tmp/wse-contract
bisheng -fPIC -shared -xcce --npu-arch=dav-2201 -O2 -std=c++17 \
  -Iinclude tests/npu/wse/calendar_kernel.cpp -o /tmp/wse-calendar.so
```

The bisheng command compiles a native probe; loading it alone does not execute
or validate a kernel. For complete FFN execution, use matching
`wse-mock-calendar-a3` branches of
[PyPTO](https://github.com/chenshengxin2026/pypto/tree/wse-mock-calendar-a3),
[PTOAS](https://github.com/chenshengxin2026/PTOAS/tree/wse-mock-calendar-a3)
and [simpler](https://github.com/chenshengxin2026/simpler/tree/wse-mock-calendar-a3).
After all PRs merge, use all four `wse-mock` branches together.
The [cross-repository run guide](https://github.com/chenshengxin2026/simpler/blob/wse-mock-calendar-a3/docs/wse-mock/README.md)
supplies portable build and CPU-golden validation scripts for a reserved A3.

## Hardware constraints and validation

Local DRAM uses private 2 KiB-aligned slots, capped at 384 MiB per worker.
Scratch fits both WSE and A3 capacities. For the original FFN, UB reaches
65,536 B, L1 uses 6,144 B, L0A/L0B use 2,048 B each, and L0C uses 1,024 B.
Status records occupy distinct 64-byte scalar cache lines.

Both route profiles have 16-byte size/alignment and expected receive bytes at
offset 12: ABI 2 uses 40 nodes/80 route bits and ABI 3 uses 48 nodes/96 bits.
ABI 3 infers the redundant landing-count/flags metadata to accommodate the
whitepaper topology. This is an explicit simulation choice.

On 2026-10-15, the four-layer stack passed 46 targeted CPU/PTOAS tests,
the separate C++ receiver contract, native Calendar/GEMM probes, and 21 FFN
executions on Ascend910_9392. Original shape `T=8,H=7168,I=3072,P=32` passed
two seeds, three deployments, and two repeats, with maximum absolute error
`0.015625` and maximum relative RMS `8.2343e-5`. Thresholds were pointwise
`atol=0.002, rtol=0.01`, relative RMS `<0.002`.

Both gathers were bit-exact against device source shards at every worker,
with `1,523,712 B` remote committed bytes per original-case run.
Same-seed output was bit-exact across mappings/repeats.
PG/Batcher/new CCE mechanisms are mocked, and mapping/NoC algorithms are stubs.
The CPU receiver tests duplicate/stale-packet handling; the A3 adapter has
no retry generator. Validation covers functional behavior and modeled
constraints, without claiming WSE NoC timing or throughput.

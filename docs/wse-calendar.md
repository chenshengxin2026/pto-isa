# WSE Calendar and the A3 CCE mock

The `wse-sim-1026` branch implements a synchronous Calendar subset in
`include/pto/wse/`. See the
[four-repository FFN guide](https://github.com/chenshengxin2026/pypto/blob/wse-sim-1026/docs/en/dev/wse-a3.md)
for a complete PyPTO → PTOAS → PTO-ISA → A3 execution and accuracy check.

- `calendar_contract.hpp`: static geometry and ordered members, axis-dependent
  offsets/strides, per-source receive validation and remote-only byte accounting.
- `calendar.hpp`: `TCalendar<Plan>` completes dependencies, enters BSP, arms
  receive, sends, waits for validated commits and returns synchronously.
  `RunCalendar<axis, members...>` adapts the compiler's context and tile operands.
- `cce_mock_a3.hpp`: mock CCE entry points `__sync_bsp`, `wait_noc_to_l1` and
  `copy_l1_to_noc_l1_align_b8`, backed by real A3 cbuf/GM DMA and scalar publication.

Receive arming is nonblocking. The send publishes its payload and geometry
after DMA completion. Receivers pull published shards into L1, count each
remote source only after DMA completes, and copy their local slice without
advancing the remote byte count. Epoch and geometry checks prevent stale,
duplicate, misrouted or oversized completion. BSP and payload polling are
bounded; the host checks the per-member error/byte/epoch status.

The 128-byte context contains logical ID, group rank, synchronization domain,
epoch, route/slot/mailbox/signal/status pointers, member count and source bytes.
Each signal/status record occupies a separate 64-byte cache line. The simpler
adapter owns those resources through stream completion. Its paired table stub
provides a complete audience mask and deterministic slot IDs. This validates
the Calendar data contract, not native NoC timing, arbitration or throughput.

CPU contract checks (no device required):

```bash
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/cpu/wse/calendar_contract.cpp -o /tmp/wse-contract
/tmp/wse-contract
```

The small transport probe is `tests/npu/wse/calendar_smoke.cpp` with the host
driver in the matching simpler branch at `tests/st/a2a3/wse/calendar_smoke.py`.
The complete FFN guide runs both physical orderings, multiple epochs, axis 0
and singleton groups, bit-exact gather checks, and CPU-reference FFN accuracy.
Every device command must run inside `task-submit`.

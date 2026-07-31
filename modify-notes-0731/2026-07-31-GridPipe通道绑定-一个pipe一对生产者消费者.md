# GridPipe 通道绑定（一个 pipe = 一对「生产者 / 消费者」）

- 日期：2026-07-31
- 分支：`backup/grid-0731`（fork = `chenshengxin2026/pto-isa`）
- 基线：`9e267a04`（上一篇笔记：寻址与搬运解耦 SlotStride / DirMask / GridPayloadWindow）
- 提交：`163c56e3`，22 文件 / +1106 −940
- 验证：NPU 实跑 4 demo + 2 smoke（bcast 含 ROW/COL/SUBRECT 三态），**8/8 PASS，max diff 全 0**

---

## 1. 背景：一个 pipe 对象承载了十个通道端点

改动前的 `GridPipe` 是这样的：

```cpp
template <typename TileT_, int SlotStride_, int SlotCount_,
          int BcastSlotCount_ = 0, int GroupMax_ = 0, int DirMask_ = kGridDirAll>
struct GridPipe {
    GridShape shape{};  GridCoord coord{};  GridRect groupRect{};
    __gm__ uint8_t*  slotBase [kGridDirectionCount] = {nullptr};   // 5
    __gm__ uint32_t* readyScb [kGridDirectionCount] = {nullptr};   // 5
    __gm__ uint32_t* freeScb  [kGridDirectionCount] = {nullptr};   // 5
    uint32_t         prodIndex[kGridDirectionCount] = {0};         // 5
    uint32_t         consIndex[kGridDirectionCount] = {0};         // 5
    GridPayloadWindow pushWindow[kGridDirectionCount] = {};        // 5
    GridPayloadWindow popWindow [kGridDirectionCount] = {};        // 5
    __gm__ uint8_t*  bcastRingBase; __gm__ uint32_t* bcastReadyLanes, *bcastFreeLanes;
    __gm__ void* runtimeCtx;  uint32_t pipeId;  GridPayloadWindow bcastWindow{};
};
```

一个 pipe 对象同时容纳 5 个方向 × 2 个角色 = **最多 10 个通道端点**，外加一整套广播态。
于是：

- `TPUSH<EAST>(pipe, t)` 和 `TPUSH<WEST>(pipe, t)` 写同一个对象，**对端在两次调用之间悄悄换了**；
- 一个纯广播 pipe（`DirMask = kGridDirNone`）仍然带着 5 组单播 scb/idx/window 字段，全是死区；
- 从代码上根本读不出「这个 pipe 复用给谁了」——这就是本次要修的「复用逻辑展示不清」。

**为什么这不只是审美问题。** pipe 里存的全部是**核内寄存器状态**：真实硬件上
`ready_scb` / `free_scb` 是 IPC_SCB（SPR），`prod_idx` / `cons_idx` 是 GPR。
寄存器不可能中途改指向另一个对端。之所以看起来能多路复用，只是因为 A3 上用 GM
window 模拟 SRAM，而 GM 里任何核的 window 都能寻址 —— **按 mock 的自由度设计接口，
而不是按寄存器模型设计**，就会出现这种「一个对象十个端点」的结构。

---

## 2. 改了什么

### 2.1 两个 pipe 模板，各自绑定自己的对端

```cpp
// 单播 SPSC 通道：生产者 = 沿 Dir 上游 Dist 跳的核，消费者 = 下游 Dist 跳的核
template <typename TileT_, GridDirection Dir_, int SlotStride_, int SlotCount_,
          int Dist_ = 1, int ScbId_ = 2 * static_cast<int>(Dir_)>
struct GridPipe { ... };

// 组集合 MPSC 通道：生产者/消费者 = 整个 group
template <typename TileT_, GridGroup Group_, int SlotStride_, int SlotCount_, int GroupMax_>
struct GridGroupPipe { ... };
```

对端身份进了**类型**，所以「换生产者或消费者 ⇒ 定义新 pipe」是编译期强制的。
广播态独立成 `GridGroupPipe` 的理由是它本来就是**另一种通道形状**：
单个 ready/free 计分板 → per-source lane 数组；私有环 → 全组共享环（按全局 index 寻址）；
且 group 一换，每一个对端都换。原来的 `BcastSlotCount` 就是它的 `SlotCount`，
`groupRect` 就是它的 `rect`。

### 2.2 字段按三类信息分组

| 成员 | 内容 | 硬件对应 |
| --- | --- | --- |
| `pipe.ctx`（`GridPipeCtx`） | `runtimeCtx`（全局上下文实例，用于寻址对端）+ `shape` / `coord` / `pipeId` | 运行时句柄 + 本核在网格中的坐标 |
| `pipe.slots`（`GridSlotRing<Stride,Count>`） | ring 基址 + `SlotStride`（单 slot 大小）+ `SlotCount`（长度）+ `Slot(idx)` | 本核 SRAM 里的 payload ring |
| `pipe.prod`（`GridProducerSem`） | `freeScb` + `prodIndex` + payload 子窗口 | IPC_SCB（SPR）+ GPR |
| `pipe.cons`（`GridConsumerSem`） | `readyScb` + `consIndex` + payload 子窗口 | IPC_SCB（SPR）+ GPR |

`GridPayloadWindow` 从「按方向的两个数组」变成**属于某一侧信号量**的字段
（`prod.window` / `cons.window`），正好对上 a5 `TPipe::Producer::entryOffset` /
`Consumer::entryOffset`。组 pipe 的 `SetWindow()` 一次写两侧（集合通信两半是同一几何）。

`prod` / `cons` 同在一个 pipe 里，是因为 SPMD 下同一个核在一条链上兼任两角：
消费上游产出、为下游生产。它们是**两个结构体**，因为那是两组绑到不同对端的寄存器 ——
与 a5 `TPipe` 的 `prod` / `cons` 同构。

### 2.3 指令签名：方向/距离/组不再是调用参数

```cpp
TPUSH(pipe, tile)                    // 原 TPUSH<Dir, Dist>(pipe, tile)
TPOP(pipe, tile)                     // 原 TPOP<Dir, Dist>(pipe, tile)
TREDUCE<Op>(pipe, acc, recv)         // 原 TREDUCE<Dir, Op, Dist>(...)  —— Op 是操作不是对端，保留
TBROADCAST(pipe, tile)               // 原 TBROADCAST<Group>(pipe, tile)
TPOP(pipe, tile, srcRank)            // 原 TPOP<Group>(pipe, tile, srcRank)
```

`Pipe::Dir` / `Pipe::Dist` 仍是编译期常量，设计文档 4.2「方向在下译时必须是常量」的
约束不变。`GRID_TRY_*_IMPL` 同步改成 `<Pipe, Tile>`。

kernel 侧的角色判断从「重新用 coord/shape 推」改成**直接问 pipe**：

```cpp
const bool isFwdSource = !fwdPipe.HasProducer();  // 无上游 → 链首
const bool isFwdSink   = !fwdPipe.HasConsumer();  // 无下游 → 链尾
```

配套访问器：`ProducerRank()` / `ConsumerRank()` / `SelfRank()`；
组 pipe 是 `GroupSize()` / `SelfGroupRank()` / `MemberRank(k)`。

### 2.4 `DirMask` 删除，`ScbId` 补位

`DirMask` 及其全套辅助（`GridDirBit` / `kGridDirAll` / `kGridDirNone` /
`GridDirInMask` / `GridDirRingCount` / `GridDirRingIndex`）**全部删除** —— 它存在的唯一
理由就是「把多个方向的环塞进一个 pipe」，而这正是本次要禁止的事。一个 pipe 一个环，
`RingCount` 恒等于 1。

但 `wait_ipc_scb(localScb, threshold, slot)` 的 `slot`（native IPC_SCB 槽号）原来是
`dirIdx` / `kGridDirectionCount + dirIdx` 算出来的，方向数组一去就没了来源。补成
**尾部模板参数 `ScbId`**（ready = `ScbId`，free = `ScbId + 1`，`static_assert(ScbId+1 < 16)`），
默认 `2 * Dir` 复现「每方向一对计分板」：

| Dir | SOURCE | NORTH | EAST | WEST | SOUTH |
| --- | --- | --- | --- | --- | --- |
| ready / free slot | 0 / 1 | 2 / 3 | 4 / 5 | 6 / 7 | 8 / 9 |

这就是 a5 `TPipe<FlagID, ...>` 的同构物：同一个核上共用同一方向的两个 pipe 必须显式给
不同 id。A3 mock 读 GM 字、忽略槽号；native `WAIT_SPR` 用它。

### 2.5 window 布局：一个 pipe 一个窗口

```
单播 pipe:   [0]   ready scb (u32)      ← 生产者对端 SYNC_HSCB(READY) 写
             [4]   free  scb (u32)      ← 消费者对端 SYNC_HSCB(FREE)  写
             [8..127] 保留（fault 哨兵在 +kFaultFlagWordOffset = word 10/11）
             [128] slot ring  [SlotCount * SlotStride]

组 pipe:     [0..127] 保留（信号量就是下面的 lane，这里没有计分板对）
             [128] 共享 MPSC ring [SlotCount * SlotStride]
             [+]   per-source ready lanes [GroupMax * 64B]
             [+]   per-source free  lanes [GroupMax * 64B]
```

需要两个通道的核就在 cell arena 里切两段，前向在前、后向在后
（`FFN_NCUT_TPUSH_PIPE_WIN_P*` / `FFN_RS_REDUCE_PIPE_WIN`）。
**两段必须不重叠** —— 否则第二相会从第一相留下的 ready/free 计数起步。
以前这个风险被「5 组计分板共用一个 flags 块、EAST 用 word2 / SOUTH 用 word4」掩盖了，
现在它变成了一条显式的布局约束（`ffn_config.hpp` 里写进了注释）。

每 cell window 字节数（before → after）：

| 例子 | before | after | 差 |
| --- | --- | --- | --- |
| tpush AllGather P1（EAST+WEST，slot 12288×2） | 49280 | 2×24704 = 49408 | +128 |
| tpush AllGather P2（SOUTH+NORTH，slot 49152×2） | 196736 | 2×98432 = 196864 | +128 |
| ReduceSum（EAST+SOUTH，slot 32768×7） | 458880 | 2×229504 = 459008 | +128 |
| tbroadcast AllGather P1 / P2（组 pipe） | 13440 / 49792 | 13440 / 49792 | 0 |

即每多一条通道多付一个 128 B flag header，环字节数不变；组 pipe 逐字节不变。

### 2.6 调用侧拆分

| 例子 | before | after |
| --- | --- | --- |
| `tpush_allgather` 中继 | 1 个 pipe，`DirMask = EAST\|WEST` | `FfnGatherPipeP1Fwd`(EAST) + `P1Bwd`(WEST)；P2 同理 SOUTH/NORTH |
| `tpush_reducesum` | 1 个 pipe，`DirMask = EAST\|SOUTH`，两相各自 init | `FfnReduceRowPipe`(EAST) + `FfnReduceColPipe`(SOUTH)，窗口分两段 |
| `tbroadcast_allgather` | `GridPipe<..., kGridDirNone>` 带广播区 | `GridGroupPipe<..., GridGroup::ROW/COL, ...>` |
| `khop_smoke` | `DirMask = EAST`，`TPUSH<EAST, KHOP_DIST>` | `GridPipe<Tile, EAST, ..., KHOP_DIST>`，`TPUSH(pipe, t)` |
| `bcast_smoke` | `TBROADCAST<kGroup>` | `GridGroupPipe<..., kGroup, ...>`，`TBROADCAST(pipe, t)` |
| `treduce_reducesum` | 有个从未实例化的 `FfnReducePipe` 死别名 | 删除（组归约 `GRID_TREDUCE_GROUP_IMPL` 不接 pipe，也不该假装需要） |

host 侧 fault 扫描从「每 cell 扫一个 128 B 头」改成「每 **pipe** 扫一个头」
（`main_tpush_allgather.cpp` / `main_tpush_reducesum.cpp` / `main_treduce_reducesum.cpp`）。
另外顺手修了 `main_tbroadcast_allgather.cpp` 里 `DumpArenaLanes` 的一个**既有错误**：
它按 `FFN_NCUT_GRID_DIRECTION_COUNT(5) * SLOT_COUNT * slotBytes` 计算 lane 偏移，
但那两个 pipe 是 `kGridDirNone`（0 个单播环），所以这个调试 dump 一直在读错地址。

---

## 3. 设计取舍

### 3.1 为什么把 `Dir`/`Dist`/`Group` 放进类型而不是运行期字段

两者都能满足「一个 pipe 一对生产消费者」，但：

- **类型级**把误用变成编译错误，而不是运行期踩别人的 window（跨核越界写比本地越界难查得多）；
- 设计文档 4.2 要求方向在下译时是常量，运行期字段会让 native `WAIT_SPR` 的槽号
  操作数无法折叠成立即数；
- 与 a5 `TPipe<FlagID, DirType, ...>` 同构 —— 那边 `FlagID` 和方向也都在类型里。

### 3.2 为什么拆成两个 pipe 模板而不是一个带 `if constexpr`

单播 pipe 的信号量是**一对字**，组 pipe 的是**两个 lane 数组**；单播 pipe 有
`prod_idx`/`cons_idx` 两个 GPR，组 pipe 在 `count_k = 1` 下全局 index 就等于 rank，
根本没有 run-counter。把它们塞进一个结构体只能靠「另一半字段恒为 null」——
那正是这次要消灭的东西。

### 3.3 `GridSlotRing` 为什么把 stride/count 留在编译期

用户列的「Slot 信息」是「基址、单 slot 大小、长度」三样。后两样做成运行期字段会丢掉
`(idx % SlotCount) * SlotStride` 的常量折叠，而且没人要求它们运行期可变。折中做法是
把三样**放进同一个命名的地方**：`GridSlotRing<Stride, Count>` 带 `base` 字段 +
两个 `static constexpr`，pipe 再把 `Pipe::SlotStride` / `Pipe::SlotCount` 转发出去，
既让三类信息在结构上聚齐，又不动既有的编译期语义。

---

## 4. 踩坑记录

### 4.1 重载歧义：grid `TPUSH(pipe, tile)` 撞上 TPipe 的反序重载

去掉模板参数后，`TPUSH(gridPipe, tile)` 和 `pto_instr.hpp` 里
`template <typename TileData, typename Pipe, typename... WaitEvents>
TPUSH(TileData& tile, Pipe& pipe, ...)` 完全同形（都是 `(T1&, T2&, Pack&...)`），
偏序不分胜负 ⇒ 歧义。给反序重载加负向约束，试了三种写法：

| 写法 | 结果 |
| --- | --- |
| `std::enable_if_t<!is_any_grid_pipe_v<TileData>, int> = 0`（多一个非类型参） | ✗ 第三个参会吞掉 `TPUSH<Pipe, TileProd, Split>` 的 `Split`（枚举值可转 int）⇒ 那条调用变歧义 |
| `typename = std::enable_if_t<...>`（多一个类型参） | ✗ 参数列表变成 `<typename, typename, typename, typename...>`，与 4 参数的 `TConfig` 重载**重定义** |
| `PTO_INST std::enable_if_t<..., RecordEvent> TPUSH(...)`（约束放**返回类型**） | ✓ 模板参数列表原样保持 `<typename, typename, typename...>` |

`TPOP` 同理。另外 `is_any_grid_pipe_v` 在 `__CPU_SIM` 下要有 `= false` 兜底 ——
`pto_instr.hpp` 的 `#include grid_intrinsic.hpp` 被 `#ifndef __CPU_SIM` 包着。

### 4.2 `ScbId` 不是可有可无的

如果不补 `ScbId`，`wait_ipc_scb` 的 `slot` 操作数就只能传运行期的 `pipeId`，
native 分支上无法折叠成立即数。A3 mock 忽略这个参数，所以这个坑在本地测不出来 ——
属于「现在不补、切 `PTO_GRID_CCE_NATIVE` 时才炸」的那一类。

### 4.3 window 分段必须显式检查

`tpush_reducesum` 的 EAST 相和 SOUTH 相是两次独立 launch，共用同一块 cell window。
以前靠 `readyScb[EAST]` 和 `readyScb[SOUTH]` 天然错开；现在两个 pipe 若都从
`window + 0` 起，SOUTH 相的第一个 TPOP 会看到 EAST 相留下的 `ready = 7` 而立刻
放行读到垃圾。已在 `ffn_config.hpp`（`FFN_RS_REDUCE_PIPE_WIN`）和 kernel 注释里写死这条约束。

---

## 5. 验证

| 用例 | 结果 |
| --- | --- |
| `run_treduce_reducesum.sh -r npu` | PASS，max diff 0 |
| `run_tpush_reducesum.sh -r npu` | PASS，max diff 0 |
| `run_tbroadcast_allgather.sh -r npu` | PASS，max diff 0 |
| `run_tpush_allgather.sh -r npu` | PASS，max diff 0 |
| `smoke/run_khop_smoke.sh -r npu`（EAST dist=2） | PASS，max diff 0 |
| `smoke/run_bcast_smoke.sh -r npu`（ROW src=2，1×5） | PASS，max diff 0 |
| 同上 `--span-col 1`（COL src=1，4×1） | PASS，max diff 0 |
| 同上 `--subrect 1`（3×4 rect=[r1:3,c1:4] src=2，走非等步长 fallback） | PASS，max diff 0 |

另：`pre-commit run clang-format`（CI 固定 v18.1.8）Passed；`scripts/oat_check.sh` 22 文件全过。
`README.md` / `README_zh.md` 的 API 记法同步更新，并新增「GridPipe 通道绑定」一节。

---

## 6. 与上一篇的关系 / 剩余差距

上一篇（`9e267a04`）解的是**寻址步长 vs 搬运长度**的解耦（`SlotStride` + `GridPayloadWindow`）；
这一篇解的是**通道身份**的绑定。两者正交：前者让一条通道能只搬有效前缀，后者让「这条通道
是谁到谁」在类型上说得清。

相对 a5 `TPipe` 仍然存在的差距（见 `2026-07-31-Pipe与原TPipe实现差距分析.txt`）：

1. **credit 稀疏化未补** —— a5 有 `SyncPeriod = SlotNum/2` +
   `setWaitStatus`/`setFreeStatus`/`setRecordStatus`/`setAllocateStatus`，
   N 次搬运换 1 次 credit；Grid 仍是每 push 一次 `sync_hscb(READY)`、每 pop 一次
   `sync_hscb(FREE)`。这条现在更好补了：状态位天然属于 `prod` / `cons` 结构体。
2. **本地落地环缺失** —— a5 同时维护 GM 环与本地 UB/L1 环（两个不同步长）；
   Grid 只有一个环，消费侧 tile 由 caller 提供。
3. **无真正的 Grid TALLOC** —— ISA 级限制：对端 L1 只能经 `COPY_UBUF_TO_NBR` 的
   `(dir, dist, nbr_off)` 寻址，拿不到能交给 `TSTORE` 的 `GlobalTensor`。

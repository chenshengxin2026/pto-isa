# GridPipe 寻址与搬运解耦（SlotStride / DirMask / GridPayloadWindow）

- 日期：2026-07-31
- 分支：`backup/grid-0731`（fork = `chenshengxin2026/pto-isa`）
- 基线：`24edffa4`（把 `origin/main 1f61a9a2` 合入 `feat/grid-tbroadcast-treduce-group-cce`）
- 提交：`8697793f` → `a5f87ead` → `bf42ade4`，共 21 文件 / +534 −90
- 验证：NPU 实跑 4 demo + 2 smoke，**6/6 PASS，max diff 全 0**

---

## 1. 背景：一个 `SlotBytes` 同时干了四件事

改动前 `GridPipe<TileT, SlotBytes, SlotCount, ...>` 的 `SlotBytes` 同时决定：

1. 单播环的寻址步长 `slotOff = (idx % SlotCount) * SlotBytes`；
2. 广播环的寻址步长；
3. window 布局（`grid_pipe_runtime.hpp` 的所有尺寸模板）；
4. **每一次搬运的长度**。

第 4 条是问题所在 —— `GridTPush.hpp` 里同一个常量既当步长又当长度：

```cpp
const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotBytes;   // 寻址步长
CopyTileToNeighborSramSlot<TileProd>(neighborSlot, tile, Pipe::SlotBytes); // 搬运长度
```

对照 a5 的 `TPipe`：`SLOT_SIZE` 只出现在 `entryBase = (tileIndex % SLOT_NUM) * SLOT_SIZE` 这一个乘法里，
**从不作为长度传给任何搬运指令**；长度与排布来自 tile / `GlobalTensor` 的 `Shape`×`Stride` 描述符，
再叠一层运行时 `entryOffset`（`a5/TPush.hpp:78/207/274/289`）。

值得强调的是：**塌缩发生在 PTO 指令层，不在 ISA/CCE 层。**

- `copy_ubuf_to_neighbor_ubuf` 的 V8 操作数是 `(dir, dist, nbr_off, local_off, bytes)` —— 偏移与长度本来就独立；
- `mov_ubuf_group(..., bytes, memberCount, memberStride, ...)` 更是直接把两者分开；
- `GRID_TREDUCE_GROUP_IMPL` 早就在用这个自由度：treduce demo 传的是 `bytes=32768 / memberStride=229376`（1:7）。

所以这次改动是把已有的自由度向上暴露，而不是新增能力。

---

## 2. 改了什么

### 2.1 `8697793f` — 核心机制（5 个头文件，+328 −48）

`include/pto/npu/a2a3/{grid_intrinsic,grid_pipe_runtime,GridTPush,GridTPop,GridTBroadcast}.hpp`

**(b) `SlotStride` 正名 + `DirMask`**

```cpp
template <typename TileT_, int SlotStride_, int SlotCount_,
          int BcastSlotCount_ = 0, int GroupMax_ = 0,
          int DirMask_ = kGridDirAll>          // ← 新增第 6 个参数
struct GridPipe {
    static constexpr int SlotStride = SlotStride_;  // 只管寻址
    static constexpr int SlotBytes  = SlotStride_;  // 兼容别名
    static constexpr int DirMask    = DirMask_;
    static constexpr int RingCount  = popcount(DirMask_);
```

window 的 slot 区从 `5 * SlotCount * SlotStride` 变成 `RingCount * SlotCount * SlotStride`，
环下标紧凑打包（`GridDirRingIndex`）。默认 `kGridDirAll` ⇒ `RingCount == 5` 且
`ring(dir) == GridDirectionIndex(dir)`，**与原布局逐字节一致**。
`TPUSH<Dir>` / `TPOP<Dir>` 增加 `static_assert`：方向必须在 mask 内。

**(a) `GridPayloadWindow`**

```cpp
struct GridPayloadWindow {
    uint32_t entryOffset = 0;  // 槽内字节偏移
    uint32_t rowBytes    = 0;  // 每行搬运字节
    uint32_t rowCount    = 0;  // 0 = 关闭（整槽 1-D，= 改动前行为）
    uint32_t tileStride  = 0;  // tile 内行距（0 ⇒ rowBytes）
    uint32_t slotStride  = 0;  // 槽内行距（0 ⇒ rowBytes）
};
```

按方向存在 pipe 上（`pushWindow[5]` / `popWindow[5]` + 一个 `bcastWindow`），
配 `SetPushWindow / SetPopWindow / SetBcastWindow / Reset*`。

两点设计取舍：

- **为什么是 2-D 而不是 1-D。** 中继 tile 是行主序，"前 k 个分片有效"是**列**前缀，
  不是字节前缀。1-D `(offset, length)` 表达不了，只能靠 2-D `rowCount × rowBytes` 描述。
- **为什么 stride 按 buffer 命名而不是 src/dst。** push 读 tile 写 slot、pop 读 slot 写 tile，
  src/dst 在两半里含义互换 —— 这个项目历史上已经因为 stride 弄反出过两次精度 bug
  （phase-D `GY` 用了 `kIfull` 而非 `kHfull`）。用 `tileStride`/`slotStride` 从命名上消除歧义。

下译：`rowCount > 1` 时发 `rowCount` 次 `COPY_UBUF_TO_NBR`，
**但仍然只发 1 次 ready doorbell** —— 每次 TPUSH 的握手开销不变（不恶化差距 3）。

**差距 5 运行时越界断言**

长度变成运行时值以后，没有任何静态约束兜底，而**越界的 push 是写进对端的 window**（跨核静默踩内存）。
a5 靠描述符自洽天然规避，Grid 必须显式检查：

```cpp
if (GridPayloadSlotExtent(win, Pipe::SlotStride) > Pipe::SlotStride) {
    grid_mock::MockSetFault(rangeFault, grid_mock::kFaultPushPayloadRange);
    return false;
}
```

新增 `0x401` push / `0x402` pop / `0x403` broadcast。

**一个编译期坑**：`DirMask` 系列 helper **不能**加 `AICORE`。
`grid_pipe_runtime.hpp` 的 window 尺寸模板是 host 侧 `constexpr`，
而 `[aicore]` 函数不能从 host 上下文调用（`error: no matching function for call to 'GridDirRingCount'`）。
所以这几个 helper 是裸 `inline constexpr` 且内部不调用任何 AICORE 函数（用 `static_cast<int>(d)` 而非 `GridDirectionIndex`）；
反过来，device 侧（`static_assert`、`RingCount` 初值、`InitGridPipeFromWindow`）一律用**展开的位运算**，不调这些 helper。

### 2.2 `a5f87ead` — demo / smoke 落地（10 文件，+170 −42）

| pipe | 用到的方向 | DirMask |
|---|---|---|
| tbroadcast P1/P2、bcast smoke | 无单播（走广播环） | `kGridDirNone` |
| tpush_allgather P1 | EAST + WEST | 中继对 |
| tpush_allgather P2 | SOUTH + NORTH | 中继对 |
| tpush / treduce reducesum | EAST + SOUTH | 行后列 |
| khop smoke | EAST | 单向 |

payload 钩子新增 2-D 形式 `CopyTileToNeighborSramSlot2D` / `CopyLocalSlotToTile2D`。

**中继只搬有效前缀**（`FfnRelayGather`）：

```cpp
const uint32_t fwdPushCols = ownOff + unitCols;   // 插入自己后有效的列数
const uint32_t fwdPopCols  = ownOff;              // 上游 rank k-1 发来的列数
const GridPayloadWindow fwdPushWin{0, fwdPushCols * kElemBytes, kRowsU, kRowPitch, kRowPitch};
```

接收方**自己从 rank 推**出窗口（上游发了 `k*unitCols` 列，正好等于本 cell 的 `ownOff`），
不需要随数据传递元信息 —— 和 a5 生产者/消费者各自从 tile id 推 `entryOffset` 是同一套做法。
后向 pass 携带完整 tile，`Reset*Window` 回到整槽。

### 2.3 `bf42ade4` — host 侧（6 文件，+36）

- **构建解阻塞**：合入 main 后 `common.hpp` 经 `comm_types.hpp → pto_tile.hpp` 传递性拉入
  `SdmaWorkspaceManager`，host 编译器没有 CCE 地址空间属性，六个 host main 全部报
  `decomposition declaration '[aicore]' requires an initializer`。
  `common.hpp` 自己提供了 opt-out，按 `moe_dispatch/main.cpp` 的写法在 include 前
  `#define PTO_COMM_ST_SKIP_SDMA_WORKSPACE_MANAGER`（这些二进制都不用 SdmaWorkspaceManager）。
- **fault 名字**：`GridPipeFaultName` 补 `0x401/0x402/0x403`，越界会打印
  `code=0x401 (push payload window out of slot range)` 而不是裸十六进制。

---

## 3. 收益

### 3.1 window（每 cell）

| pipe | 前 | 后 | 降幅 |
|---|---:|---:|---:|
| tbroadcast P1 | 21120 | 13440 | −36% |
| tbroadcast P2 | 111232 | 49792 | −55% |
| tpush P1 | 123008 | 49280 | −60% |
| tpush P2 | 491648 | 196736 | −60% |
| reducesum | 1147008 | 458880 | −60% |
| **合计** | **1894016** | **768128** | **−59%** |

32 cell 合计省 **34.4 MiB**。

### 3.2 中继线上字节

| 阶段 | 前 | 后 |
|---|---:|---:|
| P1 行聚合（×4 行） | 688128 | 516096 |
| P2 列聚合（×8 列） | 2359296 | 1769472 |
| **合计** | **3047424** | **2285568（−25%）** |

前向段 1,523,712 → 761,856（**减半**）；链首第 1 跳 12288 B → 1536 B（**×8**）。
后向段本来就全有效，不变。

---

## 4. 顺带修掉的既有缺陷（非本次改动引入）

`smoke/bcast_smoke_config.hpp` 的 per-source ready/free lane 区按 `sizeof(uint32_t)` = 4 B/lane 计算，
而设备侧 `grid_mock::kBcastLaneStride` 早在 2026-07-23（修 packed-cache-line lost-update 时）就改成 **64 B**。
后果：每个 window 少 `2*GroupMax*60` 字节（默认配置 = 600 B），
ready lane 尾部溢出到**下一个 cell 的 window**，接收方永远等不到落在自己窗口内的 doorbell。

表现是 `rc=507014` + `launch+sync ≈ 1.09e9 us` 的超时 —— **看起来完全像 kernel 死锁**。
在未修改的基线树上同样复现（先误判成抢卡，换卡重跑仍挂才定位）。

修复：命名常量 `BCAST_LANE_STRIDE = 64` 并加注释要求与设备侧同步。bcast smoke 恢复 PASS。

> 教训：host 侧 window 镜像与设备侧布局常量必须成对修改；这次是 (b) 重算 window 数学时才暴露出来。

---

## 5. 验证

全部经 `task-submit --device auto` 在 NPU 实跑（clang-format v18.1.8 hook 跑过后重新验证）：

| 目标 | 结果 |
|---|---|
| distributed_ffn_grid_treduce_reducesum | PASS, max diff 0 |
| distributed_ffn_grid_tpush_reducesum | PASS, max diff 0 |
| distributed_ffn_grid_tbroadcast_allgather | PASS, max diff 0 |
| distributed_ffn_grid_tpush_allgather | PASS, max diff 0 |
| khop_smoke | PASS, max diff 0 |
| bcast_smoke | PASS, max diff 0（基线是挂死的） |

**两个对照实验**，确认新代码路径真的在执行、不是静默失效：

1. **正向对照**：把前向前缀砍成 `ownOff + unitCols/2` → 结果 FAILED。
   ⇒ 2-D payload window 确实在承载中继数据。
2. **反向对照**：把 `rowCount` 故意设成 `kRowsU + 1`（越界一行）→
   `[ERROR] GridPipe fault P1 cell=20 row=2 col=4 code=0x401 (push payload window out of slot range)`，
   且在任何跨核写发生**之前**被拦下（下游随之出现 `0x301 wait ready timeout`，符合预期）。
   ⇒ 差距 5 的断言有效。

两个实验后都已把 kernel 逐字节还原（`diff -q` 对比干净备份）并重跑全绿。

---

## 6. 兼容性

- `SlotBytes` 保留为 `SlotStride` 的别名，旧写法不动即可编译。
- `DirMask` 默认 `kGridDirAll` ⇒ window 布局与改动前**逐字节一致**。
- `GridPayloadWindow` 默认全 0 ⇒ `rowCount == 0` ⇒ 整槽 1-D 搬运，与改动前**指令流一致**。

即：不 opt-in 的调用方零回归。

---

## 7. 未覆盖 / 剩余差距

对照《2026-07-31-Pipe与原TPipe实现差距分析》，本次做了 (a)+(b)+差距 5，剩下：

| # | 差距 | 状态 |
|---|---|---|
| 1 | 指令不可拆分（无真正的 Grid `TALLOC`） | **ISA 级，补不了**。a5 的 `TALLOC` 能返回普通 GM 地址让任意 `TSTORE` 写；Grid 的推送目标是对端 L1/SRAM，native 上只能由 `COPY_UBUF_TO_NBR` 的 `(dir,dist,nbr_off)` 编码寻址。Grid 版 `TALLOC` 最多返回**描述符**，不能返回 `GlobalTensor`。注意 A3 mock 里对端 window 是 GM、看着能 TSTORE，切 `PTO_GRID_CCE_NATIVE` 就不成立，别按 mock 的能力设计接口。 |
| 2 | 搬运描述符维度 | **部分缓解**。软件层用 N 次 burst 模拟了 2-D；但仍无 5D `Shape×Stride`、无 NZ2ND/NZ2DN 布局转换、无 fixpipe quant/relu/atomic。真正的一条指令 2-D 需要机器指令新增 `src_stride/dst_stride/nrepeat`。 |
| 3 | credit 稀疏化 | **未做**。仍是每 push 一次 `sync_hscb(READY)`、每 pop 一次 `sync_hscb(FREE)`，没有 a5 的 `SyncPeriod` / `shouldWaitFree` / 逐次可关的 `setWaitStatus/setFreeStatus`。本次的 2-D 下译刻意保持"1 次 TPUSH = 1 次 doorbell"，所以没有让它变差；但若将来把一次大 push 拆成 K 次子 push，必须先补这条轴。 |
| 4 | 本地落地环 | **未做**。GridPipe 仍只有一个环，消费侧 ping-pong 由 kernel 手写（`recvTile`/`relayTile`）。 |

另外记一笔：**`treduce_reducesum` 的 `FfnReducePipe` 是死代码** —— 该变体走
`GRID_TREDUCE_GROUP_IMPL`（`bytes`/`memberStride` 本就解耦），直接从 `partialBuf` 原地读，
不需要 slot 环；kernel 里 `(void)reduceWindow;` 把 host 分配的整个 window 丢掉了。
本次只给它补了一致的 DirMask 与注释；要彻底清理需要同时动 host 侧分配，属于独立改动。

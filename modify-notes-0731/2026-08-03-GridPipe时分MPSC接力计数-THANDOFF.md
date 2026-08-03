# GridPipe 时分 MPSC 接力计数（THANDOFF）

- 日期：2026-08-03
- 分支：`backup/grid-0731`（fork = `chenshengxin2026/pto-isa`）
- 基线：`f25697d6`（上一篇笔记：一个 pipe = 一对生产者/消费者）
- 提交：`3fe7a68b`，15 文件 / +1578 −41
- 验证：NPU 实跑 handoff smoke 3 种配置 ×8 次 + 回归 4 demo & 2 smoke，**全部 PASS，max diff 全 0**

> 同批推送还有 `c355df8f`（`refactor(grid): call the PTO ISA surface from the
> distributed_ffn_grid demos`，6 文件 / +95 −41）。那是另一条独立改动线——把 demo 从
> `GRID_*_IMPL` 换成 `TPUSH/TPOP/TBROADCAST/TREDUCE`，并补上 `TREDUCE<Group, Op, T>`
> 组归约重载——与本篇正交，本篇不展开。

---

## 1. 背景：换生产者时，计数器为什么不能归零

上一篇把「这条通道是谁到谁」钉进了类型：`GridPipe<TileT, Dir, SlotStride, SlotCount, Dist, ScbId>`
的生产者是 `Dir` 上游 `Dist` 跳的那个核，消费者是下游那个。好处很清楚，代价是——**换对端就得换
pipe 对象**。

时分 MPSC 场景（选型文档 §1.3 / §3.2 B1）恰好踩在这个代价上：**消费者不动，生产者跨相位换人**。
比如 AllGather 两相，第一相沿 EAST 灌、第二相沿 SOUTH 灌，接收方从头到尾是同一个核。按类型绑定
的规矩，这是两个 pipe；但物理资源一样都没动：

| 状态 | 交接时 |
|---|---|
| ring 里已经躺着的数据 | 不动。TPUSH 是写推、TPOP 是本地读，旧生产者写的 tile 就在消费者本核 SRAM 里 |
| ring 基址 / 槽数 | 不动。两个 pipe 声明在同一 window、同样几何 |
| `ready_scb` | 不动（relay 分支）。它已经等于 E，正是新序号该续的位置 |
| `cons_idx` | 不动（relay 分支）。消费者继续从原位排，只是排的是旧生产者的遗留数据 |
| `free_scb` | 重新基线化，但走的是普通记分板通路 |
| **`prod_idx`** | **必须跨核搬运** |

只有最后一行是真的要过桥的。而它**不能简单归零**：ring 里还有 `E − cons_idx` 块没被消费时，
`slot (0 % SlotCount)` 恰恰是最老的那块未消费数据，而且 free 门槛 `prod − SlotCount + 1` 也不再
描述「消费者实际消费到哪」。选型文档 §9 把「只重置计数、不排空」列为头号伪解法，说的就是这个。

所以计数要**接力**：绝对序号从旧相末的 E 连续续到 E+1、E+2……，`(idx % SlotCount)` 的回绕自动
绕开未消费块。只有在**可证明已排空**的边界上才归零——那时归零反而有价值，能把绝对计数从
IPC_SCB 的 16-bit 天花板边上拉回来。

---

## 2. 改了什么

### 2.1 `THANDOFF` 指令与三个半程

```cpp
THANDOFF(oldPipe, newPipe, handoffSeq, retiredEnd);
```

两个 pipe 必须声明在**同一个 window、同一个显式 `ScbId`**（默认 `2*Dir` 会给出不同的物理寄存器
对，所以必须显式传）。`GridTHandoff.hpp` 用 static_assert 钉住了四件事：ring 几何一致、记分板槽
一致、两个 pipe 确实指向不同对端、以及 `ScbId+3 < 16`。

每个核都调用它，跑哪几个半程由拓扑决定——和 TPUSH/TPOP/TREDUCE 一样，没有显式 root 标志：

```text
半程 A（作为消费者）
    ① WAIT_SPR(ready_scb, retiredEnd)      证明退役生产者已发完（H3）
    ② MOV_SPR2X  ready_scb → cons.batonL1  取出 E，直接落 L1
    ③ MOV_L12X   cons.batonL1 → GPR        仅为判分支：E ≤ cons_idx ?
    ④ (rebase 时)把 L1 字改写成 0
    ⑤ sync_hscb  cons.batonL1 → 新生产者 prod.batonL1
       sync_hscb  cons_idx     → 新生产者 free_scb
       publish fence
       sync_hscb  handoffSeq   → 新生产者 install_scb（门铃）

半程 B（作为生产者）
    ① WAIT_SPR(install_scb, handoffSeq)
    ② MOV_L12X   prod.batonL1 → prod_idx (GPR)
    ③ (baseline==0 时)sync_hscb 0 → 消费者 ready_scb    ← 消费者不能写自己的 IPC_SCB
       publish fence
    ④ sync_hscb  handoffSeq → 消费者 open_scb（OPEN_ACK）

半程 C（作为消费者）
    relay ：cons_idx 原样带过
    rebase：先 WAIT_SPR(open_scb, handoffSeq)，再 cons_idx = 0
```

A 先于 B、B 先于 C，且 **A 不阻塞于任何其他核的 handoff**——这是无死锁的全部理由：每个核都能走到
A，于是每个 B 都有发送者，于是每个 C 都有 acker。

### 2.2 两条 MOV 类 facade，以及为什么不能合成一条

`grid_cce_intrinsic.hpp` 的 facade 表新增两行：

```text
V8 机器指令 | CCE facade          | CCE builtin                         | 方向
-----------+---------------------+-------------------------------------+---------
MOV_SPR2X  | mov_ipc_scb_to_l1   | __builtin_cce___mov_ipc_scb_to_l1   | SPR → L1
MOV_L12X   | mov_l1_to_gpr       | __builtin_cce___mov_l1_to_gpr       | L1  → GPR
```

```cpp
void     mov_ipc_scb_to_l1(__gm__ uint32_t* dstL1, __gm__ uint32_t* srcScb, uint32_t srcSlot);
uint32_t mov_l1_to_gpr(__gm__ uint32_t* localL1);
```

签名就不一样：一个按 slot 号取源、写内存，一个按地址取源、进寄存器。**没有 SPR → GPR 形式，也
不需要**——记分板只有两种用法，要么被 WAIT_SPR 比较（阈值比较在指令内部完成），要么被
MOV_SPR2X 转发出去。稳态 TPUSH/TPOP 路径一条 MOV 都没有。

需要注意的是半程 A 的第 ③ 步：分支判断要一个 GPR 里的标量，而既然没有 SPR→GPR 通路，唯一取值
办法就是把 MOV_SPR2X 刚写下的 L1 字用 MOV_L12X 读回来。**这一步不在转发路径上，纯粹服务于分支。**

### 2.3 baton：一个 L1 字，不是记分板

```cpp
struct GridProducerSem {
    __gm__ uint32_t* freeScb;     // IPC_SCB
    __gm__ uint32_t* installScb;  // IPC_SCB，门铃
    __gm__ uint32_t* batonL1;     // L1：消费者 ST_HSCB 投递进来的 prod_idx 基线
    uint32_t prodIndex;           // GPR
    GridPayloadWindow window;
};
struct GridConsumerSem {
    __gm__ uint32_t* readyScb;    // IPC_SCB
    __gm__ uint32_t* openScb;     // IPC_SCB，OPEN_ACK
    __gm__ uint32_t* batonL1;     // L1：MOV_SPR2X 把待转发的棒放这儿
    uint32_t consIndex;           // GPR
    GridPayloadWindow window;
};
```

**baton = 交接时唯一被真正交出去的东西**（见 §1 那张表）。它必须落 L1 而不是记分板，因为
`prod_idx` 不是信号量——信号量只被问「够不够阈值」，`prod_idx` 要被**读进寄存器**（TPUSH 用它
算 `slot = idx % SC` 和 free 门槛）。SPR 没有出口到 GPR，L1 有。

出/入两个字必须分开：SPMD 下同一个核同时是交接的两端，一边把自己的棒交给上游、一边从下游接自己
的棒，共用一个字会撞。

顺带的好处：handoff 的 IPC_SCB 预算只花在两个门铃上，`ScbId..ScbId+3` 共 4 槽（稳态 2 + 交接 2），
而且交接槽只在真正用到 THANDOFF 的 pipe 上计费——`GridPipe` 自身的 static_assert 仍只要求
`ScbId+1 < 16`，`ScbId+3 < 16` 挪到 `GridTHandoff.hpp` 里断言。

### 2.4 free 基线为什么走记分板通路，且值不是 E

```cpp
sync_hscb(peerBaton, rebase ? 0 : endIndex);   // → 新生产者 L1
sync_hscb(peerFree,  rebase ? 0 : consIndex);  // → 新生产者 free_scb，普通 FREE store
```

很容易误以为两个都该是 E（§1.3 的 INSTALL_BASE 确实写 `prod = cons = E`，但那是**已 drain**的
前提）。relay 分支恰恰没 drain：

- `prod_idx = E`：新生产者第一次写落在 `slot(E % SC)`，紧贴旧生产者最后一块之后；
- `free_scb = cons_idx`：free 的语义是**已消费量**（C3），未消费的 `E − cons_idx` 块必须被反压保护。

把 TPUSH 的门槛展开，这个 gap 就是反压本身：

```text
free ≥ prod − SC + 1
cons_idx ≥ E − SC + 1
(E − cons_idx) ≤ SC − 1      ← 在飞块数 < SC，即不变量 C4
```

也就是说转发两个不同的值，是为了让**既有的** SPSC 反压公式在换了生产者之后继续成立，一行新逻辑
都不用加。

还有一处「零代码」的转发切换：半程 C 之后消费者用 `newPipe` 做 TPOP，而 TPOP 的 free 门铃目标是
`pipe.ProducerRank()`——现在自然解析到新生产者。消费者排掉旧生产者遗留块时释放的信用，正好发给
需要这些槽的人。

### 2.5 window 布局：每个外部写入字独占一条 cache line

这是本次最大的一处改动，起因见 §4.1。

```text
line 0  (  0)  ready_scb    IPC_SCB   ← 生产者对端写
line 1  ( 64)  free_scb     IPC_SCB   ← 消费者对端写
line 2  (128)  install_scb  IPC_SCB   ← 消费者对端写（INSTALL_BASE 门铃）
line 3  (192)  open_scb     IPC_SCB   ← 新生产者对端写（OPEN_ACK）
line 4  (256)  batonL1(out) L1        ← 本核写
line 5  (320)  batonL1(in)  L1        ← 消费者对端写
384..511       reserved
512            slot ring
```

- 新增 `grid_mock::kScbLineStride = 64` / `kScbLineStrideU32 = 16`，原来的
  `kBcastLaneStride` 改为它的别名（同一条规则，一处真相）。
- `kFlagsBytes` **128 → 512**，5 处 host 镜像同步：`KHOP_` / `BCAST_` / `HANDOFF_` /
  `FFN_` / `FFN_NCUT_GRID_FLAGS_BYTES`。
- 故障哨兵仍在 `scb + kFaultFlagWordOffset`（各自 line 内 +40 B），所以 host 侧扫描逻辑不变。

新增 4 个故障码：`0x501` 退役等待超时 / `0x502` INSTALL_BASE 门铃超时 / `0x503` OPEN_ACK 超时 /
`0x504` 两个 pipe 不是同一个 window（后者是唯一「不是挂死而是静默损坏」的错，所以专门查）。

### 2.6 新增 handoff smoke

`kernels/manual/a2a3/distributed_ffn_grid/smoke/handoff_smoke_*` + `run_handoff_smoke.sh`，
Vec-only 纯搬运。默认 3×4，每个 cell 一个 window 被两个生产者先后驱动：

```text
phase 1   EAST  pipe：每个 cell 往东推 P1_TILES(2) 块；接收方只 pop P1_POPS(1) 块，故意留货
THANDOFF  生产者 西邻 → 北邻
phase 2   SOUTH pipe：先排掉 phase-1 遗留（旧生产者写的数据，走新 pipe 排出），再排新块
```

`--p1-pops` 就是分支开关：`< --p1-tiles` 走 relay，`== ` 走 rebase。无论怎么设，第 0 列因为没有
phase-1 生产者、ring 天然是空的，永远走 rebase——所以一次跑里两条分支都被覆盖。

host 侧 golden 是 kernel 控制流的逐条复演，**每块 popped tile 按 stamp 校验**，所以基线错一位会
表现成「拿到别的 cell 的数据」，而不是精度超差。3×4 relay 的实测输出：

```text
cell 5 (r=1,c=1) relay pops=3 expect=[50 51 22] got=[50 51 22]
                                      ↑  ↑   ↑
                                      │  │   └ 北邻在中继基线 E=2 上写进 slot 2 的新块
                                      │  └ 西邻第 2 块：跨过交接、通过新 pipe 排出的遗留数据
                                      └ 西邻第 1 块：phase-1 就排掉的
```

---

## 3. 设计取舍

### 3.1 为什么由消费者做中继，而不是旧生产者直发新生产者

三个理由，缺一不可：

1. **寻址**：旧生产者和新生产者不一定是几何邻居（demo 里一西一北，互为对角），`sync_hscb`
   按 (dir, dist) 寻址够不着。而消费者同时是两者的直连对端——这正是「消费者身份不变」的推论。
2. **值是现成的**：消费者的 `ready_scb` 里存的就是 E。旧生产者最后一次 TPUSH 已经
   `sync_hscb(READY, idx+1)` 把它覆盖式写进来了，**第 0 跳是免费的**，稳态协议本来就在做。
3. **因果安全**：rebase 分支要清零 `ready_scb`，清零必须因果地排在「观测到已排空」之后。只有
   消费者能做这个观测。

### 3.2 rebase 分支为什么必须有 OPEN_ACK

因为消费者**不能自己清零** `ready_scb`（约束①：本核 ScalarUnit 不能写自己的 IPC_SCB），只能让
新生产者代劳。于是出现一个窗口：如果消费者先把 `cons_idx` 归零、新生产者的 `ready=0` 还没落地，
那一刻 `ready(E) ≥ cons(0)+1` 成立，下一条 TPOP 会立刻放行并排出一个陈旧槽。

OPEN_ACK 转发回去的不是数值而是**事实**——「你的 ready_scb 已经是 0 了」。两次 store 走同一条边
（新生产者 → 消费者）且中间有 fence，per-edge publish 序保证 `ready=0` 先于 ACK 落地。

relay 分支不需要 ACK（消费者侧什么都不改），所以这个往返只在 rebase 时付出。

判据 `baseline == 0 ⟺ rebase` 是可靠的：relay 分支下 `base = E > cons_idx ≥ 0`，必有 `E ≥ 1`。

### 3.3 为什么没有 baton 结构体

初版设计过一个 `GridPipeBaton { endIndex, consIndex, handoffSeq, rebase }`，后来删掉了：它要装的
两个值本来就活在退役 pipe 上——`cons.readyScb` 就是 E，`cons.consIndex` 就是排空位置——包一层
只是复制状态。半程 A 现在两个局部变量搞定。

「baton」这个名字保留下来，但只用来指那个 L1 字，含义很具体：**交接时唯一被交出去的东西**。

### 3.4 `retiredEnd` 不是可选装饰

整条链路的正确性挂在 E 上。如果旧生产者最后一次 ready store 还在途中，消费者读到 E' < E，转发出去
后新生产者从 `slot(E' % SC)` 开始写，**直接覆盖旧生产者已经填好的块**。

所以半程 A ① 是 H3 不变量的落地：`retiredEnd` 传相位的 tile 数（结构化集合通信里都是编译期已知
的），拿 WAIT_SPR 证明；传 0 表示调度自身保证静默。边界 cell 没有旧生产者，由 `HasProducer()`
自动跳过，因此调用方可以整个 mesh 传同一个常量。选型文档 §3.2 B1 明确说过「先改 owner 表再等
一段时间」不是它的合法替代。

---

## 4. 踩坑记录

### 4.1 记分板打包在一条 cache line 里 → 3×3 起偶发挂死（本次最大的坑）

现象：1×2 / 2×1 / 2×2 / 3×2 / 2×4 / 1×9 全过，**3×3 偶发挂死**（3 次里挂 2 次）。

排查过程值得记：

1. 先按拓扑做二分，确认不是块数问题（1×9 = 9 块过了，3×3 = 9 块挂）。
2. 把 kernel 里的 `TPUSH/TPOP/THANDOFF` 临时换成 `GRID_TRY_*_IMPL` 带有限自旋（3e6 ≈ 0.3 s），
   让 mock 写故障哨兵而不是死等。拿到 cell 2 / cell 5 的 `0x501`（退役等待超时）。
3. 加一段 host 侧 D2H dump，把每个 cell 的整个 flag header 打出来。**结果那一次跑过了**——
   于是问题定性为竞态，不是逻辑错。
4. 对着 dump 反推：cell 2 等的是它西邻写的 `ready_scb`（word 0），而它南邻的半程 A 正好同时在写
   cell 2 的 word 1/2/3。

根因：window 头里 5 个字打包成连续 u32，落在**同一条 64 B cache line** 上，而它们有**至多 4 个
不同的远端写者**。AICore 缓存核间不相干、`dcci` 写回是整行粒度，于是一个对端的 store 会把自己
（可能过时的）整行拍回去，**把另一个对端的门铃写没了**——等待方就永远卡在一个其实早已满足的
阈值上。

这与 TBROADCAST 那次 per-source lane 打包是同一个故障（见
[allgather-tbroadcast-prodcons-split-fix]），当时的结论「unicast 是 fan-in-1，一行一个写者，不受
影响」**是错的**：`ready`(word 0) 由生产者对端写、`free`(word 1) 由消费者对端写，一直就是两个远
端写者共一行。它之所以长期没爆，是因为单调计数有自愈性——丢的不是最后一次更新时，后一次更新会
把值追回来。本次交接给同一行又加了两个远端写者，概率被推到了肉眼可见。

修复即 §2.5。改完 3×4 连过 3 次、4×6（24 cell）连过 3 次。

### 4.2 MOV 指令的语义前后错了两版

- **第一版**：一个 `mov_ipc_scb(scb, slot)` 同时用于「消费者读自己的 ready_scb」和「新生产者读
  投递进来的基线」，把 L1 那一跳错建模成了 IPC_SCB。
- **第二版**：拆成 `mov_ipc_scb_to_gpr` + `mov_l1_to_gpr`，但前者仍是 SPR → GPR，而**根本没有
  这条指令**，也没有这个需求。
- **定版**：`mov_ipc_scb_to_l1`（SPR → L1）+ `mov_l1_to_gpr`（L1 → GPR）。转发路径上不经过 GPR。

一句话记忆：**记分板只被比较（WAIT_SPR）或被转发（MOV_SPR2X → L1），永远不落 GPR。**

### 4.3 有界自旋调试时，故障哨兵会污染诊断

哨兵在 `scb + 10` 字，和该记分板同处一条 line。写哨兵是本核写，于是那次写回可能把远端刚写进来的
记分板值拍回旧值——把一次干净的挂死变成看起来像别的错。这也是 §4.1 排查时先看到 `0x501`、后来
才发现真正的首错可能是 TPOP 的 `0x301`（两者都落 word 10，后写覆盖先写）的原因。

结论：哨兵位置留着不动（只在已经失败的跑里才写），但**调试时别把哨兵码当唯一线索**，配合 flag
header 全量 dump 一起看。

---

## 5. 验证

| 跑法 | 结果 |
|---|---|
| `smoke/run_handoff_smoke.sh`（3×4，relay，`--p1-pops 1`）×3 | PASS，max diff 0 |
| 同上 `--p1-pops 2`（3×4，rebase 全网）×2 | PASS，max diff 0 |
| 同上 `--grid-rows 4 --grid-cols 6`（24 cell，relay）×3 | PASS，max diff 0 |
| `smoke/run_khop_smoke.sh -r npu` | PASS，max diff 0 |
| `smoke/run_bcast_smoke.sh -r npu` | PASS，max diff 0 |
| `run_treduce_reducesum.sh -r npu` | PASS，max diff 0 |
| `run_tpush_reducesum.sh -r npu` | PASS，max diff 0 |
| `run_tbroadcast_allgather.sh -r npu` | PASS，max diff 0 |
| `run_tpush_allgather.sh -r npu` | PASS，max diff 0 |

另：`pre-commit run clang-format`（CI 固定 v18.1.8）Passed；`scripts/oat_check.sh` 全过；grid 头
文件在 `__CPU_SIM` 下无新增报错；`3fe7a68b` 单独 checkout 到临时 worktree 能独立编译（本次提交
是从一个混合工作区按 hunk 拆出来的，所以专门验了这一条）。

> `run_tpush_reducesum.sh` 有 **10–20% 的既有偶发挂死**，在基线 `f25697d6` 上就存在（干净
> worktree 3/16，本次改动后 2/18），与本次改动无关，但单次 PASS 不足以作为该 demo 的强证据。

---

## 6. 与上一篇的关系 / 剩余差距

上一篇（`163c56e3`）把通道身份钉进类型，本篇处理的正是那个决定的直接代价：**换对端就得换 pipe，
于是计数得有办法过桥**。两者是同一条设计线的前后两步。

剩余差距：

1. **16-bit 回绕只做了一半** —— rebase 分支能在排空边界把计数拉回 0，但没有 D2「窗口式模比较」
   （`(ready − cons) mod 2^16 ∈ [0, SC]`）。单 kernel 总 tile 数逼近 `2^16` 且中途没有排空边界
   时仍会出问题。
2. **只支持严格顺序交接（方向二）** —— 相位重叠要的是方向一（A2 double-bank / A3 通道池），那需要
   多套物理 bank，目前没有。
3. **多次交接的 `handoffSeq` 靠调用方给** —— 要求全网一致且严格递增，没有编译期或运行期校验；
   给错会表现成一个半程等在永远不会到达的门铃上（`0x502` / `0x503`）。
4. **`retiredEnd` 只对单条边成立** —— 一次 THANDOFF 只覆盖一条 (生产者, 消费者) 边的静默证明；
   若一个相位里同一 window 被多条边写过，需要调用方自己拆开。

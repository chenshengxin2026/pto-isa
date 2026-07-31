# PTO MegaMoE Dispatch + Combine 融合算子示例（A5）

[English](README.md)

## 概览

本示例使用 PTO Manual kernel 在昇腾 A5 上实现端到端 MegaMoE 融合算子，将 MoE 重排、类 AlltoAllV 数据
交换、分组 FFN 计算、Combine 和 Unpermute 放入一个 AIC/AIV 混合 kernel 中。

Device 侧主流程如下：

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

主要逻辑沿用 A2/A3 实现。A5 版本适配了核数、UB、同步机制和 GMM 参数，并增加 GMM2 到 Combine 的
CV 直通。

## 支持的 AI 处理器

- 昇腾 A5，arch35
- CCE 编译目标：`dav-c310`

构建时定义 `PTO_NPU_ARCH_A5`，并使用 PTO A5 后端。

## 目录结构

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # Host 和 Device kernel 构建配置
├── run.sh                          # 数据生成、构建、拉起和校验入口
├── main.cpp                        # ACL/HCCL/MPI 初始化、拉起和结果校验
├── kernel_launch.cpp               # Device kernel 拉起封装
├── runtime_context.*               # Runtime 和 HCCL window 管理
├── tiling_builder.*                # Host 侧 tiling 和 workspace 规划
├── data_utils.*                    # Case IO 和精度比对
├── comm_mpi.h                      # MPI 动态加载封装
├── scripts/
│   └── gen_data.py                 # 输入、权重和 Golden 数据生成
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE Device 流水入口
│   ├── dispatch_mega_combine_tiling.h
│   ├── front_reorder.h             # FrontReorder 和量化 scatter
│   ├── front_fullload_sort.h       # FullLoad 排序路径
│   ├── front_vms_sort.h            # OneCore 和 MultiCore 排序路径
│   ├── dispatch.h                  # 拉取 peer offsetA
│   ├── gmm_common.h                # GMM 公共调度
│   ├── gmm1.h                      # 第一次分组矩阵乘
│   ├── swiglu.h                    # SwiGLU 和动态量化
│   ├── gmm2.h                      # 第二次分组矩阵乘
│   ├── gmm2_combine_cv_pipe.h      # GMM2 到 Combine 的 CV pipe
│   ├── combine.h                   # 远端结果写回
│   ├── unpermute.h                 # TopK 加权归约
│   └── utils/                      # PTO、同步和 HCCL 辅助代码
├── overview.md                     # A5 适配概览
└── gmm2_combine_cv_direct_design.md # CV 直通设计说明
```

## 算子说明

### 计算功能

算子实现多 rank MoE FFN 主流程：

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> expert-major 路由结果
  -> 分组 GMM1
  -> SwiGLU + 动态量化
  -> 分组 GMM2
  -> Combine 回源 rank
  -> TopK 加权归约
  -> out[rank, M, K]
```

概念上等价于：

```text
out[token] = sum(probs[token, route] * FFN_expert(x[token]), route=0..topK-1)
```

### 规格

| 项目 | 取值 |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| 输入 | `x`：`[M, K]`，BF16；`expertId`：`[M, topK]`，int32；`probs`：`[M, topK]`，FP32；int8 packed 权重和 Fixpipe scale |
| 输出 | `out`：`[M, K]`，FP16 |
| Kernel 名称 | `dispatch_mega_combine_kernel` |
| Host 可执行文件 | `dispatch_mega_combine` |
| 默认 case | `worldSize=2, M=16, K=128, N=128, topK=2, expertPerRank=16, maxOutputSize=32` |

## 优化说明

- **Expert 粒度流水**：Dispatch、GMM1、SwiGLU、GMM2 和 Combine 按本地 expert 或 segment 推进，并在
  阶段边界使用 hard flag。
- **三种 FrontReorder 路径**：根据 route 数和 A5 UB 容量选择 FullLoad、OneCore 或 MultiCore。
- **Count-as-flag**：FrontReorder 发布路由计数，用于 peer rank 数据到达判断和地址计算。
- **PTO tile GMM**：GMM1/GMM2 使用输出 tile swizzle、L1/L0 复用和 Fixpipe 量化。
- **GMM2 L0C 双缓冲**：GMM2 使用两个 L0C stage。
- **GMM2 到 Combine CV 直通**：GMM2 把完整输出 tile 直接送给一个配对 AIV，省去 GMM2 输出的 GM
  中间缓冲区。

## Tiling 参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `M` | 每个 rank 的 token 数，通过 `run.sh --m` 设置 |
| `K` | Hidden size，要求 `K % 128 == 0` |
| `N` | FFN intermediate size，要求 `N % 64 == 0` |
| `topK` | 每个 token 的路由专家数 |
| `expertPerRank` | 每个 rank 的本地专家数 |
| `worldSize` | MPI/HCCL rank 数，支持范围为 `[1, 16]` |
| `maxOutputSize` | 每个 rank 的路由行 workspace 上限，至少为 `M * topK` |
| `aicNum` | 逻辑 AIC 数，默认 `28` |
| `aivNum` | 逻辑 AIV 数，默认 `56`，且必须等于 `2 * aicNum` |
| GMM L1 tile | `128 x 256 x 512` |
| GMM L0 tile | `128 x 256 x 128` |
| CV tile | 最大 `128 x 256` 个 FP16 元素 |

## 支持 Case

| Case | 参数 |
| --- | --- |
| 默认正确性 case | `worldSize=2, M=16, K=128, N=128, topK=2, experts=16, maxOutputSize=32` |
| 已验证四 rank case | `worldSize=4, M=2048, K=7168, N=4096, topK=8, experts=16, maxOutputSize=81940` |

该四 rank case 已在 A5 上执行，所有 rank 均输出 `PASS`。

## 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ FrontReorder (AIV)                                                          │
│   路由排序 -> 量化 offsetA + count/prefix 元数据                            │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert 粒度重叠流水                                                        │
│                                                                              │
│ AIV: Dispatch -> SwiGLU -> Combine                                           │
│ AIC:            GMM1 -> GMM2                                                 │
│                               └─ CV 直通 -> 配对 AIV                        │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   offsetD + probs + expandedRowIdx -> TopK 加权归约 -> out[M, K]             │
└──────────────────────────────────────────────────────────────────────────────┘
```

## FrontReorder 阶段

FrontReorder 按全局 expert 对 `[token, topK]` 路由排序，并把量化后的 token 行写入本地 HCCL window：

```text
x[M, K] + expertId[M, topK]
  -> offsetA[expert-major rows, K + 32]
  -> expandedRowIdx / localTokenPerExpert / preSumBeforeRank / cumsumMM
```

三种排序路径具有相同输出布局：

- **FullLoad**：工作集可放入 UB。
- **OneCore**：由一个 AIV 完成路由排序。
- **MultiCore**：多个 AIV 生成并合并有序 run。

## Dispatch 阶段

Dispatch 在目标 rank 上运行，从所有源 rank 拉取 packed 行：

```text
peer.offsetA[srcRowBase : srcRowBase + rows]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
```

一个本地 expert group 就绪后，Dispatch 设置对应的 GMM1-ready flag。

## GMM1 / SwiGLU / GMM2 阶段

### GMM1

GMM1 在 AIC 上运行：

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> Fixpipe scale1
  -> gmC[FP16]
```

### SwiGLU

SwiGLU 在 AIV 上运行：

```text
gmC * perTokenScale1
  -> SiLU(up) * gate
  -> 动态量化
  -> gmPermutedToken[int8] + perTokenScale2[FP32]
```

### GMM2

GMM2 在 AIC 上运行：

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> Fixpipe scale2
  -> FP16 CV tile
```

Tile 在两个配对 AIV 间轮转，每个被选中的 AIV 使用 `TILE_NO_SPLIT` 消费一个完整 tile。同步细节见
[gmm2_combine_cv_direct_design.md](gmm2_combine_cv_direct_design.md)。

## Combine / Unpermute 阶段

Combine 消费 CV tile，应用 `perTokenScale2`，再把对应数据行写回源 rank：

```text
CV tile[FP16]
  -> FP32
  -> * perTokenScale2
  -> FP16
  -> sourceRank.remoteWindow.offsetD
```

Small 和 large token case 使用同一条直通路径。Unpermute 随后恢复源 rank 的 token 顺序：

```text
offsetD + probs + expandedRowIdx
  -> TopK 加权累加
  -> out[M, K]
```

## 内存布局与 HCCL Window

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `offsetA` | HCCL window | FrontReorder 写 packed 行，Dispatch 从 peer rank 读取 |
| `offsetD` | HCCL window | Combine 写回结果，Unpermute 本地读取 |
| `tokenPerExpert` | HCCL window | 跨 rank 路由计数元数据 |
| `gmA` | Workspace GM | Dispatch 生成的 GMM1 输入 |
| `gmC` | Workspace GM | GMM1 输出和 SwiGLU 输入 |
| `gmPermutedToken` | Workspace GM | SwiGLU 输出和 GMM2 输入 |
| `expandedRowIdx` | Workspace GM | 路由到 expert-major 行的映射 |
| `cumsumMM / preSumBeforeRank` | Workspace GM | Dispatch 和 Combine 地址元数据 |

GMM2 输出通过 CV pipe 传输，不使用 workspace GM 输出 buffer。所选 case 需要更大 HCCL window 时，
`run.sh` 会上调 `HCCL_BUFFSIZE`。

## 性能优化指南

### 1. 先确认 FrontReorder 路径

较小 route 数通常应进入 FullLoad。进入 OneCore 或 MultiCore 时，先检查排序和 GM 流量。

### 2. 保持阶段边界不变

Dispatch、GMM1、SwiGLU、GMM2 和 Combine 依赖配对的 hard flag set/wait 操作。

### 3. 优先检查 GMM tile 效率

检查 AIC 负载均衡、输出 tile swizzle、L1/L0 复用，以及 AIV 通信带来的 HBM 竞争。

### 4. 保持 CV tile 顺序

GMM2 producer 和配对 AIV consumer 必须跨 expert 边界使用相同 tile 顺序。

## 构建与运行

配置 CANN 环境并进入算子目录：

```bash
source /path/to/cann/set_env.sh
export ASCEND_HOME_PATH=/path/to/cann
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
```

运行默认正确性 case：

```bash
bash run.sh
```

运行已验证的四 rank case：

```bash
bash run.sh --world-size 4 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### 环境变量

| 环境变量 | 用途 |
| --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装路径，构建时必须设置 |
| `DISPATCH_MEGA_COMBINE_START_SYNC` | Kernel 入口处可选的跨 rank 同步，默认 `0` |

## 修改 Case 参数

主要约束如下：

- `worldSize` 位于 `[1, 16]`。
- `K % 128 == 0`。
- `N % 64 == 0`。
- `maxOutputSize >= M * topK`。
- `aivNum == 2 * aicNum`。

## 常见问题

| 问题 | 原因与处理方法 |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | 先加载 CANN 环境并设置 `ASCEND_HOME_PATH` |
| AIC/AIV 核数报错 | 使用 1:2 混合核配置，例如 `28/56` |
| HCCL window 太小 | 增大 `HCCL_BUFFSIZE`，或由 `run.sh` 自动调整 |
| MPI 拉起失败 | 检查 `run.sh` 使用的 MPI 环境 |
| 结果比对失败 | 重新生成 case 数据，不要复用不兼容的数据文件 |

## 构建系统

- **编译器**：`bisheng`
- **语言标准**：C++17
- **Device 架构**：`dav-c310`
- **构建目标**：`dispatch_mega_combine_kernel` 和 `dispatch_mega_combine`
- **PTO include**：仓库根目录下的 `include/`

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-07-23 | 新增 A5 中文 README |

# PTO MegaMoE Dispatch + Combine Fusion Example (A5)

[中文文档](README_zh.md)

## Overview

This example implements an end-to-end MegaMoE fused operator for Ascend A5 with PTO Manual kernels. It fuses MoE
reorder, AlltoAllV-style data exchange, grouped FFN computation, combine, and unpermute into one mixed AIC/AIV kernel.

The main device pipeline is:

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

The main logic follows the A2/A3 implementation. The A5 version adapts the core count, UB usage, synchronization, and
GMM parameters, and adds a direct GMM2-to-Combine CV path.

## Supported AI Processors

- Ascend A5, arch35
- CCE target: `dav-c310`

The build defines `PTO_NPU_ARCH_A5` and uses the PTO A5 backend.

## Directory Layout

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # Host and device-kernel build configuration
├── run.sh                          # Data generation, build, launch, and verification entry
├── main.cpp                        # ACL/HCCL/MPI setup, launch, and result verification
├── kernel_launch.cpp               # Device-kernel launch wrapper
├── runtime_context.*               # Runtime and HCCL-window management
├── tiling_builder.*                # Host tiling and workspace planning
├── data_utils.*                    # Case IO and accuracy comparison
├── comm_mpi.h                      # MPI dynamic-loading wrapper
├── scripts/
│   └── gen_data.py                 # Input, weight, and golden-data generation
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE device pipeline entry
│   ├── dispatch_mega_combine_tiling.h
│   ├── front_reorder.h             # Front reorder and quantized scatter
│   ├── front_fullload_sort.h       # FullLoad sorting path
│   ├── front_vms_sort.h            # OneCore and MultiCore sorting paths
│   ├── dispatch.h                  # Peer offsetA gather
│   ├── gmm_common.h                # Shared GMM scheduling
│   ├── gmm1.h                      # First grouped matmul
│   ├── swiglu.h                    # SwiGLU and dynamic quantization
│   ├── gmm2.h                      # Second grouped matmul
│   ├── gmm2_combine_cv_pipe.h      # GMM2-to-Combine CV pipe
│   ├── combine.h                   # Remote result writeback
│   ├── unpermute.h                 # TopK weighted reduction
│   └── utils/                      # PTO, synchronization, and HCCL helpers
├── overview.md                     # A5 adaptation overview
└── gmm2_combine_cv_direct_design.md # CV-direct design details
```

## Operator Description

### Functionality

The operator implements the multi-rank MoE FFN path:

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> expert-major routed rows
  -> grouped GMM1
  -> SwiGLU + dynamic quantization
  -> grouped GMM2
  -> combine to source ranks
  -> TopK weighted reduction
  -> out[rank, M, K]
```

Conceptually:

```text
out[token] = sum(probs[token, route] * FFN_expert(x[token]), route=0..topK-1)
```

### Specification

| Item | Value |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| Input | `x`: `[M, K]`, BF16; `expertId`: `[M, topK]`, int32; `probs`: `[M, topK]`, FP32; int8 packed weights and Fixpipe scales |
| Output | `out`: `[M, K]`, FP16 |
| Kernel name | `dispatch_mega_combine_kernel` |
| Host executable | `dispatch_mega_combine` |
| Default case | `worldSize=2, M=16, K=128, N=128, topK=2, expertPerRank=16, maxOutputSize=32` |

## Optimization Notes

- **Expert-level overlap**: Dispatch, GMM1, SwiGLU, GMM2, and Combine progress by local expert or segment and use hard
  flags at stage boundaries.
- **Three FrontReorder paths**: FullLoad, OneCore, and MultiCore are selected from route size and A5 UB capacity.
- **Count-as-flag**: FrontReorder publishes route counts for peer-rank data arrival and address calculation.
- **PTO tile GMM**: GMM1 and GMM2 use output-tile swizzle, L1/L0 reuse, and Fixpipe quantization.
- **GMM2 L0C double buffering**: GMM2 uses two L0C stages.
- **GMM2-to-Combine CV direct**: GMM2 sends each complete output tile directly to one paired AIV, avoiding an
  intermediate GMM2-output GM buffer.

## Tiling Parameters

| Parameter | Default / Description |
| --- | --- |
| `M` | Tokens per rank; set by `run.sh --m` |
| `K` | Hidden size; must satisfy `K % 128 == 0` |
| `N` | FFN intermediate size; must satisfy `N % 64 == 0` |
| `topK` | Routed experts per token |
| `expertPerRank` | Local experts per rank |
| `worldSize` | MPI/HCCL rank count; supported range is `[1, 16]` |
| `maxOutputSize` | Per-rank routed-row workspace limit; must be at least `M * topK` |
| `aicNum` | Logical AIC count; default `28` |
| `aivNum` | Logical AIV count; default `56` and must equal `2 * aicNum` |
| GMM L1 tile | `128 x 256 x 512` |
| GMM L0 tile | `128 x 256 x 128` |
| CV tile | Up to `128 x 256` FP16 elements |

## Supported Cases

| Case | Parameters |
| --- | --- |
| Default correctness case | `worldSize=2, M=16, K=128, N=128, topK=2, experts=16, maxOutputSize=32` |
| Validated four-rank case | `worldSize=4, M=2048, K=7168, N=4096, topK=8, experts=16, maxOutputSize=81940` |

The four-rank case has been run on A5 with all ranks reporting `PASS`.

## Overall Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ FrontReorder (AIV)                                                          │
│   sort routes -> quantized offsetA + count/prefix metadata                  │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert-level overlapped pipeline                                             │
│                                                                              │
│ AIV: Dispatch -> SwiGLU -> Combine                                           │
│ AIC:            GMM1 -> GMM2                                                 │
│                               └─ CV direct -> paired AIV                     │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   offsetD + probs + expandedRowIdx -> TopK weighted reduce -> out[M, K]      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## FrontReorder Stage

FrontReorder sorts `[token, topK]` routes by global expert and writes quantized token rows into the local HCCL window:

```text
x[M, K] + expertId[M, topK]
  -> offsetA[expert-major rows, K + 32]
  -> expandedRowIdx / localTokenPerExpert / preSumBeforeRank / cumsumMM
```

The three sorting paths have the same output layout:

- **FullLoad**: the working set fits in UB.
- **OneCore**: one AIV completes the route sorting.
- **MultiCore**: multiple AIVs generate and merge sorted runs.

## Dispatch Stage

Dispatch runs on the destination rank and gathers packed rows from all source ranks:

```text
peer.offsetA[srcRowBase : srcRowBase + rows]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
```

After one local-expert group is ready, Dispatch sets the corresponding GMM1-ready flag.

## GMM1 / SwiGLU / GMM2 Stages

### GMM1

GMM1 runs on AIC:

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> Fixpipe scale1
  -> gmC[FP16]
```

### SwiGLU

SwiGLU runs on AIV:

```text
gmC * perTokenScale1
  -> SiLU(up) * gate
  -> dynamic quantization
  -> gmPermutedToken[int8] + perTokenScale2[FP32]
```

### GMM2

GMM2 runs on AIC:

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> Fixpipe scale2
  -> FP16 CV tile
```

Tiles alternate between the two paired AIVs. Each selected AIV consumes one complete tile with `TILE_NO_SPLIT`.
Detailed synchronization is documented in [gmm2_combine_cv_direct_design.md](gmm2_combine_cv_direct_design.md).

## Combine / Unpermute Stages

Combine consumes the CV tile, applies `perTokenScale2`, and writes the corresponding rows back to their source ranks:

```text
CV tile[FP16]
  -> FP32
  -> * perTokenScale2
  -> FP16
  -> sourceRank.remoteWindow.offsetD
```

Small and large token counts use the same direct path. Unpermute then restores the source token order:

```text
offsetD + probs + expandedRowIdx
  -> TopK weighted accumulation
  -> out[M, K]
```

## Memory Layout and HCCL Window

| Buffer | Location | Purpose |
| --- | --- | --- |
| `offsetA` | HCCL window | FrontReorder writes packed rows; Dispatch reads from peer ranks |
| `offsetD` | HCCL window | Combine writes results back; Unpermute reads locally |
| `tokenPerExpert` | HCCL window | Cross-rank route-count metadata |
| `gmA` | Workspace GM | GMM1 input generated by Dispatch |
| `gmC` | Workspace GM | GMM1 output and SwiGLU input |
| `gmPermutedToken` | Workspace GM | SwiGLU output and GMM2 input |
| `expandedRowIdx` | Workspace GM | Route-to-expert-major-row mapping |
| `cumsumMM / preSumBeforeRank` | Workspace GM | Dispatch and Combine address metadata |

GMM2 output is transferred through the CV pipe instead of a workspace-GM output buffer. `run.sh` raises
`HCCL_BUFFSIZE` when the selected case requires a larger HCCL window.

## Performance Tuning Guide

### 1. Check the FrontReorder path first

Small route counts should normally select FullLoad. For OneCore or MultiCore, check sorting and GM traffic first.

### 2. Keep stage boundaries unchanged

Dispatch, GMM1, SwiGLU, GMM2, and Combine depend on paired hard-flag set/wait operations.

### 3. Prioritize GMM tile efficiency

Check AIC balance, output-tile swizzle, L1/L0 reuse, and HBM contention with AIV communication.

### 4. Preserve the CV tile order

The GMM2 producer and paired AIV consumers must use the same tile order across expert boundaries.

## Build and Run

Configure the CANN environment and enter this directory:

```bash
source /path/to/cann/set_env.sh
export ASCEND_HOME_PATH=/path/to/cann
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
```

Run the default correctness case:

```bash
bash run.sh
```

Run the validated four-rank case:

```bash
bash run.sh --world-size 4 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### Environment Variables

| Environment Variable | Purpose |
| --- | --- |
| `ASCEND_HOME_PATH` | CANN installation path; required by the build |
| `DISPATCH_MEGA_COMBINE_START_SYNC` | Optional cross-rank synchronization at the kernel entry; default `0` |

## Changing Case Parameters

The main constraints are:

- `worldSize` is in `[1, 16]`.
- `K % 128 == 0`.
- `N % 64 == 0`.
- `maxOutputSize >= M * topK`.
- `aivNum == 2 * aicNum`.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | Source the CANN environment and set `ASCEND_HOME_PATH` before running `run.sh` |
| AIC/AIV count error | Use a 1:2 mixed-core shape, such as `28/56` |
| HCCL window too small | Increase `HCCL_BUFFSIZE` or let `run.sh` adjust it |
| MPI launch fails | Check the MPI installation used by `run.sh` |
| Result comparison fails | Regenerate the case data instead of reusing incompatible files |

## Build System

- **Compiler**: `bisheng`
- **Language level**: C++17
- **Device architecture**: `dav-c310`
- **Targets**: `dispatch_mega_combine_kernel` and `dispatch_mega_combine`
- **PTO include**: repository root `include/`

## Changelog

| Date | Change |
| --- | --- |
| 2026-07-23 | Added the A5 English README |

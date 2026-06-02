/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP
#define DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP

#include <cstdint>

// ReduceSum mixed Cube/Vec kernel for the single-device multi-block FFN path.
//
// gridRows*gridCols blocks form a single-device logical grid.  Each block uses
// get_block_idx() as its row-major cell id.
//
// The kernel is compiled for dav-c220 mixed Cube/Vector.  Cube and Vec branches
// run concurrently and exchange gate/up/hidden/down intermediates through
// regular A2/A3 TPipe ready/free synchronization.  The final row-local EAST
// reduce still uses GridPipe windows.
void launchDistributedFfnGridMixedKernel(uint8_t *ffts, uint8_t *reducePipeWindow, uint8_t *x, uint8_t *wGate,
                                         uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial, uint8_t *upPartial,
                                         uint8_t *hiddenIn, uint8_t *downPartial, uint8_t *yOutput, uint8_t *hcclCtx,
                                         int gridRows, int gridCols, void *stream);

// AllGather split variant.  The GridPipe window carries fp16 hidden shards
// [T, Fi] across columns, then each column computes and stores its [T, Hc]
// output shard directly.
void launchDistributedFfnGridAllGatherMixedKernel(uint8_t *ffts, uint8_t *gatherPipeWindow, uint8_t *x, uint8_t *wGate,
                                                  uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial,
                                                  uint8_t *upPartial, uint8_t *hiddenIn, uint8_t *downPartial,
                                                  uint8_t *yOutput, uint8_t *hcclCtx, int gridRows, int gridCols,
                                                  void *stream);

// ===========================================================================
// pto::comm variants.  The cross-column combine is done with the existing
// collective communication API (pto::comm::TREDUCE / pto::comm::TGATHER)
// instead of the GridPipe mock.  Each variant runs in two launches with a host
// stream-sync barrier between them: a compute kernel publishes each cell's
// partial into its HCCL window slot, then a collective kernel combines them.
// ===========================================================================

// ReduceSum: phase-1 compute writes fp32 down partials [T,H] into window slots.
void launchFfnCommReduceSumComputeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *x, uint8_t *wGate, uint8_t *wUp,
                                         uint8_t *wDown, uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
                                         uint8_t *downPartial, int gridRows, int gridCols, void *stream);
// ReduceSum: phase-2 collective, gridRows roots run pto::comm::TREDUCE(Sum).
// windowsBase is the window arena base (== windowsIn[0]); the kernel folds the
// row offset into the HcclRemotePtr pe index.
void launchFfnCommReduceSumCollectiveKernel(uint8_t *ffts, uint8_t *windowsBase, uint8_t *yOutput, uint8_t *hcclCtx,
                                            int gridRows, int gridCols, void *stream);

// AllGather: phase-1 compute writes fp16 hidden shards [T,Fi] into window slots.
void launchFfnCommAllGatherComputeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *x, uint8_t *wGate, uint8_t *wUp,
                                         uint8_t *gatePartial, uint8_t *upPartial, int gridRows, int gridCols,
                                         void *stream);
// AllGather: phase-2 collective, each cell runs pto::comm::TGATHER then the down
// GEMM.  gatherScratch holds the [gridCols*T,Fi] TGATHER destination; hidden /
// down scratch back the V2C / C2V FIFOs of the in-kernel down projection.
void launchFfnCommAllGatherCollectiveKernel(uint8_t *ffts, uint8_t *windowsBase, uint8_t *gatherScratch,
                                            uint8_t *hiddenScratch, uint8_t *downScratch, uint8_t *wDown,
                                            uint8_t *yOutput, uint8_t *hcclCtx, int gridRows, int gridCols,
                                            void *stream);

#endif // DISTRIBUTED_FFN_GRID_KERNEL_LAUNCH_HPP

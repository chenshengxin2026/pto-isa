/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HANDOFF_SMOKE_LAUNCH_HPP
#define HANDOFF_SMOKE_LAUNCH_HPP

#include <cstdint>

// GridPipe 接力计数 producer-handoff smoke kernel (THANDOFF).
//
// gridRows*gridCols blocks form a single-device logical grid.  Each cell drives
// ONE window through two phases with two different producers -- EAST then SOUTH
// -- with a THANDOFF in between that relays (or rebases) the ring counters.  See
// handoff_smoke_config.hpp for the schedule and which branch fires where.
void launchHandoffSmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, int gridRows, int gridCols,
    void* stream);

#endif // HANDOFF_SMOKE_LAUNCH_HPP

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TARGET_TEST_PROBE_H
#define PTO_CPU_TARGET_TEST_PROBE_H

#include <cstddef>
#include <cstdint>

struct TargetProbe {
    int architecture;
    std::size_t ubBytes;
    std::size_t accBytes;
    std::uintptr_t memoryModel;
    std::uintptr_t ubStorage;
    float rowSums[8];
    float loadPadding;
    bool rejectsInitialize;
    bool rejectsDefault;
    bool rejectsMemoryInit;
    bool a5HighAddress;
    int capabilityArch;
    bool supportsFp8;
    bool convertsBfloat16;
};

using ProbeFunction = void (*)(TargetProbe*, int);
extern "C" void probe_a2a3(TargetProbe*, int);
extern "C" void probe_a5(TargetProbe*, int);
extern "C" void probe_legacy(TargetProbe*, int);

#endif

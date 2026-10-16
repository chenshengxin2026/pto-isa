// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include <pto/wse/a3.hpp>

__global__ AICORE void gemm(__gm__ half* a, __gm__ half* b, __gm__ float* c, __gm__ int32_t* mapping)
{
    pto::wse::LocalGemm<2, 16, 32, 512>(a, b, c, mapping);
}

extern "C" void launch(uint32_t blocks, void* stream, void* a, void* b, void* c, void* mapping)
{
    gemm<<<blocks, nullptr, stream>>>((__gm__ half*)a, (__gm__ half*)b, (__gm__ float*)c, (__gm__ int32_t*)mapping);
}

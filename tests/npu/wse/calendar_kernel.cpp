// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include <pto/wse/a3.hpp>

template <int Slot>
__global__ AICORE void calendar(
    __gm__ half* src, __gm__ half* dst, __gm__ int64_t* routes, __gm__ int32_t* mapping, __gm__ int32_t* members,
    __gm__ int32_t* ctrl, __gm__ int32_t* status)
{
    if constexpr (Slot == 0)
        pto::wse::CalendarKernel<2, 16, 8, 0, 8192, 32, 48>(src, dst, routes, mapping, members, ctrl, status);
    else
        pto::wse::CalendarKernel<2, 128, 4, 1, 8192, 32, 48>(src, dst, routes, mapping, members, ctrl, status);
}

extern "C" void launch_calendar(
    uint32_t slot, void* stream, void* src, void* dst, void* routes, void* mapping, void* members, void* ctrl,
    void* status)
{
    if (slot == 0)
        calendar<0><<<32, nullptr, stream>>>(
            (half*)src, (half*)dst, (int64_t*)routes, (int32_t*)mapping, (int32_t*)members, (int32_t*)ctrl,
            (int32_t*)status);
    else
        calendar<1><<<32, nullptr, stream>>>(
            (half*)src, (half*)dst, (int64_t*)routes, (int32_t*)mapping, (int32_t*)members, (int32_t*)ctrl,
            (int32_t*)status);
}

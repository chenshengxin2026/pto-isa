/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/wse/calendar.hpp>

extern "C" __global__ AICORE void calendar_smoke(__gm__ half* input, __gm__ half* output, __gm__ uint64_t* contexts)
{
#if defined(__CCE_AICORE__)
    auto context = contexts + get_block_idx() * 16;
    uint32_t rank = context[1];
    using Send = pto::Tile<pto::TileType::Mat, half, 2, 16, pto::BLayout::RowMajor, 2, 16, pto::SLayout::NoneBox>;
    using Recv = pto::Tile<pto::TileType::Mat, half, 2, 32, pto::BLayout::RowMajor, 2, 32, pto::SLayout::NoneBox>;
    Send send;
    Recv recv;
    pto::TASSIGN(send, 0);
    pto::TASSIGN(recv, 4096);
    pto::GlobalTensor<half, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>> src(input + rank * 32);
    pto::GlobalTensor<half, pto::Shape<1, 1, 1, 2, 32>, pto::Stride<64, 64, 64, 32, 1>> dst(output + rank * 64);
    pto::TLOAD(send, src);
    pto::wse::RunCalendar<1, 2, 9>(reinterpret_cast<uint64_t>(context), send, recv, 64);
    pto::TSTORE(dst, recv);
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" void launch_smoke(void* stream, void* input, void* output, void* contexts)
{
    calendar_smoke<<<2, nullptr, stream>>>(
        static_cast<half*>(input), static_cast<half*>(output), static_cast<uint64_t*>(contexts));
}

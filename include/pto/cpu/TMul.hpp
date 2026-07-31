/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMUL_HPP
#define TMUL_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename TileDst, typename TileSrc0, typename TileSrc1>
void TMul_Impl(TileDst& dst, TileSrc0& src0, TileSrc1& src1, unsigned validRow, unsigned validCol)
{
    cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
        PTO_CPU_VECTORIZE_LOOP
        for (std::size_t c = 0; c < validCol; ++c) {
            dst.SetElement(r, c, src0.GetElement(r, c) * src1.GetElement(r, c));
        }
    });
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TMUL_IMPL(TileDst& dst, TileSrc0& src0, TileSrc1& src1)
{
    static_assert(
        std::is_same_v<typename TileDst::DType, typename TileSrc0::DType> &&
            std::is_same_v<typename TileDst::DType, typename TileSrc1::DType>,
        "TMUL input and output dtypes must match.");
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == row && src0.GetValidCol() == col,
        "Fix: TMUL input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == row && src1.GetValidCol() == col,
        "Fix: TMUL input tile src1 valid shape mismatch with output tile dst shape.");
    TMul_Impl(dst, src0, src1, row, col);
}
} // namespace pto
#endif

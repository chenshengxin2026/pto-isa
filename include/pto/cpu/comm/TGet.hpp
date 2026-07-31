/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGET_HPP
#define TGET_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
namespace comm {

template <typename GlobalDstData, typename GlobalSrcData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void Copy_Data(GlobalDstData& dstTensor, GlobalSrcData& srcTensor)
{
    typename GlobalDstData::DType* dst = dstTensor.data();
    typename GlobalSrcData::DType* src = srcTensor.data();
    int64_t shape[] = {
        dstTensor.GetShape(0), dstTensor.GetShape(1), dstTensor.GetShape(2), dstTensor.GetShape(3),
        dstTensor.GetShape(4)};
    int64_t dstStride[] = {
        dstTensor.GetStride(0), dstTensor.GetStride(1), dstTensor.GetStride(2), dstTensor.GetStride(3),
        dstTensor.GetStride(4)};
    int64_t srcStride[] = {
        srcTensor.GetStride(0), srcTensor.GetStride(1), srcTensor.GetStride(2), srcTensor.GetStride(3),
        srcTensor.GetStride(4)};

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                for (size_t l = 0; l < shape[3]; l++) {
                    for (size_t m = 0; m < shape[4]; m++) {
                        const int64_t dstIndex = i * dstStride[0] + j * dstStride[1] + k * dstStride[2] +
                                                 l * dstStride[3] + m * dstStride[4];
                        const int64_t srcIndex = i * srcStride[0] + j * srcStride[1] + k * srcStride[2] +
                                                 l * srcStride[3] + m * srcStride[4];
                        if constexpr (atomicType == AtomicType::AtomicNone) {
                            dst[dstIndex] = src[srcIndex];
                        } else {
                            dst[dstIndex] += src[srcIndex];
                        }
                    }
                }
            }
        }
    }
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& src1)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData& dst, GlobalSrcData& src, TileData& ping, TileData& pong)
{
    Copy_Data(dst, src);
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_IMPL(GlobalDstData& dst, GlobalSrcData& src, const AsyncSession& session)
{
    Copy_Data(dst, src);
    return AsyncEvent(0, engine);
}

} // namespace comm
} // namespace pto
#endif

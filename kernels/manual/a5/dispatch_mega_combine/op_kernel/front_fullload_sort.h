/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H
#define DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H

#include "front_reorder.h"

template <typename InputElement>
class FrontReorderFullLoad : public FrontReorderContext {
public:
    AICORE inline void Init(
        GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData)
    {
        InitCommonInputs(xGM, expertIdGM, expertTokenNumsGM, workspaceGM, tilingData);
        const auto& front = tilingData->frontReorderTiling;
        InitMinimalSortTiling(front);
        InitCommonCoreIdx();
        InitCommonWorkspacePtrs(front, false);
        InitCommonPeerWindow();
    }

    struct FullLoadRouteTiling {
        uint32_t needCoreNum = 0;
        uint32_t perCoreRows = 0;
        uint32_t lastCoreRows = 0;
        uint32_t activateRows = 0;
        uint32_t coreRows = 0;
    };

    AICORE inline FullLoadRouteTiling BuildFullLoadRouteTiling() const
    {
        FullLoadRouteTiling tiling;
        tiling.activateRows = routeElems_;
        if (routeElems_ == 0U || coreNum_ == 0U) {
            return tiling;
        }
        tiling.perCoreRows = static_cast<uint32_t>(ceilDiv(routeElems_, coreNum_));
        if (tiling.perCoreRows == 0U) {
            return tiling;
        }
        tiling.needCoreNum = static_cast<uint32_t>(ceilDiv(routeElems_, tiling.perCoreRows));
        tiling.lastCoreRows = routeElems_ - tiling.perCoreRows * (tiling.needCoreNum - 1U);
        if (coreIdx_ < tiling.needCoreNum) {
            tiling.coreRows = coreIdx_ == tiling.needCoreNum - 1U ? tiling.lastCoreRows : tiling.perCoreRows;
        }
        return tiling;
    }

    AICORE inline uint32_t FullLoadTileLength() const
    {
        const uint32_t lastCorePerLoop = sortLastCorePerLoopElems_ == 0U ? routeElems_ : sortLastCorePerLoopElems_;
        return static_cast<uint32_t>(alignUp(lastCorePerLoop, sizeof(int32_t)));
    }

    AICORE inline uint32_t FullLoadSortNum() const
    {
        return static_cast<uint32_t>(alignUp(FullLoadTileLength(), kFrontSortAlignElement));
    }

    AICORE inline uint64_t FullLoadRequiredUbBytes() const
    {
        constexpr uint64_t kOneCoreSortBuffer = 6U;
        constexpr uint64_t kOtherRouteBuffer = 3U;
        constexpr uint64_t kDynamicQuantFullLoadColsBuffer = 13U;
        constexpr uint64_t kScaleOutBytes = 64U;
        const uint64_t alignedRouteElems = alignUp(routeElems_, UB_ALIGN);
        const uint64_t sortSpace = alignedRouteElems * sizeof(int32_t) * kOneCoreSortBuffer;
        const uint64_t otherSpace = alignedRouteElems * sizeof(int32_t) * kOtherRouteBuffer;
        const uint64_t expertSpace = alignUp(static_cast<uint64_t>(expertNum_) * sizeof(int32_t), UB_ALIGN);
        const uint64_t quantSpace =
            alignUp(static_cast<uint64_t>(problemK_), UB_ALIGN) * kDynamicQuantFullLoadColsBuffer;
        return sortSpace + otherSpace + expertSpace + quantSpace + kScaleOutBytes;
    }

    AICORE inline bool FullLoadDynamicCapable() const
    {
        return routeElems_ != 0U && routeElems_ <= sortLoopMaxElement_ && problemK_ <= kLargeFullRowMaxK &&
               problemK_ % UB_ALIGN == 0U && FullLoadRequiredUbBytes() <= AtlasA5::UB_SIZE;
    }

    AICORE inline uint64_t FullLoadRouteBytes() const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(FullLoadSortNum()) * sizeof(int32_t));
    }

    AICORE inline uint64_t FullLoadPackedSortBytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FullLoadSortNum()) * 2U * sizeof(float));
    }

    AICORE inline uint64_t FullLoadSortKeyBytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FullLoadSortNum()) * sizeof(float));
    }

    AICORE inline uint64_t FullLoadCountBytes() const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(expertNumAligned_) * sizeof(int32_t));
    }

    AICORE inline uint64_t QuantRawBytes() const
    {
        return AlignBytes<InputElement>(static_cast<uint64_t>(problemK_) * sizeof(InputElement));
    }

    AICORE inline uint64_t QuantFp32Bytes() const
    {
        return AlignBytes<float>(static_cast<uint64_t>(problemK_) * sizeof(float));
    }

    AICORE inline uint64_t QuantOutBytes() const
    {
        return AlignBytes<int8_t>(static_cast<uint64_t>(problemK_) * sizeof(int8_t));
    }

    AICORE inline uint64_t QuantScaleBytes() const { return AlignBytes<float>(8U * sizeof(float)); }

    AICORE inline uint64_t FullLoadPayloadUb() const { return FullLoadRouteBytes(); }

    AICORE inline uint64_t FullLoadPackedSortUb() const { return FullLoadPayloadUb() + FullLoadRouteBytes(); }

    AICORE inline uint64_t FullLoadMergeTmpUb() const { return FullLoadPackedSortUb() + FullLoadPackedSortBytes(); }

    AICORE inline uint64_t FullLoadSortKeyUb() const { return FullLoadMergeTmpUb() + FullLoadPackedSortBytes(); }

    AICORE inline uint64_t FullLoadExpandedExpertUb() const { return FullLoadSortKeyUb() + FullLoadSortKeyBytes(); }

    AICORE inline uint64_t FullLoadExpandDstToSrcUb() const
    {
        return FullLoadExpandedExpertUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadExpandedRowIdxUb() const
    {
        return FullLoadExpandDstToSrcUb() + FullLoadRouteBytes();
    }

    AICORE inline uint64_t FullLoadSortScratchUb() const { return FullLoadExpandedRowIdxUb() + FullLoadRouteBytes(); }

    AICORE inline uint64_t FullLoadCountUb() const { return FullLoadSortScratchUb() + FullLoadSortKeyBytes(); }

    AICORE inline uint64_t QuantRawUb() const { return FullLoadSortScratchUb(); }

    AICORE inline uint64_t QuantFp32Ub() const { return QuantRawUb() + QuantRawBytes(); }

    AICORE inline uint64_t QuantTmpUb() const { return QuantFp32Ub() + QuantFp32Bytes(); }

    AICORE inline uint64_t QuantOutUb() const { return QuantTmpUb() + QuantFp32Bytes(); }

    AICORE inline uint64_t QuantScaleUb() const { return QuantOutUb() + QuantOutBytes(); }

    AICORE inline uint64_t QuantActualUbBytes() const { return QuantScaleUb() + QuantScaleBytes(); }
    AICORE inline uint64_t FullLoadSortActualUbBytes() const { return FullLoadCountUb() + FullLoadCountBytes(); }

    AICORE inline bool FullLoadSortEnabled() const
    {
        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        return coreIdx_ < tiling.needCoreNum && FullLoadDynamicCapable() && FullLoadSortNum() <= kFrontSortMaxElems &&
               FullLoadSortActualUbBytes() <= AtlasA5::UB_SIZE;
    }

    AICORE inline void BuildInverseExpandedRowIdx(uint32_t totalLength, uint32_t sortNum) const
    {
        PtoCastUb<float, int32_t>(
            FullLoadSortKeyUb(), FullLoadExpandDstToSrcUb(), totalLength, pto::RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_ALL);
        PtoMulScalarUb<float>(FullLoadSortKeyUb(), FullLoadSortKeyUb(), totalLength, -1.0F);
        pipe_barrier(PIPE_ALL);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        for (uint32_t idx = totalLength; idx < sortNum; ++idx) {
            PtoSetValue<float>(FullLoadSortKeyUb(), idx, kFrontSortNegInf);
            PtoSetValue<uint32_t>(FullLoadPayloadUb(), idx, 0U);
        }
        if (sortNum > totalLength) {
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
        }
        PtoFillArithProgressionInt32(FullLoadPayloadUb(), 0, 1, totalLength);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        FrontPackedSortTile packedTile(1, sortNum * 2U);
        FrontPackedSortTile mergeTmpTile(1, sortNum * 2U);
        FrontSortKeyTile srcTile(1, sortNum);
        FrontSortPayloadTile payloadTile(1, sortNum);
        pto::TASSIGN(srcTile, FullLoadSortKeyUb());
        pto::TASSIGN(payloadTile, FullLoadPayloadUb());
        pto::TASSIGN(packedTile, FullLoadPackedSortUb());
        pto::TASSIGN(mergeTmpTile, FullLoadMergeTmpUb());
        pto::TSORT32(packedTile, srcTile, payloadTile);
        pipe_barrier(PIPE_ALL);
        FrontMergePackedSortRecords(packedTile, mergeTmpTile, sortNum * 2U);

        FrontPackedPayloadTile packedPayloadTile(1, totalLength * 2U);
        FrontSortPayloadTile expandedRowIdxTile(1, totalLength);
        pto::TASSIGN(packedPayloadTile, FullLoadPackedSortUb());
        pto::TASSIGN(expandedRowIdxTile, FullLoadExpandedRowIdxUb());
        pto::TGATHER<FrontSortPayloadTile, FrontPackedPayloadTile, pto::MaskPattern::P1010>(
            expandedRowIdxTile, packedPayloadTile);
        pipe_barrier(PIPE_ALL);
    }

    AICORE inline void RunFullLoadSort() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }

        const uint32_t totalLength = routeElems_;
        const uint32_t sortNum = FullLoadSortNum();
        PtoLoadVector<int32_t>(0U, expertIdPtr_, totalLength);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(FullLoadPayloadUb(), 0, 1, totalLength);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        FrontSortInt32ToPackedUb(
            0U, FullLoadPayloadUb(), FullLoadPackedSortUb(), FullLoadMergeTmpUb(), FullLoadSortKeyUb(), totalLength,
            sortNum);
        pipe_barrier(PIPE_ALL);
        FrontExtractPackedSortResult(
            FullLoadExpandedExpertUb(), FullLoadExpandDstToSrcUb(), FullLoadSortScratchUb(), FullLoadPackedSortUb(),
            totalLength);
        pipe_barrier(PIPE_ALL);

        BuildInverseExpandedRowIdx(totalLength, sortNum);
    }

    AICORE inline void StoreExpandedRowIdxToGm() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }
        if (coreIdx_ == 0U) {
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            PtoStoreVector<int32_t>(expandedRowIdxPtr_, FullLoadExpandedRowIdxUb(), routeElems_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
    }

    AICORE inline void BuildLocalTokenPerExpertFromSort() const
    {
        if (!FullLoadSortEnabled()) {
            return;
        }

        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        const uint32_t ownerCore = tiling.needCoreNum == 0U ? 0U : tiling.needCoreNum - 1U;
        if (coreIdx_ != ownerCore) {
            return;
        }

        PtoFillUb<int32_t>(FullLoadCountUb(), 0, expertNumAligned_);
        pipe_barrier(PIPE_ALL);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();

        int32_t lastExpertId = PtoGetValue<int32_t>(FullLoadExpandedExpertUb(), 0U);
        int32_t tokenCount = 0;
        for (uint32_t idx = 0; idx < routeElems_; ++idx) {
            const int32_t curExpertId = PtoGetValue<int32_t>(FullLoadExpandedExpertUb(), idx);
            ++tokenCount;
            while (lastExpertId < curExpertId) {
                if (lastExpertId >= 0 && static_cast<uint32_t>(lastExpertId) < expertNumAligned_) {
                    PtoSetValue<int32_t>(FullLoadCountUb(), static_cast<uint32_t>(lastExpertId), tokenCount - 1);
                }
                tokenCount = 1;
                ++lastExpertId;
            }
        }
        if (lastExpertId >= 0 && static_cast<uint32_t>(lastExpertId) < expertNumAligned_) {
            PtoSetValue<int32_t>(FullLoadCountUb(), static_cast<uint32_t>(lastExpertId), tokenCount);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        PtoStoreVector<int32_t>(localTokenPerExpertPtr_, FullLoadCountUb(), expertNumAligned_);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    AICORE inline bool FullLoadQuantEnabled() const
    {
        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        return FullLoadSortEnabled() && topK_ != 0U && tiling.coreRows != 0U && problemK_ % UB_ALIGN == 0U &&
               QuantActualUbBytes() <= AtlasA5::UB_SIZE;
    }

    AICORE inline void QuantAndScatterPackedRows() const
    {
        if (!FullLoadQuantEnabled()) {
            return;
        }

        const FullLoadRouteTiling tiling = BuildFullLoadRouteTiling();
        uint32_t curRowsStart = coreIdx_ * tiling.perCoreRows;
        const uint32_t curRowsEnd = curRowsStart + tiling.coreRows - 1U;
        const uint32_t startXRow = curRowsStart / topK_;
        const uint32_t endXRow = curRowsEnd / topK_;

        for (uint32_t row = startXRow; row <= endXRow && row < problemM_; ++row) {
            FrontDynamicQuantRowToUb<InputElement>(
                reinterpret_cast<__gm__ InputElement*>(xPtr_) + static_cast<uint64_t>(row) * problemK_, problemK_,
                QuantRawUb(), QuantFp32Ub(), QuantTmpUb(), QuantOutUb(), QuantScaleUb());

            bool rowStored = false;
            while (curRowsStart <= curRowsEnd && curRowsStart / topK_ == row) {
                const int32_t outIndex = PtoGetValue<int32_t>(FullLoadExpandedRowIdxUb(), curRowsStart);
                ++curRowsStart;
                if (outIndex < 0 || static_cast<uint32_t>(outIndex) >= tiling.activateRows) {
                    continue;
                }
                FrontStoreQuantPackedRow(offsetAPtr_, static_cast<uint32_t>(outIndex), problemK_, QuantOutUb());
                rowStored = true;
            }
            if (rowStored) {
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>();
            }
        }
    }
};

#endif // DISPATCH_MEGA_COMBINE_FRONT_FULLLOAD_SORT_H

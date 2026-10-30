/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_FP4_LUT_CASES_H_
#define TCVT_FP4_LUT_CASES_H_

// name, source rows/columns, destination rows/columns, valid rows/columns,
// NZ destination, compact mode (0 Null, 1 Normal, 2 RowPlusOne), dynamic valid extents.
#define FP4_LUT_CASES(M)                                                  \
    M(MainND, 32, 512, 32, 512, 32, 512, false, 0, false)                 \
    M(MainNZ, 32, 512, 32, 512, 32, 512, true, 0, false)                  \
    M(MinimumND, 1, 64, 1, 64, 1, 64, false, 0, false)                    \
    M(MinimumNZ, 1, 64, 16, 64, 1, 64, true, 0, false)                    \
    M(SingleNibbleND, 1, 64, 1, 96, 1, 1, false, 0, true)                 \
    M(SingleNibbleNZ, 1, 64, 32, 96, 1, 1, true, 1, true)                 \
    M(StridedStaticND, 5, 640, 7, 704, 3, 579, false, 0, false)           \
    M(StridedDynamicND, 5, 640, 7, 704, 3, 579, false, 0, true)           \
    M(StridedStaticNZ, 5, 640, 32, 704, 3, 579, true, 0, false)           \
    M(StridedDynamicNZ, 5, 640, 32, 704, 3, 579, true, 0, true)           \
    M(CompactStaticNZ, 5, 640, 32, 704, 3, 579, true, 1, false)           \
    M(CompactDynamicNZ, 5, 640, 32, 704, 3, 579, true, 1, true)           \
    M(RectangularNZ, 19, 128, 48, 192, 17, 93, true, 0, false)            \
    M(RectangularCompactNZ, 19, 128, 48, 192, 17, 93, true, 1, true)      \
    M(WideND, 3, 1088, 5, 1152, 2, 1025, false, 0, false)                 \
    M(WideNZ, 3, 1088, 32, 1152, 2, 1025, true, 0, false)                 \
    M(SecondRegisterTailND, 2, 512, 4, 544, 1, 257, false, 0, true)       \
    M(SecondRegisterTailNZ, 2, 512, 32, 544, 1, 257, true, 0, true)       \
    M(Bulk64ND, 20, 128, 32, 192, 20, 128, false, 0, true)                \
    M(Bulk64NZ, 20, 128, 32, 192, 20, 128, true, 0, true)                 \
    M(Bulk64CompactNZ, 20, 128, 32, 192, 20, 128, true, 1, true)          \
    M(Bulk128ND, 20, 256, 32, 320, 20, 256, false, 0, true)               \
    M(Bulk128NZ, 20, 256, 32, 320, 20, 256, true, 0, true)                \
    M(Bulk128CompactNZ, 20, 256, 32, 320, 20, 256, true, 1, true)         \
    M(RowPlusOneFull, 128, 256, 128, 256, 128, 256, true, 2, false)       \
    M(RowPlusOneDynamicFull, 128, 256, 128, 256, 128, 256, true, 2, true) \
    M(RowPlusOneLegacyPadded, 32, 128, 33, 128, 32, 128, true, 2, false)  \
    M(RowPlusOneTail, 5, 640, 31, 704, 3, 579, true, 2, true)             \
    M(RowPlusOneMinimum, 1, 64, 16, 96, 1, 1, true, 2, true)              \
    M(RowPlusOneBulk64, 20, 128, 32, 192, 19, 128, true, 2, true)         \
    M(RowPlusOneBulk128, 20, 256, 32, 320, 19, 256, true, 2, true)        \
    M(RowPlusOneRegisterTail, 2, 512, 32, 544, 1, 257, true, 2, true)

constexpr int kFp4LutStoreSourceRows = 19;
constexpr int kFp4LutStoreColumns = 128;
constexpr int kFp4LutStoreRows = 32;
constexpr int kFp4LutStorePanelColumns = 32;

struct Fp4LutCase {
    const char* name;
    int sourceRows;
    int sourceCols;
    int destinationRows;
    int destinationCols;
    int validRows;
    int validCols;
    bool nz;
    int compact;
    bool dynamic;
};

#define FP4_LUT_DESCRIPTION(Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic) \
    {#Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic},
constexpr Fp4LutCase kFp4LutCases[] = {FP4_LUT_CASES(FP4_LUT_DESCRIPTION)};
#undef FP4_LUT_DESCRIPTION

enum class Fp4LutCaseId {
#define FP4_LUT_ENUM(Name, ...) Name,
    FP4_LUT_CASES(FP4_LUT_ENUM)
#undef FP4_LUT_ENUM
};

void launchFp4Lut(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream);
void launchFp4LutRegression(int pair, unsigned char* dst, unsigned char* src, void* stream);
void launchFp4LutBulkPrimitive(unsigned char* dst, unsigned char* src, int sourceOffset, bool rotation, void* stream);
void launchFp4LutStore(unsigned char* dst, unsigned char* src, int compactMode, void* stream);

#endif

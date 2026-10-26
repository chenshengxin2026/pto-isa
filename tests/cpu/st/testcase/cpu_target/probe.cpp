/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "probe.h"

#include <array>
#include <stdexcept>
#include <pto/pto-inst.hpp>

#if defined(PTO_URMA_SUPPORTED) || defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
#error "CPU target selection must not enable an NPU or communication target"
#endif

using namespace pto;

#if defined(PTO_CPU_SIM_TARGET_A5)
static_assert(GetCpuSimArch() == NPUArch::A5);
static_assert(GetCurrentArch() == ChipArch::A5);
#elif defined(PTO_CPU_SIM_TARGET_A2A3)
static_assert(GetCpuSimArch() == NPUArch::A2A3);
static_assert(GetCurrentArch() == ChipArch::A2A3);
#endif

extern "C" void PTO_TARGET_TEST_SYMBOL(TargetProbe* result, int requestedArch)
{
    if (requestedArch != 0) {
        NPU_MEMORY_INIT(requestedArch == 5 ? NPUArch::A5 : NPUArch::A2A3);
    }
    auto& memory = NPUMemoryModel::Instance();
    memory.EnsureInitialized();
    const bool a5 = memory.GetArch() == NPUArch::A5;
    result->architecture = a5 ? 5 : 2;
    result->ubBytes = memory.GetSizes()[1];
    result->accBytes = memory.GetSizes()[5];
    result->memoryModel = reinterpret_cast<std::uintptr_t>(&memory);
    result->ubStorage = reinterpret_cast<std::uintptr_t>(memory.GetUBBase());
    result->capabilityArch = static_cast<int>(GetCurrentArch());
    result->supportsFp8 = CurrArch::SupportsFp8;

    using Input = Tile<TileType::Vec, float, 8, 64>;
    using Output = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor>;
    Input input;
    Output output;
    TASSIGN(input, 0);
    TASSIGN(output, 4096);
    for (int i = 0; i < 8 * 64; i += 4) {
        input.data()[i] = 1.0e8f;
        input.data()[i + 1] = 1.0f;
        input.data()[i + 2] = -1.0e8f;
        input.data()[i + 3] = 1.0f;
    }
    TROWSUM(output, input, input);
    for (int row = 0; row < 8; ++row) {
        result->rowSums[row] = output.GetValue(row);
    }

    using LoadTile = Tile<TileType::Mat, float, 1, 8, BLayout::RowMajor, 1, 3, SLayout::NoneBox, 512, PadValue::Max>;
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, 3>, Stride<3, 3, 3, 3, 1>>;
    std::array<float, 3> source{2.0f, 3.0f, 4.0f};
    Global global(source.data());
    LoadTile tile;
    TASSIGN(tile, 0);
    std::fill_n(tile.data(), 8, -42.0f);
    TLOAD(tile, global);
    result->loadPadding = tile.data()[3];

    using FloatTile = Tile<TileType::Vec, float, 1, 16>;
    using BfloatTile = Tile<TileType::Vec, bfloat16_t, 1, 16>;
    FloatTile convertInput, convertOutput, convertTmp;
    BfloatTile convertBfloat;
    TASSIGN(convertInput, 8192);
    TASSIGN(convertOutput, 8256);
    TASSIGN(convertTmp, 8320);
    TASSIGN(convertBfloat, 8384);
    for (int i = 0; i < 16; ++i) {
        convertInput.data()[i] = static_cast<float>(i - 8) / 2.0f;
    }
    result->convertsBfloat16 = true;
    for (int overload = 0; overload < 4; ++overload) {
        std::fill_n(convertBfloat.data(), 16, static_cast<bfloat16_t>(42.0f));
        std::fill_n(convertOutput.data(), 16, 42.0f);
        if (overload == 0) {
            TCVT(convertBfloat, convertInput, RoundMode::CAST_RINT);
            TCVT(convertOutput, convertBfloat, RoundMode::CAST_RINT);
        } else if (overload == 1) {
            TCVT(convertBfloat, convertInput, RoundMode::CAST_RINT, SaturationMode::ON);
            TCVT(convertOutput, convertBfloat, RoundMode::CAST_RINT, SaturationMode::ON);
        } else if (overload == 2) {
            TCVT(convertBfloat, convertInput, convertTmp, RoundMode::CAST_RINT);
            TCVT(convertOutput, convertBfloat, convertTmp, RoundMode::CAST_RINT);
        } else {
            TCVT(convertBfloat, convertInput, convertTmp, RoundMode::CAST_RINT, SaturationMode::ON);
            TCVT(convertOutput, convertBfloat, convertTmp, RoundMode::CAST_RINT, SaturationMode::ON);
        }
        for (int i = 0; i < 16; ++i) {
            result->convertsBfloat16 &= convertOutput.data()[i] == convertInput.data()[i];
        }
    }

    if (a5) {
        Input high;
        TASSIGN(high, 200 * 1024);
        high.data()[0] = 42.0f;
        result->a5HighAddress = high.data()[0] == 42.0f;
    }
    if constexpr (CPU_SIM_TARGET != CpuSimTarget::Legacy) {
        memory.Initialize();
        NPU_MEMORY_INIT();
        const auto wrong = a5 ? NPUArch::A2A3 : NPUArch::A5;
        try {
            memory.Initialize(wrong);
        } catch (const std::invalid_argument&) {
            result->rejectsInitialize = true;
        }
        try {
            NPUMemoryModel::SetDefaultArch(wrong);
        } catch (const std::invalid_argument&) {
            result->rejectsDefault = true;
        }
        try {
            NPU_MEMORY_INIT(wrong);
        } catch (const std::invalid_argument&) {
            result->rejectsMemoryInit = true;
        }
    }
}

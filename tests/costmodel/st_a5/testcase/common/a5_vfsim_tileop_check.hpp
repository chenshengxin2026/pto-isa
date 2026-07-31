/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include "pto/costmodel/trace.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace pto::test::a5 {

namespace vf = ::pto::mocker::vf;

inline void ExpectLastVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    const uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());

    const auto& record = trace.executed_pto.back();
    EXPECT_GT(cycles, 0U);
    ASSERT_EQ(record.vf_infos.size(), 1U);

    const vf::VfInfo& info = record.vf_infos[0];
    ASSERT_EQ(info.tree.size(), 1U);
    ASSERT_TRUE(vf::IsLoop(info.tree[0]));
    EXPECT_EQ(vf::LoopDepth(info.tree), 1U);

    const vf::VfLoop& loop = vf::AsLoop(info.tree[0]);
    EXPECT_EQ(loop.count, expectedRepeat);
    EXPECT_EQ(vf::FlattenLeafInstrs(loop.body), expectedBody);
    EXPECT_EQ(loop.body.size(), expectedBody.size());
}

inline void ExpectLastBinaryVecTileOp(const std::vector<std::string>& expectedBody, uint64_t expectedRepeat)
{
    ExpectLastVecTileOp(expectedBody, expectedRepeat);

    const auto& trace = ::pto::mocker::GetTrace();
    ASSERT_FALSE(trace.executed_pto.empty());
    const auto& record = trace.executed_pto.back();
    ASSERT_EQ(record.vf_infos.size(), 1U);
    const vf::VfInfo& info = record.vf_infos[0];
    ASSERT_EQ(info.tree.size(), 1U);
    ASSERT_TRUE(vf::IsLoop(info.tree[0]));

    const vf::VfLoop& loop = vf::AsLoop(info.tree[0]);
    ASSERT_GE(loop.body.size(), 4U);
    const vf::VfInst& load0 = vf::AsInst(loop.body[0]);
    const vf::VfInst& load1 = vf::AsInst(loop.body[1]);
    const vf::VfInst& op = vf::AsInst(loop.body[2]);
    const vf::VfInst& store = vf::AsInst(loop.body[3]);

    EXPECT_EQ(load0.dst.size(), 1U);
    EXPECT_EQ(load0.src.size(), 1U);
    EXPECT_EQ(load0.dst[0].location, vf::MemLocation::PhyRegister);
    EXPECT_EQ(load0.src[0].location, vf::MemLocation::UB);
    EXPECT_EQ(load1.dst.size(), 1U);
    EXPECT_EQ(load1.src.size(), 1U);
    EXPECT_EQ(load1.dst[0].location, vf::MemLocation::PhyRegister);
    EXPECT_EQ(load1.src[0].location, vf::MemLocation::UB);
    EXPECT_EQ(op.dst.size(), 1U);
    ASSERT_GE(op.src.size(), 2U);
    EXPECT_EQ(op.src[0].name, load0.dst[0].name);
    EXPECT_EQ(op.src[1].name, load1.dst[0].name);
    EXPECT_EQ(store.dst.size(), 1U);
    EXPECT_EQ(store.src.size(), 1U);
    EXPECT_EQ(store.dst[0].location, vf::MemLocation::UB);
    EXPECT_EQ(store.src[0].name, op.dst[0].name);
}

} // namespace pto::test::a5

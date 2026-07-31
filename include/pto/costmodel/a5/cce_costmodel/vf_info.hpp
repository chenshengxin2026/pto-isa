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

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pto::mocker::vf {

enum class VfNodeKind : uint8_t {
    LOOP,
    INST,
    MEMBAR,
};

enum class MemLocation : uint8_t {
    PhyRegister,
    UB,
};

struct MemInfo {
    std::string name;
    MemLocation location = MemLocation::PhyRegister;
    std::string dtype;
};

inline bool operator==(const MemInfo& lhs, const MemInfo& rhs)
{
    return lhs.name == rhs.name && lhs.location == rhs.location && lhs.dtype == rhs.dtype;
}

struct VfInst {
    std::string opName;
    std::vector<MemInfo> dst;
    std::vector<MemInfo> src;
};

inline bool operator==(const VfInst& lhs, const VfInst& rhs)
{
    return lhs.opName == rhs.opName && lhs.dst == rhs.dst && lhs.src == rhs.src;
}

struct VfMemBar {
    std::string name;
};

struct VfNode;

struct VfLoop {
    uint64_t count = 0;
    std::vector<VfNode> body;
};

struct VfNode {
    VfNodeKind kind;
    std::variant<VfLoop, VfInst, VfMemBar> v;
};

struct VfInfo {
    std::string op;
    std::string shape;
    std::vector<VfNode> tree;
};

inline VfNode MakeLoop(uint64_t count, std::vector<VfNode> body)
{
    return VfNode{VfNodeKind::LOOP, VfLoop{count, std::move(body)}};
}
inline VfNode MakeInst(VfInst inst) { return VfNode{VfNodeKind::INST, std::move(inst)}; }
inline VfNode MakeInst(std::string_view name) { return MakeInst(VfInst{std::string{name}, {}, {}}); }
inline VfNode MakeMemBar(std::string_view name) { return VfNode{VfNodeKind::MEMBAR, VfMemBar{std::string{name}}}; }

inline const VfLoop& AsLoop(const VfNode& n) { return std::get<VfLoop>(n.v); }
inline const VfInst& AsInst(const VfNode& n) { return std::get<VfInst>(n.v); }
inline const VfMemBar& AsMemBar(const VfNode& n) { return std::get<VfMemBar>(n.v); }
inline VfLoop& AsLoop(VfNode& n) { return std::get<VfLoop>(n.v); }
inline VfInst& AsInst(VfNode& n) { return std::get<VfInst>(n.v); }
inline VfMemBar& AsMemBar(VfNode& n) { return std::get<VfMemBar>(n.v); }
inline bool IsLoop(const VfNode& n) { return n.kind == VfNodeKind::LOOP; }
inline bool IsInst(const VfNode& n) { return n.kind == VfNodeKind::INST; }
inline bool IsMemBar(const VfNode& n) { return n.kind == VfNodeKind::MEMBAR; }

} // namespace pto::mocker::vf

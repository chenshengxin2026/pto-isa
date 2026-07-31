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

#include "vf_info.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pto::mocker::vf::trace {

enum class EvKind : uint8_t {
    LoopEnter,
    LoopIter,
    LoopExit,
    Op,
    MemBar,
};

struct Event {
    EvKind kind;
    uint64_t loopId = 0;
    std::string opName;
    VfInst inst;
};

inline std::vector<Event>& Events()
{
    static thread_local std::vector<Event> e;
    return e;
}
inline bool& Armed()
{
    static thread_local bool a = false;
    return a;
}

inline void Reset() { Events().clear(); }
inline void Arm(bool v) { Armed() = v; }

inline void RecordOp(VfInst inst)
{
    if (Armed())
        Events().push_back({EvKind::Op, 0, inst.opName, std::move(inst)});
}
inline void RecordOp(std::string name) { RecordOp(VfInst{std::move(name), {}, {}}); }
inline void RecordMemBar(std::string name)
{
    if (Armed())
        Events().push_back({EvKind::MemBar, 0, std::move(name)});
}

} // namespace pto::mocker::vf::trace

#define PTO_TRACE_HOOK __attribute__((used)) inline
extern "C" {
PTO_TRACE_HOOK void __pto_trace_loop_enter(uint64_t loopId, const char* /*file*/, int /*line*/, int /*col*/)
{
    if (!pto::mocker::vf::trace::Armed())
        return;
    pto::mocker::vf::trace::Events().push_back({pto::mocker::vf::trace::EvKind::LoopEnter, loopId, {}});
}
PTO_TRACE_HOOK void __pto_trace_loop_iter(uint64_t loopId)
{
    if (!pto::mocker::vf::trace::Armed())
        return;
    pto::mocker::vf::trace::Events().push_back({pto::mocker::vf::trace::EvKind::LoopIter, loopId, {}});
}
PTO_TRACE_HOOK void __pto_trace_loop_exit(uint64_t loopId)
{
    if (!pto::mocker::vf::trace::Armed())
        return;
    pto::mocker::vf::trace::Events().push_back({pto::mocker::vf::trace::EvKind::LoopExit, loopId, {}});
}
PTO_TRACE_HOOK void __pto_vf_scope_enter() {}
PTO_TRACE_HOOK void __pto_vf_scope_exit() {}
}
#undef PTO_TRACE_HOOK

namespace pto::mocker::vf::trace {

namespace detail {

constexpr uint64_t kNoEnclosing = static_cast<uint64_t>(-1);

inline bool NodesEqual(const std::vector<VfNode>& a, const std::vector<VfNode>& b);

inline bool NodeEqual(const VfNode& a, const VfNode& b)
{
    if (a.kind != b.kind)
        return false;
    switch (a.kind) {
        case VfNodeKind::INST:
            return AsInst(a) == AsInst(b);
        case VfNodeKind::MEMBAR:
            return AsMemBar(a).name == AsMemBar(b).name;
        case VfNodeKind::LOOP: {
            const VfLoop &la = AsLoop(a), &lb = AsLoop(b);
            return la.count == lb.count && NodesEqual(la.body, lb.body);
        }
    }
    return false;
}

inline bool NodesEqual(const std::vector<VfNode>& a, const std::vector<VfNode>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!NodeEqual(a[i], b[i]))
            return false;
    return true;
}

struct Parser {
    const std::vector<Event>& ev;
    size_t i = 0;
    bool ok = true;
    std::string err;

    explicit Parser(const std::vector<Event>& e) : ev(e) {}

    std::vector<VfNode> parseBody(uint64_t enclosingId)
    {
        std::vector<VfNode> nodes;
        while (i < ev.size()) {
            const Event& e = ev[i];
            if (e.kind == EvKind::LoopIter || e.kind == EvKind::LoopExit) {
                if (enclosingId != kNoEnclosing && e.loopId == enclosingId)
                    return nodes;
                ok = false;
                err = "stray loop_iter/exit for loopId=" + std::to_string(e.loopId);
                return nodes;
            }
            if (e.kind == EvKind::Op) {
                nodes.push_back(MakeInst(e.inst));
                ++i;
            } else if (e.kind == EvKind::MemBar) {
                nodes.push_back(MakeMemBar(e.opName));
                ++i;
            } else if (e.kind == EvKind::LoopEnter) {
                std::vector<VfNode> loop = parseLoop();
                if (!ok)
                    return nodes;
                nodes.push_back(MakeLoop(0, {}));
                nodes.back() = std::move(loop.front());
            } else {
                ok = false;
                err = "unexpected event kind in body";
                return nodes;
            }
        }
        return nodes;
    }

    std::vector<VfNode> parseLoop()
    {
        uint64_t id = ev[i].loopId;
        ++i; // consume LoopEnter
        uint64_t count = 0;
        std::vector<VfNode> canonBody;
        bool haveCanon = false;
        while (i < ev.size()) {
            const Event& e = ev[i];
            if (e.kind == EvKind::LoopIter && e.loopId == id) {
                ++i; // consume iter
                std::vector<VfNode> body = parseBody(id);
                if (!ok)
                    return {};
                if (body.empty())
                    continue;
                ++count;
                if (!haveCanon) {
                    canonBody = std::move(body);
                    haveCanon = true;
                } else if (!NodesEqual(canonBody, body)) {
                    ok = false;
                    err = "non-uniform loop body for loopId=" + std::to_string(id) +
                          " (VF template should be a uniform loop)";
                    return {};
                }
            } else if (e.kind == EvKind::LoopExit && e.loopId == id) {
                ++i; // consume exit
                break;
            } else {
                ok = false;
                err = "malformed loop: expected iter/exit for loopId=" + std::to_string(id);
                return {};
            }
        }
        return {MakeLoop(count, std::move(canonBody))};
    }
};

} // namespace detail

struct BuildResult {
    bool ok = false;
    VfInfo info;
    std::string err;
};

inline BuildResult BuildVfInfo(std::string_view op, std::string_view shape)
{
    BuildResult r;
    detail::Parser p(Events());
    std::vector<VfNode> tree = p.parseBody(detail::kNoEnclosing);
    if (!p.ok || p.i != Events().size()) {
        r.ok = false;
        r.err = p.ok ? ("unconsumed events at index " + std::to_string(p.i)) : p.err;
        return r;
    }
    r.ok = true;
    r.info.op = op;
    r.info.shape = shape;
    r.info.tree = std::move(tree);
    return r;
}

inline void FormatNodes(const std::vector<VfNode>& nodes, int depth, std::string& out)
{
    std::string ind(depth * 2, ' ');
    for (const VfNode& n : nodes) {
        if (IsLoop(n)) {
            const VfLoop& lp = AsLoop(n);
            out += ind + "loop{n=" + std::to_string(lp.count) + "}\n";
            FormatNodes(lp.body, depth + 1, out);
        } else if (IsInst(n)) {
            out += ind + "op(" + AsInst(n).opName + ")\n";
        } else {
            out += ind + "membar(" + AsMemBar(n).name + ")\n";
        }
    }
}

inline std::string FormatTree(const VfInfo& vf)
{
    std::string out = "program(" + vf.op + ")\n";
    FormatNodes(vf.tree, 1, out);
    return out;
}

} // namespace pto::mocker::vf::trace

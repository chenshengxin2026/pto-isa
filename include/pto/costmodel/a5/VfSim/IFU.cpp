/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/IFU.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vfsim {
namespace {

int64_t resolveInt(
    const std::string& text, const ProgramAnalysis::ParamMap& params, int64_t defaultValue, int64_t minValue)
{
    if (text.empty())
        return defaultValue;
    bool digits = true;
    size_t pos = 0;
    if (text[0] == '-') {
        pos = 1;
        digits = text.size() > 1;
    }
    for (; digits && pos < text.size(); ++pos) {
        if (!std::isdigit(static_cast<unsigned char>(text[pos])))
            digits = false;
    }
    if (digits) {
        try {
            return std::max<int64_t>(minValue, std::stoll(text));
        } catch (...) {
            return defaultValue;
        }
    }
    auto it = params.find(text);
    if (it != params.end())
        return std::max<int64_t>(minValue, it->second);
    return defaultValue;
}

} // namespace

IFU::IFU(
    const std::vector<LinearProgramNode>& linearNodes, ProgramAnalysis::ParamMap params, const ParamDB* db,
    std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds, int64_t totalTopBlocks, std::string dtype)
    : nodes_(linearNodes), analysis_(std::move(params)), dtype_(std::move(dtype))
{
    db_ = db;
    buildIndices();
    if (!topBlockLoopBounds.empty())
        topBlockLoopBounds_ = std::move(topBlockLoopBounds);
    else
        topBlockLoopBounds_ = analysis_.inferTopBlockLoopBounds({});
    if (totalTopBlocks > 0)
        totalTopBlocks_ = totalTopBlocks;
}

bool IFU::isInst(const LinearProgramNode& node) { return node.type == "inst"; }
bool IFU::isLoopBegin(const LinearProgramNode& node) { return node.type == "loop_begin"; }
bool IFU::isLoopEnd(const LinearProgramNode& node) { return node.type == "loop_end"; }

bool IFU::containsAnyLoop(const std::vector<LinearProgramNode>& nodes)
{
    for (const auto& node : nodes) {
        if (node.type == "loop_begin")
            return true;
    }
    return false;
}

void IFU::buildIndices()
{
    const std::vector<int64_t> begins = collectLoopBegins();
    markInnermostLoops(begins);
    assignTopBlockIds(begins);
    cacheLoopBodies(begins);
    cacheTopBlockLastInsts();
}

std::vector<int64_t> IFU::collectLoopBegins()
{
    std::vector<int64_t> stack;
    for (int64_t i = 0; i < static_cast<int64_t>(nodes_.size()); ++i) {
        const auto& node = nodes_[static_cast<size_t>(i)];
        if (isLoopBegin(node)) {
            stack.push_back(i);
            beginLoopId_[i] = static_cast<int64_t>(beginLoopId_.size());
        } else if (isLoopEnd(node)) {
            if (stack.empty())
                throw std::runtime_error("Unmatched loop_end in linear program");
            const int64_t begin = stack.back();
            stack.pop_back();
            beginToEnd_[begin] = i;
        }
    }
    if (!stack.empty())
        throw std::runtime_error("Unmatched loop_begin in linear program");

    std::vector<int64_t> begins;
    begins.reserve(beginToEnd_.size());
    for (const auto& [b, _] : beginToEnd_)
        begins.push_back(b);
    std::sort(begins.begin(), begins.end());
    return begins;
}

void IFU::markInnermostLoops(const std::vector<int64_t>& begins)
{
    for (const auto b : begins) {
        const auto e = beginToEnd_.at(b);
        bool nested = false;
        for (const auto b2 : begins) {
            if (b2 == b)
                continue;
            if (b < b2 && b2 < e) {
                nested = true;
                break;
            }
        }
        isInnermostBegin_[b] = !nested;
    }
}

void IFU::assignTopBlockIds(const std::vector<int64_t>& begins)
{
    int64_t topBid = 0;
    for (const auto b : begins) {
        bool enclosed = false;
        for (const auto b2 : begins) {
            if (b2 == b)
                continue;
            const auto e2 = beginToEnd_.at(b2);
            if (b2 < b && b < e2) {
                enclosed = true;
                break;
            }
        }
        if (!enclosed)
            beginTopBlockId_[b] = topBid++;
    }
    totalTopBlocks_ = topBid;
}

std::optional<int64_t> IFU::findLastInstIdx(int64_t begin, int64_t end) const
{
    std::optional<int64_t> lastIdx;
    for (int64_t i = begin + 1; i < end; ++i) {
        if (isInst(nodes_[static_cast<size_t>(i)]))
            lastIdx = i;
    }
    return lastIdx;
}

void IFU::cacheLoopBodies(const std::vector<int64_t>& begins)
{
    for (const auto b : begins) {
        const auto e = beginToEnd_.at(b);
        std::vector<LinearProgramNode> body;
        for (int64_t i = b + 1; i < e; ++i) {
            if (isInst(nodes_[static_cast<size_t>(i)]))
                body.push_back(nodes_[static_cast<size_t>(i)]);
        }
        if (isInnermostBegin_.at(b))
            loopBodyCache_[b] = std::move(body);

        loopLastInstIdx_[b] = findLastInstIdx(b, e);
    }
}

void IFU::cacheTopBlockLastInsts()
{
    for (const auto& [b, tbid] : beginTopBlockId_) {
        const auto e = beginToEnd_.at(b);
        topBlockLastInstIdx_[tbid] = findLastInstIdx(b, e);
    }
}

bool IFU::done() const { return pc_ >= static_cast<int64_t>(nodes_.size()) && pending_.empty(); }

std::pair<std::vector<int64_t>, std::vector<int64_t>> IFU::snapshot() const
{
    std::vector<int64_t> loopIds;
    std::vector<int64_t> iterNow;
    loopIds.reserve(frames_.size());
    iterNow.reserve(frames_.size());
    for (const auto& fr : frames_) {
        loopIds.push_back(fr.loopId);
        iterNow.push_back(fr.iterNow);
    }
    return {std::move(loopIds), std::move(iterNow)};
}

int64_t IFU::currentTopBlockId() const
{
    if (!frames_.empty())
        return frames_.front().topBlockId;
    return 0;
}

std::pair<std::string, std::vector<int64_t>> IFU::makeKey(
    int64_t topBlockId, const std::string& loopId, std::vector<int64_t> it) const
{
    it.insert(it.begin(), topBlockId);
    return {loopId, std::move(it)};
}

std::pair<std::string, std::vector<int64_t>> IFU::normalizeBlockKey(
    const std::pair<std::string, std::vector<int64_t>>& raw, int64_t topBlockId) const
{
    if (raw.second.empty())
        return raw;
    std::vector<int64_t> it = raw.second;
    if (!it.empty())
        it.insert(it.begin(), topBlockId);
    return {raw.first, std::move(it)};
}

std::vector<std::pair<std::string, std::vector<int64_t>>> IFU::buildBlockKeyByLevel(
    const std::vector<int64_t>& loopStack, const std::vector<int64_t>& iterStack) const
{
    std::vector<std::pair<std::string, std::vector<int64_t>>> out;
    for (size_t lv = 0; lv < loopStack.size(); ++lv) {
        std::vector<int64_t> prefix(iterStack.begin(), iterStack.begin() + static_cast<std::ptrdiff_t>(lv));
        out.emplace_back("loop" + std::to_string(lv), std::move(prefix));
    }
    return out;
}

std::vector<int64_t> IFU::calcBlockEndLevelsNormal() const
{
    if (frames_.empty())
        return {};

    const int64_t deepest = static_cast<int64_t>(frames_.size()) - 1;
    const LoopFrame& deepestFr = frames_.back();
    const auto it = loopLastInstIdx_.find(deepestFr.beginIdx);
    if (it == loopLastInstIdx_.end() || !it->second.has_value() || pc_ != it->second.value())
        return {};

    std::vector<int64_t> endLevels;
    for (int64_t lv = deepest; lv >= 0; --lv) {
        if (areFinalLoopIterations(lv, deepest, nullptr, true))
            endLevels.push_back(lv);
        else
            break;
    }
    return endLevels;
}

bool IFU::areFinalLoopIterations(
    int64_t firstLevel, int64_t deepest, const LoopFrame* unrolledFrame, bool isLastSuperIter) const
{
    for (int64_t kk = firstLevel; kk <= deepest; ++kk) {
        const auto& fr = frames_[static_cast<size_t>(kk)];
        const bool finalNow = unrolledFrame != nullptr && fr.beginIdx == unrolledFrame->beginIdx ?
                                  isLastSuperIter :
                                  fr.iterNow == fr.itersTotal - 1;
        if (!finalNow)
            return false;
    }
    return true;
}

bool IFU::isLastInTopBlockNormal() const
{
    if (frames_.empty())
        return false;
    const int64_t tbid = currentTopBlockId();
    const auto it = topBlockLastInstIdx_.find(tbid);
    if (it == topBlockLastInstIdx_.end() || !it->second.has_value() || pc_ != it->second.value())
        return false;
    for (const auto& fr : frames_) {
        if (fr.iterNow != fr.itersTotal - 1)
            return false;
    }
    return true;
}

DynamicInst IFU::emitNormalInst(const LinearProgramNode& node)
{
    DynamicInst out;
    out.instId = instId_++;
    out.type = node.type;
    out.op = node.op;
    out.form = node.form.empty() ? dtype_ : node.form;
    out.src = node.src;
    out.dst = node.dst;

    const auto [loopStack, iterStack] = snapshot();
    out.loopStack = loopStack;
    out.iterStack = iterStack;
    out.loopDepth = static_cast<int64_t>(loopStack.size());
    out.inLoop = !loopStack.empty();
    out.unrollFactor = 1;
    out.lane = -1;
    out.topBlockId = currentTopBlockId();
    out.isLastInTopBlock = isLastInTopBlockNormal();
    out.blockKeyByLevel = buildBlockKeyByLevel(loopStack, iterStack);
    out.blockEndLevels = calcBlockEndLevelsNormal();
    return out;
}

DynamicInst IFU::emitUnrolledInst(
    const LinearProgramNode& node, const LoopFrame& frame, const std::vector<int64_t>& loopStack,
    const std::vector<int64_t>& iterStack, int64_t superIter, int64_t lane)
{
    DynamicInst inst;
    inst.instId = instId_++;
    inst.type = node.type;
    inst.op = node.op;
    inst.form = node.form.empty() ? dtype_ : node.form;
    inst.src = node.src;
    inst.dst = node.dst;
    inst.loopStack = loopStack;
    if (!iterStack.empty()) {
        inst.iterStack = iterStack;
        inst.iterStack.back() = superIter;
    }
    inst.loopDepth = static_cast<int64_t>(loopStack.size());
    inst.inLoop = true;
    inst.unrollFactor = frame.unroll;
    inst.unrollGroup = unrollGroup_;
    inst.lane = lane;
    inst.origIterBase = frame.iterNow;
    for (auto& src : inst.src)
        src += "_lane" + std::to_string(lane);
    for (auto& dst : inst.dst)
        dst += "_lane" + std::to_string(lane);
    inst.topBlockId = frame.topBlockId;
    inst.isLastInTopBlock = false;
    inst.blockKeyByLevel = buildBlockKeyByLevel(loopStack, inst.iterStack);
    inst.blockEndLevels.clear();
    return inst;
}

std::vector<int64_t> IFU::calcBlockEndLevelsUnrolled(
    const std::vector<int64_t>& loopStack, const LoopFrame& frame, bool isLastSuperIter) const
{
    std::vector<int64_t> endLevels;
    if (!isLastSuperIter)
        return endLevels;

    for (int64_t lv = static_cast<int64_t>(loopStack.size()) - 1; lv >= 0; --lv) {
        const int64_t deepest = static_cast<int64_t>(frames_.size()) - 1;
        if (areFinalLoopIterations(lv, deepest, &frame, isLastSuperIter))
            endLevels.push_back(lv);
        else
            break;
    }
    return endLevels;
}

bool IFU::isLastUnrolledTopBlock(const LoopFrame& frame) const
{
    for (const auto& fr : frames_) {
        if (fr.beginIdx == frame.beginIdx)
            continue;
        if (fr.iterNow != fr.itersTotal - 1)
            return false;
    }
    return true;
}

void IFU::markLastPendingUnrolled(
    std::vector<DynamicInst>& pending, const std::vector<int64_t>& loopStack, const LoopFrame& frame,
    bool isLastSuperIter) const
{
    if (pending.empty())
        return;
    pending.back().blockEndLevels = calcBlockEndLevelsUnrolled(loopStack, frame, isLastSuperIter);
    if (isLastSuperIter)
        pending.back().isLastInTopBlock = isLastUnrolledTopBlock(frame);
}

void IFU::buildPendingUnrolled(LoopFrame& frame)
{
    const auto it = loopBodyCache_.find(frame.beginIdx);
    const std::vector<LinearProgramNode> empty;
    const std::vector<LinearProgramNode>& body = it == loopBodyCache_.end() ? empty : it->second;

    const auto [loopStack, iterStack] = snapshot();
    const int64_t U = frame.unroll;
    const int64_t origBase = frame.iterNow;
    const int64_t superIter = U > 0 ? origBase / U : origBase;
    const bool isLastSuperIter = (origBase + U >= frame.itersTotal);

    std::vector<DynamicInst> pending;
    pending.reserve(body.size() * static_cast<size_t>(std::max<int64_t>(1, U)));

    for (const auto& ins : body) {
        for (int64_t lane = 0; lane < U; ++lane) {
            pending.push_back(emitUnrolledInst(ins, frame, loopStack, iterStack, superIter, lane));
        }
    }

    markLastPendingUnrolled(pending, loopStack, frame, isLastSuperIter);

    ++unrollGroup_;
    for (auto& inst : pending)
        pending_.push_back(std::move(inst));
    frame.iterNow += U;
}

std::optional<DynamicInst> IFU::popPending()
{
    if (pending_.empty())
        return std::nullopt;
    DynamicInst out = std::move(pending_.front());
    pending_.pop_front();
    return out;
}

void IFU::enterLoop(const LinearProgramNode& node)
{
    const int64_t iters = resolveInt(node.itersRaw, analysis_.params(), 1, 0);
    const int64_t end = beginToEnd_.at(pc_);
    const int64_t loopId = beginLoopId_.at(pc_);
    const bool isInnermost = isInnermostBegin_.at(pc_);
    const int64_t unroll = resolveInt(node.unrollRaw, analysis_.params(), 1, 1);
    if (iters <= 0) {
        pc_ = end + 1;
        return;
    }

    const int64_t topBlockId = frames_.empty() ? beginTopBlockId_.at(pc_) : frames_.front().topBlockId;
    if (isInnermost && unroll > 1 && iters % unroll != 0)
        throw std::runtime_error("Invalid unroll: iters not divisible by unroll");

    frames_.push_back(
        LoopFrame{pc_, end, loopId, iters, 0, isInnermost, (isInnermost && unroll > 1) ? unroll : 1, topBlockId});
    pc_ = isInnermost && unroll > 1 ? end : pc_ + 1;
}

std::optional<DynamicInst> IFU::handleLoopEnd()
{
    if (frames_.empty())
        throw std::runtime_error("loop_end encountered with empty runtime stack");
    LoopFrame& top = frames_.back();
    if (top.endIdx != pc_)
        throw std::runtime_error("loop_end mismatch with runtime top frame");

    if (top.isInnermost && top.unroll > 1) {
        if (top.iterNow < top.itersTotal) {
            buildPendingUnrolled(top);
            return popPending();
        }
        frames_.pop_back();
        ++pc_;
        return std::nullopt;
    }

    if (top.iterNow + 1 < top.itersTotal) {
        ++top.iterNow;
        pc_ = top.beginIdx + 1;
        return std::nullopt;
    }

    frames_.pop_back();
    ++pc_;
    return std::nullopt;
}

std::optional<DynamicInst> IFU::nextInst()
{
    if (auto out = popPending())
        return out;

    while (pc_ < static_cast<int64_t>(nodes_.size())) {
        const auto& n = nodes_[static_cast<size_t>(pc_)];
        if (n.type == "loop_begin") {
            enterLoop(n);
            continue;
        }

        if (n.type == "loop_end") {
            if (auto out = handleLoopEnd())
                return out;
            continue;
        }

        if (n.type != "inst") {
            ++pc_;
            continue;
        }

        DynamicInst out = emitNormalInst(n);
        ++pc_;
        return out;
    }

    return std::nullopt;
}

std::vector<DynamicInst> IFU::take(int64_t n)
{
    std::vector<DynamicInst> out;
    for (int64_t i = 0; i < std::max<int64_t>(0, n); ++i) {
        auto inst = nextInst();
        if (!inst.has_value())
            break;
        out.push_back(std::move(*inst));
    }
    return out;
}

} // namespace vfsim

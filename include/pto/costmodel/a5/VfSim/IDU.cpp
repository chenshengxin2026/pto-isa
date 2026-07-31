/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/IDU.h"

#include "pto/costmodel/a5/VfSim/ISATraits.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vfsim {
namespace {

std::string joinInts(const std::vector<int64_t>& values)
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ',';
        oss << values[i];
    }
    return oss.str();
}

} // namespace

IDU::IDU(
    const UarchConfig& uarch, const ParamDB& db, ProgramAnalysis::ParamMap params, std::vector<int64_t> loopBounds,
    int64_t totalTopBlocks, std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds, std::string dtype,
    std::unordered_map<std::string, ValueInfo> values)
    : db_(db),
      dtype_(std::move(dtype)),
      analysis_(std::move(params), std::move(values)),
      loopBounds_(std::move(loopBounds)),
      totalTopBlocks_(totalTopBlocks),
      topBlockLoopBounds_(std::move(topBlockLoopBounds))
{
    windowWidth_ = uarch.iduWindowWidth;
    issueWidth_ = uarch.iduIssueWidth;
    theoreticalLimitMode_ = false;
    theoreticalLimitVloopOnly_ = false;
    vfStartupCost_ = db_.isaDefaults().vfStartupCost;
    iduDispatchStartAdvance_ = uarch.iduDispatchStartAdvance;
    vloopToDispatchDelay_ = uarch.vloopToDispatchDelay;
    initialTopBlockVloopStartCycle_ = uarch.initialTopBlockVloopStartCycle;
    nestedVloopInitialStartGap_ = uarch.nestedVloopInitialStartGap;
    loop1MinFeedbackGap_ = uarch.loop1MinFeedbackGap;
    innermostIterDispatchStride_ = uarch.innermostIterDispatchStride;
    globalShqPregGate_ = uarch.globalShqPregGate;
    initVloopStarts();
}

bool IDU::canAccept() const
{
    if (theoreticalLimitMode_)
        return true;
    return static_cast<int64_t>(window_.size()) < windowWidth_;
}

void IDU::accept(const DynamicInst& inst)
{
    if (theoreticalLimitMode_ || static_cast<int64_t>(window_.size()) < windowWidth_)
        window_.push_back(inst);
}

void IDU::setTopBlockVloop(int64_t topBlockId, int64_t startCycle)
{
    if (topBlockVloopStart_.count(topBlockId))
        return;
    topBlockVloopStart_[topBlockId] = startCycle;
    topBlockBodyOpenTime_[topBlockId] = startCycle + vloopToDispatchDelay_;
    vloopTrace_.push_back(VloopTraceRecord{topBlockId, "top_block", {}, startCycle});
}

void IDU::initTopBlockNestedStarts(int64_t topBlockId, int64_t topVloopStart)
{
    const auto it = topBlockLoopBounds_.find(static_cast<int>(topBlockId));
    const std::vector<int64_t>& bounds = it == topBlockLoopBounds_.end() ? loopBounds_ : it->second;
    const int64_t depth = static_cast<int64_t>(bounds.size());
    if (depth <= 0)
        return;

    setTopBlockVloop(topBlockId, topVloopStart);
    if (depth >= 1) {
        const std::string key0 = makeKey(topBlockId, "loop0", {});
        vloopStart_[key0] = topVloopStart;
        bodyOpenTime_[key0] = topVloopStart + vloopToDispatchDelay_;
    }
    if (depth >= 2 && bounds[0] > 0) {
        const std::string key1 = makeKey(topBlockId, "loop1", {0});
        vloopStart_[key1] = topVloopStart + nestedVloopInitialStartGap_;
        bodyOpenTime_[key1] = vloopStart_[key1] + vloopToDispatchDelay_;
    }
    if (depth >= 3 && bounds[0] > 0 && bounds[1] > 0) {
        const std::string key2 = makeKey(topBlockId, "loop2", {0, 0});
        vloopStart_[key2] = topVloopStart + 2 * nestedVloopInitialStartGap_;
        bodyOpenTime_[key2] = vloopStart_[key2] + vloopToDispatchDelay_;
    }
}

void IDU::initVloopStarts()
{
    if (totalTopBlocks_ <= 0)
        return;
    setTopBlockVloop(0, initialTopBlockVloopStartCycle_);
    initTopBlockNestedStarts(0, initialTopBlockVloopStartCycle_);
}

std::string IDU::makeKey(int64_t topBlockId, const std::string& loopId, const std::vector<int64_t>& iters) const
{
    return std::to_string(topBlockId) + "|" + loopId + "|" + joinInts(iters);
}

std::optional<std::string> IDU::normalizeBlockKey(
    const std::pair<std::string, std::vector<int64_t>>& raw, int64_t topBlockId) const
{
    if (raw.first.empty())
        return std::nullopt;
    return makeKey(topBlockId, raw.first, raw.second);
}

std::optional<std::string> IDU::currentInnerBlockKey(const DynamicInst& inst) const
{
    const int64_t topBlockId = inst.topBlockId;
    const auto& bk = inst.blockKeyByLevel;
    if (!bk.empty()) {
        return normalizeBlockKey(bk.back(), topBlockId);
    }

    const auto& iterStack = inst.iterStack;
    const int64_t depth =
        std::min<int64_t>(static_cast<int64_t>(loopBounds_.size()), static_cast<int64_t>(inst.loopDepth));
    if (depth <= 0)
        return std::nullopt;
    if (depth == 1)
        return makeKey(topBlockId, "loop0", {});
    if (depth == 2 && iterStack.size() >= 1)
        return makeKey(topBlockId, "loop1", {iterStack[0]});
    if (depth >= 3 && iterStack.size() >= 2)
        return makeKey(topBlockId, "loop2", {iterStack[0], iterStack[1]});
    return std::nullopt;
}

bool IDU::isLastInstOfTopBlock(const DynamicInst& inst) const
{
    if (inst.blockEndLevels.empty())
        return false;
    return inst.isLastInTopBlock;
}

const std::vector<int64_t>& IDU::loopBoundsForTopBlock(int64_t topBlockId) const
{
    const auto it = topBlockLoopBounds_.find(static_cast<int>(topBlockId));
    return it == topBlockLoopBounds_.end() ? loopBounds_ : it->second;
}

bool IDU::hasBlockEndLevel(const DynamicInst& inst, int64_t level) const
{
    return std::find(inst.blockEndLevels.begin(), inst.blockEndLevels.end(), level) != inst.blockEndLevels.end();
}

void IDU::triggerNextTopBlock(const DynamicInst& inst, int64_t cycle)
{
    if (!inst.isLastInTopBlock)
        return;
    const int64_t nextTop = inst.topBlockId + 1;
    if (nextTop >= totalTopBlocks_ || topBlockVloopStart_.count(nextTop))
        return;
    setTopBlockVloop(nextTop, cycle);
    initTopBlockNestedStarts(nextTop, cycle);
}

int64_t IDU::lastDispatchOrCycle(const std::string& key, int64_t cycle) const
{
    const auto it = lastDispatchTime_.find(key);
    return it == lastDispatchTime_.end() ? cycle : it->second;
}

void IDU::openLoopBody(const std::string& key, int64_t startCycle)
{
    vloopStart_[key] = startCycle;
    bodyOpenTime_[key] = startCycle + vloopToDispatchDelay_;
}

std::optional<int64_t> IDU::nextLoop1Start(
    const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle) const
{
    if (!hasBlockEndLevel(inst, 1) || inst.iterStack.empty())
        return std::nullopt;
    const int64_t topBlockId = inst.topBlockId;
    const int64_t i = inst.iterStack[0];
    const std::string curKey = makeKey(topBlockId, "loop1", {i});
    const int64_t endCy = lastDispatchOrCycle(curKey, cycle);
    if (i + 1 >= bounds[0])
        return std::nullopt;
    const auto startIt = vloopStart_.find(curKey);
    const int64_t prevStart = startIt == vloopStart_.end() ? endCy : startIt->second;
    return std::max<int64_t>(endCy, prevStart + loop1MinFeedbackGap_);
}

void IDU::triggerDepth2Vloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle)
{
    const auto nextStart = nextLoop1Start(inst, bounds, cycle);
    if (!nextStart.has_value())
        return;
    const int64_t topBlockId = inst.topBlockId;
    const int64_t i = inst.iterStack[0];
    const std::string nextKey = makeKey(topBlockId, "loop1", {i + 1});
    openLoopBody(nextKey, *nextStart);
    vloopTrace_.push_back(VloopTraceRecord{topBlockId, "loop1", {i + 1}, *nextStart});
}

void IDU::triggerDepth3InnerVloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle)
{
    if (!hasBlockEndLevel(inst, 2) || inst.iterStack.size() < 2)
        return;
    const int64_t topBlockId = inst.topBlockId;
    const int64_t i = inst.iterStack[0];
    const int64_t j = inst.iterStack[1];
    const std::string curKey = makeKey(topBlockId, "loop2", {i, j});
    const int64_t endCy = lastDispatchOrCycle(curKey, cycle);
    if (j + 1 >= bounds[1])
        return;
    const std::string nextKey = makeKey(topBlockId, "loop2", {i, j + 1});
    openLoopBody(nextKey, endCy);
    vloopTrace_.push_back(VloopTraceRecord{topBlockId, "loop2", {i, j + 1}, endCy});
}

void IDU::triggerDepth3OuterVloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle)
{
    const auto nextStart = nextLoop1Start(inst, bounds, cycle);
    if (!nextStart.has_value())
        return;
    const int64_t topBlockId = inst.topBlockId;
    const int64_t i = inst.iterStack[0];
    openLoopBody(makeKey(topBlockId, "loop1", {i + 1}), *nextStart);
    vloopTrace_.push_back(VloopTraceRecord{topBlockId, "loop1", {i + 1}, *nextStart});
    if (bounds[1] <= 0)
        return;
    const int64_t childStart = *nextStart + nestedVloopInitialStartGap_;
    openLoopBody(makeKey(topBlockId, "loop2", {i + 1, 0}), childStart);
    vloopTrace_.push_back(VloopTraceRecord{topBlockId, "loop2", {i + 1, 0}, childStart});
}

void IDU::updateLastDispatch(const DynamicInst& inst, int64_t cycle)
{
    const int64_t topBlockId = inst.topBlockId;
    const auto& bk = inst.blockKeyByLevel;
    if (!bk.empty()) {
        for (const auto& raw : bk) {
            if (auto key = normalizeBlockKey(raw, topBlockId))
                lastDispatchTime_[*key] = cycle;
        }
        return;
    }

    const auto& iterStack = inst.iterStack;
    const int64_t depth = static_cast<int64_t>(inst.loopDepth);
    if (depth == 1) {
        lastDispatchTime_[makeKey(topBlockId, "loop0", {})] = cycle;
    } else if (depth == 2 && iterStack.size() >= 1) {
        lastDispatchTime_[makeKey(topBlockId, "loop1", {iterStack[0]})] = cycle;
    } else if (depth >= 3 && iterStack.size() >= 2) {
        lastDispatchTime_[makeKey(topBlockId, "loop2", {iterStack[0], iterStack[1]})] = cycle;
        lastDispatchTime_[makeKey(topBlockId, "loop1", {iterStack[0]})] = cycle;
        lastDispatchTime_[makeKey(topBlockId, "loop0", {})] = cycle;
    }
}

void IDU::triggerNextVloops(const DynamicInst& inst, int64_t cycle)
{
    const int64_t topBlockId = inst.topBlockId;
    const std::vector<int64_t>& bounds = loopBoundsForTopBlock(topBlockId);
    const int64_t depth = static_cast<int64_t>(bounds.size());

    triggerNextTopBlock(inst, cycle);

    if (depth <= 0 || inst.blockEndLevels.empty())
        return;

    if (depth == 1)
        return;
    if (depth == 2) {
        triggerDepth2Vloops(inst, bounds, cycle);
        return;
    }
    if (depth == 3) {
        triggerDepth3InnerVloops(inst, bounds, cycle);
        triggerDepth3OuterVloops(inst, bounds, cycle);
    }
}

IDU::DispatchResources IDU::makeDispatchResources(const IDUDispatchBudget& budget) const
{
    constexpr int64_t kUnlimited = 1LL << 60;
    if (budget.theoreticalLimitMode) {
        return DispatchResources{kUnlimited, kUnlimited, kUnlimited, kUnlimited, static_cast<int64_t>(window_.size())};
    }
    return DispatchResources{budget.freePreg, budget.freeShqQueue, budget.freeLsq, budget.freeShq, budget.issueBudget};
}

bool IDU::hasInitialDispatchCredit(const DispatchResources& resources) const
{
    if (resources.shqQueueFree <= 0 && resources.lsqFree <= 0)
        return false;
    return !globalShqPregGate_ || (resources.credits > 0 && resources.shqFree > 0);
}

bool IDU::isVloopDispatchOpen(const DynamicInst& inst, int64_t cycle, const IDUDispatchBudget& budget)
{
    const auto topOpenIt = topBlockBodyOpenTime_.find(inst.topBlockId);
    if (topOpenIt == topBlockBodyOpenTime_.end() || cycle < topOpenIt->second)
        return false;

    const auto innerKey = currentInnerBlockKey(inst);
    if (!budget.theoreticalLimitVloopOnly && innerKey) {
        const auto openIt = bodyOpenTime_.find(*innerKey);
        if (openIt == bodyOpenTime_.end() || cycle < openIt->second)
            return false;
    }
    if (budget.theoreticalLimitMode || budget.theoreticalLimitVloopOnly || !innerKey) {
        return true;
    }

    const int64_t iterId = inst.iterStack.empty() ? 0 : inst.iterStack.back();
    if (iterId == 0 && !blockBaseCycle_.count(*innerKey))
        blockBaseCycle_[*innerKey] = cycle;
    const auto baseIt = blockBaseCycle_.find(*innerKey);
    if (baseIt == blockBaseCycle_.end())
        return false;
    return cycle >= baseIt->second + iterId * innermostIterDispatchStride_;
}

bool IDU::hasQueueResource(const DynamicInst& inst, const std::string& form, const DispatchResources& resources) const
{
    if (isLoadOp(db_, inst.op, form))
        return resources.lsqFree > 0;
    if (isStoreOp(db_, inst.op, form))
        return resources.lsqFree > 0 && resources.shqFree > 0;
    return resources.shqQueueFree > 0 && resources.shqFree > 0;
}

int64_t IDU::countRegisterDst(const DynamicInst& inst) const
{
    int64_t dstCount = 0;
    for (const auto& dst : inst.dst) {
        if (analysis_.isVregName(dst))
            ++dstCount;
    }
    return dstCount;
}

void IDU::consumeDispatchResources(
    const DynamicInst& inst, const std::string& form, int64_t dstCount, DispatchResources& resources) const
{
    resources.credits -= dstCount;
    if (usesLsq(db_, inst.op, form)) {
        --resources.lsqFree;
        if (usesSharedShqCredit(db_, inst.op, form))
            --resources.shqFree;
    } else if (usesShqQueue(db_, inst.op, form)) {
        --resources.shqQueueFree;
        --resources.shqFree;
    }
}

void IDU::recordDispatch(const DynamicInst& inst, int64_t cycle, const DispatchResources& resources)
{
    window_.pop_front();
    dispatchLog_.push_back(IDUDispatchRecord{
        cycle,
        inst.instId,
        inst.op,
        inst.dst,
        inst.src,
        inst.topBlockId,
        resources.credits,
        resources.shqQueueFree,
        resources.lsqFree,
        resources.shqFree,
    });
    updateLastDispatch(inst, cycle);
    triggerNextVloops(inst, cycle);
}

std::vector<DynamicInst> IDU::dispatch(int64_t cycle, const IDUDispatchBudget& budget)
{
    if (window_.empty())
        return {};

    const int64_t dispatchStartGate = std::max<int64_t>(0, vfStartupCost_ - iduDispatchStartAdvance_);
    if (cycle < dispatchStartGate)
        return {};

    DispatchResources resources = makeDispatchResources(budget);

    std::vector<DynamicInst> dispatched;
    dispatched.reserve(static_cast<size_t>(std::max<int64_t>(0, resources.issueBudget)));

    if (!hasInitialDispatchCredit(resources))
        return {};

    for (const auto& inst : window_) {
        if (static_cast<int64_t>(dispatched.size()) >= resources.issueBudget)
            break;

        if (!isVloopDispatchOpen(inst, cycle, budget))
            break;

        const std::string& form = inst.form.empty() ? dtype_ : inst.form;
        if (!hasQueueResource(inst, form, resources))
            break;

        const int64_t dstCount = countRegisterDst(inst);
        if (resources.credits < dstCount)
            break;
        dispatched.push_back(inst);
        consumeDispatchResources(inst, form, dstCount, resources);
    }

    for (const auto& inst : dispatched) {
        recordDispatch(inst, cycle, resources);
    }

    return dispatched;
}

} // namespace vfsim

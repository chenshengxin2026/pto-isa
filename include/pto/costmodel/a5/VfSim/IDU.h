/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef VFSIM_NATIVE_IDU_H
#define VFSIM_NATIVE_IDU_H

#include "pto/costmodel/a5/VfSim/IFU.h"

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vfsim {

struct IDUDispatchBudget {
    int64_t freePreg = 0;
    int64_t freeShqQueue = 0;
    int64_t freeLsq = 0;
    int64_t freeShq = 0;
    int64_t issueBudget = 0;
    bool theoreticalLimitMode = false;
    bool theoreticalLimitVloopOnly = false;
};

struct IDUDispatchRecord {
    int64_t cycle = 0;
    int64_t instId = 0;
    std::string op;
    std::vector<std::string> dst;
    std::vector<std::string> src;
    int64_t topBlockId = 0;
    int64_t vreg = 0;
    int64_t shqQueue = 0;
    int64_t lsq = 0;
    int64_t shq = 0;
};

struct VloopTraceRecord {
    int64_t topBlockId = 0;
    std::string loopId;
    std::vector<int64_t> iter;
    int64_t startCycle = 0;
};

class IDU {
public:
    IDU(const UarchConfig& uarch, const ParamDB& db, ProgramAnalysis::ParamMap params = {},
        std::vector<int64_t> loopBounds = {}, int64_t totalTopBlocks = 1,
        std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds = {}, std::string dtype = "fp32",
        std::unordered_map<std::string, ValueInfo> values = {});

    bool empty() const noexcept { return window_.empty(); }
    bool canAccept() const;
    void accept(const DynamicInst& inst);

    std::vector<DynamicInst> dispatch(int64_t cycle, const IDUDispatchBudget& budget);

    const ParamDB& db() const noexcept { return db_; }
    const std::vector<IDUDispatchRecord>& dispatchLog() const noexcept { return dispatchLog_; }
    const std::vector<VloopTraceRecord>& vloopTrace() const noexcept { return vloopTrace_; }

private:
    struct DispatchResources {
        int64_t credits = 0;
        int64_t shqQueueFree = 0;
        int64_t lsqFree = 0;
        int64_t shqFree = 0;
        int64_t issueBudget = 0;
    };

    const ParamDB& db_;
    std::string dtype_;
    int64_t windowWidth_ = 0;
    int64_t issueWidth_ = 0;
    bool theoreticalLimitMode_ = false;
    bool theoreticalLimitVloopOnly_ = false;
    int64_t vfStartupCost_ = 0;
    int64_t iduDispatchStartAdvance_ = 0;
    int64_t vloopToDispatchDelay_ = 4;
    int64_t initialTopBlockVloopStartCycle_ = 19;
    int64_t nestedVloopInitialStartGap_ = 1;
    int64_t loop1MinFeedbackGap_ = 7;
    int64_t innermostIterDispatchStride_ = 1;
    bool globalShqPregGate_ = false;

    std::deque<DynamicInst> window_;
    ProgramAnalysis analysis_;
    std::vector<int64_t> loopBounds_;
    int64_t totalTopBlocks_ = 1;
    std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds_;

    std::unordered_map<int64_t, int64_t> topBlockVloopStart_;
    std::unordered_map<int64_t, int64_t> topBlockBodyOpenTime_;
    std::unordered_map<std::string, int64_t> vloopStart_;
    std::unordered_map<std::string, int64_t> bodyOpenTime_;
    std::unordered_map<std::string, int64_t> lastDispatchTime_;
    std::unordered_map<std::string, int64_t> blockBaseCycle_;
    std::vector<VloopTraceRecord> vloopTrace_;
    std::vector<IDUDispatchRecord> dispatchLog_;

    void initVloopStarts();
    void setTopBlockVloop(int64_t topBlockId, int64_t startCycle);
    void initTopBlockNestedStarts(int64_t topBlockId, int64_t topVloopStart);
    const std::vector<int64_t>& loopBoundsForTopBlock(int64_t topBlockId) const;
    bool hasBlockEndLevel(const DynamicInst& inst, int64_t level) const;
    void triggerNextTopBlock(const DynamicInst& inst, int64_t cycle);
    int64_t lastDispatchOrCycle(const std::string& key, int64_t cycle) const;
    void openLoopBody(const std::string& key, int64_t startCycle);
    std::optional<int64_t> nextLoop1Start(
        const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle) const;
    void triggerDepth2Vloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle);
    void triggerDepth3InnerVloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle);
    void triggerDepth3OuterVloops(const DynamicInst& inst, const std::vector<int64_t>& bounds, int64_t cycle);

    std::string makeKey(int64_t topBlockId, const std::string& loopId, const std::vector<int64_t>& iters) const;
    std::optional<std::string> normalizeBlockKey(
        const std::pair<std::string, std::vector<int64_t>>& raw, int64_t topBlockId) const;
    std::optional<std::string> currentInnerBlockKey(const DynamicInst& inst) const;

    void updateLastDispatch(const DynamicInst& inst, int64_t cycle);
    void triggerNextVloops(const DynamicInst& inst, int64_t cycle);
    DispatchResources makeDispatchResources(const IDUDispatchBudget& budget) const;
    bool hasInitialDispatchCredit(const DispatchResources& resources) const;
    bool isVloopDispatchOpen(const DynamicInst& inst, int64_t cycle, const IDUDispatchBudget& budget);
    bool hasQueueResource(const DynamicInst& inst, const std::string& form, const DispatchResources& resources) const;
    int64_t countRegisterDst(const DynamicInst& inst) const;
    void consumeDispatchResources(
        const DynamicInst& inst, const std::string& form, int64_t dstCount, DispatchResources& resources) const;
    void recordDispatch(const DynamicInst& inst, int64_t cycle, const DispatchResources& resources);
    bool isLastInstOfTopBlock(const DynamicInst& inst) const;
};

} // namespace vfsim

#endif // VFSIM_NATIVE_IDU_H

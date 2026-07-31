/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/ProgramVregLiveRangeNormalization.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace vfsim {
namespace {

struct VregVersion {
    std::string name;
    int64_t generation = 0;
};

struct FlatLoopVregState {
    std::unordered_map<std::string, VregVersion> currentVersionByVreg;
    std::unordered_map<std::string, int64_t> versionCounter;
    std::vector<std::vector<std::optional<std::string>>> srcVersions;
    std::vector<std::vector<std::optional<std::string>>> dstVersions;
    std::unordered_map<std::string, int64_t> lastUse;
    std::unordered_map<std::string, std::string> currentSlotByVreg;
    std::unordered_map<std::string, std::string> slotOfVersion;
    std::unordered_map<std::string, std::optional<std::string>> slotOccupant;
    std::vector<std::string> slotPool;
};

std::string makeVersionKey(const VregVersion& version)
{
    return version.name + "#" + std::to_string(version.generation);
}

std::pair<int64_t, std::string> vregSortKey(const std::string& name)
{
    if (name.size() <= 1)
        return {std::numeric_limits<int64_t>::max(), name};
    int64_t value = 0;
    for (size_t i = 1; i < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i])))
            return {std::numeric_limits<int64_t>::max(), name};
        value = value * 10 + static_cast<int64_t>(name[i] - '0');
    }
    return {value, name};
}

bool containsSlot(const std::vector<std::string>& slots, const std::string& slot)
{
    return std::find(slots.begin(), slots.end(), slot) != slots.end();
}

std::string nextFreshVreg(const std::vector<std::string>& slotPool)
{
    std::unordered_set<std::string> used(slotPool.begin(), slotPool.end());
    int64_t maxIndex = -1;
    for (const std::string& name : slotPool) {
        auto key = vregSortKey(name);
        if (key.first != std::numeric_limits<int64_t>::max())
            maxIndex = std::max(maxIndex, key.first);
    }

    for (int64_t candidate = maxIndex + 1;; ++candidate) {
        std::string name = "v" + std::to_string(candidate);
        if (used.find(name) == used.end())
            return name;
    }
}

void countFieldChanges(
    const ProgramInstNode& before, const ProgramInstNode& after, ProgramVregLiveRangeNormalizationStats& stats)
{
    const size_t srcCount = std::min(before.src.size(), after.src.size());
    for (size_t i = 0; i < srcCount; ++i) {
        if (before.src[i] != after.src[i])
            ++stats.changedFields;
    }
    const size_t dstCount = std::min(before.dst.size(), after.dst.size());
    for (size_t i = 0; i < dstCount; ++i) {
        if (before.dst[i] != after.dst[i])
            ++stats.changedFields;
    }
}

void ensureRegisterValue(
    std::unordered_map<std::string, ValueInfo>* values, const std::string& slot, const std::string& sourceValueId)
{
    if (values == nullptr || values->find(slot) != values->end())
        return;

    ValueInfo value;
    auto sourceIt = values->find(sourceValueId);
    if (sourceIt != values->end())
        value = sourceIt->second;
    value.valueId = slot;
    value.storage = ValueStorageKind::Register;
    values->emplace(slot, std::move(value));
}

void collectFlatLoopVersions(
    const std::vector<ProgramNode>& out, const ProgramAnalysis& analysis, FlatLoopVregState& state)
{
    state.srcVersions.assign(out.size(), {});
    state.dstVersions.assign(out.size(), {});
    for (size_t idx = 0; idx < out.size(); ++idx) {
        const ProgramInstNode& inst = out[idx].inst;
        state.srcVersions[idx].reserve(inst.src.size());
        for (const std::string& src : inst.src) {
            if (!analysis.isVregName(src)) {
                state.srcVersions[idx].push_back(std::nullopt);
                continue;
            }
            auto it = state.currentVersionByVreg.find(src);
            if (it == state.currentVersionByVreg.end()) {
                state.srcVersions[idx].push_back(std::nullopt);
                continue;
            }
            std::string key = makeVersionKey(it->second);
            state.srcVersions[idx].push_back(key);
            state.lastUse[key] = static_cast<int64_t>(idx);
        }

        state.dstVersions[idx].reserve(inst.dst.size());
        for (const std::string& dst : inst.dst) {
            if (!analysis.isVregName(dst)) {
                state.dstVersions[idx].push_back(std::nullopt);
                continue;
            }
            int64_t& generation = state.versionCounter[dst];
            ++generation;
            VregVersion version{dst, generation};
            state.currentVersionByVreg[dst] = version;
            state.dstVersions[idx].push_back(makeVersionKey(version));
        }
    }
}

std::string versionBaseName(const std::optional<std::string>& version, const std::string& fallback)
{
    if (!version)
        return fallback;
    const size_t hash = version->find('#');
    return hash == std::string::npos ? fallback : version->substr(0, hash);
}

std::string resolveSrcSlot(
    const std::string& src, const std::optional<std::string>& version, const FlatLoopVregState& state)
{
    if (version) {
        auto slotIt = state.slotOfVersion.find(*version);
        if (slotIt != state.slotOfVersion.end())
            return slotIt->second;
        const std::string versionName = versionBaseName(version, src);
        auto curIt = state.currentSlotByVreg.find(versionName);
        return curIt == state.currentSlotByVreg.end() ? versionName : curIt->second;
    }
    auto curIt = state.currentSlotByVreg.find(src);
    return curIt == state.currentSlotByVreg.end() ? src : curIt->second;
}

std::vector<std::string> rewriteSrcs(
    const ProgramInstNode& inst, size_t idx, const ProgramAnalysis& analysis, const FlatLoopVregState& state)
{
    std::vector<std::string> newSrcs = inst.src;
    for (size_t pos = 0; pos < inst.src.size(); ++pos) {
        if (!analysis.isVregName(inst.src[pos]))
            continue;
        const std::optional<std::string>& version =
            pos < state.srcVersions[idx].size() ? state.srcVersions[idx][pos] : std::nullopt;
        newSrcs[pos] = resolveSrcSlot(inst.src[pos], version, state);
    }
    return newSrcs;
}

std::vector<std::string> reusableSlots(const FlatLoopVregState& state, size_t idx)
{
    std::vector<std::string> candidates;
    for (const std::string& slot : state.slotPool) {
        auto occIt = state.slotOccupant.find(slot);
        bool reusable = occIt == state.slotOccupant.end() || !occIt->second;
        if (!reusable) {
            auto lastIt = state.lastUse.find(*occIt->second);
            int64_t last = lastIt == state.lastUse.end() ? -1 : lastIt->second;
            reusable = last < static_cast<int64_t>(idx);
        }
        if (reusable)
            candidates.push_back(slot);
    }
    return candidates;
}

void addLastUseSrcSlots(
    std::vector<std::string>& candidates, const std::vector<std::string>& newSrcs, const FlatLoopVregState& state,
    size_t idx)
{
    for (size_t pos = 0; pos < state.srcVersions[idx].size(); ++pos) {
        const std::optional<std::string>& version = state.srcVersions[idx][pos];
        if (!version)
            continue;
        auto lastIt = state.lastUse.find(*version);
        if (lastIt == state.lastUse.end() || lastIt->second != static_cast<int64_t>(idx))
            continue;
        if (pos < newSrcs.size() && !containsSlot(candidates, newSrcs[pos]))
            candidates.push_back(newSrcs[pos]);
    }
}

std::string chooseDstSlot(
    const std::vector<std::string>& newSrcs, std::vector<std::string> candidates, const std::string& dstName,
    const std::vector<std::string>& slotPool)
{
    if (newSrcs.size() == 1 && containsSlot(candidates, newSrcs[0]))
        return newSrcs[0];
    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(), [](const std::string& lhs, const std::string& rhs) {
            return vregSortKey(lhs) < vregSortKey(rhs);
        });
        return candidates.front();
    }
    if (!containsSlot(slotPool, dstName))
        return dstName;
    return nextFreshVreg(slotPool);
}

std::vector<std::string> rewriteDsts(
    const ProgramInstNode& inst, const std::vector<std::string>& newSrcs, size_t idx, const ProgramAnalysis& analysis,
    FlatLoopVregState& state, std::unordered_map<std::string, ValueInfo>* values)
{
    std::vector<std::string> newDsts = inst.dst;
    if (inst.dst.size() != 1 || !analysis.isVregName(inst.dst.front()))
        return newDsts;
    const std::optional<std::string>& dstVersion =
        state.dstVersions[idx].empty() ? std::nullopt : state.dstVersions[idx].front();
    if (!dstVersion)
        return newDsts;

    const std::string& dstName = inst.dst.front();
    std::vector<std::string> candidates = reusableSlots(state, idx);
    addLastUseSrcSlots(candidates, newSrcs, state, idx);
    const std::string chosenSlot = chooseDstSlot(newSrcs, std::move(candidates), dstName, state.slotPool);
    if (!containsSlot(state.slotPool, chosenSlot))
        state.slotPool.push_back(chosenSlot);
    ensureRegisterValue(values, chosenSlot, dstName);
    state.slotOfVersion[*dstVersion] = chosenSlot;
    state.currentSlotByVreg[dstName] = chosenSlot;
    state.slotOccupant[chosenSlot] = *dstVersion;
    newDsts[0] = chosenSlot;
    return newDsts;
}

void rewriteFlatLoopInst(
    ProgramInstNode& inst, size_t idx, const ProgramAnalysis& analysis, FlatLoopVregState& state,
    ProgramVregLiveRangeNormalizationStats& stats, std::unordered_map<std::string, ValueInfo>* values)
{
    ProgramInstNode before = inst;
    std::vector<std::string> newSrcs = rewriteSrcs(inst, idx, analysis, state);
    std::vector<std::string> newDsts = rewriteDsts(inst, newSrcs, idx, analysis, state, values);
    inst.src = std::move(newSrcs);
    inst.dst = std::move(newDsts);
    countFieldChanges(before, inst, stats);
}

std::vector<ProgramNode> normalizeFlatLoopVregs(
    const std::vector<ProgramNode>& body, const ProgramAnalysis& analysis,
    ProgramVregLiveRangeNormalizationStats& stats, std::unordered_map<std::string, ValueInfo>* values)
{
    std::vector<ProgramNode> out = body;
    FlatLoopVregState state;
    collectFlatLoopVersions(out, analysis, state);
    for (size_t idx = 0; idx < out.size(); ++idx) {
        if (out[idx].kind != ProgramNode::Kind::Inst)
            continue;
        rewriteFlatLoopInst(out[idx].inst, idx, analysis, state, stats, values);
    }
    return out;
}

std::vector<ProgramNode> normalizeNodes(
    const std::vector<ProgramNode>& program, const ProgramAnalysis& analysis,
    ProgramVregLiveRangeNormalizationStats& stats, std::unordered_map<std::string, ValueInfo>* values)
{
    std::vector<ProgramNode> out;
    out.reserve(program.size());
    for (const ProgramNode& node : program) {
        if (node.kind != ProgramNode::Kind::Loop || !node.loop) {
            out.push_back(node);
            continue;
        }

        const bool flatInstBody = std::all_of(
            node.loop->body.begin(), node.loop->body.end(),
            [](const ProgramNode& op) { return op.kind == ProgramNode::Kind::Inst; });
        ProgramLoopNode rewritten = *node.loop;
        if (flatInstBody) {
            rewritten.body = normalizeFlatLoopVregs(rewritten.body, analysis, stats, values);
        } else {
            rewritten.body = normalizeNodes(rewritten.body, analysis, stats, values);
        }
        out.push_back(ProgramNode::makeLoop(std::move(rewritten)));
    }
    return out;
}

} // namespace

std::vector<ProgramNode> normalizeProgramVregLiveRanges(
    const std::vector<ProgramNode>& program, const ProgramAnalysis::ParamMap& params,
    const std::unordered_map<std::string, ValueInfo>& values, ProgramVregLiveRangeNormalizationStats* stats)
{
    ProgramVregLiveRangeNormalizationStats localStats;
    ProgramAnalysis analysis(params, values);
    auto normalized = normalizeNodes(program, analysis, localStats, nullptr);
    if (stats != nullptr)
        *stats = localStats;
    return normalized;
}

void normalizeProgramVregLiveRanges(VfInfo& vfInfo, ProgramVregLiveRangeNormalizationStats* stats)
{
    ProgramVregLiveRangeNormalizationStats localStats;
    ProgramAnalysis analysis(vfInfo.params, vfInfo.values);
    vfInfo.body = normalizeNodes(vfInfo.body, analysis, localStats, &vfInfo.values);
    if (stats != nullptr)
        *stats = localStats;
}

} // namespace vfsim

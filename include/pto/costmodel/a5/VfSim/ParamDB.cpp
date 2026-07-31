/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/ParamDB.h"

#include "pto/costmodel/a5/VfSim/VfSimParamsGenerated.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vfsim {
namespace {

struct IntUarchParam {
    std::string_view key;
    int64_t UarchConfig::*member;
};

struct BoolUarchParam {
    std::string_view key;
    bool UarchConfig::*member;
};

inline constexpr std::array<IntUarchParam, 27> kIntUarchParams = {{
    {"issue_ports", &UarchConfig::issuePorts},
    {"load_ports", &UarchConfig::loadPorts},
    {"store_ports", &UarchConfig::storePorts},
    {"IDU_window_width", &UarchConfig::iduWindowWidth},
    {"IDU_issue_width", &UarchConfig::iduIssueWidth},
    {"LDQ_width", &UarchConfig::ldqWidth},
    {"vreg_num", &UarchConfig::vregNum},
    {"shq_depth", &UarchConfig::shqDepth},
    {"exq_depth", &UarchConfig::exqDepth},
    {"shq_release_delay", &UarchConfig::shqReleaseDelay},
    {"idu_visible_preg_delay", &UarchConfig::iduVisiblePregDelay},
    {"idu_visible_shq_delay", &UarchConfig::iduVisibleShqDelay},
    {"idu_to_ooo_delay", &UarchConfig::iduToOooDelay},
    {"vloop_to_dispatch_delay", &UarchConfig::vloopToDispatchDelay},
    {"idu_dispatch_start_advance", &UarchConfig::iduDispatchStartAdvance},
    {"initial_top_block_vloop_start_cycle", &UarchConfig::initialTopBlockVloopStartCycle},
    {"nested_vloop_initial_start_gap", &UarchConfig::nestedVloopInitialStartGap},
    {"loop1_min_feedback_gap", &UarchConfig::loop1MinFeedbackGap},
    {"innermost_iter_dispatch_stride", &UarchConfig::innermostIterDispatchStride},
    {"consumer_release_start_offset", &UarchConfig::consumerReleaseStartOffset},
    {"load_done_latency", &UarchConfig::loadDoneLatency},
    {"ooo_to_shq_delay", &UarchConfig::oooToShqDelay},
    {"ooo_to_lsq_delay", &UarchConfig::oooToLsqDelay},
    {"exq_recv_delay", &UarchConfig::exqRecvDelay},
    {"shq_to_exq_port_per_cycle", &UarchConfig::shqToExqPortPerCycle},
    {"compute_inflight_cap", &UarchConfig::computeInflightCap},
    {"exq_issue_inflight_cap_per_port", &UarchConfig::exqIssueInflightCapPerPort},
}};

inline constexpr std::array<BoolUarchParam, 9> kBoolUarchParams = {{
    {"enable_isu_queue_model", &UarchConfig::enableIsuQueueModel},
    {"admit_blocked_to_exq", &UarchConfig::admitBlockedToExq},
    {"enable_shq_credit_model", &UarchConfig::enableShqCreditModel},
    {"enable_credit_visibility_delay", &UarchConfig::enableCreditVisibilityDelay},
    {"global_shq_preg_gate", &UarchConfig::globalShqPregGate},
    {"use_explicit_idu_credit_bank", &UarchConfig::useExplicitIduCreditBank},
    {"exq_capacity_counts_inflight", &UarchConfig::exqCapacityCountsInflight},
    {"enforce_same_cycle_src_hazard", &UarchConfig::enforceSameCycleSrcHazard},
    {"enable_cross_fu_ii", &UarchConfig::enableCrossFuIi},
}};

std::string toString(std::string_view value) { return std::string(value.data(), value.size()); }

int64_t parseInt(std::string_view value) { return std::stoll(toString(value)); }

bool parseBool(std::string_view value) { return value == "true" || value == "1"; }

std::pair<std::string, std::string> splitQualifiedOp(const std::string& name)
{
    const std::size_t dot = name.rfind('.');
    if (dot == std::string::npos)
        return {name, ""};
    return {name.substr(0, dot), name.substr(dot + 1)};
}

std::string qualifyOp(const std::string& op, const std::string& form) { return form.empty() ? op : op + "." + form; }

void applyUarchParam(UarchConfig& uarch, std::string_view key, std::string_view type, std::string_view value)
{
    (void)type;
    const std::string keyText = toString(key);
    const auto intIt = std::find_if(
        kIntUarchParams.begin(), kIntUarchParams.end(), [&](const auto& param) { return param.key == keyText; });
    if (intIt != kIntUarchParams.end()) {
        uarch.*(intIt->member) = parseInt(value);
        return;
    }

    const auto boolIt = std::find_if(
        kBoolUarchParams.begin(), kBoolUarchParams.end(), [&](const auto& param) { return param.key == keyText; });
    if (boolIt != kBoolUarchParams.end()) {
        uarch.*(boolIt->member) = parseBool(value);
        return;
    }

    if (keyText == "mem_bar_mode")
        uarch.memBarMode = toString(value);
}

void addRelation(
    std::unordered_map<DTypeName, std::unordered_map<OpName, std::unordered_map<OpName, int64_t>>>& byDtype,
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>& byForm, const std::string& lhs,
    const std::string& rhs, int64_t cycles)
{
    const auto [lhsOp, lhsForm] = splitQualifiedOp(lhs);
    const auto [rhsOp, rhsForm] = splitQualifiedOp(rhs);
    if (!lhsForm.empty()) {
        byForm[lhs][rhs] = cycles;
        if (rhsForm.empty() || rhsForm == lhsForm)
            byDtype[lhsForm][lhsOp][rhsOp] = cycles;
        return;
    }
    byDtype[lhs][rhsOp][rhsForm] = cycles;
}

} // namespace

ParamDB::ParamDB(std::filesystem::path baseDir) : baseDir_(resolveBaseDir(std::move(baseDir)))
{
    for (const auto& param : generated::kIsaDefaultParams) {
        const std::string key = toString(param.key);
        if (key == "vf_startup_cost")
            bundle_.isaDefaults.vfStartupCost = param.value;
        else if (key == "vf_drain_cost")
            bundle_.isaDefaults.vfDrainCost = param.value;
    }

    for (const auto& param : generated::kIsaInstParams) {
        InstConfig config;
        config.pipelineStartupCost = param.pipelineStartupCost;
        config.latency = param.latency;
        config.throughput = param.throughput;
        config.pipelineDrainCost = param.pipelineDrainCost;
        config.dataLoadCost = param.dataLoadCost;
        config.dataStoreCost = param.dataStoreCost;
        config.exu = toString(param.exu);
        config.dispatchExu = toString(param.dispatchExu);
        config.opClass = toString(param.opClass);
        bundle_.isa[toString(param.op)][toString(param.form)] = std::move(config);
    }

    for (const auto& param : generated::kUarchParams)
        applyUarchParam(bundle_.uarch, param.key, param.type, param.value);

    for (const auto& param : generated::kForwardingParams)
        addRelation(
            bundle_.forwarding, bundle_.forwardingByForm, toString(param.lhs), toString(param.rhs), param.cycles);

    for (const auto& param : generated::kInitiationIntervalParams)
        addRelation(
            bundle_.initiationInterval, bundle_.initiationIntervalByForm, toString(param.lhs), toString(param.rhs),
            param.cycles);
}

std::filesystem::path ParamDB::resolveBaseDir(std::filesystem::path baseDir)
{
    if (baseDir.empty())
        return std::filesystem::absolute(std::filesystem::current_path());
    return std::filesystem::absolute(std::move(baseDir));
}

bool ParamDB::hasInst(const std::string& op, const std::string& dtype) const
{
    const auto opIt = bundle_.isa.find(op);
    if (opIt == bundle_.isa.end())
        return false;
    return opIt->second.find(dtype) != opIt->second.end();
}

const InstConfig& ParamDB::inst(const std::string& op, const std::string& dtype) const
{
    const auto opIt = bundle_.isa.find(op);
    if (opIt == bundle_.isa.end())
        throw std::runtime_error("Instruction not found: op=" + op + ", dtype=" + dtype);
    const auto dtypeIt = opIt->second.find(dtype);
    if (dtypeIt == opIt->second.end())
        throw std::runtime_error("Instruction not found: op=" + op + ", dtype=" + dtype);
    return dtypeIt->second;
}

int64_t ParamDB::forwardingCycles(const std::string& dtype, const std::string& prod, const std::string& cons) const
{
    const auto dtypeIt = bundle_.forwarding.find(dtype);
    if (dtypeIt != bundle_.forwarding.end()) {
        const auto prodIt = dtypeIt->second.find(prod);
        if (prodIt != dtypeIt->second.end()) {
            const auto consIt = prodIt->second.find(cons);
            if (consIt != prodIt->second.end())
                return std::max<int64_t>(0, consIt->second);
        }
    }
    const int64_t latency = hasInst(prod, dtype) ? inst(prod, dtype).latency : 0;
    return std::max<int64_t>(0, latency - 3);
}

int64_t ParamDB::forwardingCycles(
    const std::string& prod, const std::string& prodForm, const std::string& cons, const std::string& consForm) const
{
    const auto prodIt = bundle_.forwardingByForm.find(qualifyOp(prod, prodForm));
    if (prodIt != bundle_.forwardingByForm.end()) {
        const auto consIt = prodIt->second.find(qualifyOp(cons, consForm));
        if (consIt != prodIt->second.end())
            return std::max<int64_t>(0, consIt->second);
    }
    if (prodForm == consForm)
        return forwardingCycles(prodForm, prod, cons);
    const int64_t latency = hasInst(prod, prodForm) ? inst(prod, prodForm).latency : 0;
    return std::max<int64_t>(0, latency - 3);
}

int64_t ParamDB::initiationInterval(const std::string& dtype, const std::string& prev, const std::string& cur) const
{
    const auto dtypeIt = bundle_.initiationInterval.find(dtype);
    if (dtypeIt != bundle_.initiationInterval.end()) {
        const auto prevIt = dtypeIt->second.find(prev);
        if (prevIt != dtypeIt->second.end()) {
            const auto curIt = prevIt->second.find(cur);
            if (curIt != prevIt->second.end())
                return std::max<int64_t>(1, curIt->second);
        }
    }
    return 1;
}

int64_t ParamDB::initiationInterval(
    const std::string& prev, const std::string& prevForm, const std::string& cur, const std::string& curForm) const
{
    const auto prevIt = bundle_.initiationIntervalByForm.find(qualifyOp(prev, prevForm));
    if (prevIt != bundle_.initiationIntervalByForm.end()) {
        const auto curIt = prevIt->second.find(qualifyOp(cur, curForm));
        if (curIt != prevIt->second.end())
            return std::max<int64_t>(1, curIt->second);
    }
    if (prevForm == curForm)
        return initiationInterval(prevForm, prev, cur);
    return 1;
}

} // namespace vfsim

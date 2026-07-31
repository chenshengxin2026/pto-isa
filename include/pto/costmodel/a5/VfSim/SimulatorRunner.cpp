/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/SimulatorRunner.h"

#include "pto/costmodel/a5/VfSim/ISATraits.h"
#include "pto/costmodel/a5/VfSim/JsonDumpUtils.h"
#include "pto/costmodel/a5/VfSim/ProgramCanonicalization.h"
#include "pto/costmodel/a5/VfSim/ProgramFlatten.h"
#include "pto/costmodel/a5/VfSim/ProgramVregLiveRangeNormalization.h"
#include "pto/costmodel/a5/VfSim/ValueStorage.h"

#include <deque>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace vfsim {

namespace {

template <typename T>
void dumpJsonLines(const std::vector<T>& records, const std::string& path)
{
    std::ofstream os(path);
    for (const auto& r : records) {
        os << r << '\n';
    }
}

struct Reservation {
    int64_t preg = 0;
    int64_t shqQueue = 0;
    int64_t lsq = 0;
    int64_t shq = 0;
};

using IduToOooPipe = std::deque<std::pair<int64_t, DynamicInst>>;

struct SimulationState {
    IduToOooPipe iduToOooPipe;
    int64_t iduPregCredit = 0;
    int64_t iduShqCredit = 0;
    int64_t iduPendingShqQueue = 0;
    int64_t cycle = 0;
    bool completed = false;
};

Reservation reservationForInst(
    const DynamicInst& inst, const ParamDB& db, const std::string& defaultDtype, const ValueStorageLookup& valueStorage)
{
    const std::string& form = inst.form.empty() ? defaultDtype : inst.form;
    Reservation out;
    for (const auto& d : inst.dst) {
        if (valueStorage.isRegister(d))
            ++out.preg;
    }
    if (usesShqQueue(db, inst.op, form))
        out.shqQueue = 1;
    if (usesLsq(db, inst.op, form))
        out.lsq = 1;
    if (usesSharedShqCredit(db, inst.op, form))
        out.shq = 1;
    return out;
}

int64_t applyMaxCyclesEnv(int64_t maxCycles)
{
    const char* envMax = std::getenv("PTOAS_VFSIM_MAX_CYCLES");
    if (envMax == nullptr)
        return maxCycles;
    try {
        const int64_t parsed = std::stoll(envMax);
        return parsed > 0 ? parsed : maxCycles;
    } catch (...) {
        return maxCycles;
    }
}

void logCycleBegin(int64_t cycle, const IFU& ifu, const IDU& idu, const OoOCoreMainline& ooo)
{
    std::cerr << "[vfsim] cycle " << cycle << " begin"
              << " ifu_done=" << (ifu.done() ? 1 : 0) << " idu_empty=" << (idu.empty() ? 1 : 0)
              << " rob=" << ooo.getRobSize() << " lsq=" << ooo.getLsqSize() << " shq=" << ooo.getShqSize() << "\n";
}

void updateExplicitIduCredits(
    OoOCoreMainline& ooo, int64_t cycle, bool enabled, int64_t& pregCredit, int64_t& shqCredit)
{
    auto visibleDelta = ooo.updateIduVisibility(cycle);
    if (!enabled)
        return;
    pregCredit += visibleDelta["preg_free"];
    shqCredit += visibleDelta["shq_release"];
}

void drainIduToOooPipe(
    IduToOooPipe& pipe, OoOCoreMainline& ooo, int64_t cycle, bool useExplicitIduCreditBank,
    const ValueStorageLookup& valueStorage, int64_t& iduPendingShqQueue)
{
    while (!pipe.empty() && pipe.front().first <= cycle) {
        auto item = std::move(pipe.front());
        pipe.pop_front();
        if (useExplicitIduCreditBank) {
            for (const auto& dst : item.second.dst) {
                if (valueStorage.isRegister(dst))
                    --iduPendingShqQueue;
            }
        }
        ooo.accept(item.second);
    }
}

Reservation pendingPipeReservations(const IduToOooPipe& pipe, const IDU& idu, const ValueStorageLookup& valueStorage)
{
    Reservation pending;
    for (const auto& item : pipe) {
        const auto r = reservationForInst(item.second, idu.db(), "fp32", valueStorage);
        pending.preg += r.preg;
        pending.shqQueue += r.shqQueue;
        pending.lsq += r.lsq;
        pending.shq += r.shq;
    }
    return pending;
}

void fillIdu(IFU& ifu, IDU& idu)
{
    while (idu.canAccept()) {
        if (ifu.done())
            break;
        auto inst = ifu.nextInst();
        if (!inst.has_value())
            break;
        idu.accept(*inst);
    }
}

IDUDispatchBudget makeDispatchBudget(
    const OoOCoreMainline& ooo, const Reservation& pending, bool useExplicitIduCreditBank, int64_t iduPregCredit,
    int64_t iduShqCredit)
{
    IDUDispatchBudget budget;
    budget.theoreticalLimitMode = false;
    budget.theoreticalLimitVloopOnly = false;
    budget.freePreg = useExplicitIduCreditBank ? iduPregCredit : std::max<int64_t>(0, ooo.getFreePreg() - pending.preg);
    budget.freeShqQueue = std::max<int64_t>(0, ooo.getFreeShqQueue() - pending.shqQueue);
    budget.freeLsq = std::max<int64_t>(0, ooo.getFreeLsq() - pending.lsq);
    budget.freeShq = useExplicitIduCreditBank ? iduShqCredit : std::max<int64_t>(0, ooo.getFreeShq() - pending.shq);
    budget.issueBudget = 5;
    return budget;
}

void logDispatchBegin(int64_t cycle, const IDUDispatchBudget& budget)
{
    std::cerr << "[vfsim] cycle " << cycle << " dispatch begin"
              << " freePreg=" << budget.freePreg << " freeShqQueue=" << budget.freeShqQueue
              << " freeLsq=" << budget.freeLsq << " freeShq=" << budget.freeShq << "\n";
}

void forwardDispatchedToOoo(
    const std::vector<DynamicInst>& dispatched, IduToOooPipe& pipe, OoOCoreMainline& ooo, int64_t cycle,
    int64_t iduToOooDelay)
{
    for (const auto& inst : dispatched) {
        if (iduToOooDelay > 0)
            pipe.emplace_back(cycle + iduToOooDelay, inst);
        else
            ooo.accept(inst);
    }
}

bool isSimulationComplete(const IFU& ifu, const IDU& idu, const OoOCoreMainline& ooo, const IduToOooPipe& pipe)
{
    return ifu.done() && idu.empty() && ooo.getRobSize() == 0 && ooo.getLsqSize() == 0 && ooo.getShqSize() == 0 &&
           pipe.empty();
}

void dumpDispatchLog(const IDU& idu, const std::string& path)
{
    std::ofstream os(path);
    for (const auto& r : idu.dispatchLog()) {
        os << "{\"cy\":" << r.cycle << ",\"inst_id\":" << r.instId << ",\"op\":\"" << jsonEscape(r.op) << "\""
           << ",\"dst\":" << joinJsonArray(r.dst) << ",\"src\":" << joinJsonArray(r.src)
           << ",\"top_block_id\":" << r.topBlockId << ",\"vreg\":" << r.vreg << ",\"SHQ_QUEUE\":" << r.shqQueue
           << ",\"LSQ\":" << r.lsq << ",\"SHQ\":" << r.shq << "}\n";
    }
}

void dumpVloopTrace(const IDU& idu, const std::string& path)
{
    std::ofstream os(path);
    for (const auto& r : idu.vloopTrace()) {
        os << "{\"top_block_id\":" << r.topBlockId << ",\"loop_id\":\"" << jsonEscape(r.loopId) << "\""
           << ",\"iter\":" << joinJsonArray(r.iter) << ",\"start_cycle\":" << r.startCycle << "}\n";
    }
}

void dumpSimulationLogs(const std::string& resultsDir, const IDU& idu, const OoOCoreMainline& ooo)
{
    if (resultsDir.empty())
        return;
    std::filesystem::create_directories(resultsDir);
    ooo.dumpHistory(resultsDir + "/sim_history.json");
    ooo.dumpSimpleLogs(resultsDir + "/start_by_cycle.json", resultsDir + "/done_by_cycle.json");
    dumpDispatchLog(idu, resultsDir + "/idu_to_ooo.json");
    dumpVloopTrace(idu, resultsDir + "/vloop_trace.json");
}

bool runOneSimulationCycle(
    IFU& ifu, IDU& idu, OoOCoreMainline& ooo, SimulationState& state, bool useExplicitIduCreditBank,
    int64_t iduToOooDelay, const ValueStorageLookup& valueStorage, bool debugCycles)
{
    const int64_t cycle = state.cycle;
    if (debugCycles)
        logCycleBegin(cycle, ifu, idu, ooo);
    updateExplicitIduCredits(ooo, cycle, useExplicitIduCreditBank, state.iduPregCredit, state.iduShqCredit);
    drainIduToOooPipe(state.iduToOooPipe, ooo, cycle, useExplicitIduCreditBank, valueStorage, state.iduPendingShqQueue);

    if (debugCycles)
        std::cerr << "[vfsim] cycle " << cycle << " fill_idu begin\n";
    Reservation pending;
    if (!useExplicitIduCreditBank)
        pending = pendingPipeReservations(state.iduToOooPipe, idu, valueStorage);
    fillIdu(ifu, idu);
    if (debugCycles)
        std::cerr << "[vfsim] cycle " << cycle << " fill_idu end\n";

    const IDUDispatchBudget budget =
        makeDispatchBudget(ooo, pending, useExplicitIduCreditBank, state.iduPregCredit, state.iduShqCredit);

    if (debugCycles)
        logDispatchBegin(cycle, budget);
    auto dispatched = idu.dispatch(cycle, budget);
    if (debugCycles)
        std::cerr << "[vfsim] cycle " << cycle << " dispatch end n=" << dispatched.size() << "\n";
    forwardDispatchedToOoo(dispatched, state.iduToOooPipe, ooo, cycle, iduToOooDelay);

    if (debugCycles)
        std::cerr << "[vfsim] cycle " << cycle << " ooo begin\n";
    ooo.step();
    if (debugCycles)
        std::cerr << "[vfsim] cycle " << cycle << " ooo end\n";

    return isSimulationComplete(ifu, idu, ooo, state.iduToOooPipe);
}

std::string incompleteSimulationMessage(const IFU& ifu, const IDU& idu, const OoOCoreMainline& ooo)
{
    return "Simulation did not complete before maxCycles"
           " (ifu_done=" +
           std::string(ifu.done() ? "true" : "false") + ", idu_empty=" + std::string(idu.empty() ? "true" : "false") +
           ", rob=" + std::to_string(ooo.getRobSize()) + ", lsq=" + std::to_string(ooo.getLsqSize()) +
           ", shq=" + std::to_string(ooo.getShqSize()) + ", free_preg=" + std::to_string(ooo.getFreePreg()) +
           ", free_shq=" + std::to_string(ooo.getFreeShq()) + ", free_lsq=" + std::to_string(ooo.getFreeLsq()) +
           ", free_shqq=" + std::to_string(ooo.getFreeShqQueue()) + ")";
}

} // namespace

SimulationResult runVfInfo(const VfInfo& input, const ParamDB& db, const std::string& resultsDir, int64_t maxCycles)
{
    VfInfo vfInfo = input;
    canonicalizeVfInfo(vfInfo);
    normalizeProgramVregLiveRanges(vfInfo);
    const auto program = canonicalizeSingleSuperIterationLoops(vfInfo.body, vfInfo.params, db, vfInfo.defaultDtype);
    ProgramAnalysis analysis(vfInfo.params, vfInfo.values);
    const auto loopBounds = analysis.inferTopBlockLoopBounds(program);
    ProgramFlatten flattener(vfInfo.params);
    const auto& linear = flattener.flatten(program);
    const int topBlocks = static_cast<int>(loopBounds.size());

    IFU ifu(linear, vfInfo.params, &db, loopBounds, topBlocks, vfInfo.defaultDtype);
    IDU idu(db.uarch(), db, vfInfo.params, {}, topBlocks, loopBounds, vfInfo.defaultDtype, vfInfo.values);
    OoOCoreMainline ooo(db.uarch(), db, vfInfo.defaultDtype, vfInfo.values);
    return runSimulation(ifu, idu, ooo, db.uarch(), vfInfo.params, resultsDir, maxCycles, vfInfo.values);
}

SimulationResult runSimulation(
    IFU& ifu, IDU& idu, OoOCoreMainline& ooo, const UarchConfig& uarch, const ProgramAnalysis::ParamMap& params,
    const std::string& resultsDir, int64_t maxCycles, const std::unordered_map<std::string, ValueInfo>& values)
{
    maxCycles = applyMaxCyclesEnv(maxCycles);
    const bool debugCycles = std::getenv("PTOAS_VFSIM_DEBUG_CYCLES") != nullptr;
    const int64_t iduToOooDelay = uarch.iduToOooDelay;
    const bool useExplicitIduCreditBank = uarch.useExplicitIduCreditBank;
    const ValueStorageLookup valueStorage(values);
    (void)params;

    SimulationState state;
    state.iduPregCredit = ooo.getFreePreg();
    state.iduShqCredit = ooo.getFreeShq();

    while (state.cycle < maxCycles) {
        if (runOneSimulationCycle(
                ifu, idu, ooo, state, useExplicitIduCreditBank, iduToOooDelay, valueStorage, debugCycles)) {
            state.completed = true;
            break;
        }
        ++state.cycle;
    }

    if (!state.completed) {
        dumpSimulationLogs(resultsDir, idu, ooo);
        throw std::runtime_error(incompleteSimulationMessage(ifu, idu, ooo));
    }

    dumpSimulationLogs(resultsDir, idu, ooo);
    return SimulationResult{state.cycle, ooo.vfEndCycle(), resultsDir};
}

} // namespace vfsim

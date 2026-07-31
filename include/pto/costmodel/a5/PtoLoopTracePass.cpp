/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h" // formDedicatedExitBlocks

#include <cstring>
#include <string>
#include <vector>

using namespace llvm;

namespace {

static bool nameStartsWith(StringRef s, const char* p)
{
    size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}
static PointerType* i8PtrTy(LLVMContext& C) { return PointerType::getUnqual(Type::getInt8Ty(C)); }

static FunctionCallee getOrInsertLoopEnter(Module& M)
{
    LLVMContext& C = M.getContext();
    auto* i64 = Type::getInt64Ty(C);
    auto* i8p = i8PtrTy(C);
    auto* i32 = Type::getInt32Ty(C);
    return M.getOrInsertFunction(
        "__pto_trace_loop_enter", FunctionType::get(Type::getVoidTy(C), {i64, i8p, i32, i32}, false));
}
static FunctionCallee getOrInsertLoopIter(Module& M)
{
    LLVMContext& C = M.getContext();
    return M.getOrInsertFunction(
        "__pto_trace_loop_iter", FunctionType::get(Type::getVoidTy(C), {Type::getInt64Ty(C)}, false));
}
static FunctionCallee getOrInsertLoopExit(Module& M)
{
    LLVMContext& C = M.getContext();
    return M.getOrInsertFunction(
        "__pto_trace_loop_exit", FunctionType::get(Type::getVoidTy(C), {Type::getInt64Ty(C)}, false));
}

static DebugLoc getLoopDebugLoc(Loop* L)
{
    if (DebugLoc loc = L->getStartLoc())
        return loc;
    for (Instruction& I : *L->getHeader())
        if (I.getDebugLoc())
            return I.getDebugLoc();
    return DebugLoc();
}

static uint64_t computeLoopId(Function& F, Loop* L)
{
    DebugLoc loc = getLoopDebugLoc(L);
    std::string key;
    raw_string_ostream os(key);
    os << F.getName() << "|";
    if (loc) {
        if (auto* scope = dyn_cast_or_null<DILocalScope>(loc.getScope()))
            if (auto* file = scope->getFile())
                os << file->getFilename() << "|";
        os << loc.getLine() << ":" << loc.getCol();
    }
    os.flush();
    return static_cast<uint64_t>(hash_value(StringRef(key)));
}

struct LoopWork {
    uint64_t loopId;
    BasicBlock* preheader;
    BasicBlock* header;
    std::string file;
    int line;
    int col;
    std::vector<BasicBlock*> exitBlocks;
};

static bool isScopeCtorCall(const CallBase* call)
{
    const Function* callee = call->getCalledFunction();
    return callee != nullptr && callee->getName().contains("ScopeSentinelC");
}

static std::vector<const Instruction*> collectScopeEnters(Function& F)
{
    std::vector<const Instruction*> scopeEnters;
    for (BasicBlock& BB : F)
        for (Instruction& I : BB)
            if (const auto* call = dyn_cast<CallBase>(&I))
                if (isScopeCtorCall(call))
                    scopeEnters.push_back(&I);
    return scopeEnters;
}

template <class InScopeFn>
static void collectLoopInScope(Loop* L, Function& F, std::vector<LoopWork>& out, InScopeFn&& inScope)
{
    for (Loop* sub : L->getSubLoops())
        collectLoopInScope(sub, F, out, inScope);

    if (!inScope(L))
        return;

    BasicBlock* pre = L->getLoopPreheader();
    DebugLoc loc = getLoopDebugLoc(L);
    if (!pre) {
        errs() << "[PtoLoopTrace] WARN: loop without preheader skipped (" << F.getName() << ")\n";
        return;
    }
    if (!loc) {
        errs() << "[PtoLoopTrace] WARN: loop without DebugLoc skipped (" << F.getName() << ")\n";
        return;
    }

    LoopWork w;
    w.loopId = computeLoopId(F, L);
    w.preheader = pre;
    w.header = L->getHeader();
    if (auto* scope = dyn_cast_or_null<DILocalScope>(loc.getScope()))
        if (auto* file = scope->getFile())
            w.file = file->getFilename().str();
    w.line = static_cast<int>(loc.getLine());
    w.col = static_cast<int>(loc.getCol());

    if (!L->hasDedicatedExits()) {
        errs() << "[PtoLoopTrace] WARN: loop without dedicated exits skipped (" << F.getName() << ")\n";
        return;
    }
    SmallVector<BasicBlock*, 4> exitBlks;
    L->getUniqueExitBlocks(exitBlks);
    for (BasicBlock* eb : exitBlks)
        w.exitBlocks.push_back(eb);
    out.push_back(std::move(w));
}

template <class InScopeFn>
static bool formDedicatedExitsForScopedLoops(LoopInfo& LI, DominatorTree& DT, InScopeFn&& loopInScope)
{
    bool cfgChanged = false;
    for (Loop* L : LI) {
        SmallVector<Loop*, 8> nest;
        nest.push_back(L);
        for (size_t i = 0; i < nest.size(); ++i)
            for (Loop* sub : nest[i]->getSubLoops())
                nest.push_back(sub);
        for (Loop* cur : nest)
            if (loopInScope(cur))
                cfgChanged |= formDedicatedExitBlocks(
                    cur, &DT, &LI,
                    /*MSSAU=*/nullptr,
                    /*PreserveLCSSA=*/false);
    }
    return cfgChanged;
}

static void insertLoopEnterAndIter(LoopWork& w, LLVMContext& C, FunctionCallee enterFn, FunctionCallee iterFn)
{
    ConstantInt* loopIdC = ConstantInt::get(C, APInt(64, w.loopId));
    IRBuilder<> enterBuilder(w.preheader->getTerminator());
    Value* filePtr = enterBuilder.CreateGlobalStringPtr(w.file.empty() ? StringRef("") : StringRef(w.file));
    enterBuilder.CreateCall(
        enterFn, {loopIdC, filePtr, ConstantInt::get(C, APInt(32, static_cast<uint64_t>(w.line))),
                  ConstantInt::get(C, APInt(32, static_cast<uint64_t>(w.col)))});

    IRBuilder<> iterBuilder(w.header, w.header->getFirstInsertionPt());
    iterBuilder.CreateCall(iterFn, {loopIdC});
}

static void insertLoopExits(LoopWork& w, LLVMContext& C, FunctionCallee exitFn)
{
    ConstantInt* loopIdC = ConstantInt::get(C, APInt(64, w.loopId));
    for (BasicBlock* eb : w.exitBlocks) {
        IRBuilder<> b(eb, eb->getFirstInsertionPt());
        b.CreateCall(exitFn, {loopIdC});
    }
}

struct PtoLoopTracePass : PassInfoMixin<PtoLoopTracePass> {
    PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM)
    {
        if (F.isDeclaration())
            return PreservedAnalyses::all();

        Module& M = *F.getParent();
        LoopInfo& LI = FAM.getResult<LoopAnalysis>(F);
        DominatorTree& DT = FAM.getResult<DominatorTreeAnalysis>(F);

        std::vector<const Instruction*> scopeEnters = collectScopeEnters(F);

        const bool isDemo = nameStartsWith(F.getName(), "__pto_demo_");
        if (scopeEnters.empty() && !isDemo) {
            return PreservedAnalyses::all();
        }

        auto loopInScope = [&](Loop* L) -> bool {
            if (isDemo)
                return true;
            BasicBlock* h = L->getHeader();
            for (const Instruction* en : scopeEnters)
                if (DT.dominates(en, &*h->getFirstInsertionPt()))
                    return true;
            return false;
        };

        bool cfgChanged = formDedicatedExitsForScopedLoops(LI, DT, loopInScope);
        if (cfgChanged)
            DT.recalculate(F);

        std::vector<LoopWork> work;
        for (Loop* L : LI)
            collectLoopInScope(L, F, work, loopInScope);
        if (work.empty())
            return PreservedAnalyses::all();

        FunctionCallee enterFn = getOrInsertLoopEnter(M);
        FunctionCallee iterFn = getOrInsertLoopIter(M);
        FunctionCallee exitFn = getOrInsertLoopExit(M);
        LLVMContext& C = M.getContext();

        bool changed = false;
        for (LoopWork& w : work) {
            insertLoopEnterAndIter(w, C, enterFn, iterFn);
            changed = true;
        }

        for (LoopWork& w : work) {
            insertLoopExits(w, C, exitFn);
        }

        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

    static bool isRequired() { return true; }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, "PtoLoopTrace", LLVM_VERSION_STRING, [](PassBuilder& PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager& FPM, ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name != "pto-loop-trace")
                            return false;
                        FPM.addPass(PtoLoopTracePass());
                        return true;
                    });
                PB.registerPipelineStartEPCallback([](ModulePassManager& MPM, OptimizationLevel) {
                    MPM.addPass(createModuleToFunctionPassAdaptor(PtoLoopTracePass()));
                });
            }};
}

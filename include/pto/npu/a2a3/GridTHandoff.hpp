/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe THANDOFF -- 接力计数 (relay counting) across a
// time-division producer handoff.
//
// Design_spec: Grid_TPUSH_TPOP_WSE核间握手机制选型-时分MPSC与真同时MPSC综合.md
//   §1.3 接力计数 (INSTALL_BASE / OPEN_ACK / START)
//   §3.2 B1  drain/close + 单调序号接力棒
//   §8.2 H1-H4 (时分 MPSC 写者切换不变量)
//
// THE PROBLEM.  A channel's CONSUMER stays put while its PRODUCER changes between
// phases (时分 MPSC: the writer's identity moves, but at any instant there is
// still exactly one writer).  Because a GridPipe is bound to one (producer,
// consumer) pair, the new producer means a NEW PIPE -- yet the physical resources
// do not move: same ring, same ready/free IPC_SCB pair, same core draining them.
// Both pipes are therefore declared over the SAME window with the SAME ScbId, and
// only the COUNTERS have to cross the boundary.  That crossing is this file.
//
// WHY NOT JUST RESET.  The successor's prod_idx cannot restart at 0 while the ring
// still holds tiles the retiring producer wrote: slot (0 % SlotCount) is the
// oldest UNDRAINED tile, and the free-credit threshold `prod - SlotCount + 1` no
// longer describes what the consumer has actually consumed.  §9 names naked
// counter resets as the classic pseudo-solution.  So the counters RELAY -- the
// absolute sequence continues across the boundary (E -> E+1 -> ...) -- except at a
// boundary where the ring is provably drained, where restarting at 0 is both safe
// and useful (it keeps the absolute counts away from the IPC_SCB's 16-bit
// ceiling).  Both branches are decided in half A below from two values the retiring
// pipe already holds -- its ready_scb and its cons_idx -- so nothing needs to be
// copied into a side structure to describe the boundary.
//
// THE THREE HALVES.  Every core runs the same THANDOFF; its roles come from the
// topology, exactly as TPUSH/TPOP/TREDUCE derive theirs, and under SPMD one core
// plays both roles on the same call:
//
//   A (as CONSUMER)  MOV_SPR2X E out of my OWN ready_scb (its last writer was the
//                    retiring producer's final SYNC_HSCB(READY), so the value is
//                    already there) straight into my outgoing baton L1 word -- no
//                    GPR on this path, L1 is where ST_HSCB forwards from.  Read it
//                    back with MOV_L12X only to compare against cons_idx and pick
//                    the branch, then INSTALL_BASE into my incoming producer: the
//                    baton shipped by ST_HSCB into ITS L1, the free-credit baseline
//                    as an ordinary FREE store into its free_scb, then -- after a
//                    publish fence -- the doorbell (install_scb).
//   B (as PRODUCER)  block on that doorbell from MY incoming consumer (WAIT_SPR,
//                    no MOV: the threshold test lives in the instruction),
//                    MOV_L12X the delivered baseline out of L1 into my prod_idx
//                    GPR, zero that consumer's ready_scb if it rebased (it cannot
//                    zero its own -- 约束①: a core may not write its own IPC_SCB),
//                    and send OPEN_ACK.
//   C (as CONSUMER)  adopt my own cons_idx: carried forward on the relay branch,
//                    or zeroed on the rebase branch -- but only AFTER OPEN_ACK,
//                    because until my successor's zeroing store has landed my
//                    ready_scb still reads E, and `ready(E) >= cons(0)+1` would
//                    let the very next TPOP drain a stale slot.
//
// A precedes B on every core and B precedes C, which is what makes the whole
// thing deadlock-free: A never blocks on another core's handoff, so every core
// reaches its A; every B therefore has a sender; every C therefore has an acker.
//
// WHY prod_idx TAKES THE L1 ROUTE AND free TAKES THE SCOREBOARD ROUTE.  free_scb is
// a semaphore: the producer only ever asks "is it >= my threshold", which WAIT_SPR
// answers against the SPR directly, so free never leaves the scoreboard file.
// prod_idx is not a semaphore -- it is a run-counter the incoming producer must
// LOAD into a GPR -- and the cross-core transport that carries it (ST_HSCB) both
// sources from and lands in L1.  Hence the two MOV-class facades, and hence their
// asymmetry: mov_ipc_scb_to_l1 (SPR -> L1, forwarding) and mov_l1_to_gpr (L1 -> GPR,
// loading).  There is no SPR -> GPR instruction because nothing wants one: a
// scoreboard is only ever compared or forwarded.
//
// PRECONDITION (H3, the caller's).  The retiring producer must have stopped
// publishing and its final SYNC_HSCB(READY) must have LANDED before its consumer
// reads the baton -- otherwise the relayed E is short and the successor overwrites
// a slot the retiring producer already filled.  Pass `retiredEnd` (the phase's
// end-exclusive tile count, statically known in every scheduled collective) and
// half A proves it with a WAIT_SPR; pass 0 only when the schedule establishes
// quiescence some other way.  §3.2 B1 is emphatic that "flip the owner and wait a
// while" is NOT a substitute for that proof.

#ifndef PTO_A2A3_GRID_THANDOFF_HPP
#define PTO_A2A3_GRID_THANDOFF_HPP

#include <cstdint>

#include <pto/npu/a2a3/GridTPush.hpp> // for the a2a3_grid_payload::RemoteScbPtr hook
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

namespace grid_handoff_detail {

// Fault sentinel for a header word, null-safe: nullptr + offset is UB and would
// slip a non-null (but invalid) pointer past MockSetFault's null guard.
AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

// Overwrite a LOCAL L1 word from a scalar.  An ordinary store, not a new machine
// instruction; the handoff needs it only on the rebase branch, where the value to
// forward (0) is not the one MOV_SPR2X deposited.
AICORE inline void StoreBatonL1(__gm__ uint32_t* localL1, uint32_t value)
{
    grid_cce_detail::write_local_word(localL1, value);
}

} // namespace grid_handoff_detail

// Hand the producer role of one physical channel from `oldPipe` to `newPipe`.
//
// `handoffSeq` is the window's monotone handoff generation (1 for the first
// handoff, 2 for the next, ...).  It is the value carried by both doorbells, so
// it must be identical on every participating core and strictly increasing per
// window -- the same discipline as a5 TPipe's FlagID, and the reason the doc
// insists OPEN_ACK is an IDENTITY confirmation and not merely a numeric one:
// with a bare "non-zero" doorbell the second handoff would be satisfied by the
// first one's leftovers.
//
// `retiredEnd` is the retiring phase's end-exclusive tile count on this edge (0 =
// "no proof available, the caller guarantees quiescence"); see the header note.
//
// Returns false on a mock spin-timeout or a mis-wiring, having written a fault
// sentinel; GRID_THANDOFF_IMPL below is the block-forever form.
template <typename OldPipe, typename NewPipe>
AICORE bool GRID_TRY_THANDOFF_IMPL(
    OldPipe& oldPipe, NewPipe& newPipe, uint32_t handoffSeq, uint32_t retiredEnd,
    uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    // The two pipes must be the SAME physical channel under two producer
    // bindings.  Ring geometry and IPC_SCB slots are part of that identity, and
    // getting either wrong is silent corruption rather than a hang, so both are
    // compile-time errors.
    static_assert(
        OldPipe::SlotStride == NewPipe::SlotStride && OldPipe::SlotCount == NewPipe::SlotCount,
        "THANDOFF: the retiring and incoming pipes share one ring, so SlotStride and SlotCount must match.");
    static_assert(
        OldPipe::ReadyScbSlot == NewPipe::ReadyScbSlot && OldPipe::FreeScbSlot == NewPipe::FreeScbSlot,
        "THANDOFF: the retiring and incoming pipes share one scoreboard pair, so they must be declared with the "
        "same ScbId (the default 2*Dir differs per direction -- pass an explicit shared ScbId).");
    static_assert(
        !(OldPipe::Dir == NewPipe::Dir && OldPipe::Dist == NewPipe::Dist),
        "THANDOFF: both pipes name the same (Dir, Dist), so they are bound to the same producer -- there is no "
        "handoff to perform.");
    static_assert(
        NewPipe::Dir != GridDirection::SOURCE,
        "THANDOFF: a SOURCE-bound pipe has no cross-core producer to install a baseline into.");
    // Register budget, charged only where it is spent: the steady-state pipe
    // occupies ScbId..ScbId+1, and participating in a handoff additionally
    // occupies the 接力计数 trio ScbId+2..ScbId+4.
    static_assert(
        NewPipe::OpenScbSlot < 16,
        "THANDOFF: a handoff-capable pipe occupies IPC_SCB slots ScbId..ScbId+3 (ready, free, install, open), "
        "so ScbId + 3 must stay inside the 0..15 slot file.");

    // Same window, or the relayed counters describe a ring the successor will
    // never write.  Cheap pointer identity check -- the mock's addresses, the
    // native lowering's slot ids.
    if (oldPipe.cons.readyScb != newPipe.cons.readyScb || oldPipe.slots.base != newPipe.slots.base) {
        grid_mock::MockSetFault(
            grid_handoff_detail::FaultWord(newPipe.prod.batonL1), grid_mock::kFaultHandoffWindowMismatch);
        return false;
    }

    // =====================================================================
    // Half A -- as CONSUMER: relay the retiring prod_idx to my successor.
    // =====================================================================

    // (A1) H3: prove the retiring producer's last READY store has landed, so the
    //      E read below is its FINAL prod_idx and not a short prefix.  This is a
    //      pure WAIT_SPR: the threshold comparison happens inside the instruction,
    //      against the SPR, with nothing moved into a register.  Boundary cells
    //      have no retiring producer and nothing to wait for, which lets the
    //      caller pass one uniform constant for the whole mesh.
    if (retiredEnd != 0 && oldPipe.HasProducer()) {
        if (!wait_ipc_scb_sim(oldPipe.cons.readyScb, retiredEnd, OldPipe::ReadyScbSlot, maxSpins)) {
            grid_mock::MockSetFault(
                grid_handoff_detail::FaultWord(oldPipe.cons.readyScb), grid_mock::kFaultHandoffRetireTimeout);
            return false;
        }
    }

    // (A2) MOV_SPR2X: move E -- the retiring producer's final prod_idx, which its
    //      last SYNC_HSCB(READY) already left in this core's ready_scb -- out of the
    //      SPR file and into the outgoing baton L1 word.  L1 is where it has to be
    //      anyway (that is what ST_HSCB forwards from), so the count is staged and
    //      forwarded without ever passing through a GPR.
    mov_ipc_scb_to_l1(newPipe.cons.batonL1, oldPipe.cons.readyScb, OldPipe::ReadyScbSlot);

    // (A3) MOV_L12X: read the staged word back into a scalar, purely to DECIDE.
    //      The forwarding above does not need this; the branch does, and a scalar
    //      is the only thing the unit can compare cons_idx against.  `<=` rather
    //      than `==` so a never-used channel (both zero) takes the drained branch.
    const uint32_t endIndex = mov_l1_to_gpr(newPipe.cons.batonL1);
    const uint32_t consIndex = oldPipe.cons.consIndex;
    bool rebase = (endIndex <= consIndex);

    if (newPipe.HasProducer()) {
        // (A4) INSTALL_BASE.  Two DIFFERENT absolute counts go out, by two
        //      different routes:
        //        prod_idx baseline -> the successor's L1, because prod_idx is a GPR
        //          run-counter on both ends and ST_HSCB-to-L1 is the transport that
        //          a MOV_L12X can pick up (half B);
        //        free credit       -> the successor's free_scb, the ordinary FREE
        //          scoreboard store, carrying cons_idx exactly as a TPOP would.
        //      They differ on the relay branch -- E tiles published, only cons_idx
        //      of them consumed -- and that gap IS the backpressure that keeps the
        //      successor off the undrained slots.
        const uint32_t prodBase = rebase ? 0U : endIndex;
        const uint32_t freeBase = rebase ? 0U : consIndex;

        // The relay branch forwards the word MOV_SPR2X already staged; only the
        // rebase branch has to overwrite it, because 0 is not what was in ready_scb.
        if (rebase) {
            grid_handoff_detail::StoreBatonL1(newPipe.cons.batonL1, prodBase);
        }

        const int newProdRank = newPipe.ProducerRank();
        __gm__ uint32_t* peerBaton =
            a2a3_grid_payload::RemoteScbPtr(newPipe.ctx.runtimeCtx, newPipe.prod.batonL1, newProdRank);
        __gm__ uint32_t* peerFree =
            a2a3_grid_payload::RemoteScbPtr(newPipe.ctx.runtimeCtx, newPipe.prod.freeScb, newProdRank);
        sync_hscb(peerBaton, prodBase);
        sync_hscb(peerFree, freeBase);

        // Publish fence (data-before-doorbell, C2 -- the same discipline TPUSH
        // applies between payload and ready).  Without it the successor can see
        // its doorbell and MOV an L1 word that still holds the previous handoff's
        // baseline.
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        __gm__ uint32_t* peerInstall =
            a2a3_grid_payload::RemoteScbPtr(newPipe.ctx.runtimeCtx, newPipe.prod.installScb, newProdRank);
        sync_hscb(peerInstall, handoffSeq);
    } else {
        // No successor producer: nobody can zero my ready_scb (约束①), so I must
        // not zero my cons_idx either -- carry both forward untouched.
        rebase = false;
    }

    // =====================================================================
    // Half B -- as PRODUCER: adopt the baseline my successor consumer installed.
    // =====================================================================
    if (newPipe.HasConsumer()) {
        if (!wait_ipc_scb_sim(newPipe.prod.installScb, handoffSeq, NewPipe::InstallScbSlot, maxSpins)) {
            grid_mock::MockSetFault(
                grid_handoff_detail::FaultWord(newPipe.prod.installScb), grid_mock::kFaultHandoffInstallTimeout);
            return false;
        }

        // MOV_L12X: the count crossed the fabric as an ST_HSCB into this core's
        // L1, and prod_idx is a GPR -- TPUSH derives the ring slot and the free
        // threshold from it -- so the delivered word has to be moved into a
        // register before it means anything.  The doorbell above is what makes
        // this read safe; the MOV carries no synchronisation of its own.
        newPipe.prod.prodIndex = mov_l1_to_gpr(newPipe.prod.batonL1);

        const int newConsRank = newPipe.ConsumerRank();
        // A zero baseline means -- and can only mean -- that my consumer took the
        // rebase branch: the relay branch installs E, and E > cons_idx >= 0 there.
        // So this is where the rebase actually happens on the consumer's side,
        // because only an EXTERNAL writer may store into its ready_scb.
        if (newPipe.prod.prodIndex == 0) {
            __gm__ uint32_t* peerReady =
                a2a3_grid_payload::RemoteScbPtr(newPipe.ctx.runtimeCtx, newPipe.cons.readyScb, newConsRank);
            sync_hscb(peerReady, 0);
        }

        // OPEN_ACK must not overtake that zeroing store; per-edge publish order
        // plus this fence is what lets the consumer treat the ack as proof that
        // its ready_scb is already 0.
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        __gm__ uint32_t* peerOpen =
            a2a3_grid_payload::RemoteScbPtr(newPipe.ctx.runtimeCtx, newPipe.cons.openScb, newConsRank);
        sync_hscb(peerOpen, handoffSeq);
    }

    // =====================================================================
    // Half C -- as CONSUMER: adopt my own cons_idx (H2: prod == cons).
    // =====================================================================
    if (rebase) {
        if (!wait_ipc_scb_sim(newPipe.cons.openScb, handoffSeq, NewPipe::OpenScbSlot, maxSpins)) {
            grid_mock::MockSetFault(
                grid_handoff_detail::FaultWord(newPipe.cons.openScb), grid_mock::kFaultHandoffOpenTimeout);
            return false;
        }
        newPipe.cons.consIndex = 0;
    } else {
        // Relay: keep draining, through the new pipe, the tiles the OLD producer
        // left in the ring -- the payload is already local, so a TPOP does not
        // care who wrote it, and the free doorbells it emits now reach the NEW
        // producer, which is exactly what un-blocks the slots it wants next.
        newPipe.cons.consIndex = consIndex;
    }
    return true;
}

template <typename OldPipe, typename NewPipe>
AICORE void GRID_THANDOFF_IMPL(OldPipe& oldPipe, NewPipe& newPipe, uint32_t handoffSeq, uint32_t retiredEnd)
{
    (void)GRID_TRY_THANDOFF_IMPL<OldPipe, NewPipe>(oldPipe, newPipe, handoffSeq, retiredEnd, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_THANDOFF_HPP

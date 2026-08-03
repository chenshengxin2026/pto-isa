/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host driver for the GridPipe 接力计数 producer-handoff smoke kernel (THANDOFF).
//
// Layout: a gridRows x gridCols logical grid on one device, backed by per-cell GM
// windows + a fake CommDeviceContext (same mock as the FFN GridPipe demos).  Cell
// c's input tile k is stamped (c+1)*10 + k.  The golden model below replays the
// kernel's pop schedule exactly, so every output slot is checked by STAMP -- a
// relayed baseline that is off by one shows up as another cell's data, not as a
// tolerance miss.  Verified in-process (no data files).

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "runtime/rt.h"
#ifdef RT_STREAM_PRIORITY_DEFAULT
#undef RT_STREAM_PRIORITY_DEFAULT
#endif

#ifdef AICORE
#undef AICORE
#endif
#define AICORE

#ifndef __gm__
#define __gm__
#endif

// Skip SdmaWorkspaceManager pull-in from common.hpp (needs CCE attrs on host).
#define PTO_COMM_ST_SKIP_SDMA_WORKSPACE_MANAGER
#include "common.hpp"

#include "handoff_smoke_config.hpp"
#include "handoff_smoke_launch.hpp"

static bool ParseDeviceIdValue(const char* value, int& deviceId)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    deviceId = static_cast<int>(parsed);
    return true;
}

static int GetDeviceId(int argc, char** argv)
{
    int deviceId = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device-id") == 0 || std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 >= argc || !ParseDeviceIdValue(argv[i + 1], deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
        constexpr const char* kPrefix = "--device-id=";
        constexpr size_t kPrefixLen = 12;
        if (std::strncmp(argv[i], kPrefix, kPrefixLen) == 0) {
            if (!ParseDeviceIdValue(argv[i] + kPrefixLen, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
    }
    if (const char* env = std::getenv("ASCEND_DEVICE_ID")) {
        (void)ParseDeviceIdValue(env, deviceId);
    }
    return deviceId;
}

static bool InitAcl(int deviceId)
{
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    aRet = aclrtSetDevice(deviceId);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    return true;
}

struct Resources {
    aclrtStream stream = nullptr;
    void* windows_dev = nullptr;
    void* in_dev = nullptr;
    void* out_dev = nullptr;
    void* hccl_ctx_dev = nullptr;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    int rows = HANDOFF_ROWS;
    int cols = HANDOFF_COLS;
    size_t cells = static_cast<size_t>(HANDOFF_ROWS) * static_cast<size_t>(HANDOFF_COLS);
    size_t windowsBytes = 0;
    size_t inBytes = 0;
    size_t outBytes = 0;
};

// Golden pop schedule for one cell -- a replay of the kernel's own control flow.
// Returns the number of slots written; `expect` is filled with their stamps and
// the rest of the cell's output region stays zero.
static int GoldenPops(int row, int col, int cols, float* expect)
{
    const int cell = row * cols + col;
    (void)cell;
    const bool hasWestProducer = (col > 0);
    const bool hasNorthProducer = (row > 0);
    const int west = row * cols + (col - 1);
    const int north = (row - 1) * cols + col;

    int slot = 0;
    // Phase 1: pops from the west producer, only P1_POPS of the P1_TILES pushed.
    if (hasWestProducer) {
        for (int i = 0; i < HANDOFF_P1_POPS; ++i) {
            expect[slot++] = HandoffStamp(west, i);
        }
    }
    // Phase 2, leftovers: still the WEST producer's payload, drained through the
    // SOUTH pipe after the handoff relayed the counters.  Only reachable where a
    // successor producer exists (a TPOP needs one).
    if (hasWestProducer && hasNorthProducer) {
        for (int i = HANDOFF_P1_POPS; i < HANDOFF_P1_TILES; ++i) {
            expect[slot++] = HandoffStamp(west, i);
        }
    }
    // Phase 2, new tiles: written by the north producer from the relayed (or
    // rebased) baseline.
    if (hasNorthProducer) {
        for (int i = 0; i < HANDOFF_P2_TILES; ++i) {
            expect[slot++] = HandoffStamp(north, HANDOFF_P1_TILES + i);
        }
    }
    return slot;
}

static bool BuildFakeHcclCtx(Resources& r)
{
    CommDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(HANDOFF_WINDOW_BYTES);
    uint64_t base = reinterpret_cast<uint64_t>(r.windows_dev);
    for (size_t i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + i * static_cast<size_t>(HANDOFF_WINDOW_BYTES);
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }
    if (aclrtMalloc(&r.hccl_ctx_dev, sizeof(CommDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(
            r.hccl_ctx_dev, sizeof(CommDeviceContext), &hostCtx, sizeof(CommDeviceContext),
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(hccl_ctx) failed" << std::endl;
        return false;
    }
    return true;
}

static bool Allocate(Resources& r)
{
    if (r.cells == 0 || r.cells > HCCL_MAX_RANK_NUM) {
        std::cerr << "[ERROR] invalid cell count " << r.cells << std::endl;
        return false;
    }
    if (aclrtCreateStream(&r.stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return false;
    }

    r.windowsBytes = r.cells * static_cast<size_t>(HANDOFF_WINDOW_BYTES);
    r.inBytes = r.cells * static_cast<size_t>(HANDOFF_IN_TILES) * static_cast<size_t>(HANDOFF_TILE_BYTES);
    r.outBytes = r.cells * static_cast<size_t>(HANDOFF_OUT_TILES) * static_cast<size_t>(HANDOFF_TILE_BYTES);

    aclrtMalloc(&r.windows_dev, r.windowsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.in_dev, r.inBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.out_dev, r.outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!r.windows_dev || !r.in_dev || !r.out_dev) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }
    aclrtMemset(r.windows_dev, r.windowsBytes, 0, r.windowsBytes);
    aclrtMemset(r.out_dev, r.outBytes, 0, r.outBytes);

    // Stamp cell c's tile k with (c+1)*10 + k.
    std::vector<float> hostIn(r.cells * static_cast<size_t>(HANDOFF_IN_TILES) * HANDOFF_TILE_ELEMS);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        for (int k = 0; k < HANDOFF_IN_TILES; ++k) {
            float stamp = HandoffStamp(static_cast<int>(cell), k);
            float* dst = hostIn.data() +
                         (cell * static_cast<size_t>(HANDOFF_IN_TILES) + k) * static_cast<size_t>(HANDOFF_TILE_ELEMS);
            for (int e = 0; e < HANDOFF_TILE_ELEMS; ++e) {
                dst[e] = stamp;
            }
        }
    }
    if (aclrtMemcpy(r.in_dev, r.inBytes, hostIn.data(), r.inBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(in_dev) failed" << std::endl;
        return false;
    }

    if (!BuildFakeHcclCtx(r)) {
        return false;
    }

    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }
    return true;
}

static bool Verify(Resources& r)
{
    std::vector<float> outHost(r.cells * static_cast<size_t>(HANDOFF_OUT_TILES) * HANDOFF_TILE_ELEMS);
    if (aclrtMemcpy(outHost.data(), r.outBytes, r.out_dev, r.outBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] out D2H memcpy failed" << std::endl;
        return false;
    }

    double maxDiff = 0.0;
    size_t badCells = 0;
    for (int row = 0; row < r.rows; ++row) {
        for (int col = 0; col < r.cols; ++col) {
            const size_t cell = static_cast<size_t>(row) * r.cols + col;
            float expect[HANDOFF_OUT_TILES] = {0.0f};
            const int used = GoldenPops(row, col, r.cols, expect);
            const bool rebased = (col == 0) || (HANDOFF_P1_POPS == HANDOFF_P1_TILES);

            double cellDiff = 0.0;
            for (int s = 0; s < HANDOFF_OUT_TILES; ++s) {
                const float* tile =
                    outHost.data() + (cell * static_cast<size_t>(HANDOFF_OUT_TILES) + s) * HANDOFF_TILE_ELEMS;
                for (int e = 0; e < HANDOFF_TILE_ELEMS; ++e) {
                    double d = std::abs(static_cast<double>(tile[e]) - static_cast<double>(expect[s]));
                    if (d > cellDiff) {
                        cellDiff = d;
                    }
                }
            }
            if (cellDiff > maxDiff) {
                maxDiff = cellDiff;
            }
            if (cellDiff > 0.0) {
                ++badCells;
            }

            std::cout << "[INFO] cell " << cell << " (r=" << row << ",c=" << col << ") "
                      << (rebased ? "rebase" : "relay ") << " pops=" << used << " expect=[";
            for (int s = 0; s < HANDOFF_OUT_TILES; ++s) {
                std::cout << expect[s] << (s + 1 < HANDOFF_OUT_TILES ? " " : "");
            }
            std::cout << "] got=[";
            for (int s = 0; s < HANDOFF_OUT_TILES; ++s) {
                const float* tile =
                    outHost.data() + (cell * static_cast<size_t>(HANDOFF_OUT_TILES) + s) * HANDOFF_TILE_ELEMS;
                std::cout << tile[0] << (s + 1 < HANDOFF_OUT_TILES ? " " : "");
            }
            std::cout << "]" << std::endl;
        }
    }

    std::cout << "[INFO] handoff smoke: grid=" << r.rows << "x" << r.cols << " p1=" << HANDOFF_P1_TILES
              << " pops=" << HANDOFF_P1_POPS << " p2=" << HANDOFF_P2_TILES << " tile=" << HANDOFF_T << "x" << HANDOFF_W
              << " max diff=" << maxDiff << std::endl;
    if (maxDiff != 0.0) {
        std::cerr << "[ERROR] mismatch (max diff " << maxDiff << ", " << badCells << " bad cells)" << std::endl;
        return false;
    }
    return true;
}

static bool CheckFaults(Resources& r)
{
    constexpr size_t kFlagWords = static_cast<size_t>(HANDOFF_GRID_FLAGS_BYTES) / sizeof(uint32_t);
    // Mirror of grid_pipe_runtime.hpp: one cache line (16 u32 words) per
    // scoreboard, so the live counters sit at word 0 of each line and everything
    // else in the header is a fault sentinel or reserved.
    constexpr size_t kScbWordStride = 16;
    std::vector<uint32_t> flags(r.cells * kFlagWords, 0);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        auto* src = reinterpret_cast<uint8_t*>(r.windows_dev) + cell * HANDOFF_WINDOW_BYTES;
        auto* dst = flags.data() + cell * kFlagWords;
        if (aclrtMemcpy(
                dst, kFlagWords * sizeof(uint32_t), src, kFlagWords * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
            std::cerr << "[ERROR] flag D2H memcpy failed for cell " << cell << std::endl;
            return false;
        }
    }
    bool ok = true;
    for (size_t cell = 0; cell < r.cells; ++cell) {
        const uint32_t* cf = flags.data() + cell * kFlagWords;
        for (size_t i = 0; i < kFlagWords; ++i) {
            if (i % kScbWordStride == 0) {
                continue; // a live scoreboard counter, not a sentinel
            }
            if (cf[i] >= 0x100U) {
                std::cerr << "[ERROR] GridPipe fault cell=" << cell << " flagWord=" << i << " code=0x" << std::hex
                          << cf[i] << std::dec << std::endl;
                ok = false;
            }
        }
    }
    return ok;
}

static void Cleanup(Resources& r)
{
    if (r.hccl_ctx_dev) {
        aclrtFree(r.hccl_ctx_dev);
    }
    if (r.windows_dev) {
        aclrtFree(r.windows_dev);
    }
    if (r.in_dev) {
        aclrtFree(r.in_dev);
    }
    if (r.out_dev) {
        aclrtFree(r.out_dev);
    }
    if (r.stream) {
        aclrtDestroyStream(r.stream);
    }
}

static bool Run()
{
    Resources r;
    if (!Allocate(r)) {
        Cleanup(r);
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    launchHandoffSmokeKernel(
        reinterpret_cast<uint8_t*>(r.ffts), reinterpret_cast<uint8_t*>(r.windows_dev),
        reinterpret_cast<uint8_t*>(r.in_dev), reinterpret_cast<uint8_t*>(r.out_dev),
        reinterpret_cast<uint8_t*>(r.hccl_ctx_dev), r.rows, r.cols, r.stream);
    aclError syncRet = aclrtSynchronizeStream(r.stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    bool faultsOk = (syncRet == ACL_SUCCESS) && CheckFaults(r);
    std::cout << "[INFO] launch+sync " << us << " us (rc=" << static_cast<int>(syncRet)
              << " faults_ok=" << (faultsOk ? 1 : 0) << ")" << std::endl;

    bool ok = (syncRet == ACL_SUCCESS) && faultsOk && Verify(r);
    Cleanup(r);
    return ok;
}

int main(int argc, char** argv)
{
    int deviceId = GetDeviceId(argc, argv);
    std::cout << "[INFO] using device " << deviceId << std::endl;
    if (!InitAcl(deviceId)) {
        return 1;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  GridPipe 接力计数 producer-handoff smoke test (THANDOFF)" << std::endl;
    std::cout << "  grid=" << HANDOFF_ROWS << "x" << HANDOFF_COLS << "  phase1=" << HANDOFF_P1_TILES << " tiles / "
              << HANDOFF_P1_POPS << " popped  phase2=" << HANDOFF_P2_TILES << " tiles" << std::endl;
    std::cout << "  branch: col 0 rebases; cols > 0 "
              << (HANDOFF_P1_POPS == HANDOFF_P1_TILES ? "rebase (ring drained)" : "relay (ring undrained)")
              << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = Run();
    std::cout << (ok ? "[SUCCESS] GridPipe producer-handoff smoke PASS." :
                       "[FAILED] GridPipe producer-handoff smoke FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}

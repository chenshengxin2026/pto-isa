/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Single-device multi-block FFN driver, AllGather combine via pto::comm::TGATHER.
//
// gridRows*gridCols blocks form a logical grid on one NPU.  A compute kernel
// publishes each cell's fp16 hidden shard [T,Fi] into its HCCL window slot; a
// host stream-sync barrier follows; then a collective kernel has every cell run
// pto::comm::TGATHER over the row's column window slots (gathering all hidden
// shards), re-lays-out to hidden_full[T,F], down-projects with its own
// W_down[:,Hc] shard, and stores its [T,Hc] output shard.  The HCCL context is
// the same single-device fake context as the GridPipe demo.

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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

#include "common.hpp"

#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#include "ffn_config.hpp"
#include "kernel_launch.hpp"

struct DeviceResources {
    aclrtStream stream = nullptr;
    void *x_dev = nullptr;
    void *w_gate_dev = nullptr;
    void *w_up_dev = nullptr;
    void *w_down_dev = nullptr;
    void *gate_partial_dev = nullptr;
    void *up_partial_dev = nullptr;
    void *hidden_dev = nullptr;
    void *down_partial_dev = nullptr;
    void *gather_scratch_dev = nullptr;
    void *y_output_dev = nullptr;
    void *windows_dev = nullptr;
    void *fake_hccl_ctx_dev = nullptr;
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;

    size_t rows = static_cast<size_t>(FFN_GRID_ROWS);
    size_t cols = static_cast<size_t>(FFN_GRID_COLS);
    size_t cells = static_cast<size_t>(FFN_GRID_ROWS) * static_cast<size_t>(FFN_GRID_COLS);
    size_t xBytes = 0;
    size_t wGateBytes = 0;
    size_t wUpBytes = 0;
    size_t wDownBytes = 0;
    size_t gatePartialBytes = 0;
    size_t upPartialBytes = 0;
    size_t hiddenBytes = 0;
    size_t downPartialBytes = 0;
    size_t gatherScratchBytes = 0;
    size_t yOutputBytes = 0;
    size_t windowBytes = 0;

    std::string dataDir = "./out";
};

static bool ParseDeviceIdValue(const char *value, int &deviceId)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    deviceId = static_cast<int>(parsed);
    return true;
}

static bool ParseDeviceIdEnv(const char *name, int &deviceId)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    if (!ParseDeviceIdValue(value, deviceId)) {
        std::cerr << "[WARN] ignoring invalid " << name << "=" << value << std::endl;
        return false;
    }
    return true;
}

static int GetDeviceId(int argc, char **argv)
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
        constexpr const char *kPrefix = "--device-id=";
        constexpr size_t kPrefixLen = 12;
        if (std::strncmp(argv[i], kPrefix, kPrefixLen) == 0) {
            if (!ParseDeviceIdValue(argv[i] + kPrefixLen, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return deviceId;
        }
    }

    if (ParseDeviceIdEnv("FFN_GRID_DEVICE_ID", deviceId) || ParseDeviceIdEnv("ASCEND_DEVICE_ID", deviceId) ||
        ParseDeviceIdEnv("DEVICE_ID", deviceId)) {
        return deviceId;
    }
    return 0;
}

static bool ShouldUseRtSetDevice()
{
    const char *value = std::getenv("FFN_GRID_USE_RT_SET_DEVICE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool InitAcl(int device_id)
{
    constexpr int kAclRepeatInit = 100002;
    std::cout << "[INFO] aclInit begin" << std::endl;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    std::cout << "[INFO] aclInit done rc=" << static_cast<int>(aRet) << std::endl;

    if (ShouldUseRtSetDevice()) {
        std::cout << "[INFO] rtSetDevice(" << device_id << ") begin" << std::endl;
        rtError_t rtRet = rtSetDevice(device_id);
        if (rtRet != RT_ERROR_NONE) {
            std::cerr << "[ERROR] rtSetDevice(" << device_id << ") failed: " << static_cast<int>(rtRet) << std::endl;
            return false;
        }
        std::cout << "[INFO] rtSetDevice(" << device_id << ") done" << std::endl;
    } else {
        std::cout << "[INFO] rtSetDevice skipped; set FFN_GRID_USE_RT_SET_DEVICE=1 to enable" << std::endl;
    }

    std::cout << "[INFO] aclrtSetDevice(" << device_id << ") begin" << std::endl;
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(" << device_id << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    std::cout << "[INFO] aclrtSetDevice(" << device_id << ") done" << std::endl;
    return true;
}

// Build the single-device fake HcclDeviceContext (see main_comm_reducesum.cpp).
static bool InitFakeHcclContext(DeviceResources &r)
{
    r.windowBytes = r.cells * static_cast<size_t>(FFN_GRID_WINDOW_BYTES);
    if (aclrtMalloc(&r.windows_dev, r.windowBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(windows) failed" << std::endl;
        return false;
    }
    aclrtMemset(r.windows_dev, r.windowBytes, 0, r.windowBytes);

    HcclDeviceContext hostCtx{};
    hostCtx.rankId = 0;
    hostCtx.rankNum = static_cast<uint32_t>(r.cells);
    hostCtx.winSize = static_cast<uint64_t>(FFN_GRID_WINDOW_BYTES);
    uint64_t base = reinterpret_cast<uint64_t>(r.windows_dev);
    for (size_t i = 0; i < r.cells && i < HCCL_MAX_RANK_NUM; ++i) {
        hostCtx.windowsIn[i] = base + i * static_cast<size_t>(FFN_GRID_WINDOW_BYTES);
        hostCtx.windowsOut[i] = hostCtx.windowsIn[i];
    }

    if (aclrtMalloc(&r.fake_hccl_ctx_dev, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(fake_hccl_ctx) failed" << std::endl;
        return false;
    }
    if (aclrtMemcpy(r.fake_hccl_ctx_dev, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(fake_hccl_ctx) failed" << std::endl;
        return false;
    }
    return true;
}

static bool AllocateResources(DeviceResources &r)
{
    if (r.cells == 0 || r.cells > HCCL_MAX_RANK_NUM) {
        std::cerr << "[ERROR] invalid cell count " << r.cells << "; supported range is 1.." << HCCL_MAX_RANK_NUM
                  << std::endl;
        return false;
    }
    if (FFN_MODEL_TILE % FFN_GRID_COLS != 0) {
        std::cerr << "[ERROR] AllGather split requires MODEL_TILE divisible by GRID_COLS" << std::endl;
        return false;
    }
    if (const char *env = std::getenv("FFN_GRID_DATA_DIR")) {
        r.dataDir = env;
    }

    if (aclrtCreateStream(&r.stream) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed" << std::endl;
        return false;
    }

    r.xBytes = r.cells * static_cast<size_t>(FFN_X_BYTES);
    r.wGateBytes = r.cells * static_cast<size_t>(FFN_W_GATE_BYTES);
    r.wUpBytes = r.cells * static_cast<size_t>(FFN_W_UP_BYTES);
    r.wDownBytes = r.cells * static_cast<size_t>(FFN_W_DOWN_BYTES);
    r.gatePartialBytes = r.cells * static_cast<size_t>(FFN_GATE_PARTIAL_BYTES);
    r.upPartialBytes = r.cells * static_cast<size_t>(FFN_UP_PARTIAL_BYTES);
    r.hiddenBytes = r.cells * static_cast<size_t>(FFN_HIDDEN_FULL_BYTES);
    r.downPartialBytes = r.cells * static_cast<size_t>(FFN_DOWN_PARTIAL_BYTES);
    r.gatherScratchBytes = r.cells * static_cast<size_t>(FFN_HIDDEN_FULL_BYTES);
    r.yOutputBytes = r.rows * static_cast<size_t>(FFN_Y_OUTPUT_BYTES);

    aclrtMalloc(&r.x_dev, r.xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.w_gate_dev, r.wGateBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.w_up_dev, r.wUpBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.w_down_dev, r.wDownBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.gate_partial_dev, r.gatePartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.up_partial_dev, r.upPartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.hidden_dev, r.hiddenBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.down_partial_dev, r.downPartialBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.gather_scratch_dev, r.gatherScratchBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&r.y_output_dev, r.yOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!r.x_dev || !r.w_gate_dev || !r.w_up_dev || !r.w_down_dev || !r.gate_partial_dev || !r.up_partial_dev ||
        !r.hidden_dev || !r.down_partial_dev || !r.gather_scratch_dev || !r.y_output_dev) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }

    aclrtMemset(r.x_dev, r.xBytes, 0, r.xBytes);
    aclrtMemset(r.gate_partial_dev, r.gatePartialBytes, 0, r.gatePartialBytes);
    aclrtMemset(r.up_partial_dev, r.upPartialBytes, 0, r.upPartialBytes);
    aclrtMemset(r.hidden_dev, r.hiddenBytes, 0, r.hiddenBytes);
    aclrtMemset(r.down_partial_dev, r.downPartialBytes, 0, r.downPartialBytes);
    aclrtMemset(r.gather_scratch_dev, r.gatherScratchBytes, 0, r.gatherScratchBytes);
    aclrtMemset(r.y_output_dev, r.yOutputBytes, 0, r.yOutputBytes);

    if (!InitFakeHcclContext(r)) {
        return false;
    }

    rtGetC2cCtrlAddr(&r.ffts, &r.fftsLen);
    if (r.ffts == 0) {
        std::cerr << "[ERROR] rtGetC2cCtrlAddr returned null FFTS address" << std::endl;
        return false;
    }

    std::cout << "[INFO] grid=" << r.rows << "x" << r.cols << " cells=" << r.cells << " dataDir=" << r.dataDir
              << " windowBytes=" << r.windowBytes << " yOutputBytes=" << r.yOutputBytes << std::endl;
    return true;
}

static bool LoadInputs(DeviceResources &r)
{
    std::vector<uint8_t> hostX(r.xBytes);
    for (size_t cell = 0; cell < r.cells; ++cell) {
        std::string xPath = r.dataDir + "/pe_" + std::to_string(cell) + "_x.bin";
        size_t fileSize = 0;
        uint8_t *dst = hostX.data() + cell * static_cast<size_t>(FFN_X_BYTES);
        if (!PtoTestCommon::ReadFile(xPath, fileSize, dst, static_cast<size_t>(FFN_X_BYTES)) ||
            fileSize != static_cast<size_t>(FFN_X_BYTES)) {
            std::cerr << "[ERROR] X file load mismatch: " << xPath << " (got " << fileSize << " bytes, expected "
                      << FFN_X_BYTES << ")" << std::endl;
            return false;
        }
    }
    if (aclrtMemcpy(r.x_dev, r.xBytes, hostX.data(), r.xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(x_dev) failed" << std::endl;
        return false;
    }
    return true;
}

static bool LoadWeights(DeviceResources &r)
{
    struct WeightSpec {
        const char *suffix;
        void *dev;
        size_t tileBytes;
        size_t totalBytes;
    };
    WeightSpec specs[] = {
        {"_w_gate.bin", r.w_gate_dev, static_cast<size_t>(FFN_W_GATE_BYTES), r.wGateBytes},
        {"_w_up.bin", r.w_up_dev, static_cast<size_t>(FFN_W_UP_BYTES), r.wUpBytes},
        {"_w_down.bin", r.w_down_dev, static_cast<size_t>(FFN_W_DOWN_BYTES), r.wDownBytes},
    };

    for (const auto &w : specs) {
        std::vector<uint8_t> hostBuf(w.totalBytes);
        for (size_t cell = 0; cell < r.cells; ++cell) {
            std::string path = r.dataDir + "/pe_" + std::to_string(cell) + w.suffix;
            size_t fileSize = 0;
            uint8_t *dst = hostBuf.data() + cell * w.tileBytes;
            if (!PtoTestCommon::ReadFile(path, fileSize, dst, w.tileBytes) || fileSize != w.tileBytes) {
                std::cerr << "[ERROR] weight load mismatch: " << path << " (got " << fileSize << " bytes, expected "
                          << w.tileBytes << ")" << std::endl;
                return false;
            }
        }
        if (aclrtMemcpy(w.dev, w.totalBytes, hostBuf.data(), w.totalBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemcpy(weight " << w.suffix << ") failed" << std::endl;
            return false;
        }
    }
    return true;
}

static bool VerifyOutput(DeviceResources &r)
{
    const size_t outputElems = r.rows * static_cast<size_t>(FFN_TILE_ELEMS);
    const size_t outputBytes = r.rows * static_cast<size_t>(FFN_Y_OUTPUT_BYTES);

    std::vector<float> outHost(outputElems);
    if (aclrtMemcpy(outHost.data(), outputBytes, r.y_output_dev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        std::cerr << "[ERROR] y_output D2H memcpy failed" << std::endl;
        return false;
    }

    std::string goldenPath = r.dataDir + "/golden.bin";
    std::vector<float> golden(outputElems);
    size_t fileSize = 0;
    if (!PtoTestCommon::ReadFile(goldenPath, fileSize, golden.data(), outputBytes) || fileSize != outputBytes) {
        std::cerr << "[ERROR] golden.bin mismatch: " << goldenPath << " (got " << fileSize << " bytes, expected "
                  << outputBytes << ")" << std::endl;
        return false;
    }

    std::cout << "[INFO] ResultCmp single-device pto::comm TGATHER output shards vs golden:" << std::endl;
    return PtoTestCommon::ResultCmp(golden, outHost.data(), 0.001f);
}

static void Cleanup(DeviceResources &r)
{
    void *buffers[] = {r.fake_hccl_ctx_dev, r.windows_dev,      r.w_gate_dev,       r.w_up_dev,
                       r.w_down_dev,        r.x_dev,            r.gate_partial_dev, r.up_partial_dev,
                       r.hidden_dev,        r.down_partial_dev, r.gather_scratch_dev, r.y_output_dev};
    for (void *p : buffers) {
        if (p) {
            aclrtFree(p);
        }
    }
    r.fake_hccl_ctx_dev = r.windows_dev = r.w_gate_dev = r.w_up_dev = r.w_down_dev = nullptr;
    r.x_dev = r.gate_partial_dev = r.up_partial_dev = r.hidden_dev = nullptr;
    r.down_partial_dev = r.gather_scratch_dev = r.y_output_dev = nullptr;
    if (r.stream) {
        aclrtDestroyStream(r.stream);
        r.stream = nullptr;
    }
}

static bool RunSingleDevice()
{
    DeviceResources r;
    if (!AllocateResources(r) || !LoadInputs(r) || !LoadWeights(r)) {
        Cleanup(r);
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Phase 1: each cell computes its fp16 [T,Fi] hidden shard and stores it into
    // its window slot.
    launchFfnCommAllGatherComputeKernel(
        reinterpret_cast<uint8_t *>(r.ffts), reinterpret_cast<uint8_t *>(r.windows_dev),
        reinterpret_cast<uint8_t *>(r.x_dev), reinterpret_cast<uint8_t *>(r.w_gate_dev),
        reinterpret_cast<uint8_t *>(r.w_up_dev), reinterpret_cast<uint8_t *>(r.gate_partial_dev),
        reinterpret_cast<uint8_t *>(r.up_partial_dev), FFN_GRID_ROWS, FFN_GRID_COLS, r.stream);
    aclError computeRet = aclrtSynchronizeStream(r.stream); // global barrier before the collective reads windows

    // Phase 2: every cell TGATHERs the row's hidden shards, re-lays-out, and
    // down-projects its own output H shard.
    launchFfnCommAllGatherCollectiveKernel(
        reinterpret_cast<uint8_t *>(r.ffts), reinterpret_cast<uint8_t *>(r.windows_dev),
        reinterpret_cast<uint8_t *>(r.gather_scratch_dev), reinterpret_cast<uint8_t *>(r.hidden_dev),
        reinterpret_cast<uint8_t *>(r.down_partial_dev), reinterpret_cast<uint8_t *>(r.w_down_dev),
        reinterpret_cast<uint8_t *>(r.y_output_dev), reinterpret_cast<uint8_t *>(r.fake_hccl_ctx_dev), FFN_GRID_ROWS,
        FFN_GRID_COLS, r.stream);
    aclError collectiveRet = aclrtSynchronizeStream(r.stream);

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "[INFO] launch+sync " << us << " us  (compute rc=" << static_cast<int>(computeRet)
              << " collective rc=" << static_cast<int>(collectiveRet) << ")" << std::endl;

    bool syncOk = (computeRet == ACL_SUCCESS) && (collectiveRet == ACL_SUCCESS);
    bool verifyOk = syncOk && VerifyOutput(r);
    Cleanup(r);
    return syncOk && verifyOk;
}

int main(int argc, char **argv)
{
    int deviceId = GetDeviceId(argc, argv);
    std::cout << "[INFO] using device " << deviceId << std::endl;
    if (!InitAcl(deviceId)) {
        return 1;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Single-device multi-block FFN pto::comm TGATHER demo" << std::endl;
    std::cout << "  grid=" << FFN_GRID_ROWS << "x" << FFN_GRID_COLS << " cells=" << (FFN_GRID_ROWS * FFN_GRID_COLS)
              << " tile=" << FFN_TOKEN_TILE << "x" << FFN_MODEL_TILE << " ffnTile=" << FFN_FFN_TILE << std::endl;
    std::cout << "  Mode: row=data-parallel, col=model-parallel; combine via pto::comm::TGATHER + sharded down GEMM"
              << std::endl;
    std::cout << "================================================================" << std::endl;

    bool ok = RunSingleDevice();
    std::cout << (ok ? "[SUCCESS] Single-device multi-block FFN pto::comm TGATHER PASS." :
                       "[FAILED] Single-device multi-block FFN pto::comm TGATHER FAILED.")
              << std::endl;
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ok ? 0 : 1;
}

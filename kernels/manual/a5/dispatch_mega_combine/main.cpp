/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int kDefaultAicNum = 28;
constexpr int kDefaultAivNum = 56;

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void* hostSrc = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0U) {
        return buffer;
    }
    if (aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    if (hostSrc != nullptr &&
        aclrtMemcpy(buffer.ptr, bytes, hostSrc, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0U) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    for (size_t i = 0; i < out.size(); ++i) {
        const size_t byteOffset = i * sizeof(uint16_t);
        out[i] = static_cast<uint16_t>(bytes[byteOffset]) | (static_cast<uint16_t>(bytes[byteOffset + 1U]) << 8U);
    }
    return out;
}

int ParseEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        throw std::invalid_argument("alignment must be non-zero");
    }
    return (value + alignment - 1U) / alignment * alignment;
}

uint64_t SwigluFullRowUbBytes(uint32_t n)
{
    auto alignUb = [](uint64_t value) { return AlignUp(value, 32U); };
    uint64_t bytes = 0;
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(int8_t));
    bytes += 2U * 32U;
    return bytes;
}

void ValidateConfiguration(const CaseConfig& cfg)
{
    if (cfg.world_size == 0U || cfg.world_size > 16U) {
        throw std::runtime_error("front cumsum requires world_size in [1, 16]");
    }
    if (cfg.expert_per_rank > MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS) {
        throw std::runtime_error(
            "dispatch V2C hard flag budget exceeded: expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
            " max=" + std::to_string(MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS));
    }
    if (cfg.k % 128U != 0U) {
        throw std::runtime_error("GMM1 requires K % 128 == 0");
    }
    if (cfg.n % 64U != 0U) {
        throw std::runtime_error("SwiGLU requires N % 64 == 0");
    }
    if (SwigluFullRowUbBytes(cfg.n) > AtlasA5::UB_SIZE) {
        throw std::runtime_error(
            "SwiGLU full-row UB capacity exceeded: ub_bytes=" + std::to_string(SwigluFullRowUbBytes(cfg.n)) +
            " max=" + std::to_string(AtlasA5::UB_SIZE));
    }
    if ((cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("combine/unpermute requires K * sizeof(float16) to be 32-byte aligned");
    }
    if (static_cast<uint64_t>(cfg.max_output_size) < static_cast<uint64_t>(cfg.m) * cfg.topk) {
        throw std::runtime_error("unpermute requires max_output_size >= M * topK");
    }
}

void ZeroDeviceBuffer(const DeviceBuffer& buffer, const char* name)
{
    if (buffer.bytes != 0U && aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void PrepareLaunchBuffers(
    const StandaloneRankRuntime& runtime, const DeviceBuffer& out, const DeviceBuffer& expertTokenNums,
    const DeviceBuffer& workspace)
{
    const uint64_t windowBytes = runtime.hccl.WindowClearBytes();
    void* window = runtime.hccl.WindowClearBase(static_cast<uint32_t>(runtime.hccl.rank_id));
    if (aclrtMemset(window, windowBytes, 0, windowBytes) != ACL_SUCCESS) {
        throw std::runtime_error("failed to zero HCCL window");
    }
    ZeroDeviceBuffer(out, "out buffer");
    ZeroDeviceBuffer(expertTokenNums, "expert_token_nums");
    ZeroDeviceBuffer(workspace, "workspace");
}

std::string BuildAccuracyReport(int rankId, const AccuracyReport& report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rankId << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err << " err=" << report.mismatch_count << "/" << report.err_threshold
       << " -> " << (report.pass ? "PASS" : "FAIL");
    return os.str();
}

void PrintOrderedByRank(int rankId, int worldSize, const std::string& text)
{
    for (int turn = 0; turn < worldSize; ++turn) {
        CommMpiBarrier();
        if (turn == rankId) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

bool RunOneRank(int rankId, int worldSize, const std::string& caseDir, const HcclRootInfo& rootInfo)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rankId, worldSize, rootInfo)) {
        return false;
    }

    bool ok = false;
    try {
        CaseConfig cfg = LoadCaseConfig(caseDir + "/case.json");
        if (cfg.world_size != static_cast<uint32_t>(worldSize)) {
            throw std::runtime_error("case world_size does not match MPI world size");
        }

        const int aicNum = ParseEnvInt("DISPATCH_MEGA_COMBINE_AIC_NUM", kDefaultAicNum);
        const int aivNum = ParseEnvInt("DISPATCH_MEGA_COMBINE_AIV_NUM", kDefaultAivNum);
        if (aicNum <= 0 || aivNum != aicNum * 2) {
            throw std::runtime_error("A5 requires positive AIC_NUM and AIV_NUM == AIC_NUM * 2");
        }
        cfg.aic_num = static_cast<uint32_t>(aicNum);
        cfg.aiv_num = static_cast<uint32_t>(aivNum);
        ValidateConfiguration(cfg);

        const RankFileSet files = BuildRankFileSet(caseDir, rankId);
        MegaMoeBuildResult build = BuildMegaMoeTiling(cfg, runtime);

        const std::vector<uint8_t> x = ReadBinaryFile(files.x);
        const std::vector<uint8_t> weight1 = ReadBinaryFile(files.weight1);
        const std::vector<uint8_t> weight2 = ReadBinaryFile(files.weight2);
        const std::vector<uint8_t> expertIdx = ReadBinaryFile(files.expert_idx);
        const std::vector<uint8_t> scale1 = ReadBinaryFile(files.scale1);
        const std::vector<uint8_t> scale2 = ReadBinaryFile(files.scale2);
        const std::vector<uint8_t> probs = ReadBinaryFile(files.probs);
        const std::vector<uint16_t> expectedOut = BytesToU16(ReadBinaryFile(files.expected_out));

        DeviceBuffer xDev = MakeDeviceBuffer(x.size(), x.data());
        DeviceBuffer weight1Dev = MakeDeviceBuffer(weight1.size(), weight1.data());
        DeviceBuffer weight2Dev = MakeDeviceBuffer(weight2.size(), weight2.data());
        DeviceBuffer expertIdxDev = MakeDeviceBuffer(expertIdx.size(), expertIdx.data());
        DeviceBuffer scale1Dev = MakeDeviceBuffer(scale1.size(), scale1.data());
        DeviceBuffer scale2Dev = MakeDeviceBuffer(scale2.size(), scale2.data());
        DeviceBuffer probsDev = MakeDeviceBuffer(probs.size(), probs.data());
        DeviceBuffer outDev = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
        DeviceBuffer expertTokenNumsDev = MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
        DeviceBuffer workspaceDev = MakeDeviceBuffer(build.workspace_bytes);
        DeviceBuffer tilingDev = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);

        uint64_t fftsAddr = 0;
        uint32_t fftsLen = 0;
        if (rtGetC2cCtrlAddr(&fftsAddr, &fftsLen) != 0) {
            fftsAddr = 0;
        }

        MegaMoeLaunchArgs args;
        args.ffts = reinterpret_cast<void*>(fftsAddr);
        args.x = xDev.ptr;
        args.weight1 = weight1Dev.ptr;
        args.weight2 = weight2Dev.ptr;
        args.expert_idx = expertIdxDev.ptr;
        args.scale1 = scale1Dev.ptr;
        args.scale2 = scale2Dev.ptr;
        args.probs = probsDev.ptr;
        args.out = outDev.ptr;
        args.expert_token_nums = expertTokenNumsDev.ptr;
        args.workspace = workspaceDev.ptr;
        args.tiling = tilingDev.ptr;
        args.block_dim = build.block_dim;
        args.start_sync = ParseEnvInt("DISPATCH_MEGA_COMBINE_START_SYNC", 0) != 0 ? 1U : 0U;

        PrepareLaunchBuffers(runtime, outDev, expertTokenNumsDev, workspaceDev);
        CommMpiBarrier();
        launchMegaMoe(args, runtime.compute_stream);
        if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
            throw std::runtime_error("stream sync failed");
        }
        CommMpiBarrier();

        std::vector<uint16_t> actualOut(static_cast<size_t>(cfg.m) * cfg.k);
        const size_t outputBytes = actualOut.size() * sizeof(uint16_t);
        if (aclrtMemcpy(actualOut.data(), outputBytes, outDev.ptr, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
            throw std::runtime_error("device->host output copy failed");
        }
        WriteBinaryFile(caseDir + "/output_rank" + std::to_string(rankId) + ".bin", actualOut.data(), outputBytes);

        const AccuracyReport report = CompareFp16File(expectedOut, actualOut, cfg.compare_atol, cfg.compare_rtol);
        ok = report.pass;
        PrintOrderedByRank(rankId, worldSize, BuildAccuracyReport(rankId, report));
    } catch (const std::exception& ex) {
        std::cerr << "rank=" << rankId << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rankId = CommMpiRank();
    const int worldSize = CommMpiSize();
    const char* caseDirEnv = std::getenv("DISPATCH_MEGA_COMBINE_CASE_DIR");
    const std::string caseDir = caseDirEnv == nullptr ? "../out" : caseDirEnv;

    if (aclInit(nullptr) != ACL_SUCCESS) {
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(rankId) != 0 || aclrtSetDevice(rankId) != ACL_SUCCESS) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo rootInfo{};
    if (rankId == 0 && HcclGetRootInfo(&rootInfo) != HCCL_SUCCESS) {
        aclrtResetDevice(rankId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rankId, worldSize, caseDir, rootInfo);

    CommMpiBarrier();
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}

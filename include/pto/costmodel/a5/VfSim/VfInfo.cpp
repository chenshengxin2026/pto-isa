/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/costmodel/a5/VfSim/VfInfo.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace vfsim {
namespace {

using ValueNameMap = std::unordered_map<std::string, std::string>;
using ReservedValueMap = std::unordered_map<std::string, bool>;

std::string lower(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::pair<std::string, std::string> conversionDtypes(const std::string& form)
{
    const std::size_t pos = form.find("_to_");
    if (pos == std::string::npos)
        return {"", ""};
    auto normalize = [](const std::string& dtype) {
        if (dtype == "f32")
            return std::string("fp32");
        if (dtype == "f16")
            return std::string("fp16");
        if (dtype == "s32")
            return std::string("int32");
        if (dtype == "u32")
            return std::string("uint32");
        return dtype;
    };
    return {normalize(form.substr(0, pos)), normalize(form.substr(pos + 4))};
}

std::string compactDtype(const std::string& dtype)
{
    if (dtype == "fp32")
        return "f32";
    if (dtype == "fp16")
        return "f16";
    if (dtype == "int32")
        return "s32";
    if (dtype == "uint32")
        return "u32";
    return dtype;
}

void registerValue(VfInfo& vfInfo, const std::string& valueId)
{
    if (vfInfo.values.find(valueId) != vfInfo.values.end())
        return;
    ValueInfo value;
    value.valueId = valueId;
    value.storage = inferValueStorage(valueId);
    vfInfo.values.emplace(valueId, std::move(value));
}

void registerInstValues(VfInfo& vfInfo, const ProgramInstNode& inst)
{
    for (const std::string& valueId : inst.src)
        registerValue(vfInfo, valueId);
    for (const std::string& valueId : inst.dst)
        registerValue(vfInfo, valueId);
}

void fillValueDtypes(
    VfInfo& vfInfo, const std::vector<std::string>& valueIds, const std::string& conversionDtype,
    const std::string& simpleForm)
{
    for (const std::string& valueId : valueIds) {
        ValueInfo& value = vfInfo.values.at(valueId);
        if (!value.dtype.empty())
            continue;
        if (!conversionDtype.empty())
            value.dtype = conversionDtype;
        else
            value.dtype = !simpleForm.empty() ? simpleForm : vfInfo.defaultDtype;
    }
}

void fillInstValueDtypes(VfInfo& vfInfo, const ProgramInstNode& inst)
{
    const auto [srcConversion, dstConversion] = conversionDtypes(inst.form);
    const std::string simpleForm = inst.form.find("_to_") == std::string::npos ? inst.form : "";
    fillValueDtypes(vfInfo, inst.src, srcConversion, simpleForm);
    fillValueDtypes(vfInfo, inst.dst, dstConversion, simpleForm);
}

std::string firstValueDtype(const VfInfo& vfInfo, const std::vector<std::string>& valueIds)
{
    if (valueIds.empty())
        return "";
    return vfInfo.values.at(valueIds.front()).dtype;
}

std::string inferInstForm(const VfInfo& vfInfo, const ProgramInstNode& inst)
{
    const std::string srcDtype = firstValueDtype(vfInfo, inst.src);
    const std::string dstDtype = firstValueDtype(vfInfo, inst.dst);
    if (!srcDtype.empty() && !dstDtype.empty() && srcDtype != dstDtype)
        return compactDtype(srcDtype) + "_to_" + compactDtype(dstDtype);
    if (!dstDtype.empty())
        return dstDtype;
    if (!srcDtype.empty())
        return srcDtype;
    return vfInfo.defaultDtype;
}

void canonicalizeInstNode(ProgramInstNode& inst, VfInfo& vfInfo)
{
    registerInstValues(vfInfo, inst);
    fillInstValueDtypes(vfInfo, inst);
    if (inst.form.empty())
        inst.form = inferInstForm(vfInfo, inst);
}

void canonicalizeNodes(std::vector<ProgramNode>& nodes, VfInfo& vfInfo)
{
    for (ProgramNode& node : nodes) {
        if (node.kind == ProgramNode::Kind::Loop) {
            if (!node.loop)
                throw std::runtime_error("VfInfo loop node has no body");
            canonicalizeNodes(node.loop->body, vfInfo);
            continue;
        }
        canonicalizeInstNode(node.inst, vfInfo);
    }
}

bool keepsStoragePrefix(const std::string& valueId, const ValueInfo& value)
{
    const std::string normalized = lower(valueId);
    return (value.storage == ValueStorageKind::Register && normalized.rfind("v", 0) == 0) ||
           (value.storage == ValueStorageKind::UB && normalized.rfind("mem", 0) == 0);
}

void reserveExistingLoweredNames(const VfInfo& vfInfo, ValueNameMap& names, ReservedValueMap& reserved)
{
    for (const auto& [valueId, value] : vfInfo.values) {
        if (!keepsStoragePrefix(valueId, value))
            continue;
        names[valueId] = valueId;
        reserved[valueId] = true;
    }
}

std::string nextLoweredName(const std::string& prefix, int64_t& next, ReservedValueMap& reserved)
{
    std::string candidate;
    do {
        candidate = prefix + std::to_string(next++);
    } while (reserved.find(candidate) != reserved.end());
    reserved[candidate] = true;
    return candidate;
}

void assignLoweredName(
    const std::string& valueId, const ValueInfo& value, ValueNameMap& names, ReservedValueMap& reserved,
    int64_t& nextRegister, int64_t& nextUb)
{
    switch (value.storage) {
        case ValueStorageKind::Register:
            names[valueId] = nextLoweredName("V", nextRegister, reserved);
            break;
        case ValueStorageKind::UB:
            names[valueId] = nextLoweredName("mem", nextUb, reserved);
            break;
        case ValueStorageKind::Scalar:
            names[valueId] = valueId;
            break;
    }
}

ValueNameMap buildLoweredNames(const VfInfo& vfInfo)
{
    ValueNameMap names;
    ReservedValueMap reserved;
    int64_t nextRegister = 0;
    int64_t nextUb = 0;
    reserveExistingLoweredNames(vfInfo, names, reserved);
    for (const auto& [valueId, value] : vfInfo.values) {
        if (names.find(valueId) == names.end())
            assignLoweredName(valueId, value, names, reserved, nextRegister, nextUb);
    }
    return names;
}

void rewriteNodeValueIds(std::vector<ProgramNode>& nodes, const ValueNameMap& names)
{
    for (ProgramNode& node : nodes) {
        if (node.kind == ProgramNode::Kind::Loop) {
            rewriteNodeValueIds(node.loop->body, names);
            continue;
        }
        for (std::string& valueId : node.inst.src)
            valueId = names.at(valueId);
        for (std::string& valueId : node.inst.dst)
            valueId = names.at(valueId);
    }
}

std::unordered_map<std::string, ValueInfo> buildLoweredValues(
    std::unordered_map<std::string, ValueInfo> values, const ValueNameMap& names)
{
    std::unordered_map<std::string, ValueInfo> loweredValues;
    for (auto& [valueId, value] : values) {
        value.valueId = names.at(valueId);
        loweredValues.emplace(value.valueId, std::move(value));
    }
    return loweredValues;
}

} // namespace

ValueStorageKind inferValueStorage(const std::string& valueId)
{
    const std::string normalized = lower(valueId);
    if (normalized.rfind("mem", 0) == 0)
        return ValueStorageKind::UB;
    if (normalized.rfind("v", 0) == 0)
        return ValueStorageKind::Register;
    return ValueStorageKind::Scalar;
}

std::string valueStorageName(ValueStorageKind storage)
{
    switch (storage) {
        case ValueStorageKind::Register:
            return "Register";
        case ValueStorageKind::UB:
            return "UB";
        case ValueStorageKind::Scalar:
            return "Scalar";
    }
    return "Scalar";
}

void canonicalizeVfInfo(VfInfo& vfInfo)
{
    if (vfInfo.defaultDtype.empty())
        vfInfo.defaultDtype = "fp32";
    for (auto& [valueId, value] : vfInfo.values) {
        if (value.valueId.empty())
            value.valueId = valueId;
        if (value.valueId != valueId)
            throw std::runtime_error("VfInfo value map key does not match valueId: " + valueId);
    }
    canonicalizeNodes(vfInfo.body, vfInfo);
}

void lowerVfInfoValueIds(VfInfo& vfInfo)
{
    canonicalizeVfInfo(vfInfo);
    const ValueNameMap names = buildLoweredNames(vfInfo);
    rewriteNodeValueIds(vfInfo.body, names);
    vfInfo.values = buildLoweredValues(std::move(vfInfo.values), names);
}

} // namespace vfsim

/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef VFSIM_NATIVE_VALUE_STORAGE_H
#define VFSIM_NATIVE_VALUE_STORAGE_H

#include "pto/costmodel/a5/VfSim/VfInfo.h"

#include <string>
#include <unordered_map>

namespace vfsim {

class ValueStorageLookup {
public:
    ValueStorageLookup() = default;
    explicit ValueStorageLookup(const std::unordered_map<std::string, ValueInfo>& values)
    {
        for (const auto& [key, value] : values) {
            const std::string id = value.valueId.empty() ? key : value.valueId;
            storageById_.emplace(id, value.storage);
        }
    }

    ValueStorageKind storageOf(const std::string& name) const
    {
        auto it = storageById_.find(name);
        if (it != storageById_.end())
            return it->second;

        const std::string suffix = "_lane";
        const auto pos = name.rfind(suffix);
        if (pos != std::string::npos && pos + suffix.size() < name.size()) {
            bool digits = true;
            for (size_t i = pos + suffix.size(); i < name.size(); ++i) {
                if (name[i] < '0' || name[i] > '9') {
                    digits = false;
                    break;
                }
            }
            if (digits) {
                it = storageById_.find(name.substr(0, pos));
                if (it != storageById_.end())
                    return it->second;
            }
        }

        return inferValueStorage(name);
    }

    bool isRegister(const std::string& name) const { return storageOf(name) == ValueStorageKind::Register; }

    bool isUB(const std::string& name) const { return storageOf(name) == ValueStorageKind::UB; }

private:
    std::unordered_map<std::string, ValueStorageKind> storageById_;
};

} // namespace vfsim

#endif // VFSIM_NATIVE_VALUE_STORAGE_H

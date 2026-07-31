/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef VFSIM_NATIVE_JSON_DUMP_UTILS_H
#define VFSIM_NATIVE_JSON_DUMP_UTILS_H

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace vfsim {

inline std::string jsonEscape(const std::string& text)
{
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

template <typename T>
inline void appendJsonValue(std::ostringstream& oss, const T& value)
{
    oss << value;
}

inline void appendJsonValue(std::ostringstream& oss, const std::string& value)
{
    oss << '"' << jsonEscape(value) << '"';
}

inline void appendJsonValue(std::ostringstream& oss, const std::optional<std::string>& value)
{
    if (value.has_value())
        appendJsonValue(oss, *value);
    else
        oss << "null";
}

template <typename T>
inline std::string joinJsonArray(const std::vector<T>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ", ";
        appendJsonValue(oss, values[i]);
    }
    oss << "]";
    return oss.str();
}

} // namespace vfsim

#endif // VFSIM_NATIVE_JSON_DUMP_UTILS_H

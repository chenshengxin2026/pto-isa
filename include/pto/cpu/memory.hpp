/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_MEMORY_HPP
#define PTO_CPU_MEMORY_HPP

#if defined(__CPU_SIM)
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <pto/common/utils.hpp>

namespace pto::cpu {

#if defined(__CPU_SIM)
PTO_INTERNAL bool IsMappedAddress(const void* addr)
{
    // TLOAD uses this to avoid reading past mapped host memory in CPU-SIM.
    if (addr == nullptr) {
        return false;
    }
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return true;
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(addr);
    void* page = reinterpret_cast<void*>(raw & ~(static_cast<std::uintptr_t>(pageSize) - 1));
    unsigned char vec = 0;
    errno = 0;
    if (mincore(page, static_cast<std::size_t>(pageSize), &vec) == 0) {
        return true;
    }
    return errno != ENOMEM;
}
#endif

} // namespace pto::cpu

#endif

/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_LOG_HOST_LOG_H_
#define PTO_COMM_LOG_HOST_LOG_H_

#if defined(__CCE_KT_TEST__)
#error "host_log.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstring>
#include <sstream>

#include "pto/comm/logger/host_logger.hpp"

namespace pto::comm::hostlog {

inline int32_t PtoSetHostLogLevel(int32_t level) { return PtoHostLogger::Instance().SetLogLevel(level); }

inline int32_t PtoSetExternalLogger(ExternalLog logger) { return PtoHostLogger::Instance().SetExternalLogger(logger); }

} // namespace pto::comm::hostlog

#define PTO_HOST_LOG_FILENAME_SHORT (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

#define PTO_HOST_OUT_LOG(LEVEL, MODULE, ARGS)                                                                   \
    do {                                                                                                        \
        auto& ptoHostLogger_ = ::pto::comm::hostlog::PtoHostLogger::Instance();                                 \
        const auto ptoHostLogLevel_ = (LEVEL);                                                                  \
        if (ptoHostLogger_.ShouldLog(ptoHostLogLevel_)) {                                                       \
            const char* const ptoHostLogModule_ = (MODULE);                                                     \
            std::ostringstream ptoHostLogStream_;                                                               \
            ptoHostLogStream_ << "[PTO_HOST " << ptoHostLogModule_ << " " << PTO_HOST_LOG_FILENAME_SHORT << ":" \
                              << __LINE__ << "] " << ARGS;                                                      \
            ptoHostLogger_.Log(ptoHostLogLevel_, ptoHostLogStream_.str());                                      \
        }                                                                                                       \
    } while (0)

#define PTO_HOST_LOG_DEBUG(MODULE, ARGS) PTO_HOST_OUT_LOG(::pto::comm::hostlog::DEBUG_LEVEL, MODULE, ARGS)
#define PTO_HOST_LOG_INFO(MODULE, ARGS) PTO_HOST_OUT_LOG(::pto::comm::hostlog::INFO_LEVEL, MODULE, ARGS)
#define PTO_HOST_LOG_WARN(MODULE, ARGS) PTO_HOST_OUT_LOG(::pto::comm::hostlog::WARN_LEVEL, MODULE, ARGS)
#define PTO_HOST_LOG_ERROR(MODULE, ARGS) PTO_HOST_OUT_LOG(::pto::comm::hostlog::ERROR_LEVEL, MODULE, ARGS)

#endif // PTO_COMM_LOG_HOST_LOG_H_

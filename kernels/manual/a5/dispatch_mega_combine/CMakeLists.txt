# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER "bisheng" CACHE STRING "Compiler used for host and CCE sources")
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${CMAKE_COMPILER}")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CMAKE_COMPILER}")
endif()

project(pto_dispatch_mega_combine_a5 LANGUAGES C CXX)

# bisheng/cce-ld does not support the linker depfile option that CMake 3.27+ may enable.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
    )

# A5 (arch35 / dav-c310) build: route PTO headers to npu/a5 and match CANN host ABI.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A5)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CCE_AICORE_ARCH "dav-c310" CACHE STRING "CCE AI Core architecture for this A5 kernel")
set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

get_filename_component(PTO_REPO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../../../include" ABSOLUTE)

if(EXISTS ${ASCEND_HOME_PATH}/x86_64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/aarch64-linux)
else()
    message(FATAL_ERROR
        "Cannot find Ascend kernel include root under ${ASCEND_HOME_PATH}/x86_64-linux or aarch64-linux")
endif()

# Host extra include dirs for hccl_tiling (path differs across CANN packages).
set(HOST_EXTRA_INCLUDE_DIRS)
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_EXTRA_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
endif()

include_directories(
    ${PTO_REPO_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/experiment/runtime
)

add_library(dispatch_mega_combine_kernel SHARED kernel_launch.cpp)
target_compile_options(dispatch_mega_combine_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=${CCE_AICORE_ARCH}
)
target_include_directories(dispatch_mega_combine_kernel BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/op_kernel
    ${PROJECT_SOURCE_DIR}/op_kernel/utils
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc/runtime/runtime
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
)
target_link_options(dispatch_mega_combine_kernel PRIVATE --cce-fatobj-link)

add_executable(dispatch_mega_combine
    main.cpp
    runtime_context.cpp
    tiling_builder.cpp
    data_utils.cpp
)
target_compile_options(dispatch_mega_combine PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(dispatch_mega_combine BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/op_kernel
    ${PROJECT_SOURCE_DIR}/op_kernel/utils
    ${ASCEND_HOME_PATH}/include/hccl
    ${HOST_EXTRA_INCLUDE_DIRS}
)

target_link_directories(dispatch_mega_combine PUBLIC
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(dispatch_mega_combine PRIVATE
    dispatch_mega_combine_kernel
    stdc++ ascendcl hcomm runtime m tiling_api register platform c_sec dl nnopbase pthread
)

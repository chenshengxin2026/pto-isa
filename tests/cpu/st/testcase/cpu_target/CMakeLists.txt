# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
pto_cpu_sim_st(cpu_target)

foreach(family IN ITEMS a2a3 a5 legacy)
    add_library(cpu_target_${family}_object OBJECT probe.cpp)
    target_include_directories(cpu_target_${family}_object PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
    target_compile_definitions(cpu_target_${family}_object PRIVATE PTO_TARGET_TEST_SYMBOL=probe_${family} NDEBUG)
    target_compile_options(cpu_target_${family}_object PRIVATE -O2)
    if(NOT family STREQUAL "legacy")
        string(TOUPPER "${family}" family_macro)
        target_compile_definitions(cpu_target_${family}_object PRIVATE PTO_CPU_SIM_TARGET_${family_macro})
        add_library(cpu_target_${family}_shared SHARED probe.cpp)
        target_include_directories(cpu_target_${family}_shared PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
        target_compile_definitions(cpu_target_${family}_shared PRIVATE
            PTO_TARGET_TEST_SYMBOL=probe_dynamic_${family} PTO_CPU_SIM_TARGET_${family_macro} NDEBUG)
        target_compile_options(cpu_target_${family}_shared PRIVATE -O2)
        target_link_libraries(cpu_target_${family}_shared PRIVATE Threads::Threads)
        target_compile_definitions(cpu_target PRIVATE
            PTO_TARGET_LIB_${family_macro}="$<TARGET_FILE:cpu_target_${family}_shared>")
        add_dependencies(cpu_target cpu_target_${family}_shared)
    endif()
    target_sources(cpu_target PRIVATE $<TARGET_OBJECTS:cpu_target_${family}_object>)
endforeach()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${PROJECT_SOURCE_DIR}/../../../include)
set(CMAKE_REQUIRED_DEFINITIONS -D__CPU_SIM -DPTO_CPU_SIM_TARGET_A5)
check_cxx_source_compiles("#include <pto/cpu/target_config.hpp>
int main() {}" PTO_CPU_TARGET_VALID)
if(NOT PTO_CPU_TARGET_VALID)
    message(FATAL_ERROR "Valid CPU target configuration did not compile")
endif()
foreach(invalid IN ITEMS BOTH NON_CPU NPU_ARCH NPU_FAMILY)
    if(invalid STREQUAL "BOTH")
        set(CMAKE_REQUIRED_DEFINITIONS -D__CPU_SIM -DPTO_CPU_SIM_TARGET_A5 -DPTO_CPU_SIM_TARGET_A2A3)
    elseif(invalid STREQUAL "NON_CPU")
        set(CMAKE_REQUIRED_DEFINITIONS -DPTO_CPU_SIM_TARGET_A5)
    elseif(invalid STREQUAL "NPU_ARCH")
        set(CMAKE_REQUIRED_DEFINITIONS -D__CPU_SIM -DPTO_CPU_SIM_TARGET_A5 -D__NPU_ARCH__=3101)
    else()
        set(CMAKE_REQUIRED_DEFINITIONS -D__CPU_SIM -DPTO_CPU_SIM_TARGET_A5 -DPTO_NPU_ARCH_A5)
    endif()
    check_cxx_source_compiles("#include <pto/cpu/target_config.hpp>
int main() {}" PTO_CPU_TARGET_ACCEPTS_${invalid})
    if(PTO_CPU_TARGET_ACCEPTS_${invalid})
        message(FATAL_ERROR "Invalid CPU target configuration ${invalid} was accepted")
    endif()
endforeach()
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_DEFINITIONS)

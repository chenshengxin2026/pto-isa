# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tcvt_fp4_lut)
# Preserve generated device assembly for the register LUT/spill review.
target_compile_options(tcvt_fp4_lut_kernel PRIVATE -save-temps=obj)

# Build compile-only type/allocation checks even when only the ST executable is requested.
# Keep the explicit and automatic guards separate from the runnable kernel.
set(ROWPLUSONE_CCE_OPTIONS ${CMAKE_CCE_COMPILE_OPTIONS})
list(REMOVE_ITEM ROWPLUSONE_CCE_OPTIONS --cce-pto-auto-enable --cce-enable-pto-passes)
# Match the parent directory and probe the same compiler command used for the guards.
set(ROWPLUSONE_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
execute_process(
    COMMAND "${ROWPLUSONE_CXX_COMPILER}" --help
    RESULT_VARIABLE ROWPLUSONE_COMPILER_HELP_RESULT
    OUTPUT_VARIABLE ROWPLUSONE_COMPILER_HELP_OUT
    ERROR_VARIABLE ROWPLUSONE_COMPILER_HELP_ERR)
set(ROWPLUSONE_COMPILER_HELP "${ROWPLUSONE_COMPILER_HELP_OUT}\n${ROWPLUSONE_COMPILER_HELP_ERR}")
set(ROWPLUSONE_AUTO_OPTION "")
if(ROWPLUSONE_COMPILER_HELP MATCHES "--cce-enable-pto-passes")
    set(ROWPLUSONE_AUTO_OPTION --cce-enable-pto-passes)
elseif(ROWPLUSONE_COMPILER_HELP MATCHES "--cce-pto-auto-enable")
    set(ROWPLUSONE_AUTO_OPTION --cce-pto-auto-enable)
endif()
foreach(MODE manual auto)
    if(MODE STREQUAL "auto" AND NOT ROWPLUSONE_AUTO_OPTION)
        message(WARNING
            "Skipping the RowPlusOne AUTO compile guard because neither AUTO option was detected; "
            "compiler help exited with ${ROWPLUSONE_COMPILER_HELP_RESULT}")
        continue()
    endif()
    set(GUARD_TARGET tcvt_fp4_rowplusone_${MODE}_compile)
    add_library(${GUARD_TARGET} OBJECT rowplusone_compile.cpp)
    target_compile_options(${GUARD_TARGET} PRIVATE ${ROWPLUSONE_CCE_OPTIONS}
        --cce-aicore-arch=dav-c310-vec --cce-aicore-only -DREGISTER_BASE -std=c++17)
    if(MODE STREQUAL "auto")
        target_compile_definitions(${GUARD_TARGET} PRIVATE __PTO_AUTO__)
        target_compile_options(${GUARD_TARGET} PRIVATE ${ROWPLUSONE_AUTO_OPTION})
    endif()
    add_dependencies(tcvt_fp4_lut_kernel ${GUARD_TARGET})
endforeach()

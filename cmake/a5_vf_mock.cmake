
set(_pto_a5_self_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_pto_a5_repo_dir "${CMAKE_CURRENT_LIST_DIR}/..")

function(_pto_a5_build_vfsim)
    if(TARGET pto_a5_vfsim)
        return()
    endif()
    set(_vfsim_dir "${_pto_a5_repo_dir}/include/pto/costmodel/a5/VfSim")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
            "${_pto_a5_repo_dir}/include/pto/costmodel/a5/formula_costmodel/gen_formula_params_header.py"
        WORKING_DIRECTORY "${_pto_a5_repo_dir}"
        RESULT_VARIABLE _a5_formula_param_gen_result)
    if(NOT _a5_formula_param_gen_result EQUAL 0)
        message(FATAL_ERROR "a5_vf_mock: failed to generate a5 formula_params_generated.hpp")
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${_vfsim_dir}/gen_vfsim_params_header.py"
        WORKING_DIRECTORY "${_pto_a5_repo_dir}"
        RESULT_VARIABLE _vfsim_param_gen_result)
    if(NOT _vfsim_param_gen_result EQUAL 0)
        message(FATAL_ERROR "a5_vf_mock: failed to generate VfSimParamsGenerated.h")
    endif()
    add_library(pto_a5_vfsim STATIC
        ${_vfsim_dir}/VfInfo.cpp
        ${_vfsim_dir}/ParamDB.cpp
        ${_vfsim_dir}/ISATraits.cpp
        ${_vfsim_dir}/ProgramAnalysis.cpp
        ${_vfsim_dir}/ProgramCanonicalization.cpp
        ${_vfsim_dir}/ProgramVregLiveRangeNormalization.cpp
        ${_vfsim_dir}/ProgramFlatten.cpp
        ${_vfsim_dir}/IFU.cpp
        ${_vfsim_dir}/IDU.cpp
        ${_vfsim_dir}/OOO.cpp
        ${_vfsim_dir}/SimulatorRunner.cpp
        ${_vfsim_dir}/VfSimCostModel.cpp)
    target_include_directories(pto_a5_vfsim PUBLIC "${_pto_a5_repo_dir}/include")
    target_compile_features(pto_a5_vfsim PUBLIC cxx_std_20)
    target_compile_definitions(pto_a5_vfsim PRIVATE PTO_VFSIM_SOURCE_ROOT="${_vfsim_dir}")
endfunction()

macro(_pto_a5_find_llvm)
    if(NOT PTO_A5_LLVM_CONFIG)
        if(DEFINED ENV{LLVM_CONFIG} AND EXISTS "$ENV{LLVM_CONFIG}")
            set(PTO_A5_LLVM_CONFIG "$ENV{LLVM_CONFIG}")
        else()
            file(GLOB _pto_a5_bindirs /usr/lib/llvm-*/bin
                 /usr/local/opt/llvm@*/bin /usr/local/opt/llvm/bin)
            find_program(PTO_A5_LLVM_CONFIG
                NAMES llvm-config llvm-config-20 llvm-config-19 llvm-config-18
                      llvm-config-17 llvm-config-16 llvm-config-15 llvm-config-14
                PATHS ${_pto_a5_bindirs} /usr/local/bin /usr/bin)
        endif()
    endif()
    if(NOT PTO_A5_LLVM_CONFIG)
        message(FATAL_ERROR "a5_vf_mock: llvm-config not found. Mac: brew install llvm@18; "
            "Linux: sudo apt install clang-N llvm-N-dev (N>=14), or set LLVM_CONFIG env.")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --version
        OUTPUT_VARIABLE PTO_A5_LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --bindir
        OUTPUT_VARIABLE PTO_A5_LLVM_BINDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "^([0-9]+)" PTO_A5_LLVM_MAJOR "${PTO_A5_LLVM_VERSION}")
    if(NOT PTO_A5_CLANGXX OR NOT EXISTS "${PTO_A5_CLANGXX}")
        unset(PTO_A5_CLANGXX CACHE)
        find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++
            PATHS ${PTO_A5_LLVM_BINDIR} /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(NOT PTO_A5_CLANGXX)
            find_program(PTO_A5_CLANGXX NAMES clang++-${PTO_A5_LLVM_MAJOR} clang++)
        endif()
    endif()
    if(NOT PTO_A5_CLANGXX)
        message(FATAL_ERROR "a5_vf_mock: matching clang++ for LLVM ${PTO_A5_LLVM_MAJOR} not found")
    endif()
    execute_process(COMMAND ${PTO_A5_LLVM_CONFIG} --cxxflags
        OUTPUT_VARIABLE PTO_A5_LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(PTO_A5_LLVM_CXXFLAGS NATIVE_COMMAND "${PTO_A5_LLVM_CXXFLAGS}")
    message(STATUS "a5_vf_mock: LLVM ${PTO_A5_LLVM_VERSION} (${PTO_A5_LLVM_CONFIG})")
endmacro()

macro(_pto_a5_build_pass)
    if(NOT TARGET PtoLoopTracePass)
    _pto_a5_find_llvm()
    set(_pass_src "${_pto_a5_self_dir}/../include/pto/costmodel/a5/PtoLoopTracePass.cpp")
    add_library(PtoLoopTracePass MODULE ${_pass_src})
    target_compile_options(PtoLoopTracePass PRIVATE ${PTO_A5_LLVM_CXXFLAGS}
        -Wno-unused-command-line-argument -Wno-unknown-warning-option)
    if(APPLE)
        file(GLOB _sdks "/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
        set(_sdk "")
        foreach(s IN LISTS _sdks)
            if(EXISTS "${s}/usr/include/c++/v1" AND EXISTS "${s}/usr/lib/libc++.tbd")
                get_filename_component(_n "${s}" NAME)
                string(REGEX MATCH "MacOSX([0-9]+\\.[0-9]+)\\.sdk" _ "${_n}")
                if(CMAKE_MATCH_1 AND CMAKE_MATCH_1 VERSION_LESS 16)
                    set(_sdk "${s}")
                endif()
            endif()
        endforeach()
        if(_sdk)
            target_compile_options(PtoLoopTracePass PRIVATE -isysroot ${_sdk})
            target_link_options(PtoLoopTracePass PRIVATE -undefined dynamic_lookup -nostdlib++
                -Wl,-syslibroot,${_sdk})
        else()
            target_link_options(PtoLoopTracePass PRIVATE -undefined dynamic_lookup -nostdlib++)
        endif()
    endif()
    endif()
endmacro()

function(target_enable_a5_vf_mock target)
    _pto_a5_build_pass()
    _pto_a5_build_vfsim()
    add_dependencies(${target} PtoLoopTracePass)
    target_link_libraries(${target} PRIVATE pto_a5_vfsim)
    target_compile_options(${target} PRIVATE
        -O0 -g -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>)
    message(STATUS "a5_vf_mock: ${target} enabled pass instrumentation (-O0 -g -fpass-plugin)")
endfunction()

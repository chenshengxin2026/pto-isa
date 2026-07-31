# st_a5 — A5 CCE mock 单 ISA 指令测试(对标 st/,但 __NPU_ARCH__=3101 + pass 插桩)。
# 验证 PTO 入口(TADD)→ __VEC_SCOPE__ → pass 拦截 for + stub rec → VF 结算 全链路。
#
# 独立 CMake 工程(走 LLVM pass,不经 run_costmodel.py 的 gtest 流程)。构建+跑测:
#   cmake -S tests/costmodel/st_a5 -B tests/costmodel/st_a5/build -DCMAKE_BUILD_TYPE=Debug
#   cmake --build tests/costmodel/st_a5/build -j 4
#   ctest --test-dir tests/costmodel/st_a5/build --output-on-failure
cmake_minimum_required(VERSION 3.18)

# 探测 clang18(须与 pass 同 LLVM),在 project() 前 pin 编译器。
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/a5_vf_mock.cmake)
_pto_a5_find_llvm()
set(CMAKE_CXX_COMPILER ${PTO_A5_CLANGXX})

# Mac 工具链坑(复用 a5_vf_mock.cmake 调通的表达):homebrew clang 默认 isysroot 坏 +
# conda ld 盖 Apple ld。选真 SDK(<16)给编译/链接,用 -B 强制 Apple ld。
if(APPLE)
    file(GLOB _sdks "/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
    set(_sdk "")
    set(_best 0)
    foreach(s IN LISTS _sdks)
        if(EXISTS "${s}/usr/include/c++/v1" AND EXISTS "${s}/usr/lib/libc++.tbd")
            get_filename_component(_n "${s}" NAME)
            string(REGEX MATCH "MacOSX([0-9]+\\.[0-9]+)\\.sdk" _ "${_n}")
            if(CMAKE_MATCH_1 AND CMAKE_MATCH_1 VERSION_LESS 16 AND CMAKE_MATCH_1 VERSION_GREATER _best)
                set(_best "${CMAKE_MATCH_1}")
                set(_sdk "${s}")
            endif()
        endif()
    endforeach()
    set(_cltb "/Library/Developer/CommandLineTools/usr/bin")
    set(CMAKE_OSX_SYSROOT "${_sdk}" CACHE PATH "" FORCE)
    set(CMAKE_CXX_FLAGS "-isysroot ${_sdk} -B${_cltb}" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-syslibroot,${_sdk}" CACHE STRING "" FORCE)
    set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-syslibroot,${_sdk}" CACHE STRING "" FORCE)
endif()

project(pto_test_costmodel_a5 CXX)
message(STATUS "st_a5: LLVM ${PTO_A5_LLVM_VERSION}, CXX=${PTO_A5_CLANGXX}")

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# 注册 CTest:st_a5 是独立 CMake 工程(走 LLVM pass,不经 run_costmodel.py 的 gtest 流程),
# 故在此 enable_testing 让 `ctest --test-dir <build>` 能发现用例(各 testcase 下 add_test)。
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)
add_compile_options(-Wno-macro-redefined -Wno-ignored-attributes
    "SHELL:-include stdint.h" "SHELL:-include stddef.h")

add_subdirectory(testcase)

function(pto_a5_vfsim_tileop_test NAME)
    get_filename_component(_repo_inc "${CMAKE_SOURCE_DIR}/../../../include" ABSOLUTE)
    add_executable(${NAME} main.cpp)
    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${_repo_inc})
    target_link_libraries(${NAME} PRIVATE
        GTest::gtest_main
        Threads::Threads)
    target_enable_a5_vf_mock(${NAME})
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

set(ALL_TESTCASES
    tadd
    tadds
    tsub
    tsubs
    tmul
    tmuls
    tdiv
    tmax
    tmin
    tmins
    tand
    tshls
    tshrs
)

foreach(TESTCASE ${ALL_TESTCASES})
    if(NOT DEFINED TEST_CASE OR TEST_CASE STREQUAL "${TESTCASE}")
        add_subdirectory(${TESTCASE})
    endif()
endforeach()

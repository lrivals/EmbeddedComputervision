cmake_minimum_required(VERSION 3.16)
project(yolo_hls_csim CXX)

# C-simulation du noyau HLS avec g++ (sans Vitis) : en-têtes ap_int open source, pragmas
# ignorés. Mêmes sources que hls/scripts/csim.tcl (T6.1-T6.4).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Tuiles : défauts KV260 (hls/configs/kv260.tcl).
set(ACC_TM 32 CACHE STRING "Tm")
set(ACC_TN 24 CACHE STRING "Tn")
set(ACC_TR 13 CACHE STRING "Tr")
set(ACC_TC 13 CACHE STRING "Tc")

include(FetchContent)
FetchContent_Declare(hls_ap_types
  GIT_REPOSITORY https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git
  GIT_TAG 200a9aecaadf471592558540dc5a88256cbf880f)
FetchContent_MakeAvailable(hls_ap_types)

add_subdirectory(../cpp/golden ${CMAKE_BINARY_DIR}/golden)

add_library(driver ../sw/driver/program.cpp)
target_include_directories(driver PUBLIC ../sw/driver kernels)
target_link_libraries(driver PUBLIC golden)
target_compile_options(driver PRIVATE -Wall -Wextra)

add_library(accel kernels/conv_pe.cpp)
target_include_directories(accel PUBLIC kernels)
target_include_directories(accel SYSTEM PUBLIC ${hls_ap_types_SOURCE_DIR}/include)
target_link_libraries(accel PUBLIC golden)
target_compile_definitions(accel PUBLIC ACC_TM=${ACC_TM} ACC_TN=${ACC_TN} ACC_TR=${ACC_TR}
                                        ACC_TC=${ACC_TC})
target_compile_options(accel PRIVATE -Wall -Wextra -Wno-unused-label)
# Faux positifs de ap_private.h (en-têtes 2019) avec g++ récent en -O3.
target_compile_options(accel PUBLIC -Wno-unknown-pragmas -Wno-maybe-uninitialized
                                    -Wno-uninitialized)

# Testbenchs : seulement en projet principal (sw/ réutilise golden, driver et accel).
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  get_filename_component(ACC_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model" ABSOLUTE)
  include(CTest)
  foreach(tb tb_conv tb_net)
    add_executable(${tb} tb/${tb}.cpp)
    target_link_libraries(${tb} PRIVATE accel driver)
    target_compile_definitions(${tb} PRIVATE ACC_MODEL_DIR="${ACC_MODEL_DIR}")
    add_test(NAME ${tb} COMMAND ${tb})
  endforeach()
endif()

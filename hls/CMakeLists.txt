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
# Options M10 (accel_config.hpp), ex. "ACC_WORD_BYTES=1;ACC_RQ=1" ; vide : défauts du fichier.
set(ACC_OPTIONS "" CACHE STRING "définitions supplémentaires du noyau")

include(FetchContent)
FetchContent_Declare(hls_ap_types
  GIT_REPOSITORY https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git
  GIT_TAG 200a9aecaadf471592558540dc5a88256cbf880f)
FetchContent_MakeAvailable(hls_ap_types)

add_subdirectory(../cpp/golden ${CMAKE_BINARY_DIR}/golden)

# Configuration du noyau, partagée par le noyau et le driver (ordre des poids, tuiles).
# Les ports sont lus en mots : accès int8 ↔ mots sur la même mémoire.
add_library(accel_config INTERFACE)
target_compile_definitions(accel_config INTERFACE ACC_TM=${ACC_TM} ACC_TN=${ACC_TN}
                           ACC_TR=${ACC_TR} ACC_TC=${ACC_TC} ${ACC_OPTIONS})
target_compile_options(accel_config INTERFACE -fno-strict-aliasing)

add_library(driver ../sw/driver/program.cpp)
target_include_directories(driver PUBLIC ../sw/driver kernels)
target_link_libraries(driver PUBLIC golden accel_config)
target_compile_options(driver PRIVATE -Wall -Wextra)

add_library(accel kernels/conv_pe.cpp kernels/postproc.cpp)
target_include_directories(accel PUBLIC kernels)
target_include_directories(accel SYSTEM PUBLIC ${hls_ap_types_SOURCE_DIR}/include)
target_link_libraries(accel PUBLIC golden accel_config)
target_compile_options(accel PRIVATE -Wall -Wextra -Wno-unused-label)
# Faux positifs de ap_private.h (en-têtes 2019) avec g++ récent en -O3.
target_compile_options(accel PUBLIC -Wno-unknown-pragmas -Wno-maybe-uninitialized
                                    -Wno-uninitialized)

# Architecture streaming (T9.4.2) : testbench, et backend sim du driver (T10.9).
add_library(accel_stream stream/yolo_stream.cpp)
target_include_directories(accel_stream PUBLIC stream)
target_link_libraries(accel_stream PUBLIC golden)
target_compile_options(accel_stream PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

# Testbenchs : seulement en projet principal (sw/ réutilise golden, driver et accel).
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  get_filename_component(ACC_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model" ABSOLUTE)
  include(CTest)
  foreach(tb tb_conv tb_net tb_post)
    add_executable(${tb} tb/${tb}.cpp)
    target_link_libraries(${tb} PRIVATE accel driver)
    target_compile_definitions(${tb} PRIVATE ACC_MODEL_DIR="${ACC_MODEL_DIR}")
    add_test(NAME ${tb} COMMAND ${tb})
  endforeach()

  # T9.4.2 : architecture streaming (Tiny-YOLOv2).
  add_executable(tb_stream tb/tb_stream.cpp)
  target_link_libraries(tb_stream PRIVATE accel_stream)
  target_compile_definitions(tb_stream PRIVATE ACC_MODEL_DIR="${ACC_MODEL_DIR}")
  target_compile_options(tb_stream PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
  add_test(NAME tb_stream COMMAND tb_stream --image 000001)

  # T10.8 : mêmes étages, poids sur la puce en ROM générée depuis l'export (gen_stream_rom.py).
  find_package(Python3 COMPONENTS Interpreter)
  set(STREAM_ROM_MODEL "${ACC_MODEL_DIR}/tiny-yolov2-voc")
  if(Python3_FOUND AND EXISTS "${STREAM_ROM_MODEL}/manifest.json")
    set(STREAM_ROM_DIR "${CMAKE_CURRENT_BINARY_DIR}/stream_rom")
    add_custom_command(
      OUTPUT ${STREAM_ROM_DIR}/stream_rom.hpp ${STREAM_ROM_DIR}/stream_rom.cpp
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_stream_rom.py
              --model ${STREAM_ROM_MODEL} --out ${STREAM_ROM_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_stream_rom.py
              ${CMAKE_CURRENT_SOURCE_DIR}/../tools/stream_model.py
              ${CMAKE_CURRENT_SOURCE_DIR}/stream/yolo_stream.hpp
              ${STREAM_ROM_MODEL}/manifest.json ${STREAM_ROM_MODEL}/weights.bin
      COMMENT "ROM des poids de yolo_stream")
    add_executable(tb_stream_rom tb/tb_stream.cpp stream/yolo_stream.cpp
                   ${STREAM_ROM_DIR}/stream_rom.cpp)
    target_include_directories(tb_stream_rom PRIVATE stream ${STREAM_ROM_DIR})
    target_link_libraries(tb_stream_rom PRIVATE golden)
    target_compile_definitions(tb_stream_rom PRIVATE STREAM_ROM
                               ACC_MODEL_DIR="${ACC_MODEL_DIR}")
    target_compile_options(tb_stream_rom PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    add_test(NAME tb_stream_rom COMMAND tb_stream_rom)
  endif()

  # T9.2.4 : PE à décalages (REQ-YOLO) sur le modèle à poids puissances de 2
  # (tools/quant_lowbit.py --weights pow2 → build/m9/models/), testé s'il existe.
  add_library(accel_pow2 kernels/conv_pe.cpp kernels/postproc.cpp)
  target_include_directories(accel_pow2 PUBLIC kernels)
  target_include_directories(accel_pow2 SYSTEM PUBLIC ${hls_ap_types_SOURCE_DIR}/include)
  target_link_libraries(accel_pow2 PUBLIC golden accel_config)
  target_compile_definitions(accel_pow2 PUBLIC ACC_WMODE_POW2)
  target_compile_options(accel_pow2 PUBLIC -Wno-unknown-pragmas -Wno-maybe-uninitialized
                                           -Wno-uninitialized)
  add_executable(tb_net_pow2 tb/tb_net.cpp)
  target_link_libraries(tb_net_pow2 PRIVATE accel_pow2 driver)
  get_filename_component(ACC_M9_MODELS "${CMAKE_CURRENT_SOURCE_DIR}/../build/m9/models"
                         ABSOLUTE)
  # T10.10 : précision mixte retenue (poids 4 bits paquetés), tools/mixed_precision.py full.
  get_filename_component(ACC_M10_MODELS "${CMAKE_CURRENT_SOURCE_DIR}/../build/m10/models"
                         ABSOLUTE)
  if(EXISTS "${ACC_M10_MODELS}/tiny-yolov2-voc-mixed/manifest.json")
    add_test(NAME tb_net_mixed COMMAND tb_net --model ${ACC_M10_MODELS}
             --net tiny-yolov2-voc-mixed)
  endif()
  if(EXISTS "${ACC_M9_MODELS}/tiny-yolov2-voc-pow2-ptq/manifest.json")
    add_test(NAME tb_net_pow2 COMMAND tb_net_pow2 --model ${ACC_M9_MODELS}
             --net tiny-yolov2-voc-pow2-ptq --image 000001)
  endif()
endif()

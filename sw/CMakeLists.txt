cmake_minimum_required(VERSION 3.16)
project(yolo_sw CXX)

# Logiciel ARM (M7) : driver, post-traitement, applications.
#   SW_BACKEND=sim (PC)  : registres émulés devant le noyau C-sim (hls/), make sw-sim ;
#   SW_BACKEND=uio (KV260, build natif sous Ubuntu) : /dev/uioN + u-dma-buf, sans hls/.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SW_BACKEND sim CACHE STRING "sim | uio")
set_property(CACHE SW_BACKEND PROPERTY STRINGS sim uio)

if(SW_BACKEND STREQUAL "sim")
  # golden, driver (program.cpp) et accel (noyau C-sim, tuiles ACC_* du cache, défauts KV260).
  add_subdirectory(../hls ${CMAKE_BINARY_DIR}/hls)
elseif(SW_BACKEND STREQUAL "uio")
  add_subdirectory(../cpp/golden ${CMAKE_BINARY_DIR}/golden)
  add_library(driver driver/program.cpp)
  target_include_directories(driver PUBLIC driver ../hls/kernels)
  target_link_libraries(driver PUBLIC golden)
  target_compile_options(driver PRIVATE -Wall -Wextra)
else()
  message(FATAL_ERROR "SW_BACKEND : sim ou uio")
endif()

include(FetchContent)
FetchContent_Declare(stb GIT_REPOSITORY https://github.com/nothings/stb.git
                         GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20)
FetchContent_MakeAvailable(stb)

set(SW_DRIVER_SOURCES driver/device.cpp driver/uio_device.cpp driver/accel_driver.cpp
                      postproc/heads.cpp)
if(SW_BACKEND STREQUAL "sim")
  list(APPEND SW_DRIVER_SOURCES driver/sim_device.cpp)
endif()
add_library(sw_driver ${SW_DRIVER_SOURCES})
target_include_directories(sw_driver PUBLIC driver postproc app)
target_link_libraries(sw_driver PUBLIC driver)
target_compile_options(sw_driver PRIVATE -Wall -Wextra)
if(SW_BACKEND STREQUAL "sim")
  target_link_libraries(sw_driver PRIVATE accel)
  target_compile_definitions(sw_driver PUBLIC SW_HAVE_SIM)
endif()

get_filename_component(SW_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model" ABSOLUTE)

add_executable(run_compare app/run_compare.cpp)
target_include_directories(run_compare PRIVATE ../hls/tb)
target_link_libraries(run_compare PRIVATE sw_driver)
target_compile_definitions(run_compare PRIVATE SW_MODEL_DIR="${SW_MODEL_DIR}")
target_compile_options(run_compare PRIVATE -Wall -Wextra)

add_executable(yolo_app app/yolo_app.cpp)
target_include_directories(yolo_app SYSTEM PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(yolo_app PRIVATE sw_driver)
target_compile_options(yolo_app PRIVATE -Wall -Wextra)

# Mesures M8 : temps par étage, détections au seuil de la mAP, puissance (T8.1, T8.2).
find_package(Threads REQUIRED)
add_executable(yolo_bench app/yolo_bench.cpp)
target_include_directories(yolo_bench SYSTEM PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(yolo_bench PRIVATE sw_driver Threads::Threads)
target_compile_options(yolo_bench PRIVATE -Wall -Wextra)

# Tests sur PC (backend sim) : 0 écart couche par couche et détections == dumps (T7.2-T7.4).
include(CTest)
if(SW_BACKEND STREQUAL "sim")
  # Couches isolées : première conv, conv 13×13 de v2, conv 21 de v3 (route en 2 segments,
  # dont L19 upsample lu par adressage).
  add_test(NAME run_compare_layer0 COMMAND run_compare --image 000001 --layer 0)
  add_test(NAME run_compare_layer13 COMMAND run_compare --net tiny-yolov2-voc --layer 13)
  add_test(NAME run_compare_layer21 COMMAND run_compare --net tiny-yolov3-coco --layer 21)
  add_test(NAME run_compare COMMAND run_compare)
  # T9.1 : post-traitement par le noyau yolo_post == golden hwpp.
  add_test(NAME run_compare_hw_post COMMAND run_compare --image 000002 --hw-post)
  # T10.7 : séquenceur (une fin par image) puis yolo_post == golden hwpp.
  add_test(NAME run_compare_chain COMMAND run_compare --image 000001 --chain --hw-post)
  add_test(NAME yolo_app_input COMMAND yolo_app --model ${SW_MODEL_DIR}/tiny-yolov2-voc
           --input ${SW_MODEL_DIR}/tiny-yolov2-voc/dumps/000001/input.npy
           --out ${CMAKE_BINARY_DIR}/det_000001.json)
  # yolo_bench (T8.2) : une entrée de dump au seuil des dumps → mêmes détections.
  add_test(NAME yolo_bench_input COMMAND ${CMAKE_COMMAND}
           -DBENCH=$<TARGET_FILE:yolo_bench> -DMODEL=${SW_MODEL_DIR}/tiny-yolov2-voc
           -DIMAGE=000001 -DWORK=${CMAKE_BINARY_DIR}/bench_test
           -P ${CMAKE_CURRENT_SOURCE_DIR}/app/test_yolo_bench.cmake)
  # T10.5 : --pipeline (deux arènes, deux threads) → mêmes détections qu'en séquentiel.
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME yolo_bench_pipeline COMMAND ${CMAKE_COMMAND}
             -DBENCH=$<TARGET_FILE:yolo_bench> -DMODEL=${SW_MODEL_DIR}/tiny-yolov2-voc
             -DPYTHON=${Python3_EXECUTABLE} -DWORK=${CMAKE_BINARY_DIR}/bench_pipeline
             -P ${CMAKE_CURRENT_SOURCE_DIR}/app/test_yolo_bench_pipeline.cmake)
  endif()
  add_test(NAME yolo_app_pipeline COMMAND yolo_app --model ${SW_MODEL_DIR}/tiny-yolov2-voc
           --input ${SW_MODEL_DIR}/tiny-yolov2-voc/dumps/000001/input.npy --repeat 3
           --pipeline --out ${CMAKE_BINARY_DIR}/det_000001_pipeline.json)
endif()

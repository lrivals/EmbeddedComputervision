cmake_minimum_required(VERSION 3.16)
project(yolo_golden CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Bibliothèque du golden model (M5) : en-têtes partagés avec hls/ (conv.hpp) et sw/
# (postproc.hpp).
add_library(golden src/golden.cpp src/json.cpp src/npy.cpp src/model.cpp src/engine.cpp)
target_include_directories(golden PUBLIC include)

add_executable(golden_run tools/golden_run.cpp)
target_link_libraries(golden_run PRIVATE golden)

option(GOLDEN_TESTS "Construire les tests (Catch2 via FetchContent)" OFF)
if(GOLDEN_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.4)
  FetchContent_MakeAvailable(Catch2)
  file(GLOB TEST_SOURCES tests/*.cpp)
  add_executable(golden_tests ${TEST_SOURCES})
  target_link_libraries(golden_tests PRIVATE golden Catch2::Catch2WithMain)
  # Export réel (T4.7) : model/<net>/ ; les tests qui en ont besoin sont sautés s'il manque.
  get_filename_component(GOLDEN_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../model" ABSOLUTE)
  target_compile_definitions(golden_tests PRIVATE GOLDEN_MODEL_DIR="${GOLDEN_MODEL_DIR}")
  include(CTest)
  add_test(NAME golden_tests COMMAND golden_tests)
endif()

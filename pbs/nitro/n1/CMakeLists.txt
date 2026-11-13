cmake_minimum_required(VERSION 3.18)
project(n1_selftest LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)
get_filename_component(N1_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(ENG "${N1_DIR}/../../../src/ML-UMA/uma-engine" ABSOLUTE)
add_executable(n1_selftest
  n1_selftest.cpp
  ${ENG}/src/neighbor_list.cpp
  ${ENG}/src/runtime_config.cpp)
target_include_directories(n1_selftest PRIVATE ${ENG}/include)
target_link_libraries(n1_selftest PRIVATE ${TORCH_LIBRARIES})
target_compile_options(n1_selftest PRIVATE -O2)

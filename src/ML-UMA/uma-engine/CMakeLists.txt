cmake_minimum_required(VERSION 3.18)
project(uma_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(UMA_ENGINE_USE_CUDA "Expect CUDA LibTorch" ON)
if(NOT UMA_ENGINE_USE_CUDA)
  set(USE_CUDA OFF CACHE BOOL "" FORCE)
endif()

# Prefer Torch from the active uma312 env:
#   python -c "import torch; print(torch.utils.cmake_prefix_path)"
find_package(Torch REQUIRED)

add_library(uma_engine
  src/metadata.cpp
  src/neighbor_list.cpp
  src/postprocess.cpp
  src/predictor.cpp
  src/graph_parallel.cpp
)
target_include_directories(uma_engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(uma_engine PUBLIC ${TORCH_LIBRARIES})
# Locate uma_gp_worker.py for devices>1 FairChem eager GP backend.
target_compile_definitions(uma_engine PRIVATE
  "UMA_ENGINE_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
if(UMA_ENGINE_USE_CUDA)
  target_compile_definitions(uma_engine PUBLIC UMA_ENGINE_USE_CUDA)
endif()

# Optional CUDA vesin neighbor list (borrowed from graspa-mlip vesin_nl.h).
set(VESIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vesin" CACHE PATH
    "Vendored vesin-torch root (include/ + lib/libvesin_torch.so)")
if(VESIN_ROOT AND EXISTS "${VESIN_ROOT}/include/vesin_torch.hpp"
   AND EXISTS "${VESIN_ROOT}/lib/libvesin_torch.so")
  target_compile_definitions(uma_engine PUBLIC VESIN_ROOT)
  target_include_directories(uma_engine PUBLIC ${VESIN_ROOT}/include)
  # Absolute .so path so dependents (lmp) link without needing -L propagation.
  target_link_libraries(uma_engine PUBLIC "${VESIN_ROOT}/lib/libvesin_torch.so")
  set_target_properties(uma_engine PROPERTIES
    BUILD_RPATH "${VESIN_ROOT}/lib"
    INSTALL_RPATH "${VESIN_ROOT}/lib"
    INTERFACE_LINK_DIRECTORIES "${VESIN_ROOT}/lib")
  message(STATUS "uma-engine: vesin CUDA NL enabled (${VESIN_ROOT})")
else()
  message(STATUS "uma-engine: vesin not found; using CPU neighbor list")
endif()


add_executable(uma_parity_cli tests/parity_cli.cpp)
target_link_libraries(uma_parity_cli PRIVATE uma_engine)

add_executable(graph_shard_smoke tests/graph_shard_smoke.cpp)
target_link_libraries(graph_shard_smoke PRIVATE uma_engine)

install(TARGETS uma_engine uma_parity_cli graph_shard_smoke
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

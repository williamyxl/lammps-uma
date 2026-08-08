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
  src/peer_context.cpp
  src/libtorch_mp.cpp
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
set(UMA_ENGINE_HAS_NCCL OFF)
if(UMA_ENGINE_USE_CUDA)
  target_compile_definitions(uma_engine PUBLIC UMA_ENGINE_USE_CUDA)
  # Opt-in UMA_PEER_TRANSPORT=nccl (raw libnccl behind uma_peer collectives).
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS
      ENV NCCL_ROOT
      ENV CUDA_HOME
      /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl-2.25/include
      /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl/include
      /usr/local/cuda/include
    PATH_SUFFIXES include
  )
  find_library(NCCL_LIBRARY nccl
    HINTS
      ENV NCCL_ROOT
      ENV CUDA_HOME
      /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl-2.25/lib
      /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl/lib
      /usr/local/cuda/lib64
    PATH_SUFFIXES lib lib64
  )
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(UMA_ENGINE_HAS_NCCL ON)
    get_filename_component(NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)
    target_include_directories(uma_engine PUBLIC ${NCCL_INCLUDE_DIR})
    target_link_libraries(uma_engine PUBLIC ${NCCL_LIBRARY})
    target_compile_definitions(uma_engine PUBLIC UMA_ENGINE_USE_NCCL)
    # Ensure dependents (worker / lmp) resolve libnccl at runtime.
    set_property(TARGET uma_engine APPEND PROPERTY BUILD_RPATH "${NCCL_LIB_DIR}")
    set_property(TARGET uma_engine APPEND PROPERTY INSTALL_RPATH "${NCCL_LIB_DIR}")
    set_property(TARGET uma_engine APPEND PROPERTY INTERFACE_LINK_DIRECTORIES "${NCCL_LIB_DIR}")
    message(STATUS "uma-engine: NCCL enabled (${NCCL_LIBRARY})")
  else()
    set(UMA_ENGINE_HAS_NCCL OFF)
    message(WARNING "uma-engine: NCCL not found; UMA_PEER_TRANSPORT=nccl unavailable")
  endif()
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

add_executable(kokkos_peer_smoke tests/kokkos_peer_smoke.cpp)
target_link_libraries(kokkos_peer_smoke PRIVATE uma_engine)

add_executable(kokkos_peer_device_smoke tests/kokkos_peer_device_smoke.cpp)
target_link_libraries(kokkos_peer_device_smoke PRIVATE uma_engine)

add_executable(uma_libtorch_mp_worker tests/uma_libtorch_mp_worker.cpp)
target_link_libraries(uma_libtorch_mp_worker PRIVATE uma_engine)
# Header-only NCCL paths live in shared_peer.h (included by the worker TU).
# Explicitly attach defs + libnccl so the worker is not relying solely on
# transitive propagation from the static uma_engine archive.
if(UMA_ENGINE_HAS_NCCL)
  target_compile_definitions(uma_libtorch_mp_worker PRIVATE UMA_ENGINE_USE_NCCL UMA_ENGINE_USE_CUDA)
  target_include_directories(uma_libtorch_mp_worker PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(uma_libtorch_mp_worker PRIVATE ${NCCL_LIBRARY})
  get_filename_component(_uma_nccl_lib_dir "${NCCL_LIBRARY}" DIRECTORY)
  set_target_properties(uma_libtorch_mp_worker PROPERTIES
    BUILD_RPATH "${_uma_nccl_lib_dir}"
    INSTALL_RPATH "${_uma_nccl_lib_dir}")
  message(STATUS "uma-engine: worker explicitly linked to NCCL (${NCCL_LIBRARY})")
endif()

install(TARGETS uma_engine uma_parity_cli graph_shard_smoke kokkos_peer_smoke
                kokkos_peer_device_smoke uma_libtorch_mp_worker
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

cmake_minimum_required(VERSION 3.18)
project(uma_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(UMA_ENGINE_USE_CUDA "Expect CUDA LibTorch" ON)
# Intel XPU (Aurora): torch 2.13.0+xpu native backend. Mutually exclusive with
# CUDA for the accelerator device path; single-tile needs no collectives.
option(UMA_ENGINE_USE_XPU "Expect Intel XPU (Aurora) LibTorch" OFF)
if(UMA_ENGINE_USE_XPU)
  set(UMA_ENGINE_USE_CUDA OFF CACHE BOOL "" FORCE)
endif()
if(NOT UMA_ENGINE_USE_CUDA)
  set(USE_CUDA OFF CACHE BOOL "" FORCE)
endif()

# Prefer Torch from the active env:
#   python -c "import torch; print(torch.utils.cmake_prefix_path)"
find_package(Torch REQUIRED)

if(UMA_ENGINE_USE_XPU)
  message(STATUS "uma-engine: building for Intel XPU (torch+xpu)")
endif()

if(UMA_ENGINE_USE_XPU)
  # XPU build: single-tile (predictor.cpp) + graph-parallel edge-parallel
  # (mpi_peer_predictor.cpp + peer_context.cpp) using host-staged MPI (Step 1)
  # or native XCCL (Step 2) collectives. libtorch_mp (C++ MP fork runtime) is
  # CUDA/NCCL-only -> stubbed. graph_parallel.cpp keeps the eager fork path.
  set(UMA_ENGINE_SOURCES
    src/metadata.cpp
    src/neighbor_list.cpp
    src/postprocess.cpp
    src/predictor.cpp
    src/graph_parallel.cpp
    src/libtorch_mp_xpu_stub.cpp
    src/peer_context.cpp
    src/block_context.cpp
    src/mpi_peer_predictor.cpp
  )
else()
  set(UMA_ENGINE_SOURCES
    src/metadata.cpp
    src/neighbor_list.cpp
    src/postprocess.cpp
    src/predictor.cpp
    src/graph_parallel.cpp
    src/peer_context.cpp
    src/block_context.cpp
    src/libtorch_mp.cpp
    src/mpi_peer_predictor.cpp
  )
endif()

add_library(uma_engine ${UMA_ENGINE_SOURCES})
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
if(UMA_ENGINE_USE_XPU)
  target_compile_definitions(uma_engine PUBLIC UMA_ENGINE_USE_XPU)
  # Graph-parallel edge-parallel on XPU: native oneCCL (XCCL) on-device
  # collectives. MPI is used ONLY for the one-time KVS-address rendezvous and as
  # the launcher; the data path is pure on-device oneCCL.
  find_package(MPI REQUIRED)
  target_link_libraries(uma_engine PUBLIC MPI::MPI_CXX)

  option(UMA_ENGINE_USE_XCCL "Native oneCCL (XCCL) collectives on XPU" ON)
  if(UMA_ENGINE_USE_XCCL)
    # xccl_peer.cpp uses SYCL + oneCCL -> compile with icpx via a custom command
    # (CMake has one CXX compiler per dir; the rest of the engine is GCC). The
    # resulting object is linked into uma_engine.
    if(NOT DEFINED UMA_ICPX)
      set(UMA_ICPX "icpx" CACHE STRING "SYCL compiler for xccl_peer.cpp")
    endif()
    if(NOT DEFINED ONECCL_ROOT)
      set(ONECCL_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "oneCCL root")
    endif()
    # Gather torch include dirs as -I flags for the icpx invocation.
    set(_xccl_incs
      -I${CMAKE_CURRENT_SOURCE_DIR}/include
      -I${ONECCL_ROOT}/include)
    foreach(_d ${TORCH_INCLUDE_DIRS})
      list(APPEND _xccl_incs -I${_d})
    endforeach()
    set(_xccl_obj "${CMAKE_CURRENT_BINARY_DIR}/xccl_peer.o")
    add_custom_command(
      OUTPUT ${_xccl_obj}
      COMMAND ${UMA_ICPX} -fsycl -O2 -std=c++17 -fPIC
              -DUMA_ENGINE_USE_XPU -DUMA_ENGINE_USE_XCCL
              ${_xccl_incs}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/src/xccl_peer.cpp -o ${_xccl_obj}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/xccl_peer.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/include/uma/xccl_peer.h
      COMMENT "Compiling xccl_peer.cpp with icpx (SYCL+oneCCL)")
    add_custom_target(uma_xccl_obj DEPENDS ${_xccl_obj})
    add_dependencies(uma_engine uma_xccl_obj)
    target_link_libraries(uma_engine PUBLIC ${_xccl_obj}
      ${ONECCL_ROOT}/lib/libccl.so)
    target_compile_definitions(uma_engine PUBLIC UMA_ENGINE_USE_XCCL)
    target_include_directories(uma_engine PUBLIC ${ONECCL_ROOT}/include)
    # The icpx-produced xccl_peer.o needs Intel compiler runtime + SYCL libs at
    # the final (GCC-driven) link. Point at the oneAPI compiler lib dir and pull
    # libintlc/libimf/libsvml/libirng + sycl.
    if(NOT DEFINED UMA_ICPX_LIBDIR)
      # derive from icpx location: <root>/bin/icpx -> <root>/lib
      get_filename_component(_icpx_bin "${UMA_ICPX}" DIRECTORY)
      get_filename_component(_icpx_root "${_icpx_bin}" DIRECTORY)
      set(UMA_ICPX_LIBDIR "${_icpx_root}/lib" CACHE PATH "icpx runtime lib dir")
    endif()
    target_link_options(uma_engine PUBLIC
      -L${ONECCL_ROOT}/lib -L${UMA_ICPX_LIBDIR}
      -Wl,-rpath,${UMA_ICPX_LIBDIR} -Wl,-rpath,${ONECCL_ROOT}/lib)
    target_link_libraries(uma_engine PUBLIC
      ${UMA_ICPX_LIBDIR}/libintlc.so ${UMA_ICPX_LIBDIR}/libimf.so
      ${UMA_ICPX_LIBDIR}/libsvml.so ${UMA_ICPX_LIBDIR}/libirng.so)
    # xccl_peer.o pulls the full SYCL runtime chain; libsycl needs its UR loader
    # on the link line. Use the conda one that matches torch's libsycl.9.
    if(EXISTS "${ONECCL_ROOT}/lib/libur_loader.so")
      target_link_libraries(uma_engine PUBLIC ${ONECCL_ROOT}/lib/libur_loader.so)
    endif()
    message(STATUS "uma-engine: XCCL enabled (icpx object); icpx libs ${UMA_ICPX_LIBDIR}")
  endif()
endif()
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

# Multi-node / GP smoke tests + worker are CUDA/NCCL-only (Phase 4 on XPU).
if(NOT UMA_ENGINE_USE_XPU)
add_executable(graph_shard_smoke tests/graph_shard_smoke.cpp)
target_link_libraries(graph_shard_smoke PRIVATE uma_engine)

add_executable(kokkos_peer_smoke tests/kokkos_peer_smoke.cpp)
target_link_libraries(kokkos_peer_smoke PRIVATE uma_engine)

add_executable(kokkos_peer_device_smoke tests/kokkos_peer_device_smoke.cpp)
target_link_libraries(kokkos_peer_device_smoke PRIVATE uma_engine)

add_executable(uma_libtorch_mp_worker tests/uma_libtorch_mp_worker.cpp)
target_link_libraries(uma_libtorch_mp_worker PRIVATE uma_engine)
endif()
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

set(UMA_ENGINE_INSTALL_TARGETS uma_engine uma_parity_cli)
if(NOT UMA_ENGINE_USE_XPU)
  list(APPEND UMA_ENGINE_INSTALL_TARGETS graph_shard_smoke kokkos_peer_smoke
       kokkos_peer_device_smoke uma_libtorch_mp_worker)
endif()
install(TARGETS ${UMA_ENGINE_INSTALL_TARGETS}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

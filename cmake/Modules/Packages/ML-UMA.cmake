# ML-UMA: LibTorch UMA pair style (GPU-persistent engine)

set(UMA_ENGINE_ROOT "${LAMMPS_SOURCE_DIR}/ML-UMA/uma-engine" CACHE PATH
    "Path to uma-engine (LibTorch UMA inference library)")
get_filename_component(UMA_ENGINE_ROOT "${UMA_ENGINE_ROOT}" ABSOLUTE)

if(NOT EXISTS "${UMA_ENGINE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "ML-UMA requires uma-engine at ${UMA_ENGINE_ROOT}. "
    "Set -DUMA_ENGINE_ROOT=/path/to/uma-engine")
endif()

# Torch from conda uma312: python -c "import torch; print(torch.utils.cmake_prefix_path)"
# Pip/conda LibTorch often leaves MKL_INCLUDE_DIR unset; stub empty to satisfy INTERFACE.
if(NOT MKL_INCLUDE_DIR)
  set(MKL_INCLUDE_DIR "" CACHE PATH "MKL include (unused stub for pip torch)" FORCE)
endif()
find_package(Torch REQUIRED)

if(NOT TARGET uma_engine)
  set(UMA_ENGINE_USE_CUDA ON CACHE BOOL "" FORCE)
  add_subdirectory(${UMA_ENGINE_ROOT} ${CMAKE_BINARY_DIR}/uma-engine EXCLUDE_FROM_ALL)
endif()

target_link_libraries(lammps PRIVATE uma_engine ${TORCH_LIBRARIES})
target_include_directories(lammps PRIVATE ${UMA_ENGINE_ROOT}/include)

# Propagate vesin .so rpath onto the LAMMPS binary when present.
set(_VESIN_ROOT "${UMA_ENGINE_ROOT}/third_party/vesin")
if(DEFINED VESIN_ROOT AND VESIN_ROOT)
  set(_VESIN_ROOT "${VESIN_ROOT}")
endif()
if(EXISTS "${_VESIN_ROOT}/lib/libvesin_torch.so")
  target_link_libraries(lammps PRIVATE "${_VESIN_ROOT}/lib/libvesin_torch.so")
  set_property(TARGET lammps APPEND PROPERTY BUILD_RPATH "${_VESIN_ROOT}/lib")
  set_property(TARGET lammps APPEND PROPERTY INSTALL_RPATH "${_VESIN_ROOT}/lib")
endif()


# Do NOT set -DPREC_POS/FORCE/ENERGY on this LAMMPS tag: it lacks KOKKOS_PREC=mixed
# and those macros break Kokkos atom/comm types. Mixed precision is enforced in the
# pair↔engine path (FP32 positions/energy, FP64 forces) instead.

message(STATUS "ML-UMA: linked uma_engine from ${UMA_ENGINE_ROOT}")
message(STATUS "ML-UMA: Torch from ${TORCH_INSTALL_PREFIX}")
message(STATUS "ML-UMA: mixed prec via pair/engine casts (pos/energy FP32, forces FP64)")

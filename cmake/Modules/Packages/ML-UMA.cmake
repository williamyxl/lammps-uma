# ML-UMA: LibTorch pair_style uma (GPU-persistent engine)
#
# Default LAMMPS CI does not enable this package: LibTorch is an external
# dependency and UMA weights cannot be shipped (FAIR Chemistry License).
# Use cmake/presets/nolib.cmake (or leave PKG_ML-UMA=off) for library-free builds.

set(UMA_ENGINE_ROOT "${LAMMPS_SOURCE_DIR}/ML-UMA/uma-engine" CACHE PATH
    "Path to uma-engine (LibTorch UMA inference library)")
get_filename_component(UMA_ENGINE_ROOT "${UMA_ENGINE_ROOT}" ABSOLUTE)

if(NOT EXISTS "${UMA_ENGINE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "ML-UMA requires uma-engine at ${UMA_ENGINE_ROOT}. "
    "Set -DUMA_ENGINE_ROOT=/path/to/uma-engine")
endif()

# Pip/conda LibTorch often leaves MKL_INCLUDE_DIR unset on the Caffe2
# INTERFACE target. An empty cache entry satisfies the imported requirement.
if(NOT MKL_INCLUDE_DIR)
  set(MKL_INCLUDE_DIR "" CACHE PATH "MKL include (unused stub for pip/conda LibTorch)")
endif()

# If Torch_DIR / CMAKE_PREFIX_PATH is unset, try the active Python's Torch.
if(NOT Torch_DIR AND NOT DEFINED ENV{Torch_DIR})
  set(_UMA_PY "${Python_EXECUTABLE}")
  if(NOT _UMA_PY)
    set(_UMA_PY "${Python3_EXECUTABLE}")
  endif()
  if(NOT _UMA_PY)
    find_program(_UMA_PY NAMES python3 python)
  endif()
  if(_UMA_PY)
    execute_process(
      COMMAND ${_UMA_PY} -c "import torch; print(torch.utils.cmake_prefix_path)"
      RESULT_VARIABLE _UMA_TORCH_RC
      OUTPUT_VARIABLE _UMA_TORCH_PREFIX
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_UMA_TORCH_RC EQUAL 0 AND _UMA_TORCH_PREFIX)
      list(PREPEND CMAKE_PREFIX_PATH "${_UMA_TORCH_PREFIX}")
      message(STATUS "ML-UMA: Torch CMake prefix from Python: ${_UMA_TORCH_PREFIX}")
    endif()
  endif()
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(FATAL_ERROR
    "ML-UMA requires LibTorch (find_package(Torch) failed).\n"
    "Point CMake at it, for example:\n"
    "  -D CMAKE_PREFIX_PATH=$(python3 -c \"import torch; print(torch.utils.cmake_prefix_path)\")\n"
    "or set Torch_DIR to the directory that contains TorchConfig.cmake.\n"
    "Leave PKG_ML-UMA=off for default LAMMPS CI; LibTorch and UMA weights are not shipped.")
endif()

option(UMA_ENGINE_USE_XPU "Build ML-UMA for Intel GPU (XPU) LibTorch" OFF)
option(UMA_ENGINE_USE_CUDA "Build ML-UMA for CUDA LibTorch" ON)
if(UMA_ENGINE_USE_XPU)
  if(UMA_ENGINE_USE_CUDA)
    message(STATUS "ML-UMA: UMA_ENGINE_USE_XPU=ON; disabling CUDA backend")
  endif()
  set(UMA_ENGINE_USE_CUDA OFF CACHE BOOL "Build ML-UMA for CUDA LibTorch" FORCE)
endif()

if(NOT TARGET uma_engine)
  add_subdirectory(${UMA_ENGINE_ROOT} ${CMAKE_BINARY_DIR}/uma-engine EXCLUDE_FROM_ALL)
endif()

target_link_libraries(lammps PRIVATE uma_engine ${TORCH_LIBRARIES})
target_include_directories(lammps PRIVATE ${UMA_ENGINE_ROOT}/include)
if(UMA_ENGINE_USE_XPU)
  target_compile_definitions(lammps PRIVATE UMA_ENGINE_USE_XPU)
endif()

# Optional vesin neighbor-list .so/.dylib; rpath on the LAMMPS binary.
set(_VESIN_ROOT "${UMA_ENGINE_ROOT}/third_party/vesin")
if(DEFINED VESIN_ROOT AND VESIN_ROOT)
  set(_VESIN_ROOT "${VESIN_ROOT}")
endif()
set(_VESIN_LIB "${_VESIN_ROOT}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vesin_torch${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(EXISTS "${_VESIN_LIB}")
  target_link_libraries(lammps PRIVATE "${_VESIN_LIB}")
  set_property(TARGET lammps APPEND PROPERTY BUILD_RPATH "${_VESIN_ROOT}/lib")
  set_property(TARGET lammps APPEND PROPERTY INSTALL_RPATH "${_VESIN_ROOT}/lib")
endif()

message(STATUS "ML-UMA: linked uma_engine from ${UMA_ENGINE_ROOT}")
message(STATUS "ML-UMA: Torch from ${TORCH_INSTALL_PREFIX}")
message(STATUS "ML-UMA: CUDA=${UMA_ENGINE_USE_CUDA} XPU=${UMA_ENGINE_USE_XPU}")
message(STATUS "ML-UMA: pair_style uma is FP64-only (LibTorch artifact must be float64)")

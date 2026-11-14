cmake_minimum_required(VERSION 3.18)
project(s4_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

if(NOT DEFINED UMA_ICPX)
  set(UMA_ICPX "icpx")
endif()
if(NOT DEFINED CONDA_PREFIX)
  message(FATAL_ERROR "set CONDA_PREFIX to the fxpu env")
endif()

set(ENG "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ML-UMA/uma-engine")
set(_inc_flags "")
foreach(_d ${TORCH_INCLUDE_DIRS})
  list(APPEND _inc_flags -I${_d})
endforeach()
list(APPEND _inc_flags -I${CMAKE_CURRENT_SOURCE_DIR}/../s1)

set(_silu_src ${CMAKE_CURRENT_SOURCE_DIR}/../s1/uma_sycl_silu.sycl.cpp)
set(_silu_obj "${CMAKE_CURRENT_BINARY_DIR}/uma_sycl_silu.o")
add_custom_command(
  OUTPUT ${_silu_obj}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
          ${_inc_flags}
          -c ${_silu_src} -o ${_silu_obj}
  DEPENDS ${_silu_src} ${CMAKE_CURRENT_SOURCE_DIR}/../s1/uma_sycl_schema.h
          ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
  COMMENT "icpx uma_sycl_silu for S4"
  VERBATIM)

set(_xccl_src "${ENG}/src/xccl_peer.cpp")
set(_xccl_obj "${CMAKE_CURRENT_BINARY_DIR}/xccl_peer.o")
set(_xccl_inc ${_inc_flags} -I${ENG}/include -I${CONDA_PREFIX}/include)
foreach(_d ${MPI_CXX_INCLUDE_DIRS})
  list(APPEND _xccl_inc -I${_d})
endforeach()
add_custom_command(
  OUTPUT ${_xccl_obj}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
          -DUMA_ENGINE_USE_XPU -DUMA_ENGINE_USE_XCCL
          ${_xccl_inc}
          -c ${_xccl_src} -o ${_xccl_obj}
  DEPENDS ${_xccl_src} ${ENG}/include/uma/xccl_peer.h
          ${ENG}/include/uma/runtime_config.h
          ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
  COMMENT "icpx xccl_peer for S4"
  VERBATIM)

add_custom_target(s4_objs DEPENDS ${_silu_obj} ${_xccl_obj})
add_executable(s4_test s4_test.cpp ${ENG}/src/runtime_config.cpp)
add_dependencies(s4_test s4_objs)
target_include_directories(s4_test PRIVATE ${ENG}/include)
set(CMAKE_CXX_LINK_EXECUTABLE "${UMA_ICPX} -fsycl <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
target_link_libraries(s4_test PRIVATE ${_silu_obj} ${_xccl_obj} ${TORCH_LIBRARIES} MPI::MPI_CXX
  ${CONDA_PREFIX}/lib/libccl.so)
target_compile_options(s4_test PRIVATE -O2)
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(s4_test PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(s4_test PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(s4_test PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR}
    -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
endif()

cmake_minimum_required(VERSION 3.18)
project(s2_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)

if(NOT DEFINED UMA_ICPX)
  set(UMA_ICPX "icpx")
endif()
set(_inc_flags "")
foreach(_d ${TORCH_INCLUDE_DIRS})
  list(APPEND _inc_flags -I${_d})
endforeach()
# The schema header lives next to the step 12 source.
list(APPEND _inc_flags -I${CMAKE_CURRENT_SOURCE_DIR}/../s1)

set(_src ${CMAKE_CURRENT_SOURCE_DIR}/../s1/uma_sycl_silu.sycl.cpp)
set(_obj "${CMAKE_CURRENT_BINARY_DIR}/uma_sycl_silu.o")
add_custom_command(
  OUTPUT ${_obj}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
          ${_inc_flags}
          -c ${_src} -o ${_obj}
  DEPENDS ${_src} ${CMAKE_CURRENT_SOURCE_DIR}/../s1/uma_sycl_schema.h
          ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
  COMMENT "icpx uma_sycl_silu for S2"
  VERBATIM)

add_custom_target(s2_obj DEPENDS ${_obj})
add_executable(s2_test s2_test.cpp)
add_dependencies(s2_test s2_obj)
set(CMAKE_CXX_LINK_EXECUTABLE "${UMA_ICPX} -fsycl <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
target_link_libraries(s2_test PRIVATE ${_obj} ${TORCH_LIBRARIES})
target_compile_options(s2_test PRIVATE -O2)
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(s2_test PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(s2_test PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(s2_test PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR}
    -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
endif()

cmake_minimum_required(VERSION 3.18)
project(n54_f2 LANGUAGES CXX)
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
set(_obj "${CMAKE_CURRENT_BINARY_DIR}/f2_gate.o")
add_custom_command(
  OUTPUT ${_obj}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../s1/compile_sycl.sh
          ${_inc_flags}
          -c ${CMAKE_CURRENT_SOURCE_DIR}/f2_gate.sycl.cpp -o ${_obj}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/f2_gate.sycl.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../../s1/compile_sycl.sh
  COMMENT "icpx f2 gate"
  VERBATIM)
add_custom_target(f2_obj DEPENDS ${_obj})

set(CMAKE_CXX_CREATE_SHARED_LIBRARY
  "${UMA_ICPX} -fsycl -shared <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
add_library(uma_n5f2 SHARED keep.cpp)
add_dependencies(uma_n5f2 f2_obj)
target_link_libraries(uma_n5f2 PRIVATE ${_obj} ${TORCH_LIBRARIES})
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(uma_n5f2 PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(DEFINED CONDA_PREFIX AND EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(uma_n5f2 PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(uma_n5f2 PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR})
  if(DEFINED CONDA_PREFIX)
    target_link_options(uma_n5f2 PRIVATE
      -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
  endif()
endif()

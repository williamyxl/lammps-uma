cmake_minimum_required(VERSION 3.18)
project(n2_bench LANGUAGES CXX)
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
set(_obj "${CMAKE_CURRENT_BINARY_DIR}/n2_segreduce.o")
add_custom_command(
  OUTPUT ${_obj}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
          ${_inc_flags}
          -c ${CMAKE_CURRENT_SOURCE_DIR}/n2_segreduce.sycl.cpp -o ${_obj}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/n2_segreduce.sycl.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/../s1/compile_sycl.sh
  COMMENT "icpx n2_segreduce"
  VERBATIM)
add_custom_target(n2_obj DEPENDS ${_obj})
add_executable(n2_bench n2_bench.cpp)
add_dependencies(n2_bench n2_obj)
set(CMAKE_CXX_LINK_EXECUTABLE "${UMA_ICPX} -fsycl <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
target_link_libraries(n2_bench PRIVATE ${_obj} ${TORCH_LIBRARIES})
target_compile_options(n2_bench PRIVATE -O2)
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(n2_bench PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(DEFINED CONDA_PREFIX AND EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(n2_bench PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(n2_bench PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR})
  if(DEFINED CONDA_PREFIX)
    target_link_options(n2_bench PRIVATE
      -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
  endif()
endif()

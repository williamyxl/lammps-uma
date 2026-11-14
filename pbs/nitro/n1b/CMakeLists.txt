cmake_minimum_required(VERSION 3.18)
project(n1b_selftest LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)

if(NOT DEFINED UMA_ICPX)
  set(UMA_ICPX "icpx")
endif()
get_filename_component(N1B_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(ENG "${N1B_DIR}/../../../src/ML-UMA/uma-engine" ABSOLUTE)
get_filename_component(S1 "${N1B_DIR}/../s1" ABSOLUTE)

set(_inc_flags -I${ENG}/include)
foreach(_d ${TORCH_INCLUDE_DIRS})
  list(APPEND _inc_flags -I${_d})
endforeach()
set(_obj "${CMAKE_CURRENT_BINARY_DIR}/n1b_graph.o")
add_custom_command(
  OUTPUT ${_obj}
  COMMAND ${S1}/compile_sycl.sh ${_inc_flags}
          -c ${ENG}/src/n1b_graph.sycl.cpp -o ${_obj}
  DEPENDS ${ENG}/src/n1b_graph.sycl.cpp ${ENG}/include/uma/n1b_graph.h
          ${S1}/compile_sycl.sh
  COMMENT "icpx n1b_graph"
  VERBATIM)
add_custom_target(n1b_obj DEPENDS ${_obj})

add_executable(n1_selftest
  ${N1B_DIR}/../n1/n1_selftest.cpp
  ${ENG}/src/neighbor_list.cpp
  ${ENG}/src/runtime_config.cpp)
add_dependencies(n1_selftest n1b_obj)
target_include_directories(n1_selftest PRIVATE ${ENG}/include)
target_compile_definitions(n1_selftest PRIVATE UMA_N1B_SYCL)
set(CMAKE_CXX_LINK_EXECUTABLE "${UMA_ICPX} -fsycl <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
target_link_libraries(n1_selftest PRIVATE ${_obj} ${TORCH_LIBRARIES})
target_compile_options(n1_selftest PRIVATE -O2)
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(n1_selftest PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(DEFINED CONDA_PREFIX AND EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(n1_selftest PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(n1_selftest PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR})
  if(DEFINED CONDA_PREFIX)
    target_link_options(n1_selftest PRIVATE
      -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
  endif()
endif()

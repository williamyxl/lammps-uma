cmake_minimum_required(VERSION 3.18)
project(s1_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Torch REQUIRED)

if(NOT DEFINED UMA_ICPX)
  set(UMA_ICPX "icpx")
endif()
set(_incs -I${TORCH_INCLUDE_DIRS})
# TORCH_INCLUDE_DIRS is a CMake list; pass each as -I.
set(_inc_flags "")
foreach(_d ${TORCH_INCLUDE_DIRS})
  list(APPEND _inc_flags -I${_d})
endforeach()

set(_obj "${CMAKE_CURRENT_BINARY_DIR}/uma_sycl_silu.o")
string(JOIN " " _inc_str ${_inc_flags})
add_custom_command(
  OUTPUT ${_obj}
  COMMAND bash -c "${UMA_ICPX} -fsycl -O2 -std=c++17 -fPIC -fp-model=precise -D_GLIBCXX_USE_CXX11_ABI=1 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device pvc' ${_inc_str} -c ${CMAKE_CURRENT_SOURCE_DIR}/uma_sycl_silu.sycl.cpp -o ${_obj}"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/uma_sycl_silu.sycl.cpp
  COMMENT "icpx AOT PVC uma_sycl_silu"
  VERBATIM)

add_custom_target(s1_obj DEPENDS ${_obj})
add_executable(s1_test s1_test.cpp)
add_dependencies(s1_test s1_obj)
target_link_libraries(s1_test PRIVATE ${_obj} ${TORCH_LIBRARIES})
target_compile_options(s1_test PRIVATE -O2)
if(DEFINED UMA_ICPX_LIBDIR)
  target_link_libraries(s1_test PRIVATE
    ${UMA_ICPX_LIBDIR}/libintlc.so
    ${UMA_ICPX_LIBDIR}/libimf.so
    ${UMA_ICPX_LIBDIR}/libsvml.so
    ${UMA_ICPX_LIBDIR}/libirng.so)
  if(EXISTS "${CONDA_PREFIX}/lib/libur_loader.so")
    target_link_libraries(s1_test PRIVATE ${CONDA_PREFIX}/lib/libur_loader.so)
  endif()
  target_link_options(s1_test PRIVATE
    -L${UMA_ICPX_LIBDIR} -Wl,-rpath,${UMA_ICPX_LIBDIR}
    -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib)
endif()

cmake_minimum_required (VERSION 3.0)
project (Render)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_ARCH -gencode arch=compute_35,code=sm_35)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -Xcompiler -fPIC -shared -o)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " --Wno-deprecated-gpu-targets -std=c++11 -Xcompiler -fPIC -shared")

CUDA_ADD_LIBRARY(render_cuda_f SHARED render_cuda_filter.cu)
add_custom_command(
   TARGET render_cuda_f POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/librender_cuda_f.so${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/"
)

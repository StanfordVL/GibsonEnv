# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (Gibson_Environment)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

MESSAGE( STATUS "CMAKE_BINARY_DIR:         " ${CMAKE_BINARY_DIR} )
MESSAGE( STATUS "CMAKE_SOURCE_DIR:         " ${CMAKE_SOURCE_DIR} )


# ------ Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_ARCH -gencode arch=compute_35,code=sm_35)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -Xcompiler -fPIC -shared -o)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " --Wno-deprecated-gpu-targets -std=c++11")


if( CMAKE_SOURCE_DIR MATCHES " " )
    message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
    message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

## use this to globally use C++11 with in our project
set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CXX_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")

SET(CMAKE_CXX_FLAGS_DEBUG "-O0")
SET(CMAKE_C_FLAGS_DEBUG "-O0")

## load in pkg-config support
find_package(PkgConfig)
## use pkg-config to get hints for zmq locations
# add_subdirectory(cppzmq)
pkg_check_modules(PC_ZeroMQ REQUIRED)

## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
    add_definitions(${GLEW_DEFINITIONS})
    message(${GLEW_LIBRARIES})
else()
    message("GLEW NOT FOUND")
endif()

add_subdirectory(assimp)
SET(GLFW_BUILD_DOCS OFF CACHE BOOL "Build libfoo shared library")
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(glfw)

include_directories("glm")

#add_subdirectory (external)

add_definitions(-DGLEW_EGL)

include_directories(
    ${GLEW_INCLUDE_DIRS}
    glfw/include/GLFW
    external/lodepng/
    assimp/include/assimp
    ${GLUT_INCLUDE_DIRS}
    .
    ${ZeroMQ_INCLUDE_DIR}
    cppzmq/
    glad
)

set(OPENGL_LIBRARY
    ${OPENGL_LIBRARY}
    -lGL -lGLU -lXrandr -lXext -lX11 -lrt
    ${CMAKE_DL_LIBS}
    ${GLFW_LIBRARIES}
)

set(ALL_LIBS
    ${OPENGL_LIBRARY}
    ${GLEW_LIBRARY}
    ${GLUT_LIBRARY}
    ${ZeroMQ_LIBRARY}
    glfw
    dl
    pthread
)

add_definitions(
    -DTW_STATIC
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)



cuda_add_executable(depth_render
    glad/egl.c glad/gl.c
    depth_render/render.cpp
    common/render_cuda_f.cu
    common/render_cuda_f.h
    common/shader.cpp
    common/shader.hpp
    common/controls.cpp
    common/controls.hpp
    common/MTLtexture.cpp
    common/MTLtexture.hpp
    common/texture.cpp
    common/texture.hpp
    common/MTLobjloader.cpp
    common/MTLobjloader.hpp
    common/MTLplyloader.cpp
    common/MTLplyloader.hpp
    common/objloader.cpp
    common/objloader.hpp
    common/tinyply.cpp
    common/tinyply.h
    common/vboindexer.cpp
    common/vboindexer.hpp
    common/text2D.hpp
    common/text2D.cpp
    common/picojson.h
    common/cmdline.h
    common/semantic_color.hpp
    external/lodepng/lodepng.h
    external/lodepng/lodepng.cpp
    depth_render/StandardShadingRTT.vertexshader
    depth_render/MistShadingRTT.fragmentshader
    depth_render/NormalShadingRTT.fragmentshader
    depth_render/Passthrough.vertexshader
    depth_render/WobblyTexture.fragmentshader
)


target_link_libraries(depth_render
    ${ALL_LIBS}
)

# Xcode and Visual working directories
set_target_properties(depth_render PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/depth_render/")


SOURCE_GROUP(common REGULAR_EXPRESSION ".*/common/.*" )
SOURCE_GROUP(shaders REGULAR_EXPRESSION ".*/.*shader$" )


add_custom_command(
   TARGET depth_render POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/depth_render${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/depth_render/"
)




cuda_add_executable(semantic
    glad/egl.c
    glad/gl.c
    depth_render/semantic.cpp
    common/controls.cpp
    common/controls.hpp
    common/MTLtexture.cpp
    common/MTLtexture.hpp
    common/MTLobjloader.cpp
    common/MTLobjloader.hpp
    common/MTLplyloader.cpp
    common/MTLplyloader.hpp
    common/objloader.cpp
    common/objloader.hpp
    common/tinyply.cpp
    common/tinyply.h
    common/vboindexer.cpp
    common/vboindexer.hpp
    common/picojson.h
    common/cmdline.h
    common/semantic_color.hpp

    external/lodepng/lodepng.h
    external/lodepng/lodepng.cpp
)


target_link_libraries(semantic
    ${ALL_LIBS}
)

# Xcode and Visual working directories
set_target_properties(semantic PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/depth_render/")


SOURCE_GROUP(common REGULAR_EXPRESSION ".*/common/.*" )
SOURCE_GROUP(shaders REGULAR_EXPRESSION ".*/.*shader$" )


add_custom_command(
   TARGET semantic POST_BUILD
   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/semantic${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/depth_render/"
)
add_definitions(-D GLM_ENABLE_EXPERIMENTAL)

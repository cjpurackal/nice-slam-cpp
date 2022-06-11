cmake_minimum_required(VERSION 3.5)
project(nice-slam-cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
set(CMAKE_CXX_STANDARD 14)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

# libtorch
list(APPEND CMAKE_PREFIX_PATH "/home/developer/deps/libtorch/")
find_package(Torch REQUIRED)

#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_HOST_COMPILER "/usr/bin/g++")
set(BUILD_TESTS OFF CACHE BOOL "Build unit tests")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

#opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB miniz ${CMAKE_CURRENT_SOURCE_DIR}/deps/tinyexr/miniz.c)
file(GLOB ns src/main.cpp)
file(GLOB mapper src/Mapper.cpp)
file(GLOB tracker src/Tracker.cpp)
file(GLOB inputs src/inputs/*.cpp)

add_executable(${PROJECT_NAME}
               ${miniz}
               ${ns}
               ${mapper}
               ${tracker}
               ${inputs}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/tinyexr
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_link_libraries(${PROJECT_NAME}
    ${Eigen_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    pthread
    config++
	)


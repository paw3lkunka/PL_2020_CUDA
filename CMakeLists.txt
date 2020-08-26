cmake_minimum_required(VERSION 3.17)
project(CMake_CUDA_Kirsch LANGUAGES CXX CUDA)

if(NOT EXISTS "libs/stb/README.md")
    execute_process(COMMAND git submodule update --init "libs/stb/" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(NOT EXISTS "libs/cuda-samples/README.md")
    execute_process(COMMAND git submodule update --init "libs/cuda-samples" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include_directories(
    include
    include/kirsch
    libs/stb
    libs/cuda-samples/Common
)

add_library(kirsch STATIC
    include/kirsch.hpp
    include/kirsch/cpu.hpp
    include/kirsch/gpu.cuh
    include/kirsch/helpers.inl
    src/kirsch/cpu.cpp
    src/kirsch/gpu.cu
)
target_compile_features(kirsch PUBLIC cxx_std_20)
set_target_properties(kirsch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CUDA_Kirsch main.cpp)
target_compile_features(CUDA_Kirsch PUBLIC cxx_std_20)
set_target_properties(CUDA_Kirsch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CUDA_Kirsch PUBLIC kirsch)

if(CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_13")  # try out sm_20 or sm_30 if it does not work with older standard
endif()

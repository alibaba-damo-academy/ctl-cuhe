cmake_minimum_required(VERSION 3.10)
project(DB LANGUAGES CXX CUDA)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -g -O3 -fopenmp")

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Adjust the way CMake's find_package function works. This determines the order in which versions of packages are found.
SET(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

# Try to find and load settings from an external package (SEAL).
find_package(SEAL REQUIRED)

# GB_CUDA
add_subdirectory(include/gb_cuda/)

# TFHE
add_library(glwe OBJECT)
include_directories(include)
add_subdirectory(include)
target_link_libraries(glwe PUBLIC SEAL::seal)

add_executable(tfhe_test tfhe_test.cc)
target_link_libraries(tfhe_test glwe gb_cuda)

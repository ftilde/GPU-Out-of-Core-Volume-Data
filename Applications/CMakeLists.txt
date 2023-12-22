cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (3dns)

set(PROJECT_APPLICATION "")

set(PROJECT_BINARY          "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
set(PROJECT_CONFIG          "${CMAKE_CURRENT_SOURCE_DIR}/../config")
set(PROJECT_GCCORE          "${CMAKE_CURRENT_SOURCE_DIR}/../GcCore")

#--- STATIC or SHARED library ---
# option(STATIC "Compile STATIC libraries" OFF)
option(SHARED "Compile SHARED libraries" OFF)
if (SHARED)
    message("-- SHARED COMPILATION")
else()
    message("-- STATIC COMPILATION")
    add_definitions(-DTDNS_STATIC)
endif()

#--- Check compiler version ---
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least gcc 5.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    endif()
else()
   message(WARNING "You are using an unsupported compiler! Compilation has only been tested with GCC.")
endif()

#--- Support of C++17 (or C++14 or C++11, check compiler version) ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# link the cuda runtime dynamically
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-arch=sm_30")
    message("-- CUDA COMPUTE CAPABILITY NOT DEFINED. DEFAULT : ${CMAKE_CUDA_FLAGS}")
else()
    message("-- CUDA COMPUTE CAPABILITY : ${CMAKE_CUDA_FLAGS}")
endif()

#--- Configure NVCC and CXX flags ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Pass options to NVCC
    # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g -gencode ${CMAKE_CUDA_FLAGS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -lineinfo -cudart shared ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -std=c++17 -g -pg")
    # pre-processor define for debug mode
    add_definitions(-DTDNS_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Pass options to NVCC // for profiling, add -lineinfo
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -std=c++17 -O3")
    # pre-processor define for release mode
    add_definitions(-DTDNS_RELEASE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    # Pass options to NVCC
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g ${CMAKE_CUDA_FLAGS})
    # GCC flags
    set(CMAKE_CXX_FLAGS "-Wall -std=c++17 -g -pg -ftest-coverage -fprofile-arcs")
    # pre-processor define for debug mode
    add_definitions(-DTDNS_DEBUG)
endif()

#--- Local dependencies ---
set(PROJECT_DEPENDENCIES    "${CMAKE_CURRENT_SOURCE_DIR}/../GcCore/dependencies")
set(PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR ${PROJECT_DEPENDENCIES}/cmake)

# pre-processor define for linux
add_definitions(-DTDNS_LINUX)
# add_definitions(-DTDNS_BENCHMARK)

#set(CUDA_SEPARABLE_COMPILATION ON)

# set the output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY}/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY}/${CMAKE_BUILD_TYPE})

#--- Add all subdirectories ---
add_subdirectory(../GcCore/src LIBRARY_OUTPUT_PATH)
add_subdirectory(DVR-ray-caster)
# add_subdirectory(exemple)
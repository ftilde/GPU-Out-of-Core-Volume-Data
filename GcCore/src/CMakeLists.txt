cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (3dns)

if(NOT DEFINED PROJECT_APPLICATION)

    set(PROJECT_BINARY          "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
    set(PROJECT_CONFIG          "${CMAKE_CURRENT_SOURCE_DIR}/../../config")
    set(PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR ${PROJECT_DEPENDENCIES}/cmake)

    # set the output path
    set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY}/${CMAKE_BUILD_TYPE})

    #--- STATIC or SHARED library ---
    # option(STATIC "Compile STATIC libraries" OFF)
    option(SHARED "Compile SHARED libraries" OFF)
    if (SHARED)
        message("-- SHARED COMPILATION")
    else()
        message("-- STATIC COMPILATION")
        add_definitions(-DTDNS_STATIC)
    endif()

    #--- Check des versions des compilateurs ---
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # require at least gcc 5.1
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
            message(FATAL_ERROR "GCC version must be at least 5.1!")
        #else() 
        #set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}"}
        endif()
    else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with GCC.")
    endif()

    #--- Activation du support C++11 ou C++14 mais dans ce cas, verifiez les versions des compilateurs ---
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED on)

    if (NOT CMAKE_CUDA_FLAGS)
        set(CMAKE_CUDA_FLAGS "-arch=sm_30")
        message("-- CUDA COMPUTE CAPABILITY NOT DEFINED. DEFAULT : ${CMAKE_CUDA_FLAGS}")
    else()
        message("-- CUDA COMPUTE CAPABILITY : ${CMAKE_CUDA_FLAGS}")
    endif()

    #--- Configure NVCC and CXX flags ---
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Pass options to NVCC
        #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g -gencode arch=compute_30,code=sm_30)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g ${CMAKE_CUDA_FLAGS})
        # GCC flags
        set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -pg")
        # pre-processor define for debug mode
        add_definitions(-DTDNS_DEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Pass options to NVCC // for profiling, add -lineinfo
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 ${CMAKE_CUDA_FLAGS})
        # GCC flags
        set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3")
        # pre-processor define for release mode
        add_definitions(-DTDNS_RELEASE)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Coverage")
        # Pass options to NVCC
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --device-debug -g ${CMAKE_CUDA_FLAGS})
        # GCC flags
        set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -pg -ftest-coverage -fprofile-arcs")
        # pre-processor define for debug mode
        add_definitions(-DTDNS_DEBUG)
    endif()

    # pre-processor define for linux
    add_definitions(-DTDNS_LINUX)
    # add_definitions(-DTDNS_BENCHMARK)

endif(NOT DEFINED PROJECT_APPLICATION)



#--- Local dependencies ---
set(PROJECT_DEPENDENCIES    "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies")
set(PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR ${PROJECT_DEPENDENCIES}/cmake)
find_package(lz4 REQUIRED PATHS ${PROJECT_DEPENDENCIES_CONFIGFILEPACKAGES_DIR} NO_DEFAULT_PATH)

# pre-processor define for export libraries compilation
add_definitions(-DTDNS_EXPORT)

set(INCLUDE_PATH          "${CMAKE_CURRENT_SOURCE_DIR}/../include")

#--- Add all subdirectories ---
add_subdirectory(libCommon)
add_subdirectory(libData)
# add_subdirectory(libGraphics)
add_subdirectory(libMath)
add_subdirectory(libNetwork)
add_subdirectory(libPreprocessor)
add_subdirectory(libTinyXml)
add_subdirectory(libPython)

add_subdirectory(cuda/libCommon)
add_subdirectory(cuda/libGPUCache)
# add_subdirectory(cuda/libGraphics)
add_subdirectory(cuda/libPreprocessor)
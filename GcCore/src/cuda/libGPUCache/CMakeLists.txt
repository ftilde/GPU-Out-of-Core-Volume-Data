cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libcudaGPUCache)

find_package(CUDA QUIET REQUIRED)

# --------------- Library configuration ---------------

# CAREFUL : DEPRECATED !! (used for CUDA) (cuda_include_directories() doesn't works)
include_directories(${INCLUDE_PATH}/)

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/cuda/libGPUCache/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (SHARED)
    # Create a shared library named libcudaGPUCache
    cuda_add_library(cudaGPUCache SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libcudaGPUCache
    cuda_add_library(cudaGPUCache STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the cudaGPUCache target
target_link_libraries(cudaGPUCache
    Common
    cudaCommon
    Math
    Data
)

# define an include directory to the cudaGPUCache target
target_include_directories(cudaGPUCache PRIVATE ${INCLUDE_PATH}/)
target_include_directories(cudaGPUCache PUBLIC ${INCLUDE_PATH}/)
cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0146 OLD)


# --------------- Project configuration ---------------

project (libcudaCommon)

find_package(CUDA QUIET REQUIRED)

# --------------- Library configuration ---------------

# CAREFUL : DEPRECATED !! (used for CUDA) (cuda_include_directories() doesn't works)
include_directories(${INCLUDE_PATH}/)

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/cuda/libCommon/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (SHARED)
    # Create a shared library named libCommon
    cuda_add_library(cudaCommon SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libCommon
    cuda_add_library(cudaCommon STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Common target
target_link_libraries(cudaCommon
    Math
)

# define an include directory to the Common target
target_include_directories(cudaCommon PRIVATE ${INCLUDE_PATH}/)
target_include_directories(cudaCommon PUBLIC ${INCLUDE_PATH}/)

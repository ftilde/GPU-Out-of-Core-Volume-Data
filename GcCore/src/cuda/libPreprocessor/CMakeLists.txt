cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libcudaPreprocessor)

find_package(CUDA QUIET REQUIRED)

# --------------- Library configuration ---------------

# CAREFUL : DEPRECATED !! (used for CUDA) (cuda_include_directories() doesn't works)
include_directories(${INCLUDE_PATH}/)

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/cuda/libPreprocessor/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (SHARED)
    # Create a shared library named libPreprocessor
    cuda_add_library(cudaPreprocessor SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libPreprocessor
    cuda_add_library(cudaPreprocessor STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Preprocessor target
target_link_libraries(cudaPreprocessor
    Common
    Math
    Data
)

# define an include directory to the Preprocessor target
target_include_directories(cudaPreprocessor PRIVATE ${INCLUDE_PATH}/)
target_include_directories(cudaPreprocessor PUBLIC ${INCLUDE_PATH}/)
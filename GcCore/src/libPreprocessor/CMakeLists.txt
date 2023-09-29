cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libPreprocessor)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libPreprocessor/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (SHARED)
    # Create a shared library named libPreprocessor
    add_library(Preprocessor SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libPreprocessor
    add_library(Preprocessor STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Preprocessor target
target_link_libraries(Preprocessor
    Common
    Math
    Data
)

# define an include directory to the Preprocessor target
target_include_directories(Preprocessor PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Preprocessor PUBLIC ${INCLUDE_PATH}/)
cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libMath)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libMath/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (SHARED)
    # Create a shared library named libMath
    add_library(Math SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libMath
    add_library(Math STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# define an include directory to the Math target
target_include_directories(Math PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Math PUBLIC ${INCLUDE_PATH}/)

#target_compile_features(Math PUBLIC x);
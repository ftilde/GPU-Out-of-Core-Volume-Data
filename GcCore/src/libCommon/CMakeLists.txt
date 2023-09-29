cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libCommon)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libCommon/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (SHARED)
    # Create a shared library named libCommon
    add_library(Common SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libCommon
    add_library(Common STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Common target
target_link_libraries(Common)

# define an include directory to the Common target
target_include_directories(Common PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Common PUBLIC ${INCLUDE_PATH}/)
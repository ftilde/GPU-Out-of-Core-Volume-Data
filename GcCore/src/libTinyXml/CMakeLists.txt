cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libTinyXml)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libTinyXml/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (SHARED)
    # Create a shared library named libTinyXml
    add_library(TinyXml SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libTinyXml
    add_library(TinyXml STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

target_include_directories(TinyXml PRIVATE ${INCLUDE_PATH}/)
target_include_directories(TinyXml PUBLIC ${INCLUDE_PATH}/)
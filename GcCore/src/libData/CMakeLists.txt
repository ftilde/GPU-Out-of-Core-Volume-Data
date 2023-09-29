cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libData)

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libData/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

if (SHARED)
    # Create a shared library named libData
    add_library(Data SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libData
    add_library(Data STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Data target
target_link_libraries(Data
    Common
    Math
    TinyXml
    lz4
)

# define an include directory to the Data target
target_include_directories(Data PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Data PUBLIC ${INCLUDE_PATH}/)

#target_compile_features(Data PUBLIC x);
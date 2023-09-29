cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

include(ExternalProject)

# Target directory for the config-file packages exported by the dependencies
set(DEPENDENCIES_CONFIGFILEPACKAGES_DIR	${CMAKE_SOURCE_DIR}/cmake)

file(MAKE_DIRECTORY ${DEPENDENCIES_CONFIGFILEPACKAGES_DIR})

###############################################################################
# Dependency build toggle options
###############################################################################
option(BUILD_LZ4        "Build LZ4"         ON)

if (BUILD_LZ4)
    include(lz4.cmake)
endif()
# Config file for the lz4 package.
#
# Defines the following macros:
#
# lz4_BINARY_DIR	- Binary directory
# lz4_INCLUDE_DIR	- Include directories

# Imported targets definition file
if(NOT TARGET lz4)
  include("/home/jo/code/3dns_gitHub/GcCore/dependencies/cmake/lz4-targets.cmake")
endif()

# Macro definitions
set(lz4_BINARY_DIR		/home/jo/code/3dns_gitHub/GcCore/dependencies/lz4-1.8.1/bin)
set(lz4_INCLUDE_DIR		/home/jo/code/3dns_gitHub/GcCore/dependencies/lz4-1.8.1/include)

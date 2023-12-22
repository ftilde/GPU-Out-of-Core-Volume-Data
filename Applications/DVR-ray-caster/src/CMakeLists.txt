cmake_minimum_required(VERSION 2.8.12)

# --------------- Project configuration ---------------

project (libGraphics)

if (NODISPLAY)
else()
    # since cmake 3.11, set the cmake policy for opengl preference to choose LEGACY to avoid Warnings
    set(OpenGL_GL_PREFERENCE "LEGACY")
    find_package(OpenGL REQUIRED)
    # pre-processor define for GLEW
    add_definitions(-DGLEW_STATIC)
endif()

#message(${CUDA_INCLUDE_DIRS})

# --------------- Library configuration ---------------

file(GLOB_RECURSE HEADER_FILES "${INCLUDE_PATH}/GcCore/libGraphics/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

if (SHARED)
    # Create a shared library named libGraphics
    add_library(Graphics SHARED 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
else()
    # Create a static library named libGraphics
    add_library(Graphics STATIC 
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
endif()

# link libraries to the Graphics target
if (NODISPLAY)
    target_link_libraries(Graphics
        Common
        Math
        Data
    )
else()
    include_directories(
        ${glew_INCLUDE_DIR}
    )
    target_link_libraries(Graphics
        Common
        Math
        Data
        ${OPENGL_gl_LIBRARY}
        glew
        -lSDL2
    )
endif()

# define an include directory to the Graphics target
target_include_directories(Graphics PRIVATE ${INCLUDE_PATH}/)
target_include_directories(Graphics PUBLIC ${INCLUDE_PATH}/)

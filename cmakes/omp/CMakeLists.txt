cmake_minimum_required(VERSION 3.2)
project(hs_omp)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../../build/) 
include_directories(
        ../../include
        ../../imgui
        ../../imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        ../../imgui/*.cpp
        ../../imgui/backends/imgui_impl_sdl.cpp
        ../../imgui/backends/imgui_impl_opengl2.cpp
        ../../imgui/misc/freetype/imgui_freetype.cpp
        ../../imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC})
add_executable(hs_omp ../../src/graphic.cpp ../../src/omp.cpp)
get_filename_component(FONT_PATH ../../imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
target_link_libraries(hs_omp core)
target_compile_options(hs_omp PRIVATE
          $<$<COMPILE_LANGUAGE:C>:${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}>)
target_link_options(hs_omp PRIVATE ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp=libomp)
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
target_compile_definitions(hs_omp PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
if (WIN32)
    target_link_options(hs_omp BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()
cmake_minimum_required(VERSION 3.15)

project(particle_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

#add_compile_definitions(ENABLE_LOG)

add_subdirectory(render_engine_family/engine_core)
add_subdirectory(render_engine_family/model_render_core)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(particle_sim
        src/physical_particle.h
        src/main.cpp
        src/basic_thread_pool.h
)

target_compile_definitions(particle_sim PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(particle_sim PRIVATE -march=native -ffast-math)
endif()

target_link_libraries(particle_sim
        PRIVATE
        engine_core
        model_render_core
        glfw
        OpenGL::GL
)

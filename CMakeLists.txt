Cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
project("VulkanCompute" VERSION "0.1.0" DESCRIPTION "My work on the vulkan tutorial compute chapter")

set(GLM_ENABLE_CXX_20 "ON")
set(GLFW_BUILD_WAYLAND "ON")
set(GLFW_BUILD_X11 "ON")
find_package(Vulkan REQUIRED)
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG  bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.0
)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 00e86d4b733103a23278fe53ce58a0d14dd47d32
)
FetchContent_MakeAvailable(glm)
FetchContent_MakeAvailable(glfw)

add_executable(VulkanCompute "./main.cpp")

message(STATUS $ENV{SHADER_COMPILE_SCRIPT})

target_include_directories(${PROJECT_NAME} PRIVATE libs/)

add_custom_target(compile_shaders ALL
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Compiling shaders")

add_custom_target ( compile_tags ALL
    COMMAND ctags main.cpp --c++-kinds=+p --fields=+iaS --extras=+q . 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Compiling project tags")

add_custom_target ( copy_resources ALL 
    COMMAND cp -r shaders/ ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying resources to build"
)

add_dependencies(${PROJECT_NAME} compile_shaders)
add_dependencies(${PROJECT_NAME} compile_tags)
add_dependencies(${PROJECT_NAME} copy_resources)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC Vulkan::Vulkan
    PUBLIC glm::glm
    PUBLIC glfw)


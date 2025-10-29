cmake_minimum_required(VERSION 3.25)

# if(APPLE)
#     message(FATAL_ERROR "Apple systems are not supported!")
# endif()

set(MI "   ") # Message indents
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")


# Set up project
project(LiveVisionKit)
message(STATUS "\nConfiguring LiveVisionKit...")

# Multi-platform support (placeholder)
if(APPLE)
    find_package(Metal QUIET)
    if(Metal_FOUND)
        set(USE_METAL ON)
        message(STATUS "Using Metal for GPU acceleration")
    else()
        message(WARNING "Metal not found, falling back to OpenCL")
        set(USE_OPENCL ON)
    endif()
elseif(WIN32)
    find_package(DirectX QUIET)
    if(DirectX_FOUND)
        set(USE_DIRECTX ON)
        message(STATUS "Using DirectX for GPU acceleration")
    else()
        message(WARNING "DirectX not found, falling back to OpenCL")
        set(USE_OPENCL ON)
    endif()
else()
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        set(USE_VULKAN ON)
        message(STATUS "Using Vulkan for GPU acceleration")
    else()
        set(USE_OPENCL ON)
        message(STATUS "Using OpenCL for GPU acceleration")
    endif()
endif()

# Project defines
set(LVK_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LiveVisionKit/")
set(LVK_RELEASES_DIR "Releases")
set(LVK_INCLUDES_DIR "Include")
set(LVK_BINARY_DIR "Binaries")
set(LVK_DEBUG_POSTFIX "d")

# Set default install location
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(
        CMAKE_INSTALL_PREFIX
        "${CMAKE_BINARY_DIR}/Install/"
        CACHE PATH
        "The location to which build files will be installed" 
        FORCE
    )
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Project settings
set(BUILD_OBS_PLUGIN "ON" CACHE BOOL "Build the OBS-Studio plugin")
set(BUILD_VIDEO_EDITOR "ON" CACHE BOOL "Build the video editor CLT")
set(DISABLE_CHECKS "OFF" CACHE BOOL "Compile without asserts and pre-condition checks")
# Load common dependencies (OpenCV)
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "${MI}Found OpenCV: YES")
    message(STATUS "${MI}OpenCV Version: ${OpenCV_VERSION}")
    message(STATUS "${MI}OpenCV Modules: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "Failed to find OpenCV!")
endif()

find_package(TensorflowLite QUIET)
find_package(obs-studio QUIET)

# If we are on Linux, we also need to load QT for OpenCV
if(UNIX)
    find_package(Qt5 REQUIRED Core Gui Widgets Test Concurrent OpenGL)
endif()

# Load sub-modules
add_subdirectory(LiveVisionKit)

if(BUILD_VIDEO_EDITOR)                                 
    message(STATUS "\nBuilding with video editor CLT...")
    add_subdirectory(Modules/VideoEditor)
endif()

if(BUILD_OBS_PLUGIN)
    message(STATUS "\nBuilding with LVK OBS-Studio plugin...")
    add_subdirectory(Modules/OBS-Plugin)
endif()

message(STATUS "\n")
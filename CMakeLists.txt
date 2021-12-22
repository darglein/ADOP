cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ADOP VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/saiga/cmake/")

include(helper_macros)
include(ExternalProject)
DefaultBuildType(RelWithDebInfo)
message("Build Options")

OptionsHelper(ADOP_HEADLESS "Skips the adop_viewer and other apps that require a window." OFF)
OptionsHelper(ADOP_ASAN "Adds sanitize=address compiler flag" OFF)

if (ADOP_HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)
endif ()

############# Required LIBRARIES ###############

# Saiga
set(SAIGA_BUILD_SAMPLES OFF)
set(SAIGA_BUILD_TESTS OFF)
set(SAIGA_BUILD_SHARED ON)
set(SAIGA_MODULE_VISION OFF)
set(SAIGA_MODULE_VULKAN OFF)
#set(SAIGA_MODULE_CUDA OFF)
set(SAIGA_NO_INSTALL ON)
set(SAIGA_USE_SUBMODULES ON)
set(SAIGA_WITH_FFMPEG OFF)
set(SAIGA_WITH_FREETYPE OFF)
add_subdirectory(External/saiga)
PackageHelperTarget(saiga_core SAIGA_FOUND)
if (NOT ADOP_HEADLESS)
    PackageHelperTarget(saiga_opengl SAIGA_FOUND)
endif ()
PackageHelperTarget(saiga_opengl SAIGA_FOUND)
PackageHelperTarget(saiga_cuda SAIGA_FOUND)

# Torch
find_package(Torch REQUIRED)
PackageHelperTarget(torch TORCH_FOUND)

# Torchvision
add_subdirectory(External/torchvision)
target_include_directories(torchvision PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/External/torchvision/>
        $<INSTALL_INTERFACE:External/torchvision/>
        )
PackageHelperTarget(torchvision TORCHVISION_FOUND)


#add_subdirectory(External/tensorboard_logger)
#target_include_directories(tensorboard_logger PUBLIC External/tensorboard_logger/include ${CMAKE_CURRENT_BINARY_DIR}/External/tensorboard_logger)
#set_property(TARGET tensorboard_logger PROPERTY POSITION_INDEPENDENT_CODE ON)
#PackageHelperTarget(tensorboard_logger TBL_FOUND)

############# COMPILER FLAGS ###############

if (MSVC)
    #multiprocessor compilation for visual studio
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")
endif ()


message(STATUS CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})

set(LIBS ${LIBS} ${LIB_TARGETS})

############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



############# SOURCE ###############

add_subdirectory(src)








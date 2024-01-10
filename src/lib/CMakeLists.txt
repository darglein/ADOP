set(TARGET_NAME "NeuralPoints")


## Generate source file with git commit
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC MY_GIT_SHA1 ALLOW_LOOKING_ABOVE_CMAKE_SOURCE_DIR)
set(MY_BUILDCONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

configure_file (
        "${CMAKE_CURRENT_SOURCE_DIR}/git_sha1.h.in"
        "${MY_BUILDCONFIG_DIR}/git_sha1.h"
        @ONLY
)

configure_file (
        "${CMAKE_CURRENT_SOURCE_DIR}/build_config.h.in"
        "${MY_BUILDCONFIG_DIR}/build_config.h"
        @ONLY
)

cmake_policy(SET CMP0104 OLD)
add_library(${TARGET_NAME} SHARED "")
generate_export_header(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "." ${MY_BUILDCONFIG_DIR})
target_sources(${TARGET_NAME} PRIVATE "${MY_BUILDCONFIG_DIR}/build_config.h" "${MY_BUILDCONFIG_DIR}/git_sha1.h")
target_link_libraries(${TARGET_NAME} ${LIBS})

if (ADOP_ASAN)
    target_compile_options(
            ${TARGET_NAME} PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    )
    target_link_options(${TARGET_NAME} PUBLIC -fsanitize=address)
endif ()

list(APPEND MY_CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND MY_CUDA_FLAGS "-lineinfo")


#############  source #############


# Source files in "src/" to level directory
FILE(GLOB_RECURSE DATA_SRC data/*.cpp)
FILE(GLOB_RECURSE MODELS_SRC models/*.cpp)
if (TARGET saiga_opengl)
    FILE(GLOB_RECURSE OPENGL_SRC opengl/*.cpp)
endif ()
FILE(GLOB_RECURSE RENDER_SRC rendering/*.cpp)
FILE(GLOB_RECURSE RENDER_CUDA_SRC rendering/*.cu)
FILE(GLOB_RECURSE PROG_HEADER *.h ${SAIGA_BUILDCONFIG_FILE})
target_sources(${TARGET_NAME} PRIVATE ${DATA_SRC} ${MODELS_SRC} ${OPENGL_SRC} ${RENDER_SRC} ${RENDER_CUDA_SRC} PUBLIC ${PROG_HEADER})


set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if (PR_NO_WINDOW)
    target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
endif ()

if(WIN32)
	#build shared lib flags
    target_compile_definitions(${TARGET_NAME} PRIVATE -DBUILD_NEURAL_POINTS_DLL)
    target_compile_definitions(${TARGET_NAME} PRIVATE -DBUILD_SHARED_LIBS)
	
    #torch_cpu and torch_cuda interface compiler flags which need to be "escaped" with -Xcompiler
    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
        message("replacing ${EXISTING_TARGET} cuda flags:")
        message(${old_flags})
        if(NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
                "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
                )
        endif()
    endfunction()

    CUDA_CONVERT_FLAGS(torch_cpu)
    CUDA_CONVERT_FLAGS(torch_cuda)

endif()
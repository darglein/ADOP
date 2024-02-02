function(add_app TARGET_NAME ADD_DLL_POSTBUILD)

    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_include_directories(${TARGET_NAME} PUBLIC ".")
    target_link_libraries(${TARGET_NAME} NeuralPoints)

    if (PR_NO_WINDOW)
        target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
    endif ()


    # ----------------------------------------------------------
    # dll copy
    if (WIN32)
        if (ADD_DLL_POSTBUILD)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_BINARY_DIR}/src/lib/$<CONFIG>/NeuralPoints.dll" $<TARGET_FILE_DIR:${TARGET_NAME}>)

            set(copyDest "${PROJECT_BINARY_DIR}/bin/$<CONFIG>/")

            function(SANITIZE_DLL_PATHS_AND_COPY PATHS_DLLS PATHS_OUTPUT)
                #Sanitizes paths to remove backslashes
                STRING(REGEX REPLACE "/" "\\\\" copyInput \"${PATHS_DLLS}\")
                STRING(REGEX REPLACE "/" "\\\\" copyDestination \"${PATHS_OUTPUT}\")
                message("dll post build copy: copying ${copyInput} to ${copyDestination}.")

                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND COMMAND copy ${copyInput} ${copyDestination})
            endfunction()

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/../External/libtorch/lib/*.dll" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${CONDA_P_PATH}/bin/libiomp5md.dll" ${copyDest})
			SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/bin/$<CONFIG>/*.dll" ${copyDest})
			SANITIZE_DLL_PATHS_AND_COPY("${CUDA_P_PATH}/bin/*.dll" ${copyDest})

        endif ()
    endif ()

    message(STATUS "App enabled:       ${TARGET_NAME}")
endfunction()

add_executable(colmap2adop colmap2adop.cpp)
add_app(colmap2adop FALSE)

add_executable(scannet2adop scannet2adop.cpp)
add_app(scannet2adop FALSE)

add_executable(preprocess_pointcloud preprocess_pointcloud.cpp)
add_app(preprocess_pointcloud FALSE)

add_executable(adop_train adop_train.cpp)
add_app(adop_train TRUE)

if (TARGET saiga_opengl)
    add_executable(adop_viewer adop_viewer.cpp adop_viewer.h)
    add_app(adop_viewer FALSE)


    find_package(OpenVR QUIET)
    if (${OPENVR_FOUND})
        add_executable(adop_vr_viewer adop_vr_viewer.cpp)
        add_app(adop_vr_viewer FALSE)
    endif ()
endif ()

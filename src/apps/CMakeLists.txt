function(add_app TARGET_NAME)

    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_include_directories(${TARGET_NAME} PUBLIC ".")
    target_link_libraries(${TARGET_NAME} NeuralPoints)

    if (PR_NO_WINDOW)
        target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
    endif ()


    message(STATUS "App enabled:       ${TARGET_NAME}")
endfunction()

add_executable(colmap2adop colmap2adop.cpp)
add_app(colmap2adop)

add_executable(preprocess_pointcloud preprocess_pointcloud.cpp)
add_app(preprocess_pointcloud)

add_executable(adop_train adop_train.cpp)
add_app(adop_train)

if (TARGET saiga_opengl)
    add_executable(adop_viewer adop_viewer.cpp adop_viewer.h)
    add_app(adop_viewer)


    find_package(OpenVR QUIET)
    if (${OPENVR_FOUND})
        add_executable(adop_vr_viewer adop_vr_viewer.cpp)
        add_app(adop_vr_viewer)
    endif ()
endif ()
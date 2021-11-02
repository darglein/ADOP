function(saiga_test FILE_NAME)
    string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

    set(TARGET_NAME ${NAME})

    add_executable(${TARGET_NAME} ${FILE_NAME})
    message(STATUS "Test enabled:      ${NAME}")

    target_link_libraries(${TARGET_NAME} PUBLIC NeuralPoints gtest gtest_main)
    target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

    add_test(
            NAME ${TARGET_NAME}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_NAME}
    )

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER tests/${PREFIX})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests")
endfunction()

saiga_test(test_loss.cpp)
add_subdirectory(lib)
if(UNIX)
#currently, tests are not supported
add_subdirectory(tests)
endif()
add_subdirectory(apps)

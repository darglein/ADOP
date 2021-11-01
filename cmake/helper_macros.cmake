macro(GroupSources startDir curdir)
    file(GLOB children RELATIVE ${startDir}/${curdir} ${startDir}/${curdir}/*)
    foreach(child ${children})
        if(IS_DIRECTORY ${startDir}/${curdir}/${child})
            GroupSources(${startDir} ${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${startDir}/${curdir}/${child})
        endif()
    endforeach()
endmacro()

macro(GroupSources2 startDir)
    file(GLOB children RELATIVE ${startDir} ${startDir}/*)
    foreach(child ${children})
        if(IS_DIRECTORY ${startDir}/${child})
            GroupSources(${startDir} ${child})
        else()
            source_group("" FILES ${startDir}/${child})
        endif()
    endforeach()
endmacro()

macro(OptionsHelper _variableName _description _defaultValue)
    option (${_variableName} "${_description}" "${_defaultValue}")
    # Create a padding string to align the console output
    string(LENGTH ${_variableName} SIZE)
    math(EXPR SIZE 20-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Option ${_variableName} ${padding} ${${_variableName}}")
endmacro()

macro(PackageHelper _name _found _include_dir _libraries)
    if (${_found})
        SET(LIB_MSG "Yes")
        SET(LIBS ${LIBS} ${_libraries})

        if(NOT "${_include_dir}" STREQUAL "" )
            #include_directories(${_include_dir})
            SET(PACKAGE_INCLUDES ${PACKAGE_INCLUDES} ${_include_dir})
            SET(LIB_MSG "Yes, at ${_include_dir}")
        endif()
    else ()
        SET(LIB_MSG "No")
    endif ()
    # Create a padding string to align the console output
    string(LENGTH ${_name} SIZE)
    math(EXPR SIZE 25-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Package ${_name} ${padding} ${LIB_MSG}")
endmacro()

macro(PackageHelperTarget _target _found)
    if (TARGET ${_target})
        SET(LIB_MSG "Yes")
        SET(LIB_TARGETS ${LIB_TARGETS} ${_target})
        set(${_found} 1)
        get_target_property(tmp_interface_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT "${tmp_interface_includes}" STREQUAL "" )
            SET(LIB_MSG "Yes, at ${tmp_interface_includes}")
        endif()
    else ()
        SET(LIB_MSG "No")
        set(${_found} 0)
    endif ()
    # Create a padding string to align the console output
    string(LENGTH ${_target} SIZE)
    math(EXPR SIZE 25-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Package ${_target} ${padding} ${LIB_MSG}")
endmacro()

####################################################################################

macro(DefaultBuildType _default_value)
    # Set a default build type if none was specified
    set(default_build_type ${_default_value})

    # Create build type drop down
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()

    message(STATUS "\nBuild Type: ${CMAKE_BUILD_TYPE}")
endmacro()

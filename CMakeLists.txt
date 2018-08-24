cmake_minimum_required(VERSION 3.1)
project (deedsBCV)

set (CMAKE_CXX_STANDARD 11)

# Versionning
# ===========
set(CPP_CONFIG_HEADER_NAME deeds_config.h)
set(PROJECT_CONFIG_FILE src/${CPP_CONFIG_HEADER_NAME})

file(STRINGS ${PROJECT_CONFIG_FILE} project_version_defines
     REGEX "#define DEEDS_VERSION_(MAJOR|MINOR|PATCH)")
message(STATUS ${PROJECT_CONFIG_FILE})

foreach(ver ${project_version_defines})
    if(ver MATCHES "#define DEEDS_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(${PROJECT_NAME}_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(${PROJECT_NAME}_VERSION
    ${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH})
message(STATUS "Building project v${${PROJECT_NAME}_VERSION}")

find_package( ZLIB REQUIRED )

add_subdirectory("${PROJECT_SOURCE_DIR}/src")

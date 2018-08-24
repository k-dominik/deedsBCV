cmake_minimum_required(VERSION 3.1)
project (deedsBCV)

set (CMAKE_CXX_STANDARD 11)

find_package( ZLIB REQUIRED )

add_subdirectory("${PROJECT_SOURCE_DIR}/src")

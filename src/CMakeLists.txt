add_executable(deedsBCV0 deedsBCV0.cpp)
add_executable(linearBCV linearBCV.cpp)
add_executable(applyBCV applyBCV.cpp)

if ( ZLIB_FOUND )
    include_directories( ${ZLIB_INCLUDE_DIRS} )
    target_link_libraries( deedsBCV0 ${ZLIB_LIBRARIES} )
    target_link_libraries( linearBCV ${ZLIB_LIBRARIES} )
    target_link_libraries( applyBCV ${ZLIB_LIBRARIES} )
endif( ZLIB_FOUND )

install (TARGETS deedsBCV0 linearBCV applyBCV DESTINATION bin)

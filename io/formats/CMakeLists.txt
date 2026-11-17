# nxr_io_formats: the pure readers and writers of the field's file formats (bytes <-> plain C++ objects). Included by ../CMakeLists.txt.
# Its rule (cmake/check_formats_layering.cmake): no store, database, writer, dataset or package header; the one nxr/io header it may include
# is errors.h. nxr_io links it, and so can the wheel, the MEX and the WebAssembly client, without the store.
add_library(nxr_io_formats
  src/byte_source.cpp
)
target_include_directories(nxr_io_formats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# errors.h is the one nxr/io header a pure format sees: the io include folder, for that header only (the layering check holds the line)
target_include_directories(nxr_io_formats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
if(MSVC)
  target_compile_options(nxr_io_formats PRIVATE /W4)
else()
  target_compile_options(nxr_io_formats PRIVATE -Wall -Wextra)
endif()

if(NXR_IO_BUILD_TESTS)
  set(NXR_FORMATS_TESTS byte_source recording)
  foreach(t ${NXR_FORMATS_TESTS})
    add_executable(test_formats_${t} test/test_${t}.cpp)
    target_link_libraries(test_formats_${t} PRIVATE nxr_io_formats)   # the formats alone: a test that needs the store does not belong here
    target_include_directories(test_formats_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../test)   # io's check.h
    target_compile_definitions(test_formats_${t} PRIVATE NXR_FORMATS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test" NXR_FORMATS_WORK_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME test_formats_${t} COMMAND test_formats_${t})
  endforeach()
  # the layering rule, and the check itself (a bad include is caught, a good tree passes)
  add_test(NAME formats_layering COMMAND ${CMAKE_COMMAND} -DFORMATS_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/check_formats_layering.cmake)
  add_test(NAME formats_layering_check_catches COMMAND ${CMAKE_COMMAND} -DCHECK=${CMAKE_SOURCE_DIR}/cmake/check_formats_layering.cmake -DWORK=${CMAKE_CURRENT_BINARY_DIR}/layering_work -P ${CMAKE_CURRENT_SOURCE_DIR}/test/layering_catches.cmake)
endif()

# nxr_io_formats: the pure readers and writers of the field's file formats (bytes <-> plain C++ objects). Included by ../CMakeLists.txt.
# Its rule (cmake/check_formats_layering.cmake): no store, database, writer, dataset or package header; the one nxr/io header it may include
# is errors.h. nxr_io links it, and so can the wheel, the MEX and the WebAssembly client, without the store.
add_library(nxr_io_formats
  src/byte_source.cpp
  src/deflate.cpp
  src/nifti.cpp
  src/gifti.cpp
  src/cifti.cpp
  src/tsv.cpp
  src/bids.cpp
  src/edf.cpp
  src/brainvision.cpp
  src/eeglab.cpp
  src/fif_tag.cpp
  src/fif_info.cpp
  src/fif_raw.cpp
  src/geometry.cpp
  src/ctf.cpp
  src/kit.cpp
  src/fif_write.cpp
  src/fif_objects.cpp
  src/stc.cpp
  src/zip.cpp
  src/trx.cpp
)

# pugixml (MIT; one new dependency, the XML of GIFTI): its single source file compiled into the library, its warnings not ours
include(FetchContent)
FetchContent_Declare(pugixml
  URL https://github.com/zeux/pugixml/releases/download/v1.14/pugixml-1.14.tar.gz
  URL_HASH SHA256=2f10e276870c64b1db6809050a75e11a897a8d7456c4be5c6b2e35a11168a015
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(pugixml)
if(NOT pugixml_POPULATED)
  FetchContent_Populate(pugixml)
endif()
target_sources(nxr_io_formats PRIVATE ${pugixml_SOURCE_DIR}/src/pugixml.cpp)
set_source_files_properties(${pugixml_SOURCE_DIR}/src/pugixml.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_include_directories(nxr_io_formats PRIVATE ${pugixml_SOURCE_DIR}/src)
if(EMSCRIPTEN)
  target_compile_options(nxr_io_formats PUBLIC -sUSE_ZLIB=1)   # Emscripten's zlib port
else()
  find_package(ZLIB REQUIRED)
  target_link_libraries(nxr_io_formats PUBLIC ZLIB::ZLIB)
endif()
target_include_directories(nxr_io_formats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# errors.h is the one nxr/io header a pure format sees: the io include folder, for that header only (the layering check holds the line)
target_include_directories(nxr_io_formats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(nxr_io_formats PUBLIC nlohmann_json PRIVATE matio)   # the JSON sidecars; matio reads the MATLAB files of EEGLAB
if(MSVC)
  target_compile_options(nxr_io_formats PRIVATE /W4)
else()
  target_compile_options(nxr_io_formats PRIVATE -Wall -Wextra)
endif()

if(NXR_IO_BUILD_TESTS)
  set(NXR_FORMATS_TESTS byte_source recording nifti gifti tsv edf brainvision eeglab fif fif_objects bids ctf kit cifti stc zip trx)
  foreach(t ${NXR_FORMATS_TESTS})
    add_executable(test_formats_${t} test/test_${t}.cpp)
    target_link_libraries(test_formats_${t} PRIVATE nxr_io_formats)   # the formats alone: a test that needs the store does not belong here
    target_include_directories(test_formats_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../test ${CMAKE_CURRENT_SOURCE_DIR}/../extern)   # io's check.h; nlohmann/json to read the goldens
    target_compile_definitions(test_formats_${t} PRIVATE NXR_FORMATS_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test" NXR_FORMATS_WORK_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME test_formats_${t} COMMAND test_formats_${t})
  endforeach()
  # the layering rule, and the check itself (a bad include is caught, a good tree passes)
  add_test(NAME formats_layering COMMAND ${CMAKE_COMMAND} -DFORMATS_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/check_formats_layering.cmake)
  add_test(NAME formats_layering_check_catches COMMAND ${CMAKE_COMMAND} -DCHECK=${CMAKE_SOURCE_DIR}/cmake/check_formats_layering.cmake -DWORK=${CMAKE_CURRENT_BINARY_DIR}/layering_work -P ${CMAKE_CURRENT_SOURCE_DIR}/test/layering_catches.cmake)
endif()

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS io_formats)

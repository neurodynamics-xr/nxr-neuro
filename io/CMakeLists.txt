# nxr_io — included by ../CMakeLists.txt (the top-level project, C++17 floor and options are there).

# WEBASSEMBLY (Emscripten; bindings/wasm-io): nxr_io WHOLE — the core and the Brainstorm importers. matio reads MATLAB v4–v7 .mat files with zlib; WITHOUT HDF5,
# so a v7.3 file (an HDF5 file) is refused with matio's error. SQLite is the 3.51 amalgamation, zlib Emscripten's port; zstd is the copy c-blosc carries
# (one zstd for both codecs; c-blosc builds it into blosc_static).
if(EMSCRIPTEN)
  set(NXR_IO_WASM ON)
else()
  set(NXR_IO_WASM OFF)
endif()

if(NOT NXR_IO_WASM)
# SQLite is ALWAYS the pinned 3.51 amalgamation, compiled into nxr_sqlite3 with the same options on every platform: one database engine, one
# behaviour (the system SQLite differs by platform and age: Apple's lacks ORDER BY inside aggregates, which the hub's Homebrew and Ubuntu 24.04 have).
include(FetchContent)
enable_language(C)
FetchContent_Declare(sqlite_amalgamation
  URL https://www.sqlite.org/2025/sqlite-amalgamation-3510000.zip
  URL_HASH SHA256=1caf7116f2910600d04473ad69d37ec538fa62fa36adccd37b5e0e43647c98be
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(sqlite_amalgamation)
add_library(nxr_sqlite3 STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
target_include_directories(nxr_sqlite3 PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
target_compile_definitions(nxr_sqlite3 PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION=1 SQLITE_ENABLE_MATH_FUNCTIONS=1 SQLITE_DEFAULT_FOREIGN_KEYS=0)
target_compile_options(nxr_sqlite3 PRIVATE -w)
set_target_properties(nxr_sqlite3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(nxr_sqlite3 PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
add_library(SQLite3::SQLite3 ALIAS nxr_sqlite3)
find_package(ZLIB REQUIRED)
# NXR_NEURO_PORTABLE (release builds): zstd from the copy c-blosc carries, linked into the binary, so the command needs no Homebrew or
# distribution libraries (zlib stays the system's: libz.a on Linux, the system libz on macOS).
if(NXR_NEURO_PORTABLE)
  # zstd: set after c-blosc is fetched (below)
else()
# --- libzstd (system; the Zarr v3 `zstd` codec body) ---
find_path(ZSTD_INCLUDE_DIR zstd.h HINTS /opt/homebrew/include /usr/local/include /usr/include)
find_library(ZSTD_LIBRARY NAMES zstd HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
if(NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
  message(FATAL_ERROR "libzstd not found (try: brew install zstd)")
endif()

endif()
endif()

# --- c-blosc 1.x (fetched and built inside the build directory; the `blosc` codec, nxr writes zstd · level 3 · shuffle) ---
include(FetchContent)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(DEACTIVATE_SNAPPY ON CACHE BOOL "" FORCE)
if(NXR_IO_WASM)
  set(PREFER_EXTERNAL_ZSTD OFF CACHE BOOL "" FORCE)   # its own zstd (also ours, below)
  set(PREFER_EXTERNAL_ZLIB OFF CACHE BOOL "" FORCE)
  set(DEACTIVATE_SSE2 ON CACHE BOOL "" FORCE)         # no x86 SIMD in WebAssembly
  set(DEACTIVATE_AVX2 ON CACHE BOOL "" FORCE)
else()
  if(NXR_NEURO_PORTABLE)
    set(PREFER_EXTERNAL_ZSTD OFF CACHE BOOL "" FORCE)   # c-blosc builds its own zstd into blosc_static; nxr_io uses that one
  else()
    set(PREFER_EXTERNAL_ZSTD ON CACHE BOOL "" FORCE)
  endif()
  set(PREFER_EXTERNAL_ZLIB ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(c-blosc
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG v1.21.6
  GIT_SHALLOW TRUE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # c-blosc 1.x declares an old minimum
FetchContent_MakeAvailable(c-blosc)
target_compile_options(blosc_static PRIVATE -w)  # third-party code: its warnings are not ours
if(NXR_NEURO_PORTABLE AND NOT NXR_IO_WASM)
  set(ZSTD_INCLUDE_DIR ${c-blosc_SOURCE_DIR}/internal-complibs/zstd-1.5.6)
  set(ZSTD_LIBRARY "")
endif()

if(NXR_IO_WASM)
  set(ZSTD_INCLUDE_DIR ${c-blosc_SOURCE_DIR}/internal-complibs/zstd-1.5.6)
  set(ZSTD_LIBRARY "")
  add_library(nlohmann_json INTERFACE)
  target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/model.sql NXR_MODEL_SQL)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/model_sql.inc "R\"NXRMODEL(${NXR_MODEL_SQL})NXRMODEL\"")
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures.json NXR_MEASURES_JSON)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/measures_json.inc "R\"NXRMEAS(${NXR_MEASURES_JSON})NXRMEAS\"")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures_nxr.json NXR_MEASURES_NXR_JSON)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/measures_nxr_json.inc "R\"NXRMEAS(${NXR_MEASURES_NXR_JSON})NXRMEAS\"")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures_nxr.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/statistics.json NXR_STATISTICS_JSON)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/statistics_json.inc "R\"NXRSTAT(${NXR_STATISTICS_JSON})NXRSTAT\"")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/statistics.json)
  # --- matio without HDF5 (the same release as the native build); its zlib is Emscripten's port ---
  execute_process(COMMAND em-config CACHE OUTPUT_VARIABLE NXR_EM_CACHE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(ZLIB_INCLUDE_DIR ${NXR_EM_CACHE}/sysroot/include CACHE PATH "" FORCE)
  set(ZLIB_LIBRARY ${NXR_EM_CACHE}/sysroot/lib/wasm32-emscripten/libz.a CACHE FILEPATH "" FORCE)
  set(MATIO_WITH_HDF5 OFF CACHE BOOL "" FORCE)
  set(MATIO_MAT73 OFF CACHE BOOL "" FORCE)
  set(MATIO_WITH_ZLIB ON CACHE BOOL "" FORCE)
  set(MATIO_SHARED OFF CACHE BOOL "" FORCE)
  set(MATIO_EXTENDED_SPARSE ON CACHE BOOL "" FORCE)
  set(MATIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(MATIO_ENABLE_CPPCHECK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(matio GIT_REPOSITORY https://github.com/tbeu/matio.git GIT_TAG v1.5.28 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(matio)
  target_compile_options(matio PRIVATE -w -sUSE_ZLIB=1)
  add_subdirectory(formats)   # the pure formats (own target; nxr_io links it)
  add_library(nxr_io
    src/codec_zstd.cpp src/codec_pipeline.cpp src/dtype.cpp src/metadata.cpp src/chunking.cpp src/zarr_store.cpp src/sparse.cpp src/schema.cpp
    src/dataset.cpp src/array_reader.cpp src/json_text.cpp src/database.cpp src/records.cpp src/selection.cpp src/task.cpp src/task_dataset.cpp src/named_paths.cpp src/layout.cpp src/writer_lock.cpp src/array_writer.cpp src/compact.cpp
    src/writer.cpp src/subject_writer.cpp src/mesh.cpp src/surface.cpp src/npy.cpp src/default_subject.cpp src/bundle.cpp src/cells.cpp src/measures.cpp src/spatial_bands.cpp src/rollup.cpp src/statistic.cpp src/yaml.cpp src/finder.cpp src/hash.cpp
    src/bst_mat.cpp src/bst_protocol.cpp src/bst_subject.cpp src/bst_entities.cpp src/bst_sensors.cpp src/bst_projectors.cpp src/bst_mri.cpp src/bst_naming.cpp
    src/bst_session.cpp src/bst_recording.cpp src/bst_events.cpp src/bst_resample.cpp src/bst_frequency.cpp src/bst_inverse.cpp src/bst_maps.cpp src/bst_timefreq.cpp
    src/bst_fibers.cpp src/fibers.cpp src/bst_templates.cpp src/chunk_loop.cpp src/constant_q.cpp src/store.cpp src/small_file_store.cpp
    src/subject_package.cpp src/subject_filter.cpp src/catalogue.cpp src/participants.cpp src/dataset_index.cpp src/migrate.cpp src/templates.cpp)
  target_include_directories(nxr_io PRIVATE ${CMAKE_CURRENT_BINARY_DIR} src ${ZSTD_INCLUDE_DIR} ${c-blosc_SOURCE_DIR}/blosc ${CMAKE_CURRENT_SOURCE_DIR}/extern)
  set_source_files_properties(src/yaml.cpp PROPERTIES COMPILE_OPTIONS "-w")   # rapidyaml (vendored) compiled here, its warnings not ours
  target_include_directories(nxr_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # -fwasm-exceptions: nxr_io reports every error by throwing (Emscripten turns C++ exceptions off by default, and a throw then aborts); WebAssembly's own
  # exception handling, in every current browser and Node
  target_compile_options(nxr_io PUBLIC -sUSE_ZLIB=1 -fwasm-exceptions PRIVATE -Wall -Wextra)
  # the writer lock's host bridges (writer_lock.cpp: process.pid, os.hostname, process.kill) return a string through stringToNewUTF8
  target_link_options(nxr_io PUBLIC -sUSE_ZLIB=1 -fwasm-exceptions "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$stringToNewUTF8")
  # SQLite 3.51.0 from its amalgamation, not Emscripten's port (3.39, 2022): the schema's views (model.sql, the Studio's) use ORDER BY inside
  # json_group_array, which needs 3.44 or newer. One thread (WebAssembly here has none), no extensions; JSON is built in.
  enable_language(C)
  FetchContent_Declare(sqlite_amalgamation
    URL https://www.sqlite.org/2025/sqlite-amalgamation-3510000.zip
    URL_HASH SHA256=1caf7116f2910600d04473ad69d37ec538fa62fa36adccd37b5e0e43647c98be
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(sqlite_amalgamation)
  add_library(nxr_sqlite3 STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
  target_include_directories(nxr_sqlite3 PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
  target_compile_definitions(nxr_sqlite3 PRIVATE SQLITE_THREADSAFE=0 SQLITE_OMIT_LOAD_EXTENSION=1 SQLITE_ENABLE_MATH_FUNCTIONS=1 SQLITE_DEFAULT_FOREIGN_KEYS=0)
  target_compile_options(nxr_sqlite3 PRIVATE -w -fwasm-exceptions)
  target_link_libraries(nxr_io PUBLIC nxr_io_formats nxr_selection nlohmann_json nxr_sqlite3 PRIVATE blosc_static matio)
  return()
endif()

# --- HDF5 (fetched and built inside the build directory; the file format of MATLAB v7.3 .mat files). The official HDFGroup release, C library
#     only (no Fortran, C++, high-level library, tools, examples or tests), static, thread-unsafe (matio is called from one thread at a time),
#     with the zlib filter (the system zlib found above; MATLAB's v7.3 datasets are deflated) and no szip. HDF5 marks its API
#     visible whatever the preset says, so keeping its symbols private is done at the final link of a shared object (a MEX or a Python
#     extension, which would otherwise clash with the HDF5 that MATLAB or h5py loads in the same process): see nxr_io_hide_static_symbols() below. ---
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_UTILS OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_THREADSAFE OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SUBFILING_VFD OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_ROS3_VFD OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_HDFS OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_MAP_API OFF CACHE BOOL "" FORCE)
set(HDF5_EXTERNALLY_CONFIGURED ON CACHE BOOL "" FORCE)
set(HDF5_ALLOW_EXTERNAL_SUPPORT NO CACHE STRING "" FORCE)
set(ZLIB_USE_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(hdf5
  GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
  GIT_TAG hdf5_1.14.6
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(hdf5)
set_target_properties(hdf5-static PROPERTIES C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_compile_options(hdf5-static PRIVATE -w)
if(NOT TARGET hdf5::hdf5-static)
  add_library(hdf5::hdf5-static ALIAS hdf5-static)   # the name matio looks for (an installed HDF5 defines it)
endif()
set(HDF5_VERSION "1.14.6")   # for matio's version check (matio finds this build, see cmake/patch_matio_hdf5.cmake)

# --- matio (fetched and built inside the build directory; reads Brainstorm's MATLAB v4 to v7.3 .mat files): zlib for v5/v7, the HDF5 above for v7.3 ---
set(MATIO_WITH_HDF5 ON CACHE BOOL "" FORCE)
set(MATIO_WITH_ZLIB ON CACHE BOOL "" FORCE)
set(MATIO_MAT73 ON CACHE BOOL "" FORCE)
set(HDF5_USE_STATIC_LIBRARIES TRUE CACHE BOOL "" FORCE)
set(MATIO_SHARED OFF CACHE BOOL "" FORCE)
set(MATIO_PIC ON CACHE BOOL "" FORCE)
set(MATIO_EXTENDED_SPARSE ON CACHE BOOL "" FORCE)
set(MATIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(MATIO_ENABLE_CPPCHECK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(matio
  GIT_REPOSITORY https://github.com/tbeu/matio.git
  GIT_TAG v1.5.28
  GIT_SHALLOW TRUE
  PATCH_COMMAND ${CMAKE_COMMAND} -DMATIO_SRC=<SOURCE_DIR> -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_matio_hdf5.cmake)
FetchContent_MakeAvailable(matio)

# For a shared object (MEX, Python extension) that links nxr_io: call nxr_io_hide_static_symbols(<target>) so that HDF5 and matio (static archives)
# export nothing from it. macOS: -load_hidden of their archives; GNU ld / lld: --exclude-libs. Verified on macOS (a dylib linking nxr_io
# exports no _H5*/_Mat_* symbol with it, 90+ without); the Linux branch is the documented flag, to be checked when the first binding is built there.
function(nxr_io_hide_static_symbols target)
  if(APPLE)
    target_link_options(${target} PRIVATE "LINKER:-load_hidden,$<TARGET_FILE:hdf5-static>" "LINKER:-load_hidden,$<TARGET_FILE:matio>")
  elseif(UNIX)
    target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
endfunction()

# --- nlohmann/json (vendored single header) ---
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern)

# --- the schema (model.sql, a byte copy of the Studio's) embedded as a raw string ---
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/model.sql NXR_MODEL_SQL)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/model_sql.inc "R\"NXRMODEL(${NXR_MODEL_SQL})NXRMODEL\"")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/model.sql)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures.json NXR_MEASURES_JSON)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/measures_json.inc "R\"NXRMEAS(${NXR_MEASURES_JSON})NXRMEAS\"")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures.json)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures_nxr.json NXR_MEASURES_NXR_JSON)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/measures_nxr_json.inc "R\"NXRMEAS(${NXR_MEASURES_NXR_JSON})NXRMEAS\"")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/measures_nxr.json)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/schema/statistics.json NXR_STATISTICS_JSON)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/statistics_json.inc "R\"NXRSTAT(${NXR_STATISTICS_JSON})NXRSTAT\"")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/schema/statistics.json)

# --- the pure formats (their own target; nxr_io links it) ---
add_subdirectory(formats)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/convert/manifest/template_sources.json NXR_TEMPLATE_SOURCES_JSON)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/template_sources_json.inc "R\"NXRSRC(${NXR_TEMPLATE_SOURCES_JSON})NXRSRC\"")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/convert/manifest/template_sources.json)

# --- core library (sources added as TDD adds them) ---
add_library(nxr_io
  src/codec_zstd.cpp
  src/codec_pipeline.cpp
  src/dtype.cpp
  src/metadata.cpp
  src/chunking.cpp
  src/zarr_store.cpp
  src/store.cpp
  src/small_file_store.cpp
  src/sparse.cpp
  src/schema.cpp
  src/dataset.cpp
  src/array_reader.cpp
  src/json_text.cpp
  src/database.cpp
  src/records.cpp
  src/selection.cpp
  src/task.cpp
  src/task_dataset.cpp
  src/named_paths.cpp
  src/layout.cpp
  src/writer_lock.cpp
  src/array_writer.cpp
  src/compact.cpp
  src/writer.cpp
  src/subject_writer.cpp
  src/subject_package.cpp
  src/subject_filter.cpp
  src/catalogue.cpp
  src/participants.cpp
  src/dataset_index.cpp
  src/migrate.cpp
  src/templates.cpp
  src/mesh.cpp
  src/surface.cpp
  src/npy.cpp
 
  src/default_subject.cpp
  src/bundle.cpp
  src/chunk_loop.cpp
  src/constant_q.cpp
  src/cells.cpp
  src/measures.cpp
  src/spatial_bands.cpp
  src/rollup.cpp
  src/statistic.cpp
  src/yaml.cpp
  src/finder.cpp
  src/hash.cpp
  src/bst_mat.cpp
  src/bst_protocol.cpp
  src/bst_subject.cpp
  src/bst_entities.cpp
  src/bst_sensors.cpp
  src/bst_projectors.cpp
  src/bst_mri.cpp
  src/bst_naming.cpp
  src/bst_session.cpp
  src/bst_recording.cpp
  src/bst_events.cpp
  src/bst_resample.cpp
  src/bst_frequency.cpp
  src/bst_inverse.cpp
  src/bst_maps.cpp
  src/bst_timefreq.cpp
  src/bst_fibers.cpp
  src/fibers.cpp
  src/bst_templates.cpp
  convert/trk.cpp
  convert/fibers_import.cpp
  convert/templates.cpp
  convert/recording.cpp
  convert/fif_objects.cpp
  convert/import_origin.cpp
  convert/nifti_import.cpp
  convert/world_frames.cpp
  convert/freesurfer_import.cpp
  convert/gifti_import.cpp
  convert/cifti_import.cpp
  convert/sensor_frames.cpp
  convert/recording_operators.cpp
  convert/bids.cpp
  convert/synthetic.cpp
  export/common.cpp
  export/gifti.cpp
  export/tsv.cpp
  export/cortex.cpp
  export/fif.cpp
  export/bids.cpp
)
target_include_directories(nxr_io PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/gen src ${CMAKE_CURRENT_SOURCE_DIR}/extern)
set_source_files_properties(src/yaml.cpp PROPERTIES COMPILE_OPTIONS "-w")   # rapidyaml (vendored: extern/ryml, v0.15.2) compiled here, its warnings not ours
target_include_directories(nxr_io
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${ZSTD_INCLUDE_DIR} ${c-blosc_SOURCE_DIR}/blosc)
target_link_libraries(nxr_io PUBLIC nxr_io_formats nxr_selection nlohmann_json SQLite3::SQLite3 ZLIB::ZLIB PRIVATE ${ZSTD_LIBRARY} blosc_static matio)
if(MSVC)
  target_compile_options(nxr_io PRIVATE /W4)
else()
  target_compile_options(nxr_io PRIVATE -Wall -Wextra)
endif()

# The test-only probe (io/include/nxr/io/testing/probe.h): linked by test executables only (cmake/check_records_layering.cmake).
add_library(nxr_io_testing STATIC testing/probe.cpp)
target_link_libraries(nxr_io_testing PUBLIC nxr_io)

# The conformance scenario written through the API (test/parity compares it with nxr's Python).
add_executable(nxr-neuro-scenario test/scenario_driver.cpp)
target_link_libraries(nxr-neuro-scenario PRIVATE nxr_io)

# The `bst` reading layer's dump (JSON of every variable of a .mat: class, size, values), for test/parity to compare with scipy's files.
add_executable(nxr-neuro-bst-dump test/bst_dump.cpp)
target_link_libraries(nxr-neuro-bst-dump PRIVATE nxr_io)

# Two .mat files read through nxr::bst and required equal (the v7.3 smoke test: a -v7.3 file against its -v7 twin).
add_executable(nxr-neuro-bst-compare test/bst_compare.cpp)
target_link_libraries(nxr-neuro-bst-compare PRIVATE nxr_io)

# --- tests (hand-written + ctest) ---
if(NXR_IO_BUILD_TESTS)
  set(NXR_IO_TESTS export_gifti export_tsv export_fif export_bids import_recording import_fif_objects import_nifti import_surfaces import_cifti import_bids synthetic records task codec codec_pipeline dtype array_io groups read_existing crud sparse schema writer bst default_subject value_types chunk_loop constant_q measures statistic store small_file_store rollup chunk_separator compact format52 subject_package subject_filter catalogue migrate templates template_import fibers schema51 split_groups task_dataset)
  foreach(t ${NXR_IO_TESTS})
    add_executable(test_${t} test/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE nxr_io nxr_io_testing)
    target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_compile_definitions(test_${t} PRIVATE
      NXR_IO_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
      NXR_IO_SCHEMA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/schema")
    add_test(NAME test_${t} COMMAND test_${t})
  endforeach()
  # the generated row structs are current with schema/model.sql (tools/gen_record_rows.py)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME records_rows_current COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_record_rows.py --check)
  endif()
  # v7.3 smoke: files MATLAB wrote with -v7.3 read as their -v7 twins (synthetic, test/data/mat-v73; regenerate with make_v73_smoke.m)
  set(NXR_V73_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/data/mat-v73)
  add_test(NAME bst_v73_tess_etc_vs_v7 COMMAND nxr-neuro-bst-compare ${NXR_V73_DIR}/t73.mat ${NXR_V73_DIR}/t7.mat)
  add_test(NAME bst_v73_utf8_vs_v7 COMMAND nxr-neuro-bst-compare ${NXR_V73_DIR}/utf73.mat ${NXR_V73_DIR}/utf7.mat)
  add_test(NAME store_identity COMMAND ${CMAKE_COMMAND} -DSCENARIO=$<TARGET_FILE:nxr-neuro-scenario> -DOBJ=${CMAKE_CURRENT_SOURCE_DIR}/test/data/identity_sphere.obj
           -DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/test/golden/store_identity.txt -DWORK=${CMAKE_CURRENT_BINARY_DIR}/store_identity_work -P ${CMAKE_CURRENT_SOURCE_DIR}/test/store_identity.cmake)
  # the layout of format 50 (c/0/1) stays readable: the written scenario turned into it equals the golden made before format 51
  add_test(NAME store_identity_v50_layout COMMAND ${CMAKE_COMMAND} -DLAYOUT=slash -DSCENARIO=$<TARGET_FILE:nxr-neuro-scenario> -DOBJ=${CMAKE_CURRENT_SOURCE_DIR}/test/data/identity_sphere.obj
           -DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/test/golden/store_identity_v50.txt -DWORK=${CMAKE_CURRENT_BINARY_DIR}/store_identity_v50_work -P ${CMAKE_CURRENT_SOURCE_DIR}/test/store_identity.cmake)
  # `dataset compact` on the format-50 layout gives the tree the writer makes now (golden/store_identity.txt), file by file
  add_test(NAME store_identity_v50_compacts COMMAND ${CMAKE_COMMAND} -DLAYOUT=slash -DCOMPACT=1 -DSCENARIO=$<TARGET_FILE:nxr-neuro-scenario> -DOBJ=${CMAKE_CURRENT_SOURCE_DIR}/test/data/identity_sphere.obj
           -DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/test/golden/store_identity.txt -DWORK=${CMAKE_CURRENT_BINARY_DIR}/store_identity_compact_work -P ${CMAKE_CURRENT_SOURCE_DIR}/test/store_identity.cmake)
  target_link_libraries(test_bst PRIVATE matio)   # the test writes its .mat files with matio itself
  target_link_libraries(test_template_import PRIVATE matio)
  target_link_libraries(test_statistic PRIVATE matio)

  # Optional Eigen interop (header-only). The core library never needs Eigen.
  option(NXR_IO_WITH_EIGEN "Build the optional Eigen interop test" OFF)
  if(NXR_IO_WITH_EIGEN)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
      HINTS /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
      message(FATAL_ERROR "NXR_IO_WITH_EIGEN=ON but Eigen headers not found")
    endif()
    add_executable(test_io_eigen test/test_eigen.cpp)
    target_link_libraries(test_io_eigen PRIVATE nxr_io)
    target_include_directories(test_io_eigen PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test ${EIGEN3_INCLUDE_DIR})
    target_compile_definitions(test_io_eigen PRIVATE
      NXR_IO_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
    add_test(NAME test_io_eigen COMMAND test_io_eigen)
  endif()
endif()

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS io)

# nxr_meeg — MEG/EEG source imaging. Included by ../CMakeLists.txt. Depends on Eigen and nxr_core only (dated amendment 2026-11-17; never nxr_compute, nxr_io, no file reading).
include(${CMAKE_CURRENT_SOURCE_DIR}/../compute/cmake/nxr_core.cmake)
add_library(nxr_meeg STATIC
  src/inverse.cpp
  src/sphere.cpp
  src/forward.cpp
  src/sources.cpp
  src/beamformer.cpp
  src/scouts.cpp
)
target_include_directories(nxr_meeg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(nxr_meeg PUBLIC cxx_std_17)
target_link_libraries(nxr_meeg PUBLIC Eigen3::Eigen nxr_core)
set_target_properties(nxr_meeg PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(nxr_meeg PRIVATE /W4)
else()
  target_compile_options(nxr_meeg PRIVATE -Wall -Wextra)
endif()

if(NXR_NEURO_BUILD_TESTS)
  foreach(t test_inverse test_inverse_parity test_forward test_forward_parity test_sources test_beamformer test_beamformer_parity test_beamformer_mne_parity)
    add_executable(${t} test/${t}.cpp)
    target_link_libraries(${t} PRIVATE nxr_meeg)
    target_compile_definitions(${t} PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME meeg_${t} COMMAND ${t})
  endforeach()
endif()

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS meeg)

# the kind of every test of this folder (tests/kinds.tsv); a test missing from the table fails the configure
if(COMMAND nxr_test_kinds)
  nxr_test_kinds()
endif()

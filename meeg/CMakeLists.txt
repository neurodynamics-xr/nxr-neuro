# nxr_meeg — MEG/EEG source imaging. Included by ../CMakeLists.txt. Depends on Eigen only (no nxr_compute, no nxr_io, no file reading).
add_library(nxr_meeg STATIC
  src/inverse.cpp
  src/covariance.cpp
  src/minimise.cpp
  src/sphere.cpp
  src/forward.cpp
  src/sources.cpp
  src/tensor.cpp
  src/wave.cpp
  src/crossspectrum.cpp
)
target_include_directories(nxr_meeg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(nxr_meeg PUBLIC cxx_std_17)
target_link_libraries(nxr_meeg PUBLIC Eigen3::Eigen)
set_target_properties(nxr_meeg PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(nxr_meeg PRIVATE /W4)
else()
  target_compile_options(nxr_meeg PRIVATE -Wall -Wextra)
endif()

if(NXR_NEURO_BUILD_TESTS)
  foreach(t test_inverse test_covariance test_inverse_parity test_forward test_forward_parity test_sources test_tensor test_wave test_crossspectrum_bins)
    add_executable(${t} test/${t}.cpp)
    target_link_libraries(${t} PRIVATE nxr_meeg)
    target_compile_definitions(${t} PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME meeg_${t} COMMAND ${t})
  endforeach()
endif()

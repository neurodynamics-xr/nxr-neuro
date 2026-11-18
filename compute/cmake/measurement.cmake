# nxr_compute: the measurement toolbox (compute/include/nxr/measurement/, compute/src/measurement/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/measurement/band_energy.cpp
  src/measurement/congruence_norms.cpp
  src/measurement/measurement_estimate.cpp
  src/measurement/measurement_measures.cpp
  src/measurement/measurement_reduce.cpp
  src/measurement/measurement_roll_up.cpp
  src/measurement/measurement_summarize.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_measurement test/test_measurement.cpp)
  target_link_libraries(test_measurement PRIVATE nxr_compute)
  target_compile_definitions(test_measurement PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_measurement COMMAND test_measurement)
endif()

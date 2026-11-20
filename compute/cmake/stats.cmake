# nxr_compute: the stats toolbox (compute/include/nxr/stats/, compute/src/stats/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/stats/bandwidth.cpp
  src/stats/circular.cpp
  src/stats/covariance.cpp
  src/stats/histogram.cpp
  src/stats/identification.cpp
  src/stats/lag.cpp
  src/stats/nulltest.cpp
  src/stats/pca.cpp
  src/stats/summary.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_stats test/test_stats.cpp)
  target_link_libraries(test_stats PRIVATE nxr_compute)
  target_compile_definitions(test_stats PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_stats COMMAND test_stats)
  add_executable(test_histogram test/test_histogram.cpp)
  target_link_libraries(test_histogram PRIVATE nxr_compute)
  target_compile_definitions(test_histogram PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_histogram COMMAND test_histogram)
  add_executable(test_pca test/test_pca.cpp)
  target_link_libraries(test_pca PRIVATE nxr_compute)
  target_compile_definitions(test_pca PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_pca COMMAND test_pca)
  add_executable(test_identification test/test_identification.cpp)
  target_link_libraries(test_identification PRIVATE nxr_compute)
  target_compile_definitions(test_identification PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_identification COMMAND test_identification)
  add_executable(test_covariance test/test_covariance.cpp)
  target_link_libraries(test_covariance PRIVATE nxr_compute)
  target_compile_definitions(test_covariance PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_covariance COMMAND test_covariance)
  add_executable(test_lag test/test_lag.cpp)
  target_link_libraries(test_lag PRIVATE nxr_compute)
  add_test(NAME test_lag COMMAND test_lag)
endif()

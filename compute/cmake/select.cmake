# nxr_compute: the select toolbox (compute/include/nxr/select/, compute/src/select/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/select/cores.cpp
  src/select/crossings.cpp
  src/select/distance_pairs.cpp
  src/select/extent.cpp
  src/select/intervals.cpp
  src/select/ladder.cpp
  src/select/regions.cpp
  src/select/risefall.cpp
  src/select/scale.cpp
  src/select/scalespace.cpp
  src/select/selection_bands.cpp
  src/select/selection_constant_q.cpp
  src/select/selection_dyadic.cpp
  src/select/selection_grid.cpp
  src/select/selection_partition.cpp
  src/select/selection_time_rule.cpp
  src/select/tracks.cpp
)

# ladder.cpp is a port of the Studio's ladder.ts (no FMA in JavaScript): contraction off, so areas, distances and the equal-mass cuts tie as they do there
set_source_files_properties(src/select/ladder.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(src/select/selection_dyadic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the Studio's ladder.ts, moved here
# measurement_measures.cpp / measurement_reduce.cpp are the Studio's measureOverPartition (no FMA in JavaScript): contraction off, so the sums equal the Studio's to the last bit
set_source_files_properties(src/measurement/measurement_measures.cpp src/measurement/measurement_reduce.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the same arithmetic, moved into the measurement module

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_intervals test/test_intervals.cpp)
  target_link_libraries(test_intervals PRIVATE nxr_compute)
  add_test(NAME test_intervals COMMAND test_intervals)
  add_executable(test_regions test/test_regions.cpp)
  target_link_libraries(test_regions PRIVATE nxr_compute)
  add_test(NAME test_regions COMMAND test_regions)
  add_executable(test_extent test/test_extent.cpp)
  target_link_libraries(test_extent PRIVATE nxr_compute)
  add_test(NAME test_extent COMMAND test_extent)
  add_executable(test_scale test/test_scale.cpp)
  target_link_libraries(test_scale PRIVATE nxr_compute)
  add_test(NAME test_scale COMMAND test_scale)
  add_executable(test_risefall test/test_risefall.cpp)
  target_link_libraries(test_risefall PRIVATE nxr_compute)
  target_compile_definitions(test_risefall PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_risefall COMMAND test_risefall)
  add_executable(test_burst_stages test/test_burst_stages.cpp)
  target_link_libraries(test_burst_stages PRIVATE nxr_compute)
  target_compile_definitions(test_burst_stages PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_burst_stages COMMAND test_burst_stages)
  add_executable(test_scalespace test/test_scalespace.cpp)
  target_link_libraries(test_scalespace PRIVATE nxr_compute)
  add_test(NAME test_scalespace COMMAND test_scalespace)
  add_executable(test_cores test/test_cores.cpp)
  target_link_libraries(test_cores PRIVATE nxr_compute)
  target_compile_definitions(test_cores PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_cores COMMAND test_cores)
  add_executable(test_tracks test/test_tracks.cpp)
  target_link_libraries(test_tracks PRIVATE nxr_compute)
  target_compile_definitions(test_tracks PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_tracks COMMAND test_tracks)
  add_executable(test_crossings test/test_crossings.cpp)
  target_link_libraries(test_crossings PRIVATE nxr_compute)
  target_compile_definitions(test_crossings PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_crossings COMMAND test_crossings)
  add_executable(test_ladder test/test_ladder.cpp)
  target_link_libraries(test_ladder PRIVATE nxr_compute)
  target_compile_definitions(test_ladder PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_ladder COMMAND test_ladder)
  add_executable(test_group_tree test/test_group_tree.cpp)
  target_link_libraries(test_group_tree PRIVATE nxr_compute)
  target_compile_definitions(test_group_tree PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_group_tree COMMAND test_group_tree)
  add_executable(test_time_tiles test/test_time_tiles.cpp)
  target_link_libraries(test_time_tiles PRIVATE nxr_compute)
  target_compile_definitions(test_time_tiles PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_time_tiles COMMAND test_time_tiles)
endif()

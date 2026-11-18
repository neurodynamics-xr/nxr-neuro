# nxr_compute: the synthesis toolbox (compute/include/nxr/synthesis/, compute/src/synthesis/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/synthesis/field_generators.cpp
  src/synthesis/lagfree.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_field_generators test/test_field_generators.cpp)
  target_link_libraries(test_field_generators PRIVATE nxr_compute)
  add_test(NAME test_field_generators COMMAND test_field_generators)
  add_executable(test_visualization_primitives test/test_visualization_primitives.cpp)
  target_link_libraries(test_visualization_primitives PRIVATE nxr_compute)
  add_test(NAME test_visualization_primitives COMMAND test_visualization_primitives)
  add_executable(test_surrogate test/test_surrogate.cpp)
  target_link_libraries(test_surrogate PRIVATE nxr_compute)
  target_compile_definitions(test_surrogate PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_surrogate COMMAND test_surrogate)
endif()

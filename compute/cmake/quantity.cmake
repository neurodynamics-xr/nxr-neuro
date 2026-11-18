# nxr_compute: the quantity toolbox (compute/include/nxr/quantity/, compute/src/quantity/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/quantity/quantity.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_quantity test/test_quantity.cpp)
  target_link_libraries(test_quantity PRIVATE nxr_compute)
  add_test(NAME test_quantity COMMAND test_quantity)
endif()

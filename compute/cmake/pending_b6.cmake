# nxr_compute: the top-level sources that stay until the behaviour batch deletes them (pyramid.h, region.h, integrals.h and rotary.h are in the plan's removed list).
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/integrals.cpp
  src/pyramid.cpp
  src/region.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_region test/test_region.cpp)
  target_link_libraries(test_region PRIVATE nxr_compute)
  add_test(NAME test_region COMMAND test_region)
  add_executable(test_integrals test/test_integrals.cpp)
  target_link_libraries(test_integrals PRIVATE nxr_compute)
  target_compile_definitions(test_integrals PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_integrals COMMAND test_integrals)
endif()

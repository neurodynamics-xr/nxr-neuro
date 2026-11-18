# nxr_compute: the core toolbox (compute/include/nxr/core/, compute/src/core/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/. The library nxr_core (core + optimisation, Eigen and threads only) is defined in nxr_core.cmake, which meeg/CMakeLists.txt also includes.

include(${CMAKE_CURRENT_LIST_DIR}/nxr_core.cmake)
target_link_libraries(nxr_compute PUBLIC nxr_core)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_scaling test/test_scaling.cpp)
  target_link_libraries(test_scaling PRIVATE nxr_compute)
  add_test(NAME test_scaling COMMAND test_scaling)
  add_executable(test_cancellation test/test_cancellation.cpp)
  target_link_libraries(test_cancellation PRIVATE nxr_compute)
  add_test(NAME test_cancellation COMMAND test_cancellation)
  add_executable(test_progress test/test_progress.cpp)
  target_link_libraries(test_progress PRIVATE nxr_compute)
  add_test(NAME test_progress COMMAND test_progress)
  add_executable(test_parallel_allocation test/test_parallel_allocation.cpp)
  target_link_libraries(test_parallel_allocation PRIVATE nxr_compute)
  add_test(NAME test_parallel_allocation COMMAND test_parallel_allocation)
endif()

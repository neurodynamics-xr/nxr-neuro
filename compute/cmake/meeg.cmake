# nxr_compute: the meeg toolbox (compute/include/nxr/meeg/, compute/src/meeg/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/meeg/connectome.cpp
  src/meeg/montage.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_montage test/test_montage.cpp)   # the 10-05 rule: names, a sphere against MNE (golden/montage_golden.txt, golden/gen_montage_golden.py), an ellipsoid
  target_link_libraries(test_montage PRIVATE nxr_compute)
  target_compile_definitions(test_montage PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_montage COMMAND test_montage)
  add_executable(test_connectome test/test_connectome.cpp)
  target_link_libraries(test_connectome PRIVATE nxr_compute)
  add_test(NAME test_connectome COMMAND test_connectome)
  add_executable(test_connectome_golden test/test_connectome_golden.cpp)   # against the MATLAB reference (golden/connectome_golden.txt, golden/gen_connectome_golden.m)
  target_link_libraries(test_connectome_golden PRIVATE nxr_compute)
  target_compile_definitions(test_connectome_golden PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_connectome_golden COMMAND test_connectome_golden)
  add_executable(test_connectome_partition test/test_connectome_partition.cpp)
  target_link_libraries(test_connectome_partition PRIVATE nxr_compute)
  add_test(NAME test_connectome_partition COMMAND test_connectome_partition)
  add_executable(test_connectome_gate test/test_connectome_gate.cpp)
  target_link_libraries(test_connectome_gate PRIVATE nxr_compute)
  add_test(NAME test_connectome_gate COMMAND test_connectome_gate)
  add_executable(test_connectome_registration test/test_connectome_registration.cpp)
  target_link_libraries(test_connectome_registration PRIVATE nxr_compute)
  add_test(NAME test_connectome_registration COMMAND test_connectome_registration)
endif()

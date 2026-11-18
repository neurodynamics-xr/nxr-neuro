# Benchmarks (NXR_BUILD_BENCH, OFF by default). Never part of the test suite: nothing here is add_test'd; tools/bench.py runs the binaries and compares them to a baseline.
# Each benchmark times a callable on a fixed synthetic size and prints one JSON line (bench/bench.h).
foreach(b wt_at psd isolines)
  add_executable(bench_${b} bench_${b}.cpp)
  target_link_libraries(bench_${b} PRIVATE nxr_compute)
  target_include_directories(bench_${b} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/compute/test)   # the synthetic-mesh helpers are compute's test headers
endforeach()

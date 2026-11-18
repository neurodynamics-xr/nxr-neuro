// A tiny benchmark harness. Time a callable N times on a fixed synthetic size and print ONE JSON line: name, parameters, runs, median and min in milliseconds.
// Benchmarks are never part of the test suite (never add_test): performance is measured here and compared by tools/bench.py.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace nxr_bench {

// Runs `f` once untimed (warm-up: caches, allocations), then `runs` times timed; prints {"name":..,"params":..,"runs":..,"median_ms":..,"min_ms":..}.
template <class F>
void run(const std::string& name, const std::string& params, int runs, F&& f) {
  using clock = std::chrono::steady_clock;
  f();
  std::vector<double> ms;
  for (int i = 0; i < runs; ++i) {
    const auto t0 = clock::now();
    f();
    ms.push_back(std::chrono::duration<double, std::milli>(clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const double median = ms.size() % 2 ? ms[ms.size() / 2] : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
  std::printf("{\"name\":\"%s\",\"params\":\"%s\",\"runs\":%d,\"median_ms\":%.4f,\"min_ms\":%.4f}\n", name.c_str(), params.c_str(), runs, median, ms.front());
}

// Keeps a result alive so the compiler cannot drop the timed work.
template <class T>
void keep(const T& v) {
  static volatile double sink = 0;
  sink = sink + static_cast<double>(v);
}

}  // namespace nxr_bench

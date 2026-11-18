// Test helper: an estimator of nxr::signal::spectral fed a record in chunks the way the executor feeds it: each block owns `chunk` samples and carries `halo` samples of right
// context (n - 1 for a segment of n samples), fewer where the record ends. A loop over chunks belongs to io; a test stands in for io here.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <vector>
namespace nxrtest {
template <class Acc, class Matrix>
void feed_chunks(Acc& acc, const Matrix& data, const std::vector<std::uint8_t>* good, std::int64_t chunk, std::int64_t halo) {
  const std::int64_t T = data.cols();
  for (std::int64_t a = 0; a < T; a += chunk) {
    const std::int64_t own = std::min(chunk, T - a), w = std::min(own + halo, T - a);
    const Matrix block = data.middleCols(a, w);
    const std::vector<std::uint8_t> g = good ? std::vector<std::uint8_t>(good->begin() + a, good->begin() + a + w) : std::vector<std::uint8_t>(static_cast<std::size_t>(w), 1);
    acc.add(block, g, a, own);
  }
}
}  // namespace nxrtest

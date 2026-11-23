// nxr::meeg::channelPositions (see the header).
#include "nxr/meeg/channels.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nxr::meeg {

std::vector<double> channelPositions(const std::vector<double>& points, const std::vector<double>& coils, const std::vector<std::int32_t>& indptr,
                                     const std::vector<std::int32_t>& indices, int coilPoints) {
  if (points.size() % 3 != 0) throw std::invalid_argument("channelPositions: the points are not [M x 3]");
  if (coilPoints < 1) throw std::invalid_argument("channelPositions: coilPoints is at least 1");
  std::vector<double> out = points;
  if (indptr.empty() && indices.empty() && coils.empty()) return out;
  const std::size_t M = points.size() / 3;
  if (coils.size() % 3 != 0) throw std::invalid_argument("channelPositions: the coils are not [Q x 3]");
  if (indptr.size() != M + 1) throw std::invalid_argument("channelPositions: indptr has " + std::to_string(indptr.size()) + " entries for " + std::to_string(M) + " channels (M + 1)");
  const std::int64_t Q = static_cast<std::int64_t>(coils.size() / 3);
  for (std::size_t m = 0; m < M; ++m) {
    const std::int64_t a = indptr[m], b = indptr[m + 1];
    if (a < 0 || b < a || b > static_cast<std::int64_t>(indices.size())) throw std::invalid_argument("channelPositions: indptr is not a CSR row pointer over the indices");
    const std::int64_t k = std::min<std::int64_t>(b - a, coilPoints);
    if (k == 0) continue;
    double s[3] = {0, 0, 0};
    for (std::int64_t j = a; j < a + k; ++j) {
      const std::int64_t q = indices[static_cast<std::size_t>(j)];
      if (q < 0 || q >= Q) throw std::invalid_argument("channelPositions: a coil index is outside the coils");
      for (int c = 0; c < 3; ++c) s[c] += coils[static_cast<std::size_t>(3 * q + c)];
    }
    for (int c = 0; c < 3; ++c) out[3 * m + static_cast<std::size_t>(c)] = s[c] / static_cast<double>(k);
  }
  return out;
}

}  // namespace nxr::meeg

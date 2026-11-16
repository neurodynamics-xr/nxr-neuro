// numpy's float64 reductions, as numpy does them (pairwise summation), for the numbers nxr's Python computes with np.sum / np.mean.
#pragma once
#include <cstddef>

namespace nxr::bst {

inline double pairwise_sum(const double* a, std::size_t n) {
  if (n < 8) {
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// np.sum of a float64 array: the identity 0 plus the pairwise sum of all of it
inline double np_sum(const double* a, std::size_t n) { return 0.0 + pairwise_sum(a, n); }

}  // namespace nxr::bst

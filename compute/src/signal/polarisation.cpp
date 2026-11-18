// nxr::signal — polarisation of a source current's cross-spectral tensor (polarisation.h).
#include "nxr/signal/polarisation.h"

namespace nxr::signal {

namespace {
// e_a^T M e_b for a 3x3 row-major M
double form(const double* M, const std::array<double, 3>& a, const std::array<double, 3>& b) {
  double s = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) s += a[static_cast<std::size_t>(i)] * M[3 * i + j] * b[static_cast<std::size_t>(j)];
  return s;
}
}  // namespace

Stokes stokes(const double* re, const double* im, const std::array<double, 3>& e1, const std::array<double, 3>& e2) {
  const double p11 = form(re, e1, e1), p22 = form(re, e2, e2), p12re = form(re, e1, e2), p12im = form(im, e1, e2);
  return Stokes{p11 + p22, p11 - p22, 2 * p12re, 2 * p12im};
}

double normalPower(const double* re, const std::array<double, 3>& n) { return form(re, n, n); }

double tracePower(const double* re) { return re[0] + re[4] + re[8]; }

}  // namespace nxr::signal

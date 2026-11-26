// decimate: scipy.signal.decimate's FIR route with zero_phase (resample_poly(x, 1, q, window=firwin(20 q + 1, 1 / q))), written as the centred convolution at the kept samples.
#include "nxr/signal/decimation.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "nxr/signal/filters_fir_design.h"

namespace nxr::signal {

int decimationOrder(int q, int order) {
  if (q < 1) throw std::invalid_argument("decimate: the factor q must be at least 1, got " + std::to_string(q));
  if (order == 0) return 20 * q;
  if (order < 2 || order % 2) throw std::invalid_argument("decimate: the low-pass order must be even and at least 2 (a centred filter), got " + std::to_string(order));
  return order;
}

Eigen::Index decimationHalo(int q, int order) { return q == 1 ? 0 : decimationOrder(q, order) / 2; }

namespace {

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> decimated(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& x, int q, Eigen::Index left, Eigen::Index width, Eigen::Index offset,
                                                          int order) {
  const int n = decimationOrder(q, order);
  const Eigen::Index cols = x.cols();
  if (width < 0) width = cols - left;
  if (left < 0 || left > cols || left + width > cols) throw std::invalid_argument("decimate: the own columns [left, left + width) must lie in the block's " + std::to_string(cols) + " columns");
  if (offset < 0 || offset >= q) throw std::invalid_argument("decimate: the offset of the first kept column must be in [0, q), got " + std::to_string(offset));
  const Eigen::Index kept = width > offset ? (width - offset + q - 1) / q : 0;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> y(x.rows(), kept);
  if (q == 1) {
    y = x.middleCols(left + offset, kept);
    return y;
  }
  const Eigen::VectorXd h = fir1(n, {1.0 / q}, Band::Lowpass);   // order + 1 taps, symmetric about n / 2
  const Eigen::Index half = n / 2;
  for (Eigen::Index k = 0; k < kept; ++k) {
    const Eigen::Index u = left + offset + k * q;               // the kept column, the filter's centre on it
    const Eigen::Index a = std::max<Eigen::Index>(0, u - half), b = std::min<Eigen::Index>(cols - 1, u + half);
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(x.rows());
    for (Eigen::Index j = a; j <= b; ++j) acc += h(half + (u - j)) * x.col(j).template cast<double>();
    y.col(k) = acc.template cast<T>();
  }
  return y;
}

}  // namespace

Eigen::MatrixXd decimate(const Eigen::MatrixXd& x, int q, Eigen::Index left, Eigen::Index width, Eigen::Index offset, int order) { return decimated<double>(x, q, left, width, offset, order); }
Eigen::MatrixXf decimate(const Eigen::MatrixXf& x, int q, Eigen::Index left, Eigen::Index width, Eigen::Index offset, int order) { return decimated<float>(x, q, left, width, offset, order); }


Eigen::MatrixXcd downsample(const Eigen::MatrixXcd& Z, int d, int offset) {
  if (d < 1) throw std::invalid_argument("downsample: the factor d is at least 1");
  if (offset < 0 || offset >= d) throw std::invalid_argument("downsample: the offset is in [0, d)");
  const Eigen::Index T = Z.cols(), n = T > offset ? (T - offset + d - 1) / d : 0;
  Eigen::MatrixXcd out(Z.rows(), n);
  for (Eigen::Index j = 0; j < n; ++j) out.col(j) = Z.col(offset + j * d);
  return out;
}

}  // namespace nxr::signal

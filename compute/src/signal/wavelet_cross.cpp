// xwt and wcoherence: Grinsted et al. (2004) wtc.m / smoothwavelet.m, in C++.
#include "nxr/signal/wavelet_cross.h"
#include "nxr/signal/fourier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::signal {
namespace {
using Eigen::Index;
// Grinsted's smoothwavelet: Gaussian in time at each row's scale (FFT, padded to a power of two), then the boxcar across scales
Eigen::MatrixXcd smooth(const Eigen::MatrixXcd& W, const WaveletCoherenceOptions& o) {
  const Index S = W.rows(), n = W.cols();
  Index npad = 1; while (npad < n) npad <<= 1;
  const double pi = std::acos(-1.0);
  // k = [0 1 ... npad/2 -(...)] 2 pi / (npad dt), exactly Grinsted's ordering (fftfreq in angular frequency)
  Eigen::VectorXd k = fftfreq(static_cast<std::size_t>(npad), o.dt) * (2.0 * pi);
  Eigen::MatrixXcd padded = Eigen::MatrixXcd::Zero(npad, S);
  for (Index s = 0; s < S; ++s) padded.block(0, s, n, 1) = W.row(s).transpose();
  Eigen::MatrixXcd F = fft(padded, 0);   // along time (down the rows of the transposed layout)
  for (Index s = 0; s < S; ++s) {
    // Grinsted's snorm = scale / dt with k in rad per sample; here k is in rad/s, so exp(-0.5 (scale k)^2): the same Gaussian
    for (Index j = 0; j < npad; ++j) F(j, s) *= std::exp(-0.5 * (o.scale(s) * k(j)) * (o.scale(s) * k(j)));
  }
  const Eigen::MatrixXcd back = ifft(F, 0);
  Eigen::MatrixXcd out(S, n);
  for (Index s = 0; s < S; ++s) out.row(s) = back.block(0, s, n, 1).transpose();
  if (o.scaleSmoothingSteps > 0.0) {   // kernel = [frac; ones(2 round(steps) - 1); frac], normalised, conv 'same' along the scale axis (zero ends)
    const double steps = o.scaleSmoothingSteps;
    const double frac = steps - std::floor(steps);
    const Index core = std::max<Index>(0, 2 * static_cast<Index>(std::llround(steps)) - 1);
    Eigen::VectorXd ker(core + 2);
    ker(0) = frac; ker(core + 1) = frac;
    for (Index i = 0; i < core; ++i) ker(1 + i) = 1.0;
    ker /= ker.sum();
    const Index L = ker.size();
    Eigen::MatrixXcd sm = Eigen::MatrixXcd::Zero(S, n);
    const Index off = L / 2;   // MATLAB conv2(..., 'same'): the central part of the full convolution starts at floor(L/2)
    for (Index r = 0; r < S; ++r)
      for (Index j = 0; j < L; ++j) {
        const Index src = r + off - j;   // full(r') = sum_j ker(j) out(r' - j), r' = r + off
        if (src >= 0 && src < S) sm.row(r) += ker(j) * out.row(src);
      }
    out = sm;
  }
  return out;
}
}  // namespace

Eigen::MatrixXcd xwt(const Eigen::MatrixXcd& Wx, const Eigen::MatrixXcd& Wy) {
  if (Wx.rows() != Wy.rows() || Wx.cols() != Wy.cols()) throw std::invalid_argument("xwt: the two transforms must have the same shape");
  return Wx.cwiseProduct(Wy.conjugate());
}
std::vector<Eigen::MatrixXcd> xwt(const std::vector<Eigen::MatrixXcd>& Wx, const std::vector<Eigen::MatrixXcd>& Wy) {
  if (Wx.size() != Wy.size()) throw std::invalid_argument("xwt: the two transforms must have the same number of members");
  std::vector<Eigen::MatrixXcd> out;
  for (std::size_t m = 0; m < Wx.size(); ++m) out.push_back(xwt(Wx[m], Wy[m]));
  return out;
}

Eigen::MatrixXd wcoherence(const Eigen::MatrixXcd& Wx, const Eigen::MatrixXcd& Wy, const WaveletCoherenceOptions& o) {
  if (Wx.rows() != Wy.rows() || Wx.cols() != Wy.cols()) throw std::invalid_argument("wcoherence: the two transforms must have the same shape");
  if (o.scale.size() != Wx.rows()) throw std::invalid_argument("wcoherence: one scale per row is needed: " + std::to_string(o.scale.size()) + " given, " + std::to_string(Wx.rows()) + " rows");
  if (!(o.dt > 0)) throw std::invalid_argument("wcoherence: dt must be positive");
  for (Index i = 0; i < o.scale.size(); ++i) if (!(o.scale(i) > 0)) throw std::invalid_argument("wcoherence: the scales must be positive");
  Eigen::MatrixXcd sinv(Wx.rows(), Wx.cols());
  for (Index s = 0; s < Wx.rows(); ++s) sinv.row(s).setConstant(1.0 / o.scale(s));
  const Eigen::MatrixXcd cross = smooth(sinv.cwiseProduct(xwt(Wx, Wy)), o);
  const Eigen::MatrixXcd px = smooth(sinv.cwiseProduct(Wx.cwiseAbs2().cast<std::complex<double>>()), o);
  const Eigen::MatrixXcd py = smooth(sinv.cwiseProduct(Wy.cwiseAbs2().cast<std::complex<double>>()), o);
  Eigen::MatrixXd R(Wx.rows(), Wx.cols());
  for (Index s = 0; s < R.rows(); ++s)
    for (Index t = 0; t < R.cols(); ++t) R(s, t) = std::norm(cross(s, t)) / (px(s, t).real() * py(s, t).real());
  return R;
}

Eigen::VectorXd coneOfInfluence(Eigen::Index nTime, double dt, double coiScalar) {
  if (nTime < 1) throw std::invalid_argument("coneOfInfluence: the series needs at least one sample");
  if (!(dt > 0.0) || !(coiScalar > 0.0)) throw std::invalid_argument("coneOfInfluence: dt and the wavelet's constant must be positive");
  Eigen::VectorXd coi(nTime);
  for (Eigen::Index n = 0; n < nTime; ++n) coi(n) = coiScalar * dt * static_cast<double>(std::min<Eigen::Index>(n + 1, nTime - n));
  return coi;
}

Eigen::VectorXd coneOfInfluenceFrequency(Eigen::Index nTime, double dt, double coiScalar, double maxFrequency) {
  if (!(maxFrequency > 0.0)) throw std::invalid_argument("coneOfInfluenceFrequency: the largest frequency must be positive");
  Eigen::VectorXd f = coneOfInfluence(nTime, dt, coiScalar).cwiseInverse();
  return f.cwiseMin(maxFrequency);
}

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> insideConeOfInfluence(const Eigen::VectorXd& period, const Eigen::VectorXd& coi) {
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> m(period.size(), coi.size());
  for (Eigen::Index s = 0; s < period.size(); ++s) for (Eigen::Index n = 0; n < coi.size(); ++n) m(s, n) = period(s) > coi(n);
  return m;
}

}  // namespace nxr::signal

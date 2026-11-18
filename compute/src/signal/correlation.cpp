// xcorr (the FFT route of scipy.signal.correlate) and mscohere (scipy.signal.coherence: Welch's cross- and auto-spectra with their scaling cancelling in the ratio).
#include "nxr/signal/correlation.h"
#include "nxr/signal/fourier.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::signal {
namespace {
using Eigen::Index;

template <class V>
auto fullCorrelation(const V& x, const V& y) {   // r[m + (M - 1)] = sum_n x[n + m] conj(y[n]), m = -(M-1) .. N-1  (N = |x|, M = |y|)
  const std::size_t N = x.size(), M = y.size();
  std::size_t L = 1; while (L < N + M - 1) L <<= 1;
  Eigen::MatrixXcd a = Eigen::MatrixXcd::Zero(L, 1), b = Eigen::MatrixXcd::Zero(L, 1);
  for (std::size_t i = 0; i < N; ++i) a(i, 0) = x(i);
  for (std::size_t i = 0; i < M; ++i) b(i, 0) = y(i);
  const Eigen::MatrixXcd A = fft(a), B = fft(b);
  Eigen::MatrixXcd P(L, 1);
  for (std::size_t i = 0; i < L; ++i) P(i, 0) = A(i, 0) * std::conj(B(i, 0));
  const Eigen::MatrixXcd c = ifft(P);   // c[m mod L] = r(m)
  Eigen::VectorXcd r(N + M - 1);
  for (long m = -(static_cast<long>(M) - 1); m < static_cast<long>(N); ++m) r(m + static_cast<long>(M) - 1) = c((m + static_cast<long>(L)) % static_cast<long>(L), 0);
  return r;
}

template <class V, class R, class Out>
Out xcorrImpl(const V& x, const V& y, int maxlag, XcorrScale scale, bool realOut) {
  (void)realOut;
  const long N = x.size(), M = y.size();
  if (N == 0 || M == 0) throw std::invalid_argument("xcorr: a signal is empty");
  if (scale != XcorrScale::None && N != M) throw std::invalid_argument("xcorr: Biased, Unbiased and Coeff scaling need two signals of equal length, got " + std::to_string(N) + " and " + std::to_string(M));
  const Eigen::VectorXcd full = fullCorrelation(x, y);
  const long lo = maxlag >= 0 ? -static_cast<long>(maxlag) : -(M - 1), hi = maxlag >= 0 ? static_cast<long>(maxlag) : N - 1;
  Out out;
  out.lags.resize(hi - lo + 1);
  out.r.resize(hi - lo + 1);
  double coeff = 1.0;
  if (scale == XcorrScale::Coeff) {
    double ex = 0, ey = 0;
    for (long i = 0; i < N; ++i) { ex += std::norm(std::complex<double>(x(i))); ey += std::norm(std::complex<double>(y(i))); }
    coeff = 1.0 / std::sqrt(ex * ey);
  }
  for (long m = lo; m <= hi; ++m) {
    std::complex<double> v = (m >= -(M - 1) && m <= N - 1) ? full(m + M - 1) : std::complex<double>(0.0);
    if (scale == XcorrScale::Biased) v /= static_cast<double>(N);
    else if (scale == XcorrScale::Unbiased) v /= static_cast<double>(N - std::labs(m) > 0 ? N - std::labs(m) : 1);
    else if (scale == XcorrScale::Coeff) v *= coeff;
    out.lags(m - lo) = static_cast<int>(m);
    if constexpr (std::is_same_v<R, double>) out.r(m - lo) = v.real(); else out.r(m - lo) = v;
  }
  return out;
}
}  // namespace

Xcorr xcorr(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int maxlag, XcorrScale scale) { return xcorrImpl<Eigen::VectorXd, double, Xcorr>(x, y, maxlag, scale, true); }
XcorrComplex xcorr(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int maxlag, XcorrScale scale) { return xcorrImpl<Eigen::VectorXcd, std::complex<double>, XcorrComplex>(x, y, maxlag, scale, false); }

Coherence mscohere(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double fs, const Eigen::VectorXd& window, int noverlap, int nfft, spectral::Detrend detrend) {
  if (x.size() != y.size()) throw std::invalid_argument("mscohere: the signals must have the same length");
  if (!(fs > 0)) throw std::invalid_argument("mscohere: the sampling rate must be positive");
  const long nper = window.size(), n = x.size();
  if (nper < 1 || nper > n) throw std::invalid_argument("mscohere: the window (the segment) must have between 1 and " + std::to_string(n) + " samples, got " + std::to_string(nper));
  const long nov = noverlap < 0 ? nper / 2 : noverlap;
  if (nov >= nper) throw std::invalid_argument("mscohere: the overlap must be less than the segment length");
  const long nf = nfft > 0 ? nfft : nper;
  if (nf < nper) throw std::invalid_argument("mscohere: nfft must be at least the segment length");
  const long step = nper - nov, nseg = (n - nov) / step;
  if (nseg < 1) throw std::invalid_argument("mscohere: no whole segment fits");
  const Index bins = nf / 2 + 1;
  Eigen::VectorXd Pxx = Eigen::VectorXd::Zero(bins), Pyy = Eigen::VectorXd::Zero(bins);
  Eigen::VectorXcd Pxy = Eigen::VectorXcd::Zero(bins);
  auto prepared = [&](const Eigen::VectorXd& s, long start) {   // detrend, window, zero-pad to nfft, rfft
    Eigen::VectorXd seg = s.segment(start, nper);
    if (detrend == spectral::Detrend::Constant) seg.array() -= seg.mean();
    else if (detrend == spectral::Detrend::Linear) {   // least-squares line over the segment
      const double tm = 0.5 * (nper - 1);
      double sxy = 0, sxx = 0, mean = seg.mean();
      for (long i = 0; i < nper; ++i) { sxy += (i - tm) * (seg(i) - mean); sxx += (i - tm) * (i - tm); }
      const double slope = nper > 1 ? sxy / sxx : 0.0;
      for (long i = 0; i < nper; ++i) seg(i) -= mean + slope * (i - tm);
    }
    Eigen::VectorXd padded = Eigen::VectorXd::Zero(nf);
    padded.head(nper) = seg.cwiseProduct(window);
    return rfft(Eigen::MatrixXd(padded)).col(0).eval();
  };
  for (long s = 0; s < nseg; ++s) {
    const Eigen::VectorXcd X = prepared(x, s * step), Y = prepared(y, s * step);
    for (Index k = 0; k < bins; ++k) { Pxx(k) += std::norm(X(k)); Pyy(k) += std::norm(Y(k)); Pxy(k) += X(k) * std::conj(Y(k)); }
  }
  Coherence c;
  c.f = rfftfreq(static_cast<std::size_t>(nf), 1.0 / fs);
  c.Cxy.resize(bins);
  for (Index k = 0; k < bins; ++k) c.Cxy(k) = std::norm(Pxy(k)) / (Pxx(k) * Pyy(k));
  return c;
}

}  // namespace nxr::signal

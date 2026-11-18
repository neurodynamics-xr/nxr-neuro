#pragma once
// nxr::signal — correlation in the lag domain and its spectral twin: `xcorr` (MATLAB xcorr; SciPy scipy.signal.correlate with correlation_lags) and
// `mscohere` (MATLAB mscohere; SciPy scipy.signal.coherence).
//
// PURE MATHS: vectors in, vectors out. No file, no chunk, no choice of which signal: the caller passes the two.
//
// xcorr.  r(m) = sum_n x[n + m] conj(y[n]), the lag m in samples from -(N-1) to M-1 ('full', scipy.signal.correlate's convention): a peak at m > 0 means x is y delayed by m.
//   `maxlag` >= 0 keeps the lags in [-maxlag, maxlag] (zero where the signals do not overlap, as MATLAB); the default keeps all of them. The scaling is MATLAB's:
//   None (raw sums), Biased (/N), Unbiased (/(N - |m|)), Coeff (/sqrt(sum|x|^2 sum|y|^2): the zero-lag autocorrelation is 1). Biased, Unbiased and Coeff need equal lengths.
// mscohere.  The magnitude-squared coherence C_xy(f) = |P_xy(f)|^2 / (P_xx(f) P_yy(f)) from Welch's averaged cross- and auto-spectra: the caller's window (its length is
//   the segment), the overlap (default half the segment), the FFT length (default the segment's), the detrend of each segment (default constant, scipy's). One-sided
//   frequencies f = k fs / nfft, k = 0 .. nfft/2. With a single segment C is 1 at every bin (the estimator has no averaging to tell coupled from not): the caller
//   chooses the segments, so the caller knows how many there are.
#include "nxr/signal/spectral.h"   // Detrend
#include <Eigen/Dense>
#include <complex>

namespace nxr::signal {

enum class XcorrScale { None, Biased, Unbiased, Coeff };

struct Xcorr {
  Eigen::VectorXi lags;   ///< the lags in samples, ascending
  Eigen::VectorXd r;      ///< r(lags)
};
struct XcorrComplex {
  Eigen::VectorXi lags;
  Eigen::VectorXcd r;
};

Xcorr xcorr(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int maxlag = -1, XcorrScale scale = XcorrScale::None);
XcorrComplex xcorr(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int maxlag = -1, XcorrScale scale = XcorrScale::None);

struct Coherence {
  Eigen::VectorXd f;     ///< Hz, k fs / nfft, k = 0 .. nfft / 2
  Eigen::VectorXd Cxy;   ///< in [0, 1]
};

/** Welch magnitude-squared coherence of two real signals of the same length. `window`: the segment's window (symmetric or periodic as the caller made it); `noverlap` < 0: half the
 *  segment; `nfft` 0: the segment's length. Throws std::invalid_argument when a segment does not fit or fewer than one segment is available. */
Coherence mscohere(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double fs, const Eigen::VectorXd& window, int noverlap = -1, int nfft = 0, spectral::Detrend detrend = spectral::Detrend::Constant);

}  // namespace nxr::signal

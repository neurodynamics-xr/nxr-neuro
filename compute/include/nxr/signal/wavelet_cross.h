#pragma once
// nxr::signal — the CROSS-WAVELET TRANSFORM and WAVELET COHERENCE of two signals: `xwt` and `wcoherence` (Torrence & Compo 1998; Grinsted, Moore & Jevrejeva 2004, "Application of the
// cross wavelet transform and wavelet coherence to geophysical time series", Nonlin. Processes Geophys. 11:561-566; MATLAB's wcoherence).
//
// PURE MATHS on coefficient matrices: the wavelet transform of each signal is made by `wt` (nxr/signal/wavelet.h); these functions take its coefficients, a row per scale (or band),
// a column per time sample ([scales x time], complex).
//
//   xwt(Wx, Wy)        W_xy(s, t) = W_x(s, t) conj(W_y(s, t)). Its modulus is the common power, its angle the local phase difference (Wx leads Wy where the angle is positive).
//   wcoherence(...)    R^2(s, t) = | S(W_xy / s) |^2 / ( S(|W_x|^2 / s) S(|W_y|^2 / s) ),  in [0, 1],
//                      S the smoothing operator of Grinsted et al. (their `wtc`): a Gaussian in TIME whose width is the scale (the FFT route, the series zero-padded to the next
//                      power of two: the circular-convolution edge of the reference), then a boxcar across SCALES of `scaleSmoothingSteps` (Grinsted's dj0 / (2 dj), dj0 = 0.6, the
//                      decorrelation length of the Morlet wavelet; end weights are the fractional part).
// What it cannot show: with no smoothing the coherence is 1 everywhere (one estimate has no spread); the result depends on the smoothing widths, which the caller states. The edge
// samples (the cone of influence) are not masked here.
#include <Eigen/Dense>
#include <vector>

namespace nxr::signal {

struct WaveletCoherenceOptions {
  double dt = 1.0;                       ///< the sampling interval, s
  Eigen::VectorXd scale;                 ///< the scale s of each row, s (the Gaussian's width in time; the 1/s weight); > 0, one per row
  double scaleSmoothingSteps = 0.0;      ///< the boxcar's half-width in rows (Grinsted: 0.6 / (2 dj), dj the scale spacing in octaves); 0: no smoothing across scales
};

/** The cross-wavelet transform of two coefficient matrices of one shape, elementwise W_x conj(W_y). */
Eigen::MatrixXcd xwt(const Eigen::MatrixXcd& Wx, const Eigen::MatrixXcd& Wy);
/** The same for the members' blocks of a bank's transform (`wt`): block m of each, elementwise (the blocks of the two transforms must have the same shapes). */
std::vector<Eigen::MatrixXcd> xwt(const std::vector<Eigen::MatrixXcd>& Wx, const std::vector<Eigen::MatrixXcd>& Wy);
/** Wavelet coherence R^2 [scales x time] of the transforms Wx, Wy [scales x time]. Throws std::invalid_argument on a shape or option mismatch. */
Eigen::MatrixXd wcoherence(const Eigen::MatrixXcd& Wx, const Eigen::MatrixXcd& Wy, const WaveletCoherenceOptions& options);

}  // namespace nxr::signal

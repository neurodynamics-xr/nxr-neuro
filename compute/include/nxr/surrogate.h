#pragma once
// nxr::surrogate — the LAG-FREE NULL of the vortex chain (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10). PURE MATHS on arrays, the same functions a test calls with matrices made anywhere:
// a function never knows where the spectrum, the covariance or the envelope came from (stored statistics, a synthetic recording) nor which chunk it is asked for. A port of nxr's `lagfree_real`, `envelope_matched_null`, `wishart` and
// `dof_fraction` (nxr/detect/vortex.py), the references, with ONE difference stated below.
//
// THE SURROGATE (lagfree_real, envelope_matched_null): Gaussian noise with the data's own spectrum (never flat: the project's trap), mixed by the square root of the data's REAL covariance (the zero-lag structure and power kept, no lag),
// scaled to the data's amplitude envelope (its RMS over one cycle). Written as a function of the SAMPLE INDEX, so a chunk's result does not depend on the chunk size:
//     x(c, t) = scale(t) . sum_m Rh(c, m) . sum_k h(k) . w(m, t + k)           w(m, s) = normalAt(seed, m, s)  (one standard normal per channel and sample: a counter-based generator, never a stream)
//   h       zero-phase taps made from the amplitude spectrum sqrt(PSD) (`shapingTaps`), unit energy: the filtered noise has unit variance
//   Rh      sqrt(Re cov) of the band signal (`sqrtPsd`)
//   scale   envelope(t) / envelope_expected: the data's one-cycle RMS envelope over the surrogate's EXPECTED one (sqrt(trace(Re cov)/C)): pointwise, no reduction in the node (the cardinal rule). The difference from nxr's
//           `envelope_matched_null`: nxr divides by the RECALISED RMS of the draw (a reduction over channels and a cycle), so its draw's envelope IS the data's; here the draw's envelope is the data's up to the noise's own sampling
//           variation, which is what a null of the data's amplitude modulation has.
// WHAT IT CANNOT SAY: the null has the data's covariance at zero lag and its spectrum, and nothing else: a core rate, a direction consistency or a repeat beyond it is beyond what mixing and amplitude modulation produce, never a proof of a wave.
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::surrogate {

/** the standard normal of (seed, stream, index): a counter-based generator (splitmix64 of the three, Box-Muller): the same value however the samples are cut into chunks */
double normalAt(std::uint64_t seed, std::uint64_t stream, std::uint64_t index);
/** [rows x n] standard normals, column j the sample `first + j` (row r the stream r); `first` may be negative (a halo before the run) */
Eigen::MatrixXd whiteBlock(std::uint64_t seed, int rows, std::int64_t first, std::int64_t n);

/** zero-phase taps (2 half + 1, symmetric, Hann-windowed, unit energy) whose amplitude response follows sqrt(psd) at `freq` (Hz, increasing; psd power per Hz, interpolated linearly in between; sampling rate fs). A psd with no power is refused. */
Eigen::VectorXd shapingTaps(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, double fs, int half);
/** [rows x n] from [rows x (n + 2 half)]: the taps (symmetric, 2 half + 1) applied along the columns, valid part only (output column j is centred on input column j + half) */
Eigen::MatrixXd filterValid(const Eigen::MatrixXd& white, const Eigen::VectorXd& taps);
/** the square root of a symmetric positive semi-definite matrix (negative eigenvalues clipped to 0) */
Eigen::MatrixXd sqrtPsd(const Eigen::MatrixXd& A);

/** THE LAG-FREE BLOCK: [C x n], samples first .. first + n - 1 of draw `seed`: Rh . (the taps applied to w) . scale. `scale` [n] the per-sample factor (empty: 1). Rh [C x C] (sqrtPsd of the data's real covariance). */
Eigen::MatrixXd lagfreeBlock(const Eigen::MatrixXd& Rh, const Eigen::VectorXd& taps, std::uint64_t seed, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale = Eigen::VectorXd());

/** a series stored every `stride` samples (value k at sample k . stride), read at samples first .. first + n - 1: linear between stored values, held beyond the ends */
Eigen::VectorXd interpolateSeries(const std::vector<double>& values, std::int64_t stride, std::int64_t first, std::int64_t n);

/** the effective bandwidth of a spectrum (Hz): (sum P)^2 dF / sum P^2 on its grid (nxr `dof_fraction` x fs: the number of independent samples a second of a signal with this power spectrum holds; a flat spectrum of width B gives B). `gain` (optional, same grid) multiplies the power (a band's filter, squared). */
double effectiveBandwidth(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain = Eigen::VectorXd());

/** A LAG-FREE SECOND MOMENT: Q ~ complex Wishart(Re Q0 / k, k), k = max(round(dof), 2): the sum of k outer products of complex Gaussian vectors of covariance Re Q0 / k (mean Re Q0; no lag, its imaginary part has zero mean) (nxr `wishart`). Draw `draw` of `seed`. */
Eigen::MatrixXcd wishartDraw(const Eigen::MatrixXd& ReQ0, double dof, std::uint64_t seed, std::uint64_t draw);

/** the sequential p of Besag and Clifford (1991): `exceed` null values at least the data's after `n` draws, h the exceedances at which sampling stops: h / n when it stopped (exceed >= h), else (exceed + 1) / (n + 1) */
double besagCliffordP(std::int64_t exceed, std::int64_t n, std::int64_t h);
/** Benjamini-Hochberg q-values of p (NaN stays NaN and does not count among the tests) */
Eigen::VectorXd benjaminiHochberg(const Eigen::VectorXd& p);

}  // namespace nxr::surrogate

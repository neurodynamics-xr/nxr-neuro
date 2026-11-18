#pragma once
// nxr::synthesis — THE LAG-FREE SURROGATES (spec docs/specs/2026-11-17-surrogates.md). PURE MATHS on arrays, the same functions a test calls with matrices made anywhere: a function never knows
// where a spectrum, a covariance or an envelope came from (stored statistics, a synthetic recording), nor which block it is asked for.
// References: Rheome +rheome/+cores lagfreenull (the lag-free draw), envelopenull (rescaled to the data's envelope), pairnull (the lag-free Wishart second moment), pairsqrt, pairshare; the Python
// reference they port. With ONE difference stated below (the envelope's scale is the expected, not the realised, power).
//
// THE SURROGATE, as a function of the SAMPLE INDEX (so a block's values never depend on how the samples are cut into blocks):
//     x(c, t) = sum_m mixing(c, m) . scale(t) . sum_k taps(k) . w(m, t + k)        w(m, s) = standardNormalAt(seed, m, s)  (a counter-based generator, never a stream)
//   taps     zero-phase taps whose amplitude response follows sqrt(PSD) (`spectrumShapingTaps`), unit energy: each shaped stream has unit variance
//   mixing   sqrt(Re C) of the band's cross-spectrum at unit mean channel variance (`zeroLagMixing`)
//   scale    envelope(t) / expected: the data's band power over the draw's EXPECTED band power through the band's analytic gain (`envelopeScale`, `analyticGainPower`): pointwise, no reduction.
//            Rheome's envelopenull divides by the REALISED envelope of the draw (a reduction over channels and a cycle), so its draw's envelope IS the data's; here the draw's envelope is the data's
//            up to the noise's own sampling variation, which is what a null of the data's amplitude modulation has.
// THE ORDER (spec section 4): white -> taps -> scale (`shapedNoiseBlock`, the UNMIXED draw), and the mixing applied after the linear time steps, composed into the spatial operator. The scale is the
// same number on every stream of a sample and the time filters are the same on every stream, so both commute with the mixing: `lagfreeBlock` (mixed first) equals the unmixed draw mixed later.
// WHAT IT CANNOT SAY: the surrogate has the data's zero-lag covariance, its spectrum and its envelope, and nothing else: a result beyond it is beyond what mixing and amplitude modulation produce,
// never a proof of a wave.
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::synthesis {

/** the standard normal of (seed, stream, index): a counter-based generator (splitmix64 of the three, Box-Muller): the same value however the samples are cut into blocks */
double standardNormalAt(std::uint64_t seed, std::uint64_t stream, std::uint64_t index);
/** [rows x n] standard normals, column j the sample `first + j` (row r the stream r); `first` may be negative (a halo before the run) */
Eigen::MatrixXd whiteNoiseBlock(std::uint64_t seed, int rows, std::int64_t first, std::int64_t n);

/** zero-phase taps (2 half + 1, symmetric, Hann-windowed, unit energy) whose amplitude response follows sqrt(psd) at `freq` (Hz, increasing; psd power per Hz, interpolated linearly in between;
 *  sampling rate fs). A psd with no power is refused (std::invalid_argument). */
Eigen::VectorXd spectrumShapingTaps(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, double fs, int half);
/** [rows x n] from [rows x (n + 2 half)]: the symmetric taps (2 half + 1) applied along the columns, the valid part only (output column j is centred on input column j + half) */
Eigen::MatrixXd convolveValid(const Eigen::MatrixXd& white, const Eigen::VectorXd& taps);

/** THE ZERO-LAG MIXING of a band: sqrt(Re C / (trace(Re C) / M)), the symmetric square root of the real cross-spectrum scaled to unit mean channel variance [M x M]. A cross-spectrum with no power is
 *  refused (std::invalid_argument). */
Eigen::MatrixXd zeroLagMixing(const Eigen::MatrixXd& realCrossSpectrum);

/** THE UNMIXED DRAW: [rows x n], samples first .. first + n - 1 of draw `seed`, one stream a row: the taps applied to white noise, times `scale` per sample (empty: 1). */
Eigen::MatrixXd shapedNoiseBlock(const Eigen::VectorXd& taps, std::uint64_t seed, int rows, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale = Eigen::VectorXd());
/** THE MIXED DRAW: mixing [C x C] times `shapedNoiseBlock(taps, seed, C, first, n, scale)` (the reference form; equal to mixing after any linear time step, spec section 4) */
Eigen::MatrixXd lagfreeBlock(const Eigen::MatrixXd& mixing, const Eigen::VectorXd& taps, std::uint64_t seed, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale = Eigen::VectorXd());

/** the expected mean |z|^2 of unit-variance noise with spectrum `psd` through a band's analytic gain `gain` (both on `freq`): 2 sum psd g^2 / sum psd (trapezoid weights; the analytic signal's power
 *  is twice the band-passed real signal's). No power in the band is refused (std::invalid_argument). */
double analyticGainPower(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain);
/** the per-sample scale sqrt(max(power, 0) / expected) (pointwise) */
Eigen::VectorXd envelopeScale(const Eigen::VectorXd& power, double expected);
/** a series stored every `stride` samples (value k at sample k . stride), read at samples first .. first + n - 1: linear between stored values, held beyond the ends */
Eigen::VectorXd interpolateTileSeries(const std::vector<double>& values, std::int64_t stride, std::int64_t first, std::int64_t n);

/** the effective bandwidth of a spectrum (Hz): (sum P)^2 dF / sum P^2 on its grid (the number of independent samples a second of a signal with this power spectrum holds; a flat spectrum of width B
 *  gives B). `gain` (optional, same grid) multiplies the power (a band's filter, squared). */
double effectiveBandwidth(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain = Eigen::VectorXd());

/** A LAG-FREE SECOND MOMENT: Q ~ complex Wishart(Re Q0 / k, k), k = max(round(dof), 2): the mean of k outer products of complex Gaussian vectors of covariance Re Q0 (mean Re Q0; no lag, its
 *  imaginary part has zero mean) (Rheome pairnull). Draw `draw` of `seed`. */
Eigen::MatrixXcd lagfreeSecondMomentDraw(const Eigen::MatrixXd& ReQ0, double dof, std::uint64_t seed, std::uint64_t draw);
/** the LAGGED SHARE of a second moment: ||Im Q||_F^2 / ||Q||_F^2 (Rheome pairshare with L = I, squared); NaN for a zero Q */
double laggedShare(const Eigen::MatrixXcd& Q);

}  // namespace nxr::synthesis

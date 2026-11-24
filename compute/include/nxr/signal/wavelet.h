#pragma once
// nxr::signal — the wavelet transform `wt` of a time filterbank and its inverse `iwt` (Rheome's names; a bank only DESIGNS, as MATLAB's cwtfilterbank holds the wavelets and cwt transforms).
//
// PURE MATHS (arrays in, arrays out). No state: a window (a chunk with its halo, as io's chunk service hands it) in, the window's coefficients out. Every transform returns its
// coefficients at the INPUT's rate: a transform never decimates, resamples or windows its input.
//   wt(bank, x)             every member on the bank's grid, [nfft x C] a member: one FFT of the array given, each member's gain, an inverse FFT at the full length; a real signal for an
//                           Analytic bank, a complex one for a TwoSided bank.
//   wt(bank, member, x, at) the full-rate analytic signal of ONE member, evaluated at instants the CALLER gives (a frame selection's); exact, a sum over the member's bins, no resampling.
//   wtBlock(design, block, fs, left, own): the full-rate transform of a haloed block, the own columns kept.
// iwt(bank, W, n, mode) / iwtComplex: synthesis from full-rate coefficients.
//
// The bank is a nxr::signal::timefilterbank::TimeFilterbank, ANALYTIC (a real recording). Its family, gain and sides are REQUIRED by timebank::Design (no default). The bank is
// designed ONCE on a reference grid `bank.nfft` (its frequency resolution: the lowest member spans >= minBins bins of fs/nfft); the record's length plays no part.
// A chunked run gives the transform a block with the halo timefilterbank::haloOf(bank, member, tolerance) on each side; which instants are kept is the caller's selection, never the transform's.
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/timefilterbank.h"
#include <cstdint>
#include <vector>

namespace nxr::signal::timefilterbank {

// ── the time-domain primitives the transforms are built from (wt / iwt and the joint transform); they take the bank for its grid and sides ─────────────────────────────────────────────────────────────

/** Forward spectrum of a record on the bank's grid: rfft (Analytic, real) or fft (TwoSided) of x zero-padded to nfft → [bins × C]. */
Eigen::MatrixXcd spectrum(const TimeFilterbank& bank, const Eigen::MatrixXd& x);
Eigen::MatrixXcd spectrum(const TimeFilterbank& bank, const Eigen::MatrixXcd& x);   // Analytic refuses
/** ifft of (spec ⊙ gain) on the grid → [nfft × C] (Analytic: the negative bins are zero). `gain` has one entry per bin. */
Eigen::MatrixXcd filterSpectrum(const TimeFilterbank& bank, const Eigen::VectorXd& gain, const Eigen::MatrixXcd& spec);
/** Back to the record: irfft (Analytic) or ifft (TwoSided) of a spectrum [bins × C], cropped to n rows. Analytic returns the real part in `real`. */
Eigen::MatrixXd inverseReal(const TimeFilterbank& bank, const Eigen::MatrixXcd& spec, std::size_t n);
Eigen::MatrixXcd inverseComplex(const TimeFilterbank& bank, const Eigen::MatrixXcd& spec, std::size_t n);
/** Spectrum of a member's coefficients [nfft × C] on the bank's grid (fft, first `bins` rows). */
Eigen::MatrixXcd coefficientSpectrum(const TimeFilterbank& bank, const Eigen::MatrixXcd& coefficients);

}  // namespace nxr::signal::timefilterbank

namespace nxr::signal {

// ── FULL RATE ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Analytic: a REAL signal x [n x C] (n <= nfft; zero-padded): W_m = ifft(rfft(x) g_m) on the one-sided bins, [nfft x C] a member. */
timefilterbank::Coefficients wt(const timefilterbank::TimeFilterbank& bank, const Eigen::MatrixXd& x);
/** TwoSided: a complex (or real) signal, W_m = ifft(fft(x) g_m). An Analytic bank REFUSES this overload (std::invalid_argument). */
timefilterbank::Coefficients wt(const timefilterbank::TimeFilterbank& bank, const Eigen::MatrixXcd& x);
/** Analytic synthesis to the real record [n x C]. Dual: sum (g/S) conv W, exact wherever the bank covers; Tight: sum g conv W / S_nominal (G^2, G^2/2 at DC and Nyquist), exact when the bank is tight. */
Eigen::MatrixXd iwt(const timefilterbank::TimeFilterbank& bank, const timefilterbank::Coefficients& W, std::size_t n, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);
/** TwoSided synthesis to the complex record [n x C]. */
Eigen::MatrixXcd iwtComplex(const timefilterbank::TimeFilterbank& bank, const timefilterbank::Coefficients& W, std::size_t n, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual);

/** THE ANALYTIC SIGNAL OF BANDS (the transform of the signal.analytic and signal.envelope nodes, the Studio's band of the Signal chain): one member per band [lo, hi] Hz of `shape`
 *  (Morlet: a Gaussian of full width at half maximum hi − lo; Meyer: Meyer edges, `halfWidth` octaves; PowerPartition: squared gains adding to 1), gain 2 (the one-sided analytic signal), DC and Nyquist
 *  dropped, designed on x's OWN length: x is the window the caller hands in (periodic on it; the caller's halo holds the edge effect). x [n × C] real → one [n × C] complex block per band. */
timefilterbank::Coefficients bandAnalytic(const Eigen::MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, timefilterbank::Family shape, double halfWidth = 1.0 / 3.0);
/** The bands' ENVELOPES: |bandAnalytic(x, ...)|, one [n × C] real block per band. */
std::vector<Eigen::MatrixXd> bandEnvelope(const Eigen::MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, timefilterbank::Family shape, double halfWidth = 1.0 / 3.0);

/** The transform of a haloed block: a block [channels x width] whose own columns are [left, left + own). The bank is designed on the BLOCK's grid (gridLength(width, FastLen): zero-padded, no wrap), the block
 *  transformed, and only the own columns kept: one [channels x own] complex block per member. With a halo of at least timefilterbank::haloOf(bank, member) on each side the own columns are what the whole record
 *  gives there; where a recording ENDS the block has no context and the transform zero-pads (the taps route reflects there). The band's choice is the caller's Design; the member a run keeps is the caller's too. */
struct BlockTransform {
  timefilterbank::TimeFilterbank bank;              // the bank as designed on the block's grid (its members, their nominal bands)
  timefilterbank::Coefficients own;                 // per member: [channels x own] complex, the block's own columns
};
BlockTransform wtBlock(const timefilterbank::Design& design, const Eigen::MatrixXd& block, double fs, std::size_t left, std::size_t own);

/** wt of ONE member of an Analytic bank EVALUATED AT GIVEN INSTANTS (a frame Selection's: samples from column 0 of the window, not whole numbers): x [channels x width] a window (a chunk with its halo, zero outside),
 *  `at` [J] the instants; returns the member's analytic signal a(t_j) = (1/N) sum_b X_b g(f_b) e^{2 pi i b t_j / N} exactly (a sum over the member's bins, no interpolation), [channels x J], on the window's own grid N =
 *  next_fast_len(width) (the gain at f_b = b fs / N: the continuous design, masked to the member's support). The window is taken periodic on N: the caller's halo keeps the ends out of the instants. */
Eigen::MatrixXcd wt(const timefilterbank::TimeFilterbank& bank, int member, const Eigen::MatrixXd& x, const Eigen::VectorXd& at);

/** wt of ONE member of an Analytic bank at EVERY D-th SAMPLE of the bank's grid (sample m·D, m = 0 .. nfft/D − 1), from the record's spectrum `spectrum` (timefilterbank::spectrum: [bins x C]): exactly the full-rate
 *  coefficients there (not a filter: a band-limited complex signal is read at instants), computed on the member's own bins only by an inverse FFT of nfft/D points (the reference detector's channel_atoms):
 *  W(mD) = (1/nfft) Σ_k X_k g_k e^{2πi k m D / nfft}. The member's support (its bins whose gain is above 1e-16 of its peak) must fit in nfft/D bins, and D must divide nfft (std::invalid_argument otherwise).
 *  Returns [C x nfft/D]. MEMORY: C x nfft/D complex. */
Eigen::MatrixXcd wtDecimated(const timefilterbank::TimeFilterbank& bank, int member, const Eigen::MatrixXcd& spectrum, int decimation);

}  // namespace nxr::signal

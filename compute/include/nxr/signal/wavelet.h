#pragma once
// nxr::signal — the wavelet transform `wt` of a time filterbank and its inverse `iwt` (Rheome's names; a bank only DESIGNS, as MATLAB's cwtfilterbank holds the wavelets and cwt transforms).
//
// PURE MATHS (arrays in, arrays out). No state: a window (a chunk with its halo, as io's chunk service hands it) in, the window's coefficients out. Three forms, one name:
//   wt(bank, x)            FULL RATE: every member on the bank's grid, [nfft x C] a member (was timefilterbank::analysis); a real signal for an Analytic bank, a complex one for a TwoSided bank.
//   wt(layout, x, members) OWN RATE: each member at its own lattice rate (below), the window `x` [channels x samples] treated as the whole record: the coefficients for t in [-margin, n + margin)
//                          (the window's tails included). `layout` = wtLayout(bank, options) declares the margin, the halo a node must give a window; `members` restricts the work (empty: all).
//   wtOwnRate(bank, x)     Rheome's whole-array own-rate form of ONE signal (was analysisOwnRate), the FFT of the zero-padded record.
// iwt(bank, W, n, mode) / iwtComplex: synthesis from full-rate coefficients (was synthesis / synthesisComplex); iwt(layout, blocks, n, mode): from the own-rate blocks wt(layout, ...) gave.
// wtBlock(design, block, fs, left, own): the full-rate transform of a haloed block, the own columns kept (was analysisOfBlock).
//
// The bank is a nxr::signal::timefilterbank::TimeFilterbank, ANALYTIC (a real recording). Its family, gain and sides are REQUIRED by timebank::Design (no default). The bank is
// designed ONCE on a reference grid `bank.nfft` (its frequency resolution: the lowest member spans >= minBins bins of fs/nfft); the record's length plays no part.
//
// THE COEFFICIENTS (Rheome's wt = timebank::analysisOwnRate, made streaming). Member m is the analytic sub-band a_m(t) = (1/nfft) sum_b X_b g_m(f_b) e^{2 pi i f_b t}
// shifted to baseband by its lowest support bin: c_m(t) = a_m(t) e^{-2 pi i f_lo t}, f_lo = i1 fs/nfft. It is stored at the member's OWN fixed rate,
// rate_m = Nm fs/nfft (timebank::rates(bank, oversample)), at the instants t_k = k nfft/(Nm fs) = k D_m samples, D_m = p_m/q_m (lowest terms), k an integer on a
// lattice that is THE SAME FOR ALL CHUNKS (global k; sample 0 is k = 0).
//
// HOW A CHUNK IS DONE. A window of N samples (N, the look-behind L and the look-ahead R are multiples of P = lcm of the p_m, so every window starts on every lattice) is FFT-ed;
// each member's gain is evaluated on the window's own frequency grid, g_m(f) at f = b fs/N (the continuous design, timebank::gains(bank, f): the DC / Nyquist rule is a property
// of a finite periodic grid and is not applied); the bins of the member's support are inverse-FFT-ed at Mm = N q/p points; the core of the window is kept (overlap-save).
// L = R is the shortest length after which the sum of the |impulse response| of the longest member (computed on a fine grid from the masked gains) is below `tolerance`
// times its L1 norm (default 1e-12), capped by maxMargin; `WtLayout::margin` and `WtLayout::achieved` report both. A member with a gain that jumps (e.g. a high-pass that
// is nonzero at Nyquist) has a 1/t tail: then achieved > tolerance and the chunked result differs from the whole record by about that much, for signals
// with energy near the jump.
//
// RECORD ENDS. The record is zero-padded on both sides: APERIODIC (as timebank::Padding::FastLen), because a long recording cannot be periodic. The coefficients are
// emitted for the instants t in [-L, n + R): the tails beyond the record are coefficients too (they carry what iwt needs to reconstruct the ends); n = the samples fed.
//
// iwt: synthesis from those coefficients (Dual g/S or Tight g/G^2, as timebank::synthesis); the same windows, the same margin; it returns x[0, n).
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/timefilterbank.h"
#include <cstdint>
#include <vector>

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

/** Rheome's wt of one real signal: per member the coefficients at the member's OWN rate (inverse FFT of its bins alone, oversampled). Analytic only. coef[t] = ifft(bins g) (Nm / nfft) at t = k nfft / (Nm fs). */
struct OwnRate { Eigen::VectorXcd coef; Eigen::VectorXd t; double rate = 0, weight = 0; };
std::vector<OwnRate> wtOwnRate(const timefilterbank::TimeFilterbank& bank, const Eigen::VectorXd& x, double oversample = 2.0);

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

// ── OWN RATE: the lattice, the margin, the window ─────────────────────────────────────────────────────────────────────

struct WtOptions {
    double oversample = 2.0;                  // timefilterbank::rates: the member's rate = next_fast_len(oversample * bins) fs / nfft
    double tolerance = 1e-12;                 // the impulse response's tail (L1, relative to its L1 norm) that the margin may leave out
    std::size_t maxMargin = std::size_t(1) << 16;
    std::size_t block = 0;                    // core samples per window (rounded up to a multiple of P); 0: max(L + R, min(4 (L + R), maxBlock))
    std::size_t maxBlock = std::size_t(1) << 15;   // the default block's cap: a window holds channels x (block + L + R) samples and their FFT, so it bounds memory
};

/** One member's run of coefficients: instants first, first+1, ...; rows = instants, columns = channels. */
struct Block { std::int64_t first = 0; Eigen::MatrixXcd coef; };

/** What wt and iwt of one bank share, a function of the bank and the options alone (no state): the lattice, the margin (the halo a node declares: `margin` samples each side, plus P so a window starts on the lattice), the windows. */
struct WtLayout {
    timefilterbank::TimeFilterbank bank;
    WtOptions options;
    std::vector<std::size_t> Nm;              // own-rate lengths
    Eigen::VectorXd rate;                     // Hz
    std::vector<std::int64_t> p, q;           // D_m = p/q samples
    std::int64_t P = 1;                       // lcm of p
    std::int64_t margin = 0, block = 0, window = 0;      // L = R, H, N = L + H + R (samples)
    std::vector<double> tail, l1;             // per member: the tail left out relative to its L1 norm; the L1 norm of its impulse response (for X = 1 per sample)
    double achieved = 0;                      // the tail actually left out (relative to the L1 norm), the worst member
    std::vector<std::int64_t> b1, b2;         // member support on the window grid
    std::vector<Eigen::VectorXd> gain;        // g_m on window bins b1..b2 (masked to the support)
    Eigen::VectorXd frame;                    // S = sum g^2 on window bins 0..N/2
};
/** The layout of an ANALYTIC bank (a real recording). */
WtLayout wtLayout(const timefilterbank::TimeFilterbank& bank, const WtOptions& options = WtOptions());

/** The own-rate coefficients of `x` [channels x n], a window of the record taken as zero outside, for t in [-margin, n + margin) from its first column: per member, instants first = (start - margin) q/p on. `start` is the
 *  window's first sample in the record (a multiple of the lattice period P: the lattice and the baseband shift are the RECORD's, so the coefficients do not depend on where the window starts). `members`: those wanted (empty: all; the others are empty blocks). */
std::vector<Block> wt(const WtLayout& layout, const Eigen::MatrixXd& x, const std::vector<int>& members = {}, std::int64_t start = 0);
/** x[0, n) back from the blocks wt(layout, x) gave: [channels x n]. Dual g/S or Tight g/G^2. */
Eigen::MatrixXd iwt(const WtLayout& layout, const std::vector<Block>& blocks, std::int64_t n, graphfilterbank::Synthesis mode = graphfilterbank::Synthesis::Dual, std::int64_t start = 0);

}  // namespace nxr::signal

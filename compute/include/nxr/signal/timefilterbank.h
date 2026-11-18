#pragma once
// nxr::signal::timefilterbank — the TIME filterbank: gains on FFT bins (the temporal half of the joint time–vertex bank).
//
// PURE MATHS (arrays in, arrays out; FFT by nxr::signal, with scipy's fast-length padding). Same GraphFilterbank/Bounds vocabulary as nxr::signal::graphfilterbank (graph).
// Design note and where Rheome, nxr and the Studio differ: docs/specs/2026-11-10-filterbanks.md (part 2).
//
// THREE REQUIRED CHOICES — a bank has no default family, no default gain, no default sides (Design's only constructor takes all three, and the
// bank records them):
//
//   family   Itersine  Rheome's @timefilterbank: tight constant-Q frame, itersine windows 2 voices wide on ξ = V·log2(f/anchor), anchored at an
//                      absolute frequency (1 Hz), a low-pass and a high-pass member completing it.                         (tight)
//            Meyer     the Studio's / nxr's dyadic frame (dyadicFrame.ts, frames.py window_frame): Meyer-edged octave bands.        (tight)
//            Morlet    a Gaussian in frequency, unit peak, width fc/cycles (Studio kernels.ts `morlet`, nxr kernels.morlet_hat).      (not tight)
//            Morse · Amor · Bump   the continuous-wavelet banks of MATLAB's cwtfilterbank (Rheome's reference path, +ingest/bank.m).  (not tight)
//   gain     a positive number G (1, 2, ...) — ONE meaning in every family: the PEAK in-band response of each member (below). Recorded on the bank.
//   sides    Analytic | TwoSided.
//
// SIDES.  Analytic: a one-sided bank on rfftfreq bins (f ≥ 0) for a REAL signal; the coefficients are complex (envelope + phase). A COMPLEX input is
//   REFUSED (std::invalid_argument): a complex signal has content at −f that a one-sided bank would drop silently.
//   TwoSided: separate members at +f and at −f on fftfreq bins, tight over the whole axis (a complex signal has no mirror half) — for
//   complex eigenmode coefficients, rotating patterns (+f counter-clockwise, −f clockwise), x + iy fields, and the joint bank. Every member lies on
//   one sign only — the low-pass / scaling too (a rotation slower than the first band keeps its sense) — except the DC bin, which has no sense and
//   is a member of its own (Role::Dc).
// GAIN — ONE MEANING IN EVERY FAMILY: the PEAK in-band response of each member is G (any positive number, recorded on the bank). A real tone of amplitude
//   A inside a member's passband then gives |c| = (G/2)·A on the ANALYTIC bank (G = 2: |c| = A, the physical amplitude; G = 1: A/2); a complex tone of
//   amplitude A on a TWO-SIDED bank gives |c| = G·A. The tight families (Itersine, Meyer) are G times the unit-sum frame, so  Σ_m g_m² = G²  on
//   0 < f < fs/2 and the frame is exact for every G:   analytic  Σ_m ‖w_m‖² = (G²/2) Σ x²,  reconstruction  x = (2/G²) Re ifft(Σ_m g_m ⊛ w_m);
//   two-sided  Σ_m ‖w_m‖² = G² Σ|z|²,  z = (1/G²) ifft(Σ g_m ⊛ w_m).  The non-tight families (Morlet, Morse, Amor, Bump) have no constant sum; the peak is G.
//   (Before the 2026-11-10 ruling "gain 2" was a constant Σg² = 2 for the tight families — a peak of √2, |c| = 0.71·A — and a peak of 2 for the others.)
// DC and NYQUIST (real, unmirrored bins) are handled identically for every family and both gains (EdgeBins): Single (default) gives each of them the
//   interior gain / √2, i.e. S = G/2 there — a bin that stands for itself, not for itself and a mirror — so the bank is exactly invertible and Parseval
//   holds for the whole record; Drop gives them gain 0 (Rheome's timefilterbank: the bank excludes them, Σ‖w‖² = Σx² "up to the DC and Nyquist bins").
//   Nothing is ever doubled.
#include "nxr/signal/fourier.h"
#include "nxr/signal/graphfilterbank.h"
#include <complex>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nxr::signal::timefilterbank {

using cd = std::complex<double>;
using Gain1 = std::function<double(double)>;   // g(f), f in Hz (signed)

enum class Family { Itersine, Meyer, Morlet, Morse, Amor, Bump, PowerPartition };
enum class Sides { Analytic, TwoSided };
enum class EdgeBins { Drop, Single };
enum class Padding { None, FastLen };          // None: the record is periodic (Rheome); FastLen: zero-padded to next_fast_len(2n), no wrap (the Studio pads to 2n)
enum class Role { Lowpass, Band, Highpass, Scaling, Wavelet, Top, Dc };

struct Design {
    Family family; double gain; Sides sides;    // required: the gain G > 0 is the peak in-band response of every member
    Design(Family f, double g, Sides s) : family(f), gain(g), sides(s) {}
    EdgeBins edgeBins = EdgeBins::Single;
    // Itersine (Rheome's defaults: 4 voices, anchor 1 Hz, ≥ 8 bins of support)
    int voices = 4; double anchor = 1.0; std::optional<std::pair<double, double>> limits; int minBins = 8;
    // Meyer: ascending edges (empty: windowEdges(nfft, fs, limits) — one octave per edge up to fs/4), transition half-width in octaves
    std::vector<double> edges; double halfWidth = 1.0 / 3.0;
    // Meyer, the Signal chain's BANDS instead of the frame: one member a band [lo, hi] Hz, the Studio's `meyerBand` (toolbox/wavelets/dyadicFrame.ts: sin of the
    // rise at lo, cos of the fall at hi, each over ±halfWidth octaves) times G; its edges may be closer than two half-widths (alpha, [8, 12]: then its peak is
    // below G). Not a frame (not tight). When given, `edges` is not used.
    // Morlet with `bands`: the project's fixed-band rule, one member a band: a Gaussian, unit peak (times G), centred on (lo + hi)/2, FWHM = hi - lo; neighbouring bands overlap.
    // PowerPartition (needs `bands`, adjacent and ascending): nxr's `band_gains`: squared gains sum to 1 inside [first low, last high] (a cos/sin crossover half an octave wide in log2 f), so band powers add to the broadband (Parseval).
    std::vector<std::pair<double, double>> bands;
    // Morlet · Morse · Amor · Bump: the member centres, Hz (required; see logCentres / cwtCentres)
    std::vector<double> centres; double cycles = 6;       // Morlet: width fc / cycles
    double gamma = 3, timeBandwidth = 60;                 // Morse (MATLAB's default (3, 60)): β = timeBandwidth / γ
};

struct Member {
    Role role = Role::Band;
    int sign = +1;                                         // +1: f > 0 · −1: f < 0 · 0: symmetric (the shared low-pass / scaling)
    double fc = std::numeric_limits<double>::quiet_NaN();  // centre, Hz (|f|); NaN: use centerFrequencies' centroid
    double lo = std::numeric_limits<double>::quiet_NaN(), hi = std::numeric_limits<double>::quiet_NaN();   // nominal band, Hz (|f|)
    Gain1 gain;                                            // g(f) of signed f; the DC / Nyquist rule is NOT in it (see gains(bank))
};

struct TimeFilterbank {
    Family family; double gain; Sides sides; EdgeBins edgeBins;
    std::size_t nfft = 0;                                  // the grid the bank is designed on (the padded record length)
    double fs = 0;
    std::vector<Member> members;
    double total = std::numeric_limits<double>::quiet_NaN();   // Σ g² on the interior where the design makes it constant (G²), else NaN
    std::vector<std::pair<std::size_t, std::size_t>> support;  // first and last bin of each member's support on the grid (analytic: rfft bins)
    int size() const { return static_cast<int>(members.size()); }
};

/** The family named "morlet" | "power_partition" (the node parameter `shape`); throws std::invalid_argument naming both. */
Family bandFamilyOf(const std::string& name);
const char* bandFamilyName(Family family);
/** [bands x freqs] amplitude gains at |f| (freqs in Hz, any sign) of the band shapes Morlet and PowerPartition (a function of the bands alone, no grid): the gains the bank gives its members, divided by G. */
Eigen::MatrixXd bandGains(const Eigen::VectorXd& freqs, const std::vector<std::pair<double, double>>& bands, Family family);

/** The record length the bank is designed on: n (None) or next_fast_len(2n) (FastLen). */
std::size_t gridLength(std::size_t n, Padding padding);
/** MATLAB cwtfilterbank's record extension: [x; flipud(x)] (2n rows) — design the bank on 2n and keep the first n coefficient rows. */
Eigen::MatrixXd extendSymmetric(const Eigen::MatrixXd& x);
/** The Studio's / nxr's window_frame edges: fs·2^k/nfft while ≤ fs/4, kept within `limits` if given. */
std::vector<double> windowEdges(std::size_t nfft, double fs, const std::optional<std::pair<double, double>>& limits = std::nullopt);
/** The Studio's wavelet_freqs: f0·2^(k/voices) up to f1 (capped at fs/2.5), ascending. */
std::vector<double> logCentres(double fs, double f0, double f1, int voices);
/** cwtfilterbank's grid as Rheome documents it (fc(k) = fmax·2^(−k/V), down to fmin), returned ASCENDING. */
std::vector<double> cwtCentres(double fmin, double fmax, int voices);

/** Build the bank on a grid of `nfft` samples at `fs`. Throws on an unsatisfiable design (no member fits; gain <= 0; missing centres). */
TimeFilterbank design(const Design& d, std::size_t nfft, double fs);

/** The grid's frequencies: rfftfreq(nfft, 1/fs) (Analytic) or fftfreq(nfft, 1/fs) (TwoSided). */
Eigen::VectorXd bins(const TimeFilterbank& bank);
/** Every member's gain on the bank's grid, [bins × M], with the DC / Nyquist rule (Analytic) applied. */
Eigen::MatrixXd gains(const TimeFilterbank& bank);
/** Every member's gain at arbitrary signed frequencies, [F × M] — the continuous design, no DC / Nyquist rule. */
Eigen::MatrixXd gains(const TimeFilterbank& bank, const Eigen::VectorXd& f);
/** Canonical dual gains g_m / S on the grid [bins × M]; 0 where S ≤ 1e-12·max S. */
Eigen::MatrixXd dualGains(const TimeFilterbank& bank);
/** Σ_m g_m² on the grid. */
Eigen::VectorXd frameOperator(const TimeFilterbank& bank);

/** Frame bounds of S = Σ g² on the interior bins (Analytic: DC and Nyquist excluded, as Rheome; TwoSided: every bin). Raw Σg² — Rheome reports S/2. */
signal::graphfilterbank::Bounds framebounds(const TimeFilterbank& bank);
bool isframetight(const TimeFilterbank& bank, double tol = 1e-6);
/** Centre of every member, Hz: the design's, or the g²-weighted centroid of |f| over the support where it has none. */
Eigen::VectorXd centerFrequencies(const TimeFilterbank& bank);
/** Rheome's rates(): the sampling rate of each member's own sub-band (own-rate) transform: next_fast_len(oversample·bins)·fs/nfft; and the lengths. */
std::pair<Eigen::VectorXd, std::vector<std::size_t>> rates(const TimeFilterbank& bank, double oversample = 2.0);

// ── the transform ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One [nfft × C] complex block per member (C signals, one per column). Rows beyond the record's n are its zero padding. */
using Coefficients = std::vector<Eigen::MatrixXcd>;
using nxr::signal::graphfilterbank::Synthesis;

/** Analytic: a REAL signal x [n × C] (n ≤ nfft; zero-padded). W_m = ifft(rfft(x) · g_m) on the one-sided bins. */
Coefficients analysis(const TimeFilterbank& bank, const Eigen::MatrixXd& x);
/** TwoSided: a complex (or real) signal. W_m = ifft(fft(x) · g_m). Analytic REFUSES this overload (std::invalid_argument). */
Coefficients analysis(const TimeFilterbank& bank, const Eigen::MatrixXcd& x);
/** Analytic synthesis → the real record [n × C]. Dual: Σ (g/S) ⊛ W, exact wherever the bank covers; Tight: Σ g ⊛ W / S_nominal (G², G²/2 at DC and
 *  Nyquist) — exact when the bank is tight, which is x = (2/G²) Re ifft(Σ g W) on the interior bins. */
Eigen::MatrixXd synthesis(const TimeFilterbank& bank, const Coefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);
/** TwoSided synthesis → the complex record [n × C]. */
Eigen::MatrixXcd synthesisComplex(const TimeFilterbank& bank, const Coefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);

/** |W_m|² per sample: the scalogram [nfft × C] of each member. */
std::vector<Eigen::MatrixXd> scalogram(const Coefficients& W);
/** Σ over samples of |W_m|², [M × C]. Parseval for a tight analytic bank with EdgeBins::Single: Σ_m ‖w_m‖² = (G²/2) Σ x² (two-sided: G² Σ|z|²). */
Eigen::MatrixXd energies(const Coefficients& W);

/** Rheome's wt: one real signal; per member the coefficients at the member's OWN rate (inverse FFT of its bins alone, oversampled).
 *  Analytic only. coef[t] = ifft(bins · g) · (Nm / nfft) at instants t = k·nfft/(Nm·fs). */
struct OwnRate { Eigen::VectorXcd coef; Eigen::VectorXd t; double rate = 0, weight = 0; };
std::vector<OwnRate> analysisOwnRate(const TimeFilterbank& bank, const Eigen::VectorXd& x, double oversample = 2.0);

// ── the time-domain primitives the joint bank is built from ─────────────────────────────────────────────────────────────

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

// ── THE TRANSFORM OF A HALOED BLOCK (a chunk with its context, as io's chunk service hands it) ──
// A block [channels x width] whose own columns are [left, left + own): the bank is designed on the BLOCK's grid (gridLength(width, FastLen): zero-padded,
// no wrap), the block transformed (FFT, the members' gains on the bins, inverse FFT: `analysis`), and only the own columns kept: one [channels x own]
// complex block per member. With a halo of at least haloOf(bank, member) on each side, the own columns are what the whole record gives there (the halo carries the
// context the band's response reaches); where a recording ENDS the block has no context and the transform zero-pads (the taps route reflects there).
// The band's choice (a Meyer band by its edges, Morlet centres, …) is the caller's Design; the member a run keeps is the caller's too (`members`).
struct BlockTransform {
  TimeFilterbank bank;                              // the bank as designed on the block's grid (its members, their nominal bands)
  Coefficients own;                           // per member: [channels x own] complex, the block's own columns
};
BlockTransform analysisOfBlock(const Design& design, const Eigen::MatrixXd& block, double fs, std::size_t left, std::size_t own);
/** THE HALO a member needs, samples on each side: the length of its response, as the Studio's device chain sizes its HALO by the band's taps (deviceChain.ts:
 *  the gain's even and odd taps, truncated where the tail holds under nxr::signal::kTailEnergy of the energy; the longer of the two). A narrow band rings for
 *  longer than the Studio's CPU rule of thumb (3 periods of its low edge): alpha [8, 12] Hz needs about 6 periods. */
std::size_t haloOf(const TimeFilterbank& bank, int member);

// ── THE INDEPENDENT ESTIMATES a band gives over a window (for the statistics of its cross-spectrum: nxr::stats::lagStatistics) ──
/** The member's STATISTICAL (equivalent) bandwidth, Hz: B = (∫ g² df)² / ∫ g⁴ df over f ≥ 0 (the bandwidth of white noise passed through it; a
 *  rectangular band of width W gives W). Evaluated on `grid` points of [0, fs/2]. */
double equivalentBandwidth(const TimeFilterbank& bank, int member, int grid = 1 << 16);
/** The number of independent complex estimates n = B·T the member gives over `samples` at the bank's rate (T = samples / fs): the n of the coherence-
 *  phase statistics (Bendat & Piersol). Tested against a simulation (test_lag): the spread of a band's estimated lag. */
double independentEstimates(const TimeFilterbank& bank, int member, std::int64_t samples);
/** THE NOISE BANDWIDTH of an ANALYTIC member's band-pass, Hz: Re z responds to a tone at f > 0 with g(f) / 2 (a tone of amplitude A in the plateau gives
 *  |c| = (G/2) A), so white noise of one-sided density d gives var(Re z) = d ∫₀^{fs/2} (g/2)² df = d · B_n, and ⟨|z|²⟩ = 2 var(Re z). A density from
 *  the analytic signal is ⟨|z|²⟩ / (2 B_n) (nxr::spectral, Stored::BandPower). Evaluated on `grid` points of [0, fs/2]. */
double noiseBandwidth(const TimeFilterbank& bank, int member, int grid = 1 << 16);

// ── DECIMATION of a band-limited block (the analytic signal of a band, whose content sits far below the new Nyquist rate: nothing aliases) ──
/** Every d-th column of Z [channels x T], starting at column `offset` (0 ≤ offset < d): [channels x ⌈(T − offset) / d⌉]. The caller places the grid (a
 *  global one across chunks: offset = (d − (first − s0) mod d) mod d for a block whose first column is sample `first` of a span starting at s0). Not a
 *  filter: a signal that is not already band-limited below fs / (2d) aliases. Σ z zᴴ over the kept columns is the full-rate sum divided by d, up to the
 *  band-limited signal's sampling error. */
Eigen::MatrixXcd downsample(const Eigen::MatrixXcd& Z, int d, int offset = 0);

}  // namespace nxr::signal::timefilterbank

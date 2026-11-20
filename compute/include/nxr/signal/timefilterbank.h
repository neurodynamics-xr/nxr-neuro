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
//            Gabor     unit-energy Gabor atoms with a FREE width: one member an atom (σ, f), the matched filter of a Gaussian burst (gaborGrid; the reference detector,
//                      docs/specs/2026-11-20-gabor-bursts.md). Analytic only.                                                          (not tight)
//   gain     a positive number G (1, 2, ...) — ONE meaning in every family but Gabor: the PEAK in-band response of each member (below). Recorded on the bank.
//            Gabor has its own NORMALISATION (recorded on the bank, `normalisation`): G times the UNIT-ENERGY atom, Ψ(ν) = G (πσ²)^(−1/4) √(2π) σ e^(−2π²σ²(ν−f)²) on ν > 0, so that
//            W = ifft(X Ψ) is the inner product ∫ x ψ* dt (G = 1): a Gaussian burst A e^(−t²/2s²) cos(2πft + φ) gives |W| = (A/2) π^(1/4) √σ at σ = s, and A = 2|W| / (π^(1/4) √σ).
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

enum class Family { Itersine, Meyer, Morlet, Morse, Amor, Bump, PowerPartition, Gabor };
/** what the gain G multiplies: the member's PEAK response (every family but Gabor), or the UNIT-ENERGY atom (Gabor, the matched filter) */
enum class Normalisation { Peak, UnitEnergy };
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
    // Gabor: the atoms (σ in s, f in Hz), one member each, in the order given (gaborGrid makes the reference detector's grid); required
    std::vector<std::pair<double, double>> atoms;
};

struct Member {
    Role role = Role::Band;
    int sign = +1;                                         // +1: f > 0 · −1: f < 0 · 0: symmetric (the shared low-pass / scaling)
    double fc = std::numeric_limits<double>::quiet_NaN();  // centre, Hz (|f|); NaN: use centerFrequencies' centroid
    double lo = std::numeric_limits<double>::quiet_NaN(), hi = std::numeric_limits<double>::quiet_NaN();   // nominal band, Hz (|f|)
    double width = std::numeric_limits<double>::quiet_NaN();   // Gabor: the atom's width σ, s (lo, hi = f ∓ σ_f, σ_f = 1/(2πσ)); NaN for the other families
    Gain1 gain;                                            // g(f) of signed f; the DC / Nyquist rule is NOT in it (see gains(bank))
};

struct TimeFilterbank {
    Family family; double gain; Sides sides; EdgeBins edgeBins;
    Normalisation normalisation = Normalisation::Peak;     // what the gain multiplies (Gabor: the unit-energy atom)
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
/** Every family by name: "itersine" | "meyer" | "morlet" | "morse" | "amor" | "bump" | "power_partition" (for a Design described in data); throws naming them. */
Family familyOf(const std::string& name);
/** "analytic" | "two_sided"; throws naming both. */
Sides sidesOf(const std::string& name);
/** [bands x freqs] amplitude gains at |f| (freqs in Hz, any sign) of the band shapes Morlet and PowerPartition (a function of the bands alone, no grid): the gains the bank gives its members, divided by G. */
Eigen::MatrixXd bandGains(const Eigen::VectorXd& freqs, const std::vector<std::pair<double, double>>& bands, Family family);

/** The record length the bank is designed on: n (None) or next_fast_len(2n) (FastLen). */
std::size_t gridLength(std::size_t n, Padding padding);
/** MATLAB cwtfilterbank's record extension: [x; flipud(x)] (2n rows) — design the bank on 2n and keep the first n coefficient rows. */
Eigen::MatrixXd extendSymmetric(const Eigen::MatrixXd& x);
/** The Studio's / nxr's window_frame edges: fs·2^k/nfft while ≤ fs/4, kept within `limits` if given. */
std::vector<double> windowEdges(std::size_t nfft, double fs, const std::optional<std::pair<double, double>>& limits = std::nullopt);
/** THE MEMBER CENTRES a design asks for (Hz, ascending), one function with the grid as its option:
 *  Studio: the Studio's wavelet_freqs, f0 * 2^(k/voices) up to f1 (capped at fs/2.5); needs `fs`.
 *  Cwt:    cwtfilterbank's grid as Rheome documents it, fc(k) = fmax * 2^(-k/V) down to fmin (`lo` = fmin, `hi` = fmax; `fs` unused). */
enum class CentreGrid { Studio, Cwt };
std::vector<double> centerFrequencies(CentreGrid grid, double lo, double hi, int voices, double fs = 0);

/** THE GABOR GRID (the reference detector's `widths` and `channel_atoms`): widths σ_k = sigmaMin·2^(k/perOctave), k = 0 .. ⌊perOctave·log2(sigmaMax/sigmaMin)⌋; at each width the frequencies
 *  from max(fMin − σ_f, minFSigma/σ) to fMax + σ_f in steps of σ_f/2 (σ_f = 1/(2πσ): a margin of one σ_f, so a peak at the band's edge is told from a maximum); a width whose first frequency is above fMax
 *  has none. minFSigma (f·σ; 0.5 is about 1.2 cycles at half maximum) keeps the Gaussian off DC. Atoms are width-major, ascending f within a width. Throws on 0 >= sigmaMin > sigmaMax, perOctave < 1, fMin > fMax. */
struct GaborGrid {
    std::vector<double> widths;                            // σ_k, s
    std::vector<std::pair<double, double>> atoms;          // (σ, f): width-major, f ascending
    std::vector<int> widthOf;                              // per atom, its width's index k
};
GaborGrid gaborGrid(double sigmaMin, double sigmaMax, int perOctave, double fMin, double fMax, double minFSigma);

/** THE BAND'S GAIN AS A SUM OF MORLET MEMBERS (the reference detector's R² signal: its ladder of Morlet members, `voices` an octave from ladderMin to ladderMax, each of width fc/cycles with
 *  cycles = 1 / (2^(1/2V) − 2^(−1/2V)), so Σ g² is flat inside the ladder; normalised so the largest Σ g² over [2 ladderMin, ladderMax / 2] is 1): Σ g_m(f)² over the members whose centre lies in [lo, hi].
 *  A real gain of |f| (Hz); the members are the bank's Morlet family. Throws on 0 < ladderMin < ladderMax, voices >= 1 not met. */
std::function<double(double)> morletLadderGain(double lo, double hi, double ladderMin, double ladderMax, int voices);

/** Build the bank on a grid of `nfft` samples at `fs`. Throws on an unsatisfiable design (no member fits; gain <= 0; missing centres). */
TimeFilterbank timeBank(const Design& d, std::size_t nfft, double fs);

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
/** Rheome's freqz: every member's frequency response on the bank's grid H [bins x M] (gains, the DC / Nyquist rule applied) and the frequencies f (bins). */
struct Freqz { Eigen::MatrixXd H; Eigen::VectorXd f; };
Freqz freqz(const TimeFilterbank& bank);
/** Centre of every member, Hz: the design's, or the g²-weighted centroid of |f| over the support where it has none. */
Eigen::VectorXd centerFrequencies(const TimeFilterbank& bank);
/** Rheome's rates(): the sampling rate of each member's own sub-band (own-rate) transform: next_fast_len(oversample·bins)·fs/nfft; and the lengths. */
std::pair<Eigen::VectorXd, std::vector<std::size_t>> rates(const TimeFilterbank& bank, double oversample = 2.0);

// THE TRANSFORMS ARE NOT HERE: a bank only designs (nxr/signal/wavelet.h: wt, iwt, iwtComplex, wtBlock; the scalogram and energies are measurements, nxr/measurement/band_energy.h).
using nxr::signal::graphfilterbank::Synthesis;
/** One [nfft x C] complex block per member (C signals, one per column). Rows beyond the record's n are its zero padding. */
using Coefficients = std::vector<Eigen::MatrixXcd>;

// ── the time-domain primitives the transforms are built from (wt / iwt and the joint bank) ─────────────────────────────────────────────────────────────

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

// ── THE HALO of a haloed block (the transform of a block is wtBlock, nxr/signal/wavelet.h) ──
/** THE HALO a member needs, samples on each side: the length of its response, as the Studio's device chain sizes its HALO by the band's taps (deviceChain.ts:
 *  the gain's even and odd taps, truncated where the tail holds under nxr::signal::kTailEnergy of the energy; the longer of the two). A narrow band rings for
 *  longer than the Studio's CPU rule of thumb (3 periods of its low edge): alpha [8, 12] Hz needs about 6 periods. */
std::size_t haloOfTaps(const TimeFilterbank& bank, int member);

/** THE HALO a member needs to a stated tolerance, samples on each side: the shortest reach after which the sum of |impulse response| of the member (its gain masked to its support on the bank's grid, taken on a fine
 *  grid that is doubled while the tail has not decayed) beyond it is below `tolerance` times its L1 norm, at most 2^16. A block with this halo gives the member's coefficients to that tail of the whole record, whatever
 *  the chunk size. A pure function of the bank; the tolerance is required (a member whose gain jumps, e.g. a high-pass nonzero at Nyquist, has a 1/t tail that the cap ends). */
std::size_t haloOf(const TimeFilterbank& bank, int member, double tolerance);

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

/** The kind of each member (Rheome's `kinds`): "lowpass" (the low-pass and the scaling member), "band" (a band or a wavelet), "highpass" (the high-pass and the top member), or "dc" (the DC bin
 *  of a two-sided bank, which has no sense). One entry per member, in the bank's order. */
std::vector<std::string> kinds(const TimeFilterbank& bank);

}  // namespace nxr::signal::timefilterbank

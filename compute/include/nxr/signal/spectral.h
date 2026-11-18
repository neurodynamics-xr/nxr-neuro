#pragma once
// nxr::welch — Welch's power spectral density of sensor recordings, STREAMING: a state fed chunk after chunk.
//
// PURE MATHS (arrays in, arrays out; the transform is nxr::fft::r2c). No file I/O.
//
// THE ESTIMATE. A record x[0..T) of C channels is cut into segments of `segmentSamples` (n) samples, a new segment every `hop` = n - floor(n * overlap) samples,
// from sample 0, as many as fit whole (a partial last segment is dropped, as scipy.signal.welch and Brainstorm do). Each segment is detrended, multiplied by a
// window w, transformed (rfft), and |X_k|^2 is averaged over the segments:
//     Density   P_k = |X_k|^2 / (fs * sum w^2)      power per Hz   (scipy scaling="density"; Brainstorm's "physical" units)
//     Spectrum  P_k = |X_k|^2 / (sum w)^2           power          (scipy scaling="spectrum")
//     Chain     P_k = |X_k|^2 / (n * sum w^2)       the Studio's / nxr's chain Welch (nxr.spectral.chain._welch): power per sample, per bin
// and the ONE-SIDED spectrum doubles every bin but DC (and, for an even n, Nyquist): P_k *= 2, k = 1 .. ceil(n/2) - 1.
// Frequencies are k * fs / n, k = 0 .. n/2.
//
// ONE ACCUMULATION: the estimate is the mean over the family of segments of a per-segment quantity, `nxr::measurement::SegmentMean` (estimate.h, spec docs/specs/2026-11-17-consolidation.md);
// the per-segment quantities (detrend, window, FFT, |X|^2, X X^H, |A X|^2) are the pointwise transforms.
//
// STREAMING. `Welch::add(chunk)` takes [channels x samples] blocks of any size. The samples left over at the end of a chunk (the start of the next segment) are
// CARRIED into the next chunk, so segments cross chunk edges and the result does not depend on how the record was cut: segments are taken in the same order and
// summed in the same order, so the result is identical to the last bit for any chunk size. Memory: the carried samples (< n + hop) plus one chunk and one segment,
// whatever the length of the record.
//
// BAD SAMPLES. `add(chunk, good)` takes a per-sample mask; a segment that touches a sample that is not good is LEFT OUT (Brainstorm's bad segments; nxr's `good`).
// The segment grid does not move: segments keep their place, those touching a bad sample are skipped (nxr.spectral.chain._welch).
//
// NAMED OPTION SETS (each reference reachable by name): brainstormPsdOptions() (the defaults of Brainstorm's PSD), scipyWelchOptions() (scipy.signal.welch's
// defaults; nxr's qc.sensors.psd is this with a 4 s Hann window), nxrChainWelchOptions() (nxr.spectral.chain._welch).
#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/measurement/estimate.h"

namespace nxr::signal::spectral {

enum class Window { Hann, Hamming, Blackman, Rectangular };
enum class Detrend { None, Constant, Linear };
enum class Scaling { Density, Spectrum, Chain };

struct WelchOptions {
    double sfreq = 1.0;                       // sampling frequency, Hz (required in practice: sets the frequencies and the density)
    std::size_t segmentSamples = 256;         // n, samples per segment (use segmentSamplesFor to state it in seconds)
    double overlap = 0.5;                     // fraction of a segment shared with the next: hop = n - floor(n * overlap); in [0, 1)
    Window window = Window::Hann;
    bool periodic = true;                     // the DFT-even form of the window (w_j = f(2 pi j / n), scipy / numpy-based nxr); false: the symmetric form (w_j = f(2 pi j / (n - 1)), MATLAB hamming(n))
    Detrend detrend = Detrend::Constant;      // removed from each segment before the window (scipy: constant)
    Scaling scaling = Scaling::Density;
};

/** The segment length for a duration: round(seconds * sfreq) samples, then +1 if oddEven is set and the count is odd (nxr's chain wants an even n). */
std::size_t segmentSamplesFor(double seconds, double sfreq, bool makeEven = false);

/** THE SEGMENT GRID of a Welch estimate, as a pure rule (the time-tile amendment of docs/specs/2026-11-13-cycle-pattern.md, section 3).
 *  Rule `tile`: a segment is a TILE of the recording's line, 2^k samples; k is `level`, or log2 of `samples` (a power of two), or round(log2(seconds * sfreq)) (seconds snap to the nearest
 *  level; default 4 s); the hop must be a tile too (overlap 0, 1/2, 3/4, ...), so that segments start on the line's grid once the span snaps to level k. Rule `round` (Brainstorm's, scipy's):
 *  round(seconds * sfreq) samples, or `samples`, from the span's start. Rule `even` (nxr's chain): as `round`, made even. `level` applies to the rule `tile` only.
 *  `level` of the result is k under `tile` (the grid the span snaps to) and 0 otherwise (no snap). Refuses (std::invalid_argument) a segment that is not whole samples, a level outside 1..24,
 *  `level` with `samples`, a hop that is not a tile, an unknown rule. */
struct Segment {
    std::size_t samples = 0;
    int level = 0;
};
Segment segmentFor(const std::string& rule, const std::optional<double>& seconds, const std::optional<std::int64_t>& samples, const std::optional<int>& level, double sfreq, double overlap);

/** Brainstorm's PSD (process_psd / bst_psd) as its defaults and tutorial give it: 4 s, 50 % overlap, Hamming (symmetric, as MATLAB's hamming(n)),
 *  the mean of each segment removed, power per Hz (the "physical" units). */
WelchOptions brainstormPsdOptions(double sfreq, double segmentSeconds = 4.0);
/** scipy.signal.welch's defaults but for the segment length: Hann (periodic), constant detrend, density, noverlap = n // 2. */
WelchOptions scipyWelchOptions(double sfreq, std::size_t segmentSamples = 256);
/** nxr.spectral.chain._welch: periodic Hann, no detrend, the Chain scaling, n even, hop n // 2. */
WelchOptions nxrChainWelchOptions(double sfreq, double segmentSeconds);

/** The window of n samples as the options define it. */
Eigen::VectorXd windowOf(std::size_t n, Window window, bool periodic);
/** The window's EQUIVALENT NOISE BANDWIDTH per bin, Hz: fs Σw² / (Σw)² — what a Spectrum-scaled value is divided by to give the Density-scaled one
 *  (nxr::spectral). */
double enbwPerBin(const WelchOptions& options);

struct WelchResult {
    Eigen::MatrixXd psd;                      // [channels x frequencies]
    Eigen::VectorXd frequencies;              // Hz, segmentSamples/2 + 1 of them
    std::int64_t segments = 0;                // the segments averaged (0: the record held none; psd is then zero)
    std::int64_t segmentsSkipped = 0;         // whole segments left out for touching a bad sample
};

class Welch {
public:
    Welch(int channels, const WelchOptions& options);
    /** Feed the next chunk, [channels x samples]; every sample is good. */
    void add(const Eigen::MatrixXd& chunk);
    /** Feed the next chunk with a mask of the same number of samples; a segment touching a false sample is left out. */
    void add(const Eigen::MatrixXd& chunk, const std::vector<std::uint8_t>& good);
    /** The estimate of everything fed so far (the state is not changed: feeding may go on). */
    WelchResult result() const;
    std::int64_t samples() const { return samples_; }
    std::int64_t segments() const { return segments_; }
    const WelchOptions& options() const { return opt_; }

private:
    void consume();
    WelchOptions opt_;
    std::size_t n_, hop_;
    Eigen::VectorXd win_;
    double winSumSquares_ = 0, winSum_ = 0;
    Eigen::MatrixXd buf_;                     // [channels x carried]: samples from the next unprocessed segment's start on
    std::vector<std::uint8_t> bufGood_;
    std::size_t carried_ = 0;
    measurement::SegmentMean mean_;           // [channels x bins]: the mean over the segments of |X|^2 (before scaling and doubling), `estimate`'s one accumulation (its sum is what the scaling multiplies)
    std::int64_t samples_ = 0, segments_ = 0, skipped_ = 0, dropped_ = 0;   // dropped_: samples consumed from the stream's start (a segment's number is its start / hop)
};

/** A run of bins first..last (inclusive) of the one-sided spectrum, and whether its value is their mean (a band: Brainstorm's "Group in frequency bands") or their sum (a total). */
struct BinGroup {
    std::size_t first = 0, last = 0;
    bool mean = true;
};

struct CrossWelchResult {
    std::vector<Eigen::MatrixXd> S;           // per group: the REAL part of the cross-spectral density, [channels x channels], in the options' scaling (its diagonal is the Welch PSD summed or averaged over the group's bins)
    std::vector<Eigen::MatrixXd> Sq;          // per group, only when asked (`withQuadrature`): the IMAGINARY part Im C of the same cross-spectral density, [channels x channels], antisymmetric, same scaling
                                              //   (Im(X_k X_k^H) = B A^T - A B^T for X = A + iB); the complex density of a group is S + i Sq
    Eigen::VectorXd frequencies;              // Hz, of all bins
    std::int64_t segments = 0, segmentsSkipped = 0;
};

/** Welch's CROSS-spectral density, streaming, accumulated per GROUP of bins: S_g = scale/(segments) x (mean or sum over the group's bins of w_k Re(X_k X_k^H)), w_k the one-sided doubling.
 *  The same segments, detrending, window, scaling and bad-segment rule as `Welch`, which is its diagonal. Memory: channels^2 per group, never a spectrum per pair. A source kernel K gives the
 *  source PSD of the group as diag(K S_g K^T) (nxr::meeg::sourcePower), equal to Welch on K x. */
class CrossWelch {
public:
    /** `withQuadrature`: also accumulate the imaginary part Im C (the quadrature spectrum, nxr/quadrature.h) into `result().Sq`; off, `S` is exactly as before. */
    CrossWelch(int channels, const WelchOptions& options, const std::vector<BinGroup>& groups, bool withQuadrature = false);
    void add(const Eigen::MatrixXd& chunk);
    void add(const Eigen::MatrixXd& chunk, const std::vector<std::uint8_t>& good);
    CrossWelchResult result() const;
    std::int64_t segments() const { return segments_; }

private:
    void consume();
    WelchOptions opt_;
    std::size_t n_, hop_;
    std::vector<BinGroup> groups_;
    Eigen::VectorXd win_;
    double winSumSquares_ = 0, winSum_ = 0;
    Eigen::MatrixXd buf_;
    std::vector<std::uint8_t> bufGood_;
    std::size_t carried_ = 0;
    std::vector<measurement::SegmentMean> mean_;   // per group: Re (the cross-spectral density) and, with the quadrature, Im in the imaginary part
    bool quadrature_ = false;
    std::int64_t samples_ = 0, segments_ = 0, skipped_ = 0, dropped_ = 0;
};

struct LinearWelchResult {
    Eigen::MatrixXd psd;                      // [outputs x bins kept]: Welch's power of each output, in the options' scaling, one-sided doubling applied
    Eigen::VectorXd frequencies;              // Hz, of the bins kept (first..last of the constructor)
    std::int64_t segments = 0, segmentsSkipped = 0;
};

/** Welch's power spectrum of a LINEAR MAP of the channels, A x, STREAMING, without forming A x in time: every segment's Fourier coefficients X are mapped, A X, and |A X|^2 accumulated per row and
 *  bin. It is `Welch` of the rows of A x (the same segments, detrending, window, scaling, chunk carrying and bad-segment rule), kept only for the bins first..last. `rowsPerOutput` consecutive
 *  rows are SUMMED in power into one output (3 for a free-orientation kernel's x y z; 1 for none); the rows' count is a multiple of it. Memory held: [outputs x bins kept] doubles, plus a block of
 *  rows of the mapped coefficients (at most 4096 rows x the bins kept) and the carried samples; the map is READ in place and must outlive the object. Called with any matrix A: where it came from
 *  (a kernel, a spatial filter, an identity) is the caller's business. At the price of one rows x channels product per segment, against the cross-spectrum route's channels^2 per bin kept. */
class LinearWelch {
public:
    LinearWelch(const Eigen::MatrixXd& map, std::size_t rowsPerOutput, const WelchOptions& options, std::size_t firstBin, std::size_t lastBin);
    void add(const Eigen::MatrixXd& chunk);
    void add(const Eigen::MatrixXd& chunk, const std::vector<std::uint8_t>& good);
    LinearWelchResult result() const;
    std::int64_t segments() const { return segments_; }

private:
    void consume();
    const Eigen::MatrixXd& map_;
    std::size_t group_, first_, last_;
    WelchOptions opt_;
    std::size_t n_, hop_;
    Eigen::VectorXd win_;
    double winSumSquares_ = 0, winSum_ = 0;
    Eigen::MatrixXd buf_;
    std::vector<std::uint8_t> bufGood_;
    std::size_t carried_ = 0;
    measurement::SegmentMean mean_;           // [outputs x bins kept]: the mean over the segments of |A X|^2 (the one-sided doubling in it; the scaling is applied at result)
    Eigen::Index outputs_ = 0;
    std::int64_t samples_ = 0, segments_ = 0, skipped_ = 0, dropped_ = 0;
};

struct ComplexWelchResult {
    Eigen::MatrixXd psd;                      // [series x frequencies], TWO-SIDED, ascending signed frequency (numpy fftshift of fftfreq): -fs/2 ... fs/2 - df for an even n
    Eigen::VectorXd frequencies;              // Hz, segmentSamples of them, signed; the first is -fs/2 for an even n
    std::int64_t segments = 0;
    std::int64_t segmentsSkipped = 0;
};

/** Welch's PSD of COMPLEX series, two-sided, streaming (nxr.timefreq.psd of a complex series = scipy.signal.welch(z, return_onesided=False), the frequencies in ascending order).
 *  A pair z = x + i y turning from x towards y has its power at +f, the opposite sense at -f. The same segments, window, detrend (a complex segment is detrended as its real and
 *  imaginary parts are), scaling, chunk carrying and bad-segment rule as `Welch`; no bin is doubled (every bin of the DFT of n samples is its own frequency).
 *  Memory: series x (carried + chunk + segment) complex. The input is any complex array: where it came from is the caller's business. */
class ComplexWelch {
public:
    ComplexWelch(int series, const WelchOptions& options);
    void add(const Eigen::MatrixXcd& chunk);
    void add(const Eigen::MatrixXcd& chunk, const std::vector<std::uint8_t>& good);
    ComplexWelchResult result() const;
    std::int64_t segments() const { return segments_; }

private:
    void consume();
    WelchOptions opt_;
    std::size_t n_, hop_;
    Eigen::VectorXd win_;
    double winSumSquares_ = 0, winSum_ = 0;
    Eigen::MatrixXcd buf_;
    std::vector<std::uint8_t> bufGood_;
    std::size_t carried_ = 0;
    measurement::SegmentMean mean_;           // [series x n]: the mean of |X|^2 in the DFT's own order (0, 1, ..., -1)
    std::int64_t samples_ = 0, segments_ = 0, skipped_ = 0, dropped_ = 0;
};

/** The whole complex series in one call. */
ComplexWelchResult complexWelch(const Eigen::MatrixXcd& z, const WelchOptions& options, const std::vector<std::uint8_t>* good = nullptr);

/** The signed frequencies of a two-sided spectrum of n samples at fs, ascending (fftshift(fftfreq)). */
Eigen::VectorXd signedFrequencies(std::size_t n, double sfreq);

/** The whole record in one call (a Welch fed once). */
WelchResult welch(const Eigen::MatrixXd& x, const WelchOptions& options, const std::vector<std::uint8_t>* good = nullptr);

}  // namespace nxr::signal::spectral

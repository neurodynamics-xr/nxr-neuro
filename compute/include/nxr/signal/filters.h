#pragma once
// nxr::signal::filters — standard temporal filters of sensor recordings, STREAMING: a state fed chunk after chunk.
//
// PURE MATHS (arrays in, arrays out; the transform is nxr::signal). No file I/O.
//
// THE VERBS (MATLAB Signal Processing Toolbox names; `notch` for line noise):
//     lowpass   remove everything above an edge            highpass  remove everything below an edge
//     bandpass  keep one band                              bandstop  remove one band
//     notch     remove line noise at a frequency and its harmonics
// Each takes [channels x samples] and the options of a named reference. Every filter is ZERO PHASE and has no delay: the output sample n is the input sample n filtered.
//
// THE NAMED OPTION SET (the reference whose OUTPUT it reproduces):
//   brainstormFilterOptions(sfreq)  Brainstorm's filters (process_bandpass / process_notch / process_bandstop):
//       lowpass · highpass · bandpass   an even-order linear-phase FIR, Kaiser window, designed with kaiserord + fir1: ripple 1e-3 (60 dB stopband attenuation), the
//                                       transition band 0.5 Hz (1 Hz when the high-pass edge is above 5 Hz; 1 Hz for a low-pass alone), the delay (order/2) compensated;
//                                       the record's mean is removed before filtering and added back only when the filter has no high-pass part; the two ends of the record are
//                                       padded with the mean of its first / last K samples, K = the "99 % energy" duration of the impulse response (the transient).
//       notch                           the 2nd-order IIR notch of Mitra and Kuo with the given 3-dB bandwidth (default 1 Hz), forward and backward (filtfilt: zero phase),
//                                       one frequency after the other.
//       bandstop                        Butterworth (4th order: butter(4, [lower upper]/(fs/2), 'stop')), forward and backward, MATLAB filtfilt's ends.
//     IIR ends = MATLAB's filtfilt: the signal extended by an odd reflection of 3 x (the filter length - 1) samples, the filter started from its steady state.
// (MNE-Python's filter_data option set: not done.)
//
// STREAMING. `Filter::add(chunk)` takes [channels x samples] blocks of any size and returns the samples that are now FINAL: the output lags the input by the filter's
// look-ahead (R samples), and `finish()` flushes the rest (with the record's end). Every output block is computed from the input window it needs (L samples behind, R ahead);
// the record's two ends are treated as the reference treats them (ends are known: the window then begins at sample 0, or ends at the last). Memory: one chunk + L + R samples
// per channel.
//   FIR  L = R = (order)/2 : the output is the whole-record filter to rounding (~1e-12), for any chunk size.
//   IIR  forward-backward filters have an infinite impulse response, so L = R = the sum over the stages of the length after which the impulse response has decayed to
//        `tolerance` (absolute, relative to its L1 norm; default 1e-15); the stage is started from its steady state at the window's start. The output equals the whole-record
//        filter to ~`tolerance` x the signal's variation (~1e-12 for signals of order 1-10), for any chunk size; at the record's ends it is exact.
// Brainstorm's mean removal needs the RECORD's mean: a caller that streams passes the means of the window it filters (computed in a first pass, `RunningMean`).
#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace nxr::signal::filters {

enum class Kind { Lowpass, Highpass, Bandpass, Bandstop, Notch };
enum class Method { Fir, Iir };
enum class Edge { MeanOfTransient, Zeros };   // FIR: the two ends of the record are padded with the mean of the first / last K samples (Brainstorm) · nothing

struct FilterOptions {
    Kind kind = Kind::Bandpass;
    double sfreq = 1.0;                       // Hz
    double lowerEdge = 0;                     // Hz: highpass edge · bandpass lower passband edge · bandstop lower stopband edge
    double upperEdge = 0;                     // Hz: lowpass edge · bandpass upper passband edge · bandstop upper stopband edge
    std::vector<double> frequencies;          // notch: the line frequency and its harmonics, Hz
    std::optional<Method> method;             // unset: the reference's (FIR, but IIR for bandstop and notch)
    // ---- FIR
    double ripple = 1e-3;                     // Kaiser: the passband and stopband deviation (linear); the stopband attenuation is -20 log10(ripple) = 60 dB  (Brainstorm "relax": 1e-2)
    std::optional<double> transitionBandwidth;    // Hz, each transition band (the lower edge's with Firwin); unset: the reference's rule
    Edge edge = Edge::MeanOfTransient;
    double transientEnergy = 0.99;            // MeanOfTransient: K = the shortest one-sided impulse response holding this fraction of its energy
    std::optional<std::size_t> transientSamples;    // MeanOfTransient: K stated (samples), instead of the energy rule (Brainstorm's own rule for K is not published: ours agrees to within 1 sample)
    bool removeMean = true;                   // subtract the record's mean before filtering; added back only when the filter has no high-pass part (Brainstorm)
    // ---- IIR
    std::optional<double> notchBandwidth;     // Hz: Brainstorm the 3-dB bandwidth of each notch (default 1 Hz) · MNE the width of each stop band (default f / 200)
    int bandstopOrder = 4;                    // Butterworth prototype order N (the filter has 2N poles) (Brainstorm: fieldtrip's 'but', N = 4)
    double tolerance = 1e-15;                 // chunked IIR: the impulse response's decayed tail, relative to its L1 norm, that the look-behind / look-ahead may leave out
    std::size_t maxMargin = std::size_t(1) << 22;     // chunked IIR: the longest look-behind / look-ahead allowed (samples)
};

/** Brainstorm's filters as documented (see the top of this file). Set `kind`, the edges (or `frequencies`) afterwards, or use the verbs. */
FilterOptions brainstormFilterOptions(double sfreq);

struct Section { double b0, b1, b2, a1, a2; };          // one biquad, direct form II transposed, a0 = 1

struct Design {
    Method method = Method::Fir;
    Eigen::VectorXd b;                        // FIR: the coefficients (odd length, symmetric)
    std::vector<std::vector<Section>> stages; // IIR: one forward-backward stage per entry (a notch each; one for the band-stop), each a cascade of biquads
    std::size_t transient = 0;                // FIR: K, the one-sided impulse response holding `transientEnergy` of its energy (samples; at least 1)
    std::size_t lookBehind = 0, lookAhead = 0;    // L, R, samples
    double order() const { return method == Method::Fir ? static_cast<double>(b.size() - 1) : 0.0; }
    /** The frequency response at f Hz: FIR complex H (delay compensated: real, linear phase removed); IIR the zero-phase gain |H|^2 (real). */
    std::complex<double> response(double freqHz, double sfreq) const;
};

/** The filter the options describe. Invalid edges (not 0 < edge < Nyquist, lower >= upper, ...) throw std::invalid_argument. */
Design design(const FilterOptions& options);

/** Channel means over what is fed (a first pass, for Brainstorm's mean removal). */
class RunningMean {
public:
    explicit RunningMean(int channels) : sum_(Eigen::VectorXd::Zero(channels)) {}
    void add(const Eigen::MatrixXd& chunk) { sum_ += chunk.rowwise().sum(); n_ += chunk.cols(); }
    Eigen::VectorXd mean() const { return n_ ? Eigen::VectorXd(sum_ / static_cast<double>(n_)) : sum_; }
    std::int64_t samples() const { return n_; }
private:
    Eigen::VectorXd sum_;
    std::int64_t n_ = 0;
};

class Filter {
public:
    /** `mean`: the record's channel means (only read when options.removeMean; empty = each window's own mean is not used: no removal). */
    Filter(int channels, const FilterOptions& options, const Eigen::VectorXd& mean = Eigen::VectorXd());
    /** Feed the next chunk, [channels x samples]; returns the output samples that are final (maybe none), in order. */
    Eigen::MatrixXd add(const Eigen::MatrixXd& chunk);
    /** The end of the record: the rest of the output. No more chunks after this. */
    Eigen::MatrixXd finish();
    std::int64_t samplesIn() const { return total_; }
    std::int64_t samplesOut() const { return emitted_; }
    const FilterOptions& options() const { return opt_; }
    const Design& filterDesign() const { return design_; }
    std::size_t lookBehind() const { return design_.lookBehind; }
    std::size_t lookAhead() const { return design_.lookAhead; }
    /** ONE BLOCK with its halo (a windowed node): `w` [channels x width] whose own columns are [left, left + own); the filtered own columns. `atStart` / `atEnd`: the block
     *  reaches the record's first / last sample (its ends padded as the option set says). With a halo of lookBehind() on the left and lookAhead() on the right (or an end of
     *  the record), the own columns are what the streaming filter gives there. Stateless: blocks in any order. */
    Eigen::MatrixXd block(const Eigen::MatrixXd& w, std::size_t left, std::size_t own, bool atStart, bool atEnd) const;

private:
    Eigen::MatrixXd drain(bool final);
    Eigen::MatrixXd window(const Eigen::MatrixXd& w, bool atStart, bool atEnd) const;     // the filtered window; rows = channels
    Eigen::MatrixXd firWindow(const Eigen::MatrixXd& w, bool atStart, bool atEnd) const;
    Eigen::MatrixXd iirWindow(const Eigen::MatrixXd& w) const;
    FilterOptions opt_;
    Design design_;
    int channels_;
    Eigen::VectorXd mean_;
    bool restoreMean_ = false;
    std::size_t minBlock_ = 1;
    Eigen::MatrixXd buf_;                     // [channels x held]: the samples from bufStart_ on
    std::int64_t bufStart_ = 0, total_ = 0, emitted_ = 0;
    bool finished_ = false;
};

// The whole record in one call (a Filter fed once); the record's own channel means are used when options.removeMean.
Eigen::MatrixXd lowpass(const Eigen::MatrixXd& x, double upperEdge, FilterOptions options);
Eigen::MatrixXd highpass(const Eigen::MatrixXd& x, double lowerEdge, FilterOptions options);
Eigen::MatrixXd bandpass(const Eigen::MatrixXd& x, double lowerEdge, double upperEdge, FilterOptions options);
Eigen::MatrixXd bandstop(const Eigen::MatrixXd& x, double lowerEdge, double upperEdge, FilterOptions options);
Eigen::MatrixXd notch(const Eigen::MatrixXd& x, const std::vector<double>& frequencies, FilterOptions options);

/** The Kaiser window of n samples (symmetric, scipy / MATLAB kaiser(n, beta)). */
Eigen::VectorXd kaiser(std::size_t n, double beta);
/** MATLAB kaiserord for a transition of width `transition` Hz at sfreq and a deviation `ripple`: the (not yet even) order and beta. */
std::pair<std::size_t, double> kaiserord(double transitionHz, double sfreq, double ripple);

}  // namespace nxr::signal::filters

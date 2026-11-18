#pragma once
// nxr::signal — the chunked filterbank transform `wt` and its inverse `iwt` (names: Rheome's @timefilterbank and MATLAB's cwtfilterbank).
//
// PURE MATHS (arrays in, arrays out; the transform is nxr::signal). STREAMING: a state fed chunk after chunk, as nxr::signal::filters.
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
// times its L1 norm (default 1e-12), capped by maxMargin; `margin()` and `achievedTolerance()` report both. A member with a gain that jumps (e.g. a high-pass that
// is nonzero at Nyquist) has a 1/t tail: then achievedTolerance() > tolerance and the chunked result differs from the whole record by about that much, for signals
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

struct Options {
    double oversample = 2.0;                  // timebank::rates: the member's rate = next_fast_len(oversample * bins) fs / nfft
    double tolerance = 1e-12;                 // the impulse response's tail (L1, relative to its L1 norm) that the margin may leave out
    std::size_t maxMargin = std::size_t(1) << 16;
    std::size_t block = 0;                    // core samples per window (rounded up to a multiple of P); 0: max(L + R, min(4 (L + R), maxBlock))
    std::size_t maxBlock = std::size_t(1) << 15;   // the default block's cap: a window holds channels x (block + L + R) samples and their FFT, so it bounds memory
};

/** One member's run of coefficients: instants first, first+1, ...; rows = instants, columns = channels. */
struct Block { std::int64_t first = 0; Eigen::MatrixXcd coef; };

struct Plan {                                 // what wt and iwt share (a function of the bank and the options alone)
    signal::timefilterbank::TimeFilterbank bank;
    Options options;
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
    Plan(const signal::timefilterbank::TimeFilterbank& bank, const Options& options = Options());
};

class Wt {
public:
    Wt(const signal::timefilterbank::TimeFilterbank& bank, int channels, const Options& options = Options());
    /** Feed [channels x samples] (any size); returns, per member, the coefficients that are now final (possibly empty blocks). */
    std::vector<Block> add(const Eigen::MatrixXd& chunk);
    /** The end of the record: the rest, including the tails. */
    std::vector<Block> finish();
    std::int64_t samplesIn() const { return total_; }
    const Plan& plan() const { return plan_; }
    std::int64_t margin() const { return plan_.margin; }
    double achievedTolerance() const { return plan_.achieved; }
private:
    std::vector<Block> drain(bool final);
    Plan plan_; int channels_;
    Eigen::MatrixXd buf_; std::int64_t bufStart_ = 0, total_ = 0, next_;   // next_: the first sample of the next core
    bool finished_ = false;
};

class Iwt {
public:
    Iwt(const signal::timefilterbank::TimeFilterbank& bank, int channels, const Options& options = Options(), signal::graphfilterbank::Synthesis mode = signal::graphfilterbank::Synthesis::Dual);
    /** Feed the coefficients wt returned (any split in time, in order); returns the signal samples now final, [channels x samples]. */
    Eigen::MatrixXd add(const std::vector<Block>& blocks);
    /** The end: `samples` = the record's length n (wt's samplesIn()); returns the rest of x[0, n). */
    Eigen::MatrixXd finish(std::int64_t samples);
    const Plan& plan() const { return plan_; }
private:
    Eigen::MatrixXd drain(bool final, std::int64_t n);
    Plan plan_; int channels_; signal::graphfilterbank::Synthesis mode_;
    std::vector<std::vector<Eigen::VectorXcd>> have_;   // per member, per channel: the coefficients held, from instant start_[m]
    std::vector<std::int64_t> start_, end_;
    std::int64_t next_ = 0;
};

}  // namespace nxr::signal

// nxr::signal::spectral — see include/nxr/welch.h.
#include "nxr/signal/spectral.h"
#include "nxr/signal/fourier.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::signal::spectral {

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
static constexpr double kPi = 3.14159265358979323846;

std::size_t segmentSamplesFor(double seconds, double sfreq, bool makeEven) {
    if (!(seconds > 0) || !(sfreq > 0)) throw std::invalid_argument("welch: the segment length and the sampling frequency must be positive");
    std::size_t n = static_cast<std::size_t>(std::llround(seconds * sfreq));
    if (makeEven) n += n % 2;
    return n;
}

namespace {
int tileLevelOfWidth(std::int64_t n) {
    if (n < 1 || (n & (n - 1)) != 0) return -1;
    int k = 0;
    while ((std::int64_t(1) << k) < n) ++k;
    return k;
}
}  // namespace

Segment segmentFor(const std::string& rule, const std::optional<double>& seconds, const std::optional<std::int64_t>& samples, const std::optional<int>& level, double sfreq, double overlap) {
    if (rule != "tile" && rule != "round" && rule != "even") throw std::invalid_argument("welch: segment_rule is tile, round or even (got '" + rule + "')");
    if (!(sfreq > 0)) throw std::invalid_argument("welch: the sampling frequency must be positive");
    Segment out;
    if (rule != "tile") {
        if (level) throw std::invalid_argument("welch: segment_level applies to segment_rule tile (the " + rule + " rule states its segment in seconds or samples)");
        if (samples) out.samples = static_cast<std::size_t>(*samples);
        else out.samples = segmentSamplesFor(seconds.value_or(4.0), sfreq, rule == "even");
        return out;
    }
    if (level && samples) throw std::invalid_argument("welch: segment_level and segment_samples are alternatives");
    int k = 0;
    if (level) k = *level;
    else if (samples) {
        k = tileLevelOfWidth(*samples);
        if (k < 0) throw std::invalid_argument("welch: under segment_rule tile a segment is a tile of 2^k samples (got " + std::to_string(*samples) + "): give segment_level k, or segment_rule round|even");
    } else {
        const double sec = seconds.value_or(4.0);
        if (!(sec > 0)) throw std::invalid_argument("welch: segment_seconds is a positive duration");
        k = std::max(0, static_cast<int>(std::lround(std::log2(sec * sfreq))));
    }
    if (k < 1 || k > 24) throw std::invalid_argument("welch: a segment's tile level is 1 to 24 (got " + std::to_string(k) + ")");
    const std::int64_t n = std::int64_t(1) << k, hop = n - static_cast<std::int64_t>(std::floor(static_cast<double>(n) * overlap));
    if (tileLevelOfWidth(hop) < 0)
        throw std::invalid_argument("welch: under segment_rule tile the hop must be a tile too (" + std::to_string(hop) + " samples at overlap " + std::to_string(overlap) + "): overlap 0, 0.5, 0.75, 0.875, ...");
    out.samples = static_cast<std::size_t>(n);
    out.level = k;
    return out;
}

WelchOptions brainstormPsdOptions(double sfreq, double segmentSeconds) {
    WelchOptions o;
    o.sfreq = sfreq;
    o.segmentSamples = segmentSamplesFor(segmentSeconds, sfreq);
    o.overlap = 0.5;
    o.window = Window::Hamming;
    o.periodic = false;
    o.detrend = Detrend::Constant;
    o.scaling = Scaling::Density;
    return o;
}

WelchOptions scipyWelchOptions(double sfreq, std::size_t segmentSamples) {
    WelchOptions o;
    o.sfreq = sfreq;
    o.segmentSamples = segmentSamples;
    return o;   // the struct's defaults are scipy's
}

WelchOptions nxrChainWelchOptions(double sfreq, double segmentSeconds) {
    WelchOptions o;
    o.sfreq = sfreq;
    o.segmentSamples = segmentSamplesFor(segmentSeconds, sfreq, true);
    o.overlap = 0.5;
    o.window = Window::Hann;
    o.periodic = true;
    o.detrend = Detrend::None;
    o.scaling = Scaling::Chain;
    return o;
}

VectorXd windowOf(std::size_t n, Window window, bool periodic) {
    VectorXd w = VectorXd::Ones(n);
    if (window == Window::Rectangular || n < 2) return w;
    const double d = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);
    for (std::size_t j = 0; j < n; ++j) {
        const double c = std::cos(2 * kPi * static_cast<double>(j) / d);
        switch (window) {
            case Window::Hann: w[j] = 0.5 - 0.5 * c; break;
            case Window::Hamming: w[j] = 0.54 - 0.46 * c; break;
            case Window::Blackman: w[j] = 0.42 - 0.5 * c + 0.08 * std::cos(4 * kPi * static_cast<double>(j) / d); break;
            case Window::Rectangular: break;
        }
    }
    return w;
}

// the segment's row detrended (scipy's constant / linear) then windowed, in place
static void detrendAndWindow(double* row, std::size_t n_, Detrend detrend, const VectorXd& win_) {
    if (detrend == Detrend::Constant) {
        double mu = 0;
        for (std::size_t j = 0; j < n_; ++j) mu += row[j];
        mu /= static_cast<double>(n_);
        for (std::size_t j = 0; j < n_; ++j) row[j] -= mu;
    } else if (detrend == Detrend::Linear) {   // least-squares line over the segment (scipy detrend "linear")
        const double tm = 0.5 * static_cast<double>(n_ - 1);
        double mu = 0, sxy = 0, sxx = 0;
        for (std::size_t j = 0; j < n_; ++j) mu += row[j];
        mu /= static_cast<double>(n_);
        for (std::size_t j = 0; j < n_; ++j) { const double t = static_cast<double>(j) - tm; sxy += t * (row[j] - mu); sxx += t * t; }
        const double slope = sxy / sxx;
        for (std::size_t j = 0; j < n_; ++j) row[j] -= mu + slope * (static_cast<double>(j) - tm);
    }
    for (std::size_t j = 0; j < n_; ++j) row[j] *= win_[static_cast<Eigen::Index>(j)];
}

namespace {
// the segment model of a streamed run: a chunk joined to the carried tail, the whole segments transformed by `stft`, the samples they have passed dropped
template <class Matrix>
void carryAdd(SegmentCarry<Matrix>& carry, const Matrix& chunk, const std::vector<std::uint8_t>& good, const char* noun) {
    if (chunk.rows() != carry.block.rows()) throw std::invalid_argument("welch: the chunk has " + std::to_string(chunk.rows()) + " " + noun + ", expected " + std::to_string(carry.block.rows()));
    if (static_cast<Eigen::Index>(good.size()) != chunk.cols()) throw std::invalid_argument("welch: the mask must hold one flag per sample of the chunk");
    const Eigen::Index c0 = carry.block.cols(), m = chunk.cols();
    carry.block.conservativeResize(Eigen::NoChange, c0 + m);
    carry.block.middleCols(c0, m) = chunk;
    carry.good.insert(carry.good.end(), good.begin(), good.end());
}
template <class Matrix>
void carryDrop(SegmentCarry<Matrix>& carry, std::size_t start) {
    const std::size_t carried = static_cast<std::size_t>(carry.block.cols());
    carry.first += static_cast<std::int64_t>(start);
    const std::size_t keep = carried > start ? carried - start : 0;
    if (start > 0 && keep > 0) {
        carry.block.leftCols(static_cast<Eigen::Index>(keep)) = carry.block.middleCols(static_cast<Eigen::Index>(start), static_cast<Eigen::Index>(keep)).eval();
        carry.good.erase(carry.good.begin(), carry.good.begin() + static_cast<std::ptrdiff_t>(start));
    } else if (keep == 0) {
        carry.good.clear();
    }
    carry.block.conservativeResize(Eigen::NoChange, static_cast<Eigen::Index>(keep));
}
}  // namespace

SegmentFamily segmentFamily(const WelchOptions& options) {
    const std::size_t n = options.segmentSamples;
    if (n < 2) throw std::invalid_argument("welch: a segment holds at least 2 samples");
    if (!(options.sfreq > 0)) throw std::invalid_argument("welch: the sampling frequency must be positive");
    if (!(options.overlap >= 0 && options.overlap < 1)) throw std::invalid_argument("welch: the overlap is a fraction in [0, 1)");
    SegmentFamily f;
    f.samples = n;
    f.hop = n - static_cast<std::size_t>(std::floor(static_cast<double>(n) * options.overlap));
    f.window = windowOf(n, options.window, options.periodic);
    f.detrend = options.detrend;
    f.windowSumSquares = f.window.squaredNorm();
    f.windowSum = f.window.sum();
    return f;
}

double segmentScale(const WelchOptions& options, const SegmentFamily& family) {
    switch (options.scaling) {
        case Scaling::Density: return 1.0 / (options.sfreq * family.windowSumSquares);
        case Scaling::Spectrum: return 1.0 / (family.windowSum * family.windowSum);
        case Scaling::Chain: return 1.0 / (static_cast<double>(family.samples) * family.windowSumSquares);
    }
    return 0;
}

std::size_t stft(const MatrixXd& block, const std::vector<std::uint8_t>& good, const SegmentFamily& family, std::int64_t first,
                 const std::function<void(std::int64_t, const cd*, std::size_t)>& visit, const std::function<void(std::int64_t)>& skipped) {
    const Eigen::Index C = block.rows();
    const std::size_t n = family.samples, bins = rfftBins(n), width = static_cast<std::size_t>(block.cols());
    std::size_t start = 0;
    std::vector<double> seg(static_cast<std::size_t>(C) * n);
    std::vector<cd> spec(static_cast<std::size_t>(C) * bins);
    while (start + n <= width) {
        const std::int64_t id = (first + static_cast<std::int64_t>(start)) / static_cast<std::int64_t>(family.hop);   // the segment's number on the stream
        bool ok = true;
        for (std::size_t j = start; j < start + n && ok; ++j) ok = good[j] != 0;
        if (ok) {
            for (Eigen::Index c = 0; c < C; ++c) {
                double* row = seg.data() + static_cast<std::size_t>(c) * n;
                for (std::size_t j = 0; j < n; ++j) row[j] = block(c, static_cast<Eigen::Index>(start + j));
                detrendAndWindow(row, n, family.detrend, family.window);
            }
            r2c(seg.data(), spec.data(), {static_cast<std::size_t>(C), n}, 1);
            visit(id, spec.data(), bins);
        } else {
            skipped(id);
        }
        start += family.hop;
    }
    return start;
}

Welch::Welch(int channels, const WelchOptions& options) : opt_(options), n_(options.segmentSamples) {
    if (channels < 1) throw std::invalid_argument("welch: at least one channel");
    family_ = segmentFamily(opt_);
    carry_.block.resize(channels, 0);
}

void Welch::add(const MatrixXd& chunk) { add(chunk, std::vector<std::uint8_t>(static_cast<std::size_t>(chunk.cols()), 1)); }

void Welch::add(const MatrixXd& chunk, const std::vector<std::uint8_t>& good) {
    carryAdd(carry_, chunk, good, "channels");
    samples_ += chunk.cols();
    consume();
}

// the family's segments of what has been carried (stft), each one's |X|^2 into the mean (estimate)
void Welch::consume() {
    const Eigen::Index C = carry_.block.rows();
    MatrixXd power(C, static_cast<Eigen::Index>(rfftBins(n_)));
    const std::size_t passed = stft(carry_.block, carry_.good, family_, carry_.first,
        [&](std::int64_t id, const cd* spec, std::size_t bins) {
            for (Eigen::Index c = 0; c < C; ++c)
                for (std::size_t k = 0; k < bins; ++k) power(c, static_cast<Eigen::Index>(k)) = std::norm(spec[static_cast<std::size_t>(c) * bins + k]);
            mean_.add_real(id, power);
            ++segments_;
        },
        [&](std::int64_t id) { mean_.skip(id); ++skipped_; });
    carryDrop(carry_, passed);
}

WelchResult Welch::result() const {
    WelchResult r;
    const std::size_t bins = rfftBins(n_);
    r.frequencies = rfftfreq(n_, 1.0 / opt_.sfreq);
    r.segments = segments_;
    r.segmentsSkipped = skipped_;
    r.psd = MatrixXd::Zero(carry_.block.rows(), static_cast<Eigen::Index>(bins));
    if (segments_ == 0) return r;
    const double scale = segmentScale(opt_, family_);
    r.psd = mean_.sum().real() * (scale / static_cast<double>(segments_));
    const std::size_t last = (n_ % 2 == 0) ? bins - 1 : bins;   // doubled: 1 .. last - 1 (an even n keeps Nyquist single)
    for (std::size_t k = 1; k < last; ++k) r.psd.col(static_cast<Eigen::Index>(k)) *= 2.0;
    return r;
}

WelchResult psd(const MatrixXd& x, const WelchOptions& options, const std::vector<std::uint8_t>* good) {
    Welch w(static_cast<int>(x.rows()), options);
    if (good) w.add(x, *good); else w.add(x);
    return w.result();
}


// ── the two-sided Welch PSD of complex series ────────────────────────────────────────────────────────────────────

VectorXd signedFrequencies(std::size_t n, double sfreq) { return fftshift(fftfreq(n, 1.0 / sfreq)); }

// a complex segment's row detrended (the real and imaginary parts each, as scipy's lstsq on complex data does) then windowed, in place
static void detrendAndWindowC(cd* row, std::size_t n_, Detrend detrend, const VectorXd& win_) {
    using cd = cd;
    if (detrend == Detrend::Constant) {
        cd mu = 0;
        for (std::size_t j = 0; j < n_; ++j) mu += row[j];
        mu /= static_cast<double>(n_);
        for (std::size_t j = 0; j < n_; ++j) row[j] -= mu;
    } else if (detrend == Detrend::Linear) {
        const double tm = 0.5 * static_cast<double>(n_ - 1);
        cd mu = 0, sxy = 0;
        double sxx = 0;
        for (std::size_t j = 0; j < n_; ++j) mu += row[j];
        mu /= static_cast<double>(n_);
        for (std::size_t j = 0; j < n_; ++j) { const double t = static_cast<double>(j) - tm; sxy += t * (row[j] - mu); sxx += t * t; }
        const cd slope = sxy / sxx;
        for (std::size_t j = 0; j < n_; ++j) row[j] -= mu + slope * (static_cast<double>(j) - tm);
    }
    for (std::size_t j = 0; j < n_; ++j) row[j] *= win_[static_cast<Eigen::Index>(j)];
}

std::size_t stft(const MatrixXcd& block, const std::vector<std::uint8_t>& good, const SegmentFamily& family, std::int64_t first,
                 const std::function<void(std::int64_t, const cd*, std::size_t)>& visit, const std::function<void(std::int64_t)>& skipped) {
    const Eigen::Index S = block.rows();
    const std::size_t n = family.samples, width = static_cast<std::size_t>(block.cols());
    std::size_t start = 0;
    std::vector<cd> seg(static_cast<std::size_t>(S) * n), spec(static_cast<std::size_t>(S) * n);
    while (start + n <= width) {
        const std::int64_t id = (first + static_cast<std::int64_t>(start)) / static_cast<std::int64_t>(family.hop);
        bool ok = true;
        for (std::size_t j = start; j < start + n && ok; ++j) ok = good[j] != 0;
        if (ok) {
            for (Eigen::Index c = 0; c < S; ++c) {
                cd* row = seg.data() + static_cast<std::size_t>(c) * n;
                for (std::size_t j = 0; j < n; ++j) row[j] = block(c, static_cast<Eigen::Index>(start + j));
                detrendAndWindowC(row, n, family.detrend, family.window);
            }
            c2c(seg.data(), spec.data(), {static_cast<std::size_t>(S), n}, 1, true);
            visit(id, spec.data(), n);
        } else {
            skipped(id);
        }
        start += family.hop;
    }
    return start;
}


ComplexWelch::ComplexWelch(int series, const WelchOptions& options) : opt_(options), n_(options.segmentSamples) {
    if (series < 1) throw std::invalid_argument("welch: at least one series");
    family_ = segmentFamily(opt_);
    carry_.block.resize(series, 0);
}

void ComplexWelch::add(const Eigen::MatrixXcd& chunk) { add(chunk, std::vector<std::uint8_t>(static_cast<std::size_t>(chunk.cols()), 1)); }

void ComplexWelch::add(const Eigen::MatrixXcd& chunk, const std::vector<std::uint8_t>& good) {
    carryAdd(carry_, chunk, good, "series");
    samples_ += chunk.cols();
    consume();
}

void ComplexWelch::consume() {
    const Eigen::Index S = carry_.block.rows();
    MatrixXd power(S, static_cast<Eigen::Index>(n_));
    const std::size_t passed = stft(carry_.block, carry_.good, family_, carry_.first,
        [&](std::int64_t id, const cd* spec, std::size_t bins) {
            for (Eigen::Index c = 0; c < S; ++c)
                for (std::size_t k = 0; k < bins; ++k) power(c, static_cast<Eigen::Index>(k)) = std::norm(spec[static_cast<std::size_t>(c) * bins + k]);
            mean_.add_real(id, power);
            ++segments_;
        },
        [&](std::int64_t id) { mean_.skip(id); ++skipped_; });
    carryDrop(carry_, passed);
}

ComplexWelchResult ComplexWelch::result() const {
    ComplexWelchResult r;
    r.frequencies = signedFrequencies(n_, opt_.sfreq);
    r.segments = segments_;
    r.segmentsSkipped = skipped_;
    r.psd = MatrixXd::Zero(carry_.block.rows(), static_cast<Eigen::Index>(n_));
    if (segments_ == 0) return r;
    const double scale = segmentScale(opt_, family_);
    const std::size_t neg = n_ / 2;   // ascending: the DFT's bins neg-from-the-end first
    for (std::size_t k = 0; k < n_; ++k) r.psd.col(static_cast<Eigen::Index>(k)) = mean_.sum().real().col(static_cast<Eigen::Index>((k + n_ - neg) % n_)) * (scale / static_cast<double>(segments_));
    return r;
}

ComplexWelchResult psd(const Eigen::MatrixXcd& z, const WelchOptions& options, const std::vector<std::uint8_t>* good) {
    ComplexWelch w(static_cast<int>(z.rows()), options);
    if (good) w.add(z, *good); else w.add(z);
    return w.result();
}


// ── the cross-spectral density ───────────────────────────────────────────────────────────────────────────────────

CrossWelch::CrossWelch(int channels, const WelchOptions& options, const std::vector<BinGroup>& groups, bool withQuadrature) : opt_(options), n_(options.segmentSamples), groups_(groups), quadrature_(withQuadrature) {
    if (channels < 1) throw std::invalid_argument("welch: at least one channel");
    family_ = segmentFamily(opt_);
    const std::size_t bins = rfftBins(n_);
    if (groups_.empty()) throw std::invalid_argument("welch: at least one group of bins");
    for (const BinGroup& g : groups_) if (g.first > g.last || g.last >= bins) throw std::invalid_argument("welch: a group of bins lies outside 0 .. " + std::to_string(bins - 1));
    carry_.block.resize(channels, 0);
    mean_.assign(groups_.size(), measurement::SegmentMean());
}

void CrossWelch::add(const MatrixXd& chunk) { add(chunk, std::vector<std::uint8_t>(static_cast<std::size_t>(chunk.cols()), 1)); }

void CrossWelch::add(const MatrixXd& chunk, const std::vector<std::uint8_t>& good) {
    carryAdd(carry_, chunk, good, "channels");
    samples_ += chunk.cols();
    consume();
}

void CrossWelch::consume() {
    const Eigen::Index C = carry_.block.rows();
    const double last = (n_ % 2 == 0) ? static_cast<double>(rfftBins(n_) - 1) : static_cast<double>(rfftBins(n_));   // bins 1 .. last-1 are doubled (one-sided), see Welch::result
    const std::size_t passed = stft(carry_.block, carry_.good, family_, carry_.first,
        [&](std::int64_t id, const cd* spec, std::size_t bins) {
            // Re(X_k X_k^H) = A A^T + B B^T (X = A + iB), summed over the group's bins with the one-sided doubling in the weights
            for (std::size_t g = 0; g < groups_.size(); ++g) {
                const std::size_t nk = groups_[g].last - groups_[g].first + 1;
                MatrixXd A(C, static_cast<Eigen::Index>(nk)), B(C, static_cast<Eigen::Index>(nk));
                for (Eigen::Index c = 0; c < C; ++c)
                    for (std::size_t k = 0; k < nk; ++k) {
                        const std::size_t bin = groups_[g].first + k;
                        const double w = std::sqrt((bin >= 1 && static_cast<double>(bin) < last) ? 2.0 : 1.0);
                        const cd v = spec[static_cast<std::size_t>(c) * bins + bin];
                        A(c, static_cast<Eigen::Index>(k)) = w * v.real();
                        B(c, static_cast<Eigen::Index>(k)) = w * v.imag();
                    }
                MatrixXcd value(C, C);                                                        // the segment's quantity: Re in the real part, Im (the quadrature) in the imaginary part
                value.real() = A * A.transpose() + B * B.transpose();
                if (quadrature_) value.imag() = B * A.transpose() - A * B.transpose();         // Im(X X^H)
                else value.imag().setZero();
                mean_[g].add(id, value);
            }
            ++segments_;
        },
        [&](std::int64_t id) { for (auto& m : mean_) m.skip(id); ++skipped_; });
    carryDrop(carry_, passed);
}

CrossWelchResult CrossWelch::result() const {
    CrossWelchResult r;
    r.frequencies = rfftfreq(n_, 1.0 / opt_.sfreq);
    r.segments = segments_;
    r.segmentsSkipped = skipped_;
    const Eigen::Index C = carry_.block.rows();
    r.S.assign(groups_.size(), MatrixXd::Zero(C, C));
    if (quadrature_) r.Sq.assign(groups_.size(), MatrixXd::Zero(C, C));
    if (segments_ == 0) return r;
    const double scale = segmentScale(opt_, family_);
    for (std::size_t g = 0; g < groups_.size(); ++g) {
        const double nk = groups_[g].mean ? static_cast<double>(groups_[g].last - groups_[g].first + 1) : 1.0;
        r.S[g] = mean_[g].sum().real() * (scale / (static_cast<double>(segments_) * nk));
        if (quadrature_) r.Sq[g] = mean_[g].sum().imag() * (scale / (static_cast<double>(segments_) * nk));
    }
    return r;
}

// ── Welch of a linear map of the channels ───────────────────────────────────────────────────────────────────────

LinearWelch::LinearWelch(const MatrixXd& map, std::size_t rowsPerOutput, const WelchOptions& options, std::size_t firstBin, std::size_t lastBin)
    : map_(map), group_(rowsPerOutput), first_(firstBin), last_(lastBin), opt_(options), n_(options.segmentSamples) {
    if (map.cols() < 1 || map.rows() < 1) throw std::invalid_argument("welch: the map holds at least one row and one channel");
    if (group_ < 1 || static_cast<std::size_t>(map.rows()) % group_ != 0) throw std::invalid_argument("welch: the map's " + std::to_string(map.rows()) + " rows are not a multiple of " + std::to_string(group_));
    family_ = segmentFamily(opt_);
    const std::size_t bins = rfftBins(n_);
    if (first_ > last_ || last_ >= bins) throw std::invalid_argument("welch: the bins lie outside 0 .. " + std::to_string(bins - 1));
    carry_.block.resize(map.cols(), 0);
    outputs_ = map.rows() / static_cast<Eigen::Index>(group_);
}

void LinearWelch::add(const MatrixXd& chunk) { add(chunk, std::vector<std::uint8_t>(static_cast<std::size_t>(chunk.cols()), 1)); }

void LinearWelch::add(const MatrixXd& chunk, const std::vector<std::uint8_t>& good) {
    carryAdd(carry_, chunk, good, "channels");
    samples_ += chunk.cols();
    consume();
}

void LinearWelch::consume() {
    const Eigen::Index C = carry_.block.rows();
    const double lastDoubled = (n_ % 2 == 0) ? static_cast<double>(rfftBins(n_) - 1) : static_cast<double>(rfftBins(n_));   // bins 1 .. lastDoubled-1 are doubled (one-sided)
    const Eigen::Index nk = static_cast<Eigen::Index>(last_ - first_ + 1);
    const Eigen::Index blockRows = static_cast<Eigen::Index>(group_) * std::max<Eigen::Index>(1, 4096 / static_cast<Eigen::Index>(group_));
    MatrixXd A(C, nk), B(C, nk);
    const std::size_t passed = stft(carry_.block, carry_.good, family_, carry_.first,
        [&](std::int64_t id, const cd* spec, std::size_t bins) {
            // X = A + iB over the bins kept, each scaled by sqrt(2) where the one-sided spectrum doubles it (the power is then doubled)
            for (Eigen::Index c = 0; c < C; ++c)
                for (Eigen::Index k = 0; k < nk; ++k) {
                    const std::size_t bin = first_ + static_cast<std::size_t>(k);
                    const double w = std::sqrt((bin >= 1 && static_cast<double>(bin) < lastDoubled) ? 2.0 : 1.0);
                    const cd v = spec[static_cast<std::size_t>(c) * bins + bin];
                    A(c, k) = w * v.real();
                    B(c, k) = w * v.imag();
                }
            // |M X|^2 = (M A)^2 + (M B)^2 per row and bin, a block of rows at a time, the group's rows summed: the segment's quantity
            MatrixXd segment = MatrixXd::Zero(outputs_, nk);
            for (Eigen::Index r0 = 0; r0 < map_.rows(); r0 += blockRows) {
                const Eigen::Index nr = std::min(blockRows, map_.rows() - r0);
                const MatrixXd P = (map_.middleRows(r0, nr) * A).array().square() + (map_.middleRows(r0, nr) * B).array().square();
                const Eigen::Index o0 = r0 / static_cast<Eigen::Index>(group_);
                for (Eigen::Index o = 0; o < nr / static_cast<Eigen::Index>(group_); ++o)
                    for (std::size_t g = 0; g < group_; ++g) segment.row(o0 + o) += P.row(o * static_cast<Eigen::Index>(group_) + static_cast<Eigen::Index>(g));
            }
            mean_.add_real(id, segment);
            ++segments_;
        },
        [&](std::int64_t id) { mean_.skip(id); ++skipped_; });
    carryDrop(carry_, passed);
}

LinearWelchResult LinearWelch::result() const {
    LinearWelchResult r;
    const VectorXd all = rfftfreq(n_, 1.0 / opt_.sfreq);
    r.frequencies = all.segment(static_cast<Eigen::Index>(first_), static_cast<Eigen::Index>(last_ - first_ + 1));
    r.segments = segments_;
    r.segmentsSkipped = skipped_;
    r.psd = MatrixXd::Zero(outputs_, static_cast<Eigen::Index>(last_ - first_ + 1));
    if (segments_ == 0) return r;
    const double scale = segmentScale(opt_, family_);
    r.psd = mean_.sum().real() * (scale / static_cast<double>(segments_));
    return r;
}

double enbwPerBin(const WelchOptions& o) {
  const Eigen::VectorXd w = windowOf(o.segmentSamples, o.window, o.periodic);
  return o.sfreq * w.squaredNorm() / (w.sum() * w.sum());
}

CrossWelchResult xpsd(const MatrixXd& x, const WelchOptions& options, const std::vector<BinGroup>& groups, bool withQuadrature, const std::vector<std::uint8_t>* good) {
    CrossWelch w(static_cast<int>(x.rows()), options, groups, withQuadrature);
    if (good) w.add(x, *good); else w.add(x);
    return w.result();
}

}  // namespace nxr::signal::spectral

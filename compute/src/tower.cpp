#include "nxr/tower.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::tower {

double periodS(int level) { return kDayS * std::ldexp(1.0, level); }
int levelForSeconds(double s) { return static_cast<int>(std::ceil(std::log2(s / kDayS))); }
int levelAtOrUnder(double s) {
    int l = static_cast<int>(std::floor(std::log2(s / kDayS)));
    while (periodS(l + 1) <= s) ++l;      // log2 rounding at an exact power of two
    while (periodS(l) > s) --l;
    return l;
}
int frameLevelFor(double centre_hz, int frames_per_cycle) {
    if (!(centre_hz > 0)) throw std::invalid_argument("tower: the band's centre is above 0 Hz");
    if (frames_per_cycle < 1) throw std::invalid_argument("tower: frames_per_cycle is at least 1");
    return levelAtOrUnder(1.0 / (centre_hz * frames_per_cycle));
}
std::int64_t parent(std::int64_t code, int up) { return up <= 0 ? code : (code >> up); }

std::optional<int> Placement::coarsest() const {
    if (!start_s) return levelForSeconds(static_cast<double>(n_samples) / sfreq);
    if (!day) return 0;
    return std::nullopt;
}

std::vector<std::int64_t> Placement::sampleCodes(int level, std::int64_t start, std::int64_t stop) const {
    if (const auto top = coarsest(); top && level > *top)
        throw std::invalid_argument("tower: level " + std::to_string(level) + " is above what is known (anchor " + anchor() + ", top level " + std::to_string(*top) + ")");
    if (stop < 0) stop = n_samples;
    const double offset = start_s ? *start_s * sfreq : 0.0;
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(std::max<std::int64_t>(0, stop - start)));
    for (std::int64_t i = start; i < stop; ++i) {
        const double pos = offset + static_cast<double>(i);
        if (level > 0) {
            const std::int64_t days = day.value_or(0) + static_cast<std::int64_t>(std::floor(pos / (kDayS * sfreq)));
            out.push_back(days >> level);
        } else {
            std::int64_t c = static_cast<std::int64_t>(std::floor(pos / (periodS(level) * sfreq)));
            if (anchor() == "birth") c += *day * (std::int64_t(1) << -level);
            out.push_back(c);
        }
    }
    return out;
}

double FrameTiling::frame_s() const { return periodS(level); }
double FrameTiling::tile_samples() const { return periodS(level) * sfreq; }
double FrameTiling::position(std::int64_t k) const { return static_cast<double>(code0 + k) * tile_samples() - offset_samples(); }
double FrameTiling::origin_s() const { return static_cast<double>(code0) * frame_s() - start_s; }
std::int64_t FrameTiling::frameOfSample(std::int64_t i) const {
    return static_cast<std::int64_t>(std::floor((offset_samples() + static_cast<double>(i)) / tile_samples())) - code0;
}

FrameTiling makeFrameTiling(double sfreq, std::int64_t samples, int level, std::optional<double> start_s) {
    if (!(sfreq > 0)) throw std::invalid_argument("frames: the sampling rate is above 0");
    if (samples < 1) throw std::invalid_argument("frames: the recording has no samples");
    if (level > 0) throw std::invalid_argument("frames: the frame level is 0 (the day) or finer (negative)");
    if (start_s && !(*start_s >= 0 && *start_s < kDayS)) throw std::invalid_argument("frames: the start is seconds from local midnight, in [0, 86400)");
    FrameTiling t;
    t.sfreq = sfreq; t.samples = samples; t.level = level;
    t.day_anchored = start_s.has_value();
    t.start_s = start_s.value_or(0.0);
    t.code0 = static_cast<std::int64_t>(std::floor(t.offset_samples() / t.tile_samples()));
    t.count = t.frameOfSample(samples - 1) + 1;
    return t;
}

namespace {
// the first frame whose start is at or after sample s (a tolerance for products that land a hair above a whole number)
std::int64_t firstFrameAtOrAfter(const FrameTiling& t, std::int64_t s) {
    const double x = (static_cast<double>(s) + t.offset_samples()) / t.tile_samples();
    return static_cast<std::int64_t>(std::ceil(x - 1e-9 * std::max(1.0, std::fabs(x)))) - t.code0;
}
}  // namespace

std::pair<std::int64_t, std::int64_t> frameSpan(const FrameTiling& t, std::int64_t first, std::int64_t n) {
    const std::int64_t j0 = first <= 0 ? 0 : std::clamp<std::int64_t>(firstFrameAtOrAfter(t, first), 0, t.count);
    const std::int64_t j1 = std::clamp<std::int64_t>(firstFrameAtOrAfter(t, first + n), 0, t.count);
    return {j0, std::max(j0, j1)};
}

std::vector<std::int64_t> sampleCounts(const FrameTiling& t, const std::vector<std::uint8_t>& good) {
    std::vector<std::int64_t> c(static_cast<std::size_t>(t.count), 0);
    for (std::int64_t i = 0; i < t.samples; ++i) {
        if (!good.empty() && !good[static_cast<std::size_t>(i)]) continue;
        ++c[static_cast<std::size_t>(t.frameOfSample(i))];
    }
    return c;
}

std::vector<double> rollTime(const std::vector<double>& x, std::int64_t rows, std::int64_t frames, int m, Merge op, std::int64_t code0, std::int64_t* tiles) {
    if (m < 0) throw std::invalid_argument("rollTime: the level is 0 or above");
    if (rows < 0 || frames < 0 || static_cast<std::int64_t>(x.size()) != rows * frames) throw std::invalid_argument("rollTime: x is rows x frames");
    const std::int64_t k = std::int64_t(1) << m;
    const std::int64_t front = code0 % k;
    const std::int64_t pad = (-(front + frames)) % k;
    const std::int64_t tail = pad < 0 ? pad + k : pad;
    const std::int64_t nt = (front + frames + tail) / k;
    if (tiles) *tiles = nt;
    const double inf = std::numeric_limits<double>::infinity();
    const double neutral = op == Merge::Sum ? 0.0 : (op == Merge::Max ? -inf : inf);
    std::vector<double> y(static_cast<std::size_t>(rows * nt), neutral);
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t f = 0; f < frames; ++f) {
            double& a = y[static_cast<std::size_t>(r * nt + (front + f) / k)];
            const double v = x[static_cast<std::size_t>(r * frames + f)];
            if (op == Merge::Sum) a += v;
            else if (op == Merge::Max) a = (std::isnan(a) || std::isnan(v)) ? std::numeric_limits<double>::quiet_NaN() : std::max(a, v);
            else a = (std::isnan(a) || std::isnan(v)) ? std::numeric_limits<double>::quiet_NaN() : std::min(a, v);
        }
    return y;
}

}  // namespace nxr::tower

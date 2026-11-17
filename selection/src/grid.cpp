// nxr::selection — the grid on a sampled line (grid.h).
#include "nxr/selection/grid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::selection {

namespace {
// floor division for any signs (codes are negative before the origin)
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  const std::int64_t q = a / b, r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return -floor_div(-a, b); }
void need_width(std::int64_t w) { if (w < 1) throw std::invalid_argument("grid: a tile holds at least one sample (width " + std::to_string(w) + ")"); }
}  // namespace

int LineGrid::level() const {
  if (width < 1 || (width & (width - 1)) != 0) return -1;
  int l = 0;
  while ((std::int64_t(1) << l) < width) ++l;
  return l;
}
std::int64_t LineGrid::first() const { return samples > 0 ? floor_div(0 - origin, width) : 0; }
std::int64_t LineGrid::end() const { return samples > 0 ? floor_div(samples - 1 - origin, width) + 1 : 0; }

LineGrid dyadic(int level, std::int64_t samples, std::int64_t origin) {
  if (level < 0 || level > 62) throw std::invalid_argument("grid: a dyadic level is 0 to 62 (got " + std::to_string(level) + ")");
  return LineGrid{std::int64_t(1) << level, origin, std::max<std::int64_t>(0, samples)};
}
LineGrid regular(std::int64_t width, std::int64_t samples, std::int64_t origin) {
  need_width(width);
  return LineGrid{width, origin, std::max<std::int64_t>(0, samples)};
}

int level_of(double seconds, double sfreq) {
  if (!(seconds > 0) || !(sfreq > 0)) throw std::invalid_argument("grid: a duration and a rate are positive");
  return std::max(0, static_cast<int>(std::lround(std::log2(seconds * sfreq))));
}
int level_at_or_under(double seconds, double sfreq) {
  if (!(seconds > 0) || !(sfreq > 0)) throw std::invalid_argument("grid: a duration and a rate are positive");
  return std::max(0, static_cast<int>(std::floor(std::log2(seconds * sfreq) + 1e-12)));
}
int level_at_or_over(double seconds, double sfreq) {
  if (!(seconds > 0) || !(sfreq > 0)) throw std::invalid_argument("grid: a duration and a rate are positive");
  return std::max(0, static_cast<int>(std::ceil(std::log2(seconds * sfreq) - 1e-12)));
}

Tile tile(const LineGrid& g, std::int64_t code) {
  need_width(g.width);
  if (code < g.first() || code >= g.end()) throw std::out_of_range("grid: member " + std::to_string(code) + " does not touch the line (members " + std::to_string(g.first()) + " to " + std::to_string(g.end() - 1) + ")");
  const std::int64_t a = g.origin + code * g.width, b = a + g.width;
  Tile t{code, std::max<std::int64_t>(0, a), std::min(g.samples, b), true};
  t.whole = t.start == a && t.stop == b;
  return t;
}
std::int64_t tile_of(const LineGrid& g, std::int64_t sample) { need_width(g.width); return floor_div(sample - g.origin, g.width); }

std::pair<std::int64_t, std::int64_t> touching(const LineGrid& g, std::int64_t start, std::int64_t stop) {
  need_width(g.width);
  start = std::max<std::int64_t>(0, start); stop = std::min(g.samples, stop);
  if (stop <= start) return {0, 0};
  return {tile_of(g, start), tile_of(g, stop - 1) + 1};
}
std::pair<std::int64_t, std::int64_t> within(const LineGrid& g, std::int64_t start, std::int64_t stop) {
  need_width(g.width);
  start = std::max<std::int64_t>(0, start); stop = std::min(g.samples, stop);
  if (stop <= start) return {0, 0};
  const std::int64_t a = ceil_div(start - g.origin, g.width), b = floor_div(stop - g.origin, g.width);
  return b > a ? std::make_pair(a, b) : std::make_pair(std::int64_t(0), std::int64_t(0));
}
std::pair<std::int64_t, std::int64_t> snap_out(const LineGrid& g, std::int64_t start, std::int64_t stop) {
  const auto [a, b] = touching(g, start, stop);
  if (b <= a) return {0, 0};
  return {std::max<std::int64_t>(0, g.origin + a * g.width), std::min(g.samples, g.origin + b * g.width)};
}
std::pair<std::int64_t, std::int64_t> snap_in(const LineGrid& g, std::int64_t start, std::int64_t stop) {
  const auto [a, b] = within(g, start, stop);
  if (b <= a) return {0, 0};
  return {g.origin + a * g.width, g.origin + b * g.width};
}

LineGrid coarser(const LineGrid& g, int m) {
  const int l = g.level();
  if (l < 0) throw std::invalid_argument("grid: only a dyadic grid nests (width " + std::to_string(g.width) + ")");
  if (m < 0 || l + m > 62) throw std::invalid_argument("grid: " + std::to_string(m) + " levels coarser than level " + std::to_string(l) + " is past level 62");
  return LineGrid{g.width << m, g.origin, g.samples};
}
std::int64_t parent(std::int64_t code, int m) {
  if (m < 0 || m > 62) throw std::invalid_argument("grid: a parent is 0 to 62 levels up");
  return floor_div(code, std::int64_t(1) << m);
}
std::pair<std::int64_t, std::int64_t> children(std::int64_t code, int m) {
  if (m < 0 || m > 62) throw std::invalid_argument("grid: children are 0 to 62 levels down");
  return {code * (std::int64_t(1) << m), (code + 1) * (std::int64_t(1) << m)};
}

std::vector<std::int64_t> codes(const LineGrid& g, std::int64_t start, std::int64_t stop) {
  need_width(g.width);
  start = std::max<std::int64_t>(0, start); stop = std::min(g.samples, stop);
  std::vector<std::int64_t> out;
  if (stop <= start) return out;
  out.reserve(static_cast<std::size_t>(stop - start));
  for (std::int64_t s = start; s < stop; ++s) out.push_back(tile_of(g, s));
  return out;
}
std::vector<std::int64_t> counts(const LineGrid& g, const std::vector<std::uint8_t>& good) {
  need_width(g.width);
  if (!good.empty() && static_cast<std::int64_t>(good.size()) != g.samples) throw std::invalid_argument("grid: one good flag a sample of the line");
  std::vector<std::int64_t> out(static_cast<std::size_t>(g.count()), 0);
  for (std::int64_t k = g.first(); k < g.end(); ++k) {
    const Tile t = tile(g, k);
    if (good.empty()) { out[static_cast<std::size_t>(k - g.first())] = t.stop - t.start; continue; }
    std::int64_t n = 0;
    for (std::int64_t s = t.start; s < t.stop; ++s) n += good[static_cast<std::size_t>(s)] != 0;
    out[static_cast<std::size_t>(k - g.first())] = n;
  }
  return out;
}

Closed seconds(const LineGrid& g, double sfreq, double t0) {
  if (!(sfreq > 0)) throw std::invalid_argument("grid: a rate is positive");
  need_width(g.width);
  const double w = static_cast<double>(g.width) / sfreq;
  return Closed{t0 + static_cast<double>(g.origin + g.first() * g.width) / sfreq, w, w, g.count(), t0 + static_cast<double>(g.samples) / sfreq};
}
LineGrid from_seconds(const Closed& c, double sfreq, double t0, std::int64_t samples, double tolerance) {
  if (!(sfreq > 0)) throw std::invalid_argument("grid: a rate is positive");
  const double w = c.width * sfreq, h = c.hop * sfreq, o = (c.origin - t0) * sfreq;
  const double wr = std::round(w), orr = std::round(o);
  if (std::abs(w - wr) > tolerance || std::abs(o - orr) > tolerance) throw std::invalid_argument("grid: the stored tiling's width and origin are not whole samples at " + std::to_string(sfreq) + " Hz");
  if (std::abs(h - w) > tolerance) throw std::invalid_argument("grid: the stored tiling is not contiguous (hop != width): a Welch segmentation, not a partition");
  return regular(static_cast<std::int64_t>(wr), samples, static_cast<std::int64_t>(orr));
}

}  // namespace nxr::selection

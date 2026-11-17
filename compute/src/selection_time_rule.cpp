// nxr::selection (compute) — THE TIME RULE (partition.h `TimeRule`): one rule family for every time grid.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "nxr/selection/grid.h"
#include "nxr/selection/partition.h"
#include "nxr/welch.h"

namespace nxr::selection {

namespace {
std::int64_t pow2(int k) { return std::int64_t(1) << k; }

// the sample a point or a span of the line is at (a span: its start rounded outward, axis 0)
std::int64_t origin_sample(const Selection& o, const Line& line) {
  if (o.type == Type::point && o.center) return *o.center;
  if (o.type == Type::span && !o.extents.empty()) return static_cast<std::int64_t>(std::floor((o.extents.front().start - line.origin) / line.spacing + 1e-9));
  throw std::invalid_argument("time_rule: the origin is a point of the line (the sample it is at) or a span (an event: its start)");
}

int log2_exact(std::int64_t x, const char* what) {
  if (x < 1 || (x & (x - 1)) != 0) throw std::invalid_argument(std::string("time_rule: ") + what + " is a power of two (got " + std::to_string(x) + ")");
  int k = 0;
  while ((std::int64_t(1) << k) < x) ++k;
  return k;
}
}  // namespace

TimeGrid time_rule_resolved(const Line& line, const TimeRule& r) {
  if (line.count < 1 || !(line.spacing > 0)) throw std::invalid_argument("time_rule: a line of count >= 1 points, spacing > 0 apart");
  const double fs = 1.0 / line.spacing;
  const std::int64_t o = r.origin ? origin_sample(*r.origin, line) : 0;
  if (o < 0 || o >= line.count) throw std::invalid_argument("time_rule: the origin (sample " + std::to_string(o) + ") lies outside the line of " + std::to_string(line.count) + " samples");
  TimeGrid out;
  out.origin_sample = o;
  std::int64_t width = 0, hop = 0;
  double width_s = 0;       // exact, when the width is not whole samples
  bool whole_only = false;  // only whole members (segments, windows)
  switch (r.kind) {
    case TimeRule::Kind::tiles: {
      if (!r.duration_s) throw std::invalid_argument("time_rule: kind tiles needs duration_s");
      const LevelOf l = level_of_duration(*r.duration_s, fs, std::nullopt);
      out.level = l.level; width = hop = pow2(l.level);
      break;
    }
    case TimeRule::Kind::band: {
      if (!r.band_hz) throw std::invalid_argument("time_rule: kind band needs band_hz");
      out.level = time_level_of_band(r.band_hz->first, r.band_hz->second, fs, r.cycles);
      width = hop = pow2(out.level);
      break;
    }
    case TimeRule::Kind::chunks: out.level = 15; width = hop = pow2(15); break;
    case TimeRule::Kind::frames: {
      if (!r.band_hz) throw std::invalid_argument("time_rule: kind frames needs band_hz");
      if (r.frames_per_tile.has_value() == r.frames_per_cycle.has_value())
        throw std::invalid_argument("time_rule: kind frames needs exactly one of frames_per_tile (a power of two) and frames_per_cycle");
      if (r.frames_per_tile) {
        const int s = log2_exact(*r.frames_per_tile, "frames_per_tile");
        const int k = time_level_of_band(r.band_hz->first, r.band_hz->second, fs, r.cycles);
        if (s > k) throw std::invalid_argument("time_rule: " + std::to_string(*r.frames_per_tile) + " frames per tile would be finer than a sample (the band's tile is 2^" + std::to_string(k) + " samples)");
        out.level = k - s;
        width = hop = pow2(out.level);
      } else {
        if (!(*r.frames_per_cycle > 0)) throw std::invalid_argument("time_rule: frames_per_cycle > 0");
        const double centre = std::sqrt(r.band_hz->first * r.band_hz->second);
        width_s = 1.0 / (centre * *r.frames_per_cycle);
        if (!(width_s * fs >= 1)) throw std::invalid_argument("time_rule: " + std::to_string(*r.frames_per_cycle) + " frames per cycle of " + std::to_string(centre) + " Hz is finer than a sample at " + std::to_string(fs) + " Hz");
        const double w = width_s * fs;
        if (std::abs(w - std::round(w)) < 1e-9) { width = hop = static_cast<std::int64_t>(std::llround(w)); }
      }
      break;
    }
    case TimeRule::Kind::segments: {
      if (!(r.overlap >= 0 && r.overlap < 1)) throw std::invalid_argument("time_rule: the overlap is a fraction in [0, 1)");
      const welch::Segment seg = welch::segmentFor(r.segment_rule, r.duration_s, r.segment_samples, r.segment_level, fs, r.overlap);
      width = static_cast<std::int64_t>(seg.samples);
      hop = width - static_cast<std::int64_t>(std::floor(static_cast<double>(width) * r.overlap));
      out.level = seg.level > 0 ? seg.level : -1;
      whole_only = true;
      break;
    }
    case TimeRule::Kind::windows: {
      if (!r.duration_s || !r.step_s) throw std::invalid_argument("time_rule: kind windows needs duration_s and step_s");
      width = pow2(level_of_duration(*r.duration_s, fs, std::nullopt).level);
      hop = pow2(level_of_duration(*r.step_s, fs, std::nullopt).level);
      whole_only = true;
      break;
    }
  }
  const std::int64_t span = line.count - o;                           // samples from the origin to the end
  std::int64_t count = 0;
  const double dt = line.spacing;
  if (width > 0) {
    if (whole_only) count = span >= width ? (span - width) / hop + 1 : 0;
    else count = (span + hop - 1) / hop;                              // the last member may be partial
    out.width_samples = width; out.hop_samples = hop;
    out.width_s = static_cast<double>(width) * dt; out.hop_s = static_cast<double>(hop) * dt;
  } else {
    count = static_cast<std::int64_t>(std::ceil(static_cast<double>(span) / (width_s * fs) - 1e-9));
    out.width_s = out.hop_s = width_s;
  }
  if (count < 1) throw std::invalid_argument("time_rule: the line from the origin (" + std::to_string(span) + " samples) holds no member of " + std::to_string(width) + " samples");
  Selection g = tiled(Tiling{line.origin + static_cast<double>(o) * dt, out.width_s, out.hop_s, count}, line.unit);
  if (out.level >= 0 && o == 0 && r.kind != TimeRule::Kind::segments && r.kind != TimeRule::Kind::windows) { g.level = out.level; g.code = out.level; }
  out.grid = std::move(g);
  return out;
}

Selection time_rule(const Line& line, const TimeRule& rule) { return time_rule_resolved(line, rule).grid; }

}  // namespace nxr::selection

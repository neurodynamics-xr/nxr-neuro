// nxr::signal and the frame option of the time rule: the band-limited interpolation that samples a block at the frames of a time rule (selection/partition.h `TimeRule`, kind frames:
// the frame density, frames per cycle or per tile, and the origin are OPTIONS of the function; no tower, no clock anchor).
//   - the frames of a 250 Hz recording at 36 frames per cycle of 10 Hz: every frame once, from the origin to the end of the line; an origin (a point of the line) starts the first frame there
//   - frames per tile nest in the band's dyadic level; the two densities are exclusive; refusals
//   (the band's signal read at the frames: test_wt_at)
#include <cmath>
#include <cstdio>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nxr/select/grid.h"
#include "nxr/select/partition.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

int main() {
  namespace sel = nxr::selection;
  const double fs = 250.0, f0 = 10.0;
  const std::int64_t n = 5000;
  const sel::Line line{"s", n, 0.0, 1.0 / fs};
  nxr::select::TimeRule rule;
  rule.kind = nxr::select::TimeRule::Kind::frames;
  rule.band_hz = std::make_pair(f0, f0);
  rule.frames_per_cycle = 36;
  const nxr::select::TimeGrid t = nxr::select::time_rule_resolved(line, rule);
  const double tile = t.width_s * fs;                                          // the frame in samples, not whole
  const std::int64_t count = sel::count(t.grid);
  EXPECT(std::abs(t.width_s - 1.0 / (f0 * 36)) < 1e-15 && count == static_cast<std::int64_t>(std::ceil(static_cast<double>(n) / tile - 1e-9)), "36 frames a cycle of 10 Hz: the frames the samples touch");
  // an origin is an option: a point of the line (or an event's span) starts the first frame there
  nxr::select::TimeRule shifted = rule;
  shifted.origin = sel::point(500);
  const nxr::select::TimeGrid o = nxr::select::time_rule_resolved(line, shifted);
  EXPECT(o.origin_sample == 500 && sel::count(o.grid) == static_cast<std::int64_t>(std::ceil(static_cast<double>(n - 500) / tile - 1e-9)), "frames start at the origin and run to the end of the line");
  // frames per tile are a power of two nested in the band's dyadic level
  nxr::select::TimeRule nested;
  nested.kind = nxr::select::TimeRule::Kind::frames;
  nested.band_hz = std::make_pair(8.0, 12.0);
  nested.frames_per_tile = 8;
  const nxr::select::TimeGrid nd = nxr::select::time_rule_resolved(line, nested);
  const int fit = nxr::select::time_level_of_band(8.0, 12.0, fs);
  EXPECT(nd.level == fit - 3 && nd.width_samples == (std::int64_t(1) << (fit - 3)), "8 frames a tile: level " << fit << " - 3, got " << nd.level);
  auto refused = [&](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  nxr::select::TimeRule both = rule; both.frames_per_tile = 4;
  nxr::select::TimeRule odd = nested; odd.frames_per_tile = 6;
  nxr::select::TimeRule outside = rule; outside.origin = sel::point(n + 5);
  EXPECT(refused([&] { nxr::select::time_rule(line, both); }) && refused([&] { nxr::select::time_rule(line, odd); }) && refused([&] { nxr::select::time_rule(line, outside); }), "both densities, a non power of two, an origin outside the line are refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}

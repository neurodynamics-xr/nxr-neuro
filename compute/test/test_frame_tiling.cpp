// nxr::frametiling and the frame option of the time rule: the band-limited interpolation that samples a block at the frames of a time rule (selection/partition.h `TimeRule`, kind frames:
// the frame density, frames per cycle or per tile, and the origin are OPTIONS of the function; no tower, no clock anchor).
//   - the frames of a 250 Hz recording at 36 frames per cycle of 10 Hz: every frame once, from the origin to the end of the line; an origin (a point of the line) starts the first frame there
//   - frames per tile nest in the band's dyadic level; the two densities are exclusive; refusals
//   - a complex exponential in the band (10 Hz, 250 Hz sampling) interpolated at the frames equals e^{iωt} (the analytic signal of a band-limited signal): max error
#include <cmath>
#include <cstdio>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nxr/signal/resample.h"
#include "nxr/select/grid.h"
#include "nxr/select/partition.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

int main() {
  using namespace nxr::signal;
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
  // a band-limited complex exponential at 10 Hz (a band 8-12) sampled at the frames
  const double phi = 0.4;
  std::vector<float> re(n), im(n);
  for (std::int64_t i = 0; i < n; ++i) { re[i] = static_cast<float>(std::cos(2 * M_PI * f0 * i / fs + phi)); im[i] = static_cast<float>(std::sin(2 * M_PI * f0 * i / fs + phi)); }
  // a 600-sample block in the middle: the frames of its own 400 samples need a halo of the half width
  const int H = 32;
  const std::int64_t s0 = 2000, len = 400;
  const std::int64_t j0 = static_cast<std::int64_t>(std::ceil(static_cast<double>(s0) / tile - 1e-9)), j1 = static_cast<std::int64_t>(std::ceil(static_cast<double>(s0 + len) / tile - 1e-9));   // the frames that START in the block
  std::vector<double> pos;
  for (std::int64_t j = j0; j < j1; ++j) pos.push_back(static_cast<double>(j) * tile - static_cast<double>(s0 - H));
  const auto yr = sincInterpolate(re.data() + (s0 - H), 1, len + 2 * H, pos, H), yi = sincInterpolate(im.data() + (s0 - H), 1, len + 2 * H, pos, H);
  double worst = 0;
  for (std::int64_t j = j0; j < j1; ++j) {
    const double tt = static_cast<double>(j) * tile / fs;
    worst = std::max(worst, std::abs(std::complex<double>(yr[j - j0], yi[j - j0]) - std::polar(1.0, 2 * M_PI * f0 * tt + phi)));
  }
  std::printf("  %lld frames of %lld samples at %.0f Hz: max |interpolated - exact| %.2e (half width %d)\n", static_cast<long long>(j1 - j0), static_cast<long long>(len), 1.0 / t.width_s, worst, H);
  EXPECT(std::llabs((j1 - j0) - static_cast<std::int64_t>(std::llround(len / tile))) <= 1 && worst < 2e-4, "the interpolation is off");
  EXPECT(refused([&] { sincInterpolate(re.data(), 1, 10, {1.0}, 0); }), "a half width of 0 is refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}

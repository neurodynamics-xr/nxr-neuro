// nxr::frametiling: the band-limited interpolation that samples a block at the TOWER's frames (tower.h; the tiling itself is tested in test_tower).
//   - the tower frames of a 250 Hz recording at level -25 (10 Hz x 36): every frame once, whatever the chunk size
//   - a complex exponential in the band (10 Hz, 250 Hz sampling) interpolated at the frames equals e^{iωt} (the analytic signal of a band-limited signal): max error
//   - refusals
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

#include "nxr/frame_tiling.h"
#include "nxr/tower.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

int main() {
  using namespace nxr::frametiling;
  namespace tw = nxr::tower;
  const double fs = 250.0, f0 = 10.0;
  const std::int64_t n = 5000;
  const int L = tw::frameLevelFor(f0, 36);
  const tw::FrameTiling t = tw::makeFrameTiling(fs, n, L);
  EXPECT(L == -25 && t.count == static_cast<std::int64_t>(std::floor((n - 1) / t.tile_samples())) + 1, "level -25: the tiles the samples touch");
  EXPECT(std::fabs(t.position(36) - 36 * t.tile_samples()) < 1e-9, "frame 36 starts 36 tiles on");
  // every frame once, whatever the chunk size
  for (std::int64_t chunk : {std::int64_t(1), std::int64_t(7), std::int64_t(1000), std::int64_t(4096), n}) {
    std::int64_t seen = 0, next = 0;
    bool ordered = true;
    for (std::int64_t s = 0; s < n; s += chunk) {
      const auto [a, b] = tw::frameSpan(t, s, std::min(chunk, n - s));
      ordered = ordered && a == next;
      next = b; seen += b - a;
      for (std::int64_t j = a; j < b; ++j) ordered = ordered && t.position(j) >= s - 1e-6 && t.position(j) < s + std::min(chunk, n - s) + 1e-6;
    }
    EXPECT(seen == t.count && ordered, "chunk " << chunk << ": the frames are not a partition (" << seen << " of " << t.count << ")");
  }
  // a band-limited complex exponential at 10 Hz (a band 8-12) sampled at the frames
  const double phi = 0.4;
  std::vector<float> re(n), im(n);
  for (std::int64_t i = 0; i < n; ++i) { re[i] = static_cast<float>(std::cos(2 * M_PI * f0 * i / fs + phi)); im[i] = static_cast<float>(std::sin(2 * M_PI * f0 * i / fs + phi)); }
  // a 600-sample block in the middle: the frames of its own 400 samples need a halo of the half width
  const int H = 32;
  const std::int64_t s0 = 2000, len = 400;
  const auto [j0, j1] = tw::frameSpan(t, s0, len);
  std::vector<double> pos;
  for (std::int64_t j = j0; j < j1; ++j) pos.push_back(t.position(j) - static_cast<double>(s0 - H));
  const auto yr = sincInterpolate(re.data() + (s0 - H), 1, len + 2 * H, pos, H), yi = sincInterpolate(im.data() + (s0 - H), 1, len + 2 * H, pos, H);
  double worst = 0;
  for (std::int64_t j = j0; j < j1; ++j) {
    const double tt = t.position(j) / fs;
    worst = std::max(worst, std::abs(std::complex<double>(yr[j - j0], yi[j - j0]) - std::polar(1.0, 2 * M_PI * f0 * tt + phi)));
  }
  std::printf("  %lld frames of %lld samples at %.0f Hz: max |interpolated - exact| %.2e (half width %d)\n", static_cast<long long>(j1 - j0), static_cast<long long>(len), t.fps(), worst, H);
  EXPECT(std::llabs((j1 - j0) - static_cast<std::int64_t>(std::llround(len / t.tile_samples()))) <= 1 && worst < 2e-4, "the interpolation is off");
  auto refused = [&](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([] { tw::makeFrameTiling(0, 100, -5); }) && refused([] { tw::frameLevelFor(10, 0); }) && refused([&] { sincInterpolate(re.data(), 1, 10, {1.0}, 0); }), "refusals");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}

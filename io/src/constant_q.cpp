// nxr::io::constantq — the tile grid of a time Line (constant_q.h): the Studio's backend/src/db/constant-q.ts (JavaScript's Math.round rounds halves up).
// The design of a tiling is compute's; io keeps only the stored form's arithmetic.
#include "nxr/io/constant_q.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nxr/io/errors.h"

namespace nxr::io::constantq {

namespace {
std::string i2s(std::int64_t x) { return std::to_string(x); }
}  // namespace

int levelOf(double seconds, double sfreq) {
  if (!(seconds > 0) || !(sfreq > 0)) throw Error(code::invalid_argument, "levelOf: " + std::to_string(seconds) + " s at " + std::to_string(sfreq) + " Hz");
  return std::max(0, static_cast<int>(std::lround(std::log2(seconds * sfreq))));
}

int levelOfWidth(std::int64_t samples) {
  if (samples < 1 || (samples & (samples - 1)) != 0) return -1;
  int k = 0;
  while ((std::int64_t(1) << k) < samples) ++k;
  return k;
}

std::pair<double, double> tileOf(const Closed& t, std::int64_t m, double end) {
  const double start = t.origin + static_cast<double>(m) * t.hop;
  return {start, std::min(start + t.width, end)};
}

std::optional<std::int64_t> tileIndexAt(const Closed& t, double value, double end) {
  if (value < t.origin || value >= end) return std::nullopt;
  const std::int64_t m = static_cast<std::int64_t>(std::floor((value - t.origin) / t.hop));
  return m < t.count ? std::optional<std::int64_t>(m) : std::nullopt;
}

std::int64_t tilesWithin(std::int64_t coarser, std::int64_t finer) {
  if (!(finer > 0) || coarser % finer != 0)
    throw Error(code::invalid_argument, "tilesWithin: " + i2s(coarser) + " is not a whole multiple of " + i2s(finer));
  return coarser / finer;
}

}  // namespace nxr::io::constantq

// nxr::io::constantq — the tile grid of a time Line (constant_q.h): the Studio's backend/src/db/constant-q.ts (JavaScript's Math.round rounds halves up).
// The design of a tiling is compute's; io keeps only the stored form's arithmetic.
#include "nxr/io/constant_q.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nxr/io/errors.h"

namespace nxr::io::constantq {

namespace {
double jsRound(double x) { return std::floor(x + 0.5); }                 // Math.round: halves toward +∞
double pow2ceil(double x) { return std::pow(2.0, std::ceil(std::log2(x))); }
std::string i2s(std::int64_t x) { return std::to_string(x); }
}  // namespace

std::pair<double, double> anchorsFor(Method method, double sfreq, double target) {
  const double nyquist = sfreq / 2;
  if (method == Method::Cycles) return {target, nyquist};
  return {std::pow(2.0, jsRound(std::log2(target * sfreq))) / sfreq, std::min(std::pow(2.0, std::floor(std::log2(nyquist))), kCapHz)};
}

std::int64_t windowFor(double sfreq, double lo, std::optional<double> cycles) {
  if (!(sfreq > 0) || !(lo > 0)) throw Error(code::invalid_argument, "windowFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(lo) + " Hz");
  const double c = cycles ? *cycles : anchorsFor(Method::Hertz, sfreq).first;
  if (!(c > 0)) throw Error(code::invalid_argument, "windowFor: cycles must be > 0");
  return static_cast<std::int64_t>(std::llround(pow2ceil(c * sfreq / lo)));
}

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

std::int64_t strideFor(double sfreq, double hz, int min) {
  if (!(sfreq > 0) || !(hz > 0)) throw Error(code::invalid_argument, "strideFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(hz) + " Hz");
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::pow(2.0, std::floor(std::log2(sfreq / (min * hz))))));
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

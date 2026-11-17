// nxr::selection::constantq — the band-design arithmetic of a constant-Q ladder (constant_q.h): the Studio's backend/src/db/constant-q.ts (JavaScript's Math.round
// rounds halves up).
#include "nxr/selection/constant_q.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::selection::constantq {

namespace {
double js_round(double x) { return std::floor(x + 0.5); }                 // Math.round: halves toward +infinity
double pow2ceil(double x) { return std::pow(2.0, std::ceil(std::log2(x))); }
}  // namespace

std::pair<double, double> anchorsFor(Method method, double sfreq, double target) {
  const double nyquist = sfreq / 2;
  if (method == Method::Cycles) return {target, nyquist};
  return {std::pow(2.0, js_round(std::log2(target * sfreq))) / sfreq, std::min(std::pow(2.0, std::floor(std::log2(nyquist))), kCapHz)};
}

std::int64_t windowFor(double sfreq, double lo, std::optional<double> cycles) {
  if (!(sfreq > 0) || !(lo > 0)) throw std::invalid_argument("windowFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(lo) + " Hz");
  const double c = cycles ? *cycles : anchorsFor(Method::Hertz, sfreq).first;
  if (!(c > 0)) throw std::invalid_argument("windowFor: cycles must be > 0");
  return static_cast<std::int64_t>(std::llround(pow2ceil(c * sfreq / lo)));
}

std::int64_t strideFor(double sfreq, double hz, int min) {
  if (!(sfreq > 0) || !(hz > 0)) throw std::invalid_argument("strideFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(hz) + " Hz");
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::pow(2.0, std::floor(std::log2(sfreq / (min * hz))))));
}

}  // namespace nxr::selection::constantq

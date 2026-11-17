// nxr::selection — the time partition of a recording (time_partition.h): the band design of the Studio's backend/src/db/constant-q.ts (constantQTiling,
// anchorsFor), its levels named by k.
#include "nxr/selection/time_partition.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace nxr::selection {

namespace {
double js_round(double x) { return std::floor(x + 0.5); }            // the Studio's Math.round: halves toward +infinity
int k_of(double samples) { return static_cast<int>(std::lround(std::ceil(std::log2(samples)))); }   // the power of two a window is rounded up to
}  // namespace

TimePartition time_partition(std::int64_t samples, double sfreq, const ConstantQ& o) {
  if (samples < 1 || !(sfreq > 0)) throw std::invalid_argument("time partition: a recording of " + std::to_string(samples) + " samples at " + std::to_string(sfreq) + " Hz");
  if (o.method != "hertz" && o.method != "cycles") throw std::invalid_argument("time partition: constant-Q's method is hertz or cycles (got '" + o.method + "')");
  if (o.voices < 1) throw std::invalid_argument("time partition: voices is a whole number >= 1");
  const double nyquist = sfreq / 2;
  // the method's anchors (anchorsFor): hertz takes the cycles of a power-of-two window at 8 samples a cycle and caps f_max at 64 Hz; cycles takes 8 cycles to Nyquist
  const bool hertz = o.method == "hertz";
  const double anchor_cycles = hertz ? std::pow(2.0, js_round(std::log2(8.0 * sfreq))) / sfreq : 8.0;
  const double anchor_fmax = hertz ? std::min(std::pow(2.0, std::floor(std::log2(nyquist))), 64.0) : nyquist;
  const double cycles = o.cycles.value_or(anchor_cycles);
  if (!(cycles > 0)) throw std::invalid_argument("time partition: cycles is > 0");
  const double f_max = std::min(o.f_max.value_or(anchor_fmax), nyquist);
  if (!(f_max > 0)) throw std::invalid_argument("time partition: f_max is > 0");
  const double f_min = std::max(o.f_min.value_or(hertz ? 0.5 : 0.0), 0.0);
  TimePartition p;
  p.samples = samples;
  p.sfreq = sfreq;
  // the bands, down from f_max by 1/voices of an octave, while their window fits the recording
  for (int b = 0;; ++b) {
    const double hi = f_max * std::pow(2.0, -static_cast<double>(b) / o.voices), lo = f_max * std::pow(2.0, -static_cast<double>(b + 1) / o.voices);
    if (lo < f_min) break;
    const int k = k_of(cycles * sfreq / lo);
    if ((std::int64_t(1) << k) > samples) break;
    p.bands.push_back(TimeBand{lo, hi, k});
  }
  // the levels: one a distinct k, finest first, resolving the union of its bands
  std::map<int, TimeLevel> by_k;
  for (const TimeBand& band : p.bands) {
    auto [it, fresh] = by_k.try_emplace(band.k, TimeLevel{band.k, (samples + (std::int64_t(1) << band.k) - 1) >> band.k, band.lo_hz, band.hi_hz});
    if (!fresh) { it->second.lo_hz = std::min(it->second.lo_hz, band.lo_hz); it->second.hi_hz = std::max(it->second.hi_hz, band.hi_hz); }
  }
  for (const auto& [k, level] : by_k) p.levels.push_back(level);
  return p;
}

}  // namespace nxr::selection

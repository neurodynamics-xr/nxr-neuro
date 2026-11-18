// nxr::select — temporal events (intervals.h).
#include "nxr/select/events.h"

#include "nxr/stats/summary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::select {

std::vector<Span> crossings(const Eigen::Ref<const Eigen::VectorXd>& x, double entry, double exit) {
  if (!(exit <= entry)) throw std::invalid_argument("crossings: the exit level is at most the entry level");
  std::vector<Span> out;
  const Eigen::Index n = x.size();
  Eigen::Index i = 0;
  while (i < n) {
    if (!(x(i) >= exit)) { ++i; continue; }
    Span s; s.start = i; s.value = -std::numeric_limits<double>::infinity();
    bool entered = false;
    while (i < n && x(i) >= exit) {
      if (x(i) >= entry) entered = true;
      if (x(i) > s.value) { s.value = x(i); s.peak = i; }
      ++i;
    }
    s.stop = i;
    if (entered) out.push_back(s);
  }
  return out;
}

std::vector<Span> mergeSpans(const std::vector<Span>& spans, std::int64_t gap) {
  if (gap < 0) throw std::invalid_argument("mergeSpans: the gap is at least 0");
  std::vector<Span> out;
  for (const Span& s : spans) {
    if (!out.empty() && s.start < out.back().stop) throw std::invalid_argument("mergeSpans: the spans are in order and do not overlap");
    if (!out.empty() && s.start - out.back().stop < gap) {
      Span& m = out.back();
      m.stop = s.stop;
      if (s.value > m.value) { m.value = s.value; m.peak = s.peak; }
    } else {
      out.push_back(s);
    }
  }
  return out;
}

std::vector<Span> keepSpans(const std::vector<Span>& spans, std::int64_t minLength, std::int64_t maxLength) {
  std::vector<Span> out;
  for (const Span& s : spans) if (s.length() >= minLength && s.length() <= maxLength) out.push_back(s);
  return out;
}

std::vector<Span> halfWaves(const Eigen::Ref<const Eigen::VectorXd>& x, std::int64_t minLength, std::int64_t maxLength) {
  std::vector<Span> out;
  const Eigen::Index n = x.size();
  Eigen::Index i = 1;
  while (i < n) {
    if (!(x(i - 1) >= 0 && x(i) < 0)) { ++i; continue; }    // a downward zero crossing: the half-wave starts at i
    Span s; s.start = i; s.value = std::numeric_limits<double>::infinity();
    Eigen::Index j = i;
    while (j < n && x(j) < 0) { if (x(j) < s.value) { s.value = x(j); s.peak = j; } ++j; }
    if (j == n) break;                                        // no upward crossing: the record ends inside the half-wave
    s.stop = j;
    if (s.length() >= minLength && s.length() <= maxLength) out.push_back(s);
    i = j;
  }
  return out;
}

double median(const Eigen::Ref<const Eigen::VectorXd>& x) {
  if (x.size() == 0) throw std::invalid_argument("median: an empty series");
  return nxr::stats::median(std::vector<double>(x.data(), x.data() + x.size()));      // the one copy (stats/summary.h)
}

double mad(const Eigen::Ref<const Eigen::VectorXd>& x) {
  if (x.size() == 0) throw std::invalid_argument("median: an empty series");
  return nxr::stats::mad(std::vector<double>(x.data(), x.data() + x.size()));
}

std::vector<double> eventDurations(const std::vector<TileEvent>& events, double width) {
  std::vector<double> d;
  for (const auto& e : events) d.push_back(static_cast<double>(e.stop - e.first) * width);
  return d;
}

std::vector<double> eventIntervalNext(const std::vector<TileEvent>& events, double width) {
  std::vector<double> out(events.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < events.size(); ++i) {
    std::int64_t best = -1;
    for (std::size_t j = 0; j < events.size(); ++j) {
      if (j == i || events[j].first <= events[i].first) continue;
      if (events[i].row >= 0 && events[j].row != events[i].row) continue;
      if (best < 0 || events[j].first < best) best = events[j].first;
    }
    if (best >= 0) out[i] = static_cast<double>(best - events[i].stop) * width;
  }
  return out;
}

BurstLengths burstLengths(const BurstOptions& o, double step) {
  if (!(step > 0) || !std::isfinite(step)) throw std::invalid_argument("burstLengths: step (s between samples) is above 0");
  // seconds in whole samples: joined when the samples between two spans last <= mergeS; kept when minS <= length * step <= maxS
  BurstLengths L;
  L.gap = static_cast<std::int64_t>(std::floor(o.mergeS / step + 1e-9)) + 1;
  L.minLength = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(o.minS / step - 1e-9)));
  L.maxLength = std::isinf(o.maxS) ? std::numeric_limits<std::int64_t>::max() : static_cast<std::int64_t>(std::floor(o.maxS / step + 1e-9));
  return L;
}

Bursts detectBursts(const Eigen::Ref<const Eigen::VectorXd>& x, double step, const BurstOptions& o, const std::optional<Baseline>& baseline) {
  if (!(step > 0) || !std::isfinite(step)) throw std::invalid_argument("detectBursts: step (s between samples) is above 0");
  if (!(o.mergeS >= 0) || !(o.minS >= 0) || !(o.maxS >= o.minS)) throw std::invalid_argument("detectBursts: 0 <= mergeS, 0 <= minS <= maxS");
  Bursts b;
  const BurstLengths L = burstLengths(o, step);
  b.gap = L.gap; b.minLength = L.minLength; b.maxLength = L.maxLength;
  std::vector<double> good;
  for (Eigen::Index t = 0; t < x.size(); ++t) if (std::isfinite(x(t))) good.push_back(x(t));
  b.good = static_cast<std::int64_t>(good.size());
  if (good.size() < 2) { b.skipped = "fewer than 2 good samples"; return b; }
  const Eigen::Map<const Eigen::VectorXd> g(good.data(), static_cast<Eigen::Index>(good.size()));
  b.median = baseline ? baseline->median : median(g);
  b.sd = baseline ? baseline->sd : mad(g);
  if (!(b.sd > 0)) { b.skipped = "no spread (robust SD 0)"; return b; }
  b.enter = b.median + o.enterSd * b.sd;
  b.exit = b.median + o.exitSd * b.sd;
  Eigen::VectorXd z = x;
  for (Eigen::Index t = 0; t < z.size(); ++t) if (!std::isfinite(z(t))) z(t) = -std::numeric_limits<double>::infinity();   // a bad sample ends a span
  b.spans = keepSpans(mergeSpans(crossings(z, b.enter, b.exit), b.gap), b.minLength, b.maxLength);
  return b;
}

}  // namespace nxr::select

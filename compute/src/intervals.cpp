// nxr::intervals — temporal events (intervals.h).
#include "nxr/intervals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::intervals {

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
  std::vector<double> v(x.data(), x.data() + x.size());
  const std::size_t h = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h), v.end());
  if (v.size() % 2) return v[h];
  const double hi = v[h];
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h)));
}

double mad(const Eigen::Ref<const Eigen::VectorXd>& x) {
  const double m = median(x);
  return 1.4826 * median((x.array() - m).abs().matrix());
}

EventValues rollupEvents(const Eigen::MatrixXd& tiles, const Eigen::VectorXd& counts, const std::vector<TileEvent>& events, const std::string& reduction) {
  if (reduction != "mean" && reduction != "max") throw std::invalid_argument("rollupEvents: the reduction is mean or max, not '" + reduction + "'");
  if (counts.size() != tiles.cols()) throw std::invalid_argument("rollupEvents: one count per tile");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EventValues out;
  out.value = Eigen::MatrixXd::Constant(tiles.rows(), static_cast<Eigen::Index>(events.size()), nan);
  out.count = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(events.size()));
  for (std::size_t e = 0; e < events.size(); ++e) {
    const TileEvent& ev = events[e];
    if (ev.first < 0 || ev.stop > tiles.cols() || ev.stop <= ev.first || ev.row >= tiles.rows()) throw std::invalid_argument("rollupEvents: event " + std::to_string(e) + " lies outside the tiles");
    for (std::int64_t t = ev.first; t < ev.stop; ++t) out.count(static_cast<Eigen::Index>(e)) += counts(t);
    for (Eigen::Index r = 0; r < tiles.rows(); ++r) {
      if (ev.row >= 0 && r != ev.row) continue;
      double acc = 0, w = 0, mx = -std::numeric_limits<double>::infinity();
      bool ok = true;
      for (std::int64_t t = ev.first; t < ev.stop && ok; ++t) {
        const double v = tiles(r, t);
        if (!std::isfinite(v)) { ok = false; break; }
        acc += v * counts(t); w += counts(t); mx = std::max(mx, v);
      }
      if (!ok) continue;
      out.value(r, static_cast<Eigen::Index>(e)) = reduction == "max" ? mx : (w > 0 ? acc / w : nan);
    }
  }
  return out;
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

}  // namespace nxr::intervals

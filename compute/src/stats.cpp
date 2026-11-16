// nxr::stats — see include/nxr/stats.h.
#include "nxr/stats.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::stats {

namespace {
void need(const std::vector<double>& x, const char* who) { if (x.empty()) throw std::invalid_argument(std::string("stats: ") + who + " of an empty sample"); }
double moment(const std::vector<double>& x, int k, double m) { double s = 0; for (double v : x) s += std::pow(v - m, k); return s / static_cast<double>(x.size()); }
}  // namespace

double mean(const std::vector<double>& x) { need(x, "mean"); double s = 0; for (double v : x) s += v; return s / static_cast<double>(x.size()); }

double sd(const std::vector<double>& x, int ddof) {
  need(x, "sd");
  if (ddof < 0 || static_cast<std::size_t>(ddof) >= x.size()) throw std::invalid_argument("stats: sd needs more values than ddof");
  const double m = mean(x);
  double s = 0;
  for (double v : x) s += (v - m) * (v - m);
  return std::sqrt(s / static_cast<double>(x.size() - static_cast<std::size_t>(ddof)));
}

double median(std::vector<double> x) {
  need(x, "median");
  const std::size_t h = x.size() / 2;
  std::nth_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(h), x.end());
  if (x.size() % 2) return x[h];
  const double hi = x[h];
  return 0.5 * (hi + *std::max_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(h)));
}

double mad(const std::vector<double>& x) {
  const double m = median(x);
  std::vector<double> d(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) d[i] = std::abs(x[i] - m);
  return 1.4826 * median(std::move(d));
}

double percentile(std::vector<double> x, double p) {
  need(x, "percentile");
  if (!(p >= 0 && p <= 100)) throw std::invalid_argument("stats: a percentile is in [0, 100]");
  std::sort(x.begin(), x.end());
  const double r = p / 100.0 * static_cast<double>(x.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(r)), hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (r - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

double peak(const std::vector<double>& x) { need(x, "peak"); double m = 0; for (double v : x) m = std::max(m, std::abs(v)); return m; }

double kurtosis(const std::vector<double>& x) {
  need(x, "kurtosis");
  const double m = mean(x), m2 = moment(x, 2, m);
  return m2 > 0 ? moment(x, 4, m) / (m2 * m2) : std::nan("");
}

double skewness(const std::vector<double>& x) {
  need(x, "skewness");
  const double m = mean(x), m2 = moment(x, 2, m);
  return m2 > 0 ? moment(x, 3, m) / std::pow(m2, 1.5) : std::nan("");
}

double min(const std::vector<double>& x) { need(x, "min"); return *std::min_element(x.begin(), x.end()); }
double max(const std::vector<double>& x) { need(x, "max"); return *std::max_element(x.begin(), x.end()); }
double count(const std::vector<double>& x) { return static_cast<double>(x.size()); }
double sum(const std::vector<double>& x) { double s = 0; for (double v : x) s += v; return s; }

const std::vector<std::string>& summaries() {
  static const std::vector<std::string> n = {"mean", "sd", "median", "mad", "percentile", "peak", "kurtosis", "skewness", "min", "max", "count", "sum"};
  return n;
}

double summary(const std::string& name, const std::vector<double>& x, double p, int ddof) {
  if (name == "mean") return mean(x);
  if (name == "sd") return sd(x, ddof);
  if (name == "median") return median(x);
  if (name == "mad") return mad(x);
  if (name == "percentile") return percentile(x, p);
  if (name == "peak") return peak(x);
  if (name == "kurtosis") return kurtosis(x);
  if (name == "skewness") return skewness(x);
  if (name == "min") return min(x);
  if (name == "max") return max(x);
  if (name == "count") return count(x);
  if (name == "sum") return sum(x);
  std::string all;
  for (const auto& s : summaries()) all += (all.empty() ? "" : ", ") + s;
  throw std::invalid_argument("stats: no summary '" + name + "' (" + all + ")");
}

std::vector<Summary> parseSummaries(const std::vector<std::string>& names) {
  std::vector<Summary> out;
  for (const auto& n : names) {
    Summary sm;
    const auto c = n.find(':');
    sm.name = n.substr(0, c);
    if (c != std::string::npos) {
      if (sm.name != "percentile") throw std::invalid_argument("stats: only percentile takes a parameter (percentile:75), not '" + n + "'");
      try { sm.p = std::stod(n.substr(c + 1)); } catch (...) { throw std::invalid_argument("stats: '" + n + "': percentile:p, p in [0, 100]"); }
      if (!(sm.p >= 0 && sm.p <= 100)) throw std::invalid_argument("stats: '" + n + "': p in [0, 100]");
    } else if (sm.name == "percentile") {
      throw std::invalid_argument("stats: percentile needs its p (percentile:75)");
    }
    if (std::find(summaries().begin(), summaries().end(), sm.name) == summaries().end()) summary(sm.name, {1.0});   // refuses, naming the valid ones
    out.push_back(sm);
  }
  if (out.empty()) throw std::invalid_argument("stats: at least one summary");
  return out;
}

std::vector<double> summarizeRows(const std::vector<double>& values, std::int64_t rows, std::int64_t members, const std::vector<Summary>& summaries,
                                  const std::string& combine, int ddof, std::vector<std::int64_t>* used) {
  if (rows < 1 || members < 1 || static_cast<std::int64_t>(values.size()) != rows * members) throw std::invalid_argument("stats: the table is not [rows x members]");
  if (combine != "each" && combine != "pool" && combine != "mean") throw std::invalid_argument("stats: combine is each, pool or mean (got '" + combine + "')");
  std::vector<std::vector<double>> samples;
  if (combine == "each") {
    for (std::int64_t r = 0; r < rows; ++r) {
      std::vector<double> x;
      for (std::int64_t m = 0; m < members; ++m) { const double v = values[static_cast<std::size_t>(r * members + m)]; if (std::isfinite(v)) x.push_back(v); }
      samples.push_back(std::move(x));
    }
  } else if (combine == "pool") {
    std::vector<double> x;
    for (double v : values) if (std::isfinite(v)) x.push_back(v);
    samples.push_back(std::move(x));
  } else {
    std::vector<double> x;
    for (std::int64_t m = 0; m < members; ++m) {
      double s = 0;
      bool ok = true;
      for (std::int64_t r = 0; r < rows && ok; ++r) { const double v = values[static_cast<std::size_t>(r * members + m)]; ok = std::isfinite(v); s += v; }
      if (ok) x.push_back(s / static_cast<double>(rows));
    }
    samples.push_back(std::move(x));
  }
  std::vector<double> out;
  if (used) used->clear();
  for (const auto& x : samples) {
    if (used) used->push_back(static_cast<std::int64_t>(x.size()));
    for (const Summary& sm : summaries) {
      double v = std::nan("");
      try { if (!x.empty() || sm.name == "count" || sm.name == "sum") v = summary(sm.name, x, sm.p, ddof); } catch (const std::invalid_argument&) { v = std::nan(""); }   // too few values for it
      out.push_back(v);
    }
  }
  return out;
}

}  // namespace nxr::stats

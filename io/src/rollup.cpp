// nxr-io — read-time roll-up (rollup.h).
#include "nxr/io/rollup.h"

#include "nxr/io/errors.h"
#include "nxr/io/measures.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::io::rollup {

namespace {
double neutral(Merge op) { return op == Merge::Max ? -std::numeric_limits<double>::infinity() : op == Merge::Min ? std::numeric_limits<double>::infinity() : 0.0; }
void need_foldable(Merge op, const char* what) {
  if (op == Merge::None) throw std::invalid_argument(std::string(what) + ": this measure is never merged (a median, a member's own value): it cannot be rolled up");
}
void fold(Merge op, double& a, double v) {
  if (op == Merge::Sum || op == Merge::Mean) a += v;
  else if (std::isnan(a) || std::isnan(v)) a = std::numeric_limits<double>::quiet_NaN();     // numpy's maximum / minimum propagate NaN
  else if (op == Merge::Max) a = std::max(a, v);
  else a = std::min(a, v);
}
}  // namespace

Merge merge_of(const std::string& m) { return merge_of(measure_merge(m), m); }
Merge merge_of(MergeRule rule, const std::string& m) {
  switch (rule) {
    case MergeRule::sum: return Merge::Sum;
    case MergeRule::max: return Merge::Max;
    case MergeRule::min: return Merge::Min;
    case MergeRule::mean: return Merge::Mean;
    case MergeRule::none: return Merge::None;
    case MergeRule::derived: throw std::invalid_argument("rollup: '" + m + "' is derived (computed at read from its ingredients, which are merged first)");
  }
  return Merge::Sum;
}
const char* merge_name(Merge m) { return m == Merge::Sum ? "sum" : m == Merge::Max ? "max" : m == Merge::Min ? "min" : m == Merge::Mean ? "mean" : "none"; }

std::vector<double> space(const std::vector<double>& x, std::int64_t rows, std::int64_t leaves, int depth, int level, Merge op, std::int64_t* nodes) {
  need_foldable(op, "rollup space");
  if (depth < 0 || depth > 30) throw std::invalid_argument("rollup space: the depth is 0 to 30");
  if (level > depth) throw std::invalid_argument("rollup space: level " + std::to_string(level) + " is finer than the stored depth " + std::to_string(depth));
  if (level < -1) throw std::invalid_argument("rollup space: the level is -1 (the whole cortex) or more");
  const std::int64_t per = std::int64_t(1) << depth;
  if (leaves <= 0 || leaves % per != 0) throw std::invalid_argument("rollup space: " + std::to_string(leaves) + " leaves are not a whole number of pieces of 2^" + std::to_string(depth));
  if (rows < 0 || static_cast<std::int64_t>(x.size()) != rows * leaves) throw std::invalid_argument("rollup space: x is rows x leaves");
  const std::int64_t pieces = leaves / per;
  const std::int64_t groups = level < 0 ? 1 : pieces * (std::int64_t(1) << level), block = leaves / groups;
  if (nodes) *nodes = groups;
  std::vector<double> y(static_cast<std::size_t>(rows * groups), neutral(op));
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t l = 0; l < leaves; ++l) fold(op, y[static_cast<std::size_t>(r * groups + l / block)], x[static_cast<std::size_t>(r * leaves + l)]);
  return y;
}

std::vector<double> time(const std::vector<double>& x, std::int64_t rows, std::int64_t frames, int m, Merge op, std::int64_t code0, std::int64_t* tiles) {
  need_foldable(op, "rollup time");
  if (m < 0 || m > 40) throw std::invalid_argument("rollup time: the level is 0 or above");
  if (rows < 0 || frames < 0 || static_cast<std::int64_t>(x.size()) != rows * frames) throw std::invalid_argument("rollup time: x is rows x frames");
  if (code0 < 0) throw std::invalid_argument("rollup time: the tower code of frame 0 is 0 or more");
  const std::int64_t k = std::int64_t(1) << m, front = code0 % k;
  const std::int64_t nt = (front + frames + k - 1) / k;
  if (tiles) *tiles = nt;
  std::vector<double> y(static_cast<std::size_t>(rows * nt), neutral(op));
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t f = 0; f < frames; ++f) fold(op, y[static_cast<std::size_t>(r * nt + (front + f) / k)], x[static_cast<std::size_t>(r * frames + f)]);
  return y;
}

std::vector<double> members(const std::vector<double>& x, std::int64_t rows, std::int64_t n, int shift, Merge op, std::int64_t* groups) {
  need_foldable(op, "rollup members");
  if (shift < 0) throw std::invalid_argument("rollup members: the shift is 0 or more");
  if (rows < 0 || n < 0 || static_cast<std::int64_t>(x.size()) != rows * n) throw std::invalid_argument("rollup members: x is rows x members");
  const int s = std::min(shift, 62);
  const std::int64_t g = n == 0 ? 0 : ((n - 1) >> s) + 1;      // members n: the last code n - 1 >> shift, plus one
  if (groups) *groups = g;
  std::vector<double> y(static_cast<std::size_t>(rows * g), neutral(op));
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t m = 0; m < n; ++m) fold(op, y[static_cast<std::size_t>(r * g + (m >> s))], x[static_cast<std::size_t>(r * n + m)]);
  return y;
}

std::vector<double> product_factor(const std::vector<double>& x, std::int64_t rows, const std::vector<std::int64_t>& radices, std::size_t factor, int shift, Merge op, std::vector<std::int64_t>* radices_out) {
  need_foldable(op, "rollup product");
  if (factor >= radices.size()) throw std::invalid_argument("rollup product: the factor is outside the radices");
  if (shift < 0) throw std::invalid_argument("rollup product: the shift is 0 or more");
  std::int64_t total = 1;
  for (const std::int64_t r : radices) { if (r < 1) throw std::invalid_argument("rollup product: a radix is 1 or more"); total *= r; }
  if (rows < 0 || static_cast<std::int64_t>(x.size()) != rows * total) throw std::invalid_argument("rollup product: x is rows x the product of the radices");
  const int s = std::min(shift, 62);
  std::vector<std::int64_t> out_radices = radices;
  out_radices[factor] = ((radices[factor] - 1) >> s) + 1;
  std::int64_t new_total = 1;
  for (const std::int64_t r : out_radices) new_total *= r;
  std::vector<double> y(static_cast<std::size_t>(rows * new_total), neutral(op));
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t c = 0; c < total; ++c) {
      std::int64_t rest = c, place = 1, to = 0;                          // unpack c (first factor fastest), roll the factor, repack with the new radices
      for (std::size_t k = 0; k < radices.size(); ++k) {
        std::int64_t m = rest % radices[k];
        rest /= radices[k];
        if (k == factor) m >>= s;
        to += m * place;
        place *= out_radices[k];
      }
      fold(op, y[static_cast<std::size_t>(r * new_total + to)], x[static_cast<std::size_t>(r * total + c)]);
    }
  if (radices_out) *radices_out = out_radices;
  return y;
}

std::vector<double> across(const std::vector<const std::vector<double>*>& subjects, Merge op) {
  need_foldable(op, "rollup across subjects");
  if (subjects.empty() || !subjects[0]) throw std::invalid_argument("rollup across subjects: at least one subject");
  const std::size_t n = subjects[0]->size();
  std::vector<double> y(n, neutral(op));
  for (const auto* sj : subjects) {
    if (!sj || sj->size() != n) throw std::invalid_argument("rollup across subjects: the subjects' arrays differ in shape");
    for (std::size_t i = 0; i < n; ++i) {
      const double v = (*sj)[i];
      if ((op == Merge::Sum || op == Merge::Mean) && std::isnan(v)) continue;
      fold(op, y[i], v);
    }
  }
  return y;
}

}  // namespace nxr::io::rollup

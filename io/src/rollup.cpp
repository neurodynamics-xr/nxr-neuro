// nxr-io — read-time roll-up (rollup.h).
#include "nxr/io/rollup.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::io::rollup {

namespace {
double neutral(Merge op) { return op == Merge::Sum ? 0.0 : op == Merge::Max ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity(); }
void fold(Merge op, double& a, double v) {
  if (op == Merge::Sum) a += v;
  else if (std::isnan(a) || std::isnan(v)) a = std::numeric_limits<double>::quiet_NaN();     // numpy's maximum / minimum propagate NaN
  else if (op == Merge::Max) a = std::max(a, v);
  else a = std::min(a, v);
}
}  // namespace

Merge merge_of(const std::string& m) {
  if (m == "max" || (m.size() >= 4 && m.compare(m.size() - 4, 4, "_max") == 0)) return Merge::Max;
  return m == "min" ? Merge::Min : Merge::Sum;
}
const char* merge_name(Merge m) { return m == Merge::Sum ? "sum" : m == Merge::Max ? "max" : "min"; }

std::vector<double> space(const std::vector<double>& x, std::int64_t rows, std::int64_t leaves, int depth, int level, Merge op, std::int64_t* nodes) {
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

}  // namespace nxr::io::rollup

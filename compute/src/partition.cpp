// nxr::partition — the Studio's measureOverPartition (backend/src/db/measurements.ts), line for line in its accumulation and its measures.
#include <string>
#include "nxr/partition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::partition {

Sums accumulate(const std::int32_t* codes, std::size_t n, int members, const double* values, const double* weights) {
  if (members < 0) throw std::invalid_argument("partition: a negative member count");
  Sums s;
  s.members = members;
  const auto M = static_cast<std::size_t>(members);
  s.weight.assign(M, 0.0); s.sum.assign(M, 0.0); s.sum_square.assign(M, 0.0); s.sum_abs.assign(M, 0.0); s.sum_sqrt_abs.assign(M, 0.0); s.sum_cube.assign(M, 0.0); s.sum_fourth.assign(M, 0.0);
  s.max.assign(M, -std::numeric_limits<double>::infinity()); s.min.assign(M, std::numeric_limits<double>::infinity()); s.abs_max.assign(M, 0.0);
  s.count.assign(M, 0); s.nonfinite.assign(M, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t t = codes[i];
    if (t < 0) { ++s.unassigned; continue; }
    if (t >= members) throw std::invalid_argument("partition: element " + std::to_string(i) + " is in member " + std::to_string(t) + ", outside [0, " + std::to_string(members) + ")");
    const auto k = static_cast<std::size_t>(t);
    const double wi = weights ? weights[i] : 1.0;
    s.weight[k] += wi; ++s.count[k];
    if (values) {
      const double v = values[i];
      if (!std::isfinite(v)) ++s.nonfinite[k];
      s.sum[k] += v * wi; s.sum_square[k] += v * v * wi; s.sum_cube[k] += v * v * v * wi; s.sum_fourth[k] += v * v * v * v * wi;
      if (v > s.max[k]) s.max[k] = v;
      if (v < s.min[k]) s.min[k] = v;
      const double a = std::abs(v);
      if (a > s.abs_max[k]) s.abs_max[k] = a;
      s.sum_abs[k] += a * wi; s.sum_sqrt_abs[k] += std::sqrt(a) * wi;
    }
  }
  return s;
}

std::vector<double> medians(const std::int32_t* codes, std::size_t n, int members, const double* values) {
  // numpy median per member: gather the finite values, then the middle one (the mean of the two middle ones when the count is even)
  if (members < 0) throw std::invalid_argument("partition: a negative member count");
  std::vector<std::vector<double>> held(static_cast<std::size_t>(members));
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t t = codes[i];
    if (t < 0) continue;
    if (t >= members) throw std::invalid_argument("partition: element " + std::to_string(i) + " is in member " + std::to_string(t) + ", outside [0, " + std::to_string(members) + ")");
    if (std::isfinite(values[i])) held[static_cast<std::size_t>(t)].push_back(values[i]);
  }
  std::vector<double> out(held.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t k = 0; k < held.size(); ++k) {
    auto& v = held[k];
    if (v.empty()) continue;
    const std::size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h), v.end());
    const double hi = v[h];
    if (v.size() % 2 == 1) { out[k] = hi; continue; }
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h));
    out[k] = 0.5 * (lo + hi);
  }
  return out;
}

const std::vector<std::string>& field_measures() { static const std::vector<std::string> m = {"mean", "max", "min", "mean_square", "abs_max"}; return m; }
const std::vector<std::string>& geometric_measures() { static const std::vector<std::string> m = {"area", "n_elements"}; return m; }
bool known_measure(const std::string& n) {
  return n == "mean" || n == "mean_square" || n == "max" || n == "min" || n == "abs_max" || n == "sum" || n == "area" || n == "n_elements" || n == "mean_abs" || n == "mean_sqrt_abs" || n == "mean_cube" || n == "mean_fourth";
}

double measure(const Sums& s, int m, const std::string& name) {
  const auto k = static_cast<std::size_t>(m);
  const double w = s.weight[k];
  double v;
  if (name == "mean") v = w > 0 ? s.sum[k] / w : 0.0;
  else if (name == "mean_square") v = w > 0 ? s.sum_square[k] / w : 0.0;
  else if (name == "max") v = s.max[k];
  else if (name == "min") v = s.min[k];
  else if (name == "abs_max") v = s.abs_max[k];
  else if (name == "sum") v = s.sum[k];
  else if (name == "area") v = w;
  else if (name == "n_elements") v = static_cast<double>(s.count[k]);
  else if (name == "mean_abs") v = w > 0 ? s.sum_abs[k] / w : 0.0;
  else if (name == "mean_sqrt_abs") v = w > 0 ? s.sum_sqrt_abs[k] / w : 0.0;
  else if (name == "mean_cube") v = w > 0 ? s.sum_cube[k] / w : 0.0;
  else if (name == "mean_fourth") v = w > 0 ? s.sum_fourth[k] / w : 0.0;
  else throw std::invalid_argument("partition: measure '" + name + "' is not one this producer writes (mean, mean_square, max, min, abs_max, sum, area, n_elements, mean_abs, mean_sqrt_abs, mean_cube, mean_fourth)");
  return std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
}

TileReduction reduceTiles(const float* values, std::size_t rows, std::size_t samples, std::size_t width, std::size_t tiles) {
  // measurements.ts reduceTiles, line for line: per row, per tile, the run's min / max / Σx / Σx² in double, then folded into the float32 tiles
  if (width == 0) throw std::invalid_argument("reduceTiles: a tile is at least one sample wide");
  TileReduction r;
  r.min.assign(tiles, std::numeric_limits<float>::infinity());
  r.max.assign(tiles, -std::numeric_limits<float>::infinity());
  r.sum.assign(tiles, 0.0f);
  r.sumSq.assign(tiles, 0.0f);
  r.count.assign(tiles, 0.0f);
  for (std::size_t c = 0; c < rows; ++c) {
    const float* row = values + c * samples;
    for (std::size_t t = 0; t < tiles; ++t) {
      const std::size_t from = t * width, to = std::min(from + width, samples);
      double lo = r.min[t], hi = r.max[t], s = 0, q = 0;
      for (std::size_t i = from; i < to; ++i) { const double x = row[i]; if (x < lo) lo = x; if (x > hi) hi = x; s += x; q += x * x; }
      r.min[t] = static_cast<float>(lo);
      r.max[t] = static_cast<float>(hi);
      r.sum[t] = static_cast<float>(static_cast<double>(r.sum[t]) + s);
      r.sumSq[t] = static_cast<float>(static_cast<double>(r.sumSq[t]) + q);
      r.count[t] = static_cast<float>(static_cast<double>(r.count[t]) + (static_cast<double>(to) - static_cast<double>(from)));   // as the Studio: negative past the data
    }
  }
  return r;
}

CrossSpectrumTiles crossSpectrum(const Eigen::MatrixXcd& Z, std::size_t width, std::size_t tiles) {
  if (width == 0) throw std::invalid_argument("crossSpectrum: a tile is at least one sample wide");
  const Eigen::Index C = Z.rows(), n = Z.cols();
  CrossSpectrumTiles r;
  r.sum.assign(tiles, Eigen::MatrixXcd::Zero(C, C));
  r.count.assign(tiles, 0);
  for (std::size_t t = 0; t < tiles; ++t) {
    const Eigen::Index from = static_cast<Eigen::Index>(t * width), to = std::min<Eigen::Index>(from + static_cast<Eigen::Index>(width), n);
    if (to <= from) continue;
    const auto block = Z.middleCols(from, to - from);
    r.sum[t].noalias() = block * block.adjoint();          // Σ_t z zᴴ: one complex product
    r.count[t] = to - from;
  }
  return r;
}

Eigen::MatrixXd meanOverMembers(const float* data, std::size_t rows, std::size_t stride, std::size_t cols, const std::int32_t* codes, int members, const double* weights) {
  Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(members, static_cast<Eigen::Index>(cols));
  std::vector<double> w(static_cast<std::size_t>(members), 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::int32_t c = codes[i];
    if (c < 0) continue;
    if (c >= members) throw std::invalid_argument("meanOverMembers: code " + std::to_string(c) + " is past the " + std::to_string(members) + " members");
    const double wi = weights ? weights[i] : 1.0;
    w[static_cast<std::size_t>(c)] += wi;
    const float* x = data + i * stride;
    for (std::size_t j = 0; j < cols; ++j) sum(c, static_cast<Eigen::Index>(j)) += wi * static_cast<double>(x[j]);
  }
  for (int m = 0; m < members; ++m) sum.row(m) = w[static_cast<std::size_t>(m)] > 0 ? Eigen::RowVectorXd(sum.row(m) / w[static_cast<std::size_t>(m)]) : Eigen::RowVectorXd::Constant(static_cast<Eigen::Index>(cols), std::numeric_limits<double>::quiet_NaN());
  return sum;
}

MatrixSums sumOverMembers(const std::vector<Eigen::MatrixXcd>& value, const std::vector<int>& member, int members) {
  if (value.size() != member.size()) throw std::invalid_argument("sumOverMembers: one member per value");
  if (members < 0) throw std::invalid_argument("sumOverMembers: a negative number of members");
  MatrixSums out;
  out.sum.assign(static_cast<std::size_t>(members), Eigen::MatrixXcd());
  out.count.assign(static_cast<std::size_t>(members), 0);
  for (std::size_t i = 0; i < value.size(); ++i) {
    const int m = member[i];
    if (m == -1) continue;
    if (m < 0 || m >= members) throw std::invalid_argument("sumOverMembers: a member outside [0, members)");
    Eigen::MatrixXcd& s = out.sum[static_cast<std::size_t>(m)];
    if (out.count[static_cast<std::size_t>(m)] == 0) s = value[i];
    else {
      if (s.rows() != value[i].rows() || s.cols() != value[i].cols()) throw std::invalid_argument("sumOverMembers: the values of a member have different shapes");
      s += value[i];
    }
    ++out.count[static_cast<std::size_t>(m)];
  }
  return out;
}

AnalyticSums::AnalyticSums(int members_, std::int64_t tiles_) : members(members_), tiles(tiles_) {
  const std::size_t n = static_cast<std::size_t>(members_) * static_cast<std::size_t>(tiles_);
  power.assign(n, 0.0);
  env.assign(n, 0.0);
  envmax.assign(n, -std::numeric_limits<double>::infinity());
}

void analyticSums(AnalyticSums& out, const Eigen::MatrixXd& re, const Eigen::MatrixXd& im, const std::int32_t* codes, const double* weights, const std::int32_t* tile) {
  if (re.rows() != im.rows() || re.cols() != im.cols()) throw std::invalid_argument("analyticSums: re and im differ in shape");
  const Eigen::Index rows = re.rows(), samples = re.cols();
  for (Eigen::Index v = 0; v < rows; ++v) if (codes[v] >= out.members) throw std::invalid_argument("analyticSums: a code is beyond the members");
  for (Eigen::Index s = 0; s < samples; ++s) {
    if (tile[s] < 0) continue;
    if (tile[s] >= out.tiles) throw std::invalid_argument("analyticSums: a tile is beyond the time axis");
    const std::size_t base = static_cast<std::size_t>(tile[s]) * static_cast<std::size_t>(out.members);
    double* P = out.power.data() + base;
    double* E = out.env.data() + base;
    double* X = out.envmax.data() + base;
    const double* r = re.data() + s * re.rows();
    const double* i = im.data() + s * im.rows();
    for (Eigen::Index v = 0; v < rows; ++v) {
      const std::int32_t c = codes[v];
      if (c < 0) continue;
      const double p = r[v] * r[v] + i[v] * i[v], m = std::sqrt(p), a = weights ? weights[v] : 1.0;
      P[c] += a * p;
      E[c] += a * m;
      if (m > X[c]) X[c] = m;
    }
  }
}


TensorSums::TensorSums(int members_, std::int64_t tiles_) : members(members_), tiles(tiles_) {
  const std::size_t n = static_cast<std::size_t>(members_) * static_cast<std::size_t>(tiles_);
  for (auto* a : arrays()) a->assign(n, 0.0);
}
std::vector<std::vector<double>*> TensorSums::arrays() { return {&t11, &t22, &tnn, &t12r, &t12i, &t1nr, &t1ni, &t2nr, &t2ni, &total}; }
const char* const* TensorSums::keys() {
  static const char* const k[] = {"t11", "t22", "tnn", "t12r", "t12i", "t1nr", "t1ni", "t2nr", "t2ni", "total"};
  return k;
}

void tensorSums(TensorSums& out, const Eigen::MatrixXd* J, const std::int32_t* codes, const double* weights, const double* totalWeights, const std::int32_t* tile) {
  for (int k = 1; k < 6; ++k) if (J[k].rows() != J[0].rows() || J[k].cols() != J[0].cols()) throw std::invalid_argument("tensorSums: the six parts differ in shape");
  const Eigen::Index rows = J[0].rows(), samples = J[0].cols();
  for (Eigen::Index v = 0; v < rows; ++v) if (codes[v] >= out.members) throw std::invalid_argument("tensorSums: a code is beyond the members");
  for (Eigen::Index s = 0; s < samples; ++s) {
    if (tile[s] < 0) continue;
    if (tile[s] >= out.tiles) throw std::invalid_argument("tensorSums: a tile is beyond the time axis");
    const std::size_t base = static_cast<std::size_t>(tile[s]) * static_cast<std::size_t>(out.members);
    const double *j1r = J[0].data() + s * rows, *j1i = J[1].data() + s * rows, *j2r = J[2].data() + s * rows, *j2i = J[3].data() + s * rows, *jnr = J[4].data() + s * rows, *jni = J[5].data() + s * rows;
    for (Eigen::Index v = 0; v < rows; ++v) {
      const std::int32_t c = codes[v];
      if (c < 0) continue;
      const std::size_t i = base + static_cast<std::size_t>(c);
      const double a = weights[v], b = totalWeights[v];
      const double a1r = j1r[v], a1i = j1i[v], a2r = j2r[v], a2i = j2i[v], anr = jnr[v], ani = jni[v];
      const double p11 = a1r * a1r + a1i * a1i, p22 = a2r * a2r + a2i * a2i, pnn = anr * anr + ani * ani;
      out.t11[i] += a * p11; out.t22[i] += a * p22; out.tnn[i] += a * pnn;
      out.t12r[i] += a * (a1r * a2r + a1i * a2i); out.t12i[i] += a * (a1i * a2r - a1r * a2i);
      out.t1nr[i] += a * (a1r * anr + a1i * ani); out.t1ni[i] += a * (a1i * anr - a1r * ani);
      out.t2nr[i] += a * (a2r * anr + a2i * ani); out.t2ni[i] += a * (a2i * anr - a2r * ani);
      out.total[i] += b * (p11 + p22 + pnn);
    }
  }
}

}  // namespace nxr::partition

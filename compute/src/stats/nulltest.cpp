// nxr::stats — comparing a measurement with its surrogate draws (nulltest.h).
#include "nxr/stats/nulltest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nxr::stats {

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
void same_size(Eigen::Index a, Eigen::Index b, const char* who) {
  if (a != b) throw std::invalid_argument(std::string(who) + ": the data and the draw have different numbers of members (" + std::to_string(a) + ", " + std::to_string(b) + ")");
}
}  // namespace

double sequentialP(std::int64_t exceedances, std::int64_t draws, std::int64_t h) {
  if (draws < 1 || exceedances < 0 || exceedances > draws || h < 1) throw std::invalid_argument("sequentialP: draws >= 1, 0 <= exceedances <= draws, h >= 1");
  return exceedances >= h ? static_cast<double>(h) / static_cast<double>(draws) : static_cast<double>(exceedances + 1) / static_cast<double>(draws + 1);
}

Eigen::VectorXd sequentialPs(const Eigen::VectorXd& data, const Eigen::VectorXd& exceedances, const Eigen::VectorXd& draws, std::int64_t h) {
  same_size(data.size(), exceedances.size(), "sequentialPs");
  same_size(data.size(), draws.size(), "sequentialPs");
  Eigen::VectorXd p = Eigen::VectorXd::Constant(data.size(), kNaN);
  for (Eigen::Index i = 0; i < data.size(); ++i)
    if (std::isfinite(data[i]) && draws[i] > 0) p[i] = sequentialP(static_cast<std::int64_t>(exceedances[i]), static_cast<std::int64_t>(draws[i]), h);
  return p;
}

Eigen::VectorXd fixedDrawPs(const Eigen::VectorXd& data, const Eigen::VectorXd& exceedances, std::int64_t draws) {
  same_size(data.size(), exceedances.size(), "fixedDrawPs");
  if (draws < 1) throw std::invalid_argument("fixedDrawPs: at least one draw");
  Eigen::VectorXd p = Eigen::VectorXd::Constant(data.size(), kNaN);
  for (Eigen::Index i = 0; i < data.size(); ++i) if (std::isfinite(data[i])) p[i] = (exceedances[i] + 1.0) / static_cast<double>(draws + 1);
  return p;
}

MeanComparison compareMeanOverMembers(const Eigen::VectorXd& data, const Eigen::MatrixXd& draws) {
  same_size(data.size(), draws.cols(), "compareMeanOverMembers");
  if (draws.rows() < 2) throw std::invalid_argument("compareMeanOverMembers: at least 2 draws");
  MeanComparison c;
  c.draws = draws.rows();
  for (Eigen::Index i = 0; i < data.size(); ++i) if (std::isfinite(data[i])) { c.mean += data[i]; ++c.members; }
  if (c.members == 0) throw std::invalid_argument("compareMeanOverMembers: no member has a statistic");
  c.mean /= static_cast<double>(c.members);
  Eigen::VectorXd m(draws.rows());
  for (Eigen::Index j = 0; j < draws.rows(); ++j) {
    double s = 0; std::int64_t n = 0;
    for (Eigen::Index i = 0; i < data.size(); ++i) if (std::isfinite(data[i]) && std::isfinite(draws(j, i))) { s += draws(j, i); ++n; }
    m[j] = n > 0 ? s / static_cast<double>(n) : kNaN;
  }
  std::int64_t finite = 0, ge = 0;
  for (Eigen::Index j = 0; j < m.size(); ++j) if (std::isfinite(m[j])) { c.nullMean += m[j]; ++finite; ge += m[j] >= c.mean; }
  c.nullMean = finite > 0 ? c.nullMean / static_cast<double>(finite) : kNaN;
  double ss = 0;
  for (Eigen::Index j = 0; j < m.size(); ++j) if (std::isfinite(m[j])) ss += (m[j] - c.nullMean) * (m[j] - c.nullMean);
  c.nullSd = finite > 1 ? std::sqrt(ss / static_cast<double>(finite - 1)) : kNaN;
  c.p = static_cast<double>(1 + ge) / static_cast<double>(c.draws + 1);
  return c;
}

Eigen::VectorXd benjaminiHochbergQ(const Eigen::VectorXd& p) {
  std::vector<Eigen::Index> idx;
  for (Eigen::Index i = 0; i < p.size(); ++i) if (std::isfinite(p[i])) idx.push_back(i);
  std::sort(idx.begin(), idx.end(), [&](Eigen::Index a, Eigen::Index b) { return p[a] < p[b]; });
  Eigen::VectorXd q = Eigen::VectorXd::Constant(p.size(), kNaN);
  const double m = static_cast<double>(idx.size());
  double run = 1.0;
  for (std::size_t r = idx.size(); r-- > 0;) { run = std::min(run, p[idx[r]] * m / static_cast<double>(r + 1)); q[idx[r]] = std::min(run, 1.0); }
  return q;
}

Eigen::VectorXd exceedanceIndicator(const Eigen::VectorXd& data, const Eigen::VectorXd& draw, Side side) {
  same_size(data.size(), draw.size(), "exceedanceIndicator");
  Eigen::VectorXd e(data.size());
  for (Eigen::Index i = 0; i < data.size(); ++i) {
    if (!std::isfinite(data[i]) || !std::isfinite(draw[i])) { e[i] = kNaN; continue; }
    e[i] = (side == Side::upper ? draw[i] >= data[i] : draw[i] <= data[i]) ? 1.0 : 0.0;
  }
  return e;
}

Tally emptyTally(Eigen::Index members) {
  Tally t;
  t.upper = t.lower = t.finite = t.sum = Eigen::VectorXd::Zero(members);
  return t;
}

Tally tallyDraw(const Tally& tally, const Eigen::VectorXd& data, const Eigen::VectorXd& draw) {
  same_size(tally.upper.size(), draw.size(), "tallyDraw");
  same_size(data.size(), draw.size(), "tallyDraw");
  Tally t = tally;
  const Eigen::VectorXd up = exceedanceIndicator(data, draw, Side::upper), lo = exceedanceIndicator(data, draw, Side::lower);
  for (Eigen::Index i = 0; i < draw.size(); ++i) {
    if (up[i] == 1.0) t.upper[i] += 1;
    if (lo[i] == 1.0) t.lower[i] += 1;
    if (std::isfinite(draw[i])) { t.finite[i] += 1; t.sum[i] += draw[i]; }
  }
  t.draws += 1;
  return t;
}

Tally rollUp(const Tally& a, const Tally& b) {
  same_size(a.upper.size(), b.upper.size(), "rollUp");
  Tally t;
  t.upper = a.upper + b.upper; t.lower = a.lower + b.lower; t.finite = a.finite + b.finite; t.sum = a.sum + b.sum;
  t.draws = a.draws + b.draws;
  return t;
}

std::int64_t undecidedMembers(const Eigen::VectorXd& data, const Tally& tally, std::int64_t h, Side side) {
  same_size(data.size(), tally.upper.size(), "undecidedMembers");
  const Eigen::VectorXd& e = side == Side::upper ? tally.upper : tally.lower;
  std::int64_t n = 0;
  for (Eigen::Index i = 0; i < data.size(); ++i) n += std::isfinite(data[i]) && e[i] < static_cast<double>(h);
  return n;
}

Eigen::VectorXd nullMean(const Tally& tally) {
  Eigen::VectorXd m(tally.sum.size());
  for (Eigen::Index i = 0; i < m.size(); ++i) m[i] = tally.finite[i] > 0 ? tally.sum[i] / tally.finite[i] : kNaN;
  return m;
}

Eigen::VectorXd countRatio(const Eigen::VectorXd& dataCounts, const Eigen::VectorXd& nullCountSum, std::int64_t draws) {
  same_size(dataCounts.size(), nullCountSum.size(), "countRatio");
  if (draws < 1) throw std::invalid_argument("countRatio: at least one draw");
  Eigen::VectorXd r(dataCounts.size());
  for (Eigen::Index i = 0; i < r.size(); ++i) { const double mean = nullCountSum[i] / static_cast<double>(draws); r[i] = mean > 0 ? dataCounts[i] / mean : kNaN; }
  return r;
}

Eigen::VectorXd unbiasedConsistency(const Eigen::MatrixXd& sum, const Eigen::VectorXd& n, int minCount) {
  same_size(sum.rows(), n.size(), "unbiasedConsistency");
  Eigen::VectorXd r(n.size());
  for (Eigen::Index t = 0; t < n.size(); ++t) { const double c = n[t]; r[t] = c >= std::max(minCount, 2) ? (sum.row(t).squaredNorm() - c) / (c * (c - 1)) : kNaN; }
  return r;
}

Eigen::MatrixXd meanResultants(const Eigen::MatrixXd& sum, const Eigen::VectorXd& n) {
  same_size(sum.rows(), n.size(), "meanResultants");
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(sum.rows(), sum.cols());
  for (Eigen::Index t = 0; t < n.size(); ++t) if (n[t] > 0) m.row(t) = sum.row(t) / n[t];
  return m;
}

Eigen::VectorXd occupancyFraction(const std::vector<double>& v, bool occupiedOnly) {
  if (v.size() % 2 != 0) throw std::invalid_argument("occupancyFraction: the values are not [vertices x (in a patch, frames)]");
  const Eigen::Index V = static_cast<Eigen::Index>(v.size() / 2);
  Eigen::VectorXd f(V);
  for (Eigen::Index i = 0; i < V; ++i) {
    const double in = v[static_cast<std::size_t>(2 * i)], frames = v[static_cast<std::size_t>(2 * i + 1)];
    if (occupiedOnly) f[i] = frames > 0 && in > 0 ? in / frames : kNaN;
    else f[i] = frames > 0 ? in / frames : 0.0;
  }
  return f;
}

Eigen::VectorXd patchShares(const Eigen::VectorXd& counts, double patches, double frames) {
  Eigen::VectorXd s(counts.size() + 1);
  for (Eigen::Index k = 0; k < counts.size(); ++k) s[k] = patches > 0 ? counts[k] / patches : kNaN;
  s[counts.size()] = frames > 0 ? patches / frames : kNaN;
  return s;
}

}  // namespace nxr::stats

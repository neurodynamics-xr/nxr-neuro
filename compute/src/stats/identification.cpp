// nxr::stats — see include/nxr/stats/identification.h.
#include "nxr/stats/identification.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "nxr/stats/pca.h"

namespace nxr::stats {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
void same_shape(const MatrixXd& a, const MatrixXd& b, const char* who) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument(std::string(who) + ": the two tables differ in shape (" + std::to_string(a.rows()) + " x " + std::to_string(a.cols()) + " and " +
                                    std::to_string(b.rows()) + " x " + std::to_string(b.cols()) + "); row i of each is the same member");
    if (a.rows() < 2) throw std::invalid_argument(std::string(who) + ": at least two members");
}
// numpy's argmax over a row or a column: the first of the largest
Index first_max_row(const MatrixXd& r, Index i) { Index at = 0; for (Index j = 1; j < r.cols(); ++j) if (r(i, j) > r(i, at)) at = j; return at; }
Index first_max_col(const MatrixXd& r, Index j) { Index at = 0; for (Index i = 1; i < r.rows(); ++i) if (r(i, j) > r(at, j)) at = i; return at; }
}  // namespace

SpectralFeature spectralFeature(const std::vector<Eigen::MatrixXd>& spectra, double df, double f0, const std::vector<std::array<double, 2>>& bands) {
  if (spectra.empty() || spectra.size() != bands.size())
    throw std::invalid_argument("spectralFeature: one spectrum a band (" + std::to_string(spectra.size()) + " spectra, " + std::to_string(bands.size()) + " bands)");
  if (!(df > 0)) throw std::invalid_argument("spectralFeature: the frequency resolution is positive");
  const Eigen::Index R = spectra[0].rows(), K = spectra[0].cols();
  for (std::size_t b = 1; b < spectra.size(); ++b)
    if (spectra[b].rows() != R || spectra[b].cols() != K)
      throw std::invalid_argument("spectralFeature: band " + std::to_string(b) + "'s spectrum is " + std::to_string(spectra[b].rows()) + " x " + std::to_string(spectra[b].cols()) + ", the first " + std::to_string(R) + " x " + std::to_string(K));
  SpectralFeature f;
  std::vector<std::pair<std::size_t, Eigen::Index>> kept;      // (band, bin)
  const double eps = 1e-9 * df;
  for (std::size_t b = 0; b < bands.size(); ++b) {
    std::int64_t c = 0;
    for (Eigen::Index k = 0; k < K; ++k) {
      const double fk = f0 + static_cast<double>(k) * df;
      if (fk >= bands[b][0] - eps && fk <= bands[b][1] + eps) { kept.emplace_back(b, k); f.frequencies.push_back(fk); ++c; }
    }
    if (c == 0) throw std::invalid_argument("spectralFeature: band " + std::to_string(b) + " [" + std::to_string(bands[b][0]) + ", " + std::to_string(bands[b][1]) + "] Hz holds no bin");
    f.counts.push_back(c);
  }
  f.values.resize(R, static_cast<Eigen::Index>(kept.size()));
  for (std::size_t j = 0; j < kept.size(); ++j) f.values.col(static_cast<Eigen::Index>(j)) = spectra[kept[j].first].col(kept[j].second);
  return f;
}

std::vector<double> connectomeFeature(const std::vector<Eigen::MatrixXd>& M) {
  if (M.empty()) throw std::invalid_argument("connectomeFeature: at least one band's matrix");
  const Eigen::Index r = M[0].rows();
  const std::size_t B = M.size();
  for (std::size_t b = 0; b < B; ++b)
    if (M[b].rows() != r || M[b].cols() != r) throw std::invalid_argument("connectomeFeature: band " + std::to_string(b) + "'s matrix is " + std::to_string(M[b].rows()) + " x " + std::to_string(M[b].cols()) + ", not " + std::to_string(r) + " x " + std::to_string(r));
  std::vector<double> v(static_cast<std::size_t>(r * r) * B);
  for (Eigen::Index i = 0; i < r; ++i)
    for (Eigen::Index j = 0; j < r; ++j)
      for (std::size_t b = 0; b < B; ++b) v[static_cast<std::size_t>(i * r + j) * B + b] = M[b](i, j);
  return v;
}

MatrixXd featureTable(const MatrixXd& members, const std::vector<std::int64_t>& shape, FeatureLayout layout) {
    std::int64_t size = 1;
    for (const auto s : shape) {
        if (s < 1) throw std::invalid_argument("featureTable: every axis has at least one entry");
        size *= s;
    }
    if (members.cols() != size)
        throw std::invalid_argument("featureTable: each member holds " + std::to_string(members.cols()) + " values, the shape " + std::to_string(size));
    if (layout == FeatureLayout::Flatten) return members;
    if ((shape.size() != 2 && shape.size() != 3) || shape[0] != shape[1])
        throw std::invalid_argument("featureTable: the upper triangle is of a square [r x r] or a stack of them [r x r x slices]");
    const std::int64_t r = shape[0], slices = shape.size() == 3 ? shape[2] : 1, per = r * (r + 1) / 2;
    MatrixXd out(members.rows(), per * slices);
    for (Index m = 0; m < members.rows(); ++m)
        for (std::int64_t s = 0; s < slices; ++s) {
            std::int64_t k = s * per;
            for (std::int64_t i = 0; i < r; ++i)
                for (std::int64_t j = i; j < r; ++j) out(m, k++) = members(m, (i * r + j) * slices + s);
        }
    return out;
}

MatrixXd zscoreRows(const MatrixXd& X) {
    if (X.cols() < 2) throw std::invalid_argument("zscoreRows: at least two columns");
    MatrixXd Z(X.rows(), X.cols());
    for (Index i = 0; i < X.rows(); ++i) {
        if (!X.row(i).allFinite()) throw std::invalid_argument("zscoreRows: row " + std::to_string(i) + " holds a value that is not finite");
        const double mu = X.row(i).mean();
        const auto c = (X.row(i).array() - mu).matrix();
        const double sd = std::sqrt(c.squaredNorm() / static_cast<double>(X.cols()));
        if (!(sd > 0)) throw std::invalid_argument("zscoreRows: row " + std::to_string(i) + " has no variance (every value the same)");
        Z.row(i) = c / sd;
    }
    return Z;
}

MatrixXd identificationCorrelation(const MatrixXd& target, const MatrixXd& database) {
    if (target.cols() != database.cols())
        throw std::invalid_argument("identificationCorrelation: the target has " + std::to_string(target.cols()) + " features, the database " + std::to_string(database.cols()));
    return zscoreRows(target) * zscoreRows(database).transpose() / static_cast<double>(target.cols());
}

IdentificationAccuracy identificationAccuracy(const MatrixXd& r) {
    if (r.rows() != r.cols() || r.rows() < 1) throw std::invalid_argument("identificationAccuracy: the correlation matrix is square, member i on row i and column i");
    const Index n = r.rows();
    Index rows = 0, cols = 0;
    for (Index i = 0; i < n; ++i) { rows += first_max_row(r, i) == i; cols += first_max_col(r, i) == i; }
    IdentificationAccuracy a;
    a.accuracyByRow = static_cast<double>(rows) / static_cast<double>(n);
    a.accuracyByColumn = static_cast<double>(cols) / static_cast<double>(n);
    a.chance = 1.0 / static_cast<double>(n);
    return a;
}

VectorXd selfIdentifiability(const MatrixXd& r, SelfReference reference) {
    if (r.rows() != r.cols() || r.rows() < 2) throw std::invalid_argument("selfIdentifiability: the correlation matrix is square, with at least two members");
    const Index n = r.rows();
    if (reference == SelfReference::Others && n < 3) throw std::invalid_argument("selfIdentifiability (others): at least three members (two others for a spread)");
    VectorXd d(n);
    for (Index i = 0; i < n; ++i) {
        double s = 0, ss = 0;
        Index m = 0;
        for (Index j = 0; j < n; ++j) {
            if (reference == SelfReference::Others && j == i) continue;
            s += r(i, j); ++m;
        }
        const double mu = s / static_cast<double>(m);
        for (Index j = 0; j < n; ++j) {
            if (reference == SelfReference::Others && j == i) continue;
            ss += (r(i, j) - mu) * (r(i, j) - mu);
        }
        const double sd = std::sqrt(ss / static_cast<double>(m));
        if (!(sd > 0)) throw std::invalid_argument("selfIdentifiability: row " + std::to_string(i) + " has no spread");
        d[i] = (r(i, i) - mu) / sd;
    }
    return d;
}

VectorXd intraclassCorrelation(const MatrixXd& a, const MatrixXd& b) {
    same_shape(a, b, "intraclassCorrelation");
    const double n = static_cast<double>(a.rows()), k = 2;
    VectorXd icc(a.cols());
    for (Index e = 0; e < a.cols(); ++e) {
        const double grand = (a.col(e).sum() + b.col(e).sum()) / (n * k);
        double ss_t = 0, ss_w = 0;
        for (Index i = 0; i < a.rows(); ++i) {
            const double x = a(i, e), y = b(i, e), m = (x + y) / 2;
            ss_t += (x - grand) * (x - grand) + (y - grand) * (y - grand);
            ss_w += (x - m) * (x - m) + (y - m) * (y - m);
        }
        const double ms_b = (ss_t - ss_w) / (n - 1), ms_w = ss_w / (n * (k - 1));
        icc[e] = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w);
    }
    return icc;
}

VectorXd differentialPower(const MatrixXd& a, const MatrixXd& b) {
    same_shape(a, b, "differentialPower");
    const Index n = a.rows();
    const double denom = 2.0 * static_cast<double>(n - 1), floor = 1.0 / denom;
    VectorXd dp(a.cols());
    for (Index e = 0; e < a.cols(); ++e) {
        double sum = 0;
        for (Index i = 0; i < n; ++i) {
            const double ai = a(i, e), bi = b(i, e), self = ai * bi;
            Index above = 0;
            for (Index j = 0; j < n; ++j) {
                if (j == i) continue;
                above += (ai * b(j, e) > self) + (bi * a(j, e) > self);
            }
            const double p = above == 0 ? floor : static_cast<double>(above) / denom;
            sum += -std::log(p);
        }
        dp[e] = sum;
    }
    return dp;
}

VectorXd groupConsistency(const MatrixXd& a, const MatrixXd& b) {
    same_shape(a, b, "groupConsistency");
    return (a.array() * b.array()).colwise().mean().transpose();
}

Identification identification(const MatrixXd& target, const MatrixXd& database, SelfReference reference) {
    same_shape(target, database, "identification");
    Identification out;
    const MatrixXd zt = zscoreRows(target), zd = zscoreRows(database);
    out.correlation = zt * zd.transpose() / static_cast<double>(target.cols());
    out.accuracy = identificationAccuracy(out.correlation);
    out.selfCorrelation = out.correlation.diagonal();
    out.selfIdentifiability = selfIdentifiability(out.correlation, reference);
    out.icc = intraclassCorrelation(zt, zd);
    out.differentialPower = differentialPower(zt, zd);
    out.groupConsistency = groupConsistency(zt, zd);
    return out;
}

std::vector<std::int64_t> identificationSteps(std::int64_t rows) {
    std::vector<std::int64_t> s;
    for (const std::int64_t k : {std::int64_t(2), std::int64_t(5)}) s.push_back(k);
    for (std::int64_t k = 10; k < rows; k += 10) s.push_back(k);
    return s;
}

IdentificationCurve identificationCurve(const MatrixXd& scores, const MatrixXd& components, const VectorXd& mean, const VectorXd& scale,
                                        const std::vector<std::int64_t>& steps, SelfReference reference) {
    const Index rows = scores.rows(), p = components.cols(), rank = components.rows();
    if (rows < 4 || rows % 2) throw std::invalid_argument("identificationCurve: the stacked table has 2n rows (the n target rows, then the n database rows), n >= 2; got " + std::to_string(rows));
    if (scores.cols() != rank || mean.size() != p || scale.size() != p) throw std::invalid_argument("identificationCurve: the scores, components and centring are not of one decomposition");
    if (steps.empty()) throw std::invalid_argument("identificationCurve: at least one step (a number of components)");
    const Index n = rows / 2;
    IdentificationCurve c;
    c.steps = steps;
    const Index m = static_cast<Index>(steps.size());
    c.accuracyByRow.resize(m); c.accuracyByColumn.resize(m); c.meanSelfIdentifiability.resize(m);
    std::vector<Identification> each;
    for (Index s = 0; s < m; ++s) {
        const std::int64_t k = steps[static_cast<std::size_t>(s)];
        if (k < 1 || k > std::min<std::int64_t>(rows, p))
            throw std::invalid_argument("identificationCurve: a step of " + std::to_string(k) + " components; each is from 1 to min(2n, features) = " + std::to_string(std::min<std::int64_t>(rows, p)));
        const std::int64_t used = std::min<std::int64_t>(k, rank);
        c.used.push_back(used);
        const MatrixXd Xk = reconstruct(scores, components, mean, scale, static_cast<int>(used));
        Identification id = identification(Xk.topRows(n), Xk.bottomRows(n), reference);
        c.accuracyByRow[s] = id.accuracy.accuracyByRow;
        c.accuracyByColumn[s] = id.accuracy.accuracyByColumn;
        c.meanSelfIdentifiability[s] = id.selfIdentifiability.mean();
        if (s == 0 || c.meanSelfIdentifiability[s] > c.meanSelfIdentifiability[c.best]) { c.best = s; c.atBest = std::move(id); }
    }
    return c;
}

}  // namespace nxr::stats

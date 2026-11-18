// nxr::pca against its reference (golden: test/golden/pca_golden.txt, gen_pca_golden.py: numpy's SVD of the centred table, the sign convention applied; synthetic data only), and its own properties:
// the sign convention, orthonormal components, the variance shares, the reconstruction (full rank returns the table, k = 0 the mean, k components the best rank-k fit), agreement with the eigendecomposition
// of the table's covariance (nxr::signal::leading), a rank-deficient table, and the refusals.
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "golden_io.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/stats/pca.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace nxr::stats;

static bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
  std::cout << "test_pca\n";
  Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/pca_golden.txt");
  EXPECT(g.good(), "the golden is there");
  for (const std::string name : {"wide", "tall", "scaled"}) {
    const MatrixXd X = g.mat(name + "_X");
    const bool uv = g.mat(name + "_unit_variance")(0, 0) > 0.5;
    const int rank = static_cast<int>(g.mat(name + "_rank")(0, 0));
    const MatrixXd comp = g.mat(name + "_components"), S = g.mat(name + "_S"), ev = g.mat(name + "_ev"), ratio = g.mat(name + "_ratio"), scores = g.mat(name + "_scores");
    const MatrixXd mean = g.mat(name + "_mean"), scale = g.mat(name + "_scale");
    PcaOptions o;
    o.scale = uv ? Scale::UnitVariance : Scale::None;
    const PcaResult r = pca(X, o);
    EXPECT(r.rank == rank && r.components.rows() == rank && r.scores.cols() == rank, name + ": the rank is numpy's (" + std::to_string(rank) + ")");
    // singular values and variances to rounding; components and scores in the same sign
    const double s_err = (r.singularValues - S.row(0).transpose()).cwiseAbs().maxCoeff() / S.maxCoeff();
    const double e_err = (r.explainedVariance - ev.row(0).transpose()).cwiseAbs().maxCoeff() / ev.maxCoeff();
    const double q_err = (r.explainedVarianceRatio - ratio.row(0).transpose()).cwiseAbs().maxCoeff();
    const double c_err = relErr(r.components, comp), t_err = relErr(r.scores, scores);
    const double m_err = (r.mean - mean.row(0).transpose()).cwiseAbs().maxCoeff(), l_err = (r.scale - scale.row(0).transpose()).cwiseAbs().maxCoeff() / scale.maxCoeff();
    std::cout << "  " << name << ": singular " << sci(s_err) << ", variance " << sci(e_err) << ", ratio " << sci(q_err) << ", components " << sci(c_err) << ", scores " << sci(t_err) << ", mean " << sci(m_err) << ", scale " << sci(l_err) << "\n";
    EXPECT(s_err < 1e-12 && e_err < 1e-12 && q_err < 1e-12 && c_err < 1e-9 && t_err < 1e-9 && m_err < 1e-12 && l_err < 1e-12, name + ": equal to numpy's SVD, the sign convention included");
    // reconstruction at the golden's two k (the golden lists them after the table's arrays), then at every k
    const MatrixXd ks = g.mat(name + "_ks");
    for (int j = 0; j < ks.cols(); ++j) {
      const int k = static_cast<int>(ks(0, j));
      const double e = relErr(reconstruct(r.scores, r.components, r.mean, r.scale, k), g.mat(name + "_recon" + std::to_string(k)));
      EXPECT(e < 1e-9, name + ": the reconstruction at k = " + std::to_string(k) + " equals numpy's (" + sci(e) + ")");
    }
    for (int k = 0; k <= rank; ++k) {
      const MatrixXd Xk = reconstruct(r.scores, r.components, r.mean, r.scale, k);
      if (k == 0) EXPECT((Xk.rowwise() - r.mean.transpose()).cwiseAbs().maxCoeff() == 0, name + ": k = 0 is the mean");
      if (k == rank) EXPECT((Xk - X).cwiseAbs().maxCoeff() < 1e-10 * std::max(1.0, X.cwiseAbs().maxCoeff()), name + ": full rank returns the table");
    }
    // orthonormal rows, descending singular values, the shares add up to one at full rank
    const MatrixXd G = r.components * r.components.transpose();
    EXPECT((G - MatrixXd::Identity(rank, rank)).cwiseAbs().maxCoeff() < 1e-12, name + ": the components are orthonormal");
    bool sign_ok = true, desc = true;
    for (int c = 0; c < rank; ++c) {
      Eigen::Index at = 0;
      r.components.row(c).cwiseAbs().maxCoeff(&at);
      sign_ok = sign_ok && r.components(c, at) > 0;
      if (c > 0) desc = desc && r.singularValues[c] <= r.singularValues[c - 1];
    }
    EXPECT(sign_ok, name + ": each component's largest loading is positive (the fixed sign convention)");
    EXPECT(desc && std::abs(r.explainedVarianceRatio.sum() - 1.0) < 1e-12, name + ": descending, the shares add up to 1");
    // the scores are the centred (scaled) table on the components
    MatrixXd Xc = (X.rowwise() - r.mean.transpose()).array().rowwise() / r.scale.transpose().array();
    EXPECT((Xc * r.components.transpose() - r.scores).cwiseAbs().maxCoeff() < 1e-9 * std::max(1.0, r.scores.cwiseAbs().maxCoeff()), name + ": scores = centred table on the components");
  }
  // the best rank-k fit: the error falls with k and equals the discarded singular values
  {
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    MatrixXd X(20, 9);
    for (auto& v : X.reshaped()) v = nd(rng);
    const PcaResult r = pca(X);
    double prev = 1e300;
    bool mono = true;
    for (int k = 0; k <= r.rank; ++k) {
      const double e = (reconstruct(r.scores, r.components, r.mean, r.scale, k) - X).squaredNorm();
      const double want = r.singularValues.tail(r.rank - k).squaredNorm();
      mono = mono && e <= prev + 1e-12 && std::abs(e - want) < 1e-9 * std::max(1.0, want);
      prev = e;
    }
    EXPECT(mono, "the squared error at k is the sum of the discarded squared singular values (Eckart-Young), falling with k");
    // agreement with the eigendecomposition of the covariance (n x n sample covariance of the columns, p = 9)
    const MatrixXd Xc = X.rowwise() - X.colwise().mean();
    const MatrixXd cov = Xc.transpose() * Xc / static_cast<double>(X.rows() - 1);
    const nxr::signal::Leading L = nxr::signal::leading(cov.cast<std::complex<double>>(), 5);
    double ve = 0, vv = 1e300;
    for (int c = 0; c < 5; ++c) {
      ve = std::max(ve, std::abs(L.values[c] - r.explainedVariance[c]) / r.explainedVariance[0]);
      vv = std::min(vv, std::abs((L.vectors.col(c).adjoint() * r.components.row(c).transpose().cast<std::complex<double>>())(0, 0)));
    }
    EXPECT(ve < 1e-12 && vv > 1 - 1e-9, "the components and variances equal the eigendecomposition of the table's covariance (" + sci(ve) + ", |<v, w>| >= " + sci(vv) + ")");
  }
  // a rank-deficient table: a column that is the sum of two others, and a repeated row
  {
    std::mt19937 rng(3);
    std::normal_distribution<double> nd;
    MatrixXd X(15, 5);
    for (auto& v : X.reshaped()) v = nd(rng);
    X.col(4) = X.col(0) + X.col(1);
    const PcaResult r = pca(X);
    EXPECT(r.rank == 4, "a feature that is the sum of two others: rank 4, not 5");
    MatrixXd Y(6, 3);
    for (auto& v : Y.reshaped()) v = nd(rng);
    EXPECT(pca(Y).rank == 3 && pca(Y.topRows(3)).rank == 2, "at most n - 1 components after centring (3 rows: 2)");
  }
  // refusals, each with its reason
  {
    MatrixXd X = MatrixXd::Random(10, 4);
    EXPECT(refused([&] { pca(X.topRows(1)); }), "fewer than two observations is refused");
    MatrixXd Y = X;
    Y(3, 2) = std::nan("");
    EXPECT(refused([&] { pca(Y); }), "a NaN is refused");
    EXPECT(refused([&] { pca(MatrixXd::Constant(8, 3, 2.0)); }), "a table with no variance is refused");
    MatrixXd Z = X;
    Z.col(1).setConstant(1.0);
    PcaOptions o;
    o.scale = Scale::UnitVariance;
    EXPECT(refused([&] { pca(Z, o); }), "a constant feature cannot be scaled to unit variance: refused");
    EXPECT(pca(Z).rank == 3, "(without scaling it is a zero column)");
    const PcaResult r = pca(X);
    EXPECT(refused([&] { reconstruct(r.scores, r.components, r.mean, r.scale, r.rank + 1); }) && refused([&] { reconstruct(r.scores, r.components, r.mean, r.scale, -1); }), "k outside 0 .. components held is refused");
  }
  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}

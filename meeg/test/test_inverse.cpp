// nxr::meeg inverse: the whitener, depth weights and the unconstrained kernel on synthetic data (full-rank; rank-deficient with an average reference and
// SSP-like projectors), against the closed form R G~^T (G~ R G~^T + I)^-1 W built another way, and the invariances the maths states.
#include <cmath>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/meeg/inverse.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
MatrixXd randn(int r, int c, std::mt19937& g) {
  std::normal_distribution<double> n;
  MatrixXd m(r, c);
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m(i, j) = n(g);
  return m;
}
struct Case { MatrixXd G, C; std::vector<std::string> types; };
Case make(bool deficient) {
  std::mt19937 g(7);
  const int nm = 6, ng = 6, ne = 8, N = 15;
  Case c;
  for (int i = 0; i < nm; ++i) c.types.push_back("MEG MAG");
  for (int i = 0; i < ng; ++i) c.types.push_back("MEG GRAD");
  for (int i = 0; i < ne; ++i) c.types.push_back("EEG");
  const int M = nm + ng + ne;
  c.G = randn(M, 3 * N, g);
  for (int i = 0; i < nm; ++i) c.G.row(i) *= 1e-13;
  for (int i = nm; i < nm + ng; ++i) c.G.row(i) *= 1e-12;
  for (int i = nm + ng; i < M; ++i) c.G.row(i) *= 1e-6;
  const MatrixXd A = randn(M, 3 * M, g);
  c.C = A * A.transpose() / (3.0 * M);
  for (int i = 0; i < M; ++i) c.C.row(i) *= (i < nm ? 1e-13 : i < nm + ng ? 1e-12 : 1e-6), c.C.col(i) *= (i < nm ? 1e-13 : i < nm + ng ? 1e-12 : 1e-6);
  if (deficient) {
    // the average reference of the EEG block, and two SSP vectors on the magnetometers
    MatrixXd P = MatrixXd::Identity(M, M);
    P.block(nm + ng, nm + ng, ne, ne) -= MatrixXd::Constant(ne, ne, 1.0 / ne);
    for (int k = 0; k < 2; ++k) {
      VectorXd v = VectorXd::Zero(M);
      for (int i = 0; i < nm; ++i) v(i) = std::sin(1.0 + k + i);
      if (k == 1) { VectorXd u = VectorXd::Zero(M); for (int i = 0; i < nm; ++i) u(i) = std::sin(1.0 + i); u.normalize(); v -= u * u.dot(v); }
      v.normalize();
      P = (MatrixXd::Identity(M, M) - v * v.transpose()) * P;
    }
    c.G = P * c.G;
    c.C = P * c.C * P.transpose();
  }
  return c;
}
double maxrel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); }
}  // namespace

int main() {
  // ── full rank ──
  {
    const Case c = make(false);
    KernelOptions o;
    const Kernel k = unconstrainedKernel(c.G, c.C, c.types, o);
    CHECK_EQ(k.whitenerRank, 20);
    CHECK_EQ(k.K.rows(), 45);
    CHECK_EQ(k.K.cols(), 20);
    CHECK(k.depthWeights.minCoeff() >= 1 - 1e-12 && k.depthWeights.maxCoeff() <= o.depth.limit + 1e-12);
    // the closed form of the minimum-norm estimate in whitened data: R Gt^T (Gt R Gt^T + I)^-1 W, R = Lambda diag(alpha^2 x 1_3)
    const MatrixXd Gt = k.W * c.G;
    VectorXd r(45);
    for (int s = 0; s < 15; ++s) r.segment(3 * s, 3).setConstant(k.lambda * k.depthWeights(s) * k.depthWeights(s));
    const MatrixXd R = r.asDiagonal();
    const MatrixXd K2 = R * Gt.transpose() * (Gt * R * Gt.transpose() + MatrixXd::Identity(20, 20)).inverse() * k.W;
    CHECK(maxrel(k.K, K2) < 1e-9);
    // the SNR sets Lambda: trace(Gt R Gt^T) / M = snr^2
    CHECK(std::abs((Gt * R * Gt.transpose()).trace() / 20 - o.snr * o.snr) < 1e-9);
    // the form of the whitener does not change the kernel
    KernelOptions p = o;
    p.whitener.form = WhitenerForm::Pca;
    const Kernel kp = unconstrainedKernel(c.G, c.C, c.types, p);
    CHECK(maxrel(kp.K, k.K) < 1e-9);
    // regularisation None: the block whitens the covariance to the identity; no cross-type covariance is used
    WhitenerOptions w;
    w.regularisation = Regularisation::None;
    const Whitener wn = whitener(c.C, c.types, w);
    MatrixXd I = wn.W * c.C * wn.W.transpose();
    for (int i = 0; i < 20; ++i) for (int j = 0; j < 20; ++j) {
      const bool same = c.types[static_cast<std::size_t>(i)] == c.types[static_cast<std::size_t>(j)];
      if (same) CHECK(std::abs(I(i, j) - (i == j ? 1.0 : 0.0)) < 1e-8);
    }
    // depth weighting off: all ones; a larger exponent never shrinks the spread
    DepthOptions off;
    off.enabled = false;
    CHECK(depthWeights(c.G, c.types, off).isOnes());
    DepthOptions d1, d2;
    d2.exponent = 1.0;
    CHECK(depthWeights(c.G, c.types, d2).maxCoeff() >= depthWeights(c.G, c.types, d1).maxCoeff() - 1e-12);
    // MNE-Python's conventions agree with Brainstorm's when depth is off, the types are not joined and the whitener is full rank
    KernelOptions m = mneKernelOptions();
    m.depth.enabled = false;
    m.whitener.jointTypes = false;   // the types joined or not is the one difference left
    KernelOptions b;
    b.depth.enabled = false;
    CHECK(maxrel(unconstrainedKernel(c.G, c.C, c.types, m).K, unconstrainedKernel(c.G, c.C, c.types, b).K) < 1e-9);
    // a diagonal covariance is detected and said so
    MatrixXd D = c.C.diagonal().asDiagonal();
    const Whitener wd = whitener(D, c.types);
    CHECK(!wd.notes.empty());
    // errors
    bool threw = false;
    try { unconstrainedKernel(c.G.leftCols(44), c.C, c.types); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { whitener(c.C, {"MEG"}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    MatrixXd bad = c.G;
    bad(0, 0) = std::nan("");
    threw = false;
    try { depthWeights(bad, c.types); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  // ── rank-deficient: the average reference (EEG) and two SSP vectors (magnetometers) ──
  {
    const Case c = make(true);
    const Kernel k = unconstrainedKernel(c.G, c.C, c.types, {});
    CHECK_EQ(k.whitenerRank, 20 - 1 - 2);
    // the null space is removed: the common EEG mode and the projected magnetometer patterns do not reach the sources
    VectorXd common = VectorXd::Zero(20);
    common.tail(8).setOnes();
    CHECK((k.K * common).cwiseAbs().maxCoeff() < 1e-8 * k.K.cwiseAbs().maxCoeff());
    // the two Lambda normalisations differ when the whitener is rank-deficient, and only then
    KernelOptions wr;
    wr.snrNormalisation = SnrNormalisation::WhitenerRank;
    const Kernel kr = unconstrainedKernel(c.G, c.C, c.types, wr);
    CHECK(std::abs(kr.lambda / k.lambda - 17.0 / 20.0) < 1e-9);
    // the kernel solves the same closed form
    const MatrixXd Gt = k.W * c.G;
    VectorXd r(45);
    for (int s = 0; s < 15; ++s) r.segment(3 * s, 3).setConstant(k.lambda * k.depthWeights(s) * k.depthWeights(s));
    const MatrixXd R = r.asDiagonal();
    const MatrixXd K2 = R * Gt.transpose() * (Gt * R * Gt.transpose() + MatrixXd::Identity(20, 20)).inverse() * k.W;
    CHECK(maxrel(k.K, K2) < 1e-8);
    KernelOptions p;
    p.whitener.form = WhitenerForm::Pca;
    const Kernel kp = unconstrainedKernel(c.G, c.C, c.types, p);
    CHECK_EQ(kp.W.rows(), 17);
    CHECK(maxrel(kp.K, k.K) < 1e-8);
  }
  return nxrtest::finish("test_inverse");
}

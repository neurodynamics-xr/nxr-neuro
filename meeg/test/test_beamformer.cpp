// nxr::meeg::truncatedWhitener, lcmvKernel, scoutSignFlip, regionComponent, regionComponentSign on matrices made here: the median rule on known eigenvalues, the truncation to the
// rank, the types' blocks, the fixed orientation (a row follows its orientation's sign; a source's gain scale cancels), the unit output variance of an unregularised kernel, the noise
// covariance's part (none in the kernel, except the diagonal rule), and the region's component against the principal component of explicit series.
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/meeg/beamformer.h"
#include "nxr/meeg/scouts.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
MatrixXd randomOrthogonal(int n, std::mt19937& rng) {
  std::normal_distribution<double> nd;
  MatrixXd A(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A(i, j) = nd(rng);
  return Eigen::HouseholderQR<MatrixXd>(A).householderQ();
}
bool throws(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;

  // ── the median rule: eigenvalues 16, 9, 4, 1 (singular values 4, 3, 2, 1, median 2.5) become 16, 9, 6.25, 6.25 ──
  {
    const MatrixXd U = randomOrthogonal(4, rng);
    const MatrixXd C = U * VectorXd((VectorXd(4) << 16, 9, 4, 1).finished()).asDiagonal() * U.transpose();
    const std::vector<std::string> one(4, "MEG");
    const TruncatedWhitening w = truncatedWhitener(C, one);
    const MatrixXd want = U * VectorXd((VectorXd(4) << 16, 9, 6.25, 6.25).finished()).asDiagonal() * U.transpose();
    CHECK(rel(w.C, want) < 1e-12);
    CHECK(rel(w.W, U * VectorXd((VectorXd(4) << 0.25, 1.0 / 3, 0.4, 0.4).finished()).asDiagonal() * U.transpose()) < 1e-12);
    CHECK(w.ranks.size() == 1 && w.ranks[0] == 4);
    // an odd count: singular values 4, 3, 1 have median 3
    const MatrixXd U3 = randomOrthogonal(3, rng);
    const TruncatedWhitening w3 = truncatedWhitener(U3 * VectorXd((VectorXd(3) << 16, 9, 1).finished()).asDiagonal() * U3.transpose(), std::vector<std::string>(3, "MEG"));
    CHECK(rel(w3.C, U3 * VectorXd((VectorXd(3) << 16, 9, 9).finished()).asDiagonal() * U3.transpose()) < 1e-12);
    // none: the inverse square root; ridge: 1 / (s + sqrt(0.1 x the largest eigenvalue))
    TruncatedOptions none; none.regularisation = TruncatedRegularisation::None;
    CHECK(rel(truncatedWhitener(C, one, none).W * C * truncatedWhitener(C, one, none).W, MatrixXd::Identity(4, 4)) < 1e-12);
    TruncatedOptions ridge; ridge.regularisation = TruncatedRegularisation::Ridge;
    const VectorXd rw = (VectorXd(4) << 4, 3, 2, 1).finished().array() + std::sqrt(1.6);
    CHECK(rel(truncatedWhitener(C, one, ridge).W, U * rw.cwiseInverse().asDiagonal() * U.transpose()) < 1e-12);
  }
  // ── truncation to the rank, and the types' blocks (no cross-type covariance) ──
  {
    MatrixXd A(5, 3);
    for (int i = 0; i < 5; ++i) for (int j = 0; j < 3; ++j) A(i, j) = nd(rng);
    const MatrixXd C = A * A.transpose();                         // rank 3 of 5
    const TruncatedWhitening w = truncatedWhitener(C, std::vector<std::string>(5, "MEG"));
    CHECK(w.ranks[0] == 3);
    MatrixXd B(6, 6);
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) B(i, j) = nd(rng);
    const MatrixXd C2 = B * B.transpose();
    const std::vector<std::string> two = {"MEG MAG", "MEG GRAD", "MEG MAG", "MEG GRAD", "MEG MAG", "MEG GRAD"};
    const TruncatedWhitening w2 = truncatedWhitener(C2, two);
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) if (two[static_cast<std::size_t>(i)] != two[static_cast<std::size_t>(j)]) CHECK(w2.W(i, j) == 0 && w2.C(i, j) == 0);
    CHECK(w2.ranks.size() == 2);
    CHECK(throws([&] { truncatedWhitener(C2, std::vector<std::string>(5, "MEG")); }));
  }
  // ── the kernel: fixed orientation ──
  {
    const int M = 12, N = 20, T = 4000;
    MatrixXd G(M, 3 * N), O(N, 3), X(M, T), Xn(M, T);
    for (int i = 0; i < M; ++i) for (int j = 0; j < 3 * N; ++j) G(i, j) = nd(rng);
    for (int k = 0; k < N; ++k) { for (int j = 0; j < 3; ++j) O(k, j) = nd(rng); O.row(k).normalize(); }
    for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) { X(i, t) = nd(rng); Xn(i, t) = nd(rng) * (1 + 0.1 * i); }
    X.row(0) += 3.0 * X.row(1);
    auto cov = [](MatrixXd Y) { Y.colwise() -= Y.rowwise().mean(); return MatrixXd(Y * Y.transpose() / static_cast<double>(Y.cols() - 1)); };
    const MatrixXd Cd = cov(X), Cn = cov(Xn);
    const std::vector<std::string> types(M, "MEG");
    LcmvOptions none; none.covariance.regularisation = TruncatedRegularisation::None;
    const LcmvKernel k = lcmvKernel(G, O, Cd, Cn, types, none);
    // unregularised and full rank: each source's output has unit variance on the data covariance, and unit whitened gain
    CHECK(((k.K * Cd * k.K.transpose()).diagonal().array() - 1.0).abs().maxCoeff() < 1e-10);
    // the classical form: g^T C^-1 / sqrt(g^T C^-1 g)
    const MatrixXd Ci = Cd.inverse();
    for (int s = 0; s < N; ++s) {
      const VectorXd l = G.middleCols(3 * s, 3) * O.row(s).transpose();
      const Eigen::RowVectorXd w = (l.transpose() * Ci) / std::sqrt(l.dot(Ci * l));
      CHECK(rel(k.K.row(s), w) < 1e-9);
    }
    // the sign follows the orientation; a scale of the orientation (or of the source's gain) cancels
    MatrixXd O2 = O;
    O2.row(3) = -O.row(3);
    O2.row(5) = 7.0 * O.row(5);
    const LcmvKernel k2 = lcmvKernel(G, O2, Cd, Cn, types, none);
    CHECK(rel(k2.K.row(3), -k.K.row(3)) < 1e-12);
    CHECK(rel(k2.K.row(5), k.K.row(5)) < 1e-12);
    // the noise covariance does not enter the kernel (only its whitener is returned) ...
    const LcmvKernel k3 = lcmvKernel(G, O, Cd, 4.0 * Cn + MatrixXd::Identity(M, M), types, none);
    CHECK(rel(k3.K, k.K) < 1e-12);
    CHECK(rel(k3.noiseWhitener, k.noiseWhitener) > 1e-3);
    // ... except a diagonal one, which whitens both covariances by their diagonal (the reference's rule)
    const LcmvKernel k4 = lcmvKernel(G, O, Cd, MatrixXd(Cn.diagonal().asDiagonal()), types);
    CHECK(!k4.notes.empty());
    CHECK(rel(k4.dataWhitener, MatrixXd(Cd.diagonal().cwiseSqrt().cwiseInverse().asDiagonal())) < 1e-12);
    // the median rule (the default) differs from none on a covariance whose eigenvalues spread
    const LcmvKernel k5 = lcmvKernel(G, O, Cd, Cn, types);
    CHECK(k5.notes.empty());
    CHECK(rel(k5.K, k.K) > 1e-3);
    // refusals: shapes and a zero orientation
    MatrixXd Oz = O;
    Oz.row(2).setZero();
    CHECK(throws([&] { lcmvKernel(G, Oz, Cd, Cn, types); }));
    CHECK(throws([&] { lcmvKernel(G.leftCols(3 * N - 1), O, Cd, Cn, types); }));
    CHECK(throws([&] { lcmvKernel(G, O.topRows(N - 1), Cd, Cn, types); }));
  }
  // ── the region's component ──
  {
    // the flip: the majority keeps its sign
    MatrixXd O(5, 3);
    O << 0, 0, 1, 0.1, 0, 0.9, 0, 0.2, -1, 0.1, 0.1, 1, 0, 0, 0.8;
    const VectorXd f = scoutSignFlip(O);
    CHECK(f(0) == 1 && f(1) == 1 && f(2) == -1 && f(3) == 1 && f(4) == 1);
    CHECK(scoutSignFlip(-O) == f);
    CHECK(scoutSignFlip(O.topRows(1)).size() == 1 && scoutSignFlip(O.topRows(1))(0) == 1);
    // against the principal component of the series themselves
    const int M = 8, n = 5, T = 600;
    MatrixXd K(n, M), X(M, T);
    for (int i = 0; i < n; ++i) for (int j = 0; j < M; ++j) K(i, j) = nd(rng);
    for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) X(i, t) = nd(rng) + std::sin(0.05 * t * (i + 1)) + 0.3 * i;
    MatrixXd Xc = X;
    Xc.colwise() -= X.rowwise().mean();
    const MatrixXd C = Xc * Xc.transpose() / (T - 1);
    const RegionComponent rc = regionComponent(K, C, f);
    MatrixXd F = f.asDiagonal() * K * X;
    F.colwise() -= F.rowwise().mean();
    const Eigen::JacobiSVD<MatrixXd> svd(F, Eigen::ComputeThinU);
    const VectorXd u = svd.matrixU().col(0);
    CHECK(rel(rc.component.cwiseAbs(), (u.transpose() * f.asDiagonal() * K).cwiseAbs()) < 1e-9);
    CHECK(std::abs(rc.explained - svd.singularValues()(0) / svd.singularValues().sum()) < 1e-9);
    Eigen::Index pk;
    u.cwiseAbs().maxCoeff(&pk);
    CHECK(rc.peakSource == pk);
    // the sign from the extremes
    CHECK(regionComponentSign(3, -1, 0, 2, -5, 0) == -1);
    CHECK(regionComponentSign(1, -3, 0, 2, -5, 0) == 1);
    CHECK(regionComponentSign(4, 0, 1, 1, -1, 0) == 1);     // a tie (up = down): +1
    CHECK(throws([&] { regionComponent(K, C.topLeftCorner(M - 1, M - 1), f); }));
    CHECK(throws([&] { regionComponent(K, MatrixXd::Zero(M, M), f); }));
  }
  return nxrtest::finish("test_beamformer");
}

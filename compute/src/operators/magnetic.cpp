// nxr::operators — the magnetic Laplacian, its eigenpairs and the winding of a complex vertex field (magnetic.h).
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"

#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <Spectra/HermEigsSolver.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "nxr/core/errors.h"

namespace nxr::operators {

namespace {
using nxr::core::Error;
using nxr::core::ErrorCode;

// (L − σI)⁻¹ applied to a vector: the largest eigenvalues of it are L's smallest (shift-invert), factorised once (Eigen's complex LDLᵀ)
struct ShiftInvert {
  using Scalar = Complex;
  Eigen::SimplicialLDLT<Hermitian> factor;
  Eigen::Index n = 0;
  ShiftInvert(const Hermitian& L, double sigma) : n(L.rows()) {
    Hermitian I(n, n);
    I.setIdentity();
    factor.compute(L - Complex(sigma) * I);
    if (factor.info() != Eigen::Success) throw Error(ErrorCode::InternalError, "hermitianEigs: L − σI could not be factorised (σ is not below the spectrum?)");
  }
  Eigen::Index rows() const { return n; }
  Eigen::Index cols() const { return n; }
  void perform_op(const Complex* x, Complex* y) const {
    Eigen::Map<const Eigen::VectorXcd> in(x, n);
    Eigen::Map<Eigen::VectorXcd> out(y, n);
    out = factor.solve(in);
  }
};
// (L − σI + c V Vᴴ)⁻¹ applied to a vector, through the same factorisation (Woodbury): the found eigenvectors V pushed up by c, so a second solve finds
// what the first missed (Spectra's Lanczos can miss a partner of an exactly degenerate cluster)
struct Deflated {
  using Scalar = Complex;
  const ShiftInvert& base;
  Eigen::MatrixXcd Y;                                   // A⁻¹ V, A = L − σI
  Eigen::PartialPivLU<Eigen::MatrixXcd> small;          // I/c + Vᴴ A⁻¹ V
  Deflated(const ShiftInvert& b, const Eigen::MatrixXcd& V, double c) : base(b), Y(b.n, V.cols()) {
    for (Eigen::Index j = 0; j < V.cols(); ++j) Y.col(j) = b.factor.solve(V.col(j));
    small.compute(Eigen::MatrixXcd::Identity(V.cols(), V.cols()) / c + V.adjoint() * Y);
  }
  Eigen::Index rows() const { return base.n; }
  Eigen::Index cols() const { return base.n; }
  void perform_op(const Complex* x, Complex* y) const {
    Eigen::Map<const Eigen::VectorXcd> in(x, base.n);
    Eigen::Map<Eigen::VectorXcd> out(y, base.n);
    out = base.factor.solve(in);
    out -= Y * small.solve(Y.adjoint() * in);
  }
};
template <class Op>
std::pair<Eigen::VectorXd, Eigen::MatrixXcd> lowest(Op& op, Eigen::Index n, int k, double sigma, double tol, int& nconv) {
  const Eigen::Index ncv = std::min<Eigen::Index>(n, std::max<Eigen::Index>(2 * k + 1, k + 20));
  Spectra::HermEigsSolver<Op> es(op, k, ncv);
  es.init();
  nconv = static_cast<int>(es.compute(Spectra::SortRule::LargestMagn, 1000, tol));
  const Eigen::VectorXd mu = es.eigenvalues();
  Eigen::VectorXd lam(mu.size());
  for (Eigen::Index i = 0; i < mu.size(); ++i) lam(i) = sigma + 1.0 / mu(i);
  return {lam, es.eigenvectors()};
}

// unit norm, the largest component real and positive (a vector's global phase is arbitrary; this fixes it)
void fixPhase(Eigen::MatrixXcd& V) {
  for (Eigen::Index j = 0; j < V.cols(); ++j) {
    auto col = V.col(j);
    col /= col.norm();
    Eigen::Index big = 0;
    col.cwiseAbs().maxCoeff(&big);
    col *= std::conj(col(big)) / std::abs(col(big));
  }
}
void checkK(Eigen::Index n, int k) {
  if (k < 1 || k >= n) throw Error(ErrorCode::EigensolveInvalidK, "hermitianEigs: k = " + std::to_string(k) + " outside [1, " + std::to_string(n - 1) + "]");
}
}  // namespace

Normalise normaliseOf(const std::string& name) {
  if (name == "none") return Normalise::None;
  if (name == "symmetric") return Normalise::Symmetric;
  throw std::invalid_argument("normalise '" + name + "': none or symmetric");
}

Hermitian magneticLaplacian(int n, const Edges& edges, const std::vector<Complex>& W, Normalise normalise) {
  if (n < 1) throw std::invalid_argument("magneticLaplacian: no vertex");
  if (static_cast<std::size_t>(edges.rows()) != W.size()) throw std::invalid_argument("magneticLaplacian: " + std::to_string(edges.rows()) + " edges, " + std::to_string(W.size()) + " weights");
  std::vector<double> D(static_cast<std::size_t>(n), 0.0);
  std::set<std::pair<int, int>> seen;
  std::vector<Eigen::Triplet<Complex>> t;
  t.reserve(static_cast<std::size_t>(2 * edges.rows() + n));
  for (Eigen::Index e = 0; e < edges.rows(); ++e) {
    const int u = edges(e, 0), v = edges(e, 1);
    if (u < 0 || v < 0 || u >= n || v >= n) throw std::invalid_argument("magneticLaplacian: edge " + std::to_string(e) + " outside [0, " + std::to_string(n) + ")");
    if (u == v) throw std::invalid_argument("magneticLaplacian: edge " + std::to_string(e) + " is a self-loop");
    if (!seen.insert({std::min(u, v), std::max(u, v)}).second) throw std::invalid_argument("magneticLaplacian: edge (" + std::to_string(u) + ", " + std::to_string(v) + ") given twice");
    const Complex w = W[static_cast<std::size_t>(e)];
    if (!std::isfinite(w.real()) || !std::isfinite(w.imag())) throw std::invalid_argument("magneticLaplacian: edge " + std::to_string(e) + " has a non-finite weight");
    t.emplace_back(u, v, -w);
    t.emplace_back(v, u, -std::conj(w));
    D[static_cast<std::size_t>(u)] += std::abs(w);
    D[static_cast<std::size_t>(v)] += std::abs(w);
  }
  for (int u = 0; u < n; ++u) t.emplace_back(u, u, Complex(D[static_cast<std::size_t>(u)], 0.0));
  Hermitian L(n, n);
  L.setFromTriplets(t.begin(), t.end());
  if (normalise == Normalise::Symmetric) {
    Eigen::VectorXcd s(n);
    for (int u = 0; u < n; ++u) s(u) = D[static_cast<std::size_t>(u)] > 0 ? 1.0 / std::sqrt(D[static_cast<std::size_t>(u)]) : 0.0;
    L = s.asDiagonal() * L * s.asDiagonal();
  }
  L.makeCompressed();
  return L;
}

Hermitian transportLaplacian(int n, const Edges& edges, const Eigen::VectorXd& w, const Eigen::VectorXd& theta, const std::vector<Eigen::MatrixXd>& R, Normalise normalise) {
  const Eigen::Index E = edges.rows();
  if (n < 1 || w.size() != E || theta.size() != E || static_cast<Eigen::Index>(R.size()) != E || E == 0)
    throw std::invalid_argument("transportLaplacian: " + std::to_string(E) + " edges, " + std::to_string(w.size()) + " weights, " + std::to_string(theta.size()) + " lags, " +
                                std::to_string(R.size()) + " rotations");
  const Eigen::Index b = R[0].rows();
  std::vector<double> D(static_cast<std::size_t>(n), 0.0);
  std::set<std::pair<int, int>> seen;
  std::vector<Eigen::Triplet<Complex>> t;
  for (Eigen::Index e = 0; e < E; ++e) {
    const int u = edges(e, 0), v = edges(e, 1);
    if (u < 0 || v < 0 || u >= n || v >= n || u == v) throw std::invalid_argument("transportLaplacian: edge " + std::to_string(e) + " outside the vertices or a self-loop");
    if (!seen.insert({std::min(u, v), std::max(u, v)}).second) throw std::invalid_argument("transportLaplacian: edge (" + std::to_string(u) + ", " + std::to_string(v) + ") given twice");
    if (!(w(e) >= 0) || !std::isfinite(w(e)) || !std::isfinite(theta(e))) throw std::invalid_argument("transportLaplacian: edge " + std::to_string(e) + " has a negative or non-finite weight or lag");
    if (R[static_cast<std::size_t>(e)].rows() != b || R[static_cast<std::size_t>(e)].cols() != b) throw std::invalid_argument("transportLaplacian: rotation " + std::to_string(e) + " is not " + std::to_string(b) + " x " + std::to_string(b));
    const Complex ph = std::polar(w(e), theta(e));
    for (Eigen::Index i = 0; i < b; ++i)
      for (Eigen::Index j = 0; j < b; ++j) {
        const Complex x = ph * R[static_cast<std::size_t>(e)](i, j);
        if (x == Complex(0, 0)) continue;
        t.emplace_back(static_cast<int>(u * b + i), static_cast<int>(v * b + j), -x);
        t.emplace_back(static_cast<int>(v * b + j), static_cast<int>(u * b + i), -std::conj(x));
      }
    D[static_cast<std::size_t>(u)] += w(e);
    D[static_cast<std::size_t>(v)] += w(e);
  }
  for (int u = 0; u < n; ++u) for (Eigen::Index i = 0; i < b; ++i) t.emplace_back(static_cast<int>(u * b + i), static_cast<int>(u * b + i), Complex(D[static_cast<std::size_t>(u)], 0));
  Hermitian L(static_cast<Eigen::Index>(n) * b, static_cast<Eigen::Index>(n) * b);
  L.setFromTriplets(t.begin(), t.end());
  if (normalise == Normalise::Symmetric) {
    Eigen::VectorXcd s(static_cast<Eigen::Index>(n) * b);
    for (int u = 0; u < n; ++u) for (Eigen::Index i = 0; i < b; ++i) s(u * b + i) = D[static_cast<std::size_t>(u)] > 0 ? 1.0 / std::sqrt(D[static_cast<std::size_t>(u)]) : 0.0;
    L = s.asDiagonal() * L * s.asDiagonal();
  }
  L.makeCompressed();
  return L;
}

Hermitian blockMagneticLaplacian(int n, const Edges& edges, const std::vector<Eigen::MatrixXcd>& blocks, Normalise normalise) {
  const Eigen::Index E = edges.rows();
  if (n < 1 || E == 0 || static_cast<Eigen::Index>(blocks.size()) != E)
    throw std::invalid_argument("blockMagneticLaplacian: " + std::to_string(E) + " edges, " + std::to_string(blocks.size()) + " blocks");
  const Eigen::Index b = blocks[0].rows();
  std::vector<double> D(static_cast<std::size_t>(n), 0.0);
  std::set<std::pair<int, int>> seen;
  std::vector<Eigen::Triplet<Complex>> t;
  for (Eigen::Index e = 0; e < E; ++e) {
    const int u = edges(e, 0), v = edges(e, 1);
    const Eigen::MatrixXcd& W = blocks[static_cast<std::size_t>(e)];
    if (u < 0 || v < 0 || u >= n || v >= n || u == v) throw std::invalid_argument("blockMagneticLaplacian: edge " + std::to_string(e) + " outside the vertices or a self-loop");
    if (!seen.insert({std::min(u, v), std::max(u, v)}).second) throw std::invalid_argument("blockMagneticLaplacian: edge (" + std::to_string(u) + ", " + std::to_string(v) + ") given twice");
    if (W.rows() != b || W.cols() != b) throw std::invalid_argument("blockMagneticLaplacian: block " + std::to_string(e) + " is not " + std::to_string(b) + " x " + std::to_string(b));
    if (!W.allFinite()) throw std::invalid_argument("blockMagneticLaplacian: block " + std::to_string(e) + " has a non-finite entry");
    const double norm = Eigen::JacobiSVD<Eigen::MatrixXcd>(W).singularValues()(0);
    for (Eigen::Index i = 0; i < b; ++i)
      for (Eigen::Index j = 0; j < b; ++j) {
        const Complex x = W(i, j);
        if (x == Complex(0, 0)) continue;
        t.emplace_back(static_cast<int>(u * b + i), static_cast<int>(v * b + j), -x);
        t.emplace_back(static_cast<int>(v * b + j), static_cast<int>(u * b + i), -std::conj(x));
      }
    D[static_cast<std::size_t>(u)] += norm;
    D[static_cast<std::size_t>(v)] += norm;
  }
  for (int u = 0; u < n; ++u) for (Eigen::Index i = 0; i < b; ++i) t.emplace_back(static_cast<int>(u * b + i), static_cast<int>(u * b + i), Complex(D[static_cast<std::size_t>(u)], 0));
  Hermitian L(static_cast<Eigen::Index>(n) * b, static_cast<Eigen::Index>(n) * b);
  L.setFromTriplets(t.begin(), t.end());
  if (normalise == Normalise::Symmetric) {
    Eigen::VectorXcd s(static_cast<Eigen::Index>(n) * b);
    for (int u = 0; u < n; ++u) for (Eigen::Index i = 0; i < b; ++i) s(u * b + i) = D[static_cast<std::size_t>(u)] > 0 ? 1.0 / std::sqrt(D[static_cast<std::size_t>(u)]) : 0.0;
    L = s.asDiagonal() * L * s.asDiagonal();
  }
  L.makeCompressed();
  return L;
}

Eigenpairs hermitianEigs(const Hermitian& L, int k, double sigma, double tol) {
  const Eigen::Index n = L.rows();
  checkK(n, k);
  ShiftInvert op(L, sigma);
  int nconv = 0;
  auto [lam, V] = lowest(op, n, k, sigma, tol, nconv);
  if (nconv < k) throw Error(ErrorCode::EigensolveNotConverged, "hermitianEigs: " + std::to_string(nconv) + " of " + std::to_string(k) + " eigenpairs converged");
  // CLOSE THE CLUSTERS: deflate what was found and solve again; an eigenvalue below the k-th found is one the first solve missed (a degenerate
  // partner): kept, and again, until none is (at most k rounds)
  for (int round = 0; round < k; ++round) {
    const double top = lam.maxCoeff(), c = 2 * (top - sigma) + 1;
    Deflated d(op, V, c);
    int nc = 0;
    auto [lam2, V2] = lowest(d, n, std::min<int>(k, static_cast<int>(n - V.cols() - 1)), sigma, tol, nc);
    std::vector<Eigen::Index> missed;
    for (Eigen::Index i = 0; i < lam2.size(); ++i) if (lam2(i) < top - tol * std::max(1.0, std::abs(top))) missed.push_back(i);
    if (missed.empty()) break;
    Eigen::VectorXd l3(lam.size() + static_cast<Eigen::Index>(missed.size()));
    Eigen::MatrixXcd V3(n, l3.size());
    l3.head(lam.size()) = lam;
    V3.leftCols(V.cols()) = V;
    for (std::size_t j = 0; j < missed.size(); ++j) { l3(lam.size() + static_cast<Eigen::Index>(j)) = lam2(missed[j]); V3.col(V.cols() + static_cast<Eigen::Index>(j)) = V2.col(missed[j]); }
    // keep the lowest k, the vectors re-orthonormalised within the kept set (a missed partner is orthogonal to the found ones in exact arithmetic)
    std::vector<Eigen::Index> order(static_cast<std::size_t>(l3.size()));
    for (Eigen::Index i = 0; i < l3.size(); ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return l3(a) < l3(b); });
    lam.resize(k);
    V.resize(n, k);
    for (int i = 0; i < k; ++i) { lam(i) = l3(order[static_cast<std::size_t>(i)]); V.col(i) = V3.col(order[static_cast<std::size_t>(i)]); }
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(V);
    const Eigen::MatrixXcd Q = qr.householderQ() * Eigen::MatrixXcd::Identity(n, k);
    // Rayleigh–Ritz in the kept span: the exact eigenpairs of L restricted to it
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> rr(Q.adjoint() * (L * Q));
    lam = rr.eigenvalues();
    V = Q * rr.eigenvectors();
  }
  std::vector<Eigen::Index> order(static_cast<std::size_t>(lam.size()));
  for (Eigen::Index i = 0; i < lam.size(); ++i) order[static_cast<std::size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return lam(a) < lam(b); });
  Eigenpairs out;
  out.values.resize(k);
  out.vectors.resize(n, k);
  for (int i = 0; i < k; ++i) { out.values(i) = lam(order[static_cast<std::size_t>(i)]); out.vectors.col(i) = V.col(order[static_cast<std::size_t>(i)]); }
  fixPhase(out.vectors);
  out.converged = nconv;
  return out;
}

Eigenpairs hermitianEigsDense(const Hermitian& L, int k) {
  const Eigen::Index n = L.rows();
  checkK(n, k);
  if (n > 4096) throw Error(ErrorCode::InvalidInput, "hermitianEigsDense: a test oracle, n ≤ 4096 (n = " + std::to_string(n) + ")");
  const Eigen::MatrixXcd A = Eigen::MatrixXcd(L);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
  Eigenpairs out;
  out.values = es.eigenvalues().head(k);
  out.vectors = es.eigenvectors().leftCols(k);
  fixPhase(out.vectors);
  out.converged = k;
  return out;
}

std::vector<int> winding(const Eigen::VectorXcd& x, const Faces& faces) {
  auto wrap = [](double d) { return std::remainder(d, 2 * M_PI); };   // into [−π, π]
  std::vector<int> out(static_cast<std::size_t>(faces.rows()));
  for (Eigen::Index f = 0; f < faces.rows(); ++f) {
    double s = 0;
    for (int c = 0; c < 3; ++c) {
      const Eigen::Index a = faces(f, c), b = faces(f, (c + 1) % 3);
      if (a < 0 || b < 0 || a >= x.size() || b >= x.size()) throw std::invalid_argument("winding: face " + std::to_string(f) + " has a vertex outside x");
      s += wrap(std::arg(x(b)) - std::arg(x(a)));
    }
    out[static_cast<std::size_t>(f)] = static_cast<int>(std::lround(s / (2 * M_PI)));
  }
  return out;
}

}  // namespace nxr::operators

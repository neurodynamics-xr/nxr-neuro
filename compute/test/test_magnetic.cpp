// nxr::algebra::congruence and nxr::magnetic (the magnetic Laplacian, its eigenpairs, the winding of a complex field), on synthetic data only:
//   congruence           against the dense A M Aᴴ;
//   hermitianEigs        a ring with flux α (exact λ_k = 2 − 2cos(2π(k + α)/n)), with α = 0 too (doubly degenerate pairs); a random graph with random
//                        lags against scipy's eigsh (golden: gen_magnetic_golden.py) and against the dense oracle;
//   the method           on an icosphere: a plane wave's own lags are fitted exactly (λ₁ ≈ 0, the mode's phase differences the lags, |x| uniform);
//                        a rotating wave (the phase winds round the z axis): λ₁ ≈ 0 and the winding finds its two cores (+1 and −1);
//                        noise on the lags: λ₁ (the frustration) grows with it; the symmetric normalisation keeps the fit;
//   refusals             a self-loop, an edge twice, a non-finite weight, k out of range.
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include "mesh_helpers.h"
#include "nxr/congruence.h"
#include "nxr/errors.h"
#include "nxr/magnetic.h"

using namespace nxr::magnetic;
using C = std::complex<double>;

namespace {
template <class F>
bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

// the icosphere's undirected edges, once each
Edges edgesOf(const std::vector<int32_t>& F) {
  std::set<std::pair<int, int>> s;
  for (std::size_t f = 0; f < F.size(); f += 3)
    for (int c = 0; c < 3; ++c) { const int a = F[f + c], b = F[f + (c + 1) % 3]; s.insert({std::min(a, b), std::max(a, b)}); }
  Edges E(static_cast<Eigen::Index>(s.size()), 2);
  Eigen::Index i = 0;
  for (const auto& [a, b] : s) { E(i, 0) = a; E(i, 1) = b; ++i; }
  return E;
}
// W_uv = z_u conj(z_v): the lags of a field z
std::vector<C> lagsOf(const Edges& E, const std::vector<C>& z) {
  std::vector<C> W(static_cast<std::size_t>(E.rows()));
  for (Eigen::Index e = 0; e < E.rows(); ++e) W[static_cast<std::size_t>(e)] = z[static_cast<std::size_t>(E(e, 0))] * std::conj(z[static_cast<std::size_t>(E(e, 1))]);
  return W;
}
// the worst |arg(x_u conj x_v) − arg W_uv| (wrapped) over the edges
double lagMisfit(const Edges& E, const Eigen::VectorXcd& x, const std::vector<C>& W) {
  double worst = 0;
  for (Eigen::Index e = 0; e < E.rows(); ++e) {
    const C fit = x(E(e, 0)) * std::conj(x(E(e, 1)));
    worst = std::max(worst, std::abs(std::remainder(std::arg(fit) - std::arg(W[static_cast<std::size_t>(e)]), 2 * M_PI)));
  }
  return worst;
}
}  // namespace

int main() {
  std::cout << "test_magnetic\n";
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;

  // CONGRUENCE against the dense A M Aᴴ
  {
    Eigen::MatrixXd A(30, 8);
    for (Eigen::Index i = 0; i < A.size(); ++i) A.data()[i] = nd(rng);
    Eigen::MatrixXcd B(8, 8);
    for (Eigen::Index i = 0; i < B.size(); ++i) B.data()[i] = C(nd(rng), nd(rng));
    const Eigen::MatrixXcd M = B * B.adjoint();
    const Eigen::MatrixXcd full = A.cast<C>() * M * A.transpose().cast<C>();
    nxr::algebra::Pairs P(40, 2);
    for (Eigen::Index p = 0; p < 40; ++p) { P(p, 0) = static_cast<int32_t>(rng() % 30); P(p, 1) = static_cast<int32_t>(rng() % 30); }
    const auto got = nxr::algebra::congruence(A, M, P);
    double worst = 0;
    for (Eigen::Index p = 0; p < 40; ++p) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(p)] - full(P(p, 0), P(p, 1))) / full.cwiseAbs().maxCoeff());
    EXPECT(worst < 1e-14, "congruence == the dense A M A^H on 40 pairs (worst " << worst << ")");
    EXPECT(throws([&] { nxr::algebra::congruence(A, Eigen::MatrixXcd::Zero(7, 7), P); }), "congruence refuses mismatched shapes");
  }

  // THE RING with flux α: exact spectrum; α = 0 has doubly degenerate pairs
  for (const double alpha : {0.3, 0.0}) {
    const int n = 2000, k = 7;
    Edges E(n, 2);
    std::vector<C> W(n);
    for (int v = 0; v < n; ++v) { E(v, 0) = v; E(v, 1) = (v + 1) % n; W[static_cast<std::size_t>(v)] = std::polar(1.0, 2 * M_PI * alpha / n); }
    const Hermitian L = magneticLaplacian(n, E, W, Normalise::None);
    const Eigenpairs ep = hermitianEigs(L, k);
    std::vector<double> want;
    for (int j = -k; j <= k; ++j) want.push_back(2 - 2 * std::cos(2 * M_PI * (j + alpha) / n));
    std::sort(want.begin(), want.end());
    double worst = 0, resid = 0;
    for (int i = 0; i < k; ++i) {
      worst = std::max(worst, std::abs(ep.values(i) - want[static_cast<std::size_t>(i)]) / want[static_cast<std::size_t>(k - 1)]);   // relative to the k-th
      resid = std::max(resid, (L * ep.vectors.col(i) - ep.values(i) * ep.vectors.col(i)).norm());
    }
    EXPECT(worst < 1e-6 && resid < 1e-8, "ring n=2000, flux " << alpha << ": the lowest " << k << " exact" << (alpha == 0 ? ", degenerate pairs included" : "") << " (worst, relative to the k-th, " << worst << ", residual " << resid << ")");
  }

  // EXACTLY DEGENERATE CLUSTERS: the icosphere's graph with no lags (W = 1): its lowest 25 come in groups (1, 3, 5, 7, 9 by the sphere's symmetry);
  // every partner found (against the dense oracle)
  {
    std::vector<double> Vs;
    std::vector<int32_t> Fs;
    icosphere(2, Vs, Fs);
    const Edges Es = edgesOf(Fs);
    const Hermitian L = magneticLaplacian(static_cast<int>(Vs.size() / 3), Es, std::vector<C>(static_cast<std::size_t>(Es.rows()), C(1, 0)), Normalise::None);
    const Eigenpairs it = hermitianEigs(L, 25), dense = hermitianEigsDense(L, 25);
    const double worst = (it.values - dense.values).cwiseAbs().maxCoeff();
    double resid = 0;
    for (int i = 0; i < 25; ++i) resid = std::max(resid, (L * it.vectors.col(i) - it.values(i) * it.vectors.col(i)).norm());
    EXPECT(worst < 1e-9 && resid < 1e-8, "icosphere graph, the lowest 25 (clusters of 1 3 5 7 9) == the dense oracle (worst " << worst << ", residual " << resid << ")");
  }

  // A RANDOM GRAPH with random lags: against scipy's eigsh (the golden) and the dense oracle
  {
    std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/magnetic_golden.txt");
    int n = 0, m = 0, k = 0;
    in >> n >> m >> k;
    Edges E(m, 2);
    std::vector<C> W(static_cast<std::size_t>(m));
    for (int e = 0; e < m; ++e) { double re, im; in >> E(e, 0) >> E(e, 1) >> re >> im; W[static_cast<std::size_t>(e)] = C(re, im); }
    Eigen::VectorXd ref(k);
    for (int i = 0; i < k; ++i) in >> ref(i);
    Eigen::MatrixXcd vref(n, k);
    for (int j = 0; j < k; ++j) for (int i = 0; i < n; ++i) { double re, im; in >> re >> im; vref(i, j) = C(re, im); }
    EXPECT(in.good() && n == 400 && k == 8, "the scipy golden read (" << n << " vertices, " << m << " edges, " << k << " eigenpairs)");
    const Hermitian L = magneticLaplacian(n, E, W, Normalise::None);
    const Eigenpairs ep = hermitianEigs(L, k), dense = hermitianEigsDense(L, k);
    const double vs_scipy = (ep.values - ref).cwiseAbs().maxCoeff(), vs_dense = (ep.values - dense.values).cwiseAbs().maxCoeff();
    double overlap = 1;                                    // |<ours, scipy's>| per eigenvector: 1 when the same up to the global phase
    for (int j = 0; j < k; ++j) overlap = std::min(overlap, std::abs(ep.vectors.col(j).dot(vref.col(j))) / vref.col(j).norm());
    EXPECT(vs_scipy < 1e-10 && vs_dense < 1e-10 && overlap > 1 - 1e-8,
           "random graph: the lowest 8 == scipy eigsh (" << vs_scipy << ") == dense (" << vs_dense << "); eigenvectors' overlap with scipy's " << overlap);
    EXPECT(ep.converged >= k, "random graph: " << ep.converged << " eigenpairs converged");
  }

  // THE METHOD on an icosphere (level 4, 2562 vertices)
  std::vector<double> V;
  std::vector<int32_t> F;
  icosphere(4, V, F);
  const int n = static_cast<int>(V.size() / 3);
  const Edges E = edgesOf(F);
  Faces Fm(static_cast<Eigen::Index>(F.size() / 3), 3);
  for (std::size_t i = 0; i < F.size(); ++i) Fm(static_cast<Eigen::Index>(i / 3), static_cast<Eigen::Index>(i % 3)) = F[i];
  auto field = [&](auto phase) { std::vector<C> z(static_cast<std::size_t>(n)); for (int v = 0; v < n; ++v) z[static_cast<std::size_t>(v)] = std::polar(1.0, phase(&V[3 * v])); return z; };
  // a plane wave: its own lags are fitted exactly
  const std::vector<C> plane = field([](const double* x) { return 2.0 * x[0] + 1.0 * x[1]; });
  const std::vector<C> Wp = lagsOf(E, plane);
  for (const Normalise nm : {Normalise::None, Normalise::Symmetric}) {
    const Eigenpairs ep = hermitianEigs(magneticLaplacian(n, E, Wp, nm), 3);
    const Eigen::VectorXd a = ep.vectors.col(0).cwiseAbs();
    EXPECT(std::abs(ep.values(0)) < 1e-9 && ep.values(1) > 1e-4 && lagMisfit(E, ep.vectors.col(0), Wp) < 1e-6 && (nm == Normalise::Symmetric || a.maxCoeff() - a.minCoeff() < 1e-8),
           "plane wave (" << (nm == Normalise::None ? "none" : "symmetric") << "): λ₁ " << ep.values(0) << ", λ₂ " << ep.values(1) << ", the mode's lags == the wave's (worst "
                          << lagMisfit(E, ep.vectors.col(0), Wp) << ")");
  }
  // a rotating wave: the phase winds round the z axis; fitted exactly, the winding finds its two cores
  {
    const std::vector<C> rot = field([](const double* x) { return std::atan2(x[1], x[0]); });
    const std::vector<C> Wr = lagsOf(E, rot);
    const Eigenpairs ep = hermitianEigs(magneticLaplacian(n, E, Wr, Normalise::None), 2);
    const std::vector<int> w = winding(ep.vectors.col(0), Fm);
    int plus = 0, minus = 0;
    for (std::size_t f = 0; f < w.size(); ++f) {
      if (w[f] == 0) continue;
      const double zc = (V[3 * F[3 * f] + 2] + V[3 * F[3 * f + 1] + 2] + V[3 * F[3 * f + 2] + 2]) / 3;
      if (w[f] == 1 && zc > 0.9) ++plus;
      else if (w[f] == -1 && zc < -0.9) ++minus;
      else { plus += 10; minus += 10; }   // any other winding is wrong
    }
    EXPECT(std::abs(ep.values(0)) < 1e-9 && plus == 1 && minus == 1, "rotating wave: λ₁ " << ep.values(0) << "; the winding finds one core at each pole (+1 north, −1 south)");
  }
  // lags that do not close round loops: the frustration λ₁ grows with the noise on the plane wave's lags
  {
    std::vector<double> lam;
    for (const double noise : {0.0, 0.1, 0.3, 0.6}) {
      std::mt19937 g(5);
      std::normal_distribution<double> e(0.0, 1.0);
      std::vector<C> W(Wp.size());
      for (std::size_t i = 0; i < W.size(); ++i) W[i] = Wp[i] * std::polar(1.0, noise * e(g));
      lam.push_back(hermitianEigs(magneticLaplacian(n, E, W, Normalise::Symmetric), 1).values(0));
    }
    EXPECT(lam[0] < 1e-9 && lam[1] > 1e-4 && lam[2] > lam[1] && lam[3] > lam[2],
           "noisy lags (0 · 0.1 · 0.3 · 0.6 rad): λ₁ " << lam[0] << " < " << lam[1] << " < " << lam[2] << " < " << lam[3]);
  }

  // THE BLOCK MAGNETIC LAPLACIAN: the scalar one for b = 1, the transported one for blocks w e^{iθ} R, PSD for arbitrary blocks, and the DATA-BLOCK route
  {
    std::mt19937 g(41);
    std::normal_distribution<double> z;
    const int nb = 40;
    Edges Eb(80, 2);
    std::set<std::pair<int, int>> used;
    for (int e = 0; e < 80;) { const int a = static_cast<int>(g() % nb), c = static_cast<int>(g() % nb); if (a == c || !used.insert({std::min(a, c), std::max(a, c)}).second) continue; Eb(e, 0) = a; Eb(e, 1) = c; ++e; }
    std::vector<C> w1(80);
    std::vector<Eigen::MatrixXcd> b1(80), b3(80), r3(80);
    Eigen::VectorXd ww(80), th(80);
    std::vector<Eigen::MatrixXd> rots(80);
    for (int e = 0; e < 80; ++e) {
      w1[e] = C(z(g), z(g));
      b1[e] = Eigen::MatrixXcd::Constant(1, 1, w1[e]);
      Eigen::MatrixXcd B(3, 3);
      for (Eigen::Index i = 0; i < 9; ++i) B.data()[i] = C(z(g), z(g));
      b3[e] = B;
      ww(e) = std::abs(z(g)); th(e) = z(g);
      Eigen::Matrix3d Q = Eigen::Matrix3d::Random().householderQr().householderQ();
      if (Q.determinant() < 0) Q.col(0) *= -1;
      rots[e] = Q;
      r3[e] = std::polar(ww(e), th(e)) * Q.cast<C>();
    }
    const double d1 = (Eigen::MatrixXcd(blockMagneticLaplacian(nb, Eb, b1, Normalise::None)) - Eigen::MatrixXcd(magneticLaplacian(nb, Eb, w1, Normalise::None))).cwiseAbs().maxCoeff();
    const double d2 = (Eigen::MatrixXcd(blockMagneticLaplacian(nb, Eb, r3, Normalise::Symmetric)) - Eigen::MatrixXcd(transportLaplacian(nb, Eb, ww, th, rots, Normalise::Symmetric))).cwiseAbs().maxCoeff();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Eigen::MatrixXcd(blockMagneticLaplacian(nb, Eb, b3, Normalise::None)));
    EXPECT(d1 < 1e-12 && d2 < 1e-12 && es.eigenvalues().minCoeff() > -1e-10,
           "blockMagneticLaplacian: == magneticLaplacian for b = 1 (" << d1 << "), == transportLaplacian for blocks w e^{iθ} R (" << d2 << "), PSD for arbitrary 3 x 3 blocks (min eigenvalue "
           << es.eigenvalues().minCoeff() << ")");
  }
  // THE DATA-BLOCK ROUTE, NO ORIENTATION ASSUMED: currents of RANDOM orientation and sign a vertex (a_v, unit, real) with a lag map φ; each edge's block
  // is the pair's cross-spectrum a_u a_vᵀ e^{−i(φ_u − φ_v)}: the lowest mode is unique and IS the current map a_v e^{−iφ_v}, up to one global phase
  {
    std::mt19937 g(43);
    std::normal_distribution<double> z;
    std::vector<double> Vs;
    std::vector<int32_t> Fs;
    icosphere(3, Vs, Fs);
    const int ns = static_cast<int>(Vs.size() / 3);
    const Edges Es = edgesOf(Fs);
    std::vector<Eigen::Vector3d> a(ns);
    std::vector<double> ph(ns);
    for (int v = 0; v < ns; ++v) { a[v] = Eigen::Vector3d(z(g), z(g), z(g)).normalized(); ph[v] = 2.0 * Vs[3 * v] - 1.5 * Vs[3 * v + 2]; }
    std::vector<Eigen::MatrixXcd> blk(static_cast<std::size_t>(Es.rows()));
    for (Eigen::Index e = 0; e < Es.rows(); ++e) blk[e] = (a[Es(e, 0)] * a[Es(e, 1)].transpose()).cast<C>() * std::polar(1.0, -(ph[Es(e, 0)] - ph[Es(e, 1)]));
    const auto ep = hermitianEigs(blockMagneticLaplacian(ns, Es, blk, Normalise::None), 2);
    Eigen::VectorXcd truth(3 * ns);
    for (int v = 0; v < ns; ++v) truth.segment<3>(3 * v) = a[v].cast<C>() * std::polar(1.0, -ph[v]);
    truth.normalize();
    const double overlap = std::abs(truth.dot(ep.vectors.col(0)));
    EXPECT(std::abs(ep.values(0)) < 1e-9 && ep.values(1) > 1e-4 && overlap > 1 - 1e-9,
           "the data-block route, currents of random orientation and sign: λ₁ " << ep.values(0) << ", λ₂ " << ep.values(1) << " (unique); the mode == the current map up to a global phase (overlap "
           << overlap << ")");
  }

  // REFUSALS
  {
    Edges e1(1, 2); e1 << 0, 0;
    Edges e2(2, 2); e2 << 0, 1, 1, 0;
    Edges e3(1, 2); e3 << 0, 1;
    EXPECT(throws([&] { magneticLaplacian(3, e1, {C(1, 0)}, Normalise::None); }), "a self-loop refused");
    EXPECT(throws([&] { magneticLaplacian(3, e2, {C(1, 0), C(1, 0)}, Normalise::None); }), "an edge given twice refused");
    EXPECT(throws([&] { magneticLaplacian(3, e3, {C(NAN, 0)}, Normalise::None); }), "a non-finite weight refused");
    EXPECT(throws([&] { hermitianEigs(magneticLaplacian(3, e3, {C(1, 0)}, Normalise::None), 3); }), "k out of range refused");
    EXPECT(throws([] { normaliseOf("l2"); }), "an unknown normalisation refused");
  }
  std::cout << (g_failures ? "FAIL" : "PASS") << " test_magnetic (" << g_failures << " failures)\n";
  return g_failures ? 1 : 0;
}

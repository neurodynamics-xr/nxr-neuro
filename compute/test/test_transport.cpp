// nxr::transport (levi_civita, frames, compare), nxr::algebra::blockCongruence and nxr::magnetic::transportLaplacian, synthetic only:
//   the rotations: R n_v = n_u, orthogonal with det 1 (an antiparallel pair too); frames' R e1_v = e1_u; the blocks equal A M Aᴴ's;
//   A FOLDED SURFACE (z = A sin(2π x / Λ), walls up to 72° steep) carrying a current NORMAL to it with a travelling lag φ(x): the world-frame comparison
//   tr ⟨J_u J_vᴴ⟩ is off by π for pairs across a fold (normals pointing apart), the transported one tr(T R ᵀ) recovers every lag exactly (strength 1);
//   the transported magnetic Laplacian on the sheet's edges (w = 1, θ and R from the transport): a developable sheet has three parallel fields, so
//   λ₁…₃ = 0 and the normal component of each null vector carries the lags φ; ON A SPHERE (curved: the tangential fields have holonomy, the normal line is
//   carried onto itself) the lowest mode is UNIQUE and is the current, along the normal with its lags; refusals.
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <map>
#include <set>

#include "nxr/congruence.h"
#include "nxr/magnetic.h"
#include "nxr/transport.h"

using C = std::complex<double>;
static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_transport\n";
  std::mt19937 rng(31);
  std::normal_distribution<double> nd;

  // blockCongruence against A M Aᴴ
  {
    Eigen::MatrixXd A(3 * 6, 5);
    for (Eigen::Index i = 0; i < A.size(); ++i) A.data()[i] = nd(rng);
    Eigen::MatrixXcd B(5, 5);
    for (Eigen::Index i = 0; i < B.size(); ++i) B.data()[i] = C(nd(rng), nd(rng));
    const Eigen::MatrixXcd M = B * B.adjoint(), full = A.cast<C>() * M * A.transpose().cast<C>();
    nxr::algebra::Pairs P(4, 2);
    P << 0, 1, 2, 5, 3, 3, 5, 0;
    const auto blk = nxr::algebra::blockCongruence(A, M, P, 3);
    double worst = 0;
    for (Eigen::Index p = 0; p < 4; ++p) worst = std::max(worst, (blk[p] - full.block(3 * P(p, 0), 3 * P(p, 1), 3, 3)).cwiseAbs().maxCoeff());
    EXPECT(worst < 1e-12 * full.cwiseAbs().maxCoeff(), "blockCongruence == the 3 x 3 blocks of A M Aᴴ (worst " << worst << ")");
  }

  // the rotations
  {
    Eigen::MatrixXd N(4, 3);
    N << 0, 0, 1, 0.3, 0.1, 0.95, 0, 0, -1, 1, 0, 0;
    for (int i = 0; i < 4; ++i) N.row(i).normalize();
    nxr::transport::Edges E(3, 2);
    E << 1, 0, 2, 0, 3, 1;                                                   // (u, v): v's normal carried to u's
    const auto R = nxr::transport::leviCivita(N, E);
    double worst = 0;
    for (int e = 0; e < 3; ++e) {
      worst = std::max(worst, (R[e] * N.row(E(e, 1)).transpose() - N.row(E(e, 0)).transpose()).norm());
      worst = std::max(worst, (R[e].transpose() * R[e] - Eigen::Matrix3d::Identity()).norm() + std::abs(R[e].determinant() - 1));
    }
    Eigen::MatrixXd e1(4, 3), e2(4, 3);
    for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d n = N.row(i), p = std::abs(n.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
      p = (p - p.dot(n) * n).normalized();
      e1.row(i) = p; e2.row(i) = n.cross(p);
    }
    const auto Rf = nxr::transport::frames(e1, e2, N, E);
    double wf = 0;
    for (int e = 0; e < 3; ++e) wf = std::max(wf, (Rf[e] * e1.row(E(e, 1)).transpose() - e1.row(E(e, 0)).transpose()).norm() + (Rf[e] * N.row(E(e, 1)).transpose() - N.row(E(e, 0)).transpose()).norm());
    EXPECT(worst < 1e-12 && wf < 1e-12, "levi_civita: R n_v = n_u, a rotation (antiparallel normals too); frames: R carries v's frame onto u's");
  }

  // A FOLDED SHEET: z = A sin(2π x / Λ) on a grid, a current normal to it with lag φ = 2π x / Λw
  const int nx = 120, ny = 24;
  const double Lx = 6.0, Ly = 1.2, Lam = 1.0, Amp = 0.5 * Lam / M_PI * 3.0, Lw = 2.5;   // max slope 3: walls up to 72°
  const int n = nx * ny;
  Eigen::MatrixXd P(n, 3), N = Eigen::MatrixXd::Zero(n, 3);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) { const double x = Lx * i / (nx - 1), y = Ly * j / (ny - 1); P.row(j * nx + i) << x, y, Amp * std::sin(2 * M_PI * x / Lam); }
  std::vector<std::array<int, 3>> F;
  for (int j = 0; j + 1 < ny; ++j)
    for (int i = 0; i + 1 < nx; ++i) {
      const int a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      F.push_back({a, b, d}); F.push_back({a, d, c});
    }
  for (const auto& f : F) {
    const Eigen::Vector3d u = P.row(f[1]) - P.row(f[0]), w = P.row(f[2]) - P.row(f[0]), cr = u.cross(w);
    for (int k : f) N.row(k) += cr;
  }
  for (int v = 0; v < n; ++v) N.row(v).normalize();
  std::vector<double> phi(n);
  for (int v = 0; v < n; ++v) phi[v] = 2 * M_PI * P(v, 0) / Lw;
  auto Tof = [&](int u, int v) { return Eigen::MatrixXcd((N.row(u).transpose() * N.row(v)).cast<C>() * std::polar(1.0, -(phi[u] - phi[v]))); };   // ⟨J_u J_vᴴ⟩
  // pairs up to 1.5 Λ apart along x, same row: many cross a fold
  std::vector<std::pair<int, int>> pairs;
  for (int j = 0; j < ny; j += 3) for (int i = 0; i < nx; ++i) for (int k = 1; k <= 30 && i + k < nx; k += 3) pairs.push_back({j * nx + i + k, j * nx + i});
  {
    nxr::transport::Edges E(static_cast<Eigen::Index>(pairs.size()), 2);
    std::vector<Eigen::MatrixXcd> T;
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(pairs.size()));
    for (std::size_t p = 0; p < pairs.size(); ++p) { E(p, 0) = pairs[p].first; E(p, 1) = pairs[p].second; T.push_back(Tof(pairs[p].first, pairs[p].second)); }
    const auto R = nxr::transport::leviCivita(N, E);
    const auto c = nxr::transport::compare(T, R, ones, ones);
    std::vector<Eigen::Matrix3d> Id(pairs.size(), Eigen::Matrix3d::Identity());
    const auto world = nxr::transport::compare(T, Id, ones, ones);
    int piWorld = 0;
    double worst = 0, worstStrength = 0;
    for (std::size_t p = 0; p < pairs.size(); ++p) {
      const double truth = -(phi[pairs[p].first] - phi[pairs[p].second]);
      worst = std::max(worst, std::abs(std::remainder(c.lag(p) - truth, 2 * M_PI)));
      worstStrength = std::max(worstStrength, std::abs(c.strength(p) - 1));
      if (std::abs(std::remainder(world.lag(p) - truth, 2 * M_PI)) > M_PI / 2) ++piWorld;
    }
    EXPECT(piWorld > 0 && worst < 1e-12 && worstStrength < 1e-12,
           "a normal current on a folded sheet: the world-frame comparison is off by π for " << piWorld << " of " << pairs.size()
           << " pairs (across folds); the transported one recovers every lag (worst " << worst << " rad), strength 1");
  }
  // the transported magnetic Laplacian on the sheet's edges: its lowest mode is the current
  {
    std::set<std::pair<int, int>> es;
    for (const auto& f : F) for (int k = 0; k < 3; ++k) { const int a = f[k], b = f[(k + 1) % 3]; es.insert({std::min(a, b), std::max(a, b)}); }
    nxr::transport::Edges E(static_cast<Eigen::Index>(es.size()), 2);
    { Eigen::Index i = 0; for (auto [a, b] : es) { E(i, 0) = a; E(i, 1) = b; ++i; } }
    std::vector<Eigen::MatrixXcd> T;
    for (Eigen::Index e = 0; e < E.rows(); ++e) T.push_back(Tof(E(e, 0), E(e, 1)));
    const auto R = nxr::transport::leviCivita(N, E);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(E.rows());
    const auto c = nxr::transport::compare(T, R, ones, ones);
    std::vector<Eigen::MatrixXd> Rx(R.begin(), R.end());
    const auto L = nxr::magnetic::transportLaplacian(n, E, ones, c.lag, Rx, nxr::magnetic::Normalise::None);
    const auto ep = nxr::magnetic::hermitianEigs(L, 4);
    // a developable sheet has THREE parallel fields (normal, along and across the folds): λ₁ = λ₂ = λ₃ = 0, and each null vector's normal component
    // carries the current's lags (the phase map is determined; the vector's direction within the parallel fields is not, by this operator alone)
    double lagErr = 0;
    for (int m = 0; m < 3; ++m)
      for (Eigen::Index e = 0; e < E.rows(); ++e) {
        const C au = N.row(E(e, 0)).cast<C>().dot(ep.vectors.col(m).segment<3>(3 * E(e, 0))), av = N.row(E(e, 1)).cast<C>().dot(ep.vectors.col(m).segment<3>(3 * E(e, 1)));
        if (std::abs(au) < 1e-9 || std::abs(av) < 1e-9) continue;
        lagErr = std::max(lagErr, std::abs(std::remainder(std::arg(au * std::conj(av)) - (-(phi[E(e, 0)] - phi[E(e, 1)])), 2 * M_PI)));
      }
    EXPECT(ep.values.head(3).cwiseAbs().maxCoeff() < 1e-9 && ep.values(3) > 1e-6 && lagErr < 1e-6,
           "the transported magnetic Laplacian on the folded (developable) sheet: λ₁…₃ ≤ " << ep.values.head(3).cwiseAbs().maxCoeff() << " (three parallel fields), λ₄ "
           << ep.values(3) << "; the normal component of every null vector carries the current's lags (worst " << lagErr << " rad)");
  }
  // ON A CURVED SURFACE (a sphere) the tangential fields have holonomy, the normal line is carried onto itself: the lowest mode is UNIQUE and is the current
  {
    std::vector<Eigen::Vector3d> V;
    std::vector<std::array<int, 3>> Fs;
    {   // an icosphere, level 3
      const double t = (1 + std::sqrt(5.0)) / 2;
      V = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
      for (auto& v : V) v.normalize();
      Fs = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
      for (int l = 0; l < 3; ++l) {
        std::map<std::pair<int, int>, int> mid;
        auto m = [&](int a, int b) { const auto k = std::make_pair(std::min(a, b), std::max(a, b)); auto it = mid.find(k); if (it != mid.end()) return it->second; V.push_back((V[a] + V[b]).normalized()); return mid[k] = static_cast<int>(V.size()) - 1; };
        std::vector<std::array<int, 3>> nf;
        for (const auto& f : Fs) { const int a = m(f[0], f[1]), b = m(f[1], f[2]), c = m(f[2], f[0]); nf.push_back({f[0], a, c}); nf.push_back({f[1], b, a}); nf.push_back({f[2], c, b}); nf.push_back({a, b, c}); }
        Fs = nf;
      }
    }
    const int ns = static_cast<int>(V.size());
    Eigen::MatrixXd Ns(ns, 3);
    for (int v = 0; v < ns; ++v) Ns.row(v) = V[v].transpose();
    std::vector<double> ph(ns);
    for (int v = 0; v < ns; ++v) ph[v] = 2.0 * V[v].x() + V[v].z();          // a travelling lag across the sphere
    std::set<std::pair<int, int>> es;
    for (const auto& f : Fs) for (int k = 0; k < 3; ++k) { const int a = f[k], b = f[(k + 1) % 3]; es.insert({std::min(a, b), std::max(a, b)}); }
    nxr::transport::Edges E(static_cast<Eigen::Index>(es.size()), 2);
    { Eigen::Index i = 0; for (auto [a, b] : es) { E(i, 0) = a; E(i, 1) = b; ++i; } }
    std::vector<Eigen::MatrixXcd> T;
    for (Eigen::Index e = 0; e < E.rows(); ++e) T.push_back((Ns.row(E(e, 0)).transpose() * Ns.row(E(e, 1))).cast<C>() * std::polar(1.0, -(ph[E(e, 0)] - ph[E(e, 1)])));
    const auto R = nxr::transport::leviCivita(Ns, E);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(E.rows());
    const auto c = nxr::transport::compare(T, R, ones, ones);
    const auto ep = nxr::magnetic::hermitianEigs(nxr::magnetic::transportLaplacian(ns, E, ones, c.lag, std::vector<Eigen::MatrixXd>(R.begin(), R.end()), nxr::magnetic::Normalise::None), 2);
    const Eigen::VectorXcd x = ep.vectors.col(0);
    double offNormal = 0, lagErr = 0;
    for (int v = 0; v < ns; ++v) {
      const Eigen::Vector3cd xv = x.segment<3>(3 * v);
      const Eigen::Vector3cd nv = Ns.row(v).transpose().cast<C>();
      offNormal = std::max(offNormal, (xv - nv * nv.dot(xv)).norm() / xv.norm());
    }
    for (Eigen::Index e = 0; e < E.rows(); ++e) {
      const C au = Ns.row(E(e, 0)).cast<C>().dot(x.segment<3>(3 * E(e, 0))), av = Ns.row(E(e, 1)).cast<C>().dot(x.segment<3>(3 * E(e, 1)));
      lagErr = std::max(lagErr, std::abs(std::remainder(std::arg(au * std::conj(av)) - (-(ph[E(e, 0)] - ph[E(e, 1)])), 2 * M_PI)));
    }
    EXPECT(std::abs(ep.values(0)) < 1e-9 && ep.values(1) > 1e-4 && offNormal < 1e-6 && lagErr < 1e-6,
           "on a sphere (curved): λ₁ " << ep.values(0) << ", λ₂ " << ep.values(1) << " (unique); the lowest mode lies along the normal (off by ≤ " << offNormal
           << ") with the current's lags (worst " << lagErr << " rad)");
  }
  // refusals
  {
    nxr::transport::Edges E(1, 2);
    E << 0, 0;
    bool a = false, b = false, c2 = false;
    try { nxr::magnetic::transportLaplacian(2, E, Eigen::VectorXd::Ones(1), Eigen::VectorXd::Zero(1), {Eigen::MatrixXd::Identity(3, 3)}, nxr::magnetic::Normalise::None); } catch (const std::invalid_argument&) { a = true; }
    try { nxr::transport::transportOf("parallel"); } catch (const std::invalid_argument&) { b = true; }
    try { nxr::transport::compare({Eigen::MatrixXcd::Zero(2, 2)}, {Eigen::Matrix3d::Identity()}, Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1)); } catch (const std::invalid_argument&) { c2 = true; }
    EXPECT(a && b && c2, "a self-loop, an unknown transport and a block that is not 3 x 3 refused");
  }
  std::cout << (failures ? "FAIL" : "PASS") << " test_transport (" << failures << " failures)\n";
  return failures ? 1 : 0;
}

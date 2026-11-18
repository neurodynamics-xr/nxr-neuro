// nxr::signal::graphfilterbank::bandMoments / bandPower and the area-weighted memberEnergy, on an icosphere with its FULL eigenbasis (synthetic only):
//   the band power of a statistic C = c cᴴ (c a field's mode coefficients) equals the area-weighted energy of the band-filtered field (Parseval), for a
//   tight bank (the Meyer octave frame by Weyl) and a non-tight one (MexHat); a tight bank's band powers add up to the field's power; C_m = G_m C G_m is
//   Hermitian, its trace the power; a field with no lags (a real pattern times one complex phase) gives real C_m, a travelling pattern does not;
//   bandPower(diag C) == bandMoments(C).power; the unweighted memberEnergy is Rheome's (it differs on an uneven mesh); shape refusals.
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/signal/graphfilterbank.h"

namespace fb = nxr::signal::graphfilterbank;
using C = std::complex<double>;

int main() {
  std::cout << "test_band_moments\n";
  std::vector<double> V;
  std::vector<int32_t> F;
  icosphere(2, V, F, 0.05);
  for (std::size_t i = 0; i < V.size(); i += 3) V[i] *= 1.4;                         // an ellipsoid: uneven vertex areas
  const int n = static_cast<int>(V.size() / 3);
  nxr::geometry::Manifold m(V.data(), n, F.data(), static_cast<int>(F.size() / 3), true);
  const Eigen::SparseMatrix<double> K = m.operators().laplacian().cotan(), M = m.operators().mass().mixedVoronoi();
  // the full eigenbasis, mass-orthonormal (dense: a small mesh)
  const Eigen::MatrixXd Kd(K), Md(M);
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd, Md);
  fb::EigenTransform t{es.eigenvectors(), es.eigenvalues().cwiseMax(0.0), M};
  const Eigen::VectorXd area = M.diagonal();
  double total_area = area.sum();
  // a complex field, 3 columns: noise, a pattern with no lags (real times e^{i 0.7}), a travelling pattern (phase along x)
  std::mt19937 rng(9);
  std::normal_distribution<double> nd;
  Eigen::MatrixXcd X(n, 3);
  for (int v = 0; v < n; ++v) {
    const double x = V[3 * v], z = V[3 * v + 2];
    X(v, 0) = C(nd(rng), nd(rng));
    X(v, 1) = std::polar(std::cos(40 * z) + 0.3 * x * 20, 0.7);
    X(v, 2) = std::polar(1.0, 60 * x);
  }
  const Eigen::MatrixXcd c = fb::forward(t, X);
  const fb::Bank meyer = fb::weyl(total_area, 6);
  fb::Range rg; rg.lmax = t.lambda.maxCoeff(); rg.lmin = t.lambda(1);
  fb::DesignOptions o; o.numFilters = 5;
  const fb::Bank mexhat = fb::design(fb::Family::MexHat, rg, o);
  int failures = 0;
  for (const auto* bank : {&meyer, &mexhat}) {
    const auto W = fb::analysis(*bank, t, X);
    const Eigen::MatrixXd Ew = fb::memberEnergy(W, area), Eu = fb::memberEnergy(W);
    double worst = 0, unweightedDiff = 0;
    for (int j = 0; j < 3; ++j) {
      const Eigen::MatrixXcd Cj = c.col(j) * c.col(j).adjoint();
      const auto bm = fb::bandMoments(*bank, t.lambda, Cj, true);
      worst = std::max(worst, (bm.power - Ew.col(j)).cwiseAbs().maxCoeff() / Ew.col(j).cwiseAbs().maxCoeff());
      unweightedDiff = std::max(unweightedDiff, (Eu.col(j) / Eu.col(j).sum() - Ew.col(j) / Ew.col(j).sum()).cwiseAbs().maxCoeff());
      for (std::size_t b = 0; b < bm.matrices.size(); ++b) {
        const auto& Cm = bm.matrices[b];
        if ((Cm - Cm.adjoint()).cwiseAbs().maxCoeff() > 1e-12 * Cm.cwiseAbs().maxCoeff() + 1e-300 || std::abs(Cm.trace().real() - bm.power(b)) > 1e-10 * std::max(1.0, bm.power(b))) ++failures;
      }
      if ((fb::bandPower(*bank, t.lambda, Cj.diagonal().real()) - bm.power).cwiseAbs().maxCoeff() > 1e-10 * bm.power.cwiseAbs().maxCoeff()) ++failures;
    }
    const bool ok = worst < 1e-10 && failures == 0;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << bank->kind << ": band power of c cᴴ == the area-weighted energy of the filtered field (worst " << worst
              << "); C_m Hermitian, tr C_m the power; bandPower(diag) == bandMoments; the unweighted energy's shares differ by up to " << unweightedDiff << "\n";
    if (!ok) ++failures;
  }
  // a tight bank's band powers add up to the field's power
  {
    const Eigen::MatrixXcd C0 = c.col(0) * c.col(0).adjoint();
    const double sum = fb::bandMoments(meyer, t.lambda, C0).power.sum(), tot = C0.trace().real();
    const bool ok = std::abs(sum - tot) < 1e-10 * tot;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] the Meyer frame (tight): Σ_m P_m == tr C (" << sum << " vs " << tot << ")\n";
    if (!ok) ++failures;
  }
  // no lags: every band's matrix real; a travelling pattern: not
  {
    const auto still = fb::bandMoments(meyer, t.lambda, c.col(1) * c.col(1).adjoint(), true), moving = fb::bandMoments(meyer, t.lambda, c.col(2) * c.col(2).adjoint(), true);
    double im = 0, lag = 0;
    for (const auto& Cm : still.matrices) im = std::max(im, Cm.imag().norm() / std::max(Cm.norm(), 1e-300));
    for (const auto& Cm : moving.matrices) lag = std::max(lag, Cm.imag().norm() / std::max(Cm.norm(), 1e-300));
    const bool ok = im < 1e-12 && lag > 0.1;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] a pattern with no lags: every band's ‖Im C_m‖/‖C_m‖ ≤ " << im << "; a travelling pattern's reaches " << lag << "\n";
    if (!ok) ++failures;
  }
  // refusals
  {
    bool threw = false;
    try { fb::bandMoments(meyer, t.lambda, Eigen::MatrixXcd::Zero(3, 3)); } catch (const std::invalid_argument&) { threw = true; }
    bool threw2 = false;
    try { fb::memberEnergy(fb::analysis(meyer, t, X), Eigen::VectorXd::Ones(3)); } catch (const std::invalid_argument&) { threw2 = true; }
    std::cout << "  [" << (threw && threw2 ? "PASS" : "FAIL") << "] mismatched shapes refused\n";
    if (!(threw && threw2)) ++failures;
  }
  std::cout << (failures ? "FAIL" : "PASS") << " test_band_moments (" << failures << " failures)\n";
  return failures ? 1 : 0;
}

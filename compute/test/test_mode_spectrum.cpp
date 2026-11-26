// nxr::signal mode_spectrum on synthetic data (an icosphere, generated modes and signals):
//   modeAnalysis: ⋆0-orthonormal modes stored at 1/√2 (as a package stores ψ/√2) come back with rows · synthesis = I under unit_mass; the kinds select;
//   modeAnalysisOfComponents: each Cartesian component expanded on the same modes, component-major;
//   (the flow of a cross-spectrum through these coordinates: test_flow_readout)
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_mode_spectrum\n";
  namespace ns = nxr::signal;
  std::mt19937 rng(7);
  std::normal_distribution<double> nd(0, 1);
  auto randn = [&](Eigen::Index r, Eigen::Index c) { Eigen::MatrixXd m(r, c); for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) m(i, j) = nd(rng); return m; };
  auto crandn = [&](Eigen::Index r, Eigen::Index c) { return Eigen::MatrixXcd(randn(r, c).cast<std::complex<double>>() + std::complex<double>(0, 1) * randn(r, c).cast<std::complex<double>>()); };

  std::vector<double> Vx; std::vector<int32_t> F;
  icosphere(2, Vx, F, 1.0);
  const ns::IntrinsicComplex c = ns::intrinsicComplex(nxr::geometry::surfaceArrays(Vx.data(), static_cast<int>(Vx.size() / 3), F.data(), static_cast<int>(F.size() / 3)));
  const int V = c.nV, K = 8;
  const Eigen::VectorXd mass = c.star0;

  // ⋆0-orthonormal modes (Gram-Schmidt in the mass), stored at 1/√2; kinds alternate exact / coexact
  Eigen::MatrixXd Q = randn(V, K);
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < k; ++j) Q.col(k) -= Q.col(j).cwiseProduct(mass).dot(Q.col(k)) * Q.col(j);
    Q.col(k) /= std::sqrt(Q.col(k).cwiseAbs2().dot(mass));
  }
  const Eigen::MatrixXd P = Q / std::sqrt(2.0);
  std::vector<int> kind(K);
  for (int k = 0; k < K; ++k) kind[static_cast<std::size_t>(k)] = k % 2 ? 2 : 0;
  const ns::ModeAnalysis all = ns::modeAnalysis(P, mass, kind, ns::ModeKinds::all, ns::ModeNormalisation::unit_mass);
  EXPECT((all.rows * all.synthesis - Eigen::MatrixXd::Identity(K, K)).cwiseAbs().maxCoeff() < 1e-10, "unit_mass: rows · synthesis = I");
  const ns::ModeAnalysis ex = ns::modeAnalysis(P, mass, kind, ns::ModeKinds::exact, ns::ModeNormalisation::as_stored);
  EXPECT(ex.modes.size() == 4 && ex.modes[1] == 2 && (ex.rows * ex.synthesis - 0.5 * Eigen::MatrixXd::Identity(4, 4)).cwiseAbs().maxCoeff() < 1e-10,
         "exact modes only; as_stored keeps the package's 1/2 norm");

  const ns::ModeAnalysis withConstant = ns::modeAnalysis(P, mass, kind, ns::ModeKinds::exact_and_constant, ns::ModeNormalisation::unit_mass);
  EXPECT(withConstant.modes.size() == 5 && withConstant.modes[0] == -1 && std::abs(withConstant.synthesis.col(0).cwiseAbs2().dot(mass) - 1.0) < 1e-12 &&
         std::abs((withConstant.rows.row(0) * Eigen::VectorXd::Ones(V)).value() - std::sqrt(mass.sum())) < 1e-10, "exact_and_constant: the constant first, at unit ⋆0 norm");

  // the components: a free kernel [3V x C]
  const int C = 5;
  const Eigen::MatrixXd W = randn(3 * V, C);
  const Eigen::MatrixXd O = ns::modeAnalysisOfComponents(ex.rows, W, 3);
  double e = 0;
  for (int comp = 0; comp < 3; ++comp) {
    Eigen::MatrixXd Wc(V, C);
    for (int v = 0; v < V; ++v) Wc.row(v) = W.row(3 * v + comp);
    e = std::max(e, (O.middleRows(comp * 4, 4) - ex.rows * Wc).cwiseAbs().maxCoeff());
  }
  EXPECT(O.rows() == 12 && e < 1e-10, "each Cartesian component expanded on the same modes, component-major");

  std::cout << (failures ? "FAILED" : "OK") << "\n";
  return failures ? 1 : 0;
}

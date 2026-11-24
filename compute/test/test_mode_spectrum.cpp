// nxr::signal mode_spectrum on synthetic data (an icosphere, generated modes and signals):
//   modeAnalysis: ⋆0-orthonormal modes stored at 1/√2 (as a package stores ψ/√2) come back with rows · synthesis = I under unit_mass; the kinds select;
//   modeAnalysisOfComponents: each Cartesian component expanded on the same modes, component-major;
//   spectralCoefficients: a cosine at bin k0 gives a·n/2·e^{iφ}; operator-then-transform equals transform-then-operator (they commute);
//   modeCrossSpectrum: Σ over tiles and weighted bins; contractComponents: the sum of the diagonal blocks;
//   flowMap: the power and the edge phase current of the synthesised field equal |z|² and phaseCurrent(c, z) summed over the draws of the cross-spectrum;
//   cellMeasures: the ingredients against a direct sum; effectiveDegreesOfFreedom: independent bins, fully dependent bins.
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

  // the transform of a tile: a cosine at bin k0, and the commuting operator
  const int n = 256, k0 = 17;
  Eigen::MatrixXd tile = 0.1 * randn(C, n);
  const double a = 2.0, phi = 0.7;
  for (int t = 0; t < n; ++t) tile(0, t) += a * std::cos(2 * M_PI * k0 * t / n + phi);
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(1, C); E(0, 0) = 1;
  Eigen::MatrixXd tile0 = Eigen::MatrixXd::Zero(C, n);
  for (int t = 0; t < n; ++t) tile0(0, t) = a * std::cos(2 * M_PI * k0 * t / n + phi);
  const Eigen::MatrixXcd one = ns::spectralCoefficients(E, tile0, k0, 1);
  EXPECT(std::abs(one(0, 0) - std::polar(a * n / 2.0, phi)) < 1e-9, "a cosine at bin k0 gives a n/2 e^{i phi}");
  const Eigen::MatrixXcd A = ns::spectralCoefficients(O, tile, 10, 20);
  const Eigen::MatrixXcd B = ns::rfft(Eigen::MatrixXd(O * tile), 1).middleCols(10, 20);
  EXPECT(A.rows() == 12 && A.cols() == 20 && (A - B).cwiseAbs().maxCoeff() < 1e-9 * B.cwiseAbs().maxCoeff(), "operator then transform = transform then operator");

  // the statistic over tiles, and the contraction
  const Eigen::VectorXd w = Eigen::VectorXd::LinSpaced(20, 0.1, 1.0);
  const Eigen::MatrixXcd A2 = crandn(12, 20);
  const Eigen::MatrixXcd T = ns::modeCrossSpectrum({A, A2}, w);
  Eigen::MatrixXcd T2 = Eigen::MatrixXcd::Zero(12, 12);
  for (const Eigen::MatrixXcd* X : {&A, &A2}) for (int b = 0; b < 20; ++b) T2 += w(b) * X->col(b) * X->col(b).adjoint();
  EXPECT((T - T2).cwiseAbs().maxCoeff() < 1e-9 * T2.cwiseAbs().maxCoeff(), "Σ over tiles and weighted bins");
  const Eigen::MatrixXcd S3 = ns::contractComponents(T, 3);
  EXPECT((S3 - (T.block(0, 0, 4, 4) + T.block(4, 4, 4, 4) + T.block(8, 8, 4, 4))).cwiseAbs().maxCoeff() < 1e-12, "the contraction sums the diagonal blocks");

  // the flow map against the synthesised field's own power and phase current
  const int m = 6;
  const Eigen::MatrixXcd Y = crandn(K, m);
  const Eigen::MatrixXcd S = Y * Y.adjoint();
  const Eigen::VectorXd h = Eigen::VectorXd::LinSpaced(K, 1.0, 0.2);
  const Eigen::MatrixXcd Z = all.synthesis.cast<std::complex<double>>() * h.cast<std::complex<double>>().asDiagonal() * Y;
  const ns::FlowMap fm = ns::flowMap(c, all.synthesis, S, h, 50);
  const Eigen::VectorXd J = ns::phaseCurrent(c, Z).rowwise().sum();
  const Eigen::VectorXd pw = Z.cwiseAbs2().rowwise().sum();
  EXPECT((fm.phaseCurrent - J).cwiseAbs().maxCoeff() < 1e-9 * J.cwiseAbs().maxCoeff(), "the phase current = phaseCurrent(c, z) summed over the draws (blocks of 50 vertices)");
  EXPECT((fm.power - pw).cwiseAbs().maxCoeff() < 1e-9 * pw.maxCoeff(), "the power = |z|² summed over the draws");

  // the cell measures against a direct sum
  Eigen::MatrixXd H = randn(3, K).cwiseAbs();
  const ns::CellMeasures cm = ns::cellMeasures(S, H);
  double qa = 0, na = 0, ra = 0;
  for (int j = 0; j < K; ++j) for (int k = 0; k < K; ++k) if (j != k) {
    qa += H(0, j) * H(2, k) * std::pow(S(j, k).imag(), 2);
    ra += H(0, j) * H(2, k) * std::pow(S(j, k).real(), 2);
    na += H(0, j) * H(2, k) * (S(j, j).real() * S(k, k).real() - std::pow(S(j, k).real(), 2));
  }
  EXPECT(std::abs(cm.quadratureEnergy(0, 2) - qa) < 1e-9 * qa && std::abs(cm.inPhaseEnergy(0, 2) - ra) < 1e-9 * ra && std::abs(cm.quadratureNoise(0, 2) - na) < 1e-9 * std::abs(na),
         "in-phase, quadrature and noise ingredients between two scale members");
  EXPECT(std::abs(cm.power(1) - H.row(1).dot(S.diagonal().real())) < 1e-9 * cm.power(1), "power per scale member");

  // degrees of freedom
  const Eigen::VectorXd flat = Eigen::VectorXd::Ones(4);
  EXPECT(std::abs(ns::effectiveDegreesOfFreedom(flat, 9, Eigen::VectorXd()) - 36) < 1e-12, "independent bins: tiles x bins");
  EXPECT(std::abs(ns::effectiveDegreesOfFreedom(flat, 9, Eigen::VectorXd::Ones(4)) - 9) < 1e-12, "fully dependent bins: tiles");

  std::cout << (failures ? "FAILED" : "OK") << "\n";
  return failures ? 1 : 0;
}

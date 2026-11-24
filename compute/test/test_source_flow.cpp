// nxr::signal source_flow on synthetic data (random kernels and coordinates, an icosphere's complex and its Hodge-Dirac modes):
//   selectVertexRows: a vertex range and a glued-style selection, the bridge rows exactly zero;
//   synthesisFromAnalysis: analysis x synthesis = I for mass-orthonormal coordinates;
//   bandReadouts: the bands add back to the kernel exactly; a band below the last equals the vertex filter Phi g Phiᵀ M W (the reference's band kernel); with every mode kept and the
//     last band's gain zero on them, the last band is the kernel's content outside the modes, zero for a complete basis;
//   (the flow readout through these readouts: test_flow_readout)
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/source_flow.h"
#include "nxr/signal/spectral_flow.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_source_flow\n";
  namespace ns = nxr::signal;
  using cd = std::complex<double>;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd(0, 1);
  auto randn = [&](Eigen::Index r, Eigen::Index c) { Eigen::MatrixXd m(r, c); for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) m(i, j) = nd(rng); return m; };
  auto crandn = [&](Eigen::Index r, Eigen::Index c) { return Eigen::MatrixXcd(randn(r, c).cast<cd>() + cd(0, 1) * randn(r, c).cast<cd>()); };
  auto relative = [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); };

  // ── selectVertexRows ──
  {
    const Eigen::MatrixXd kernel = randn(3 * 6, 4);
    const std::vector<std::int64_t> hemisphere = {3, 4, 5}, glued = {0, 2, -1, 5};
    const Eigen::MatrixXd h = ns::selectVertexRows(kernel, hemisphere, 3), g = ns::selectVertexRows(kernel, glued, 3);
    EXPECT(h.rows() == 9 && (h - kernel.bottomRows(9)).cwiseAbs().maxCoeff() == 0, "a hemisphere: its vertex range's rows");
    EXPECT(g.middleRows(6, 3).cwiseAbs().maxCoeff() == 0 && (g.middleRows(3, 3) - kernel.middleRows(6, 3)).cwiseAbs().maxCoeff() == 0, "glued: rows selected, the bridge exactly zero");
    bool refused = false;
    try { ns::selectVertexRows(kernel, {7}, 3); } catch (const std::invalid_argument&) { refused = true; }
    EXPECT(refused, "refused: a source vertex outside the rows");
  }

  // ── an icosphere, its complex and its modes ──
  std::vector<double> Vx; std::vector<int32_t> F;
  icosphere(2, Vx, F, 1.0);
  const int nV = static_cast<int>(Vx.size() / 3), nF = static_cast<int>(F.size() / 3);
  const auto surface = nxr::geometry::surfaceArrays(Vx.data(), nV, F.data(), nF);
  const ns::IntrinsicComplex c = ns::intrinsicComplex(surface);
  const auto h = nxr::operators::hodgedirac::assemble(surface);
  nxr::operators::HodgeSolveOptions so; so.limit = 60;
  const ns::Modes md = ns::modesOf(nxr::operators::hodgeDiracModes(h, 120, false, so), h.nV, h.nE, h.nF);
  // the field coordinates: the constant and the exact modes' vertex parts, each at unit mass norm
  std::vector<int> exact;
  for (Eigen::Index k = 0; k < md.mu.size(); ++k) if (md.kind[static_cast<std::size_t>(k)] == 0) exact.push_back(static_cast<int>(k));
  const Eigen::Index K = static_cast<Eigen::Index>(exact.size()) + 1;
  Eigen::MatrixXd Phi(nV, K);
  Eigen::VectorXd mu(K);
  Phi.col(0).setConstant(1.0 / std::sqrt(c.star0.sum())); mu(0) = 0;
  for (Eigen::Index j = 1; j < K; ++j) { Phi.col(j) = md.P.col(exact[static_cast<std::size_t>(j - 1)]); Phi.col(j) /= std::sqrt(Phi.col(j).cwiseAbs2().dot(c.star0)); mu(j) = md.mu(exact[static_cast<std::size_t>(j - 1)]); }
  const Eigen::MatrixXd analysis = Phi.transpose() * c.star0.asDiagonal();
  EXPECT((analysis * Phi - Eigen::MatrixXd::Identity(K, K)).cwiseAbs().maxCoeff() < 1e-8, "the coordinates are mass-orthonormal");
  EXPECT((ns::synthesisFromAnalysis(analysis, c.star0) - Phi).cwiseAbs().maxCoeff() < 1e-12, "synthesisFromAnalysis gives the synthesis back");

  // ── bandReadouts ──
  {
    const int channels = 7;
    const Eigen::MatrixXd W = randn(3 * nV, channels);                    // a free-orientation kernel, vertex-major
    Eigen::MatrixXd modeOperator(3 * K, channels);                         // (I3 ⊗ Phiᵀ M) W, component-major
    const Eigen::MatrixXd Wc = ns::componentMajor(W, 3);
    for (int comp = 0; comp < 3; ++comp) modeOperator.middleRows(comp * K, K) = analysis * Wc.middleRows(comp * nV, nV);
    Eigen::VectorXd logMu(K);
    for (Eigen::Index j = 0; j < K; ++j) logMu(j) = mu(j) > 0 ? std::log2(mu(j)) : -std::numeric_limits<double>::infinity();
    const Eigen::MatrixXd gains = ns::cosineSquaredPartition(logMu, {std::log2(mu(2)), std::log2(mu(6)), std::log2(mu(K - 1))});
    const std::vector<Eigen::MatrixXd> bands = ns::bandReadouts(modeOperator, Phi, gains, W, 3);
    Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(3 * nV, channels);
    for (const auto& b : bands) sum += b;
    EXPECT(relative(sum, Wc) < 1e-12, "the bands add back to the kernel");
    const Eigen::MatrixXd filter = Phi * gains.row(1).transpose().asDiagonal() * analysis;   // the reference's band kernel Phi g Phiᵀ M W, per component
    EXPECT(relative(bands[1].middleRows(nV, nV), filter * Wc.middleRows(nV, nV)) < 1e-10, "a band = Phi g Phiᵀ M W");
    const Eigen::MatrixXd outside = Wc.middleRows(0, nV) - Phi * analysis * Wc.middleRows(0, nV);
    const Eigen::MatrixXd lastOwn = Phi * gains.row(2).transpose().asDiagonal() * analysis * Wc.middleRows(0, nV);
    EXPECT(relative(bands[2].middleRows(0, nV), lastOwn + outside) < 1e-10, "the last band = its own gain + the content outside the modes");
  }


  std::cout << (failures ? "FAILED " : "OK ") << failures << "\n";
  return failures ? 1 : 0;
}

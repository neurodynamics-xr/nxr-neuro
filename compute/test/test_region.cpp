// nxr::region::collapse (pca_flip), synthetic only: an icosphere cut into regions (by octant); in each region one complex signal z_R(t), carried by its
// vertices with the sign of their orientation (a fold: some normals flipped, the signal flipped with them) plus noise. The weights recover each region's
// signal (|correlation| ≈ 1, its sign that of the region's MAJORITY orientation, MNE's convention: where most sources are flipped the region's signal is
// the flipped one), so the lags between regions are the true ones up to those signs (a plain mean would cancel across the flipped sources); the first component's share is high; real weights, unit norm; refusals.
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "measurement_helpers.h"
#include "nxr/region.h"

using C = std::complex<double>;

int main() {
  std::cout << "test_region\n";
  std::vector<double> V;
  std::vector<int32_t> F;
  icosphere(2, V, F, 1.0);
  const int n = static_cast<int>(V.size() / 3), R = 8, T = 3000;
  std::mt19937 rng(23);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> ud(0, 1);
  std::vector<int32_t> codes(n);
  Eigen::MatrixXd N(n, 3);
  std::vector<double> sgn(n);
  for (int v = 0; v < n; ++v) {
    codes[v] = (V[3 * v] > 0) + 2 * (V[3 * v + 1] > 0) + 4 * (V[3 * v + 2] > 0);
    sgn[v] = ud(rng) < 0.3 ? -1.0 : 1.0;                                    // 30 % of the vertices on the far wall of a fold
    for (int d = 0; d < 3; ++d) N(v, d) = sgn[v] * V[3 * v + d];
  }
  // each region's signal: a 10 Hz analytic tone with a region-specific lag, amplitude modulated, plus independent noise a vertex
  const std::vector<double> lagOf = {0.0, 0.4, 0.8, 1.2, -0.4, -0.8, 1.6, -1.2};
  Eigen::MatrixXcd zR(R, T), Z(n, T);
  for (int t = 0; t < T; ++t) {
    const double a = 1 + 0.3 * std::sin(2 * M_PI * 0.3 * t / 1000.0);
    for (int r = 0; r < R; ++r) zR(r, t) = std::polar(a, 2 * M_PI * 10.0 * t / 1000.0 - lagOf[r]);
  }
  for (int v = 0; v < n; ++v)
    for (int t = 0; t < T; ++t) Z(v, t) = sgn[v] * zR(codes[v], t) + 0.5 * C(nd(rng), nd(rng));
  const Eigen::MatrixXcd M = crossSpectrumTiles(Z, T, 1).sum[0] / static_cast<double>(T);   // the vertices' cross-spectrum
  const auto w = nxr::region::collapse(nxr::region::Collapse::PcaFlip, codes, R, Eigen::MatrixXd::Identity(n, n), M, N);
  const Eigen::MatrixXcd Zr = w.W.cast<C>() * Z;                           // the regions' signals
  int failures = 0;
  double worstCorr = 1, worstLag = 0, minShare = 1;
  std::vector<double> expect(R, 1.0);
  for (int r = 0; r < R; ++r) {
    const C c = Zr.row(r).dot(zR.row(r));                                   // conj(Zr) · zR
    const double corr = std::abs(c) / (Zr.row(r).norm() * zR.row(r).norm());
    worstCorr = std::min(worstCorr, corr);
    minShare = std::min(minShare, w.share(r));
    // the expected sign: the majority's orientation (MNE's convention): + when most of the region's sources are unflipped, − when most are flipped
    int k = 0, flipped = 0;
    for (int v = 0; v < n; ++v) if (codes[v] == r) { ++k; flipped += sgn[v] < 0; }
    expect[r] = 2 * flipped > k ? -1.0 : 1.0;
    const double phase = std::arg(c * expect[r]);                            // 0 when the region's signal comes back with the majority's sign
    worstLag = std::max(worstLag, std::abs(phase));
  }
  // the lags between regions read from the regions' cross-spectrum equal the true ones
  const Eigen::MatrixXcd Cr = crossSpectrumTiles(Zr, T, 1).sum[0];
  double worstPair = 0;
  for (int a = 0; a < R; ++a) for (int b = 0; b < R; ++b) worstPair = std::max(worstPair, std::abs(std::remainder(std::arg(Cr(a, b) * expect[a] * expect[b]) - (lagOf[b] - lagOf[a]), 2 * M_PI)));
  double norms = 0;
  for (int r = 0; r < R; ++r) norms = std::max(norms, std::abs(Eigen::VectorXd(w.W.row(r)).norm() - 1));
  const bool ok = worstCorr > 0.98 && worstLag < 0.05 && worstPair < 0.05 && minShare > 0.5 && norms < 1e-12;
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] pca_flip recovers every region's signal (|corr| ≥ " << worstCorr << ", with its majority orientation's sign: phase ≤ " << worstLag
            << " rad) despite 30 % flipped sources; the lags between regions within " << worstPair << " rad; first-component share ≥ " << minShare << "; unit-norm real weights\n";
  if (!ok) ++failures;
  bool threw = false;
  try { nxr::region::collapse(nxr::region::Collapse::PcaFlip, std::vector<int32_t>(n, 9), R, Eigen::MatrixXd::Identity(n, n), M, N); } catch (const std::invalid_argument&) { threw = true; }
  bool threw2 = false;
  try { nxr::region::collapseOf("mean"); } catch (const std::invalid_argument&) { threw2 = true; }
  std::cout << "  [" << (threw && threw2 ? "PASS" : "FAIL") << "] a code beyond the regions refused; 'mean' is not an option\n";
  if (!(threw && threw2)) ++failures;
  std::cout << (failures ? "FAIL" : "PASS") << " test_region (" << failures << " failures)\n";
  return failures ? 1 : 0;
}

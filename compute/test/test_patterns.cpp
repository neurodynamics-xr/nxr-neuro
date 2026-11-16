// nxr::patterns and the dense algebra::congruence, on synthetic matrices only:
//   leading      a planted rank-1 pattern plus noise (a cross-spectrum of 4000 samples): recovered (similarity > 0.999), its share as planted; refusals;
//   rotateToLargest  Re of the result has the largest norm over every phase; a real pattern times e^{iθ} comes back real (‖Im‖ / ‖Re‖ ≈ 0);
//   similarity   invariant to e^{iθ}; 1 on itself; refuses a shape mismatch;
//   congruence   the whole A M Aᴴ equals the pattern call on every pair.
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "nxr/congruence.h"
#include "nxr/patterns.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using Cd = std::complex<double>;

int main() {
  std::cout << "test_patterns\n";
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  const int n = 30, T = 4000;
  Eigen::VectorXcd m(n);
  for (int i = 0; i < n; ++i) m(i) = Cd(nd(rng), nd(rng));
  m.normalize();
  // samples: a pattern swinging with a random phase each sample (amplitude 3) plus unit white noise; C = Σ z zᴴ / T
  Eigen::MatrixXcd Z(n, T);
  for (int t = 0; t < T; ++t) {
    const Cd a = 3.0 * std::polar(1.0, 2 * M_PI * t / 37.0);
    for (int i = 0; i < n; ++i) Z(i, t) = a * m(i) + Cd(nd(rng), nd(rng)) / std::sqrt(2.0);
  }
  const Eigen::MatrixXcd C = Z * Z.adjoint() / T;
  const nxr::patterns::Leading L = nxr::patterns::leading(C, 3);
  const double sim = std::abs(L.vectors.col(0).dot(m)), share = L.values(0) / L.trace, planted = (9.0 + 1.0) / (9.0 + n);   // the leading eigenvalue: the pattern's power plus the noise along it
  std::printf("  planted pattern recovered: similarity %.5f; share %.3f (planted %.3f)\n", sim, share, planted);
  EXPECT(sim > 0.999 && std::abs(share - planted) < 0.02, "leading recovers a planted pattern and its share");
  EXPECT(L.values(0) >= L.values(1) && L.values(1) >= L.values(2), "eigenvalues come largest first");
  bool refused = false;
  try { nxr::patterns::leading(C, 0); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "q outside [1, n] is refused");
  // rotateToLargest
  const Eigen::VectorXcd r = nxr::patterns::rotateToLargest(m);
  double best = 0;
  for (int k = 0; k < 360; ++k) best = std::max(best, (m * std::polar(1.0, 2 * M_PI * k / 360.0)).real().norm());
  EXPECT(r.real().norm() >= best - 1e-12, "Re(rotateToLargest(m)) has the largest norm over the phase");
  Eigen::VectorXd real(n);
  for (int i = 0; i < n; ++i) real(i) = nd(rng);
  const Eigen::VectorXcd back = nxr::patterns::rotateToLargest(real.cast<Cd>() * std::polar(1.0, 1.1));
  EXPECT(back.imag().norm() < 1e-12 * back.real().norm(), "a standing (real) pattern times e^{iθ} comes back real");
  // similarity
  Eigen::MatrixXcd U(n, 1), V(n, 1);
  U.col(0) = m; V.col(0) = m * std::polar(1.0, 0.7);
  EXPECT(std::abs(nxr::patterns::similarity(U, V)(0) - 1) < 1e-12, "similarity is phase-free and 1 on itself");
  refused = false;
  try { nxr::patterns::similarity(U, Eigen::MatrixXcd(n + 1, 1)); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "similarity refuses a shape mismatch");
  // the dense congruence equals the pattern call
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(7, n);
  const Eigen::MatrixXcd full = nxr::algebra::congruence(A, C);
  nxr::algebra::Pairs pairs(49, 2);
  for (int u = 0; u < 7; ++u) for (int v = 0; v < 7; ++v) { pairs(7 * u + v, 0) = u; pairs(7 * u + v, 1) = v; }
  const auto at = nxr::algebra::congruence(A, C, pairs);
  double worst = 0;
  for (int u = 0; u < 7; ++u) for (int v = 0; v < 7; ++v) worst = std::max(worst, std::abs(full(u, v) - at[static_cast<std::size_t>(7 * u + v)]));
  EXPECT(worst < 1e-12 * full.cwiseAbs().maxCoeff(), "the whole A M Aᴴ equals the pattern call on every pair");
  std::cout << (failures ? "FAIL" : "PASS") << " test_patterns (" << failures << " failures)\n";
  return failures ? 1 : 0;
}

// The smallest k eigenpairs are COMPLETE and accurate when eigenvalues are multiple. A single-vector Lanczos run finds one vector of a multiple eigenvalue
// in exact arithmetic; the others appear through rounding, some never do. A diagonal operator with the multiplicities of a sphere's spectrum (1, 3, 5, 7, 9,
// 11, 13 at l(l+1)) is the exact case: every k must return the k smallest eigenvalues, M-orthonormal vectors, small residuals.
#include <cmath>
#include <iostream>
#include <vector>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_eigen_complete\n";
  std::vector<double> ev{0.0};
  for (int l = 1; l <= 6; ++l) for (int m = 0; m < 2 * l + 1; ++m) ev.push_back(l * (l + 1.0));
  for (int i = 0; i < 200; ++i) ev.push_back(100.5 + i);
  const int n = static_cast<int>(ev.size());
  Eigen::SparseMatrix<double> K(n, n), M(n, n);
  for (int i = 0; i < n; ++i) { K.insert(i, i) = ev[i]; M.insert(i, i) = 1.0; }
  for (int k : {2, 4, 5, 9, 10, 16, 20, 25, 30, 40, 49}) {
    const auto r = nxr::operators::eigen(K, M, k, nxr::operators::kShiftLadder, false, false);
    double worst = 0;
    for (int i = 0; i < k; ++i) worst = std::max(worst, std::abs(r.eigenvalues(i) - ev[i]));
    Eigen::MatrixXd X = r.eigenvectors;
    const double orth = (X.transpose() * M * X - Eigen::MatrixXd::Identity(k, k)).norm();
    const double res = (K * X - M * X * r.eigenvalues.asDiagonal()).norm();
    EXPECT(r.k == k && worst < 1e-9, "k = " + std::to_string(k) + ": the k smallest eigenvalues, none missing (worst error " + std::to_string(worst) + ")");
    EXPECT(orth < 1e-8 && res < 1e-8, "k = " + std::to_string(k) + ": M-orthonormal vectors, small residual (orth " + std::to_string(orth) + ", res " + std::to_string(res) + ")");
  }
  return failures ? 1 : 0;
}

// test_psd_linear_map: the psd of a linear map from the cross-spectra (nxr::signal::spectral::psd(map, S, rowsPerOutput)) is the power of the mapped series:
// with S = X X^T, diag(K S K^T) is the sum over time of J^2 (the orientations summed or apart), and rows that are not a multiple of the rows per output are refused.
// Synthetic only.
#include "linear_map_inputs.h"
#include "mesh_helpers.h"
#include "nxr/signal/spectral.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace Eigen;
int main() {
    namespace sp = nxr::signal::spectral;
    // the power of the mapped series
    const MatrixXd K = lmi::kernel(3);
    MatrixXd X(lmi::kM, 40); for (int c = 0; c < lmi::kM; ++c) for (int t = 0; t < 40; ++t) X(c, t) = std::sin(0.31 * t * (1 + c) + c);
    const MatrixXd J = K * X, Sx = X * X.transpose();
    const MatrixXd Pc = sp::psd(K, {Sx}, 1), Ps = sp::psd(K, {Sx}, 3);
    EXPECT((Pc.col(0) - J.array().square().rowwise().sum().matrix()).cwiseAbs().maxCoeff() < 1e-9, "diag(K S K^T) is the sum of J^2 over time");
    MatrixXd Jn(lmi::kN, 40); for (int s = 0; s < lmi::kN; ++s) for (int t = 0; t < 40; ++t) Jn(s, t) = J(3 * s, t) * J(3 * s, t) + J(3 * s + 1, t) * J(3 * s + 1, t) + J(3 * s + 2, t) * J(3 * s + 2, t);
    EXPECT((Ps.col(0) - Jn.rowwise().sum()).cwiseAbs().maxCoeff() < 1e-9, "the three orientations' power summed");
    bool threw = false; try { sp::psd(K, {Sx}, 4); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "rows that are not a multiple of the rows per output are refused");
    std::cout << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}

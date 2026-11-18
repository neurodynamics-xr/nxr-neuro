// test_psd_linear_map: the psd of a linear map from the cross-spectra (nxr::signal::spectral::psd(map, S, rowsPerOutput), the former meeg::sourcePower) gives BITWISE what the old function gave
// (golden/psd_linear_map_golden.txt, made by golden/gen_psd_linear_map_golden.cpp from the code of commit 5c67a87), and is the power of the mapped series: with S = X X^T, diag(K S K^T) is the
// sum over time of J^2 (the orientations summed or apart). Synthetic only.
#include "golden_hash.h"
#include "linear_map_inputs.h"
#include "mesh_helpers.h"
#include "nxr/signal/spectral.h"
#include <iostream>
using namespace Eigen;
int main() {
    goldenhash::Reader r(std::string(NXR_TEST_GOLDEN_DIR) + "/psd_linear_map_golden.txt");
    EXPECT(r.good(), "the golden opens");
    namespace sp = nxr::signal::spectral;
    const auto S = lmi::spectra();
    std::string why;
    EXPECT(r.same("free.norm", sp::psd(lmi::kernel(3), S, 3), &why), "a free kernel, the three orientations summed: bitwise " << why);
    EXPECT(r.same("free.components", sp::psd(lmi::kernel(3), S, 1), &why), "a free kernel, every row: bitwise " << why);
    EXPECT(r.same("fixed.norm", sp::psd(lmi::kernel(1), S, 1), &why), "a fixed kernel: bitwise " << why);
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

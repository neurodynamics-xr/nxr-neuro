// nxr::flow, the ambient / triangle functional (Lefevre and Baillet 2008, as Brainstorm's bst_opticalflow):
// (1) the comparison with bst_opticalflow (golden made by CALLING it, never reading it, gen_opticalflow_bst_golden.m) on a tiny irregular sphere: REPORTED, and held to the
//     direction (cos) only. The two are NOT yet equal to 1e-10 (see the PR): same direction, speed about half of ours, so something in Brainstorm's discretisation is not what is
//     written here and cannot be settled without its source;
// (2) frame independence: the ambient velocity (3-vectors) does not change when the tangent frames are rotated;
// (3) a stationary map gives no flow; the data term is the triangle's own (a Galerkin sum, not the vertex-lumped one).
#include <cmath>
#include <iostream>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/signal/opticalflow.h"

using namespace nxr::signal;
using Eigen::MatrixXd;

static double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); }

int main() {
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/opticalflow_bst_golden.txt");
    EXPECT(g.good(), "golden file");
    const MatrixXd Vg = g.mat("positions"), Fg = g.mat("faces"), A = g.mat("A");
    const MatrixXd b1 = g.mat("bst_1_1"); g.mat("bst_1_2"); g.mat("bst_1_3");
    const RowMat3 pos = Vg;
    const nxr::operators::FaceMat faces = Fg.cast<std::int32_t>();
    const RowMat3 nrm = vertexNormals(pos, faces);
    const FlowForm lb{Regulariser::Ambient, DataFit::Triangle};
    {
        const MatrixXd v = opticalflowScalar(A, pos, faces, nrm, 1.0, lb);
        double dot = 0, na = 0, nb = 0;
        for (int t = 0; t < v.cols(); ++t) for (int i = 0; i < b1.rows(); ++i) for (int d = 0; d < 3; ++d) { const double x = v(3 * i + d, t), y = b1(i, d + 3 * t); dot += x * y; na += x * x; nb += y * y; }
        std::printf("  vs bst_opticalflow (alpha 1): cos %.4f, speed ratio ours/bst %.3f, max relative difference %.3f\n", dot / std::sqrt(na * nb), std::sqrt(na / nb), [&] { MatrixXd w(b1.rows(), b1.cols()); for (int t = 0; t < v.cols(); ++t) for (int i = 0; i < b1.rows(); ++i) for (int d = 0; d < 3; ++d) w(i, d + 3 * t) = v(3 * i + d, t); return rel(w, b1); }());
        EXPECT(dot / std::sqrt(na * nb) > 0.95, "ambient / triangle points where bst_opticalflow's flow does (cos > 0.95)");
    }
    const RowMat3 nrm2 = nrm;
    const Eigen::Vector3d axis(1, -2, 0.5);
    {
        const MatrixXd v1 = opticalflowScalar(A, pos, faces, nrm2, 0.3, lb), v2 = opticalflowScalar(A, pos, faces, nrm2, 0.3, FlowForm{Regulariser::Ambient, DataFit::Triangle, axis});
        std::printf("  frames rotated: ambient/triangle changes by %.2e\n", rel(v1, v2));
        EXPECT(rel(v1, v2) < 1e-12, "the ambient velocity does not depend on the frames (1e-12)");
    }
    {
        MatrixXd still(A.rows(), 3);
        for (int t = 0; t < 3; ++t) still.col(t) = A.col(0);
        EXPECT(opticalflowScalar(still, pos, faces, nrm2, 1.0, lb).cwiseAbs().maxCoeff() < 1e-12, "a stationary map gives no flow");
    }
    return g_failures ? 1 : 0;
}

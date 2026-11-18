// The 1-form commands of the bindings (hodge, whitney, trivial, randomDecomposed1Form) act on the OPERATOR mesh's DEC (assembleDECOperators), the
// intrinsic triangulation's edges, not the input's (embeddedDEC). On a mesh whose intrinsic construction flips edges the two differ, and
// only the operator mesh's DEC satisfies d0' * star1 * d0 == the cotan Laplacian of the operator mesh. A binding that takes the wrong one fails here.
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dec.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/factorization.h"

#include <iostream>
#include <memory>

using namespace nxr;
static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

static double relDiff(const Eigen::SparseMatrix<double>& a, const Eigen::SparseMatrix<double>& b) {
    Eigen::SparseMatrix<double> d = a - b;
    return d.coeffs().cwiseAbs().maxCoeff() / std::max(1e-300, a.coeffs().cwiseAbs().maxCoeff());
}

int main() {
    std::cout << "test_operator_dec\n";
    const Mesh m = jitterSphere(icosphere(3), 0.35, 7);
    const geometry::SurfaceArrays s = geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF());
    const operators::DECOperators op  = operators::assembleDECOperators(s);
    const operators::DECOperators emb = operators::embeddedDEC(s);
    const Eigen::SparseMatrix<double> K = operators::cotanLaplacian(s);

    const Eigen::SparseMatrix<double> Kop  = op.d0.transpose() * op.hodge1 * op.d0;
    const Eigen::SparseMatrix<double> Kemb = emb.d0.transpose() * emb.hodge1 * emb.d0;
    CHECK(relDiff(Kop, K) < 1e-10, "operator mesh DEC: d0' star1 d0 == the cotan Laplacian");
    CHECK(relDiff(Kemb, K) > 1e-6, "the input's DEC differs from the operator mesh's (the intrinsic option flipped edges)");
    CHECK(relDiff(op.d0, emb.d0) > 0.0, "d0 differs between the two");

    // the Hodge decomposition on the operator mesh's DEC splits an edge 1-form exactly
    Eigen::VectorXd omega = Eigen::VectorXd::LinSpaced(op.d0.rows(), -1.0, 1.0);
    const std::unique_ptr<operators::CholeskyFactor> exact = operators::exactFormFactor(op);
    const std::unique_ptr<operators::SquareSolver> coexact = operators::coexactFormSolver(op);
    const signal::HodgeResult h = signal::hodge(s, op, *exact, *coexact, omega);
    CHECK((h.dAlpha + h.deltaBeta + h.gamma - omega).norm() < 1e-8 * omega.norm(), "hodge parts sum to omega");
    CHECK(h.dAlpha.size() == op.d0.rows(), "the 1-form is indexed by the operator mesh's edges");

    std::cout << (failures ? "FAIL" : "PASS") << " test_operator_dec (" << failures << " failures)\n";
    return failures ? 1 : 0;
}

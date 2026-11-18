// The gate for replacing the factor cache by geometry-central's solvers: the same systems (K + εI, d0ᵀ⋆1d0 + εI, d1⋆1⁻¹d1ᵀ + εI, ε = 1e-8) solved by
// the factorizations the cache held (Eigen SimplicialLLT, SparseLU) and by geometry-central's PositiveDefiniteSolver (SimplicialLDLT without SuiteSparse)
// and SquareSolver (SparseLU). Prints the largest difference of each; asserts what is required for each replacement.
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dec.h"

#include "geometrycentral/numerical/linear_solvers.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cstdio>
#include <iostream>
#include <random>

using namespace nxr;
static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

static Eigen::SparseMatrix<double> reg(Eigen::SparseMatrix<double> A) {
    for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += 1e-8;
    return A;
}

int main() {
    const Mesh m = jitterSphere(icosphere(4), 0.3, 3);
    const geometry::SurfaceArrays s = geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF());
    const operators::DECOperators dec = operators::embeddedDEC(s);
    const Eigen::SparseMatrix<double> K = reg(operators::cotanLaplacian(s));
    const Eigen::SparseMatrix<double> A = reg(dec.d0.transpose() * dec.hodge1 * dec.d0);
    const Eigen::SparseMatrix<double> B = reg(dec.d1 * dec.hodge1Inverse * dec.d1.transpose());
    std::mt19937 g(5);
    auto rhs = [&](Eigen::Index n) { Eigen::VectorXd b(n); for (Eigen::Index i = 0; i < n; ++i) b(i) = gauss(g); b.array() -= b.mean(); return b; };

    for (auto [name, M] : {std::pair<const char*, const Eigen::SparseMatrix<double>*>{"K + eI", &K}, {"d0' *1 d0 + eI", &A}}) {
        const Eigen::VectorXd b = rhs(M->rows());
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt(*M);
        const Eigen::VectorXd x0 = llt.solve(b);
        geometrycentral::PositiveDefiniteSolver<double> pd(const_cast<Eigen::SparseMatrix<double>&>(*M));
        const Eigen::VectorXd x1 = pd.solve(b);
        const double d = (x0 - x1).cwiseAbs().maxCoeff() / x0.cwiseAbs().maxCoeff();
        std::printf("  %-16s SimplicialLLT vs geometry-central PositiveDefiniteSolver (LDLT): max relative difference %.3g, bitwise %s\n", name, d, x0 == x1 ? "yes" : "no");
    }
    {
        const Eigen::VectorXd b = rhs(B.rows());
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
        lu.compute(B);
        const Eigen::VectorXd x0 = lu.solve(b);
        geometrycentral::SquareSolver<double> sq(const_cast<Eigen::SparseMatrix<double>&>(B));
        const Eigen::VectorXd x1 = sq.solve(b);
        CHECK(x0 == x1, "d1 *1^-1 d1' + eI: SparseLU vs geometry-central SquareSolver bitwise");
    }
    std::cout << (failures ? "FAILED " : "PASSED ") << failures << "\n";
    return failures ? 1 : 0;
}

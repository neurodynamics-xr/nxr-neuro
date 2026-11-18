#include "nxr/operators/factorization.h"

namespace nxr::operators {

using core::Error;
using core::ErrorCode;

namespace {
Eigen::SparseMatrix<double> regularize(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> R = A;
    const int n = static_cast<int>(R.rows());
    for (int i = 0; i < n; i++) R.coeffRef(i, i) += kFactorRegularization;
    return R;
}
std::unique_ptr<CholeskyFactor> cholesky(const Eigen::SparseMatrix<double>& A, const char* what, const char* hint) {
    auto llt = std::make_unique<CholeskyFactor>();
    llt->compute(regularize(A));
    if (llt->info() != Eigen::Success) throw Error(ErrorCode::GeometryDegenerate, std::string("Cholesky factorization of ") + what + " failed", hint);
    return llt;
}
}  // namespace

std::unique_ptr<CholeskyFactor> laplacianFactor(const Eigen::SparseMatrix<double>& K) {
    return cholesky(K, "(K + εI)", "Matrix may be singular or non-PSD (degenerate mesh, disconnected graph component, etc.)");
}

std::unique_ptr<CholeskyFactor> exactFormFactor(const DECOperators& dec) {
    const Eigen::SparseMatrix<double> A = dec.d0.transpose() * dec.hodge1 * dec.d0;
    return cholesky(A, "(d0ᵀ ★₁ d0 + εI)", "Mesh may be degenerate or have boundary inconsistencies");
}

std::unique_ptr<SquareSolver> coexactFormSolver(const DECOperators& dec) {
    // B = d1 ★₁⁻¹ d1ᵀ (the face Laplacian for the co-exact solve)
    Eigen::SparseMatrix<double> B = regularize(dec.d1 * dec.hodge1Inverse * dec.d1.transpose());
    try {
        return std::make_unique<SquareSolver>(B);
    } catch (const std::exception& e) {
        throw Error(ErrorCode::GeometryDegenerate, std::string("LU factorization of (d1 ★₁⁻¹ d1ᵀ + εI) failed: ") + e.what(),
                    "Mesh may be degenerate or have boundary inconsistencies");
    }
}

} // namespace nxr::operators

#include "nxr/compute.h"
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <cmath>

namespace nxr::operators {

Eigen::MatrixXd normalize(
    const Eigen::MatrixXd& U,
    const Eigen::SparseMatrix<double>& M
) {
    int k = static_cast<int>(U.cols());

    // Step 1: Gram matrix G = U' * M * U (k x k)
    Eigen::MatrixXd MU = M * U;
    Eigen::MatrixXd G = U.transpose() * MU;
    // Symmetrize for numerical safety
    G = (G + G.transpose()) * 0.5;

    Eigen::MatrixXd U_norm = U;

    // Step 2: Try Cholesky decomposition (fast path)
    Eigen::LLT<Eigen::MatrixXd> llt(G);
    if (llt.info() == Eigen::Success) {
        // G = R'R where R is upper triangular
        // U_new = U * R^{-1}
        Eigen::MatrixXd R = llt.matrixU();
        U_norm = U * R.inverse();
    } else {
        // Step 3: Fallback — eigen-decomposition based sqrt inverse
        std::cout << "[normalize] Cholesky failed, using eigen fallback" << std::endl;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(G);
        Eigen::VectorXd s = eig.eigenvalues();
        Eigen::MatrixXd Q = eig.eigenvectors();

        double smax = s.maxCoeff();
        if (smax <= 0) {
            std::cerr << "[normalize] WARNING: Gram matrix not positive. Returning unchanged." << std::endl;
            return U;
        }

        // Clamp small/negative eigenvalues (tol = 1e-12 * max)
        double tol = 1e-12 * smax;
        for (int i = 0; i < s.size(); i++) {
            if (s(i) < tol) s(i) = tol;
        }

        // G^{-1/2} = Q * diag(1/sqrt(s)) * Q'
        Eigen::VectorXd invSqrtS = s.array().sqrt().inverse().matrix();
        Eigen::MatrixXd invSqrtG = Q * invSqrtS.asDiagonal() * Q.transpose();

        U_norm = U * invSqrtG;
    }

    // Step 4: Sanity check
    Eigen::MatrixXd check = U_norm.transpose() * (M * U_norm);
    double err = (check - Eigen::MatrixXd::Identity(k, k)).norm();
    if (err > 1e-8 * std::max(1, k)) {
        std::cerr << "[normalize] WARNING: Frobenius error = " << err
                  << " (threshold " << 1e-8 * std::max(1, k) << ")" << std::endl;
    } else {
        std::cout << "[normalize] M-orthonormality verified: ||U'MU - I||_F = "
                  << err << std::endl;
    }

    return U_norm;
}

EigenResult removeDC(const EigenResult& result, int* nRemoved, double relTol) {
    if (nRemoved) *nRemoved = 0;
    if (result.k == 0) return result;

    // The null space: every mode with |λ| <= relTol · max|λ| (one per connected component for a Laplacian; the relative Dirac's
    // constant quaternions are 4-fold). Nothing is removed when no mode is that small (e.g. a window away from zero).
    const double scale = result.eigenvalues.head(result.k).cwiseAbs().maxCoeff();
    std::vector<char> drop(result.k, 0);
    int nDrop = 0;
    for (int i = 0; i < result.k; i++)
        if (std::abs(result.eigenvalues(i)) <= relTol * scale) { drop[i] = 1; ++nDrop; }
    if (nRemoved) *nRemoved = nDrop;
    if (nDrop == 0) return result;

    std::cout << "[removeDC] Removing " << nDrop << " null-space mode(s) (|λ| <= " << relTol << " · max|λ|)" << std::endl;

    EigenResult out;
    out.k = result.k - nDrop;
    out.nConverged = result.nConverged;
    out.eigenvalues.resize(out.k);
    out.eigenvectors.resize(result.eigenvectors.rows(), out.k);
    const bool hasKind = result.kind.size() == result.k;
    if (hasKind) out.kind.resize(out.k);

    int j = 0;
    for (int i = 0; i < result.k; i++) {
        if (drop[i]) continue;
        out.eigenvalues(j) = result.eigenvalues(i);
        out.eigenvectors.col(j) = result.eigenvectors.col(i);
        if (hasKind) out.kind(j) = result.kind(i);
        j++;
    }

    return out;
}

} // namespace nxr::operators
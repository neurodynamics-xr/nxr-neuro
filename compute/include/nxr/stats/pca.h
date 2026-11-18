#pragma once
// nxr::stats — principal components of a table of observations x features (the members of a partition by the features measured on each), by the thin singular value decomposition.
//
// PURE MATHS (matrices in, matrices out; no file I/O). Called with any matrix: where the table came from (the cohort's members, a parcellation's tiles, a synthetic array) is the caller's business.
//
// THE DECOMPOSITION (scikit-learn's PCA, svd_solver="full", whose maths is the textbook's): X [n x p] has each column's mean removed, mu; optionally each column is then divided by its standard deviation
// (`Scale::UnitVariance`, ddof 1; the correlation matrix's components); the thin SVD  Xc = U S V^T  gives
//     components      the rows of V^T, [k x p], unit length, mutually orthogonal
//     singular values S
//     explained variance   S^2 / (n - 1)          (the eigenvalues of the sample covariance of Xc)
//     ratio                explained variance / total variance, the total being ||Xc||_F^2 / (n - 1)
//     scores          U S = Xc V, [n x k]
// It is never the p x p covariance matrix: p can be tens of thousands where n is a few hundred. k = the rank of Xc, the components whose singular value exceeds S_max * max(n, p) * eps (numpy's
// matrix_rank rule), at most min(n - 1, p) after centring; every component up to the rank is returned, so any smaller k is a read of the first k.
//
// SIGN CONVENTION (fixed; scikit-learn's svd_flip with u_based_decision=False): each component's sign is chosen so that its entry of largest absolute value (the first, if several are equal) is
// positive; the scores' column takes the same sign. Without one, an SVD's signs are arbitrary and a stored result would not reproduce.
//
// RECONSTRUCTION at k components: Xhat_k = scores[:, :k] components[:k, :] (x scale) + mu: the best rank-k approximation of Xc in the least-squares sense (Eckart-Young). Derived at read, never stored per k.
//
// What it cannot show: a component is a direction of variance in THIS sample; with n observations at most n - 1 components exist, and the leading ones of a small sample are not stable estimates of a
// population's. Components are not "sources" or "networks" of anything until a separate argument says so.
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::stats {

enum class Scale { None, UnitVariance };

struct PcaOptions {
    Scale scale = Scale::None;    // none: the columns keep their units (the covariance's components); unit-variance: divided by their standard deviation (ddof 1; the correlation's)
};

struct PcaResult {
    Eigen::MatrixXd components;                // [k x p], unit rows, the sign convention above
    Eigen::VectorXd singularValues;            // [k], of the centred (and scaled) table, descending
    Eigen::VectorXd explainedVariance;         // [k], singular value^2 / (n - 1)
    Eigen::VectorXd explainedVarianceRatio;    // [k], over the table's total variance (sums to 1 when k is the full rank)
    Eigen::MatrixXd scores;                    // [n x k], U S
    Eigen::VectorXd mean;                      // [p], the columns' means removed
    Eigen::VectorXd scale;                     // [p], the divisor of each column after centring (all 1 for Scale::None)
    std::int64_t rank = 0;                     // k
};

/** The components of X [n x p], n >= 2. A non-finite value, a table with no variance, or (UnitVariance) a constant column is refused with std::invalid_argument saying why. */
PcaResult pca(const Eigen::MatrixXd& X, const PcaOptions& options = {});

/** The rank-k reconstruction scores[:, :k] components[:k, :] (x scale) + mean, [n x p]; k from 0 (the mean) to the components held. The same function reads a stored result. */
Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& scores, const Eigen::MatrixXd& components, const Eigen::VectorXd& mean, const Eigen::VectorXd& scale, int k);

}  // namespace nxr::stats

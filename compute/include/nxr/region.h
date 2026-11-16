#pragma once
// nxr::region — COLLAPSING A REGION to one signal: the weights w a region's elements (vertices) are summed with, z_R = Σ_v w_v z_v, so that region-level
// cross-spectra, lags and maps are made from one signal a region (an ROI, a tile). A LINEAR spatial map (the chain's "members" stage), applied to whatever
// gives the field: with A the field's map from a sufficient statistic (a kernel K, the band's synthesis Φ G) and M that statistic (a cross-spectrum C,
// C_Φ), the regions' cross-spectrum is congruence(W A, M, pairs) (algebra/congruence.h).
//
// THE COLLAPSE IS A NAMED OPTION (Collapse); the plain mean is deliberately not one: it cancels where sources of opposite orientation meet.
//   pca_flip   MNE-Python's `pca_flip` (extract_label_time_course; Brainstorm's PCA scout): the region's first principal component. Here from second
//              moments: w the leading eigenvector of the region's REAL covariance block Re(A_R M A_Rᵀ) (the covariance of the band-passed real signals,
//              up to a factor), unit norm. REAL weights: a complex eigenvector would rotate the region's phase by an arbitrary angle and corrupt every lag
//              to it. Its SIGN as MNE's flip: agreeing with the region's dominant orientation (the first right-singular vector n̄ of its normals, itself
//              signed so that most normals point along it; then the sign making Σ_v sign(n_v · n̄) w_v ≥ 0).
//   The share of the region's power its first component holds (λ₁ / tr) is returned: near 1 a region dominated by one pattern; low, a region whose one
//   signal says little.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::region {

enum class Collapse { PcaFlip };
/** the option by name ("pca_flip"); std::invalid_argument naming the valid ones otherwise */
Collapse collapseOf(const std::string& name);

struct Weights {
  Eigen::SparseMatrix<double> W;   // [regions x n]: row r the weights of region r (0 off it)
  Eigen::VectorXd share;           // [regions] the first component's share of the region's power
  Eigen::VectorXi size;            // [regions] its elements
};
/** the regions' weights. `codes` [n] the region of every element (−1: in none), `regions` how many, A [n x m] real, M [m x m] complex Hermitian, `normals`
 *  [n x 3] (for the sign). Throws std::invalid_argument on mismatched shapes or a code ≥ regions; a region with no element gets an empty row. */
Weights collapse(Collapse how, const std::vector<int32_t>& codes, int regions, const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Eigen::MatrixXd& normals);

}  // namespace nxr::region

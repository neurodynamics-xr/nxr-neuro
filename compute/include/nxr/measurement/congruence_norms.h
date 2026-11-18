#pragma once
// nxr::measurement: the sums of squares behind a quadrature fraction, over groups of rows and pairs of groups.
// Moved from quadrature.h (toolbox refactor): declarations unchanged.
#include <Eigen/Dense>
#include <utility>
#include <vector>

#include "nxr/selection/resolved.h"

namespace nxr::measurement {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

/** The same for every group of rows of A [n x m] (rowGroup[i] in [0, groups) or −1: no group) and every matrix of Cs: out.total and out.quadrature are [groups x Cs.size()], NaN for a group with no row or
 *  a matrix with no value (an empty 0 x 0 matrix in Cs). The Gram of each group is formed once. Throws std::invalid_argument on a shape mismatch.
 *  AREA WEIGHTS (rowWeights [n], empty: all 1): the sums become Σ_ij w_i w_j |T_ij|² and Σ_ij w_i w_j |Im T_ij|² (w the vertex areas: a tile's pairs of vertices count by the mass they stand for, so the value does not
 *  change when a region is sampled more finely). With W = diag(w) on the group's rows, ‖W^½ B X Bᵀ W^½‖² = tr(X G X G) for real X, G = Bᵀ W B: the same trace form with the weighted Gram, nothing else changes. */
struct GroupedCongruenceNorms { MatrixXd total, quadrature; };
GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights = VectorXd());

/** The same for the BLOCKS BETWEEN groups: for each pair (a, b) of groups, T_ab = B_a C B_bᵀ (the part of the source cross-spectrum that couples tile a with tile b; a wave running from one tile into the next is there),
 *  Σ_ij w_i w'_j |T_ab,ij|² and Σ_ij w_i w'_j |Im T_ab,ij|² without forming T_ab. ‖W_a^½ B_a X B_bᵀ W_b^½‖² = tr(X G_b Xᵀ G_a) for real X (G_a = B_aᵀ W_a B_a): with C = Cr + i Cq, Cr symmetric and Cq
 *  antisymmetric, ‖Re T_ab‖² = tr(Cr G_b Cr G_a), ‖Im T_ab‖² = −tr(Cq G_b Cq G_a), ‖T_ab‖² = their sum; each trace is Σ_ij (X G_b)_ij (X G_a)_ji, so with X G_g held for every group once per C a pair costs m² and
 *  nothing more (a = b is the tile's own block, the grouped form above). T_ba is T_abᴴ: the same norms, so a pair is listed once. Also returns the groups' own blocks (as groupedCongruenceNorms).
 *  Memory: the groups' Grams and two m x m products per group at a time. Throws std::invalid_argument on a shape mismatch or a pair outside [0, groups). */
struct PairCongruenceNorms { GroupedCongruenceNorms tiles; MatrixXd pairTotal, pairQuadrature; };   // pair* are [pairs x Cs.size()], NaN for a pair with an empty group or an empty C
PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs,
                                        const VectorXd& rowWeights = VectorXd());
/** The same two functions with the groups as the RESOLVED SELECTION FORM (selection/resolved.h, section 8 of the selection spec): the member number of each row (-1: in no member) and the member count.
 *  A grid (a level of a dyadic grid, an elements grid) or an irregular set whose members are disjoint, resolved by resolve_member_of; the groups are the members, an empty member included (its row is NaN). */
GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights = VectorXd());
PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs,
                                        const VectorXd& rowWeights = VectorXd());

}  // namespace nxr::measurement

#pragma once
// nxr::measurement — the energies of a filterbank's coefficients: MEASUREMENTS over the members, not parts of a bank (a bank only designs; the transforms are wt / gwt / jwt).
//
// PURE MATHS on arrays: `Coefficients` are what a transform returned, one array per member (a time bank's [samples x signals], a graph bank's [vertices x columns], a joint bank's [vertices or modes x time]).
// The reductions here sum |W|^2 over an axis; they were the filterbank headers' `scalogram`, `energies`, `scaleEnergy`, `memberEnergy`, `bandMoments`, `bandPower`.
//
// `scalogram` is the PRESET of the measure of a coefficient block (PI, 2026-11-17: "scalogram = a preset of measure"): the power |w|^2 of each coefficient, per member, unreduced.
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace nxr::measurement {

using CoefficientsR = std::vector<Eigen::MatrixXd>;
using CoefficientsC = std::vector<Eigen::MatrixXcd>;

/** The scalogram PRESET: |w|^2 of every coefficient of every member (the same shape as the member). */
std::vector<Eigen::MatrixXd> scalogram(const CoefficientsC& W);
/** Energy of each member summed over its rows (time): [M x columns]. Parseval for a tight analytic bank with EdgeBins::Single: sum_m ||w_m||^2 = (G^2/2) sum x^2 (two-sided: G^2 sum |z|^2). */
Eigen::MatrixXd energies(const CoefficientsC& W);
/** Energy of each member summed over everything (a joint bank's members, vertices and time): [M]. */
Eigen::VectorXd memberTotals(const CoefficientsC& W);

/** Energy per vertex (and column), summed over members: sum_m |W_m|^2 -> [n x nT]  (Rheome scaleSpectrum). */
Eigen::MatrixXd scaleEnergy(const CoefficientsR& W);
/** Energy per member, summed over vertices: [M x nT]  (Rheome vertexSpectrum). */
Eigen::MatrixXd memberEnergy(const CoefficientsR& W);
/** The same, AREA-WEIGHTED: sum_v w_v |W_m(v)|^2, `weights` [n] the vertex areas (the mass's diagonal). With the mass the eigenbasis is orthonormal for, it equals the band's power in mode coefficients,
 *  sum_k g_m(lambda_k)^2 |c_k|^2 (Parseval), so a tight bank's members add up to the field's power; the unweighted sum (Rheome's reference) counts a densely meshed region more. */
Eigen::MatrixXd memberEnergy(const CoefficientsR& W, const Eigen::VectorXd& weights);
Eigen::MatrixXd memberEnergy(const CoefficientsC& W);
Eigen::MatrixXd memberEnergy(const CoefficientsC& W, const Eigen::VectorXd& weights);

/** A SECOND-MOMENT matrix in mode coefficients (a cross-spectrum, a covariance: C [K x K], Hermitian) seen through every member of a bank, `gains` [K x M] its gains at the modes' eigenvalues: the band's
 *  matrix C_m = G_m C G_m, G_m = diag(gains.col(m)), and its power P_m = tr C_m = sum_k g_m(lambda_k)^2 C_kk. The spatial band power of a STATISTIC, as memberEnergy is of a field: the same number when
 *  C = c c^H for a field's coefficients c (with the area weights). Measures of a band beyond its power (a coherency, the lag content) are read from C_m; this function computes none of them. */
struct BandMoments {
    Eigen::VectorXd power;                     // [M] P_m (real)
    std::vector<Eigen::MatrixXcd> matrices;    // [M] C_m, each [K x K]: only when asked (`withMatrices`)
};
BandMoments bandMoments(const Eigen::MatrixXd& gains, const Eigen::MatrixXcd& C, bool withMatrices = false);
/** The power alone, from the diagonal of C (each mode's power, [K]): P_m = sum_k g_m(lambda_k)^2 p_k. */
Eigen::VectorXd bandPower(const Eigen::MatrixXd& gains, const Eigen::VectorXd& modePower);

}  // namespace nxr::measurement

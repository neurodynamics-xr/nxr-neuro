// nxr::meeg — THE LCMV BEAMFORMER with Brainstorm's 2016 normalisation (the pseudo neural activity index, PNAI), fixed orientation. Written from the published maths
// (Van Veen BD, van Drongelen W, Yuchtman M, Suzuki A (1997) Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Trans
// Biomed Eng 44:867) and from the documented behaviour of Brainstorm's 2016 source estimation (bst_inverse_linear_2016, as process_inverse_2016 calls it; Brainstorm is GPL: its
// code is run only to make the goldens, never copied). Pure: arrays in, arrays out. Depends on Eigen only; no file is read.
//
// THE COVARIANCES, block by block of channel type (Brainstorm's rule since 2011): the cross-covariance between types is set to zero, each type's block made exactly symmetric,
// TRUNCATED to its rank (the singular values s_i = sqrt of the eigenvalues above n x ulp32(s_1)) and then REGULARISED (`TruncatedRegularisation`); the whitener of the block is
// U diag(1 / s) U^T on the kept eigenvectors. The data covariance and the noise covariance are treated the same way, with one rule stated once for both.
//
// THE KERNEL (fixed orientation): with W_d the data covariance's whitener and l_k = G_k n_k the gain of source k along its unit orientation n_k,
//
//     K_k = (W_d l_k)^T W_d / ||W_d l_k||          (one row per source; units: 1 / the data's unit)
//
// i.e. the LCMV weights g^T C^-1 normalised by sqrt(g^T C^-1 g): the beamformer's output for source k has unit variance on the data covariance it was built from (exactly, when
// nothing is regularised), so it is a normalised index, not a current (Brainstorm names it the pseudo neural activity index). The noise covariance does NOT enter the kernel: it is
// regularised and its whitener returned (Brainstorm stores it as the kernel's `Whitener`), as the reference does. Depth weighting does not apply (Brainstorm ignores it for LCMV,
// and any positive scale of a source's gain cancels in the normalisation). The sign of a row follows its orientation (n and -n give opposite rows).
//
// CALLER'S JOB, before the call: G and both covariances restricted to the good channels, the recording's projectors and an average reference applied as the reference does
// (Brainstorm: G := A P G, C := A P C P^T A^T for both covariances; nxr::pipelines::graph::minimum_norm_inputs). The orientations are the caller's (the surface's normals).
//
// What it cannot resolve: correlated sources (two sources whose time courses are correlated partly cancel, the beamformer's known blind spot); the regularisation trades spatial
// resolution for stability; the index is relative to the data covariance's own span, so two kernels made from different spans are not on one scale.
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace nxr::meeg {

// How a TRUNCATED covariance block is regularised (Brainstorm 2016's truncate_and_regularize_covariance), s the singular values kept (the square roots of the eigenvalues).
enum class TruncatedRegularisation {
  Median,     // s := max(s, median(s)): the eigenvalue tail flattened to the median (Brainstorm's 'median'; the reference's LCMV default)
  Ridge,      // the whitener U diag(1 / (s + sqrt(rho))) U^T, rho = ridgeFraction x the LARGEST eigenvalue of the block before truncation (Brainstorm 2016's 'reg')
  Diagonal,   // the truncated block's diagonal: W = diag(1 / sqrt(diag C))   (Brainstorm's 'diag')
  None        // W = U diag(1 / s) U^T on the kept eigenvectors (Brainstorm's 'none')
};
struct TruncatedOptions {
  TruncatedRegularisation regularisation = TruncatedRegularisation::Median;
  double ridgeFraction = 0.1;           // Brainstorm's NoiseReg: used by Ridge only (the reference's process passes 0.1 with 'median', where it has no effect)
};
struct TruncatedWhitening {
  Eigen::MatrixXd C;                    // [M x M] the covariance as regularised (block-diagonal by type, symmetric)
  Eigen::MatrixXd W;                    // [M x M] its symmetric whitener (block-diagonal by type)
  std::vector<int> ranks;               // the rank kept in each type's block, in order of the types' first appearance
};
/** The covariance C [M x M] truncated and regularised block by block of channel `types` (one string per channel), and its whitener. std::invalid_argument for a C that is not
 *  square, not finite, or whose types do not have M entries, and for a block with no eigenvalue above the rank tolerance. */
TruncatedWhitening truncatedWhitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const TruncatedOptions& opt = {});

struct LcmvOptions {
  TruncatedOptions covariance;          // the rule for both covariances (Brainstorm passes one NoiseMethod for both): median, 0.1 by default
  bool detectDiagonal = true;           // a DIAGONAL noise covariance (||C||_F - ||diag C||_F below ulp32(||diag C||_F)) switches the rule to Diagonal for BOTH covariances, as the 2016 code does
};
struct LcmvKernel {
  Eigen::MatrixXd K;                    // [N x M] one row per source (fixed orientation), in 1 / the channels' unit
  Eigen::MatrixXd dataWhitener;         // [M x M] W_d, the whitener the kernel is made with
  Eigen::MatrixXd noiseWhitener;        // [M x M] the noise covariance's whitener under the same rule (Brainstorm's `Whitener`; not used by the kernel)
  Eigen::MatrixXd dataCovariance, noiseCovariance;   // [M x M] both as regularised
  std::vector<int> dataRanks, noiseRanks;
  Eigen::VectorXd gainNorm;             // [N] ||W_d l_k||: the whitened gain's norm of each source (0: a source the data cannot see, its row left zero)
  TruncatedRegularisation applied = TruncatedRegularisation::Median;   // the rule both covariances were regularised by (Diagonal when the noise covariance is diagonal)
  std::vector<std::string> notes;       // what was decided (the diagonal rule, a source with no whitened gain)
};
/** The fixed-orientation LCMV kernel from the gain G [M x 3N] (x y z columns per source), the unit-normalised orientations [N x 3] of the sources, the data covariance and the noise
 *  covariance [M x M] (the same channels, in G's row order) with one type per channel. std::invalid_argument on a shape mismatch, a non-finite input, or a zero orientation. */
LcmvKernel lcmvKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& orientations, const Eigen::MatrixXd& dataCovariance, const Eigen::MatrixXd& noiseCovariance,
                      const std::vector<std::string>& types, const LcmvOptions& opt = {});

}  // namespace nxr::meeg

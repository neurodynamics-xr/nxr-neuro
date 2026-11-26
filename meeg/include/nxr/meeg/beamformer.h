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

// ── THE LCMV BEAMFORMER UNDER MNE-PYTHON'S CONVENTIONS (mne.beamformer.make_lcmv, MNE-Python 1.13; BSD: read, the maths written here) ─────────────────────────────────────────
// Written from Van Veen et al. 1997, Sekihara & Nagarajan 2008 (the weight normalisations) and MNE-Python's documented make_lcmv. Two steps, each a function of matrices:
//
// 1. THE WHITENER of a noise covariance C (already projected, P C P^T, by the caller): one symmetric block per MODALITY (MEG: magnetometers and gradiometers together; EEG; sEEG; ...),
//    the covariance across modalities set to zero; in each block the eigenvalues below the block's RANK set to zero, W_b = U diag(1 / sqrt(e)) U^T on the eigenvalues kept (e > 0).
//    The rank of a block is the caller's (MNE-Python's rank 'info': its channels less the projectors on them). No regularisation (MNE-Python uses the noise covariance as given).
//    With no noise covariance MNE-Python whitens by an identity covariance of the same projectors: the caller hands P P^T and the whitener is P.
//
// 2. THE FILTER, from the gain G [M x 3N] (x y z per source), the unit normals [N x 3], the data covariance C_d [M x M] (as estimated: the whitener carries the projectors) and the
//    whitener W [M x M]: with L = W G and C = W C_d W^T (made exactly symmetric),
//      C^-1 := the pseudo-inverse of C + mu I on its `rank` largest singular values, mu = reg x the mean of C's singular values (all M of them: diagonal loading, Gross & Ioannides 1999);
//      per source k, L_k = the source's M x 3 block (a source whose singular values s_1 > 1e6 s_3 is refused: MNE-Python's reduce_rank, not ported);
//      orientation: free (L_k as it is, 3 rows: x y z) | normal (L_k n_k) | max-power (L_k o_k, o_k the eigenvector of the largest |eigenvalue| of (L_k^T C^-2 L_k)^+ (L_k^T C^-1 L_k)
//      (of (L_k^T C^-1 L_k)^+ when weight_norm is none), its sign that of o_k . n_k);
//      the unit-gain filter F_k = (L_k^T C^-1 L_k)^+ L_k^T C^-1 (3 x 3 pseudo-inverse with rcond 1e-7; 1 / x for one orientation);
//      weight_norm: none (F_k) | unit-noise-gain (each row of F_k divided by its norm) | unit-noise-gain-invariant ((B B^T)^(-1/2) B, B = L_k^T C^-1: rotation invariant; for one orientation
//      B / ||B||) | nai (unit-noise-gain divided by sqrt(max(the rank-th largest eigenvalue of C, mu)): the neural activity index).
//    The operator from the channels as recorded is K = F W (one row a source, or 3: x y z interleaved for free orientation).
//
// What it cannot resolve: correlated sources partly cancel; the regularisation trades resolution for stability; with a weight normalisation the output is an index (noise-normalised),
// not a current, and its scale depends on the data covariance's span; free orientation's unit-noise-gain (not invariant) depends on the frame of x y z.
enum class LcmvOrientation { Free, Normal, MaxPower };
enum class LcmvWeightNorm { None, UnitNoiseGain, UnitNoiseGainInvariant, Nai };

/** The modality of a channel type: "MEG" for every MEG type (magnetometers, gradiometers, axial), else the type itself. */
std::string modalityOf(const std::string& type);
/** The channels of each modality, in order of the modalities' first appearance. */
std::vector<std::vector<Eigen::Index>> modalityBlocks(const std::vector<std::string>& types);

struct RankWhitening {
  Eigen::MatrixXd W;                    // [M x M] symmetric, block-diagonal by modality
  std::vector<std::string> modalities;  // in order of first appearance
  std::vector<int> ranks;               // the rank kept in each, as given
};
/** The rank-limited whitener of C [M x M] (step 1 above), `ranks` one per modality in order of first appearance. std::invalid_argument for a C that is not square or not finite, types
 *  without M entries, or a rank outside [1, the modality's channels]. */
RankWhitening rankWhitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const std::vector<int>& ranks);

struct MneLcmvOptions {
  double reg = 0.05;                                                    // MNE-Python's reg: the diagonal loading, a fraction of the mean singular value of the whitened data covariance
  LcmvOrientation orientation = LcmvOrientation::Free;                  // MNE-Python's pick_ori: None (free) | 'normal' | 'max-power'
  LcmvWeightNorm weightNorm = LcmvWeightNorm::UnitNoiseGainInvariant;   // MNE-Python's weight_norm, its default 'unit-noise-gain-invariant'
};
struct MneLcmvKernel {
  Eigen::MatrixXd K;                    // [N x M] or [3N x M] (free: x y z interleaved per source), on the channels as recorded
  Eigen::MatrixXd orientations;         // [N x 3] the max-power orientations (unit, sign along the normal); empty otherwise
  double loading = 0;                   // mu, the diagonal loading
  double noise = 0;                     // nai only: the noise level the filter is divided by (its square root)
  int rank = 0;
  std::vector<std::string> notes;
};
/** The LCMV filter of step 2 above. std::invalid_argument on a shape mismatch, a non-finite input, a zero normal, a rank outside [1, M], a negative reg, or a source whose gain is
 *  singular. */
MneLcmvKernel mneLcmvKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& normals, const Eigen::MatrixXd& dataCovariance, const Eigen::MatrixXd& whitener, int rank,
                            const MneLcmvOptions& opt = {});

}  // namespace nxr::meeg

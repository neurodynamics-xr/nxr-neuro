// nxr::meeg — the unconstrained minimum-norm kernel (imaging kernel), written from the published maths (docs/specs/2026-11-10-neuro-forward-inverse.md):
//   Hamalainen & Ilmoniemi 1994 (minimum-norm estimate), Lin et al. 2006 (depth weighting), Gramfort et al. 2014 (MNE-Python's conventions).
// Pure: arrays in, arrays out. Depends on Eigen only; no file is read.
//
//   K = diag(alpha x 1_3) . Lambda L^T V diag(1 / (Lambda sigma^2 + 1)) V^T W      (L = W G diag(alpha x 1_3), L L^T = V diag(sigma^2) V^T)
//
// The estimate of the source currents (A.m) from the channels as recorded is J = K b. The defaults reproduce Brainstorm's default unconstrained current
// kernel; the other conventions (MNE-Python's) are named options, never silent differences.
//
// CALLER'S JOB, before the call: G and C are restricted to the good channels; with SSP projectors P, G := P G and C := P C P^T; with EEG, sEEG or ECoG
// channels, the average reference A is applied, G := A G and C := A C A^T. K is then applied to the recording as read (the whitener's null space removes
// the projected and common-mode parts).
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace nxr::meeg {

// How many eigenvalues of a covariance block count as non-zero: those above n x ulp(largest). ulp of the single-precision float for the reference, of the
// double for a tighter rule.
enum class RankTolerance { SinglePrecision, DoublePrecision };
// How the whitener is regularised.
enum class Regularisation {
  Ridge,      // W = U_r diag(1 / sqrt(s_r^2 + rho)) U_r^T, rho = ridgeFraction x (mean of ALL eigenvalues of the block)   (Brainstorm 'reg', MNE's)
  Diagonal,   // W = diag(1 / sqrt(diag(C_r)))                                                                              (Brainstorm 'diag')
  None        // W = U_r diag(1 / s_r) U_r^T                                                                                (Brainstorm 'none')
};
// The form of the returned whitener. The kernel is the same either way.
enum class WhitenerForm { Symmetric /* U f(S) U^T, [M x M] (Brainstorm) */, Pca /* f(S) U^T, [rank x M] (MNE-Python) */ };
// The norm nu_k of source k that depth weighting uses.
enum class DepthNorm {
  Frobenius,  // ||G_k||_F^2 of the M x 3 block of source k, all channels together (Brainstorm)
  Spectral    // the largest eigenvalue of G_k^T G_k (MNE-Python)
};
// Where the depth limit is applied.
enum class DepthLimitOn {
  Weight,     // a_k = nu_k^p, a_k := max(a_k, max a / limit^2), alpha_k = sqrt(max a / a_k) in [1, limit]   (Brainstorm)
  Norm        // u_k = 1 / nu_k, u_hat = the smallest u_k above limit^2 x min u (max u if none), alpha_k = min(u_k / u_hat, 1)^(p/2)   (MNE-Python)
};
// Which channels the norm is computed on.
enum class DepthChannels {
  All,        // every channel of G (Brainstorm)
  OneType     // one type: "MEG GRAD" if present, else "MEG MAG", else "EEG", else all   (MNE-Python)
};
// What Lambda's mean eigenvalue is taken over.
enum class SnrNormalisation {
  AllChannels,   // Lambda = snr^2 / ((1/M) sum_i sigma_i^2), the mean over all M rows of L, zeros included (Brainstorm)
  WhitenerRank   // Lambda = snr^2 / ((1/r) sum_i sigma_i^2), r = the whitener's rank (MNE-Python)
};

struct WhitenerOptions {
  Regularisation regularisation = Regularisation::Ridge;
  double ridgeFraction = 0.1;                                 // Brainstorm's noise regularisation 'reg' (NoiseReg 0.1); MNE-Python's reg=0.1
  RankTolerance rankTolerance = RankTolerance::SinglePrecision;
  bool detectDiagonal = true;                                 // a diagonal covariance is whitened by its diagonal (Brainstorm's behaviour), and says so in `notes`
  bool jointTypes = false;                                    // true (MNE-Python): the types of a modality (MEG MAG with MEG GRAD) are whitened together, their cross-covariance kept; false (Brainstorm): one block per type. Modalities (MEG, EEG) are always separate
  bool ridgeOverRank = false;                                 // the ridge's mean eigenvalue is tr(C_t) / n_t (false, Brainstorm) or tr(C_t) / rank_t (true, MNE-Python: the ridge of a projected block is that fraction of its mean non-zero eigenvalue)
  WhitenerForm form = WhitenerForm::Symmetric;
};
struct DepthOptions {
  bool enabled = true;
  double exponent = 0.5;                                      // Brainstorm's WeightExp; MNE-Python's default is 0.8
  double limit = 10;                                          // Brainstorm's WeightLimit (the ratio of the largest to the smallest weight is limit^2 in variance)
  DepthNorm norm = DepthNorm::Frobenius;
  DepthLimitOn limitOn = DepthLimitOn::Weight;
  DepthChannels channels = DepthChannels::All;
};
struct KernelOptions {
  WhitenerOptions whitener;
  DepthOptions depth;
  double snr = 3;                                             // the amplitude ratio of Brainstorm's SnrFixed (lambda^2 = 1 / snr^2 in MNE-Python)
  SnrNormalisation snrNormalisation = SnrNormalisation::AllChannels;
};

// MNE-Python's conventions as named options: joint MEG types, ridge over the rank, PCA whitener, depth exponent 0.8 by the spectral norm limited on the norm, on one channel type, and Lambda by the
// whitener's rank.
KernelOptions mneKernelOptions();

struct Whitener {
  Eigen::MatrixXd W;                       // [M x M] (Symmetric) or [rank x M] (Pca), the channels' order
  int rank = 0;                           // the sum of the blocks' ranks
  std::vector<std::string> notes;         // what was decided (a diagonal covariance, a rank-deficient block)
};
struct Kernel {
  Eigen::MatrixXd K;                       // [3N x M], rows x, y, z interleaved per source; A.m per channel unit
  Eigen::MatrixXd W;                       // the whitener used
  Eigen::VectorXd depthWeights;            // [N] alpha_k (1 when depth weighting is off)
  double lambda = 0;                       // Lambda: the source covariance scale, from the SNR
  int whitenerRank = 0;
  std::vector<std::string> notes;
};

// The whitener of a noise covariance C [M x M], block by block of channel type. std::invalid_argument for a C that is not square, has a NaN, or whose
// `types` do not have M entries.
Whitener whitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const WhitenerOptions& opt = {});
// One scalar alpha_k per source of G [M x 3N] (all 1 when depth weighting is off). std::invalid_argument when the columns of G are not 3N, G has a NaN, or `types` do not have M entries.
Eigen::VectorXd depthWeights(const Eigen::MatrixXd& G, const std::vector<std::string>& types, const DepthOptions& opt = {});
// The kernel from the gain G [M x 3N] and the noise covariance C [M x M], with one type string per channel ("MEG MAG", "MEG GRAD", "MEG", "EEG", "SEEG", "ECOG").
Kernel unconstrainedKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& C, const std::vector<std::string>& types, const KernelOptions& opt = {});

// The spacing of the single-precision floats at x > 0 (2^(floor(log2 x) - 23)), the reference's rank unit; 0 for x = 0.
double ulp32(double x);

}  // namespace nxr::meeg

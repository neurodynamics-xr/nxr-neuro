// nxr::meeg — source time series and source power from a kernel (the maths of applying an imaging kernel). Pure: arrays in, arrays out; Eigen only.
//
//   J = K X       K [(o N) x M], o = 1 (fixed orientation) or 3 (free: rows x, y, z interleaved per source, as `unconstrainedKernel`), X [M x T] real, or complex
//                 coefficients (FFT bins, wt coefficients) of the same channels.
//   Orientation   what a free kernel's three rows become: Components (kept, 3N rows) or Norm (Brainstorm's "norm": sqrt(Jx^2 + Jy^2 + Jz^2) per source; for complex
//                 coefficients the amplitude sqrt(|Jx|^2 + |Jy|^2 + |Jz|^2)). A fixed kernel (o = 1): Components is K X, Norm is |K X|.
//
// SOURCE POWER FROM THE SENSOR CROSS-SPECTRUM. The power of source s at a frequency is the diagonal of K S(f) K^H, S(f) the channels' cross-spectral density; for a real K only
// Re S(f) enters, so sourcePower(K, S) = diag(K Re(S) K^T) is exactly Welch run on K x, at the cost of a channels x channels matrix per band, not a spectrum per source. With a free
// kernel the power of the three orientations is SUMMED (Orientation::Norm; the square of the norm) or kept apart (Components).
//
// BANDS. Brainstorm's default bands for "Group in time or frequency bands" (its tutorials: tutorial_omega): delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59,
// gamma2 60-90 Hz; a band's value is the MEAN of the spectrum's bins from lo to hi inclusive.
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace nxr::meeg {

enum class Orientation { Components, Norm };

struct Band {
  std::string name;
  double lo = 0, hi = 0;   // Hz, both inclusive
};
std::vector<Band> brainstormBands();

/** J = K X. Orientation::Components: [(oN) x T]; Norm: [N x T] with the norm over each source's `orientations` rows. */
Eigen::MatrixXd applyKernel(const Eigen::MatrixXd& K, const Eigen::MatrixXd& X, int orientations = 3, Orientation orientation = Orientation::Components);
/** J = K X for complex coefficients [M x T]: [(oN) x T] complex. */
Eigen::MatrixXcd applyKernel(const Eigen::MatrixXd& K, const Eigen::MatrixXcd& X);
/** sqrt(sum over the source's rows of |J|^2): [N x T] from [(oN) x T]. */
Eigen::MatrixXd amplitude(const Eigen::MatrixXcd& J, int orientations = 3);
/** applyKernel then amplitude, in one go: [N x T] (Orientation::Norm) or |J| of every row (Components). */
Eigen::MatrixXd applyKernelAmplitude(const Eigen::MatrixXd& K, const Eigen::MatrixXcd& X, int orientations = 3, Orientation orientation = Orientation::Norm);

/** diag(K Re(S) K^T) for each S (each [M x M], Hermitian cross-spectrum or its real part): [rows x S.size()], or [N x S.size()] with Orientation::Norm (the three powers summed). */
Eigen::MatrixXd sourcePower(const Eigen::MatrixXd& K, const std::vector<Eigen::MatrixXd>& S, int orientations = 3, Orientation orientation = Orientation::Norm);
/** Brainstorm's relative power: each column of P divided by `total` (one value per row of P). A zero total gives 0. */
Eigen::MatrixXd relativePower(const Eigen::MatrixXd& P, const Eigen::VectorXd& total);

}  // namespace nxr::meeg

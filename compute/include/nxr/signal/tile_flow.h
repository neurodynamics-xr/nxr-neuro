// nxr::signal — THE FLOW BETWEEN THE TILES OF A SURFACE read from a factored cross-spectrum, as pure functions (docs/specs/2026-11-24-connectome-flow.md). The reference is
// nxr's flow through the fibre network (`flownet.py`): each tile's current is a source kernel averaged over the tile's vertices (three Cartesian components, no orientation
// assumed); the tiles' cross-spectrum S_ij = Σ_c z_(i,c) conj(z_(j,c)) gives the edge flow F_ij = Im S_ij / sqrt(S_ii S_jj) (> 0 when tile j lags tile i).
//
//   servedSynthesis            the synthesis of a serving operator's coordinates back to its columns (channels): its pseudo-inverse [C x K] and its rank (K when complete)
//   tileOperator               the kernel's rows [3V x C] (vertex-major: 3 v + component) averaged over each tile's vertices, composed with a synthesis [C x K]: ONE operator
//                              [3T x K] (tile-major: 3 t + component), the averaging applied first (the smallest size)
//   accumulateTileCrossSpectra one chunk's tile coefficients Z [3T x bins] (the operator applied to the chunk's spectrum) added to each atom's sum
//                              S_a += Σ_c Z_c diag(g_a²) Z_cᴴ [T x T] (g_a the atom's gains on the bins; only the bins where the gain is not zero are read)
//   tileTimeCells              one chunk's weighted spectrum (the atom's gains applied) back in time through the tile operator (composed on the atom's bins first, one inverse
//                              transform per row, the chunk's own samples kept), summed over cells of `cellSamples`: per cell and pair i < j Σ_c Σ_t Im(z_i conj(z_j)), per cell
//                              and tile Σ_c Σ_t |z_i|² (a measurement over the cells; the per-sample products are never held beyond one chunk)
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <vector>

namespace nxr::signal {

struct ServedSynthesis {
  Eigen::MatrixXd synthesis;   // [C x K]
  std::int64_t rank = 0;
};
/** serving [K x C] → its pseudo-inverse [C x K] (a complete orthogonal decomposition; rank threshold 1e-10 of the largest singular value) */
ServedSynthesis servedSynthesis(const Eigen::MatrixXd& serving);

/** kernelRows [3V x C] vertex-major, tileOfVertex [V] (0 .. tiles - 1, or -1 off the tiles), synthesis [C x K] → [3T x K] tile-major. Each tile holds at least one vertex. */
Eigen::MatrixXd tileOperator(const Eigen::MatrixXd& kernelRows, const std::vector<std::int32_t>& tileOfVertex, std::int64_t tiles, const Eigen::MatrixXd& synthesis);

/** tileCoefficients [3T x bins] tile-major, gains [atoms x bins] → sums[a] += Σ_c Z_c diag(g_a²) Z_cᴴ; sums is resized to atoms x [T x T] zeros when empty */
void accumulateTileCrossSpectra(const Eigen::MatrixXcd& tileCoefficients, const Eigen::MatrixXd& gains, std::vector<Eigen::MatrixXcd>& sums);

struct TileCells {
  Eigen::MatrixXd pairImaginary;   // [P x cells] pairs i < j in nxr::meeg::pairIndex order
  Eigen::MatrixXd power;           // [T x cells]
};
/** weightedSpectrum [K x bins] (the bins [firstBin, firstBin + bins) of a transform of transformSamples), the chunk's own samples [ownFirst, ownFirst + ownSamples) of that
 *  transform, cells of cellSamples (a trailing part cell left out), tileOperator [3T x K] tile-major. */
TileCells tileTimeCells(const Eigen::MatrixXcd& weightedSpectrum, std::int64_t firstBin, std::int64_t transformSamples, std::int64_t ownFirst, std::int64_t ownSamples,
                        std::int64_t cellSamples, const Eigen::MatrixXd& tileOperator);

}  // namespace nxr::signal

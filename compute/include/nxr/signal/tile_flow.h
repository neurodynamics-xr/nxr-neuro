// nxr::signal — THE FLOW BETWEEN THE TILES OF A SURFACE read from a factored cross-spectrum, as pure functions (docs/specs/2026-11-24-connectome-flow.md). The reference is
// nxr's flow through the fibre network (`flownet.py`): each tile's current is a source kernel averaged over the tile's vertices (three Cartesian components, no orientation
// assumed); the tiles' cross-spectrum S_ij = Σ_c z_(i,c) conj(z_(j,c)) gives the edge flow F_ij = Im S_ij / sqrt(S_ii S_jj) (> 0 when tile j lags tile i).
//
//   servedSynthesis            the synthesis of a serving operator's coordinates back to its columns (channels): its pseudo-inverse [C x K] and its rank (K when complete)
//   tileOperator               the kernel's rows [3V x C] (vertex-major: 3 v + component) averaged over each tile's vertices, composed with a synthesis [C x K]: ONE operator
//                              [3T x K] (tile-major: 3 t + component), the averaging applied first (the smallest size)
//   (the tiles' cross-spectrum sums and time cells: flow_readout.h, pairFlow and timeCellPairFlow on the tile operator made component-major, every pair i < j;
//   the stored S_ij = Σ z_i conj(z_j) is the conjugate of their pair sums)
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

}  // namespace nxr::signal

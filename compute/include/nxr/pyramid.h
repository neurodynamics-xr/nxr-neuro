#pragma once
// nxr::pyramid — the TRANSFORM maths of a band's analytic signal (the chain's pointwise stages). PURE MATHS: arrays in, arrays out — no file, no chunk, no recording, no dataset.
//
// What this header held and no longer does: the band gains (Morlet, power-partition) are the families of nxr::signal::timefilterbank (`bandGains`, Family::PowerPartition),
// and a band's analytic signal is the analytic wavelet filterbank's member (timefilterbank::analysis). Only `frameKernel` is left: it moves to meeg with the behaviour round's row 3.
//   frameKernel       a free-orientation kernel projected on the canonical frames (K1, K2, Kn), the linear map a cortical chain folds into its operator
// The reductions this file used to carry (band sums, tensor sums, the frame of a sample, the good samples of a frame) are `nxr::measurement::reduce` over a product selection
// (spec docs/specs/2026-11-17-consolidation.md): a reduction across time or space is never a function here. The octave bank is `selection::octave_bands` (selection/bands.h).
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nxr::pyramid {

// ── THE FREE-ORIENTATION KERNEL (nxr `measure.vectors.band_tensor`) ──
/** A free-orientation kernel [3V x M] (rows 3v + (x, y, z): Brainstorm's order) projected on the canonical frames: K1 = e1·K, K2 = e2·K, Kn = n·K, each [V x M]. `e1`, `e2`, `n` [V x 3] (NaN at a vertex with no
 *  frame), `singular` [V] marks those: as nxr, a singular vertex takes an ARBITRARY tangent basis (tangent_basis of its normal, (0, 0, 1) where that is NaN) and a zero normal row, so it is out of the nine tensor
 *  sums (its weight is 0) yet its tangential power is in `total`. Throws std::invalid_argument on a shape mismatch. */
struct FrameKernel { Eigen::MatrixXd K1, K2, Kn; };
FrameKernel frameKernel(const Eigen::MatrixXd& K3, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& n, const std::vector<char>& singular, int threads = 1);

}  // namespace nxr::pyramid

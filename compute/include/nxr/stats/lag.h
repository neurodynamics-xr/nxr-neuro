#pragma once
// nxr::stats — DIAGNOSING LAGS: how big a lag is against its own noise, where the lags of a graph fail to close, at which distance they are measured, and
// a surrogate that destroys them. PURE MATHS (arrays in, arrays out). The lags themselves come from a cross-spectrum at pairs (S_uv = ⟨z_u z̄_v⟩: from
// `measurement::reduce` of the outer form in sensor space and `algebra::congruence` K C Kᵀ at the pairs, or in mode space through `filterbank::bandMoments`).
//
//   lagStatistics(S, Pu, Pv, n)      per pair: the lag θ = arg S_uv (rad), the coherence |γ| = |S_uv| / √(P_u P_v), its STANDARD ERROR
//                                    se(θ) ≈ √((1 − |γ|²) / (2 n |γ|²)) (Bendat & Piersol: n independent complex estimates averaged) and the ratio |θ| / se,
//                                    the lag's signal-to-noise. `n` from timebank::independentEstimates (the band's equivalent bandwidth times the window).
//                                    Option `debias`: |γ|² corrected for its upward bias at small n, (n|γ̂|² − 1) / (n − 1), floored at 0.
//   holonomy(edges, θ, faces)        per face, the sum of its three edge lags round it (CCW), wrapped into (−π, π]: 0 where the lags close (they come from
//                                    one phase map: frustration-free), non-zero where they do not. The edge-lag counterpart of magnetic::winding (which
//                                    is a field's).
//   circularShift(Z, shifts)         each row of Z rotated by its own shift (within the block): each channel's spectrum kept exactly, the lags between
//                                    channels destroyed: the time-shift SURROGATE of a null. The draw of the shifts is the caller's.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <vector>

namespace nxr::stats {

using Complex = std::complex<double>;
using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Faces = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct LagStatistics {
  Eigen::VectorXd lag;           // θ, rad, in (−π, π]
  Eigen::VectorXd coherence;     // |γ| (debiased when asked)
  Eigen::VectorXd stderror;      // se(θ), rad (∞ where |γ| = 0)
  Eigen::VectorXd snr;           // |θ| / se
};
/** per pair: S (complex), P_u, P_v (the two ends' powers), n independent complex estimates. Throws std::invalid_argument on mismatched sizes or n ≤ 1. */
LagStatistics lagStatistics(const std::vector<Complex>& S, const Eigen::VectorXd& Pu, const Eigen::VectorXd& Pv, double n, bool debias = false);

/** per face (CCW), the wrapped sum of its edge lags; `lag[e]` is θ_uv for edges(e) = (u, v) (θ_vu = −θ_uv). Throws when a face's edge is not in `edges`. */
Eigen::VectorXd holonomy(const Edges& edges, const Eigen::VectorXd& lag, const Faces& faces);

/** each row r of Z [rows x n] circularly shifted by shifts[r] samples (any integer; taken modulo n) */
Eigen::MatrixXcd circularShift(const Eigen::MatrixXcd& Z, const std::vector<std::int64_t>& shifts);

}  // namespace nxr::stats

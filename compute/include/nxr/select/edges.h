#pragma once
// nxr::select — WHERE A BURST STARTS AND ENDS ON ITS ENVELOPE: the tangent edges (spec docs/specs/2026-11-20-gabor-bursts.md, piece 8; the reference detector's
// tangent_edges).
// PURE MATHS: an envelope (any amplitude series: the caller reads it, e.g. by an atom at the burst's frequency), the burst's centre and half width in its samples, in; the edges in (fractional)
// samples of that series, out. The function never knows where the envelope came from.
//
//   peak     the envelope's maximum within centre ± halfWidth
//   troughs  going outward from the peak, the first local minimum below half the peak; else the lowest point within `reach` samples
//   onset    where the tangent at the steepest rise (between the trough before and the peak) meets the trough's level; offset: the same on the fall
// The slope is numpy's gradient (central differences, one-sided at the ends). Not ok (no edges) when a trough is under 2 samples from the peak, or the rise is not rising or the fall not falling.
#include <cstdint>

#include <Eigen/Dense>

namespace nxr::select {

struct TangentEdges {
  bool ok = false;
  double onset = 0, offset = 0;                            // fractional samples of the envelope
  std::int64_t peak = -1, troughBefore = -1, troughAfter = -1;
};
TangentEdges tangentEdges(const Eigen::Ref<const Eigen::VectorXd>& envelope, std::int64_t centre, std::int64_t halfWidth, std::int64_t reach);

}  // namespace nxr::select

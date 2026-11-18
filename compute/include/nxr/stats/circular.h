#pragma once
// nxr::stats: circular statistics of directions.
// Moved from quadrature.h (toolbox refactor): declarations unchanged.
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::stats {

using Eigen::VectorXd;

// ── directions ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** R = |mean e^{i a}| in [0, 1]. */
double meanResultantLength(const VectorXd& angles);
/** The mean direction arg mean e^{i a} (rad). */
double meanDirection(const VectorXd& angles);
struct Rayleigh { double R = 0, p = 1, meanDirection = 0; std::int64_t n = 0; };
/** Rayleigh's test of uniformity: R, and p ~ exp(-n R^2) (the large-sample approximation nxr used), and the mean direction. */
Rayleigh rayleighTest(const VectorXd& angles);

}  // namespace nxr::stats

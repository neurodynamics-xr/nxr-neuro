#pragma once
// nxr::signal — THE APERIODIC BACKGROUND OF A SPECTRUM: specparam's aperiodic model (Donoghue et al. 2020), as nxr's port `nxr.spectral.aperiodic.fit` (itself matching
// the Studio's `fitAperiodic` and Rheome's `+rheome/+spectral/aperiodic`). PURE MATHS: one spectrum in, the fit's parameters out.
//
//   fitAperiodic       log10 power against log10 frequency over the points in [low, high] Hz (all when low > high is not given: low 0, high infinity):
//                      without a knee, offset − exponent log10 f (least squares, numpy polyfit of degree 1); with a knee, offset − log10(knee + f^exponent), the
//                      exponent on the grid 0.1 .. 6 (step 0.1) and the knee on {0, 10^−2, 10^−1.5, …, 10^3}, the offset in closed form, the best kept (nxr's grid).
//                      `iterations` passes: after each but the last, the points above the fit by more than `threshold` standard deviations (n − 1) are dropped;
//                      fewer than 4 left restores every point and stops. r² and the points used of the last fit.
//   evaluateAperiodic  the fitted background's linear power at frequencies f
// Rheome's warnings hold: fit a smoothed spectrum (an average over segments or chunks), broadband; fit and target in the same units.
#include <Eigen/Dense>
#include <limits>

namespace nxr::signal {

struct AperiodicFit { double offset = 0, exponent = 0, knee = 0, r2 = 0; Eigen::Index pointsUsed = 0; };
/** `power` [points] linear, `frequencies` [points] > 0 Hz. Throws std::invalid_argument on sizes that differ, a frequency <= 0 in the range, or fewer than 2 points. */
AperiodicFit fitAperiodic(const Eigen::VectorXd& power, const Eigen::VectorXd& frequencies, bool withKnee, int iterations = 3, double threshold = 2.0,
                          double low = 0.0, double high = std::numeric_limits<double>::infinity());
Eigen::VectorXd evaluateAperiodic(const AperiodicFit& fit, const Eigen::VectorXd& frequencies);

}  // namespace nxr::signal

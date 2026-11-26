#pragma once
// nxr::signal — DOES THE FLOW MOVE AT THE SAME SPEED AND IN THE SAME DIRECTION ACROSS FREQUENCIES? (spec docs/specs/2026-11-24-bispectrum.md, "The flow across frequencies").
// PURE MATHS: arrays in, arrays out. A statistic over the members of the frequency atoms' level, read from a flow's per-atom results (the sums the flow readout stored: power, phase
// current per edge and per vertex, Hodge-Dirac flow coefficients), never from a recording. Per atom a with centre frequency f_a:
//   edgeCosine(a, b)      <J_a, J_b> / (|J_a| |J_b|), the edge phase currents compared in the edges' Hodge inner product (weights star1): 1 the same direction everywhere,
//                         0 unrelated, -1 opposite
//   travelCosine(a, b)    the same for the exact (travel) Hodge-Dirac coefficients; rotationCosine(a, b) for the coexact (rotation) coefficients (an orthonormal basis: plain dot products)
//   directionConsistency  per vertex, |Σ_a u_a| / n with u_a the unit vertex phase current of atom a (the atoms whose current is not zero): 1 every atom points the same way there
//   speed slope           per vertex, the least-squares slope of log(apparent speed) against log(f) over the atoms, the apparent speed 2 pi f_a power / |vertex phase current|
//                         (an upper bound): 0 the same speed at every frequency (no dispersion), 1 the same wavelength (speed proportional to frequency); the intercept beside it
// The sign of the phase current (travel from tail to tip gives J < 0) is common to every atom, so none of these depends on it.
#include <Eigen/Dense>
#include <vector>

namespace nxr::signal {

struct FlowAcrossFrequencies {
  Eigen::MatrixXd edgeCosine, travelCosine, rotationCosine;   // [atoms x atoms]
  Eigen::VectorXd directionConsistency;                       // [vertices]
  Eigen::VectorXd speedSlope, speedIntercept;                 // [vertices] (NaN where fewer than two atoms carry a current)
  Eigen::VectorXd atomsUsed;                                  // [vertices] the atoms with a non-zero current and power
};
/** `edgeCurrent` [E x atoms], `edgeWeights` [E] (star1), `modeCoefficients` [M x atoms], `exactModes` [M] (true: exact, false: coexact), `power` [S x atoms], `vertexCurrent` [3S x atoms]
 *  (rows 3v + component), `frequencies` [atoms] (> 0). Throws std::invalid_argument on sizes that disagree or a frequency <= 0. */
FlowAcrossFrequencies flowAcrossFrequencies(const Eigen::MatrixXd& edgeCurrent, const Eigen::VectorXd& edgeWeights, const Eigen::MatrixXd& modeCoefficients,
                                            const std::vector<bool>& exactModes, const Eigen::MatrixXd& power, const Eigen::MatrixXd& vertexCurrent,
                                            const Eigen::VectorXd& frequencies);

}  // namespace nxr::signal

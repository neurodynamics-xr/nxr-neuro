#pragma once
// nxr::phase — the PHASE of a complex field on a surface, POINTWISE (per vertex, per sample): how the timing of an oscillation changes across the cortex.
// The formulas are nxr's phase reader (nxr/detect/core_phase.py, PhaseReader.read_out: grad phi = Im(conj(z) grad z) / |z|^2, no unwrapping; velocity
// u = -omega grad phi / |grad phi|^2), the reference; the mesh gradient is its `gradient_and_kernel` GV (the P1 face gradients, area-averaged onto the
// vertices).
//
// THE CARDINAL RULE (CONTRIBUTING.md): every function here is POINTWISE: nothing is summed or averaged across samples or vertices. Over a span of time or a
// region, a phase gradient is a reduction, and reductions belong to the Selection system: measure the phase current and the power over a partition (time
// and/or space), then take the ratio at read (a derived measure).
//
// A COMPLEX FIELD Z holds `components` complex numbers a vertex (1: a scalar such as the normal current; 3: a vector such as a free-orientation current),
// row v·components + a, one column a sample (a frame, a segment's coefficient: the function does not know). With several components the formulas sum over
// them (Im(Jᴴ ∇J), |J|², |∇J|²): the phase of the vector as a whole, independent of the axes, and blind to how its orientation turns from vertex to vertex.
//
//   vertexGradient(P, F)              the mesh gradient: three [V x V] operators (x, y, z), each the area-weighted mean of the incident faces' P1 gradients
//                                     (nxr's GV). Positions in any length unit; the gradients are per that unit.
//   power(Z, c)                       |z|^2 per vertex and sample                                                 [V x m]
//   phaseCurrent(Z, c, G, n)          j = Im(z̄ ∇z), on the tangent plane (n the unit vertex normals): THE PHASE CURRENT, |z|^2 ∇φ   [3V x m] (x y z a vertex)
//   gradientPower(Z, c, G, n)         |∇z|^2 of the tangent gradient                                              [V x m]
//   phaseGradient(j, p)               ∇φ = j / |z|^2 (rad per length; points AGAINST the travel); NaN where |z|^2 = 0                  [3V x m]
//   phaseVelocity(g, omega)           u = −ω ∇φ / |∇φ|^2 (length per second, along the travel) and its length, the speed; NaN where ∇φ = 0
//   travellingIndex(j, p, q)          |j| / sqrt(|z|^2 |∇z|^2), 0 … 1 (Cauchy–Schwarz): 1 one plane wave, 0 a standing pattern; NaN where p q = 0
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cstdint>
#include <vector>

namespace nxr::signal {

using Gradient = std::array<Eigen::SparseMatrix<double>, 3>;

/** the mesh's vertex gradient (nxr's GV): P [V x 3] positions, F [nF x 3] faces (CCW) -> G[d] [V x V], d = x, y, z */
Gradient vertexGradient(const Eigen::MatrixXd& P, const Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>& F);

/** |z|^2 summed over the components: [V x m] */
Eigen::MatrixXd power(const Eigen::MatrixXcd& Z, int components);
/** the phase current Im(z̄ ∇z) summed over the components, projected on the tangent plane of the unit normals N [V x 3]: [3V x m] */
Eigen::MatrixXd phaseCurrent(const Eigen::MatrixXcd& Z, int components, const Gradient& G, const Eigen::MatrixXd& N);
/** |∇z|^2 of the tangent gradient, summed over the components: [V x m] */
Eigen::MatrixXd gradientPower(const Eigen::MatrixXcd& Z, int components, const Gradient& G, const Eigen::MatrixXd& N);

/** ∇φ = j / p: [3V x m] from j [3V x m] and p [V x m] */
Eigen::MatrixXd phaseGradient(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p);
struct Velocity {
    Eigen::MatrixXd velocity;   // [3V x m]
    Eigen::MatrixXd speed;      // [V x m]
};
/** u = −ω g / |g|^2 and |u| = ω / |g|, ω in rad/s */
Velocity phaseVelocity(const Eigen::MatrixXd& g, double omega);
/** |j| / sqrt(p q): [V x m] */
Eigen::MatrixXd travellingIndex(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p, const Eigen::MatrixXd& q);

}  // namespace nxr::signal

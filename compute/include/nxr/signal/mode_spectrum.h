#pragma once
// nxr::signal — the MODE-SPACE FOURIER COEFFICIENTS of a recording and what is read from them (the flow analysis: docs/reference.md, `spectral_coefficients`).
// PURE MATHS: arrays in, arrays out; nothing here knows a file, a chunk or a subject.
//
//   modeAnalysis        the analysis rows of a surface's modes: a selection of the modes (exact, coexact or all) of a Hodge–Dirac package's 0-form parts P [V x K],
//                       normalised as the caller states (unit_mass: each column rescaled to unit ⋆0 norm, so Φ̂ᵀ⋆0Φ̂ = I on the exact modes; as_stored: P as given),
//                       giving rows Φ̂ᵀ⋆0 [K' x V] (a field on the vertices → its coefficients) and the synthesis Φ̂ [V x K'] (coefficients → the field)
//   modeAnalysisOfComponents   the analysis rows composed with a source operator whose rows are `components` values a vertex (a free-orientation kernel: x, y, z):
//                       [components·K' x C], component-major blocks (rows c·K' .. (c+1)·K' read component c). No orientation is assumed: each Cartesian component is
//                       expanded on the same modes
//   spectralCoefficients  one time tile [C x n] (real) → its one-sided Fourier transform (numpy rfft, unscaled, no window) at the bins [firstBin, firstBin + binCount)
//                       → the spatial operator's rows applied: [R x binCount] complex. The operator is linear and acts on the channels, the transform on time: they commute
//   modeCrossSpectrum   the coefficients of several tiles [R x B] each, with a weight per bin (the weight on a QUADRATIC field: g² of a kernel member of amplitude g)
//                       → T = Σ_tiles Σ_b w_b a_b a_bᴴ [R x R] (the estimate over the tiles given: a statistic of the tile partition)
//   contractComponents  T [c·K x c·K] → Σ_c T_cc [K x K]: the cross-spectrum of a vector field's coefficients, the dot product of its components (rotation invariant)
//   flowMap             the synthesis of a cross-spectrum S [K x K] with amplitude weights h [K] (a scale member applied to the LINEAR coefficients): per vertex the power
//                       diag(Φ̂ h Re S h Φ̂ᵀ) (a 0-form), per edge of the intrinsic complex the phase current Σ Im(conj(z_tail) z_tip) = −(Φ̂ h Im S h Φ̂ᵀ)_{tail,tip}
//                       (a 1-form; the same quantity as `phaseCurrent(c, z)` summed over the cross-spectrum's draws). With the forward transform (e^{−iωt}) a
//                       pattern travelling tail → tip gives a NEGATIVE phase current: the direction of travel is −J
//   cellMeasures        the additive ingredients of a cross-spectrum S [K x K] over scale members given as weights on a QUADRATIC field H [M x K]: per member its power
//                       Σ_k H_mk Re S_kk and effective number of modes (Σ_k H_mk)² / Σ_k H_mk²; per pair of members (a, b) the in-phase energy Σ_{j≠k} H_aj H_bk (Re S_jk)²,
//                       the quadrature energy Σ_{j≠k} H_aj H_bk (Im S_jk)² and the quadrature's noise ingredient Σ_{j≠k} H_aj H_bk (S_jj S_kk − (Re S_jk)²) (the
//                       expected quadrature energy with no quadrature is that ÷ (2 · degrees of freedom)). Ratios are derived at read
//   effectiveDegreesOfFreedom  tiles · (Σ w)² / (wᵀ R w), w the weights on a quadratic field over consecutive bins, R_ij = ρ(|i − j|) the measured correlation of
//                       neighbouring bins (ρ(0) = 1; beyond ρ's length 0). Empty ρ: the bins taken as independent, tiles · (Σ w)² / Σ w²
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "nxr/signal/intrinsic_forms.h"

namespace nxr::signal {

enum class ModeKinds { exact, coexact, all, exact_and_constant };   // exact_and_constant: the constant (index −1, unit ⋆0 norm) then the exact modes: a surface field's coordinates
enum class ModeNormalisation { unit_mass, as_stored };

struct ModeAnalysis {
  Eigen::MatrixXd rows;                 // [K' x V] Φ̂ᵀ ⋆0
  Eigen::MatrixXd synthesis;            // [V x K'] Φ̂
  std::vector<std::int64_t> modes;      // [K'] the package's mode indices kept, in order
};
/** `zeroFormParts` [V x K], `vertexMass` [V] (⋆0), `kind` [K] (0 exact, 2 coexact). Throws std::invalid_argument on sizes that do not agree, a mode of zero ⋆0 norm under unit_mass,
 *  or a selection that keeps no mode. */
ModeAnalysis modeAnalysis(const Eigen::MatrixXd& zeroFormParts, const Eigen::VectorXd& vertexMass, const std::vector<int>& kind, ModeKinds kinds, ModeNormalisation normalisation);
/** `analysisRows` [K' x V], `sourceOperator` [components·V x C] with row components·v + c the c-th value of vertex v. Throws std::invalid_argument on sizes that do not agree. */
Eigen::MatrixXd modeAnalysisOfComponents(const Eigen::MatrixXd& analysisRows, const Eigen::MatrixXd& sourceOperator, int components);
/** `spatialOperator` [R x C], `tile` [C x n]. Throws std::invalid_argument when the bins fall outside [0, n/2] or the sizes do not agree. */
Eigen::MatrixXcd spectralCoefficients(const Eigen::MatrixXd& spatialOperator, const Eigen::MatrixXd& tile, std::int64_t firstBin, std::int64_t binCount);
/** `tiles` [R x B] each (complex; float or double values given as double), `quadraticWeights` [B]. Throws std::invalid_argument on no tile or sizes that do not agree. */
Eigen::MatrixXcd modeCrossSpectrum(const std::vector<Eigen::MatrixXcd>& tiles, const Eigen::VectorXd& quadraticWeights);
Eigen::MatrixXcd contractComponents(const Eigen::MatrixXcd& crossSpectrum, int components);

struct FlowMap {
  Eigen::VectorXd power;                // [V] the 0-form: power per vertex
  Eigen::VectorXd phaseCurrent;         // [nE] the 1-form: phase current per intrinsic edge (tail → tip)
};
/** `complex` the intrinsic complex of the surface the modes are on (its edges), `synthesis` Φ̂ [V x K], `crossSpectrum` S [K x K] Hermitian, `amplitudeWeights` h [K].
 *  Made in blocks of vertices (no [V x V] matrix; at most `blockVertices` x K doubles at a time). Throws std::invalid_argument on sizes that do not agree. */
FlowMap flowMap(const IntrinsicComplex& complex, const Eigen::MatrixXd& synthesis, const Eigen::MatrixXcd& crossSpectrum, const Eigen::VectorXd& amplitudeWeights,
                std::int64_t blockVertices = 4096);

struct CellMeasures {
  Eigen::VectorXd power;                // [M]
  Eigen::VectorXd effectiveModes;       // [M]
  Eigen::MatrixXd inPhaseEnergy;        // [M x M]
  Eigen::MatrixXd quadratureEnergy;     // [M x M]
  Eigen::MatrixXd quadratureNoise;      // [M x M] ÷ (2 dof) = the quadrature energy expected with no quadrature
};
CellMeasures cellMeasures(const Eigen::MatrixXcd& crossSpectrum, const Eigen::MatrixXd& quadraticWeights);
double effectiveDegreesOfFreedom(const Eigen::VectorXd& quadraticWeights, std::int64_t tiles, const Eigen::VectorXd& binCorrelation);

}  // namespace nxr::signal

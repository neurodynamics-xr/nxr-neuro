#pragma once
// nxr::signal — THE MODE ANALYSIS of a surface: the coordinates a field is served in.
// PURE MATHS: arrays in, arrays out; nothing here knows a file, a chunk or a subject.
//
//   modeAnalysis        the analysis rows of a surface's modes: a selection of the modes (exact, coexact or all) of a Hodge–Dirac package's 0-form parts P [V x K],
//                       normalised as the caller states (unit_mass: each column rescaled to unit ⋆0 norm, so Φ̂ᵀ⋆0Φ̂ = I on the exact modes; as_stored: P as given),
//                       giving rows Φ̂ᵀ⋆0 [K' x V] (a field on the vertices → its coefficients) and the synthesis Φ̂ [V x K'] (coefficients → the field)
//   modeAnalysisOfComponents   the analysis rows composed with a source operator whose rows are `components` values a vertex (a free-orientation kernel: x, y, z):
//                       [components·K' x C], component-major blocks (rows c·K' .. (c+1)·K' read component c). No orientation is assumed: each Cartesian component is
//                       expanded on the same modes
//   (the flow of a cross-spectrum in these coordinates: flow_readout.h)
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

}  // namespace nxr::signal

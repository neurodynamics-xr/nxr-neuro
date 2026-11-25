#pragma once
// nxr::signal — THE FLOW OF A WEIGHTED CROSS-SPECTRUM READ THROUGH A LINEAR READOUT ONTO A SURFACE (spec docs/specs/2026-11-24-source-flow.md): the source side of the flow, where an
// existing minimum-norm kernel, written in the cortex's eigenmode coordinates, carries the sensor-side cross-spectrum onto the cortex. PURE MATHS: arrays in, arrays out; nothing here
// knows a file, a chunk, a subject or where a kernel came from.
//
//   selectVertexRows            rows of a per-vertex operator (`components` rows a vertex, vertex-major) selected onto another vertex set: a vertex whose source is -1 gets zero rows
//                               (a glued cortex's bridge: no current is estimated there). A hemisphere is the same selection with a vertex range. Never a blend.
//   synthesisFromAnalysis       a surface's coordinates at its vertices from their analysis rows: analysis = synthesisᵀ diag(mass) with a mass-orthonormal synthesis, so
//                               synthesis = diag(mass)⁻¹ analysisᵀ (the sensor surface's field coordinates back to its vertices, which are channels)
//   bandReadouts                the readout of each band of the cortical eigenvalue line, component-major rows [components V x columns]: band q < last is
//                               (I ⊗ modeSynthesis diag(g_q)) modeOperator; the LAST band is the complement, the kernel minus every other band, so the bands add back to the
//                               kernel exactly and the content outside the kept modes lies in the last band (the reference's convention)
//   (the flow read through these readouts, its time cells and its split: flow_readout.h, pairFlow, timeCellPairFlow, flowSplit)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/spectral_flow.h"

namespace nxr::signal {

/** `perVertexRows` [components x sourceVertices, columns] (rows components v + c), `sourceVertex` [targetVertices] (the source vertex of each target vertex, -1: zero rows).
 *  Returns [components x targetVertices, columns]. Throws std::invalid_argument on a source vertex outside the rows. */
Eigen::MatrixXd selectVertexRows(const Eigen::MatrixXd& perVertexRows, const std::vector<std::int64_t>& sourceVertex, int components);

/** `analysisRows` [coordinates x vertices], `vertexMass` [vertices] > 0. Returns [vertices x coordinates]. */
Eigen::MatrixXd synthesisFromAnalysis(const Eigen::MatrixXd& analysisRows, const Eigen::VectorXd& vertexMass);

/** `modeOperator` [components K x columns] (component-major: rows c K + k), `modeSynthesis` [V x K], `bandGains` [bands x K] (at least one band),
 *  `kernelInColumns` [components V x columns] (vertex-major: rows components v + c, the kernel's own order). Returns the bands' readouts, each [components V x columns],
 *  component-major (rows c V + v). Throws std::invalid_argument on sizes that disagree. */
std::vector<Eigen::MatrixXd> bandReadouts(const Eigen::MatrixXd& modeOperator, const Eigen::MatrixXd& modeSynthesis, const Eigen::MatrixXd& bandGains,
                                          const Eigen::MatrixXd& kernelInColumns, int components);
/** A kernel's rows (vertex-major: components v + c) reordered component-major (c V + v): the whole field's readout. */
Eigen::MatrixXd componentMajor(const Eigen::MatrixXd& kernelInColumns, int components);


}  // namespace nxr::signal

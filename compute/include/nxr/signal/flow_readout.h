#pragma once
// nxr::signal — THE FLOW READOUT: the one place where the second-order flow of a weighted cross-spectrum is read through a readout, and split into travel and
// rotation, for every flow (the sensor surface's field coordinates, the source mapping's composed kernel, the connectome's tiles). PURE MATHS: arrays in, arrays out;
// nothing here knows a file, a chunk, a subject or where a readout came from.
//
//   flowSplit               an edge current [E x m] split on the STORED Hodge-Dirac edge rows: its coefficients (projectOneForm: inner products, no solve), the travel
//                           potential Σ_exact (c / mu) mode and the rotation stream Σ_coexact (c / mu) mode (the 1/mu gain: the full Poisson split when the basis is
//                           complete), its ⋆1 energy; and, when a remainder split is given, the part beyond the kept modes split by sparse solves (a cortex)
//   OneFormRemainderSplit   the part of a 1-form beyond the kept modes (the remainder, w minus its synthesis from its mode coefficients) split by sparse Poisson solves:
//                           its potential (d0ᵀ ⋆1 d0 alpha = d0ᵀ ⋆1 r: Neumann on an open surface) and its stream (the 2-form potential beta with
//                           d1 ⋆1⁻¹ d1ᵀ ⋆2 beta = d1 r: Dirichlet on an open surface), drawn at the vertices as `coexactStream` draws a mode's
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "nxr/signal/intrinsic_forms.h"

namespace nxr::signal {

/** The remainder's split, the two sparse systems factored once. `split` takes a 1-form remainder [E] and returns its potential and stream at the vertices, each with its
 *  mass-weighted mean removed. */
class OneFormRemainderSplit {
public:
  explicit OneFormRemainderSplit(const IntrinsicComplex& complex);
  struct Parts { Eigen::VectorXd potential, stream; };
  Parts split(const Eigen::VectorXd& remainder) const;
private:
  const IntrinsicComplex* complex_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> vertexLaplacian_, faceLaplacian_;
};

struct FlowSplit {
  Eigen::MatrixXd coefficients;                          // [M x m] on the stored edge rows (exact: travel; coexact: rotation)
  Eigen::MatrixXd potential, stream;                     // [V x m]
  Eigen::MatrixXd remainderPotential, remainderStream;   // [V x m], empty unless a remainder split is given
  Eigen::VectorXd flowEnergy, remainderEnergy;           // [m] ⋆1 energies; remainderEnergy empty unless a remainder split is given
};
/** `edgeCurrent` [E x m]; `remainderSplit` null when the modes are complete (a sensor surface) or the remainder is not wanted. */
FlowSplit flowSplit(const IntrinsicComplex& complex, const Modes& modes, const Eigen::MatrixXd& edgeCurrent, const OneFormRemainderSplit* remainderSplit);

}  // namespace nxr::signal

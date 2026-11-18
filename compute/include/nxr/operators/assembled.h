#pragma once
// Operators assembled ON DEMAND from a geometry::Manifold (the Manifold holds no operator: the facet accessors `m.operators().…` are gone).
// Each function assembles and returns BY VALUE; whoever reuses a matrix (the pipeline executor, a binding, a test) holds it. The factorisation is
// the caller's too: pass a CholeskyCache where one is wanted.
#include <complex>
#include <map>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/laplacians.h"

namespace nxr::operators {

/** The vertex connection Laplacian (complex, n = 1) in the Manifold's ACTIVE connection (Levi-Civita, or trivial with its singularities). */
Eigen::SparseMatrix<std::complex<double>> activeConnectionLaplacian(geometry::Manifold& m);
/** The covariant Laplacian [3N x 3N] in the active connection; `Ambient` (the default) is built from the frames and the cotan weights alone, `Product` also from the connection Laplacian. */
Eigen::SparseMatrix<double> activeCovariantLaplacian(geometry::Manifold& m,
    laplacian::connection::CovariantCoupling coupling = laplacian::connection::CovariantCoupling::Ambient);

/** The relative-Dirac family L(τ) = (1−τ)(cotanL ⊗ I₄) + τ·E, [4V x 4V] real symmetric, τ in [0,1]. τ = 0 never assembles E; τ = 1 never builds the cotan block. */
Eigen::SparseMatrix<double> diracFamily(geometry::Manifold& m, double tau);
/** The FACE-domain family L̃(τ) = (1−τ)(K̃ ⊗ I₄) + τ·Ẽ, [4F x 4F], K̃ = d₁⋆₁⁻¹d₁ᵀ of the INPUT's DEC. */
Eigen::SparseMatrix<double> diracFaceFamily(geometry::Manifold& m, double tau);

/** The realised vertex frame [nV,3] (c = e1 + i e2) in a connection: Levi-Civita / Euclidean = geometry-central's vertex grid; Trivial = the combed frame. */
Eigen::MatrixXcd connectionGrid(geometry::Manifold& m, geometry::ConnectionType type, const std::map<int,double>& singularities = {});
/** The realised face frame [nF,3] in a connection (same combing convention, over the dual graph). */
Eigen::MatrixXcd connectionFaceGrid(geometry::Manifold& m, geometry::ConnectionType type, const std::map<int,double>& singularities = {});

} // namespace nxr::operators

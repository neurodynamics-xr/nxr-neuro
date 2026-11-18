#pragma once
// Operators assembled from the surface's arrays (nxr/geometry/surface_arrays.h). Each function assembles and returns BY VALUE; whoever reuses a
// matrix (the pipeline executor, a binding, a test) holds it. The connection is an argument: nothing is held anywhere.
#include <complex>
#include <map>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/laplacians.h"

namespace nxr::operators {

/** The vertex connection Laplacian (complex, n = 1) in a connection: Levi-Civita (no singularities), or trivial with its singularities (they sum to χ). */
Eigen::SparseMatrix<std::complex<double>> vertexConnectionLaplacian(const geometry::SurfaceArrays& surface,
    geometry::ConnectionType type = geometry::ConnectionType::LeviCivita, const std::map<int,double>& singularities = {});
/** The covariant Laplacian [3N x 3N] in a connection; `Ambient` (the default) is built from the frames and the cotan weights alone, `Product` also from the connection Laplacian. */
Eigen::SparseMatrix<double> covariantLaplacian(const geometry::SurfaceArrays& surface,
    geometry::ConnectionType type = geometry::ConnectionType::LeviCivita, const std::map<int,double>& singularities = {},
    laplacian::connection::CovariantCoupling coupling = laplacian::connection::CovariantCoupling::Ambient);

/** The relative-Dirac family L(τ) = (1−τ)(cotanL ⊗ I₄) + τ·E, [4V x 4V] real symmetric, τ in [0,1]. τ = 0 never assembles E; τ = 1 never builds the cotan block. */
Eigen::SparseMatrix<double> diracFamily(const geometry::SurfaceArrays& surface, double tau);
/** The FACE-domain family L̃(τ) = (1−τ)(K̃ ⊗ I₄) + τ·Ẽ, [4F x 4F], K̃ = d₁⋆₁⁻¹d₁ᵀ of the INPUT's DEC. */
Eigen::SparseMatrix<double> diracFaceFamily(const geometry::SurfaceArrays& surface, double tau);

/** The realised vertex frame [nV,3] (c = e1 + i e2) in a connection: Levi-Civita / Euclidean = geometry-central's vertex grid; Trivial = the combed frame. */
Eigen::MatrixXcd connectionGrid(const geometry::SurfaceArrays& surface, geometry::ConnectionType type, const std::map<int,double>& singularities = {});
/** The realised face frame [nF,3] in a connection (same combing convention, over the dual graph). */
Eigen::MatrixXcd connectionFaceGrid(const geometry::SurfaceArrays& surface, geometry::ConnectionType type, const std::map<int,double>& singularities = {});

} // namespace nxr::operators

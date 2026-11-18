#pragma once
// Dirac, Hodge-Dirac and spin Dirac operators (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"

namespace nxr::operators {
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;
} // namespace nxr::operators

namespace nxr::operators::dirac {

// Extrinsic block of the relative Dirac family (Liu/Jacobson/Crane, SGP 2017):
// E = Dᵀ ⋆_F D, the [4V×4V] real-symmetric PSD Galerkin form of the relative
// Dirac operator D_N (the shape-operator / Gauss-map energy). Quaternion order
// [w,x,y,z]; index 4*v+c. Geometry-only (vertex normals + face areas) — the one
// operator assembled directly rather than wrapped from geometry-central.
// First-order (rectangular) extrinsic Dirac operator D : [4F×4V] — maps a
// quaternionic field on vertices to one on faces. This is the operator whose
// area-weighted Galerkin square is E = Dᵀ ⋆_F D (extrinsicBlock builds E from
// THIS matrix, so the two never drift). Quaternion order [w,x,y,z]; row 4*f+c,
// column 4*v+c. Geometry-only (vertex normals + face areas).
Eigen::SparseMatrix<double> matrix(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> extrinsicBlock(const geometry::SurfaceArrays& surface);
// First-order (rectangular) INTRINSIC Dirac operator D_int : [4F×4V] — identical
// block structure to matrix(), but built from the IMMERSION (vertex positions /
// opposite edge vectors) instead of the Gauss map. This is the spin-connection /
// tangent-frame-transport root; its area-weighted Galerkin square Dᵀ ⋆_F D equals
// the intrinsic Dirac² (the cotan-Laplacian-scalar-part operator). Geometry-only
// (vertex positions + face areas). Quaternion order [w,x,y,z].
Eigen::SparseMatrix<double> matrixIntrinsic(const geometry::SurfaceArrays& surface);
// Face-domain (Poincaré dual) first-order Dirac operator D̃ : [4V×4F] — maps a
// quaternionic field on faces to one on vertices, from EXACT per-face normals
// aggregated over vertex stars (the dual of matrix). Its Galerkin square is the
// face-domain extrinsic block Ẽ = D̃ᵀ ⋆_V D̃, the [4F×4F] real-symmetric PSD
// operator (extrinsicBlockFace builds Ẽ from THIS matrix). Quaternion order
// [w,x,y,z]; row 4*v+c, column 4*f+c. Geometry-only (face normals + vertex dual
// areas). Closed-mesh v1: both throw Error(InvalidInput) on an open boundary
// (vertex stars must be closed) — exact on closed cortical hemispheres.
Eigen::SparseMatrix<double> matrixFace(const geometry::SurfaceArrays& surface);

// matrixFaceIntrinsic: INTRINSIC face-domain Dirac D̃_int [4V×4F] — the centroid
// (immersion) dual of matrixFace (face centroids instead of the Gauss map), mirroring
// how matrixIntrinsic relates to matrix. Geometry-only; closed-mesh v1.
Eigen::SparseMatrix<double> matrixFaceIntrinsic(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> extrinsicBlockFace(const geometry::SurfaceArrays& surface);
} // namespace nxr::operators::dirac

namespace nxr::operators::hodgedirac {
// ── Hodge–Dirac d + δ on 0-, 1- and 2-forms of the INTRINSIC Delaunay triangulation ──
// Port of nxr-cortical-flow hodgeDirac.ts (the reference where implementations differ). The pencil (K, M),
// N = V + E + F, rows ordered 0-forms [0,V) · 1-forms [V,V+E) · 2-forms [V+E,N):
//     K = [0 d0ᵀ⋆1 0; ⋆1 d0 0 d1ᵀ⋆2; 0 ⋆2 d1 0],   M = diag(⋆0, ⋆1, ⋆2).
// ⋆0 mixed-Voronoi mass of the mollified input (obtuse corner: A/2 and A/4), ⋆1 = ½(cot α + cot β), ⋆2 = 1/area. M⁻¹K = d + δ is
// M-self-adjoint with spectrum ±μ; D² is block-diagonal in the Hodge Laplacians. The triangulation is built from the
// input's edge lengths by THE one construction (geometry-central's signpost triangulation, nxr/geometry/intrinsic_surface.h): mollify (Sharp & Crane 2020, relative factor `mollify`),
// then flipToDelaunay (tolerance `flipEps`). Both are named, recorded options (`record`).
// Vertex indices are the input's; edges and faces are the intrinsic mesh's (edge orientation = its halfedge).
// ⋆1 is the raw intrinsic cotan weight (nxr, the reference;). A named option `star1Floor` > 0 raises it to that floor so that M > 0 on
// near-cocircular edges (Rheome PR #94 floored at 1e-6; being changed to match); 0 = off, the default.
struct HodgeDirac {
    Eigen::SparseMatrix<double> K;      // [N×N] symmetric
    Eigen::SparseMatrix<double> d0, d1; // [E×V], [F×E]
    Eigen::VectorXd star0, star1, star2;// diagonal Hodge stars (star1 after the floor)
    Eigen::VectorXd M;                  // [star0; star1; star2]
    int nV = 0, nE = 0, nF = 0;
    int star1Clamped = 0;               // edges whose cotan weight was raised to the floor
    double mollifyEps = 0.0;            // length added to every edge by mollify (0 = none needed)
    Eigen::MatrixXi faces;              // [nF, 3] the intrinsic triangulation's faces (vertex indices are the input's)
    geometry::IntrinsicRecord record;             // mollify, ε, flipEps, flips, gauge of the construction
    // the intrinsic complex in the stored layout (geometry::IntrinsicSurface::Complex): corners 3f + k, edges tail → tip, signed face edges (1-based), lengths,
    // the raw signpost angles of each corner's halfedge, the cone angles and the reference neighbours (angle 0 of the gauge)
    geometry::IntrinsicSurface::Complex complex;
};
// Uses the geometry::Manifold's operator mesh when it is the intrinsic one built with these options, else builds its own by the same construction.
HodgeDirac assemble(const geometry::SurfaceArrays& surface, double star1Floor = 0.0, double mollify = 1e-6, double flipEps = 1e-6);
Eigen::SparseMatrix<double> massMatrix(const HodgeDirac& h);   // diag(M)
} // namespace nxr::operators::hodgedirac

namespace nxr::operators::spin {
// ── Spin Diracs on quaternion spinors per FACE (Hoffmann & Ye 2018; Ye et al. 2018) ──
// Port of cortical-flow spinDirac.ts / Rheome operators.spin_dirac. Closed, oriented meshes. Hyperedge of face i across
// the halfedge e to face j: E_ij = |e| tan(θ/2) + e. 4F×4F symmetric, interleaved [w,x,y,z] at 4f+c:
//   Extrinsic   D_f = ½ Σ_j E_ij − H_i      (spin transformations; kills the constants)
//   IntrinsicHY D   = ½ Σ_j E_ij            (= D_f + H)
//   IntrinsicYe D   = ½ Σ_j cos(θ/2) E_ij   (edge lengths only)
// The eigenproblem is D φ = μ (area ⊗ I₄) φ, solved on VERTEX spinors through vertexToFace (see operators::eigenSpin).
enum class Kind { Extrinsic, IntrinsicHY, IntrinsicYe };
struct SpinDirac {
    Eigen::SparseMatrix<double> D;   // [4F×4F]
    Eigen::VectorXd H;               // [F] integrated mean curvature per face
    Eigen::VectorXd area;            // [F]
};
SpinDirac assemble(const geometry::SurfaceArrays& surface, Kind kind);
// A (4F×4V): (Aψ)_f = Σ_{v∈f} p_fv ψ_v. covariant: p_fv = unit quaternion turning the area-weighted vertex normal onto the face's
// (Ye et al. adj_vf_cov; for the intrinsic operators); else 1 (adj_vf; for the extrinsic).
Eigen::SparseMatrix<double> vertexToFace(const geometry::SurfaceArrays& surface, bool covariant);
} // namespace nxr::operators::spin


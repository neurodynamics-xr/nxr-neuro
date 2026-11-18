#pragma once
// DEC operators, masses, the face gradient and the gradient of a vertex scalar (toolbox refactor P1: moved from compute.h, declarations unchanged).
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

namespace nxr::operators {

// ── Operator Assembly ────────────────────────────────────────

/**
 * Mass-matrix variant for the FEM L² inner product ⟨u, v⟩ = uᵀ M v.
 *
 * - MixedVoronoi (DEFAULT): diagonal; each vertex's mixed Voronoi area (Meyer, Desbrun, Schröder & Barr 2003) of the
 *   mollified INPUT triangulation, recorded before the first flip (`geometry::IntrinsicSurface::inputMixedVoronoiMass`) — the mass of the Studio, nxr and Rheome. With
 *   geometry::MeshKind::Embedded: of the input as it is.
 * - Galerkin: full sparse, the L² integrals of piecewise-linear hat functions on the triangulation as it is (per-triangle element
 *   matrix (A_T / 12) · [[2 1 1][1 2 1][1 1 2]]). A named option.
 * - Barycentric: diagonal, M_ii = (Σ adjacent triangle areas) / 3 on the triangulation as it is (geometry-central's "lumped"; renamed to what
 *   it is — "lumped" means the mixed Voronoi mass in nxr and the Studio). A named option.
 *
 * All conserve total surface area (Σᵢⱼ Mᵢⱼ == totalArea) and are symmetric.
 */
enum class MassMatrixVariant {
    MixedVoronoi,   // diagonal; the recorded mixed Voronoi mass of the mollified input (default)
    Galerkin,       // sparse FEM, on the operator mesh as it is
    Barycentric     // diagonal; A/3 per vertex on the operator mesh as it is
};

// ManifoldOperators / DECOperators mix view and value semantics. Field
// names follow geometry-central's canonical naming for the underlying
// cached matrices (cotanLaplacian, vertexDualAreas, vertexNormals,
// d0/d1/hodge*), so the surface here is a 1-to-1 viewer over GC's
// cache where the concept exists. Variant-aware quantities (mass)
// keep a generic name because GC has no single matrix covering them.
//
//   * View fields (const-references into geometry-central's cached matrices):
//       ManifoldOperators::cotanLaplacian, all DECOperators fields. Bound by
//       assembleManifoldOperators / assembleDECOperators after the relevant
//       require* call pins them on the geometry.
//
//   * Owned field:
//       ManifoldOperators::mass — variant-dependent (MixedVoronoi / Galerkin / Barycentric);
//       held by value because GC exposes two distinct matrices and we
//       want a single uniform field shape regardless of which variant
//       the caller requested.
//
// Lifetime contract: bindings must keep the owning geometry::Manifold alive
// for the entire lifetime of any ManifoldOperators / DECOperators they hold,
// and must NOT call unrequire* on the underlying geometry while the
// structs are alive. ContextHolder / ContextWrapper enforce this by
// keeping the operator structs and the context together.
//
// Because of the reference fields, both structs are copy-constructible
// (refs rebind to the same targets) but not copy-assignable. The
// bindings hold them via shared_ptr / unique_ptr and never reassign.

// Row-major V×3 alias. `vertexNormals` is exclusively consumed row-wise
// (each row is one (x,y,z) triple flattened directly into the §11 V*3
// row-major output buffer at the binding edge). Storing it row-major
// here makes the binding-side memcpy direct — no transpose needed,
// because the in-memory layout already matches the output convention.
// Eigen kernels read this as `m.row(v)` only; no column ops in any of
// the existing call sites.
using VertexNormalsMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct ManifoldOperators {
    ManifoldOperators(Eigen::SparseMatrix<double> cotanLaplacian_,
                      Eigen::SparseMatrix<double> mass_,
                      MassMatrixVariant massVariant_,
                      Eigen::VectorXd vertexDualAreas_,
                      VertexNormalsMatrix vertexNormals_,
                      double totalArea_)
      : cotanLaplacian(std::move(cotanLaplacian_)),
        mass(std::move(mass_)),
        massVariant(massVariant_),
        vertexDualAreas(std::move(vertexDualAreas_)),
        vertexNormals(std::move(vertexNormals_)),
        totalArea(totalArea_) {}

    Eigen::SparseMatrix<double> cotanLaplacian;         // the operator mesh's cotangent Laplacian
    Eigen::SparseMatrix<double> mass;                   // owned: depends on massVariant
    MassMatrixVariant massVariant;                      // which variant produced `mass`
    Eigen::VectorXd vertexDualAreas;                    // [nV] A/3-per-vertex on the operator mesh (variant-independent)
    VertexNormalsMatrix vertexNormals;                  // [nV, 3] row-major; see alias comment above
    double totalArea;
    // nV / nE / nF intentionally omitted — read from the owning
    // ComputeContext (ctx.nV(), ctx.nE(), ctx.nF()) to avoid duplicating
    // mesh metadata that's already canonical on the context.
};

/** Assemble operators from a context, on the geometry::Manifold's operator mesh (nxr/geometry/intrinsic_surface.h). Default mass variant is MixedVoronoi. */
ManifoldOperators assembleManifoldOperators(
    const geometry::SurfaceArrays& surface,
    MassMatrixVariant variant = MassMatrixVariant::MixedVoronoi);

/** The mass matrix of a variant on the geometry::Manifold's operator mesh (diagonal for MixedVoronoi and Barycentric). */
Eigen::SparseMatrix<double> vertexMass(const geometry::SurfaceArrays& surface, MassMatrixVariant variant = MassMatrixVariant::MixedVoronoi);

/** String → enum helper for binding shells (WASM, addon, MEX).
 *  Accepts "mixedVoronoi", "galerkin" and "barycentric" ("lumped" is retired: no alias). Throws
 *  Error(InvalidInput) on unknown. Case-sensitive. */
MassMatrixVariant parseMassMatrixVariant(const std::string& name);

// ── DEC Operators ────────────────────────────────────────────

struct DECOperators {
    DECOperators(const Eigen::SparseMatrix<double>& d0_,
                 const Eigen::SparseMatrix<double>& d1_,
                 const Eigen::SparseMatrix<double>& hodge0_,
                 const Eigen::SparseMatrix<double>& hodge1_,
                 const Eigen::SparseMatrix<double>& hodge2_,
                 const Eigen::SparseMatrix<double>& hodge1Inverse_)
      : d0(d0_), d1(d1_), hodge0(hodge0_), hodge1(hodge1_),
        hodge2(hodge2_), hodge1Inverse(hodge1Inverse_) {}

    // The matrices are OWNED (copies): nothing in compute caches, the caller holds the bundle (the pipeline executor binds it once). dec.hodge0 is the
    // same matrix as ops.mass when ops.massVariant == MixedVoronoi (the default).
    Eigen::SparseMatrix<double> d0;      // vertex → edge
    Eigen::SparseMatrix<double> d1;      // edge → face
    Eigen::SparseMatrix<double> hodge0;  // vertex → vertex (diagonal)
    Eigen::SparseMatrix<double> hodge1;  // edge → edge (diagonal)
    Eigen::SparseMatrix<double> hodge2;  // face → face (diagonal)
    Eigen::SparseMatrix<double> hodge1Inverse;
};

// The matrices of the operator mesh's DEC.
struct DecMatrices {
    Eigen::SparseMatrix<double> d0, d1, hodge0, hodge1, hodge2, hodge1Inverse;
};
/** The cotangent Laplacian of the operator mesh (positive semidefinite), by value. */
Eigen::SparseMatrix<double> cotanLaplacian(const geometry::SurfaceArrays& surface);
/** The DEC matrices of the operator mesh, by value. */
DecMatrices decMatrices(const geometry::SurfaceArrays& surface);
/** The DEC of the INPUT mesh (geometry-central's embedded one): for what is indexed by the input's edges and faces (a 1-form on the input's edges, Whitney
 *  interpolation on its faces, the trivial connection, the face Dirac). */
DECOperators embeddedDEC(const geometry::SurfaceArrays& surface);

// The stand-alone DEC FOLLOWS THE MESH OPTION: on the geometry::Manifold's operator mesh (the intrinsic mollified Delaunay mesh by default,
// the input with geometry::MeshKind::Embedded), so that d0ᵀ ⋆1 d0 == the cotan Laplacian (`cotanLaplacian(m)`) and ⋆0 == the mass of the
// default mass (the mixed-Voronoi mass of the mollified input; nxr::operators::massMatrix variants Barycentric and Galerkin are other masses). ⋆1 is the RAW cotan weight, NOT floored
// (a cocircular edge gives ⋆1 = 0 and hodge1Inverse is then singular; the floored ⋆1 of the Hodge–Dirac pencil is `star1Floor` in
// nxr::operators::hodgedirac::assemble). Edges and faces are the operator mesh's (the flipped faces under the intrinsic option; edge e keeps its index through
// the flips), vertices are the input's. What is indexed by the INPUT's edges and faces uses `embeddedDEC(m)`.
DECOperators assembleDECOperators(const geometry::SurfaceArrays& surface);

// The operator mesh's DEC matrices one at a time, by value (each call assembles; hold the result, or the bundle from assembleDECOperators).
Eigen::SparseMatrix<double> d0(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> d1(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> hodge0(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> hodge1(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> hodge2(const geometry::SurfaceArrays& surface);
Eigen::SparseMatrix<double> hodge1Inverse(const geometry::SurfaceArrays& surface);
// A/3 per vertex on the operator mesh (variant-independent), by value.
Eigen::VectorXd             vertexDualAreas(const geometry::SurfaceArrays& surface);
} // namespace nxr::operators

namespace nxr::operators::facegrad {
// Face-native (Poincaré-dual) scalar calculus, the dual of the vertex FEM
// gradient + cotan Laplacian. DOFs live on FACES (dual vertices).
// gradient: barycentric dual-mesh gradient of a per-face scalar — a per-face
// ambient 3-vector field [3F×F]. Green–Gauss / DEC construction:
//   grad ψ|_f = (1/2A_f) Σ_k (ψ_{g_k} − ψ_f) (l_k × n_f)
// over face f's three edge-neighbors g_k (l_k = boundary edge vector, l_k × n_f
// = the outward in-plane edge normal). Σ_k l_k = 0 ⇒ constants annihilated and
// the self term vanishes; output is tangent to each face. Closed-mesh v1 (every
// face has exactly 3 edge-neighbors): throws Error(InvalidInput) on a boundary.
Eigen::SparseMatrix<double> gradient(const geometry::SurfaceArrays& surface);
// laplacian: the face Laplacian K̃ = gradientᵀ ⋆_F gradient [F×F] (⋆_F = face
// areas), built from THE SAME gradient so the two never drift. Symmetric PSD;
// kernel = constants (genus-0). The dual of the cotan Laplacian.
Eigen::SparseMatrix<double> laplacian(const geometry::SurfaceArrays& surface);
} // namespace nxr::operators::facegrad

namespace nxr::operators {
/** Gradient of a vertex scalar field → face-centered 3D vectors.
 *  For triangle (vi, vj, vk): ∇u = (1 / 2A) Σ (uₖ - uᵢ) (N × eⱼᵢ)
 *  Returns [nF, 3] matrix. */
Eigen::MatrixXd gradient(
    const geometry::SurfaceArrays& surface,
    const Eigen::VectorXd& scalarField
);
} // namespace nxr::operators


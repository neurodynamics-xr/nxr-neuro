#pragma once

// nxr-compute — portable C++ math/compute library for cortical-flow,
// the NXR design system, and any other consumer that needs
// halfedge-mesh-based scientific computing. Targets one source,
// many shells (N-API addon, WASM/Embind, MATLAB MEX, CLI).
//
// Layout convention for matrix returns (hard rule, applies to
// every JS-facing binding shell — addon, WASM):
//
//   • [V × 3] / [F × 3] / [N × 3]  →  row-major flat: xyz xyz xyz …
//                                     matches three.js BufferAttribute
//                                     and NumPy `.reshape(-1, 3)`.
//   • [V × K] eigenvectors         →  row-major flat (vMajor): U[v*K+k]
//                                     matches the Zarr schema
//                                     (manifold/eigenmodes/eigenvectors
//                                     stored as [V, K]) and the GPU
//                                     spectral-synthesis access pattern
//                                     ("for each vertex, sum across modes").
//                                     C++ side keeps Eigen default
//                                     column-major storage; bindings
//                                     transpose at the flatten step.
//   • [T × V] activity time series →  row-major flat (frame-major):
//                                     frame t at data[t*V .. (t+1)*V].
//                                     matches the Zarr `recordings/.../
//                                     activity` schema.
//
// MEX is exempt from the JS flatten rule: a V×K Eigen matrix
// becomes a V×K MATLAB matrix (column-major in MATLAB's native
// storage) so MATLAB users get U(:,k) for mode k contiguously.
// See native/CONTRIBUTING.md §11 for the binding contract.

#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "nxr/delaunay.h"
#include "nxr/intrinsic_surface.h"

#include "nxr/errors.h"
#include "nxr/cancellation.h"
#include "nxr/progress.h"

// C++17 floor today. C++20 was the planned baseline (see Phase A
// design doc) but a vendored-Eigen 3.4 + Emscripten Clang + C++20
// template-deduction bug in geometry-central's BFF source forces
// us to stay at 17 until Phase B can address Eigen separately.
// Phase A's new primitives (Error, CancellationToken,
// ProgressObserver) intentionally avoid std::span and designated
// initializers so this floor is sufficient.

// Forward declarations for geometry-central types
namespace geometrycentral {
namespace surface {
class ManifoldSurfaceMesh;
class VertexPositionGeometry;
class IntrinsicGeometryInterface;
}
}

namespace nxr::manifold {

// Cross-cutting types live in nxr::core; bring them into manifold scope
// so unqualified uses below (Error, CancellationToken, ProgressObserver)
// resolve. See <nxr/errors.h>, <nxr/cancellation.h>, <nxr/progress.h>.
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;

// Forward declarations for facets and geometry bundles.
namespace facet {
    class TopologyFacet;  class EmbeddedFacet;  class IntrinsicFacet;
    class ExtrinsicFacet; class GaugeFacet;     class OperatorsFacet;
}
namespace geometry { struct MeshGeometry; struct MeshTopology; }
namespace ops { struct DECOperators; struct DecMatrices; class CholeskyCache; }
namespace ops::laplacian::connection { enum class CovariantCoupling; }

enum class GaugeType { Euclidean, LeviCivita, Trivial };

enum class OperatorId {
    LaplacianCotan, LaplacianGraph, LaplacianConnection, LaplacianCovariant,
    Dec, MassMixedVoronoi, MassBarycentric, MassGalerkin, Gradient3D, Dirac, DiracFace,
    DiracD, DiracFaceD, DiracIntrinsicD, DiracFaceIntrinsicD, GradFace, LapFace,
    ConnectionGradient, ExtrinsicWeitzenbock, HodgeDirac, SpinDirac
};

// ── Compute Context ──────────────────────────────────────────
// Owns the geometry-central mesh and geometry objects.
// All compute functions operate on a context, which can be created
// once and reused across multiple calls.

// Which mesh the operators (cotan Laplacian, masses, DEC, connection Laplacian, Hodge–Dirac) are built on.
//   Intrinsic (DEFAULT): the intrinsic mollified Delaunay mesh of nxr/delaunay.h, with the mixed-Voronoi mass of the mollified input.
//   Embedded: the input triangulation as it is (no mollification, no flips) — a NAMED OPTION, never a silent default.
enum class MeshKind { Intrinsic, Embedded };

struct ManifoldOptions {
    MeshKind                    mesh     = MeshKind::Intrinsic;
    delaunay::DelaunayOptions   delaunay = {};     // `mollify` and `flipEps`: named, recorded options (shared names with the Studio and nxr)
};

class Manifold {
public:
    // The operator mesh is the intrinsic mollified Delaunay mesh unless `options.mesh` is Embedded.
    Manifold(const double* vertices, int nV, const int32_t* faces, int nF, const ManifoldOptions& options);
    // The flag form: `intrinsicDelaunay` true (the DEFAULT) = MeshKind::Intrinsic, false = MeshKind::Embedded.
    Manifold(const double* vertices, int nV,
                   const int32_t* faces, int nF,
                   bool intrinsicDelaunay = true,
                   const delaunay::DelaunayOptions& delaunay = {});
    ~Manifold();

    // Non-copyable
    Manifold(const Manifold&) = delete;
    Manifold& operator=(const Manifold&) = delete;

    int nV() const;
    int nE() const;
    int nF() const;

    // Access to underlying geometry-central objects (for advanced use).
    // The mesh is a ManifoldSurfaceMesh — required for tangent-space-aware
    // features (curvature principal directions, BFF parametrization,
    // flip-out geodesics). Construction throws if the input is non-manifold;
    // cortical surfaces from FreeSurfer satisfy the constraint.
    geometrycentral::surface::ManifoldSurfaceMesh& mesh();
    geometrycentral::surface::VertexPositionGeometry& geometry();

    // True iff the operator mesh is the intrinsic mollified Delaunay mesh (the default).
    bool isIntrinsicDelaunay() const;

    // THE intrinsic surface (nxr/intrinsic_surface.h, geometry-central's signpost intrinsic Delaunay triangulation of the mollified input): every
    // intrinsic operator (cotan Laplacian, masses, DEC, connection Laplacian, Hodge–Dirac) and every tangent quantity is built on it, in ONE gauge —
    // geometry-central's. With MeshKind::Embedded it is the input as it is. geometry() is unchanged and still used for EXTRINSIC quantities.
    // (`intrinsic()` below is the older data facet; it keeps its name.)
    IntrinsicSurface& intrinsicSurface();
    const ManifoldOptions& options() const;
    // The cotangent Laplacian of the operator mesh, held for the Manifold's lifetime (ManifoldOperators views it).
    const Eigen::SparseMatrix<double>& operatorCotan();
    // The operator mesh's DEC matrices (d0, d1, ⋆0 = the mixed-Voronoi mass, raw ⋆1, ⋆2, ⋆1⁻¹), and A/3 per vertex on it; lazily built, held for the lifetime.
    const ops::DecMatrices& operatorDecMatrices();
    const Eigen::VectorXd&  operatorDualAreas();

    // ── Geometry facets (additive; thin views over the accessors above) ──
    // Each returns a lightweight view object; see include/nxr/facets.h.
    facet::TopologyFacet  topology();
    facet::EmbeddedFacet  embedded();
    facet::IntrinsicFacet intrinsic();
    facet::ExtrinsicFacet extrinsic();

    // Raw-input aliases (facet-agnostic literal input, retained from ctor).
    const Eigen::MatrixXd& vertexPositions() const;  // [nV,3]
    const Eigen::MatrixXi& faces() const;            // [nF,3], 0-based

    // Lazily-cached light per-element bundles backing the data facets.
    const geometry::MeshGeometry& lightGeometry();   // geometry::meshGeometry(*this), cached
    const geometry::MeshTopology& topologyData();    // geometry::meshTopology(*this), cached

    // ── Gauge workflow ──
    // Singularity-aware constructor (pattern 2): default active gauge = Trivial.
    // Throws Error(InvalidInput) if Σ singularities != eulerCharacteristic().
    Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
             const std::map<int,double>& singularities, bool intrinsicDelaunay = true,
             const delaunay::DelaunayOptions& delaunayOptions = {});

    int eulerCharacteristic() const;                 // nV - nE + nF
    GaugeType activeGaugeType() const;
    const std::map<int,double>& activeSingularities() const;
    // Re-point the active/default gauge. Trivial validates Gauss-Bonnet.
    void setGauge(GaugeType type, const std::map<int,double>& singularities = {});

    // Declared only; bodies implemented in B2 (require GaugeFacet complete type).
    facet::GaugeFacet gauge();
    facet::GaugeFacet gauge(GaugeType type,
                            const std::map<int,double>& singularities = {});

    // Lazy operator-assembly helpers shared by gauge + operators facets.
    // The DEC of the OPERATOR mesh (follows the mesh option: d0ᵀ⋆1d0 == the cotan Laplacian, ⋆0 == the mixed-Voronoi mass), cached.
    ops::DECOperators& decOperators();
    // The DEC of the INPUT mesh (geometry-central's, embedded) — edges and faces are the input's: for what is indexed by them (a 1-form given on the
    // input's edges, Whitney interpolation on its faces, the trivial connection, the face Dirac), cached.
    ops::DECOperators& decEmbedded();
    ops::CholeskyCache& choleskyCache();   // owned cache, lazy-initialised

    // ── Operators facet ──
    facet::OperatorsFacet operators();
    void releaseOperator(OperatorId id);        // drop a cached operator
    bool isOperatorCached(OperatorId id) const;

private:
    // Per-operator independent cache slots. Requesting one NEVER builds another.
    // (Dec reuses decCache_ from the gauge path.)
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianCotan_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianGraph_;
    std::unique_ptr<Eigen::SparseMatrix<std::complex<double>>>           cacheLaplacianConnection_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLaplacianCovariant_;
    // Coupling the covariant cache was built with (as int, since CovariantCoupling
    // is only forward-declared here) — rebuild on mismatch so covariant(Product)
    // and covariant(Ambient) on one handle never alias. -1 = nothing cached yet.
    int                                                                  cachedCovariantCoupling_ = -1;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheMassMixedVoronoi_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheMassBarycentric_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheMassGalerkin_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheGradient3D_;
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheExtrinsicWeitzenbock_; // Δ₃+W_ext [3V×3V], world-frame
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDirac_;      // extrinsic block E
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracFace_;  // face extrinsic block Ẽ
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracD_;     // first-order Dirac D [4F×4V]
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracFaceD_; // first-order face Dirac D̃ [4V×4F]
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracIntrinsicD_; // first-order INTRINSIC Dirac D_int [4F×4V]
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheDiracFaceIntrinsicD_; // first-order INTRINSIC face Dirac D̃_int [4V×4F]
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheTwoFormLaplacian_;  // K̃ = d₁⋆₁⁻¹d₁ᵀ (diracFace intrinsic anchor)
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheGradFace_;   // dual-mesh face gradient G̃ [3F×F]
    std::unique_ptr<Eigen::SparseMatrix<double>>                         cacheLapFace_;    // face Laplacian K̃ = G̃ᵀ⋆_F G̃ [F×F]
    std::unique_ptr<Eigen::SparseMatrix<std::complex<double>>>           cacheConnectionGradient_; // d^∇ [E×V] complex, built in active gauge
    int                                                                  cachedConnectionGradientNSym_ = -1; // rebuild on nSym change

    // Private cache-fill helpers called by OperatorsFacet::LaplacianView.
    // These source DIRECTLY from the operator mesh without
    // calling assembleManifoldOperators (which would fuse cotan+mass+normals).
    const Eigen::SparseMatrix<double>& cotanLaplacianCached_();
    const Eigen::SparseMatrix<double>& graphLaplacianCached_();

    // Private cache-fill helpers for OperatorsFacet::MassView.
    // Each sources DIRECTLY from the operator mesh — independent:
    // requesting one mass NEVER builds another.
    const Eigen::SparseMatrix<double>& massMixedVoronoiCached_();
    const Eigen::SparseMatrix<double>& massBarycentricCached_();
    const Eigen::SparseMatrix<double>& massGalerkinCached_();

    // Private cache-fill helpers for OperatorsFacet::LaplacianView::connection/covariant.
    // connection: builds in the ACTIVE gauge (LC or trivial); stores K_complex.
    // covariant: delegates to connection() + gauge().grid() + cotan() then assembles.
    const Eigen::SparseMatrix<std::complex<double>>& connectionLaplacianCached_();
    const Eigen::SparseMatrix<double>& covariantLaplacianCached_(
        ops::laplacian::connection::CovariantCoupling coupling);

    // Covariant gradient G (3E x 3N) — differential::covariantGradient(*this), cached
    // (mesh-only; depends on the frames, not the gauge or any parameter).
    const Eigen::SparseMatrix<double>& gradient3DCached_();

    // Extrinsic vector Weitzenböck Δ₃ + W_ext (3V×3V), cached (OperatorId::ExtrinsicWeitzenbock).
    // The imaginary (vector) block of the immersion squared Dirac 2·dirac(0.5).
    const Eigen::SparseMatrix<double>& extrinsicWeitzenbockCached_();

    // Relative-Dirac extrinsic block E (4V×4V), cached (OperatorId::Dirac).
    const Eigen::SparseMatrix<double>& diracExtrinsicBlockCached_();
    // Assemble L(τ) = (1−τ)(cotanL⊗I₄) + τ·E by value. τ ∈ [0,1] (validated).
    Eigen::SparseMatrix<double> diracFamily_(double tau);
    // First-order Dirac D (4F×4V), cached (OperatorId::DiracD). ops::dirac::matrix.
    const Eigen::SparseMatrix<double>& diracMatrixCached_();
    // First-order INTRINSIC Dirac D_int (4F×4V), cached (OperatorId::DiracIntrinsicD).
    // ops::dirac::matrixIntrinsic (immersion/edge-based; the spin-connection root).
    const Eigen::SparseMatrix<double>& diracIntrinsicMatrixCached_();

    // Face-domain relative-Dirac extrinsic block Ẽ (4F×4F), cached (OperatorId::DiracFace).
    const Eigen::SparseMatrix<double>& diracFaceExtrinsicBlockCached_();
    // First-order face Dirac D̃ (4V×4F), cached (OperatorId::DiracFaceD). ops::dirac::matrixFace.
    const Eigen::SparseMatrix<double>& diracFaceMatrixCached_();
    // First-order INTRINSIC face Dirac D̃_int (4V×4F), cached (OperatorId::DiracFaceIntrinsicD).
    // ops::dirac::matrixFaceIntrinsic (centroid immersion dual of matrixFace).
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicMatrixCached_();
    // DEC 2-form Laplacian K̃ = d₁⋆₁⁻¹d₁ᵀ ([F×F]) — the diracFace intrinsic anchor.
    // Cached (mesh-fixed; not user-facing, so no OperatorId) so a τ-sweep reuses it.
    const Eigen::SparseMatrix<double>& twoFormLaplacianCached_();
    // Assemble L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ by value, K̃ = d₁⋆₁⁻¹d₁ᵀ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFaceFamily_(double tau);
    // Barycentric dual-mesh face gradient G̃ (3F×F), cached (OperatorId::GradFace).
    // ops::facegrad::gradient.
    const Eigen::SparseMatrix<double>& gradFaceCached_();
    // Face Laplacian K̃ = G̃ᵀ⋆_F G̃ (F×F), cached (OperatorId::LapFace). ops::facegrad::laplacian.
    const Eigen::SparseMatrix<double>& lapFaceCached_();

    friend class facet::OperatorsFacet;
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>             mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>         geometry_;
    ManifoldOptions                          options_;
    std::unique_ptr<IntrinsicSurface>        surface_;           // the one intrinsic construction (flips only when options_.mesh == Intrinsic)
    std::unique_ptr<Eigen::SparseMatrix<double>> operatorCotan_; // its cotan Laplacian, never released
    std::unique_ptr<ops::DecMatrices>        operatorDec_;       // the DEC of the operator mesh
    std::unique_ptr<Eigen::VectorXd>         operatorDualAreas_;

    Eigen::MatrixXd                          vertexPositions_;  // [nV,3] retained input
    Eigen::MatrixXi                          faces_;            // [nF,3] retained input (0-based)
    std::unique_ptr<geometry::MeshGeometry>  lightGeometryCache_;
    std::unique_ptr<geometry::MeshTopology>  topologyDataCache_;

    GaugeType            activeGaugeType_    = GaugeType::LeviCivita;
    std::map<int,double> activeSingularities_;
    // Validates Σ singularities == eulerCharacteristic(); throws on mismatch.
    void validateSingularities_(const std::map<int,double>& s) const;

    std::unique_ptr<ops::DECOperators>  decCache_;
    std::unique_ptr<ops::DECOperators>  decEmbeddedCache_;
    std::unique_ptr<ops::CholeskyCache> choleskyCachePtr_;
};

// Extrinsic Delaunay edge-flip repair (geometry-central fixDelaunay).
// Flips edges in place — same vertices (positions/count/indices), new faces.
// Best-effort PSD (reduces obtuse/negative cotan weights), not a certificate;
// the intrinsic-Delaunay certificate is a separate facility. Input must be a
// valid manifold (the underlying ManifoldSurfaceMesh throws otherwise).
struct DelaunayNormalization {
    Eigen::MatrixXi faces;  // [nF, 3] 0-based, re-triangulated; same vertex indices
    int flips = 0;          // edge flips performed (0 => already Delaunay)
};
DelaunayNormalization fixDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF);

} // namespace nxr::manifold

namespace nxr::manifold::ops {

// ── Operator Assembly ────────────────────────────────────────

/**
 * Mass-matrix variant for the FEM L² inner product ⟨u, v⟩ = uᵀ M v.
 *
 * - MixedVoronoi (DEFAULT): diagonal; each vertex's mixed Voronoi area (Meyer, Desbrun, Schröder & Barr 2003) of the
 *   mollified INPUT triangulation, recorded before the first flip (`delaunay::Mesh::mass`) — the mass of the Studio, nxr and Rheome. With
 *   MeshKind::Embedded: of the input as it is.
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
// Lifetime contract: bindings must keep the owning Manifold alive
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
    ManifoldOperators(const Eigen::SparseMatrix<double>& cotanLaplacian_,
                      Eigen::SparseMatrix<double> mass_,
                      MassMatrixVariant massVariant_,
                      Eigen::VectorXd vertexDualAreas_,
                      VertexNormalsMatrix vertexNormals_,
                      double totalArea_)
      : cotanLaplacian(cotanLaplacian_),
        mass(std::move(mass_)),
        massVariant(massVariant_),
        vertexDualAreas(std::move(vertexDualAreas_)),
        vertexNormals(std::move(vertexNormals_)),
        totalArea(totalArea_) {}

    const Eigen::SparseMatrix<double>& cotanLaplacian;  // view: the operator mesh's cotangent Laplacian (Manifold::operatorCotan)
    Eigen::SparseMatrix<double> mass;                   // owned: depends on massVariant
    MassMatrixVariant massVariant;                      // which variant produced `mass`
    Eigen::VectorXd vertexDualAreas;                    // [nV] A/3-per-vertex on the operator mesh (variant-independent)
    VertexNormalsMatrix vertexNormals;                  // [nV, 3] row-major; see alias comment above
    double totalArea;
    // nV / nE / nF intentionally omitted — read from the owning
    // ComputeContext (ctx.nV(), ctx.nE(), ctx.nF()) to avoid duplicating
    // mesh metadata that's already canonical on the context.
};

/** Assemble operators from a context, on the Manifold's operator mesh (nxr/delaunay.h). Default mass variant is MixedVoronoi. */
ManifoldOperators assembleManifoldOperators(
    Manifold& m,
    MassMatrixVariant variant = MassMatrixVariant::MixedVoronoi);

/** The mass matrix of a variant on the Manifold's operator mesh (diagonal for MixedVoronoi and Barycentric). */
Eigen::SparseMatrix<double> vertexMass(Manifold& m, MassMatrixVariant variant = MassMatrixVariant::MixedVoronoi);

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

    // All fields are views into matrices the Manifold holds (the operator mesh's DEC, or for decEmbedded() geometry-central's cached
    // ones) — see the comment block above ManifoldOperators for the lifetime contract. dec.hodge0 is the same matrix as ops.mass when
    // ops.massVariant == MixedVoronoi (the default).
    const Eigen::SparseMatrix<double>& d0;      // vertex → edge
    const Eigen::SparseMatrix<double>& d1;      // edge → face
    const Eigen::SparseMatrix<double>& hodge0;  // vertex → vertex (diagonal)
    const Eigen::SparseMatrix<double>& hodge1;  // edge → edge (diagonal)
    const Eigen::SparseMatrix<double>& hodge2;  // face → face (diagonal)
    const Eigen::SparseMatrix<double>& hodge1Inverse;
};

// The matrices of the operator mesh's DEC, held by the Manifold (DECOperators views them).
struct DecMatrices {
    Eigen::SparseMatrix<double> d0, d1, hodge0, hodge1, hodge2, hodge1Inverse;
};

// The stand-alone DEC FOLLOWS THE MESH OPTION: on the Manifold's operator mesh (the intrinsic mollified Delaunay mesh by default,
// the input with MeshKind::Embedded), so that d0ᵀ ⋆1 d0 == the cotan Laplacian (`operators().laplacian().cotan()`) and ⋆0 == the mass of the
// default mass (the mixed-Voronoi mass of the mollified input; ops::massMatrix variants Barycentric and Galerkin are other masses). ⋆1 is the RAW cotan weight, NOT floored
// (a cocircular edge gives ⋆1 = 0 and hodge1Inverse is then singular; the floored ⋆1 of the Hodge–Dirac pencil is `star1Floor` in
// ops::hodgedirac::assemble). Edges and faces are the operator mesh's (the flipped faces under the intrinsic option; edge e keeps its index through
// the flips), vertices are the input's. What is indexed by the INPUT's edges and faces uses `Manifold::decEmbedded()`.
DECOperators assembleDECOperators(Manifold& m);

// Zero-copy passthrough accessors, valid for the Manifold's lifetime, of the operator mesh's DEC (the same matrices as decOperators()).
const Eigen::SparseMatrix<double>& d0(Manifold& m);
const Eigen::SparseMatrix<double>& d1(Manifold& m);
const Eigen::SparseMatrix<double>& hodge0(Manifold& m);
const Eigen::SparseMatrix<double>& hodge1(Manifold& m);
const Eigen::SparseMatrix<double>& hodge2(Manifold& m);
const Eigen::SparseMatrix<double>& hodge1Inverse(Manifold& m);
// A/3 per vertex on the operator mesh (variant-independent).
const Eigen::VectorXd&             vertexDualAreas(Manifold& m);

// Graph (combinatorial) Laplacian L = D − A, built as d0ᵀ d0 from the
// metric-free exterior derivative. Pure topology — independent of vertex
// positions. Symmetric PSD; diagonal = vertex degree,
// off-diagonal = −1 for adjacent vertices, 0 otherwise.
Eigen::SparseMatrix<double> graphLaplacian(Manifold& m);

// Extrinsic block of the relative Dirac family (Liu/Jacobson/Crane, SGP 2017):
// E = Dᵀ ⋆_F D, the [4V×4V] real-symmetric PSD Galerkin form of the relative
// Dirac operator D_N (the shape-operator / Gauss-map energy). Quaternion order
// [w,x,y,z]; index 4*v+c. Geometry-only (vertex normals + face areas) — the one
// operator assembled directly rather than wrapped from geometry-central.
namespace dirac {
// First-order (rectangular) extrinsic Dirac operator D : [4F×4V] — maps a
// quaternionic field on vertices to one on faces. This is the operator whose
// area-weighted Galerkin square is E = Dᵀ ⋆_F D (extrinsicBlock builds E from
// THIS matrix, so the two never drift). Quaternion order [w,x,y,z]; row 4*f+c,
// column 4*v+c. Geometry-only (vertex normals + face areas).
Eigen::SparseMatrix<double> matrix(Manifold& m);
Eigen::SparseMatrix<double> extrinsicBlock(Manifold& m);
// First-order (rectangular) INTRINSIC Dirac operator D_int : [4F×4V] — identical
// block structure to matrix(), but built from the IMMERSION (vertex positions /
// opposite edge vectors) instead of the Gauss map. This is the spin-connection /
// tangent-frame-transport root; its area-weighted Galerkin square Dᵀ ⋆_F D equals
// the intrinsic Dirac² (the cotan-Laplacian-scalar-part operator). Geometry-only
// (vertex positions + face areas). Quaternion order [w,x,y,z].
Eigen::SparseMatrix<double> matrixIntrinsic(Manifold& m);
// Face-domain (Poincaré dual) first-order Dirac operator D̃ : [4V×4F] — maps a
// quaternionic field on faces to one on vertices, from EXACT per-face normals
// aggregated over vertex stars (the dual of matrix). Its Galerkin square is the
// face-domain extrinsic block Ẽ = D̃ᵀ ⋆_V D̃, the [4F×4F] real-symmetric PSD
// operator (extrinsicBlockFace builds Ẽ from THIS matrix). Quaternion order
// [w,x,y,z]; row 4*v+c, column 4*f+c. Geometry-only (face normals + vertex dual
// areas). Closed-mesh v1: both throw Error(InvalidInput) on an open boundary
// (vertex stars must be closed) — exact on closed cortical hemispheres.
Eigen::SparseMatrix<double> matrixFace(Manifold& m);

// matrixFaceIntrinsic: INTRINSIC face-domain Dirac D̃_int [4V×4F] — the centroid
// (immersion) dual of matrixFace (face centroids instead of the Gauss map), mirroring
// how matrixIntrinsic relates to matrix. Geometry-only; closed-mesh v1.
Eigen::SparseMatrix<double> matrixFaceIntrinsic(Manifold& m);
Eigen::SparseMatrix<double> extrinsicBlockFace(Manifold& m);
}  // namespace dirac

// ── Hodge–Dirac d + δ on 0-, 1- and 2-forms of the INTRINSIC Delaunay triangulation ──
// Port of nxr-cortical-flow hodgeDirac.ts (the reference where implementations differ). The pencil (K, M),
// N = V + E + F, rows ordered 0-forms [0,V) · 1-forms [V,V+E) · 2-forms [V+E,N):
//     K = [0 d0ᵀ⋆1 0; ⋆1 d0 0 d1ᵀ⋆2; 0 ⋆2 d1 0],   M = diag(⋆0, ⋆1, ⋆2).
// ⋆0 mixed-Voronoi mass of the mollified input (obtuse corner: A/2 and A/4), ⋆1 = ½(cot α + cot β), ⋆2 = 1/area. M⁻¹K = d + δ is
// M-self-adjoint with spectrum ±μ; D² is block-diagonal in the Hodge Laplacians. The triangulation is built from the
// input's edge lengths by THE one construction (nxr/delaunay.h, the Studio's intrinsicDelaunay.ts): mollify (Sharp & Crane 2020, relative factor `mollify`),
// then flipToDelaunay (tolerance `flipEps`). Both are named, recorded options (`record`).
// Vertex indices are the input's; edges and faces are the intrinsic mesh's (edge orientation = its halfedge).
// ⋆1 is the raw intrinsic cotan weight (nxr, the reference;). A named option `star1Floor` > 0 raises it to that floor so that M > 0 on
// near-cocircular edges (Rheome PR #94 floored at 1e-6; being changed to match); 0 = off, the default.
namespace hodgedirac {
struct HodgeDirac {
    Eigen::SparseMatrix<double> K;      // [N×N] symmetric
    Eigen::SparseMatrix<double> d0, d1; // [E×V], [F×E]
    Eigen::VectorXd star0, star1, star2;// diagonal Hodge stars (star1 after the floor)
    Eigen::VectorXd M;                  // [star0; star1; star2]
    int nV = 0, nE = 0, nF = 0;
    int star1Clamped = 0;               // edges whose cotan weight was raised to the floor
    double mollifyEps = 0.0;            // length added to every edge by mollify (0 = none needed)
    Eigen::MatrixXi faces;              // [nF, 3] the intrinsic triangulation's faces (vertex indices are the input's)
    IntrinsicRecord record;             // mollify, ε, flipEps, flips, gauge of the construction
    // the intrinsic complex in the stored layout (IntrinsicSurface::Complex): corners 3f + k, edges tail → tip, signed face edges (1-based), lengths,
    // the raw signpost angles of each corner's halfedge, the cone angles and the reference neighbours (angle 0 of the gauge)
    IntrinsicSurface::Complex complex;
};
// Uses the Manifold's operator mesh when it is the intrinsic one built with these options, else builds its own by the same construction.
HodgeDirac assemble(Manifold& m, double star1Floor = 0.0, double mollify = 1e-6, double flipEps = 1e-6);
Eigen::SparseMatrix<double> massMatrix(const HodgeDirac& h);   // diag(M)
}  // namespace hodgedirac

// ── Spin Diracs on quaternion spinors per FACE (Hoffmann & Ye 2018; Ye et al. 2018) ──
// Port of cortical-flow spinDirac.ts / Rheome operators.spin_dirac. Closed, oriented meshes. Hyperedge of face i across
// the halfedge e to face j: E_ij = |e| tan(θ/2) + e. 4F×4F symmetric, interleaved [w,x,y,z] at 4f+c:
//   Extrinsic   D_f = ½ Σ_j E_ij − H_i      (spin transformations; kills the constants)
//   IntrinsicHY D   = ½ Σ_j E_ij            (= D_f + H)
//   IntrinsicYe D   = ½ Σ_j cos(θ/2) E_ij   (edge lengths only)
// The eigenproblem is D φ = μ (area ⊗ I₄) φ, solved on VERTEX spinors through vertexToFace (see solve::eigenSpin).
namespace spin {
enum class Kind { Extrinsic, IntrinsicHY, IntrinsicYe };
struct SpinDirac {
    Eigen::SparseMatrix<double> D;   // [4F×4F]
    Eigen::VectorXd H;               // [F] integrated mean curvature per face
    Eigen::VectorXd area;            // [F]
};
SpinDirac assemble(Manifold& m, Kind kind);
// A (4F×4V): (Aψ)_f = Σ_{v∈f} p_fv ψ_v. covariant: p_fv = unit quaternion turning the area-weighted vertex normal onto the face's
// (Ye et al. adj_vf_cov; for the intrinsic operators); else 1 (adj_vf; for the extrinsic).
Eigen::SparseMatrix<double> vertexToFace(Manifold& m, bool covariant);
}  // namespace spin

// Face-native (Poincaré-dual) scalar calculus, the dual of the vertex FEM
// gradient + cotan Laplacian. DOFs live on FACES (dual vertices).
namespace facegrad {
// gradient: barycentric dual-mesh gradient of a per-face scalar — a per-face
// ambient 3-vector field [3F×F]. Green–Gauss / DEC construction:
//   grad ψ|_f = (1/2A_f) Σ_k (ψ_{g_k} − ψ_f) (l_k × n_f)
// over face f's three edge-neighbors g_k (l_k = boundary edge vector, l_k × n_f
// = the outward in-plane edge normal). Σ_k l_k = 0 ⇒ constants annihilated and
// the self term vanishes; output is tangent to each face. Closed-mesh v1 (every
// face has exactly 3 edge-neighbors): throws Error(InvalidInput) on a boundary.
Eigen::SparseMatrix<double> gradient(Manifold& m);
// laplacian: the face Laplacian K̃ = gradientᵀ ⋆_F gradient [F×F] (⋆_F = face
// areas), built from THE SAME gradient so the two never drift. Symmetric PSD;
// kernel = constants (genus-0). The dual of the cotan Laplacian.
Eigen::SparseMatrix<double> laplacian(Manifold& m);
}  // namespace facegrad

// ── Factor Cache ─────────────────────────────────────────────
//
// Pre-factored Cholesky / LU decompositions of mesh-derived
// matrices, cached across compute calls. Each factor is built
// lazily on first request and reused on every subsequent call
// for the lifetime of the owning Manifold.
//
// Why this exists: mesh-derived sparse factorizations cost
// O(n^1.5) to build but only O(nnz) per back-substitution. Each
// solver call (Poisson, Hodge α-solve, direction-field) hits
// the same matrix; without caching we redo that O(n^1.5) work
// on every click and on every frame of any future per-frame
// pipeline. Both geometry-processing-js and geometry-central
// pre-factor and reuse for this reason.
//
// Bit-for-bit contract: each cached factor is built by the same
// matrix-assembly + factorization sequence the previous inline
// solvers used (same regularization ε, same coefficient order,
// same Eigen solver type). Refactored callers must produce
// numerically identical output to the pre-cache implementation.
//
// Lifetime: owned by ContextHolder in the addon; destroyed when
// the JavaScript handle releases (i.e. when the mesh changes).
//
// Not cached: the eigensolver's (K - σM) factor lives inside
// Spectra::SymShiftInvert and is discarded per call. Caching
// it would require replacing Spectra's ShiftInvertOp with a
// custom one — out of scope for v1.
//
// Perf options compatible with the modularity rule (CONTRIBUTING.md §10):
// AVX2/AVX-512 compile flags (the build enables these on x86_64),
// Eigen's separate analyzePattern() + factorize() for matrices
// that share sparsity pattern, multi-RHS batched solves, and GPU
// triangular back-substitution via TSL for hot per-frame paths.
// Not on the table: CHOLMOD / SuiteSparse / MKL / PARDISO — they
// break the WASM and MEX targets.

class CholeskyCache {
public:
    CholeskyCache() = default;
    CholeskyCache(const CholeskyCache&) = delete;
    CholeskyCache& operator=(const CholeskyCache&) = delete;

    /** Diagonal regularization added to every factored matrix.
     *  Identical to the constant the pre-cache inline solvers used. */
    static constexpr double kRegularization = 1e-8;

    /** LLT of (K + ε·I) for any symmetric PSD K. The cache is keyed
     *  by lifetime, not content — the *first* matrix passed in is the
     *  one factored, and subsequent calls return that factor regardless
     *  of what's passed. Use a fresh CholeskyCache per matrix. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const Eigen::SparseMatrix<double>& K);

    /** Convenience overload: factors ops.cotanLaplacian. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const ManifoldOperators& ops) {
        return laplacian(ops.cotanLaplacian);
    }

    /** LLT of (d0ᵀ ★₁ d0 + ε·I).
     *  Used by both Hodge α-solve and direction-field. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& hodgeExact(
        const DECOperators& dec);

    /** LU of (d1 ★₁⁻¹ d1ᵀ + ε·I), used by Hodge β-solve.
     *  B is symmetric PSD after regularization, so SimplicialLLT
     *  would also work; SparseLU preserved for bit-for-bit
     *  equivalence with the pre-cache implementation. */
    const Eigen::SparseLU<Eigen::SparseMatrix<double>>& hodgeCoExact(
        const DECOperators& dec);

private:
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> laplacian_;
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> hodgeExact_;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> hodgeCoExact_;
};

} // namespace nxr::manifold::ops

namespace nxr::manifold::ops::laplacian::connection {

// ── Connection Laplacian ─────────────────────────────────────
//
// Connection Laplacian on a chosen domain (vertex / face / edge).
// The operator that drives smoothest n-RoSy direction fields,
// parallel-transport energies, and tangent-bundle eigendecompositions.
//
// Mathematically the operator is complex Hermitian on the chosen
// domain (V, F, or E entries), built from cotangent weights and
// Levi-Civita transport rotations raised to the nSym power. The
// `Real2N` format (default) lowers each complex entry a + bi to the
// 2×2 real block [[a, -b], [b, a]], producing a 2N×2N symmetric real
// matrix that drops directly into nxr-compute's existing real-only
// eigensolver (`solve`) when paired with a block-diagonal
// real mass matrix `blkdiag(M, M)`. The smallest eigenpair then
// reproduces the smoothest n-direction field that
// `computeSmoothest{Vertex,Face}DirectionField` returns internally.
//
// `Complex` format leaves the matrix as complex Hermitian; bindings
// surface real and imaginary triplet arrays separately. Useful when
// callers want to apply a complex-Hermitian eigensolver themselves.
//
// MATLAB reference: none in `+bct` today (the existing direction-field
// reference uses trivial connections; connection-Laplacian-based
// smoothest fields delegate to geometry-central). When a MATLAB
// oracle harness lands, mirror it under
// `+bct.+manifold.+operator.connectionLaplacian`.

enum class ConnectionDomain {
    Vertex,                  // V × V (intrinsic vertex CL)
    Face,                    // F × F (intrinsic face CL — geometry-central
                             // has FIXME on face weights, propagated as a
                             // doc caveat below)
    EdgeCrouzeixRaviart      // E × E (intrinsic Crouzeix-Raviart CL)
};

enum class ConnectionLaplacianFormat {
    Real2N,                  // 2N × 2N symmetric real (default)
    Complex                  // N × N complex Hermitian, surfaced as
                             // parallel real/imag triplet arrays at
                             // the binding edge
};

/** Options bag for `assembleConnectionLaplacian`. Each field is
 *  defaulted; a value-initialized struct produces the default
 *  vertex / nSym=1 / regularization=1e-8 / Real2N variant. */
struct ConnectionLaplacianOptions {
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int    nSym                      = 1;       // 1=vector, 2=line, 4=cross
    double regularization            = 1e-8;    // additive ε·I shift
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Result of `assembleConnectionLaplacian`. Exactly one of `K_real`
 *  / `K_complex` is populated, selected by `options.format`. */
struct ConnectionLaplacian {
    Eigen::SparseMatrix<double>                K_real;     // populated iff format == Real2N
    Eigen::SparseMatrix<std::complex<double>>  K_complex;  // populated iff format == Complex
    Eigen::MatrixXd  frameE1;   // [baseDim, 3] tangent frame e1 per vertex/face (gauge)
    Eigen::MatrixXd  frameE2;   // [baseDim, 3] tangent frame e2 per vertex/face (gauge)
                                // Empty (0 rows) for EdgeCrouzeixRaviart domain.
    int  baseDim   = 0;                                    // V / F / E
    int  outputDim = 0;                                    // 2 * baseDim (Real2N) or baseDim (Complex)
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int  nSym                        = 1;
    double regularization            = 1e-8;
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Assemble the connection Laplacian on the chosen domain.
 *
 *  Walks halfedges / faces / edges directly using
 *  `geometry.edgeCotanWeights`, `geometry.transportVectorsAlongHalfedge`,
 *  `geometry.transportVectorsAcrossHalfedge`, and applies `.pow(nSym)` to
 *  each transport rotation (mirrors the file-static helpers in
 *  `geometry-central/src/surface/direction_fields.cpp` that drive
 *  `computeSmoothest{Vertex,Face}DirectionField`).
 *
 *  Throws `Error(InvalidInput)` for `nSym <= 0`. The Crouzeix-Raviart
 *  variant requires triangle meshes (geometry-central's own assertion).
 *
 *  Note: the face variant inherits geometry-central's FIXME on face
 *  weights (uniform 1.0). The numerical answer is still well-defined
 *  for nRoSy energy minimization; revisit if a more principled face
 *  weight emerges upstream. */
ConnectionLaplacian assembleConnectionLaplacian(
    Manifold& m,
    const ConnectionLaplacianOptions& opts = {}
);

/** First-order connection gradient d^∇ : tangentVertex [V] (complex) → tangentEdge [E]
 *  (complex). Per edge e=(i→j) via e.halfedge(): row e has +1 at column j and
 *  −transportVectorsAlongHalfedge[he].pow(nSym) at column i. Built in the active gauge,
 *  so (d^∇)ᴴ diag(edgeCotanWeight) d^∇ == the connection Laplacian for the same (nSym,
 *  gauge). Vertex domain (v1).
 *
 *  Throws Error(InvalidInput) for nSym <= 0. */
Eigen::SparseMatrix<std::complex<double>>
assembleConnectionGradient(Manifold& m, int nSym = 1);

/** String → enum helpers for binding shells (WASM, addon).
 *  Domain accepts "vertex", "face", "edge". Format accepts "real2N"
 *  / "real" / "complex". Throws Error(InvalidInput) on unknown. */
ConnectionDomain          parseConnectionDomain(const std::string& name);
ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name);

/**
 * Assemble the connection Laplacian using the trivial connection transport.
 *
 * The trivial connection for the given singularity map is computed first
 * (via computeTrivialConnection), then used to modulate the Levi-Civita
 * transport on each halfedge:
 *
 *   ρ^TC[he] = ρ^LC[he] · exp(i · sign(he) · φ[edge(he)])
 *
 * where sign(he) = +1 if he == edge.halfedge(), -1 otherwise, and
 * φ is the per-edge trivial connection 1-form (Eigen::VectorXd, nEdges).
 *
 * The off-diagonal entry K[i,j] for halfedge he (tail=i, tip=j) is then:
 *
 *   K[i,j] += -w_ij · (ρ^TC[he.twin()])^nSym
 *
 * Gauss-Bonnet: Σ singularityMap values must equal χ(mesh). Not checked
 * internally — throws nothing if violated, but φ will be incorrect.
 *
 * Only ConnectionDomain::Vertex is currently supported. Passing Face or
 * EdgeCrouzeixRaviart throws Error(InvalidInput).
 *
 * Output format follows opts.format (Real2N or Complex), same as
 * assembleConnectionLaplacian.
 */
ConnectionLaplacian assembleTrivialConnectionLaplacian(
    Manifold& m,
    const std::map<int, double>& singularityMap,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const ConnectionLaplacianOptions& opts = {}
);

// ── Covariant (3-frame) Laplacian ─────────────────────────────
//
// 3N×3N real symmetric sparse Laplacian on the full 3-frame
// [a; b; c] = [Re z; Im z; normal], component-major block layout.
// Two couplings: Product (blkdiag) and Ambient (frame-conjugate).
// See spec docs/superpowers/specs/2026-06-08-vector-laplacian-3d-design.md §3–4.

// Coupling for the 3-frame covariant Laplacian (see covariant-laplacian spec).
enum class CovariantCoupling {
    Product,   // blkdiag(real-expand(K), cotanL) — tangent ⊕ normal decoupled
    Ambient    // frame-conjugate of kron(I3, cotanL): L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)
};

// 3N×3N real symmetric Laplacian on the 3-frame [a;b;c] = [Re z; Im z; normal],
// component-major block layout. Pure linear algebra over pieces the caller holds:
//   K        = gauge connection Laplacian (V×V complex Hermitian) — used by Product
//              (and equals the Ambient tangent block for the matching gauge)
//   gaugeGrid= realized gauge frame, V×3 complex c = e1 + i·e2 (normal = Re×Im) — used by Ambient
//   cotanL   = scalar cotan Laplacian (V×V real)
// See spec §3–4. Symmetric PSD.
Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL);

} // namespace nxr::manifold::ops::laplacian::connection

namespace nxr::manifold::solve {

// ── Eigensolver ──────────────────────────────────────────────

struct EigenResult {
    Eigen::MatrixXd eigenvectors;  // [nV, k] columns are eigenvectors
    Eigen::VectorXd eigenvalues;   // [k] sorted ascending (hodgeDiracModes with bothSigns: ascending in signed μ)
    int k;
    int nConverged;
    Eigen::VectorXi kind;          // Hodge–Dirac only: per mode 0 exact · 1 harmonic · 2 coexact (empty otherwise)
};

/**
 * THE DEFAULT SHIFT: none. `kShiftLadder` (NaN) means "solve by the scale-aware SHIFT LADDER" (`eigenSmallest`:
 * σ = −1e-7·λmax, then −1e-4·λmax, then a Cholesky-mode last resort), never a fixed shift on the null space.
 *
 * The old default, a FIXED σ = −1e-8, sits on the operator's null space (the Laplacian's constant, the relative Dirac's four
 * constant quaternions) whatever the spectrum's scale: K − σM is nearly singular and the shift-invert Lanczos returns the low
 * spectrum wrong. Measured 2026-09-29 on a real cortex hemisphere (10 242 vertices): the cotan Laplacian's λ₃ came back 337.4
 * for 317.2, low modes mixed, every mode's residual 0.4–1.6; the relative Dirac's first vector per quartet at 2e-4 — while
 * the ladder gives residuals ~1e-11 on the same pencils. An EXPLICIT σ is still honoured as a fixed shift. Every binding
 * (MEX, Node, wasm) defaults to this.
 */
inline constexpr double kShiftLadder = std::numeric_limits<double>::quiet_NaN();

/**
 * Controls of the shift-invert Lanczos run (an explicit σ). The defaults are the library's long-standing ones (ncv = 2k+1, 1000 restarts,
 * tol 1e-10, a partial result accepted); the Hodge–Dirac reference solve asks for machine precision and for every pair (requireAll).
 */
struct EigenControls {
    int ncv = 0;               // Krylov dimension; 0 = min(max(2k+1, 20), N) (SciPy's eigsh default)
    int maxit = 1000;          // restarts
    double tol = 1e-12;        // Spectra's relative tolerance on the Ritz estimates (ARPACK's tol = 0 means machine precision)
    bool requireAll = false;   // throw EigensolveNotConverged unless all k pairs converged (ARPACK raises as well)
};

/**
 * Solve the generalized eigenvalue problem K φ = λ M φ for k eigenpairs (ascending in λ).
 * Matches MATLAB: eigs(K, M, k, 'SM') for the default shift.
 *
 * @param sigma      Shift. `kShiftLadder` (NaN, the default) = the scale-aware shift ladder, see above. A finite value is a
 *                   FIXED shift-invert centre: the k eigenvalues nearest σ are returned. (K − σM) is factored once and each
 *                   IRAM iteration is a forward/back substitution, so a shift near the wanted end is much faster than
 *                   standard mode on large sparse systems. Never put a fixed σ on the null space.
 * @param normalize  If true, M-orthonormalize eigenvectors so U' * M * U = I (Cholesky whitening). Default false: the
 *                   vectors are returned as Spectra produces them. Set true for downstream spectral synthesis.
 * @param removeDC   If true, strip the null space (every mode with |λ| within a relative tolerance of 0, see `removeDC`).
 *                   Default false.
 * @param cancel     Optional cancellation token, polled once per Spectra perform_op; on true, throws Error(Cancelled).
 * @param progress   Optional observer of solver progress.
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N - 1.
 * Throws Error(Cancelled)          on cancel.requested().
 * Throws Error(EigensolveNotConverged) on Spectra failure with no partial result.
 */
EigenResult eigen(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma                       = kShiftLadder,
    bool normalize                     = false,
    bool removeDC                      = false,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {},
    const EigenControls& controls      = {}
);

/**
 * The k smallest generalized eigenpairs of a pre-assembled symmetric pencil
 * (A, B) with B SPD, ported from Brainstorm's bst_eigs_smallest: estimate the
 * spectrum scale lmax (largest-magnitude, Cholesky mode on B), then a relative
 * shift ladder — σ=-1e-7·lmax → σ=-1e-4·lmax → last-resort Cholesky-mode
 * SmallestMagn. Returns M-orthonormal (if normalize), ascending. Reuses eigen()
 * per shift rung; the last resort factors only the SPD B (never the singular A).
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N-1.
 * Throws Error(EigensolveNotConverged) only if every rung fails.
 */
EigenResult eigenSmallest(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B,
    int k,
    bool normalize                     = true,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {}
);

/**
 * M-orthonormalize an eigenvector basis: U' * M * U = I.
 * Cholesky-whitening fast path with eigen-decomposition fallback.
 * Typically called via `eigen(..., normalize=true)`; this standalone
 * entry point is for transforming pre-computed bases (e.g. arriving
 * from MATLAB / Python).
 */
Eigen::MatrixXd normalize(
    const Eigen::MatrixXd& U,
    const Eigen::SparseMatrix<double>& M
);

/**
 * Strip the null space from an EigenResult: EVERY mode with |λ| <= relTol · max|λ| of the result (default 1e-8), not just one.
 * The null space is one constant per connected component for a Laplacian (two hemispheres: two), and four constant quaternions
 * for the relative Dirac. If no mode is that small (a window away from zero) nothing is removed. `kind`, when present, is kept
 * in step. `*nRemoved` (optional) receives how many modes were removed.
 * Caveat: the tolerance is relative to the largest |λ| in the result, so a result holding only near-zero values keeps them.
 * Typically called via `eigen(..., removeDC=true)`; this standalone entry point strips a pre-computed EigenResult.
 */
EigenResult removeDC(const EigenResult& result, int* nRemoved = nullptr, double relTol = 1e-8);


// ── Named-operator eigensolve ────────────────────────────────
//
// Single source of truth for "named operator → generalized eigenproblem".
// Each operator has a NATURAL generalized mass (the metric its eigenproblem is
// posed against); that pairing is library knowledge, not something every
// consumer should re-derive. `eigenProblemFor` centralizes it; `eigenOperator`
// assembles + eigensolves (+ optional ℍ-multiplet reconstruction / dense path).
// Design: docs/superpowers/specs/2026-06-15-operator-eigensolve-design.md.

enum class EigenOperator {
    LaplacianCotan,   // (cotanL, mass)
    LaplacianGraph,   // (graphL, I)           — standard eigenproblem
    Dirac,            // (dirac(τ) 4V, M_galerkin ⊗ I₄)   — vertex-interleaved 4v+c
    DiracFace,        // (diracFace(τ) 4F, diag(faceArea) ⊗ I₄)
    HodgeDirac,       // (d+δ on 0/1/2-forms of the intrinsic Delaunay mesh, diag(⋆0,⋆1,⋆2)), blockSize 1, N = V+E+F
    SpinExtrinsic,    // spin Diracs on face spinors, solved on vertex spinors (4V) by Ye et al.'s quotient
    SpinIntrinsicHY,
    SpinIntrinsicYe
    // Covariant (3N) deferred: it is component-major (c·N+v, mass = kron(I₃, M),
    // a different expansion than the Dirac's interleaved kron(M, I₄)) and its
    // ambient spectrum is just the triply-degenerate Laplace–Beltrami spectrum.
};

struct EigenOperatorSpec {
    EigenOperator op;
    double tau = 0.0;                                       // Dirac / DiracFace blend ∈ [0,1]
    ops::MassMatrixVariant mass = ops::MassMatrixVariant::MixedVoronoi;  // vertex-domain mass choice
    bool graphDegreeMass = false;   // LaplacianGraph: mass = diag(degree) (L φ = λ D φ) instead of identity
    double star1Floor = 0.0;        // HodgeDirac: floor of ⋆1, 0 = off (default, as nxr)
    double mollify    = 1e-6;       // HodgeDirac: the mollification margin relative to the mean edge length (shared name `mollify`)
    double flipEps    = 1e-6;       // HodgeDirac: an edge is flipped while its cotan weight is below −flipEps (shared name `flipEps`)
    bool bothSigns    = false;      // HodgeDirac, default shift: also return the −μ mirror of every +μ mode
};

struct EigenProblem {
    Eigen::SparseMatrix<double> K;   // the operator
    Eigen::SparseMatrix<double> M;   // its natural generalized mass (block-expanded)
    double sigma;                    // shift for shift-invert
    int blockSize;                   // 1 scalar / 3 covariant / 4 Dirac
};

// Build the (K, M, σ, blockSize) generalized eigenproblem for a named operator.
EigenProblem eigenProblemFor(Manifold& m, const EigenOperatorSpec& spec);

// blockKron(M, b): expand a [n×n] sparse to [bn×bn] block-diagonal (kron(M, I_b),
// entry (i,j,v) → (b·i+c, b·j+c, v) for c=0..b−1). Exposed for callers that build
// their own block masses; used internally by eigenProblemFor.
Eigen::SparseMatrix<double> blockKron(const Eigen::SparseMatrix<double>& M, int b);

// Assemble + eigensolve a named operator. With reconstructMultiplets and a
// blockSize-4 (Dirac) operator, the eigenspaces are closed under right-quaternion
// multiplication to recover exact 4-fold multiplets (Spectra miscounts the
// degenerate clusters). With dense=true, uses Eigen's dense generalized solver
// (exact eigenvalues + multiplicities; small meshes / verification only — throws
// EigensolveInvalidK if the dense size exceeds the cap unless the mesh is small).
EigenResult eigenOperator(
    Manifold& m,
    const EigenOperatorSpec& spec,
    int k,
    double sigma                       = kShiftLadder,
    bool normalize                     = true,
    bool reconstructMultiplets         = false,
    bool dense                         = false,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {}
);

// Spectrum slicing: the `count` lowest eigenvalues of a symmetric pencil (A SPSD, B SPD) in windows of `slice`
// shift-invert solves, so thousands of modes need only a (slice)-column Krylov basis. The first window is the ladder
// solve (eigenSmallest); each next window is centred at the top of the covered range and keeps the eigenvalues above it.
// Throws EigensolveInvalidK when count ≥ N.
// eigenLargestMagn: the k largest-|λ| generalized eigenpairs (Cholesky mode, B SPD, A symmetric) — the inverse-spectrum solve of the spin quotient.
EigenResult eigenLargestMagn(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, int k, bool normalize = true);
EigenResult eigenSliced(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                        int count, int slice = 400, bool normalize = true,
                        const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

// Hodge–Dirac mode kinds from M-orthonormal pencil eigenvectors X (columns): 0 exact (0-form block dominates),
// 2 coexact (2-form block dominates), 1 harmonic (|μ| ≤ tol·max|μ|).
Eigen::VectorXi classifyHodgeModes(const ops::hodgedirac::HodgeDirac& h, const Eigen::MatrixXd& X,
                                   const Eigen::VectorXd& mu, double tol = 1e-8);

// ── The Hodge–Dirac mode solve: the REFERENCE method, and its named options ──
// The reference is the method of nxr (`nxr.eigen.hodge_dirac.solve_hodge`, `nxr.pipelines.hodge.solved_package`) and of Rheome
// (`rheome.eigen.hodge_dirac`, `hodge_package`); the maintainers' decision of 2026-11-10 is that nxr-neuro uses exactly it, by default:
//   * two ladders: exact Δ0 = d0ᵀ⋆1 d0 against ⋆0, coexact Δ2 = (⋆2d1) ⋆1⁻¹ (⋆2d1)ᵀ against ⋆2;
//   * want = count/2 + 1; each ladder asks for min(want + 1, n − 2) pairs (n = nV for Δ0, nF for Δ2);
//   * ONE shift-invert solve per ladder at the absolute σ = −1e-3 (generalised symmetric), machine-precision tolerance, no
//     re-orthonormalisation afterwards; ascending, λ = max(λ, 0);
//   * the constants dropped per ladder (λ > 1e-6 · max λ of the ladder), then ONE λ cut: top = min(max kept λ0, max kept λ2),
//     both ladders truncated at it; μ = √λ; merged by μ (stable: exact before coexact on a tie); kinds 0 exact · 2 coexact;
//     modes with μ ≤ 1e-6 · max μ left out.
// The methods this library used before stay as named options (never defaults):
//   shift     Absolute (σ, default −1e-3)  |  RelativeLadder (eigenSmallest: σ = −1e-7·λmax, then −1e-4·λmax, then a Cholesky last resort)
//   slice     0 = none (one solve per ladder)  |  a window size (spectrum slicing, eigenSliced; the windows use their own shifts)
//   normalize false  |  true (Cholesky whitening: M-orthonormalise afterwards)
// The options are recorded in the stored result's provenance (nxr-pipelines), and a different option is a different cached result.
enum class HodgeShift { Absolute, RelativeLadder };

struct HodgeSolveOptions {
    HodgeShift shift = HodgeShift::Absolute;
    double sigma = -1e-3;      // Absolute: the shift (absolute, not scaled by the spectrum)
    int slice = 0;             // 0 = none
    bool normalize = false;
    double tol = 1e-14;        // Absolute, no slice: Spectra's tolerance (ARPACK's tol = 0 is machine precision; Spectra measures it relative to the Ritz value)
    int ncv = 0;               // Absolute, no slice: Krylov dimension, 0 = min(max(2k+1, 20), n) (SciPy's eigsh default)
    int maxit = 100000;        // Absolute, no slice: restarts allowed (generous: a large k converges in one solve)
    int limit = 0;             // hodgeDiracModes only: keep at most this many of the lowest modes (0 = all under the λ cut, about `count`, as the reference)
};

// One ladder: the k smallest eigenpairs of (S, M), ascending, λ clamped to ≥ 0, by the options' method.
EigenResult hodgeLadder(const Eigen::SparseMatrix<double>& S, const Eigen::SparseMatrix<double>& M, int k, const HodgeSolveOptions& opt = {},
                        const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

struct HodgeLadders {
    EigenResult exact, coexact;        // the whole ladders as solved (ψ [nV×k0], χ [nF×k2]; λ ascending)
    double top = 0;                    // the single λ cut
    struct Mode { double mu; int kind; int col; };   // kind 0 exact · 2 coexact; col indexes the ladder's columns
    std::vector<Mode> modes;           // the kept modes, merged by μ ascending (the constants and the μ ≤ 1e-6·max out)
    int nExact = 0, nCoexact = 0;      // how many of `modes` are of each kind
};

// The reference split: both ladders (`count` as above), the null cut, the single λ cut and the merge. S2 = (⋆2d1)⋆1⁻¹(⋆2d1)ᵀ.
HodgeLadders hodgeLadders(const Eigen::SparseMatrix<double>& L0, const Eigen::SparseMatrix<double>& M0,
                          const Eigen::SparseMatrix<double>& S2, const Eigen::SparseMatrix<double>& M2,
                          int count, const HodgeSolveOptions& opt = {},
                          const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

// The Hodge–Dirac eigenpairs with POSITIVE μ by the exact/coexact split (no indefinite solve), the reference method above:
//   exact    (ψ, dψ/μ, 0)/√2     from Δ0 ψ = λ ψ  (d0ᵀ⋆1 d0, mass ⋆0),                  μ = √λ
//   coexact  (0, δχ/μ, χ)/√2     from Δ2 χ = λ χ  (⋆2d1 ⋆1⁻¹ (⋆2d1)ᵀ, mass ⋆2),         μ = √λ
// merged by μ; the constant modes (μ = 0) are dropped, harmonic 1-forms are NOT produced (use the pencil solve for them).
// The result holds ABOUT `count` modes (all under the single λ cut; options.limit keeps at most that many of the lowest).
// bothSigns adds the −μ mirror (1-form block negated). Result: eigenvalues μ ascending (signed if bothSigns:
// −μ_max … −μ_min, then +μ_min … +μ_max), eigenvectors [N×k] (M-orthonormal when the solve is), kind per column.
EigenResult hodgeDiracModes(const ops::hodgedirac::HodgeDirac& h, int count, bool bothSigns = false,
                            const HodgeSolveOptions& options = {}, const CancellationToken& cancel = {},
                            const ProgressObserver& progress = {});

// Close each eigenvalue cluster of a 4-interleaved quaternion eigenbasis under right-ℍ multiplication and return an
// M-orthonormal basis of the true (multiple-of-4) eigenspaces; recovers multiplets Lanczos under-counts. M is the inner product.
EigenResult reconstructDiracMultiplets(const EigenResult& r, const Eigen::SparseMatrix<double>& M);

// The k spin-Dirac eigenpairs nearest 0 as Ye et al. (dirac_eigen.m): on vertex spinors ψ, AᵀDM⁻¹DAψ = μ AᵀDAψ,
// solved as AᵀDAψ = ν (AᵀDM⁻¹DA + ε AᵀMA) ψ for the k largest |ν|, μ = 1/ν. M = area ⊗ I₄. Returns μ ascending (signed),
// ψ [4V×k] normalised to ψᵀ AᵀMA ψ = 1. A = covariant vertexToFace for the intrinsic kinds, plain for the extrinsic.
EigenResult eigenSpin(Manifold& m, ops::spin::Kind kind, int k,
                      const CancellationToken& cancel = {}, const ProgressObserver& progress = {});


// ── Poisson Solver ───────────────────────────────────────────

/**
 * Solve the scalar Poisson equation: K * φ = -M * (ρ - ρ̄)
 * where ρ̄ is the M-weighted mean of ρ.
 *
 * Uses cache.laplacian(K) for the LLT factor of (K + ε·I), so
 * repeated calls with the same cache skip the factorization step.
 *
 * Agnostic in K and M: works on cotangent Laplacians + lumped mass
 * (mesh case), graph Laplacians + node-weight diagonals (graph case),
 * or any SPD pair. The total mass `Σᵢⱼ Mᵢⱼ` plays the role of the
 * "totalArea" normalization for the weighted mean.
 *
 * @param K           SPD stiffness matrix (Laplacian)
 * @param M           SPD mass / inner-product matrix (typically diagonal)
 * @param cache       factor cache (one per K)
 * @param densityMap  sparse density: index → density value
 * @return            harmonic potential φ
 */
Eigen::VectorXd poisson(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    ops::CholeskyCache& cache,
    const std::map<int, double>& densityMap
);

/** Convenience overload: forwards to (K, M) using ops.cotanLaplacian, ops.mass. */
inline Eigen::VectorXd poisson(
    const ops::ManifoldOperators& ops,
    ops::CholeskyCache& cache,
    const std::map<int, double>& densityMap
) {
    return poisson(ops.cotanLaplacian, ops.mass, cache, densityMap);
}

// ── Geodesic Distance (Heat Method) ──────────────────────────
//
// Stateful solver wrapping geometry-central's HeatMethodDistanceSolver,
// matching the lifetime model of VectorHeatSolver / SignedHeatSolver.
// The inner solver pre-factors both Cholesky systems (mean-curvature
// flow `M + tA` and Laplacian `A`) at construction; subsequent
// computeDistance() calls only do back-substitution.
//
// Without this caching we redo two factorizations on every call —
// see commit history for the perf regression discovered when the
// pre-cache version was constructing the solver inline. Bench
// numbers in bench/REPORT.md document the warm/cold gap before the
// fix landed.

class HeatGeodesicSolverImpl;
// Which mesh the heat method runs on. Intrinsic (DEFAULT): the manifold's operator mesh, the intrinsic Delaunay triangulation, t = tCoef (mean intrinsic edge)^2 — the
// Studio's HeatMethod (nxr/chart.h). Input: geometry-central's heat method on the input mesh (the former default), kept as the named option.
enum class HeatMesh { Intrinsic, Input };
class HeatGeodesicSolver {
public:
    explicit HeatGeodesicSolver(Manifold& m, double tCoef = 1.0, HeatMesh mesh = HeatMesh::Intrinsic);
    ~HeatGeodesicSolver();
    HeatGeodesicSolver(const HeatGeodesicSolver&) = delete;
    HeatGeodesicSolver& operator=(const HeatGeodesicSolver&) = delete;

    HeatGeodesicSolverImpl& impl();

private:
    std::unique_ptr<HeatGeodesicSolverImpl> impl_;
};

/**
 * Compute geodesic distances from source vertices via the heat method.
 * Reuses the pre-factored solver — first call pays the factor, subsequent
 * calls are back-substitution only.
 *
 * @param solver          stateful heat-geodesic solver
 * @param sourceVertices  list of source vertex indices
 * @return                per-vertex geodesic distances
 */
Eigen::VectorXd heat(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
);

// ── Signed Heat Method (Feng & Crane 2024) ───────────────────
//
// Signed geodesic distance from a curve on the surface. Stateful
// solver, same lifetime model as VectorHeatSolver. The curve is
// expressed as a polyline of vertex indices; `isLoop` controls
// whether the curve closes (last → first).
//
// Useful for region selection, flood fills, and morphological
// operations on the mesh: positive distance one side of the
// curve, negative the other. The level-set constraint keeps the
// zero-set pinned exactly at the curve.

class SignedHeatSolverImpl;
class SignedHeatSolver {
public:
    explicit SignedHeatSolver(Manifold& m, double tCoef = 1.0);
    ~SignedHeatSolver();
    SignedHeatSolver(const SignedHeatSolver&) = delete;
    SignedHeatSolver& operator=(const SignedHeatSolver&) = delete;

    SignedHeatSolverImpl& impl();

private:
    std::unique_ptr<SignedHeatSolverImpl> impl_;
};

enum class SignedHeatLevelSet {
    None = 0,        // unconstrained — curve may not lie exactly on the zero set
    ZeroSet = 1,     // pin the zero level set to the curve (default)
    Multiple = 2,    // pin each curve to its own level set
};

/** Signed geodesic distance from a curve to every vertex.
 *
 *  @param curveVertices  ordered vertex indices defining the curve polyline
 *  @param isLoop         if true, edge from last → first is included
 *  @param levelSet       which level-set constraint to enforce
 *  @return               [nV] signed distances (sign convention follows
 *                        the curve's orientation; flip the input order
 *                        to flip the sign) */
Eigen::VectorXd signedHeat(
    SignedHeatSolver& solver,
    const std::vector<int>& curveVertices,
    bool isLoop = true,
    SignedHeatLevelSet levelSet = SignedHeatLevelSet::ZeroSet
);

} // namespace nxr::manifold::solve

namespace nxr::manifold::query {

// ── Geodesic Paths ───────────────────────────────────────────

/** Geodesic path between two vertices via the flip-out algorithm
 *  (Sharp & Crane, SIGGRAPH Asia 2020). Initialised with a
 *  Dijkstra path between vStart and vEnd, then iteratively
 *  shortened by edge flips until locally straight.
 *
 *  Returns a [N, 3] matrix where each row is a 3D point along
 *  the geodesic. N depends on the path complexity (typically
 *  O(diameter / mean-edge-length) on cortical surfaces).
 *
 *  Differs from `heat` (heat method) which
 *  returns a scalar field on all vertices, not a path. */
Eigen::MatrixXd tracePath(
    Manifold& m,
    int vStart,
    int vEnd
);

/** A geodesic path WITH ITS PLACE ON THE MESH: every point of the
 *  polyline as a weighted combination of at most three input vertices
 *  (a vertex: one weight of 1; an edge crossing: two, (1−t, t); a face
 *  point: its barycentrics), so a consumer can put a delta — or any
 *  vertex quantity — at a point BETWEEN vertices by those weights.
 *  Row i of `vertices`/`weights` is point i of `positions`; unused
 *  slots repeat the first vertex with weight 0. */
struct GeodesicPath {
    Eigen::MatrixXd positions;  // [N, 3] world-space points
    Eigen::MatrixXi vertices;   // [N, 3] input-mesh vertex indices
    Eigen::MatrixXd weights;    // [N, 3] their weights (rows sum to 1)
    double length = 0.0;        // the path's length, mesh units
};

/** `tracePath`, answering where on the mesh each point lies. */
GeodesicPath tracePathSurface(
    Manifold& m,
    int vStart,
    int vEnd
);

/** The EXACT polyhedral geodesic distance between two vertices: the
 *  length of the shortest path over the faces (MMP, geometry-central's
 *  GeodesicAlgorithmExact), mesh units. The flip-out path of
 *  `tracePathSurface` is only locally shortest (shortest in the
 *  homotopy class of its Dijkstra start), so its `length` is >= this;
 *  the heat method (`solve::heat`) approximates it. */
double exactDistance(
    Manifold& m,
    int vStart,
    int vEnd
);

// ── Locus primitives (D11) ───────────────────────────────────
//
// `query::*` returns the geometric locus itself (the "where"); the
// paired `measure::*` (below) returns its scalar metric (the "how
// much"). Three loci: a point, a polyline, a region of faces.

struct PointLocus {
    int vertex;  // vertex index in the owning Manifold
};

struct PolylineLocus {
    Eigen::MatrixXd points;  // [N, 3] world-space points along the polyline
};

struct RegionLocus {
    std::vector<int> faces;  // face indices covered by the region
};

/** Trivial point query — echoes the input vertex after a range check.
 *  Throws Error(InvalidInput) on `v < 0 || v >= m.nV()`. */
PointLocus point(Manifold& m, int v);

/** Polyline between two vertices via the edge-flip geodesic algorithm
 *  (Sharp & Crane 2020). Identical to `tracePath` but typed as a
 *  PolylineLocus so it pairs with `measure::line`. */
PolylineLocus line(Manifold& m, int vStart, int vEnd);

/** Region of faces inside the geodesic ball of radius `level` around
 *  vertex `v`. A face is included iff all three of its vertices have
 *  heat-method geodesic distance `<= level` from `v` — conservative,
 *  slightly underestimates the true region area, but well-defined and
 *  free of barycentric interpolation.
 *
 *  Constructs a fresh `solve::HeatGeodesicSolver` per call (pays one
 *  Cholesky factor). For repeated queries on the same mesh, callers
 *  that want the cached path should compute the heat-distance scalar
 *  field themselves via the cached solver on `Manifold` and then
 *  threshold it manually.
 *
 *  Throws Error(InvalidInput) on `v` out of range or `level <= 0`. */
RegionLocus area(Manifold& m, int v, double level);

} // namespace nxr::manifold::query

namespace nxr::manifold::measure {

// ── Scalar metrics of query loci (D11) ───────────────────────
//
// Paired with `query::*`: each `measure::*` here takes the same
// inputs as the matching `query::*` and returns the scalar value
// (coordinate, length, area). Locus-input overloads avoid re-running
// the query when the caller already has the locus.

/** 3D position of vertex `v`, taken from the manifold's vertex
 *  positions. Throws Error(InvalidInput) on out-of-range `v`. */
Eigen::Vector3d point(Manifold& m, int v);

/** Polyline length along the edge-flip geodesic from `vStart` to `vEnd`.
 *  Equivalent to `line(query::line(m, vStart, vEnd))`. */
double line(Manifold& m, int vStart, int vEnd);

/** Polyline length of a pre-computed `PolylineLocus`. Sum of Euclidean
 *  distances between consecutive points. Returns 0 for empty / single-
 *  point polylines. */
double line(const query::PolylineLocus& locus);

/** Total face area of the geodesic-ball region of radius `level` around
 *  vertex `v`. Equivalent to `area(m, query::area(m, v, level))`. */
double area(Manifold& m, int v, double level);

/** Total face area of a pre-computed `RegionLocus`, summed from
 *  geometry-central's cached face areas. */
double area(Manifold& m, const query::RegionLocus& locus);

} // namespace nxr::manifold::measure

namespace nxr::manifold::transport {

// ── Vector Heat Method (Sharp, Soliman, Crane 2019) ──────────
//
// Stateful solver around a single (Laplacian, vector heat, Poisson)
// factor triple. Owned by the binding shell's per-mesh holder
// (ContextHolder / ContextWrapper) and lazily constructed on first
// use, mirroring CholeskyCache's lifetime model.
//
// The solver is opaque to JS — three free functions below operate
// on a long-lived solver passed by reference. The shells expose
// these as methods on their stateful context wrapper so the
// internal `VectorHeatMethodSolver*` never crosses the boundary.
//
// All tangent-vector inputs/outputs are V × 3 world-space vectors.
// Internally we project to / lift from each vertex's intrinsic
// tangent basis (geometry-central's `vertexTangentBasis`). This
// is a §11 layout extension: V × 2 tangent-frame data is lifted
// to V × 3 at the C++/JS boundary so callers never see Vector2.

class VectorHeatSolverImpl;
class VectorHeatSolver {
public:
    explicit VectorHeatSolver(Manifold& m, double tCoef = 1.0);
    ~VectorHeatSolver();
    VectorHeatSolver(const VectorHeatSolver&) = delete;
    VectorHeatSolver& operator=(const VectorHeatSolver&) = delete;

    // Internal — bindings should not call these directly.
    VectorHeatSolverImpl& impl();
    Manifold& m();

private:
    std::unique_ptr<VectorHeatSolverImpl> impl_;
};

/** Parallel-transport tangent vectors from a sparse set of sources
 *  to every vertex via the vector heat method.
 *
 *  Each source is (vertexIdx, worldVec3). The solver projects
 *  worldVec3 onto the source vertex's tangent basis, runs the
 *  vector heat solve, then lifts each destination vertex's
 *  Vector2 result to a world-space V × 3 vector using its own
 *  vertex tangent basis. Vectors at vertices very near a source
 *  reproduce the source vector almost exactly; vectors far away
 *  are the parallel transport of those sources along geodesics.
 *
 *  @return [nV, 3] row-major world-space tangent vectors. */
Eigen::MatrixXd parallel(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors  // [nSources, 3]
);

/** Extend a sparse scalar field from the given source vertices to
 *  the entire mesh via the scalar heat method. Behaves like a
 *  geodesic Voronoi: each destination vertex takes the value of
 *  its nearest source, smoothly interpolated.
 *
 *  @return [nV] per-vertex scalars. */
Eigen::VectorXd extendScalar(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues   // [nSources]
);

/** Logarithmic map at a source vertex: per-vertex 2D coordinates
 *  (logX, logY) in the *source* vertex's tangent frame.
 *
 *  The norm √(logX² + logY²) is the geodesic distance from source
 *  to that vertex; the angle atan2(logY, logX) is the direction
 *  in source's frame. Equivalent to the exponential-map inverse
 *  flattened around the source, useful for picking, brushing, and
 *  any "distance + bearing from p" UI.
 *
 *  Strategy defaults to AffineLocal (Affine Heat Method, faster
 *  and more accurate near source than the original VectorHeat
 *  approach; the inner solver pre-factors and caches its
 *  connection-Laplacian so repeated calls on different sources
 *  amortize the factorization).
 *
 *  Returns:
 *    - logCoords: [nV, 2] row-major (logX, logY)
 *    - sourceE1, sourceE2: 3D world vectors of the source frame
 *      so consumers can reconstruct world positions if needed
 *      via  `position(v) ≈ source + logX·e1 + logY·e2`. */
struct LogMapResult {
    Eigen::MatrixXd logCoords;   // [nV, 2]
    Eigen::Vector3d sourceE1;    // source vertex's tangent basis e1 in world
    Eigen::Vector3d sourceE2;    // source vertex's tangent basis e2 in world
};

enum class LogMapStrategy {
    VectorHeat = 0,      // original VHM paper (no prefactoring)
    AffineLocal = 1,     // Affine Heat Method, prefactored, default
    AffineAdaptive = 2,  // Affine Heat Method, no prefactor, most accurate
};

LogMapResult logMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy = LogMapStrategy::AffineLocal
);

/** Karcher mean / surface center of a set of source vertices.
 *  Returns the 3D world position of the SurfacePoint that
 *  minimizes ∑ d(p, source_i)^p (default p=2 → Karcher mean). */
Eigen::Vector3d findCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p = 2
);

} // namespace nxr::manifold::transport

namespace nxr::manifold::connection {

// ── Direction Field Design (Trivial Connections) ─────────────

struct DirectionFieldResult {
    Eigen::VectorXd connections;            // φ per edge (1-form angles)
    Eigen::MatrixXd directionVectors;       // [nF, 3] smooth direction field on faces
    Eigen::MatrixXd orthogonalVectors;      // [nF, 3] 90° rotated direction field on faces
    Eigen::MatrixXd vertexVectors;          // [nV, 3] smooth direction field on vertices
    Eigen::MatrixXd vertexOrthogonalVectors;// [nV, 3] 90° rotated direction field on vertices
    double eulerCharacteristic;
    bool gaussBonnetSatisfied;
};

// ── Smooth Direction Fields (NRoSy, Knöppel-Crane) ───────────
//
// Smoothest unit-norm direction field of order n via the
// Knöppel-Crane formulation — minimizes Dirichlet energy on
// the connection Laplacian, no prescribed singularities (they
// emerge automatically). Distinct from `trivial`
// above (which solves trivial connections with user-prescribed
// singularities).
//
// nSym selects the symmetry order:
//    1 → vector field (one direction per face)
//    2 → line field (two opposite directions per face)
//    4 → cross field (four directions, useful for quad meshing
//                     and stripe patterns aligned to two axes)
//
// `alignToCurvature` returns the principal-curvature-aligned
// field instead of pure smoothest (only meaningful for nSym ≥ 2).

/** Result of `smoothVertex`. The face-based variant
 *  returns a plain Eigen::MatrixXd directly, since face stripes
 *  aren't a downstream consumer of its raw form. */
struct SmoothVertexFieldResult {
    Eigen::MatrixXd vertexVectors;   // [nV, 3] world-space, per vertex
    Eigen::VectorXd vertexFieldRaw;  // [nV * 2] raw Vector2 in vertex tangent basis
    int nSym;
};

/**
 * Compute the trivial connection 1-form φ = δβ (per-edge rotation angles).
 *
 * Solves the Poisson system A β = -K + 2π σ where:
 *   A   = d0ᵀ ★₁ d0 (cotangent Laplacian, via CholeskyCache::hodgeExact)
 *   K   = per-vertex Gaussian curvature (angle defect)
 *   σ   = singularity index at each vertex (0 for non-singular)
 *
 * Returns φ = ★₁ d₀ β as an Eigen::VectorXd of length nEdges.
 * Sign convention: φ(e) is positive in the direction of e.halfedge().
 *
 * Gauss-Bonnet requirement: Σ singularityMap values must equal χ(mesh).
 * This is NOT checked here — the caller is responsible (directionField
 * checks it; assembleTrivialConnectionLaplacian documents it).
 *
 * For sphere topology (genus 0, χ=2): two vertices with σ=1.0 each.
 * For n-RoSy fields: use σ = (desired holonomy) / (2π).
 *   e.g. nSym=4 cross field on sphere: two vertices with σ=0.5.
 */
Eigen::VectorXd computeTrivialConnection(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

// ── Trivial-gauge rotations (vertex + face) ──────────────────
//
// Integrates the trivial-connection 1-form φ (from computeTrivialConnection)
// into unit-complex gauge rotations relative to the Levi-Civita frames:
//   vertex — BFS over the primal vertex graph (transportVectorsAlongHalfedge)
//   face   — BFS over the dual face graph (same transport + sign conventions
//            as propagateAngles in direction_field.cpp)
//
// Convention (the COMBING multiplier): the stored rotation is the gauge
// transform that realizes the combed (trivially-parallel) frame by complex
// multiplication —
//   combed vertex frame = vertex(v) .* vertexGridEmbedded.row(v)   (geometry-central's frame, the light Geometry bundle's `vertexGrid`)
//   combed face frame   = face(f)   .* faceGrid.row(f)
// Equivalently real(vertex .* vertexGridEmbedded) IS the trivial-connection parallel
// field (DirectionFieldResult::vertexVectors), and likewise on faces
// (directionVectors). Note this is the CONJUGATE of the accumulated transport
// coefficient e^{iα}: actively rotating a frame by α multiplies the complex
// grid c = e1 + i·e2 by e^{-iα}. Verified numerically against the trivial
// direction field (test_geometry_bundle.cpp).
//
// Anchors: vertex 0 and face 0 carry the identity rotation. The two combs are
// each globally parallel, but their relative phase (vertex comb vs face comb)
// is anchor-dependent and carries no meaning.
//
// Gauss-Bonnet: Σ singularityMap values must equal χ(mesh) (the caller's
// responsibility — not checked here).
//
// Connectivity contract: BFS starts at vertex 0 / face 0 and visits only that
// connected component; on a disconnected mesh other components receive
// 0+0i. Intended usage is one connected component (e.g. a single
// cortical hemisphere) at a time.
struct GaugeRotations {
    Eigen::VectorXcd vertex;   // [nV] combing rotation, |r_v| = 1
    Eigen::VectorXcd face;     // [nF] combing rotation, |r_f| = 1
};
GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap);

/**
 * Compute a smooth direction field with prescribed singularities.
 * Uses the trivial connections algorithm (Crane, de Goes, Desbrun).
 *
 * Input: singularityMap — vertex index → singularity index (typically ±1 or ±0.5)
 * The sum of indices MUST equal the mesh Euler characteristic (Gauss-Bonnet).
 *
 * Output: φ = δβ + γ (per-edge 1-form angles), plus the resulting
 * direction and orthogonal vector fields on faces.
 */
DirectionFieldResult trivial(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

/** Compute the smoothest face-based direction field.
 *  Returns [nF, 3] world-space vectors (principal nSym-RoSy
 *  representative; the other nSym-1 directions are obtained by
 *  rotating in the face tangent plane). */
Eigen::MatrixXd smoothFace(
    Manifold& m,
    int nSym = 4,
    bool alignToCurvature = false
);

/** Compute the smoothest vertex-based direction field (needed
 *  as input to stripe patterns).
 *  Returns the raw [nV * 2] Vector2 form alongside [nV, 3] world
 *  vectors, since stripes consume the Vector2 form internally. */
SmoothVertexFieldResult smoothVertex(
    Manifold& m,
    int nSym = 2,
    bool alignToCurvature = false
);

} // namespace nxr::manifold::connection

namespace nxr::manifold::solve {

// ── Hodge Decomposition ──────────────────────────────────────

struct HodgeResult {
    // Scalar potentials (per vertex)
    Eigen::VectorXd exactPotential;      // α
    Eigen::VectorXd coExactPotentialF;   // β (per face)
    Eigen::VectorXd coExactPotentialV;   // β averaged to vertices
    Eigen::VectorXd combinedPotential;   // α + β
    Eigen::VectorXd omega;               // input 1-form on edges

    // 1-form components of the decomposition (all [nE])
    Eigen::VectorXd dAlpha;              // exact 1-form: d(α)
    Eigen::VectorXd deltaBeta;           // co-exact 1-form: δ(β)
    Eigen::VectorXd gamma;               // harmonic 1-form: ω - dα - δβ

    // Face-centered vector fields via Whitney interpolation, each [nF, 3]
    Eigen::MatrixXd omegaVectors;
    Eigen::MatrixXd dAlphaVectors;
    Eigen::MatrixXd deltaBetaVectors;
    Eigen::MatrixXd gammaVectors;
};

/**
 * Hodge/Helmholtz decomposition of a 1-form ω:
 *   ω = dα + δβ + γ
 *
 * Given random ω, returns:
 *   - α: scalar potential (vertex-based) — gradient part
 *   - β: scalar potential (face-based, averaged to vertices) — curl part
 *
 * @param dec    DEC operators (d0, d1, hodge1, hodge1Inverse)
 * @param omega  input 1-form on edges (random or user-supplied)
 */
HodgeResult hodge(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const Eigen::VectorXd& omega
);

} // namespace nxr::manifold::solve

namespace nxr::manifold::parametrization {

// ── Surface Parametrization ──────────────────────────────────

/** Boundary First Flattening (Sawhney & Crane 2017) — produces a
 *  conformal-by-default planar parametrization of the surface.
 *
 *  Returns a [V, 2] matrix of UV coordinates (u, v) per vertex.
 *  Mesh must have at least one boundary loop; for closed meshes
 *  the user must cut the surface first. The mesh in nxr-compute is
 *  ManifoldSurfaceMesh so boundary detection works automatically.
 *
 *  Used by texture-based visualizations (LIC, isoline rendering
 *  in UV space, anisotropic shaders). Lower angular distortion
 *  than LSCM, especially near complex boundaries. Uses
 *  geometry-central's bundled BFF implementation — no extra
 *  dependency.
 *
 *  Throws if the mesh has no boundary. */
Eigen::MatrixXd bff(Manifold& m);

} // namespace nxr::manifold::parametrization

namespace nxr::manifold::parametrization::stripes {

// ── Stripe Patterns (Knöppel-Crane SIGGRAPH 2015) ────────────
//
// Procedural sinusoidal stripes aligned to a 2-RoSy direction
// field. Output is a list of 3D polyline segments (the zero
// level set of the per-corner phase function), suitable for
// drawing as three.js LineSegments.
//
// Frequencies control stripe spacing in oscillations per unit
// edge length. A constant uniformFrequency is the common case;
// pass per-vertex frequencies if you want curvature-modulated
// striping.

struct StripePatternResult {
    Eigen::MatrixXd positions;   // [2 * segmentCount, 3] endpoint pairs
    int segmentCount;
};

/** Stripe pattern with a uniform target frequency. The vertex
 *  field must be 2-RoSy (use `smoothVertex(m, 2)`
 *  or pass an existing line field). */
StripePatternResult compute(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2] in vertex tangent basis
    double uniformFrequency,
    bool connectOnSingularities = true
);

/** Stripe pattern with per-vertex target frequencies. */
StripePatternResult computeFreq(
    Manifold& m,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2]
    const Eigen::VectorXd& frequencies,     // [nV]
    bool connectOnSingularities = true
);

} // namespace nxr::manifold::parametrization::stripes

namespace nxr::manifold::geometry {

// ── Curvatures ───────────────────────────────────────────────

struct CurvatureResult {
    Eigen::VectorXd gaussian;        // K per vertex (angle defect)
    Eigen::VectorXd mean;            // H per vertex
    Eigen::VectorXd kMin;            // κ_min per vertex
    Eigen::VectorXd kMax;            // κ_max per vertex
    Eigen::MatrixXd principalDirMax; // [nV, 3] max principal direction lifted to 3D
};

CurvatureResult curvatures(Manifold& m);

// ── Normal Estimators ────────────────────────────────────────

enum class NormalType {
    AngleWeighted,    // default (matches gpjs "Tip Angle Weighted")
    AreaWeighted,     // weighted by face area
    EqualWeighted,    // equal weighting (normalized sum)
    SphereInscribed,  // (u × v) / (|u|² |v|²) per corner
    MeanCurvature,    // cotangent-weighted (Laplace-Beltrami)
    GaussCurvature    // dihedral angle / edge length weighted
};

/** Compute per-vertex normals using a given estimator.
 *  Returns [nV * 3] flat array (row-major). */
Eigen::MatrixXd normals(Manifold& m, NormalType type);

// ── Face Frames ──────────────────────────────────────────────

/** Per-face orthonormal tangent frame.
 *
 *  Each face f has an in-plane orthonormal basis (e1, e2) and a
 *  normal n, all 3D vectors in world coordinates. Used by:
 *    - GPU-side particle advection (frames define each face's
 *      local 2D coordinate system)
 *    - Tangent vector field visualization (face vectors expressed
 *      as 2D coords in (e1, e2) reconstruct in JS / shaders)
 *    - Texture mapping where a per-face frame is needed
 *
 *  Convention matches direction_field.cpp::faceOrthonormalBasis:
 *  e1 is the unit edge vector along f.halfedge(), e2 = n × e1.
 *  This is the same frame the trivial-connection direction field
 *  uses, so tangent vectors transported between faces are
 *  interpretable in the same coordinate system. */
struct FaceFrames {
    Eigen::MatrixXd e1;       // [nF, 3] — first tangent vector
    Eigen::MatrixXd e2;       // [nF, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nF, 3] — face normals
};

FaceFrames frames(Manifold& m);

/* Per-vertex orthonormal tangent frame = geometry-central's gauge of the (mollified) input mesh (IntrinsicSurface::tangentBasis; equal to its
 * `vertexTangentBasis` when nothing was mollified): the 3D realization of the per-vertex tangent space in which the vertex connection-Laplacian's complex
 * eigenvector coordinates, the vector heat method's and the transport live:
 *   n  = the corner-angle-weighted vertex normal
 *   e1 = the average of the vertex's outgoing edges, each projected on the tangent plane and rotated back by its own rescaled angle (angle 0 = the input's
 *        v.halfedge())
 *   e2 = n x e1
 * A connection-mode coordinate z=(a,b) at vertex i decodes to a*e1_i + b*e2_i. */
struct VertexFrames {
    Eigen::MatrixXd e1;       // [nV, 3] — first tangent
    Eigen::MatrixXd e2;       // [nV, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nV, 3] — vertex normals (area-weighted)
};

VertexFrames vertexFrames(Manifold& m);
// DEPRECATED alias of vertexFrames (there is one gauge now; removed with the port, spec G7).
VertexFrames vertexFramesEmbedded(Manifold& m);

// ── Complex frame ("grid") ───────────────────────────────────
//
// The per-element tangent frame packed as a complex 3-vector
// c = e1 + i·e2 ∈ ℂ³ (real part = e1, imag part = e2). The unit
// normal is recovered as real(c) × imag(c) and is not stored
// separately. This is the representation Brainstorm's Geometry.*.grid
// fields use; see the bundle design spec §5.1.
Eigen::MatrixXcd vertexGrid(Manifold& m);   // [nV, 3] complex, from vertexFrames (geometry-central's gauge: what the Levi-Civita connection Laplacian's coordinates decode against)
Eigen::MatrixXcd vertexGridEmbedded(Manifold& m);   // [nV, 3] complex, from vertexFramesEmbedded: geometry-central's frame, the base frame of the trivial gauge and of the light Geometry bundle's `vertexGrid`
Eigen::MatrixXcd faceGrid(Manifold& m);     // [nF, 3] complex

// ── Curvature as a 2-RoSy complex + mean ─────────────────────
//
// The shape operator (3 real DOF) split into its trace and deviatoric
// parts. `deviatoric` is the traceless symmetric part encoded as a
// 2-RoSy complex number q: arg(q)/2 is the max principal direction
// angle in the vertex tangent basis, |q| = (κmax − κmin)/2. `mean`
// is (κmax + κmin)/2. See bundle design spec §5.2. Everything else
// (principal curvatures/directions, extrinsic Gaussian) derives from
// these; intrinsic Gaussian comes from the angle-defect (2π − angleSum).
// If geometry-central principal directions are unavailable at a vertex, the
// deviatoric phase falls back to 0; only |q| (anisotropy) and mean (H) are
// reliable there.
struct VertexCurvature2RoSy {
    Eigen::VectorXcd deviatoric;  // [nV] q
    Eigen::VectorXd  mean;        // [nV] H
};
VertexCurvature2RoSy vertexCurvature(Manifold& m);

// ── Halfedge mesh topology ────────────────────────────────────
//
// Flat struct-of-arrays representation of the halfedge combinatorics,
// using 0-based indices matching geometry-central's internal storage.
// Bindings that want 1-based MATLAB indices convert at the binding edge
// (see indexVectorToMx1Based in marshal.h).
// A value of -1 (INVALID) means "none / boundary" for .face and .corner
// on exterior halfedges; all other entries are non-negative for a valid
// closed manifold mesh.
//
//   nH == 2 * nE   (every edge has exactly 2 halfedges)
//   nC == 3 * nF   (every interior face has 3 corners)

struct MeshTopology {
    // Element counts.
    int nV, nE, nF, nH, nC;

    // Reverse lookups: one representative halfedge per element (0-based).
    std::vector<long> vertexHalfedge;   // [nV]  he that "belongs to" vertex
    std::vector<long> edgeHalfedge;     // [nE]
    std::vector<long> faceHalfedge;     // [nF]
    std::vector<long> cornerHalfedge;   // [nC]

    // Halfedge adjacency table (0-based; -1 = none / exterior).
    std::vector<long> heTwin;      // [nH]
    std::vector<long> heNext;      // [nH]
    std::vector<long> heVertex;    // [nH]  tail vertex
    std::vector<long> heEdge;      // [nH]
    std::vector<long> heFace;      // [nH]  -1 for exterior halfedges
    std::vector<long> heCorner;    // [nH]  -1 for exterior halfedges

    // Boolean attributes (stored as char: 1 = true, 0 = false).
    std::vector<char> heOrientation;  // [nH]  true iff he == he.edge().halfedge()
    std::vector<char> heIsInterior;   // [nH]
};

/** Extract the halfedge mesh topology into a plain struct-of-arrays.
 *  All indices are 0-based (geometry-central convention). -1 encodes
 *  geometry-central's INVALID_IND for "none / boundary" slots. */
MeshTopology meshTopology(Manifold& m);

// ── Light per-element geometry bundle ────────────────────────
//
// All light O(V/E/F/H/C) per-element quantities, element-grouped.
// Heavy sparse operators are intentionally excluded (on-demand
// surface, separate). Frames are the complex `grid` (c = e1+i·e2);
// normals derive as real×imag. Curvature is the 2-RoSy deviatoric
// + mean (see vertexCurvature). See the bundle design spec §4.2.
//
// Corner-array sizing note: cornerAngles / cornerScaledAngles are
// sized to nH (not nC) for safety on boundary meshes, where
// geometry-central corner ids are halfedge-based and may be sparse.
// On a closed mesh nH == nC so the size is identical.
struct MeshGeometry {
    double totalArea = 0.0;
    // vertex
    Eigen::VectorXd  vertexDualAreas;           // [nV]
    Eigen::VectorXd  vertexAngleSums;           // [nV]
    Eigen::VectorXcd vertexCurvatureDeviatoric; // [nV] q (2-RoSy)
    Eigen::VectorXd  vertexMeanCurvature;       // [nV] H
    Eigen::MatrixXcd vertexGrid;                // [nV,3] c = e1+i·e2
    // edge
    Eigen::VectorXd  edgeLengths;              // [nE]
    Eigen::VectorXd  edgeCotanWeights;         // [nE]
    Eigen::VectorXd  edgeDihedralAngles;       // [nE]
    // face
    Eigen::VectorXd  faceAreas;                // [nF]
    Eigen::MatrixXd  faceCentroids;            // [nF,3]
    Eigen::MatrixXcd faceGrid;                 // [nF,3]
    // halfedge
    Eigen::VectorXd  halfedgeCotanWeights;     // [nH]
    Eigen::VectorXcd halfedgeVectorsInVertex;  // [nH]
    Eigen::VectorXcd halfedgeVectorsInFace;    // [nH]
    Eigen::VectorXcd halfedgeTransportAlong;   // [nH]
    Eigen::VectorXcd halfedgeTransportAcross;  // [nH]
    // corner (sized nH; on closed mesh nH==nC)
    Eigen::VectorXd  cornerAngles;             // [nH]
    Eigen::VectorXd  cornerScaledAngles;       // [nH]
};

MeshGeometry meshGeometry(Manifold& m);

} // namespace nxr::manifold::geometry

// ── Covariant differential operators on 3-vector cortical fields ──────────────
//
// Artifact-free transport + differentiation of a 3-vector field (e.g. a leadfield)
// expressed in per-vertex cortical frames Fv = [e1 | e2 | n] (columns), sourced from
// geometry::vertexFrames. The connection is the flat full-frame transport
// P_ij = Fj^T Fi — it accounts for the tangent rotation AND the normal tilt, and is
// path-independent (zero holonomy), which is exactly what removes the frame-rotation
// artifact (e.g. Cartesian-parallel leadfields on opposite sulcal walls reading as
// antiparallel in local frames). See the design doc.
namespace nxr::manifold::differential {

// 3x3 transport of a frame-local vector from vertex i's frame to vertex j's frame,
// P_ij = Fj^T Fi. Flat => exact for ANY pair (i, j) (adjacent or not). 0-based indices.
Eigen::Matrix3d frameTransport(Manifold& m, int i, int j);

// Per-vertex frames Fv stacked as [nV, 9], row v = Fv flattened ROW-major
// (Fv(0,0),Fv(0,1),Fv(0,2), Fv(1,0),...). Columns of Fv are [e1 | e2 | n].
Eigen::MatrixXd vertexFrameMatrices(Manifold& m);

// Lift a frame-local field to world (Cartesian): world[v] = Fv * Lloc[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToWorld(Manifold& m, const Eigen::MatrixXd& Lloc);

// Express a world (Cartesian) field in frames: Lloc[v] = Fv^T * Lworld[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToFrame(Manifold& m, const Eigen::MatrixXd& Lworld);

// Covariant gradient operator G : 3N -> 3E (component-major [a;b;c] blocks). For each
// oriented edge e: i->j, the covariant difference is delta_e = L_j - P_ij L_i. G has a +I3
// block at vertex j and a -P_ij block at vertex i. A Cartesian-constant field has G*L = 0.
Eigen::SparseMatrix<double> covariantGradient(Manifold& m);

// Extrinsic vector Weitzenböck Δ₃ + W_extrinsic on the ambient (ℝ³) bundle, a
// [3V×3V] real symmetric sparse matrix in WORLD-frame component-major layout
// (row/col index = c·N + v, c = world x/y/z). It is, by construction, the
// imaginary (vector) block of the immersion SQUARED Dirac
//   S = 2·dirac(0.5) = dirac(0) + dirac(1) = Δ₄ + D_N   (the [4V×4V] real
// vertex-interleaved 4v+c operator), extracting the quaternion components
// c ∈ {1,2,3} and dropping the scalar (c=0) border. This is the principled,
// geometrically-meaningful construction: the imaginary quaternion axes (x,y,z)
// ARE the ambient world axes the Dirac normals (Gauss map) are expressed in, so
// no per-vertex basis change is needed. Its "flat part" (the component-diagonal
// piece coming from dirac(0)=cotanL⊗I₄) is exactly kron(I₃, cotanLaplacian); the
// extra coupling is the extrinsic curvature endomorphism W_ext (the imag block
// of D_N), which vanishes on a developable/flat region (constant normals) and is
// nonzero on curved regions. Like the first-order Dirac, S annihilates GLOBALLY-
// constant world fields (D kills constant quaternions), so W_ext is detected as a
// nonzero MATRIX difference from the flat anchor, not as a constant-field response.
Eigen::SparseMatrix<double> assembleExtrinsicWeitzenbock(Manifold& m);

} // namespace nxr::manifold::differential

namespace nxr::field::generate {

using nxr::manifold::Manifold;
using nxr::manifold::solve::EigenResult;
using nxr::manifold::ops::ManifoldOperators;
using nxr::manifold::ops::DECOperators;

// ── Field Generators ─────────────────────────────────────────
//
// Analytic field synthesis. Mirrors the +bct.+field.+generate.*
// family from the MATLAB reference toolbox; the cortical-flow
// solvers (Hodge, Poisson, gradient flow) are source-agnostic and
// consume fields produced here interchangeably with real activity
// data loaded from Zarr.
//
// The validation pattern (see test_field_generators.cpp) is the
// geometry-processing-js demo pattern: synthesize ω from known
// α / β / γ contributions, run hodge, verify the
// algorithm recovers what was put in.

/** Sparse delta vector: zeros everywhere except the listed source
 *  vertices, which take the supplied values. Useful as an initial
 *  condition for heat-method / Poisson workflows. */
Eigen::VectorXd delta(int nV, const std::map<int, double>& sources);

/** Random vertex scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the α (gradient potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd randomVertexScalar(int nV, unsigned int seed = 42);

/** Random face scalar field, uniform in [-1, 1]. Reproducible by
 *  seed. Used as the β (curl potential) in decomposed-1-form
 *  synthesis. */
Eigen::VectorXd randomFaceScalar(int nF, unsigned int seed = 42);

/** Generate a random 1-form on edges. */
Eigen::VectorXd randomOmega(int nE, unsigned int seed = 42);

/** Extract the k-th eigenmode from a precomputed EigenResult as a
 *  vertex scalar field. */
Eigen::VectorXd eigenmodeField(const EigenResult& eig, int k_index);

/** Synthesize a 1-form on edges by combining independently-random
 *  exact and co-exact contributions:
 *
 *      ω = α_strength · (d0 · α_rand) + β_strength · (★₁⁻¹ d1ᵀ · β_rand)
 *
 *  where α_rand is a random vertex scalar (seed) and β_rand is a
 *  random face scalar (seed + 1). The Hodge decomposition of ω
 *  must recover the same dα and δβ within solver tolerance — this
 *  is the validation contract for hodge.
 *
 *  γ_strength is reserved for the harmonic component but ignored in
 *  v1; harmonic basis computation is deferred (cortical meshes are
 *  genus 3-5 due to ventricle topology, so the harmonic space is
 *  non-trivial in principle, but its construction needs its own
 *  primitive and isn't required for the gpjs-style validation). */
Eigen::VectorXd randomDecomposed1Form(
    const DECOperators& dec,
    int nV, int nE, int nF,
    double alphaStrength,
    double betaStrength,
    double gammaStrength,
    unsigned int seed = 42
);

// ── Time-Varying Field Generators ────────────────────────────
//
// Spectral-evolution generators that produce [T, V] float32 arrays
// suitable for the Timeline / activity slot. Both reuse precomputed
// eigenmodes — no per-frame Cholesky needed, which is the spectral
// method's defining feature: O(K) per timestep regardless of nV.
//
// Storage layout: Eigen::MatrixXf with shape (T, V), column-major
// (Eigen default). Frame ti is `output.row(ti)`. When flattened
// row-major to a Float32Array, this matches the [T, V] convention
// used by the Zarr `recordings/.../activity` schema.
//
// Both functions assume the eigenmodes are M-orthonormalized — i.e.
// the caller has already run normalize() on solve
// output. Failure to do so silently produces wrong amplitudes.

/** Heat diffusion via spectral evolution.
 *
 *  Solves  ∂u/∂t = -α K u  on the domain, returning the full time
 *  series. M-weighted spatial mean is preserved exactly across all
 *  frames (heat equation invariant); only the zero-mean part evolves
 *  spectrally:
 *
 *      u(t) = u_mean + Σₖ ⟨u₀ - u_mean, φₖ⟩_M · exp(-α·t·λₖ) · φₖ
 *
 *  The DC mode (if present in `eig`) is harmless — its λ ≈ 0 means
 *  exp(-αtλ) ≈ 1 and it just reproduces the mean offset.
 *
 *  Agnostic in M: works on lumped mass (mesh) or node-weight
 *  diagonals (graph). M-orthonormality of `eig.eigenvectors`
 *  is the caller's responsibility regardless.
 *
 *  @param M          mass / inner-product matrix used for projection
 *  @param eig        precomputed eigenmodes (M-orthonormal, sorted ascending)
 *  @param u0         initial condition [n]
 *  @param timesteps  T values of t (any non-negative reals)
 *  @param alpha      diffusion coefficient
 *  @return           [T, n] float32 time series, frame-major
 */
Eigen::MatrixXf heatDiffusion(
    const Eigen::SparseMatrix<double>& M,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
);

/** Convenience overload: forwards to the (M, …) form using ops.mass. */
inline Eigen::MatrixXf heatDiffusion(
    const ManifoldOperators& ops,
    const EigenResult& eig,
    const Eigen::VectorXd& u0,
    const std::vector<double>& timesteps,
    double alpha = 1.0
) {
    return heatDiffusion(ops.mass, eig, u0, timesteps, alpha);
}

/** Damped wave field via per-mode oscillation.
 *
 *  Synthesizes a superposition of independently-damped harmonic
 *  oscillators on the mesh:
 *
 *      u(t,v) = Σₘ aₘ · exp(-γₘ·t) · cos(√λ_{kₘ}·t + φₘ) · φ_{kₘ}(v)
 *
 *  The angular frequency of each mode is √λ_{kₘ} (acoustic relation
 *  between Laplace eigenvalues and wave speeds). Useful for
 *  illustrative time-varying fields and for studying mode-mixing
 *  visualizations.
 *
 *  @param eig          precomputed eigenmodes (M-orthonormal)
 *  @param modeIndices  which eigenmodes to excite (size M)
 *  @param amplitudes   per-mode amplitude aₘ (size M)
 *  @param dampings     per-mode damping rate γₘ ≥ 0 (size M)
 *  @param phases       per-mode phase φₘ in radians (size M)
 *  @param timesteps    T values of t
 *  @return             [T, nV] float32 time series, frame-major
 */
Eigen::MatrixXf dampedWave(
    const EigenResult& eig,
    const std::vector<int>& modeIndices,
    const std::vector<double>& amplitudes,
    const std::vector<double>& dampings,
    const std::vector<double>& phases,
    const std::vector<double>& timesteps
);

} // namespace nxr::field::generate

namespace nxr::field::interp {

using nxr::manifold::Manifold;
using nxr::manifold::ops::DECOperators;

// ── Vector Field Operations ──────────────────────────────────

/** Whitney interpolation: edge 1-form → face-centered 3D vectors.
 *  Formula: V(f) = (N × (a + b + c)) / (6A)
 *  Returns [nF, 3] matrix (row-major vectors). */
Eigen::MatrixXd whitney(
    Manifold& m,
    const DECOperators& dec,
    const Eigen::VectorXd& oneForm
);

} // namespace nxr::field::interp

namespace nxr::field::op {

using nxr::manifold::Manifold;

/** Gradient of a vertex scalar field → face-centered 3D vectors.
 *  For triangle (vi, vj, vk): ∇u = (1 / 2A) Σ (uₖ - uᵢ) (N × eⱼᵢ)
 *  Returns [nF, 3] matrix. */
Eigen::MatrixXd gradient(
    Manifold& m,
    const Eigen::VectorXd& scalarField
);

} // namespace nxr::field::op

namespace nxr::field::extract {

using nxr::manifold::Manifold;

// ── Isolines ─────────────────────────────────────────────────

struct IsolineResult {
    Eigen::MatrixXd positions;  // [segmentCount * 2, 3] endpoints of line segments
    int segmentCount;
};

/** Extract isoline (contours) from a per-vertex scalar field.
 *  numLevels evenly-spaced contour values in [minValue, maxValue].
 *  If minValue == maxValue, auto-detects range from the data. */
IsolineResult isoline(
    Manifold& m,
    const Eigen::VectorXd& scalarField,
    int numLevels = 20,
    double minValue = 0.0,
    double maxValue = 0.0
);

// ── Streamlines ──────────────────────────────────────────────

struct StreamlineResult {
    // Flat [2 * N, 3] array of line segment endpoints (pair-wise)
    Eigen::MatrixXd positions;
    int segmentCount;
};

/**
 * Trace streamlines through a face vector field using forward Euler
 * with face-crossing detection.
 *
 * @param faceField  [nF, 3] vector field on face centroids
 * @param numSeeds   target number of seed streamlines (quasi-uniform)
 * @param stepCoef   step size = stepCoef * meanEdgeLength
 * @param maxSteps   max integration steps per streamline
 */
StreamlineResult streamline(
    Manifold& m,
    const Eigen::MatrixXd& faceField,
    int numSeeds = 15,
    double stepCoef = 0.15,
    int maxSteps = 1000
);

} // namespace nxr::field::extract

// ── Backward-compat using-declarations ───────────────────────
//
// Existing callers (addon.cpp, MEX bindings, tests, downstream
// consumers) reference these symbols at `nxr::manifold::*`. The
// sub-namespace split above is a pure organization change; the
// using-declarations below preserve every public name at its
// pre-split location.

namespace nxr::manifold {
    // Type using-declarations only — function using-decls conflict with
    // out-of-line definitions in src/*.cpp (the .cpp opens a sub-namespace
    // block like `namespace nxr::manifold::connection { trivial(...) { … } }`
    // and the using-decl `using connection::trivial` is treated as
    // declaring `nxr::manifold::trivial`, which clashes with the
    // canonical sub-namespace definition). Types don't have this issue.
    //
    // Consumers must use the sub-namespace paths for free functions
    // (e.g. `nxr::manifold::ops::d0(m)`, `nxr::manifold::solve::eigen(...)`).
    using ops::MassMatrixVariant;
    using ops::ManifoldOperators;
    using ops::DECOperators;
    using ops::CholeskyCache;
    using ops::laplacian::connection::ConnectionDomain;
    using ops::laplacian::connection::ConnectionLaplacianFormat;
    using ops::laplacian::connection::ConnectionLaplacianOptions;
    using ops::laplacian::connection::ConnectionLaplacian;
    using solve::EigenResult;
    using solve::HeatGeodesicSolver;
    using solve::SignedHeatSolver;
    using solve::SignedHeatLevelSet;
    using solve::HodgeResult;
    using transport::VectorHeatSolver;
    using transport::LogMapResult;
    using transport::LogMapStrategy;
    using connection::DirectionFieldResult;
    using connection::SmoothVertexFieldResult;
    using parametrization::stripes::StripePatternResult;
    using geometry::CurvatureResult;
    using geometry::NormalType;
    using geometry::FaceFrames;
    using geometry::VertexFrames;
    using geometry::MeshTopology;
} // namespace nxr::manifold

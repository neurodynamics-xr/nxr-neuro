#pragma once
// Manifold, the mesh handle, and the light halfedge bundles (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <cstdint>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/core/errors.h"
#include "nxr/core/cancellation.h"
#include "nxr/core/progress.h"

namespace geometrycentral { namespace surface {
class ManifoldSurfaceMesh;
class VertexPositionGeometry;
class IntrinsicGeometryInterface;
} }

namespace nxr::geometry {

// Cross-cutting types live in nxr::core; bring them into manifold scope
// so unqualified uses below (Error, CancellationToken, ProgressObserver)
// resolve. See <nxr/core/errors.h>, <nxr/core/cancellation.h>, <nxr/core/progress.h>.
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;

// Forward declarations for facets and geometry bundles.
namespace facet {
    class TopologyFacet;  class EmbeddedFacet;  class IntrinsicFacet;
    class ExtrinsicFacet; class ConnectionFacet;     class OperatorsFacet;
}
} // namespace nxr::geometry
namespace nxr::operators { struct DECOperators; struct DecMatrices; class CholeskyCache; }
namespace nxr::operators::laplacian::connection { enum class CovariantCoupling; }
namespace nxr::geometry {
struct MeshGeometry; struct MeshTopology;

enum class ConnectionType { Euclidean, LeviCivita, Trivial };   // the choice of connection (G7: formerly GaugeType; "gauge" names only the tangent-space convention)
using GaugeType [[deprecated("nxr::geometry::ConnectionType")]] = ConnectionType;   // one release

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
//   Intrinsic (DEFAULT): geometry-central's intrinsic mollified Delaunay (signpost) triangulation (nxr/geometry/intrinsic_surface.h), with the mixed-Voronoi mass of the mollified input.
//   Embedded: the input triangulation as it is (no mollification, no flips) — a NAMED OPTION, never a silent default.
enum class MeshKind { Intrinsic, Embedded };

struct ManifoldOptions {
    MeshKind                    mesh     = MeshKind::Intrinsic;
    DelaunayOptions             delaunay = {};     // `mollify` and `flipEps`: named, recorded options (shared names with the Studio and nxr)
};

class Manifold {
public:
    // The operator mesh is the intrinsic mollified Delaunay mesh unless `options.mesh` is Embedded.
    Manifold(const double* vertices, int nV, const int32_t* faces, int nF, const ManifoldOptions& options);
    // The flag form: `intrinsicDelaunay` true (the DEFAULT) = MeshKind::Intrinsic, false = MeshKind::Embedded.
    Manifold(const double* vertices, int nV,
                   const int32_t* faces, int nF,
                   bool intrinsicDelaunay = true,
                   const DelaunayOptions& delaunay = {});
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

    // THE intrinsic surface (nxr/geometry/intrinsic_surface.h, geometry-central's signpost intrinsic Delaunay triangulation of the mollified input): every
    // intrinsic operator (cotan Laplacian, masses, DEC, connection Laplacian, Hodge–Dirac) and every tangent quantity is built on it, in ONE gauge —
    // geometry-central's. With MeshKind::Embedded it is the input as it is. geometry() is unchanged and still used for EXTRINSIC quantities.
    // (`intrinsic()` below is the older data facet; it keeps its name.)
    IntrinsicSurface& intrinsicSurface();
    const ManifoldOptions& options() const;
    // The cotangent Laplacian of the operator mesh, held for the Manifold's lifetime (operators::ManifoldOperators views it).
    const Eigen::SparseMatrix<double>& operatorCotan();
    // The operator mesh's DEC matrices (d0, d1, ⋆0 = the mixed-Voronoi mass, raw ⋆1, ⋆2, ⋆1⁻¹), and A/3 per vertex on it; lazily built, held for the lifetime.
    const operators::DecMatrices& operatorDecMatrices();
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
             const DelaunayOptions& delaunayOptions = {});

    int eulerCharacteristic() const;                 // nV - nE + nF
    ConnectionType activeConnectionType() const;
    const std::map<int,double>& activeSingularities() const;
    // Re-point the active/default gauge. Trivial validates Gauss-Bonnet.
    void setConnection(ConnectionType type, const std::map<int,double>& singularities = {});
    [[deprecated("setConnection")]] void setGauge(ConnectionType type, const std::map<int,double>& singularities = {}) { setConnection(type, singularities); }

    // Declared only; bodies implemented in B2 (require ConnectionFacet complete type).
    facet::ConnectionFacet connection();
    [[deprecated("connection()")]] facet::ConnectionFacet gauge();   // one release
    facet::ConnectionFacet connection(ConnectionType type,
                            const std::map<int,double>& singularities = {});

    // Lazy operator-assembly helpers shared by gauge + operators facets.
    // The DEC of the OPERATOR mesh (follows the mesh option: d0ᵀ⋆1d0 == the cotan Laplacian, ⋆0 == the mixed-Voronoi mass), cached.
    operators::DECOperators& decOperators();
    // The DEC of the INPUT mesh (geometry-central's, embedded) — edges and faces are the input's: for what is indexed by them (a 1-form given on the
    // input's edges, Whitney interpolation on its faces, the trivial connection, the face Dirac), cached.
    operators::DECOperators& decEmbedded();
    operators::CholeskyCache& choleskyCache();   // owned cache, lazy-initialised

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
    // covariant: delegates to connection() + connection().grid() + cotan() then assembles.
    const Eigen::SparseMatrix<std::complex<double>>& connectionLaplacianCached_();
    const Eigen::SparseMatrix<double>& covariantLaplacianCached_(
        operators::laplacian::connection::CovariantCoupling coupling);

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
    // First-order Dirac D (4F×4V), cached (OperatorId::DiracD). nxr::operators::dirac::matrix.
    const Eigen::SparseMatrix<double>& diracMatrixCached_();
    // First-order INTRINSIC Dirac D_int (4F×4V), cached (OperatorId::DiracIntrinsicD).
    // nxr::operators::dirac::matrixIntrinsic (immersion/edge-based; the spin-connection root).
    const Eigen::SparseMatrix<double>& diracIntrinsicMatrixCached_();

    // Face-domain relative-Dirac extrinsic block Ẽ (4F×4F), cached (OperatorId::DiracFace).
    const Eigen::SparseMatrix<double>& diracFaceExtrinsicBlockCached_();
    // First-order face Dirac D̃ (4V×4F), cached (OperatorId::DiracFaceD). nxr::operators::dirac::matrixFace.
    const Eigen::SparseMatrix<double>& diracFaceMatrixCached_();
    // First-order INTRINSIC face Dirac D̃_int (4V×4F), cached (OperatorId::DiracFaceIntrinsicD).
    // nxr::operators::dirac::matrixFaceIntrinsic (centroid immersion dual of matrixFace).
    const Eigen::SparseMatrix<double>& diracFaceIntrinsicMatrixCached_();
    // DEC 2-form Laplacian K̃ = d₁⋆₁⁻¹d₁ᵀ ([F×F]) — the diracFace intrinsic anchor.
    // Cached (mesh-fixed; not user-facing, so no OperatorId) so a τ-sweep reuses it.
    const Eigen::SparseMatrix<double>& twoFormLaplacianCached_();
    // Assemble L̃(τ) = (1−τ)(K̃⊗I₄) + τ·Ẽ by value, K̃ = d₁⋆₁⁻¹d₁ᵀ. τ ∈ [0,1].
    Eigen::SparseMatrix<double> diracFaceFamily_(double tau);
    // Barycentric dual-mesh face gradient G̃ (3F×F), cached (OperatorId::GradFace).
    // nxr::operators::facegrad::gradient.
    const Eigen::SparseMatrix<double>& gradFaceCached_();
    // Face Laplacian K̃ = G̃ᵀ⋆_F G̃ (F×F), cached (OperatorId::LapFace). nxr::operators::facegrad::laplacian.
    const Eigen::SparseMatrix<double>& lapFaceCached_();

    friend class facet::OperatorsFacet;
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>             mesh_;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>         geometry_;
    ManifoldOptions                          options_;
    std::unique_ptr<IntrinsicSurface>        surface_;           // the one intrinsic construction (flips only when options_.mesh == Intrinsic)
    std::unique_ptr<Eigen::SparseMatrix<double>> operatorCotan_; // its cotan Laplacian, never released
    std::unique_ptr<operators::DecMatrices>        operatorDec_;       // the DEC of the operator mesh
    std::unique_ptr<Eigen::VectorXd>         operatorDualAreas_;

    Eigen::MatrixXd                          vertexPositions_;  // [nV,3] retained input
    Eigen::MatrixXi                          faces_;            // [nF,3] retained input (0-based)
    std::unique_ptr<geometry::MeshGeometry>  lightGeometryCache_;
    std::unique_ptr<geometry::MeshTopology>  topologyDataCache_;

    ConnectionType            activeConnectionType_    = ConnectionType::LeviCivita;
    std::map<int,double> activeSingularities_;
    // Validates Σ singularities == eulerCharacteristic(); throws on mismatch.
    void validateSingularities_(const std::map<int,double>& s) const;

    std::unique_ptr<operators::DECOperators>  decCache_;
    std::unique_ptr<operators::DECOperators>  decEmbeddedCache_;
    std::unique_ptr<operators::CholeskyCache> choleskyCachePtr_;
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
} // namespace nxr::geometry

namespace nxr::geometry {
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

using RowMat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMat = Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** Area [nF] of each face of a mesh (positions [nV x 3], faces [nF x 3]). */
Eigen::VectorXd faceAreas(const RowMat3& positions, const FaceMat& faces);

} // namespace nxr::geometry


// THE ONE INTRINSIC SURFACE: the input mesh, its geometry and the intrinsic Delaunay triangulation, all on geometry-central.
//
// Specified in docs/specs/2026-11-17-geometry-central-switch.md (section 2). It replaces the line port of the Studio's construction
// (the port, deleted in G7). `nxr/chart.h` (G4) is a wrapper of this surface.
//
// Construction, in this order (each step names the geometry-central call):
//   1. ManifoldSurfaceMesh + VertexPositionGeometry of the input (positions [3 nV] row-major, triangle faces [3 nF])
//   2. mollifyIntrinsic on the input's edge lengths (option `mollify`, Sharp & Crane 2020); ε is recorded as the length added
//   3. the mixed-Voronoi mass of the mollified INPUT, recorded before any flip (option `mass`)
//   4. SignpostIntrinsicTriangulation(input mesh, mollified input geometry): angle 0 at each vertex is the input's `v.halfedge()`, through every flip
//   5. triangleTestEPS = `flipEps`, then flipToDelaunay (flips counted through edgeFlipCallbackList), isDelaunay() checked
//   6. the record: what was told and done, the geometry-central version and the gauge identity
// With `intrinsic = false` (the named option `mesh = embedded`) steps 2 to 5 are skipped and every operator is the input geometry's.
//
// THE GAUGE of nxr-neuro is geometry-central's vertex gauge of the (mollified) input mesh: angle 0 along the input's `v.halfedge()`, corner angles
// rescaled by `cornerScaledAngles` (2π; π at a boundary vertex), realised in 3D by `tangentBasis()` (equal to `vertexTangentBasis` when ε = 0).
//
// Pure: arrays in, the object out. No file access, no io. Not thread-safe to share between threads while a quantity is first required.
#pragma once

#include <Eigen/Sparse>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace geometrycentral { namespace surface {
class ManifoldSurfaceMesh;
class IntrinsicGeometryInterface;
class VertexPositionGeometry;
class EdgeLengthGeometry;
}}

namespace nxr::manifold {

/** The mass M of the generalised problem L φ = λ M φ (and of the heat solvers). */
enum class MassKind {
    InputMixedVoronoi,   // the mixed-Voronoi area (Meyer et al. 2003) of the mollified INPUT, recorded before any flip (default)
    Barycentric,         // geometry-central's own: a third of each intrinsic triangle's area
    Galerkin,            // the consistent P1 mass (a sparse matrix; `massDiagonal()` refuses it)
};

/** The two named options of the intrinsic Delaunay construction (mollification and flip tolerance), shared by every builder of an intrinsic surface. */
struct DelaunayOptions {
    double mollify = 1e-6;    // the mollification margin δ relative to the mean edge length (0 = no margin; ε is still added where a triangle is invalid)
    double flipEps = 1e-6;    // an edge is flipped while its cotangent weight is below −flipEps
};
namespace delaunay { using DelaunayOptions [[deprecated("nxr::manifold::DelaunayOptions")]] = nxr::manifold::DelaunayOptions; }   // the port's name, one release

/** The named, recorded choices of the construction. */
struct IntrinsicOptions {
    bool        intrinsic = true;                          // false: the named option `mesh = embedded` (no mollification, no flips)
    double      mollify   = 1e-6;                          // δ relative to the mean input edge length (0 = no margin; ε is still added where a triangle is invalid)
    double      flipEps   = 1e-6;                          // an edge is flipped while its cotangent weight is below −flipEps (geometry-central's triangleTestEPS)
    MassKind    mass      = MassKind::InputMixedVoronoi;
    std::string surface;                                   // the surface's id, recorded in the gauge identity ("" when the caller has none)
};

/** What the construction was told and did. */
struct IntrinsicRecord {
    std::string mesh;                // "intrinsic" | "embedded"
    double      mollify  = 0.0;      // the option used
    double      epsilon  = 0.0;      // ε: the length added to every input edge by the mollification (0 = nothing needed)
    double      flipEps  = 0.0;      // the option used
    int         flips    = 0;        // edge flips made (counted by callback)
    bool        delaunay = true;     // isDelaunay() after the flips
    std::string mass;                // "input-mixed-voronoi" | "barycentric" | "galerkin"
    int         boundaryVertices = 0;
    std::string geometryCentral;     // the pinned commit, e.g. "8579865"
    std::string fingerprint;         // 8 hex: a hash of each vertex's reference neighbour (head of v.halfedge()) and ε
    std::string gauge;               // "geometry-central@<commit>/vertex"
    std::string gaugeId;             // gauge + surface + fingerprint: two values are in the same gauge exactly when it agrees
};

/** The pinned geometry-central commit (7 hex), read from the submodule at configure time. */
const char* geometryCentralVersion();

/** The 3D realisation of the gauge: at each vertex the unit normal and the tangent basis (e1 at angle 0, e2 = n × e1), [3 nV] row-major. */
struct TangentBasis { std::vector<double> e1, e2, n; };

class IntrinsicSurface {
public:
    /** Throws core::Error(InvalidInput) on a face index outside [0, nV), a non-triangle input, a non-manifold or inconsistently oriented edge,
     *  or a non-finite or non-positive input edge length (refused before the mollification). */
    IntrinsicSurface(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& o = {});
    ~IntrinsicSurface();
    IntrinsicSurface(const IntrinsicSurface&) = delete;
    IntrinsicSurface& operator=(const IntrinsicSurface&) = delete;

    int nV() const;
    int nF() const;
    const IntrinsicOptions& options() const;
    const IntrinsicRecord& record() const;

    /** The input mesh (its `v.halfedge()` is angle 0 of the gauge). */
    geometrycentral::surface::ManifoldSurfaceMesh& inputMesh();
    /** The input's embedded geometry (positions, normals; NOT mollified). */
    geometrycentral::surface::VertexPositionGeometry& embedded();
    /** The mollified input as intrinsic geometry (`mesh = embedded`: the embedded geometry itself): the gauge's `halfedgeVectorsInVertex`. */
    geometrycentral::surface::IntrinsicGeometryInterface& inputGeometry();
    /** The geometry every operator is built on: the signpost triangulation (`mesh = embedded`: the input's). It serves the chosen mass. */
    geometrycentral::surface::IntrinsicGeometryInterface& geometry();
    /** The mesh `geometry()` lives on (the intrinsic triangulation's, or the input's). */
    geometrycentral::surface::ManifoldSurfaceMesh& mesh();

    /** The cotangent Laplacian L of `geometry()` (positive semidefinite). */
    Eigen::SparseMatrix<double> cotanLaplacian();
    /** The chosen mass as a diagonal [nV]; throws core::Error(InvalidInput) for `Galerkin`. */
    std::vector<double> massDiagonal();
    /** The chosen mass as a matrix (diagonal for the lumped kinds). */
    Eigen::SparseMatrix<double> massMatrix();
    /** The vertex connection Laplacian on the n-RoSy bundle (complex Hermitian, positive semidefinite); n = 1 is geometry-central's own, n > 1 is built
     *  from its transport raised to the n-th power and its edge cotangent weights. */
    Eigen::SparseMatrix<std::complex<double>> connectionLaplacian(int nSym = 1);

    /** The gauge in 3D: normals from the input geometry (corner-angle weighted), e1 from the (mollified) gauge angles. */
    TangentBasis tangentBasis();

    /** The mixed-Voronoi mass (Meyer et al. 2003) of the mollified input, [nV]. */
    std::vector<double> inputMixedVoronoiMass();
    /** The barycentric mass (a third of each triangle's area of `geometry()`), [nV], whatever the mass option. */
    std::vector<double> barycentricMass();
    /** The consistent P1 (Galerkin) mass of `geometry()`, whatever the mass option. */
    Eigen::SparseMatrix<double> galerkinMassMatrix();

    /** The combinatorics and raw signposts of `mesh()` (the intrinsic triangulation; with `mesh = embedded` the input), in the layout the stored
     *  intrinsic complexes use: face f has the corners 3f + k (k = 0, 1, 2, in the order of `f.halfedge()`, `.next()`), the halfedge at corner h runs from
     *  `faces[h]` to `faces[3f + (k + 1) % 3]`. Edges are numbered as `mesh()` numbers them and oriented along `e.halfedge()` (tail to tip). */
    struct Complex {
        std::vector<int32_t> faces;       // [3 nF] the vertex at each corner
        std::vector<int32_t> edges;       // [2 nE] tail, tip
        std::vector<int32_t> faceEdges;   // [3 nF] signed, 1-based edge of the halfedge at each corner: +e + 1 along the edge's orientation, −(e + 1) against
        std::vector<double>  length;      // [nE]
        std::vector<double>  angle;       // [3 nF] the signpost angle of the halfedge at each corner: raw, in [0, cone angle), measured from the input's reference halfedge
        std::vector<double>  angleSum;    // [nV] the cone angle (sum of the corner angles round the vertex)
        std::vector<int32_t> reference;   // [nV] the head of the input's `v.halfedge()`: angle 0
    };
    Complex complex();
    /** THE INPUT'S CORNERS in the face-corner layout (corner 3f + k of input face f, k in the order of its vertices), before any flip: each corner's raw signpost angle at its
     *  vertex and its angle (mollified input), and per vertex the angle sum and whether it is on the boundary. What the optical flow's gauge reads (Corners). */
    struct InputCorners {
        std::vector<double>  signpost, angle;   // [3 nF]
        std::vector<double>  angleSum;          // [nV]
        std::vector<uint8_t> boundary;          // [nV]
    };
    InputCorners inputCorners();
    /** The raw signpost angle of each of the INPUT mesh's halfedges (mollified input), indexed by halfedge index: the same coordinate as `Complex::angle`. */
    std::vector<double> inputSignpostAngles();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace nxr::manifold

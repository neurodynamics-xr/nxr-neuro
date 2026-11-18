#pragma once
// the heat method, geodesic paths and the loci (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/cache.h"

namespace nxr::geometry {
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
} // namespace nxr::geometry

namespace nxr::geometry::query {

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
} // namespace nxr::geometry::query

namespace nxr::geometry::measure {

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
} // namespace nxr::geometry::measure


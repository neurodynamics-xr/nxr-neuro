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
#include "nxr/geometry/surface_arrays.h"

namespace nxr::geometry {
// ── Geodesic Distance (Heat Method) ──────────────────────────
//
// geometry-central's HeatMethodDistanceSolver, built inside the call on the surface rebuilt from its arrays: it factors both systems (mean-curvature
// flow `M + tA` and Laplacian `A`) once per call. A caller with many queries uses the BATCH form (all source sets in one call): one factorization,
// every set back-substituted. Nothing is held between calls.

// Which mesh the heat method runs on. Intrinsic (DEFAULT): the operator mesh, the intrinsic Delaunay triangulation — the
// Studio's HeatMethod (nxr/geometry/charts.h). Input: geometry-central's heat method on the input mesh (the former default), kept as the named option.
enum class HeatMesh { Intrinsic, Input };

/** Geodesic distances [nV] from a set of source vertices via the heat method. */
Eigen::VectorXd heat(const SurfaceArrays& surface, const std::vector<int>& sourceVertices, double tCoef = 1.0, HeatMesh mesh = HeatMesh::Intrinsic);
/** The batch form: column j [nV] = the distances from sourceSets[j], all on one solver. */
Eigen::MatrixXd heat(const SurfaceArrays& surface, const std::vector<std::vector<int>>& sourceSets, double tCoef = 1.0, HeatMesh mesh = HeatMesh::Intrinsic);

// ── Signed Heat Method (Feng & Crane 2024) ───────────────────
//
// Signed geodesic distance from a curve on the surface. The curve is a polyline of vertex indices; `isLoop` closes it (last → first).
// Positive distance one side of the curve, negative the other; the level-set constraint keeps the zero-set pinned at the curve.
enum class SignedHeatLevelSet {
    None = 0,        // unconstrained — curve may not lie exactly on the zero set
    ZeroSet = 1,     // pin the zero level set to the curve (default)
    Multiple = 2,    // pin each curve to its own level set
};

/** Signed geodesic distance [nV] from a curve to every vertex (sign follows the curve's orientation; reverse the order to flip it). */
Eigen::VectorXd signedHeat(const SurfaceArrays& surface, const std::vector<int>& curveVertices, bool isLoop = true,
                           SignedHeatLevelSet levelSet = SignedHeatLevelSet::ZeroSet, double tCoef = 1.0);
/** The batch form: column j = the signed distance from curves[j], all on one solver. */
Eigen::MatrixXd signedHeat(const SurfaceArrays& surface, const std::vector<std::vector<int>>& curves, bool isLoop = true,
                           SignedHeatLevelSet levelSet = SignedHeatLevelSet::ZeroSet, double tCoef = 1.0);
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
    const SurfaceArrays& surface,
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
    const SurfaceArrays& surface,
    int vStart,
    int vEnd
);

/** The EXACT polyhedral geodesic distance between two vertices: the
 *  length of the shortest path over the faces (MMP, geometry-central's
 *  GeodesicAlgorithmExact), mesh units. The flip-out path of
 *  `tracePathSurface` is only locally shortest (shortest in the
 *  homotopy class of its Dijkstra start), so its `length` is >= this;
 *  the heat method (`geometry::heat`) approximates it. */
double exactDistance(
    const SurfaceArrays& surface,
    int vStart,
    int vEnd
);

/** The exact geodesic distance and log map FROM A SURFACE POINT to every
 *  vertex. The source is a vertex (`face < 0`, `vertex` the index) or a
 *  point in a face (`face` the face index, `bary` its barycentric
 *  coordinates in the face's vertex order, non-negative, summing to 1). */
struct ExactSource {
    int vertex = -1;
    int face = -1;
    Eigen::Vector3d bary = Eigen::Vector3d::Zero();
};
struct ExactFromPoint {
    Eigen::VectorXd distance;  // [nV] exact polyhedral distance, mesh units
    Eigen::MatrixXd log;       // [nV, 2] log map in the source point's tangent space (geometry-central's convention)
};
ExactFromPoint exactFromPoint(const SurfaceArrays& surface, const ExactSource& source);

// ── Locus primitives (D11) ───────────────────────────────────
//
// `query::*` returns the geometric locus itself (the "where"); the
// paired `measure::*` (below) returns its scalar metric (the "how
// much"). Three loci: a point, a polyline, a region of faces.

struct PointLocus {
    int vertex;  // vertex index of the surface
};

struct PolylineLocus {
    Eigen::MatrixXd points;  // [N, 3] world-space points along the polyline
};

struct RegionLocus {
    std::vector<int> faces;  // face indices covered by the region
};

/** Trivial point query — echoes the input vertex after a range check.
 *  Throws Error(InvalidInput) on `v < 0 || v >= m.nV()`. */
PointLocus point(const SurfaceArrays& surface, int v);

/** Polyline between two vertices via the edge-flip geodesic algorithm
 *  (Sharp & Crane 2020). Identical to `tracePath` but typed as a
 *  PolylineLocus so it pairs with `measure::line`. */
PolylineLocus line(const SurfaceArrays& surface, int vStart, int vEnd);

/** Region of faces inside the geodesic ball of radius `level` around
 *  vertex `v`. A face is included iff all three of its vertices have
 *  heat-method geodesic distance `<= level` from `v` — conservative,
 *  slightly underestimates the true region area, but well-defined and
 *  free of barycentric interpolation.
 *
 *  Runs the heat method once per call; for many regions, compute the
 *  distances with the batch `geometry::heat` and threshold them.
 *
 *  Throws Error(InvalidInput) on `v` out of range or `level <= 0`. */
RegionLocus area(const SurfaceArrays& surface, int v, double level);
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
Eigen::Vector3d point(const SurfaceArrays& surface, int v);

/** Polyline length along the edge-flip geodesic from `vStart` to `vEnd`.
 *  Equivalent to `line(query::line(m, vStart, vEnd))`. */
double line(const SurfaceArrays& surface, int vStart, int vEnd);

/** Polyline length of a pre-computed `PolylineLocus`. Sum of Euclidean
 *  distances between consecutive points. Returns 0 for empty / single-
 *  point polylines. */
double line(const query::PolylineLocus& locus);

/** Total face area of the geodesic-ball region of radius `level` around
 *  vertex `v`. Equivalent to `area(m, query::area(m, v, level))`. */
double area(const SurfaceArrays& surface, int v, double level);

/** Total face area of a pre-computed `RegionLocus`, summed from
 *  geometry-central's cached face areas. */
double area(const SurfaceArrays& surface, const query::RegionLocus& locus);
} // namespace nxr::geometry::measure


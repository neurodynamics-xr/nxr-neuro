// nxr::geometry::pointset — a triangle mesh whose vertices ARE a set of points lying on a cap (a sensor helmet, an electrode cap): every point a vertex, no
// point added or moved. The reference is nxr's sensor mesh (the sensors triangulated for the heat distance, the phase current and the Hodge–Dirac modes on
// the sensors); the steps and their order are its own:
//   1. the points centred on their centroid and mapped to a plane by the azimuthal equidistant projection about a POLE (polar angle θ from the pole,
//      azimuth φ: the plane point θ (cos φ, sin φ));
//   2. the plane points' Delaunay triangulation (planarDelaunay);
//   3. a triangle is CUT when one of its edges (3-D length) is not shorter than `cut` × the median edge (the median over every triangle's three edges,
//      an interior edge counted once per triangle): the long chords across the rim go;
//   4. REPAIR, pinches: a vertex where two fans of triangles meet at one point (not a manifold) is closed by one triangle (the vertex and the closest pair of
//      its rim neighbours from two different fans) when that pair is closer than the cut length, else the smaller fan is dropped (refused when a point would
//      be lost); repeated until no vertex is pinched (at most `repair_passes` passes);
//   5. REPAIR, holes: every boundary loop but the longest (the rim) is closed: three vertices by one triangle, more by splitting along the shortest
//      diagonal, recursively;
//   6. ORIENTATION: each triangle turned so that its normal points away from the points' centroid (outward), or left as the triangulation gives it.
// The result is reported with its topology (boundary loops, Euler characteristic) and the points that no triangle reached: the caller decides what to accept.
// PURE: arrays in, arrays out; no file, no mesh object. Positions in any unit (every length is relative to the median edge).
// What it cannot resolve: the triangulation is the projection's Delaunay, not the surface's own; on a cap that turns past the pole's horizon (θ near π)
// the projection folds and the triangulation is not meaningful.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace nxr::geometry::pointset {

enum class Orientation { outward, as_triangulated };

struct PointSetMeshOptions {
  double cut = 2.0;                          // a triangle is kept when all three of its edges are shorter than cut x the median edge (nxr: 2)
  std::array<double, 3> pole{{0.0, 0.0, 1.0}};   // the projection's pole, a direction from the centroid (nxr: +z of the points' frame; it need not be unit length)
  Orientation orientation = Orientation::outward;   // outward: each normal away from the centroid (nxr's sensor surface); as_triangulated: as made
  int repair_passes = 20;                    // the most passes the pinch repair makes before refusing (nxr: 20)
};

struct PointSetMesh {
  std::vector<std::int32_t> faces;           // [F x 3], 0-based indices into the points
  std::int64_t delaunay_triangles = 0;       // the planar Delaunay's triangles
  std::int64_t cut_triangles = 0;            // removed by the cut
  std::int64_t pinches_closed = 0;           // triangles added at pinched vertices
  std::int64_t fans_dropped = 0;             // fans removed at pinched vertices (their gap wider than the cut)
  std::int64_t hole_triangles = 0;           // triangles added to close holes
  std::int64_t edges = 0;                    // the mesh's edges
  int boundary_loops = 0;                    // 1: a disc
  std::int64_t euler_characteristic = 0;     // V - E + F over the points that are vertices of a triangle (1: a disc)
  std::vector<std::int32_t> points_without_face;   // points no triangle reached (none in a sensor surface)
  double median_edge = 0;                    // the median edge of the planar Delaunay, in the points' unit
};

/** The triangulation of points [n x 3] (row-major) lying on a cap: the steps above. Refuses (std::invalid_argument) fewer than 3 points, a non-finite
 *  coordinate, a point at the centroid, coincident or all-collinear projected points, a zero pole, a cut <= 0, and a pinch it cannot repair. */
PointSetMesh pointSetMesh(const std::vector<double>& points, const PointSetMeshOptions& options = {});

/** The Delaunay triangulation of plane points [n x 2] (row-major): triangles [T x 3], counter-clockwise. A sweep over the points in (x, y) order, each joined
 *  to the hull edges it sees, then edges flipped until every one is locally Delaunay (Lawson); on cocircular points one of the valid triangulations.
 *  Refuses coincident points and points that are all on one line. */
std::vector<std::int32_t> planarDelaunay(const std::vector<double>& xy);

}  // namespace nxr::geometry::pointset

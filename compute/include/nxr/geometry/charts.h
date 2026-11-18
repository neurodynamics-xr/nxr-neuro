#pragma once
// nxr::geometry: CHARTS on the intrinsic surface. The heat method's distance, the vector heat method's transport, the LOG MAP (geodesic polar coordinates), a vertex's tangent vector as an
// angle in THE gauge, and the DISC of a chart. Step 1 of the cortical coordinate system (docs/audits/2026-11-11-cortical-coordinates.md).
//
// A thin wrapper of geometry-central (docs/specs/2026-11-17-geometry-central-switch.md, G4). Per connected piece (a hemisphere) one `geometry::IntrinsicSurface` (the signpost
// intrinsic Delaunay triangulation of the mollified input, the input's mixed-Voronoi mass), with `HeatMethodDistanceSolver` and `VectorHeatMethodSolver` on it. The names and the
// arguments are the Studio's `SurfaceKit` (distance, transported, logMap, headingOf, disc, gradient); the values are geometry-central's.
//
// THE GAUGE is geometry-central's vertex gauge of the (mollified) input: angle 0 along each vertex's reference halfedge, e1 = `IntrinsicSurface::tangentBasis()` (equal to
// `vertexTangentBasis` when nothing was mollified), e2 = n x e1. Every complex tangent value of the kit (`transported`, `logMap`'s angle, `gaugeHeading`) is in it.
//
//   heatDistance(sources)      r, the geodesic distance from a set of source vertices (HeatMethodDistanceSolver on the intrinsic surface)
//   transported(v, heading)    the unit vector at v headed `heading` in v's gauge carried everywhere (VectorHeatMethodSolver::transportTangentVector)
//   logMap(v)                  (r, theta): r the heat distance, theta = arg z of geometry-central's log map z (LogMapStrategy::VectorHeat), 0 along v's reference edge; NaN off v's component
//   gaugeHeading(v, d)         a 3D direction at v as an angle in v's gauge (NaN when normal to the surface); headingVector is its inverse. A direction N in a shared frame is read as
//                              theta_N = theta - gaugeHeading(v, N(v))
//   disc(v, radius)            the vertices within a geodesic radius of v
// VALIDITY: a log map is a chart only inside the injectivity radius of its seed (about 10 mm on a pial surface; the audit). `disc` takes the radius the caller trusts (`maxRadius`,
// e.g. the injectivity radius measured elsewhere) and REFUSES a larger one; nothing here estimates the cut locus.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "nxr/geometry/intrinsic_surface.h"

namespace nxr::geometry {

/** At each vertex the unit normal (corner-angle weighted, geometry-central's `vertexNormals`), e1 (the gauge's angle 0) and e2 = n x e1. [3 n] each, double. */
struct Basis { std::vector<double> e1, e2, n; };

struct LogMap { Eigen::VectorXd r, theta; };   // [V] each, NaN off the seed's component

/** The surface's charts: per connected component an `IntrinsicSurface` and the two heat solvers on it (built when first used). */
class ChartKit {
public:
    /** `positions` [3 nV] row-major, `faces` [3 nF]. Each connected piece (a hemisphere) is solved by itself; a vertex in no face is on no piece. `o` the mollification and flip tolerance
     *  (the named options of the construction). */
    ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const DelaunayOptions& o = {});
    /** The same with the full options of the intrinsic surface (`mass`, ...); `o.intrinsic = false` is the named option `mesh = embedded`. */
    ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const geometry::IntrinsicOptions& o);
    ~ChartKit();
    ChartKit(const ChartKit&) = delete;
    ChartKit& operator=(const ChartKit&) = delete;
    int n() const;
    /** `distance(v)`: the heat distance from v over its component (Infinity on the others). */
    Eigen::VectorXd heatDistance(int v);
    /** `distanceSet`: from a set of vertices, on the first one's component (NaN elsewhere). */
    Eigen::VectorXd heatDistance(const std::vector<int>& sources);
    /** `transported`: [2 V] (re, im per vertex, each in its vertex's gauge; NaN off v's component). */
    Eigen::VectorXd transported(int v, double heading = 0.0);
    /** `logMap`. */
    LogMap logMap(int v);
    /** The angle of the 3D direction d at v in v's gauge; NaN where it has none (zero, or normal to the surface). */
    double gaugeHeading(int v, const double d[3]);
    /** The tangent vector at v headed `psi` (unit length), out[0..2]. The inverse of gaugeHeading. */
    void headingVector(int v, double psi, double out[3]);
    /** `disc`: the vertices within `radius` of v by geodesic distance (`<=`). Throws core::Error when radius > maxRadius (the chart's validity radius, if known). */
    std::vector<int> disc(int v, double radius, double maxRadius = std::numeric_limits<double>::infinity());
    /** `gradient`: a scalar's surface gradient at the vertices [3 V]: per input triangle the gradient of the linear interpolant, averaged by area. */
    Eigen::VectorXd gradient(const Eigen::VectorXd& field) const;
    /** e1, e2, n at every vertex (NaN off components with faces). */
    Basis basis() const;
    /** The intrinsic surface of v's component (its vertices are numbered locally: see `localIndex`); throws core::Error when v is on no component. */
    geometry::IntrinsicSurface& surfaceOf(int v);
    /** v's index inside `surfaceOf(v)` (the vertices of a component in ascending global order). */
    int localIndex(int v) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace nxr::geometry


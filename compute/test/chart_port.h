// THE PORT OF THE STUDIO'S CHARTS, kept ONLY for test_gc_port_comparison (the one-time comparison, G2 / appendix B) until the port is deleted (G7). Not part of the library:
// nxr/chart.h is the geometry-central wrapper since G4. Unchanged from nxr/chart.h and src/chart.cpp as of G3, in the namespace nxr::chart_port.
#pragma once
// nxr::chart — CHARTS on the intrinsic mesh: the heat method's distance, the vector heat method's transport, the LOG MAP (geodesic polar coordinates), a vertex's tangent
// vector as an angle in the signpost gauge, and the DISC of a chart. Step 1 of the cortical coordinate system (docs/audits/2026-11-11-cortical-coordinates.md).
//
// A line port of the Cortical Flow Studio (nxr-cortical-flow studio/src/engine/): `toolbox/differential/heatMethod.ts` (HeatMethod: t = (mean intrinsic edge)^2, the Dirichlet
// average on a boundary), `compute/runtime/surfaceKit.ts` (transported, #vectorHeat, logMap, distance, disc, headingOf, gradient) and `toolbox/differential/connectionLaplacian.ts`
// (tangentBasis, embedTangent). Same maths, same names, same constants; the Studio's float32 values are kept where they enter the numbers (the tangent basis).
// Everything is on the intrinsic Delaunay triangulation with the signposts (nxr/delaunay.h) and the connection Laplacian of nxr/opticalflow_connection.h (componentPencil):
// the mass is the mixed-Voronoi mass of the mollified INPUT (recorded before the flips).
//
//   heatDistance(sources)      r, the geodesic distance from a set of source vertices (heat method on the intrinsic mesh)
//   transported(v, heading)    the unit vector at v headed `heading` in v's basis (e1 = its reference edge projected, e2 = n x e1) carried everywhere: (C + M/t) X = M delta
//   logMap(v)                  (r, theta): r the heat distance, theta = arg(grad r / X), 0 along v's reference edge (the signpost gauge); NaN off v's component
//   gaugeHeading(v, d)         a 3D direction at v as an angle in v's basis (NaN when normal to the surface); headingVector is its inverse. A direction N in a shared gauge is read as
//                              theta_N = theta - gaugeHeading(v, N(v))
//   disc(v, radius)            the vertices within a geodesic radius of v
// VALIDITY: a log map is a chart only inside the injectivity radius of its seed (about 10 mm on a pial surface; the audit). The Studio and nxr have no criterion: `disc` takes the radius
// the caller trusts (`maxRadius`, e.g. the injectivity radius measured elsewhere) and REFUSES a larger one; nothing here estimates the cut locus.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "nxr/delaunay.h"

namespace nxr::chart_port {
namespace delaunay = nxr::manifold::delaunay;

/** `HeatMethod` (heatMethod.ts): the heat method on an intrinsic mesh (flipped to Delaunay; `m.mass` the recorded mass, else the mixed Voronoi mass as it is). t = tCoef (mean edge)^2. */
class HeatMethod {
public:
    explicit HeatMethod(const delaunay::Mesh& m, double tCoef = 1.0);
    ~HeatMethod();
    HeatMethod(const HeatMethod&) = delete;
    HeatMethod& operator=(const HeatMethod&) = delete;
    int n() const;
    double t() const;
    /** The geodesic distance from the source vertices to every vertex. */
    Eigen::VectorXd distance(const std::vector<int>& sources) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/** `tangentBasis`: at each vertex the normal (area-weighted from the input faces), e1 its reference input edge projected on the tangent plane, e2 = n x e1. float32 values, as the Studio's. */
struct Basis { std::vector<float> e1, e2, n; };   // [3 n] each
Basis tangentBasis(const delaunay::Mesh& m, const double* positions, const int32_t* faces, int nF);

struct LogMap { Eigen::VectorXd r, theta; };   // [V] each, NaN off the seed's component

/** The surface's charts (`SurfaceKit` of the Studio): per connected component the intrinsic Delaunay triangulation, its connection Laplacian, the tangent basis, the heat method. */
class ChartKit {
public:
    /** `positions` [3 nV] row-major, `faces` [3 nF]. Each connected piece (a hemisphere) is solved by itself. */
    ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const delaunay::DelaunayOptions& o = {});
    ~ChartKit();
    ChartKit(const ChartKit&) = delete;
    ChartKit& operator=(const ChartKit&) = delete;
    int n() const;
    /** `distance(v)`: the heat distance from v over its component (Infinity on the others). */
    Eigen::VectorXd heatDistance(int v);
    /** `distanceSet`: from a set of vertices, on the first one's component (NaN elsewhere). */
    Eigen::VectorXd heatDistance(const std::vector<int>& sources);
    /** `transported`: [2 V] (re, im per vertex, each in its vertex's basis; NaN off v's component). */
    Eigen::VectorXd transported(int v, double heading = 0.0);
    /** `logMap`. */
    LogMap logMap(int v);
    /** `headingOf` (the CL basis): the angle of the 3D direction d at v in v's basis; NaN where it has none (zero, or normal to the surface). */
    double gaugeHeading(int v, const double d[3]);
    /** `embedTangent`: the tangent vector at v headed `psi` (unit length), out[0..2], float32 values. The inverse of gaugeHeading. */
    void headingVector(int v, double psi, double out[3]);
    /** `disc`: the vertices within `radius` of v by geodesic distance (the Studio's `<=`). Throws core::Error when radius > maxRadius (the chart's validity radius, if known). */
    std::vector<int> disc(int v, double radius, double maxRadius = std::numeric_limits<double>::infinity());
    /** `gradient`: a scalar's surface gradient at the vertices [3 V]: per input triangle the gradient of the linear interpolant, averaged by area. */
    Eigen::VectorXd gradient(const Eigen::VectorXd& field) const;
    /** e1, e2, n at every vertex [3 V] (NaN off components with faces). */
    Basis basis() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace nxr::chart_port

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/signal/surface_calculus.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/marching_triangles.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>

namespace nxr::signal {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Contour lines of a per-vertex scalar field (the Studio's isoline.ts, now on geometry-central's marching triangles: the segments are the same
// edge crossings, in the order of the curves rather than of the faces).
IsolineResult isoline(
    const geometry::SurfaceArrays& surface,
    const Eigen::VectorXd& scalarField,
    int numLevels,
    double minValue,
    double maxValue
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nV = surface.nV();

    IsolineResult result;
    result.segmentCount = 0;

    if (numLevels < 2 || scalarField.size() != nV) {
        result.positions.resize(0, 3);
        return result;
    }

    // Auto-detect range if not provided
    double lo = minValue;
    double hi = maxValue;
    if (minValue == 0.0 && maxValue == 0.0) {
        lo = std::numeric_limits<double>::infinity();
        hi = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < nV; i++) {
            double v = scalarField(i);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }

    double range = hi - lo;
    if (range <= 0) {
        result.positions.resize(0, 3);
        return result;
    }

    double levelSpacing = range / numLevels;

    // Collect segments into a dynamic vector of (x,y,z) endpoints. One level at a time: geometry-central's marching triangles (a level's curves, as
    // surface points on the input's edges), each curve's consecutive points a segment.
    std::vector<double> pts;
    VertexData<double> u(mesh);
    for (int i = 0; i < nV; ++i) u[mesh.vertex(static_cast<size_t>(i))] = scalarField(i);
    for (int level = 1; level < numLevels; level++) {
        const double levelValue = lo + level * levelSpacing;
        const std::vector<std::vector<SurfacePoint>> curves = marchingTriangles(geom, u, levelValue);
        for (const auto& curve : curves)
            for (size_t k = 0; k + 1 < curve.size(); ++k) {
                const Vector3 a = curve[k].interpolate(geom.inputVertexPositions), b = curve[k + 1].interpolate(geom.inputVertexPositions);
                pts.push_back(a.x); pts.push_back(a.y); pts.push_back(a.z);
                pts.push_back(b.x); pts.push_back(b.y); pts.push_back(b.z);
            }
    }

    int segCount = static_cast<int>(pts.size() / 6);
    result.segmentCount = segCount;
    result.positions.resize(segCount * 2, 3);
    for (int i = 0; i < segCount * 2; i++) {
        result.positions(i, 0) = pts[i * 3 + 0];
        result.positions(i, 1) = pts[i * 3 + 1];
        result.positions(i, 2) = pts[i * 3 + 2];
    }

    std::cout << "[isoline] Extracted " << segCount << " segments across "
              << numLevels << " levels" << std::endl;

    return result;
}

} // namespace nxr::signal
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

// Contour lines of a per-vertex scalar field: the same edge crossings by two methods (nxr/signal/surface_calculus.h, IsolineMethod): the Studio's
// per-triangle isoline.ts (the default: faster on a cortex-sized mesh, test_isoline_methods) or geometry-central's marching triangles (curve order).
IsolineResult isoline(
    const geometry::SurfaceArrays& surface,
    const Eigen::VectorXd& scalarField,
    int numLevels,
    double minValue,
    double maxValue,
    IsolineMethod method
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

    std::vector<double> pts;
    if (method == IsolineMethod::MarchingTriangles) {
        // One level at a time: geometry-central's marching triangles (a level's curves, as surface points on the input's edges), each curve's
        // consecutive points a segment.
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
    } else {
        // The Studio's isoline.ts: face by face, every level, the two edge crossings of the face a segment (faces in order).
        for (Face f : mesh.faces()) {
            Halfedge h = f.halfedge();
            Vertex v0 = h.tailVertex(), v1 = h.next().tailVertex(), v2 = h.next().next().tailVertex();
            const int i0 = static_cast<int>(v0.getIndex()), i1 = static_cast<int>(v1.getIndex()), i2 = static_cast<int>(v2.getIndex());
            const double d0 = scalarField(i0), d1 = scalarField(i1), d2 = scalarField(i2);
            const Vector3 p0 = geom.inputVertexPositions[v0], p1 = geom.inputVertexPositions[v1], p2 = geom.inputVertexPositions[v2];
            struct Edge { double da, db; Vector3 pa, pb; };
            const Edge edges[3] = {{d0, d1, p0, p1}, {d1, d2, p1, p2}, {d2, d0, p2, p0}};
            for (int level = 1; level < numLevels; level++) {
                const double levelValue = lo + level * levelSpacing;
                Vector3 crossings[2];
                int crossCount = 0;
                for (int e = 0; e < 3; e++) {
                    const double edgeLo = std::min(edges[e].da, edges[e].db), edgeHi = std::max(edges[e].da, edges[e].db);
                    if (levelValue > edgeLo && levelValue < edgeHi) {
                        const double t = (levelValue - edges[e].da) / (edges[e].db - edges[e].da);
                        const Vector3 p = edges[e].pa + (edges[e].pb - edges[e].pa) * t;
                        if (crossCount < 2) crossings[crossCount] = p;
                        crossCount++;
                    }
                }
                if (crossCount == 2) {
                    pts.push_back(crossings[0].x); pts.push_back(crossings[0].y); pts.push_back(crossings[0].z);
                    pts.push_back(crossings[1].x); pts.push_back(crossings[1].y); pts.push_back(crossings[1].z);
                }
            }
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
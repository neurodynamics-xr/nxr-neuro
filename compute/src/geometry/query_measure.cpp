#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"

#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <iostream>

namespace {

using namespace nxr::geometry;

inline void requireVertexInRange(const SurfaceArrays& surface, int v, const char* fnName) {
    if (v < 0 || v >= surface.nV()) {
        throw nxr::core::Error(
            nxr::core::ErrorCode::InvalidInput,
            std::string(fnName) + ": vertex index out of range",
            "expected 0 <= v < " + std::to_string(surface.nV()));
    }
}

}  // namespace

namespace nxr::geometry::query {

PointLocus point(const SurfaceArrays& surface, int v) {
    requireVertexInRange(surface, v, "query::point");
    return PointLocus{v};
}

PolylineLocus line(const SurfaceArrays& surface, int vStart, int vEnd) {
    // tracePath validates its own inputs and runs the edge-flip
    // geodesic. Wrap the [N, 3] matrix in the locus type.
    return PolylineLocus{tracePath(surface, vStart, vEnd)};
}

RegionLocus area(const SurfaceArrays& surface, int v, double level) {
    geometry::IntrinsicSurface gs(surface);
    requireVertexInRange(surface, v, "query::area");
    if (!(level > 0.0)) {
        throw nxr::core::Error(
            nxr::core::ErrorCode::InvalidInput,
            "query::area: level must be > 0");
    }

    // Heat-method geodesic distance from v (one solver per call; see the header).
    Eigen::VectorXd distances = geometry::heat(surface, std::vector<int>{v});

    // Walk faces; a face is in the region iff all 3 of its vertices
    // satisfy distance <= level. Conservative (under-counts faces that
    // straddle the boundary) but well-defined.
    auto& mesh = gs.inputMesh();
    RegionLocus out;
    out.faces.reserve(mesh.nFaces());
    for (geometrycentral::surface::Face f : mesh.faces()) {
        bool allInside = true;
        for (geometrycentral::surface::Vertex u : f.adjacentVertices()) {
            const int uIdx = static_cast<int>(u.getIndex());
            if (distances(uIdx) > level) {
                allInside = false;
                break;
            }
        }
        if (allInside) {
            out.faces.push_back(static_cast<int>(f.getIndex()));
        }
    }
    return out;
}

}  // namespace nxr::geometry::query

namespace nxr::geometry::measure {

Eigen::Vector3d point(const SurfaceArrays& surface, int v) {
    geometry::IntrinsicSurface gs(surface);
    requireVertexInRange(surface, v, "measure::point");
    auto& geom = gs.embedded();
    geom.requireVertexPositions();
    auto& mesh = gs.inputMesh();
    const auto pos = geom.vertexPositions[mesh.vertex(static_cast<size_t>(v))];
    return Eigen::Vector3d(pos.x, pos.y, pos.z);
}

double line(const query::PolylineLocus& locus) {
    const auto& P = locus.points;
    if (P.rows() < 2) return 0.0;
    double L = 0.0;
    for (int i = 1; i < P.rows(); ++i) {
        L += (P.row(i) - P.row(i - 1)).norm();
    }
    return L;
}

double line(const SurfaceArrays& surface, int vStart, int vEnd) {
    return line(query::line(surface, vStart, vEnd));
}

double area(const SurfaceArrays& surface, const query::RegionLocus& locus) {
    geometry::IntrinsicSurface gs(surface);
    auto& geom = gs.embedded();
    geom.requireFaceAreas();
    auto& mesh = gs.inputMesh();
    double total = 0.0;
    for (int fIdx : locus.faces) {
        total += geom.faceAreas[mesh.face(static_cast<size_t>(fIdx))];
    }
    return total;
}

double area(const SurfaceArrays& surface, int v, double level) {
    return area(surface, query::area(surface, v, level));
}

}  // namespace nxr::geometry::measure

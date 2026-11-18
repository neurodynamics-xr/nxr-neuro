#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/synthesis/fields.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include <iostream>

namespace nxr::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── The heat method: geometry-central's solver, built inside the call ──

namespace {
Eigen::VectorXd distancesFrom(HeatMethodDistanceSolver& solver, ManifoldSurfaceMesh& solverMesh, const std::vector<int>& sourceVertices, int nV) {
    // Source vertices on the mesh the solver lives on (the intrinsic triangulation's vertices carry the input's indices)
    std::vector<Vertex> sources;
    sources.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        if (idx < 0 || idx >= nV) throw Error(ErrorCode::InvalidInput, "heat: a source vertex is out of range");
        sources.push_back(solverMesh.vertex(static_cast<size_t>(idx)));
    }
    VertexData<double> distances = solver.computeDistance(sources);
    Eigen::VectorXd result(nV);
    for (size_t i = 0; i < static_cast<size_t>(nV); ++i) result(static_cast<int>(i)) = distances[i];
    return result;
}
}  // namespace

Eigen::MatrixXd heat(const SurfaceArrays& surface, const std::vector<std::vector<int>>& sourceSets, double tCoef, HeatMesh mesh) {
    IntrinsicSurface gs(surface);
    // geometry-central's heat method on the intrinsic Delaunay triangulation (default; it reads the surface's mass, the input's mixed Voronoi) or on
    // the input mesh (HeatMesh::Input). Vertex indices are the input's in both.
    const bool intrinsic = mesh == HeatMesh::Intrinsic;
    HeatMethodDistanceSolver solver(intrinsic ? static_cast<IntrinsicGeometryInterface&>(gs.geometry()) : gs.embedded(), tCoef);
    ManifoldSurfaceMesh& solverMesh = intrinsic ? gs.mesh() : gs.inputMesh();
    const int nV = surface.nV();
    Eigen::MatrixXd out(nV, static_cast<Eigen::Index>(sourceSets.size()));
    for (size_t j = 0; j < sourceSets.size(); ++j) out.col(static_cast<Eigen::Index>(j)) = distancesFrom(solver, solverMesh, sourceSets[j], nV);
    return out;
}

Eigen::VectorXd heat(const SurfaceArrays& surface, const std::vector<int>& sourceVertices, double tCoef, HeatMesh mesh) {
    return heat(surface, std::vector<std::vector<int>>{sourceVertices}, tCoef, mesh).col(0);
}

}  // namespace nxr::geometry

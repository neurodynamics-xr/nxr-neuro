#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/signed_heat_method.h"
#include "geometrycentral/surface/surface_point.h"

#include <iostream>
#include <vector>

namespace nxr::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {
Eigen::VectorXd signedHeatOn(
    geometrycentral::surface::SignedHeatSolver& solver,
    ManifoldSurfaceMesh& mesh,
    int nV,
    const std::vector<int>& curveVertices,
    bool isLoop,
    SignedHeatLevelSet levelSet
) {

    if (curveVertices.size() < 2) {
        throw Error(ErrorCode::InvalidInput,
            "signedHeat: curve must have at least 2 vertices");
    }

    Curve curve;
    curve.isSigned = true;
    curve.nodes.reserve(curveVertices.size() + (isLoop ? 1 : 0));
    for (int idx : curveVertices) {
        if (idx < 0 || idx >= nV) {
            throw Error(ErrorCode::InvalidInput,
                "curve vertex index out of range");
        }
        curve.nodes.emplace_back(mesh.vertex(static_cast<size_t>(idx)));
    }
    if (isLoop) {
        curve.nodes.emplace_back(
            mesh.vertex(static_cast<size_t>(curveVertices.front())));
    }

    SignedHeatOptions opts;
    switch (levelSet) {
        case SignedHeatLevelSet::None:
            opts.levelSetConstraint = LevelSetConstraint::None; break;
        case SignedHeatLevelSet::ZeroSet:
            opts.levelSetConstraint = LevelSetConstraint::ZeroSet; break;
        case SignedHeatLevelSet::Multiple:
            opts.levelSetConstraint = LevelSetConstraint::Multiple; break;
    }

    std::vector<Curve> curves{std::move(curve)};
    VertexData<double> distances = solver.computeDistance(curves, opts);

    Eigen::VectorXd out(nV);
    for (Vertex v : mesh.vertices()) {
        out(static_cast<int>(v.getIndex())) = distances[v];
    }

    std::cout << "[signed_heat] " << curveVertices.size()
              << "-vertex curve, isLoop=" << isLoop
              << ", range [" << out.minCoeff() << ", " << out.maxCoeff() << "]"
              << std::endl;
    return out;
}

}  // namespace

Eigen::MatrixXd signedHeat(const SurfaceArrays& surface, const std::vector<std::vector<int>>& curves, bool isLoop, SignedHeatLevelSet levelSet, double tCoef) {
    IntrinsicSurface gs(surface);
    geometrycentral::surface::SignedHeatSolver solver(gs.embedded(), tCoef);
    Eigen::MatrixXd out(surface.nV(), static_cast<Eigen::Index>(curves.size()));
    for (size_t j = 0; j < curves.size(); ++j)
        out.col(static_cast<Eigen::Index>(j)) = signedHeatOn(solver, gs.inputMesh(), surface.nV(), curves[j], isLoop, levelSet);
    return out;
}

Eigen::VectorXd signedHeat(const SurfaceArrays& surface, const std::vector<int>& curveVertices, bool isLoop, SignedHeatLevelSet levelSet, double tCoef) {
    return signedHeat(surface, std::vector<std::vector<int>>{curveVertices}, isLoop, levelSet, tCoef).col(0);
}

}  // namespace nxr::geometry

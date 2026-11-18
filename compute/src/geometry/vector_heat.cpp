#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/facets.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/surface_point.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nxr::geometry::transport {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Solver impl (PIMPL) ───────────────────────────────────
//
// Owns the geometry-central VectorHeatMethodSolver plus a
// reference to the Manifold so wrappers can resolve
// vertex / face indices and reach the embedded geometry for
// tangent-basis lifting.

class VectorHeatSolverImpl {
public:
    // The solver lives on the INTRINSIC Delaunay triangulation (geometry-central's gauge, the one every tangent operator uses); its vertices carry the
    // input's indices. `basis` is the 3D realisation of that gauge (IntrinsicSurface::tangentBasis).
    VectorHeatSolverImpl(Manifold& c, double tCoef)
        : m(c), tCoef(tCoef), solver(c.intrinsicSurface().geometry(), tCoef), basis(c.intrinsicSurface().tangentBasis()) {}
    Manifold& m;
    double tCoef;
    VectorHeatMethodSolver solver;
    TangentBasis basis;
    std::unique_ptr<VectorHeatMethodSolver> inputSolver;   // findCenter only: it returns a point of the INPUT mesh, so it runs there
    Vertex vertex(int i) { return m.intrinsicSurface().mesh().vertex(static_cast<size_t>(i)); }
    std::array<Vector3, 2> frame(int i) const {
        const size_t k = 3 * static_cast<size_t>(i);
        return {Vector3{basis.e1[k], basis.e1[k + 1], basis.e1[k + 2]}, Vector3{basis.e2[k], basis.e2[k + 1], basis.e2[k + 2]}};
    }
};

VectorHeatSolver::VectorHeatSolver(Manifold& m, double tCoef)
    : impl_(std::make_unique<VectorHeatSolverImpl>(m, tCoef)) {}

VectorHeatSolver::~VectorHeatSolver() = default;

VectorHeatSolverImpl& VectorHeatSolver::impl()  { return *impl_; }
Manifold&       VectorHeatSolver::m()   { return impl_->m; }

// ── Helpers ───────────────────────────────────────────────

namespace {

void checkVertexInRange(int idx, int nV) {
    if (idx < 0 || idx >= nV) {
        throw Error(ErrorCode::InvalidInput,
            "vertex index out of range",
            "expected 0 <= idx < " + std::to_string(nV));
    }
}

// Project a 3D world vector into a vertex's intrinsic tangent
// basis. Anything in the normal direction is dropped.
Vector2 worldToTangent(const Vector3& w,
                       const std::array<Vector3, 2>& basis) {
    return {dot(w, basis[0]), dot(w, basis[1])};
}

// Lift a 2D tangent vector back to 3D world using the same basis.
Vector3 tangentToWorld(const Vector2& t,
                       const std::array<Vector3, 2>& basis) {
    return t.x * basis[0] + t.y * basis[1];
}

} // namespace

// ── Tangent vector transport ──────────────────────────────

Eigen::MatrixXd parallel(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors
) {
    auto& s = solver.impl();
    int nV = s.m.nV();

    if (static_cast<int>(sourceVertices.size()) != sourceVectors.rows()) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVertices length must match sourceVectors row count");
    }
    if (sourceVectors.cols() != 3) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVectors must be Nx3 (world-space tangent vectors)");
    }

    std::vector<std::tuple<Vertex, Vector2>> sources;
    sources.reserve(sourceVertices.size());
    for (std::size_t i = 0; i < sourceVertices.size(); i++) {
        checkVertexInRange(sourceVertices[i], nV);
        Vertex v = s.vertex(sourceVertices[i]);
        Vector3 w{sourceVectors(i, 0), sourceVectors(i, 1), sourceVectors(i, 2)};
        Vector2 t = worldToTangent(w, s.frame(sourceVertices[i]));
        sources.emplace_back(v, t);
    }

    VertexData<Vector2> transported = s.solver.transportTangentVectors(sources);

    Eigen::MatrixXd out(nV, 3);
    for (int vi = 0; vi < nV; ++vi) {
        Vector3 w = tangentToWorld(transported[static_cast<size_t>(vi)], s.frame(vi));
        out(vi, 0) = w.x;
        out(vi, 1) = w.y;
        out(vi, 2) = w.z;
    }

    std::cout << "[vector_heat] transported " << sourceVertices.size()
              << " sources to " << nV << " vertices" << std::endl;
    return out;
}

// ── Scalar extension ──────────────────────────────────────

Eigen::VectorXd extendScalar(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues
) {
    auto& s = solver.impl();
    int nV = s.m.nV();

    if (static_cast<int>(sourceVertices.size()) != sourceValues.size()) {
        throw Error(ErrorCode::InvalidInput,
            "sourceVertices and sourceValues must have the same length");
    }

    std::vector<std::tuple<Vertex, double>> sources;
    sources.reserve(sourceVertices.size());
    for (std::size_t i = 0; i < sourceVertices.size(); i++) {
        checkVertexInRange(sourceVertices[i], nV);
        sources.emplace_back(
            s.vertex(sourceVertices[i]),
            sourceValues[static_cast<Eigen::Index>(i)]);
    }

    VertexData<double> extended = s.solver.extendScalar(sources);

    Eigen::VectorXd out(nV);
    for (int vi = 0; vi < nV; ++vi) out(vi) = extended[static_cast<size_t>(vi)];
    return out;
}

// ── Log map ───────────────────────────────────────────────

LogMapResult logMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy
) {
    auto& s = solver.impl();
    int nV = s.m.nV();
    checkVertexInRange(sourceVertex, nV);

    Vertex src = s.vertex(sourceVertex);

    geometrycentral::surface::LogMapStrategy gcStrategy;
    switch (strategy) {
        case LogMapStrategy::VectorHeat:
            gcStrategy = geometrycentral::surface::LogMapStrategy::VectorHeat;
            break;
        case LogMapStrategy::AffineLocal:
            gcStrategy = geometrycentral::surface::LogMapStrategy::AffineLocal;
            break;
        case LogMapStrategy::AffineAdaptive:
            gcStrategy = geometrycentral::surface::LogMapStrategy::AffineAdaptive;
            break;
        default:
            throw Error(ErrorCode::InvalidInput, "unknown LogMapStrategy");
    }

    VertexData<Vector2> logMap = s.solver.computeLogMap(src, gcStrategy);

    LogMapResult out;
    out.logCoords.resize(nV, 2);
    for (int vi = 0; vi < nV; ++vi) {
        out.logCoords(vi, 0) = logMap[static_cast<size_t>(vi)].x;
        out.logCoords(vi, 1) = logMap[static_cast<size_t>(vi)].y;
    }
    const auto srcBasis = s.frame(sourceVertex);
    out.sourceE1 << srcBasis[0].x, srcBasis[0].y, srcBasis[0].z;
    out.sourceE2 << srcBasis[1].x, srcBasis[1].y, srcBasis[1].z;
    return out;
}

// ── Karcher mean / find center ────────────────────────────

Eigen::Vector3d findCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p
) {
    auto& s = solver.impl();
    auto& mesh = s.m.mesh();
    auto& geom = s.m.geometry();
    int nV = s.m.nV();

    if (sourceVertices.empty()) {
        throw Error(ErrorCode::InvalidInput,
            "findCenter requires at least one source vertex");
    }

    std::vector<Vertex> srcs;
    srcs.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        checkVertexInRange(idx, nV);
        srcs.push_back(mesh.vertex(static_cast<size_t>(idx)));
    }

    // findCenter requires a ManifoldSurfaceMesh — Manifold stores
    // exactly that, but the base reference is SurfaceMesh. Cast down.
    auto* manifold = dynamic_cast<ManifoldSurfaceMesh*>(&mesh);
    if (!manifold) {
        throw Error(ErrorCode::NonManifold,
            "findCenter requires a manifold surface mesh");
    }

    if (!s.inputSolver) s.inputSolver = std::make_unique<VectorHeatMethodSolver>(geom, s.tCoef);
    SurfacePoint center = findCenter(*manifold, geom, *s.inputSolver, srcs, p);

    // Resolve SurfacePoint to a 3D world position via barycentric blend.
    Vector3 pos = center.interpolate(geom.inputVertexPositions);
    Eigen::Vector3d out;
    out << pos.x, pos.y, pos.z;
    return out;
}

} // namespace nxr::geometry::transport
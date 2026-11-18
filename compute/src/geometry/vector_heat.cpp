#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"

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

// ── The solver, built inside each call ────────────────────
//
// geometry-central's VectorHeatMethodSolver on the surface rebuilt from its arrays; it lives for one call, which may answer many queries (the batch
// forms below). geometry-central keeps its own factorizations inside the solver.

namespace {
struct Solver {
    // The solver lives on the INTRINSIC Delaunay triangulation (geometry-central's gauge, the one every tangent operator uses); its vertices carry the
    // input's indices. `basis` is the 3D realisation of that gauge (IntrinsicSurface::tangentBasis).
    Solver(const SurfaceArrays& surface, double tCoef)
        : gs(surface), nV(surface.nV()), tCoef(tCoef), solver(gs.geometry(), tCoef), basis(gs.tangentBasis()) {}
    IntrinsicSurface gs;
    int nV;
    double tCoef;
    VectorHeatMethodSolver solver;
    TangentBasis basis;
    Vertex vertex(int i) { return gs.mesh().vertex(static_cast<size_t>(i)); }
    std::array<Vector3, 2> frame(int i) const {
        const size_t k = 3 * static_cast<size_t>(i);
        return {Vector3{basis.e1[k], basis.e1[k + 1], basis.e1[k + 2]}, Vector3{basis.e2[k], basis.e2[k + 1], basis.e2[k + 2]}};
    }
};

}  // namespace

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

static Eigen::MatrixXd parallelOn(
    Solver& s,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors
) {
    int nV = s.nV;

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

static Eigen::VectorXd extendScalarOn(
    Solver& s,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues
) {
    int nV = s.nV;

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

static LogMapResult logMapOn(
    Solver& s,
    int sourceVertex,
    LogMapStrategy strategy
) {
    int nV = s.nV;
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
    const SurfaceArrays& surface,
    const std::vector<int>& sourceVertices,
    int p,
    double tCoef
) {
    IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nV = surface.nV();

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

    // findCenter returns a point of the INPUT mesh, so its solver runs there
    VectorHeatMethodSolver inputSolver(geom, tCoef);
    SurfacePoint center = geometrycentral::surface::findCenter(mesh, geom, inputSolver, srcs, p);

    // Resolve SurfacePoint to a 3D world position via barycentric blend.
    Vector3 pos = center.interpolate(geom.inputVertexPositions);
    Eigen::Vector3d out;
    out << pos.x, pos.y, pos.z;
    return out;
}

// ── The public forms: one solver per call, every query of the call on it ──

Eigen::MatrixXd parallel(const SurfaceArrays& surface, const std::vector<int>& sourceVertices, const Eigen::MatrixXd& sourceVectors, double tCoef) {
    Solver s(surface, tCoef);
    return parallelOn(s, sourceVertices, sourceVectors);
}

std::vector<Eigen::MatrixXd> parallel(const SurfaceArrays& surface, const std::vector<std::vector<int>>& sourceVertices,
                                      const std::vector<Eigen::MatrixXd>& sourceVectors, double tCoef) {
    if (sourceVertices.size() != sourceVectors.size()) throw Error(ErrorCode::InvalidInput, "parallel: one vector set per source set");
    Solver s(surface, tCoef);
    std::vector<Eigen::MatrixXd> out;
    out.reserve(sourceVertices.size());
    for (size_t i = 0; i < sourceVertices.size(); ++i) out.push_back(parallelOn(s, sourceVertices[i], sourceVectors[i]));
    return out;
}

Eigen::VectorXd extendScalar(const SurfaceArrays& surface, const std::vector<int>& sourceVertices, const Eigen::VectorXd& sourceValues, double tCoef) {
    Solver s(surface, tCoef);
    return extendScalarOn(s, sourceVertices, sourceValues);
}

LogMapResult logMap(const SurfaceArrays& surface, int sourceVertex, LogMapStrategy strategy, double tCoef) {
    Solver s(surface, tCoef);
    return logMapOn(s, sourceVertex, strategy);
}

std::vector<LogMapResult> logMap(const SurfaceArrays& surface, const std::vector<int>& sourceVertices, LogMapStrategy strategy, double tCoef) {
    Solver s(surface, tCoef);
    std::vector<LogMapResult> out;
    out.reserve(sourceVertices.size());
    for (int v : sourceVertices) out.push_back(logMapOn(s, v, strategy));
    return out;
}

}  // namespace nxr::geometry::transport

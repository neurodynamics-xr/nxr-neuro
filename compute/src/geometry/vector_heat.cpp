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
#include <cmath>
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

// ── Log map at a point of an input face (the blend of the corners' log maps; see the header) ──

namespace {
// The rotation taking the unit vector a to the unit vector b (Rodrigues; the identity when they coincide).
Eigen::Matrix3d rotationBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d v = a.cross(b);
    const double c = a.dot(b);
    if (c < -0.999999) throw Error(ErrorCode::InvalidInput, "log map at a face point: a corner's normal is opposite the face's");
    Eigen::Matrix3d K;
    K << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return Eigen::Matrix3d::Identity() + K + K * K / (1.0 + c);
}
Eigen::Vector3d toEigen(const Vector3& v) { return {v.x, v.y, v.z}; }
}  // namespace

static FaceLogMapResult logMapAtFacePoint(Solver& s, const SurfaceArrays& surface, const FacePoint& pt, LogMapStrategy strategy) {
    if (pt.face < 0 || pt.face >= surface.nF()) {
        throw Error(ErrorCode::InvalidInput, "face index out of range", "expected 0 <= face < " + std::to_string(surface.nF()));
    }
    double sum = 0;
    for (double w : pt.barycentric) {
        if (!std::isfinite(w) || w < 0) throw Error(ErrorCode::InvalidInput, "barycentric weights must be finite and non-negative");
        sum += w;
    }
    if (std::abs(sum - 1.0) > 1e-9) throw Error(ErrorCode::InvalidInput, "barycentric weights must sum to 1 (within 1e-9)");

    int corner[3];
    Eigen::Vector3d p[3];
    for (int k = 0; k < 3; ++k) {
        corner[k] = surface.faces(pt.face, k);
        p[k] = surface.positions.row(corner[k]).transpose();
    }
    Eigen::Vector3d nF = (p[1] - p[0]).cross(p[2] - p[0]);
    if (!(nF.norm() > 0)) throw Error(ErrorCode::InvalidInput, "log map at a face point: the face is degenerate");
    nF.normalize();

    int big = 0;
    for (int k = 1; k < 3; ++k) if (pt.barycentric[k] > pt.barycentric[big]) big = k;

    FaceLogMapResult out;
    out.logCoords = Eigen::MatrixXd::Zero(s.nV, 2);
    out.sourcePoint.setZero();
    for (int k = 0; k < 3; ++k) out.sourcePoint += pt.barycentric[k] * p[k];

    // the common frame: the heaviest corner's, rotated into the face's plane
    {
        const auto f = s.frame(corner[big]);
        const Eigen::Vector3d e1 = toEigen(f[0]), e2 = toEigen(f[1]);
        const Eigen::Matrix3d R = rotationBetween(e1.cross(e2).normalized(), nF);
        out.sourceE1 = R * e1;
        out.sourceE2 = R * e2;
        out.sourceNormal = nF;
    }
    for (int k = 0; k < 3; ++k) {
        if (pt.barycentric[k] == 0) continue;
        const LogMapResult lm = logMapOn(s, corner[k], strategy);
        const auto f = s.frame(corner[k]);
        const Eigen::Vector3d e1 = toEigen(f[0]), e2 = toEigen(f[1]);
        const Eigen::Matrix3d R = rotationBetween(e1.cross(e2).normalized(), nF);
        // the corner's tangent vector (x e1 + y e2) carried into the face plane, read in the common frame
        const Eigen::Vector3d a = R * e1, b = R * e2;
        const double a1 = a.dot(out.sourceE1), a2 = a.dot(out.sourceE2), b1 = b.dot(out.sourceE1), b2 = b.dot(out.sourceE2);
        const double w = pt.barycentric[k];
        out.logCoords.col(0) += w * (a1 * lm.logCoords.col(0) + b1 * lm.logCoords.col(1));
        out.logCoords.col(1) += w * (a2 * lm.logCoords.col(0) + b2 * lm.logCoords.col(1));
    }
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

FaceLogMapResult logMap(const SurfaceArrays& surface, const FacePoint& point, LogMapStrategy strategy, double tCoef) {
    Solver s(surface, tCoef);
    return logMapAtFacePoint(s, surface, point, strategy);
}

std::vector<FaceLogMapResult> logMap(const SurfaceArrays& surface, const std::vector<FacePoint>& points, LogMapStrategy strategy, double tCoef) {
    Solver s(surface, tCoef);
    std::vector<FaceLogMapResult> out;
    out.reserve(points.size());
    for (const FacePoint& p : points) out.push_back(logMapAtFacePoint(s, surface, p, strategy));
    return out;
}

}  // namespace nxr::geometry::transport

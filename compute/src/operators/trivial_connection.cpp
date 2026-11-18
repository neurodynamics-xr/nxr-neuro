/**
 * direction_field.cpp — Trivial Connections + streamline tracing.
 *
 * Ports geometry-processing-js projects/direction-field-design/
 *   - trivial-connections.js
 *   - streamlines.js
 *
 * Reference: "Trivial Connections on Discrete Surfaces"
 *            Crane, Desbrun, Schröder (2010)
 *
 * Algorithm summary:
 *   Given singularity indices σ_v at vertices (Gauss-Bonnet: Σ σ_v = χ),
 *   solve for a connection φ on edges such that a vector transported
 *   around any vertex picks up the prescribed angular defect.
 *
 *     φ = δβ + γ
 *
 *   where:
 *     - δβ is the co-exact component, β solves A β = -K + 2π σ
 *       (A = d0ᵀ ⋆₁ d0 is the 0-form Laplacian, K is Gaussian curvature)
 *     - γ is the harmonic component, zero for simply-connected surfaces
 *       (we currently don't implement the harmonic component — it only
 *        matters for genus > 0)
 */

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>
#include <queue>
#include <cmath>
#include <vector>
#include <unordered_map>

namespace nxr::operators::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using geometry::SurfaceArrays;

// ─────────────────────────────────────────────────────────────
// Helper: 2D orthonormal basis on a face from its normal and
// first halfedge direction.
// ─────────────────────────────────────────────────────────────

static std::array<Vector3, 2> faceOrthonormalBasis(
    VertexPositionGeometry& geom, Face f
) {
    geom.requireFaceNormals();

    Halfedge h = f.halfedge();
    Vector3 e0 = geom.inputVertexPositions[h.tipVertex()] -
                 geom.inputVertexPositions[h.tailVertex()];
    e0 = e0.normalize();

    Vector3 N = geom.faceNormals[f];
    Vector3 e1 = cross(N, e0).normalize();

    return {e0, e1};
}

// ─────────────────────────────────────────────────────────────
// transportNoRotation: Crane's parallel transport formula.
// αⱼ = αᵢ - θᵢⱼ + θⱼᵢ
// ─────────────────────────────────────────────────────────────

static double transportNoRotation(
    VertexPositionGeometry& geom, Halfedge h
) {
    if (!h.isInterior() || !h.twin().isInterior()) return 0.0;

    // Edge vector (tail → tip) of this halfedge
    Vector3 u = geom.inputVertexPositions[h.tipVertex()] -
                geom.inputVertexPositions[h.tailVertex()];

    // Bases for the two faces sharing this halfedge
    auto [e1, e2] = faceOrthonormalBasis(geom, h.face());
    double thetaIJ = std::atan2(dot(u, e2), dot(u, e1));

    auto [f1, f2] = faceOrthonormalBasis(geom, h.twin().face());
    double thetaJI = std::atan2(dot(u, f2), dot(u, f1));

    return -thetaIJ + thetaJI;
}

// ─────────────────────────────────────────────────────────────
// Solve Aβ = -K + 2π σ for the co-exact scalar potential β,
// then return δβ = ⋆₁ d₀ β (as an edge 1-form).
// Public primitive — see include/nxr/compute.h for contract.
// ─────────────────────────────────────────────────────────────

Eigen::VectorXd computeTrivialConnection(
    const SurfaceArrays& surface,
    const operators::DECOperators& dec,
    const nxr::operators::CholeskyFactor& exactFactor,
    const std::map<int, double>& singularityMap
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nV = surface.nV();

    geom.requireVertexGaussianCurvatures();

    Eigen::VectorXd rhs(nV);
    for (Vertex v : mesh.vertices()) {
        int i = static_cast<int>(v.getIndex());
        double K = geom.vertexGaussianCurvatures[v];
        double sigma = 0.0;
        auto it = singularityMap.find(i);
        if (it != singularityMap.end()) sigma = it->second;
        rhs(i) = -K + 2.0 * M_PI * sigma;
    }

    const auto& llt = exactFactor;
    Eigen::VectorXd betaTilde = llt.solve(rhs);
    return dec.hodge1 * (dec.d0 * betaTilde);
}

// ─────────────────────────────────────────────────────────────
// Propagate angles across faces via dual-spanning-tree BFS,
// following the connection φ on edges. Returns per-face angle
// α(f) that defines the direction field.
// ─────────────────────────────────────────────────────────────

static Eigen::VectorXd propagateAngles(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& phi,   // connection 1-form on edges
    const operators::DECOperators& /*dec*/
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nF = surface.nF();

    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(nF);
    std::vector<bool> visited(nF, false);

    // BFS from face 0
    std::queue<Face> queue;
    Face root = mesh.face(0);
    visited[0] = true;
    alpha(0) = 0.0;
    queue.push(root);

    while (!queue.empty()) {
        Face f = queue.front();
        queue.pop();
        int fi = static_cast<int>(f.getIndex());
        double alphaI = alpha(fi);

        for (Halfedge h : f.adjacentHalfedges()) {
            if (!h.twin().isInterior()) continue;
            Face g = h.twin().face();
            int gi = static_cast<int>(g.getIndex());
            if (visited[gi]) continue;

            double transport = transportNoRotation(geom, h);

            // Sign of the connection φ on this edge, depending on halfedge orientation
            Edge e = h.edge();
            double sign = (h == e.halfedge()) ? 1.0 : -1.0;
            double phiEdge = phi(static_cast<int>(e.getIndex()));

            double alphaJ = alphaI + transport + sign * phiEdge;
            alpha(gi) = alphaJ;
            visited[gi] = true;
            queue.push(g);
        }
    }

    return alpha;
}

// ─────────────────────────────────────────────────────────────
// Propagate angles across vertices via primal-spanning-tree BFS,
// following the connection φ on edges and the Levi-Civita transport
// from geometry-central's transportVectorsAlongHalfedge.
// Returns per-vertex angle α(v) that defines the vertex direction field.
// ─────────────────────────────────────────────────────────────

static Eigen::VectorXd propagateAnglesVertex(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& phi
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nV = surface.nV();

    geom.requireVertexIndices();
    geom.requireEdgeIndices();
    geom.requireTransportVectorsAlongHalfedge();

    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(nV);
    std::vector<bool> visited(nV, false);

    std::queue<Vertex> queue;
    Vertex root = mesh.vertex(0);
    visited[0] = true;
    alpha(0) = 0.0;
    queue.push(root);

    while (!queue.empty()) {
        Vertex vi = queue.front();
        queue.pop();
        int ii = static_cast<int>(vi.getIndex());

        for (Halfedge he : vi.outgoingHalfedges()) {
            if (!he.isInterior()) continue;
            Vertex vj = he.tipVertex();
            int jj = static_cast<int>(vj.getIndex());
            if (visited[jj]) continue;

            // LC transport angle from vi to vj along this halfedge
            const Vector2 rot = geom.transportVectorsAlongHalfedge[he];
            double transport = std::atan2(rot.y, rot.x);

            // φ sign for this halfedge
            Edge e = he.edge();
            double sign = (he == e.halfedge()) ? 1.0 : -1.0;
            double phiEdge = phi(static_cast<int>(geom.edgeIndices[e]));

            alpha(jj) = alpha(ii) + transport + sign * phiEdge;
            visited[jj] = true;
            queue.push(vj);
        }
    }

    return alpha;
}

// ─────────────────────────────────────────────────────────────
// Build per-face direction vectors from per-face angles.
// For each face f, the direction vector is:
//   u = cos(α(f)) e₁(f) + sin(α(f)) e₂(f)
// where {e₁, e₂} is the face tangent basis.
// ─────────────────────────────────────────────────────────────

static Eigen::MatrixXd buildFaceVectorsFromAngles(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& alpha
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nF = surface.nF();

    Eigen::MatrixXd out(nF, 3);
    for (Face f : mesh.faces()) {
        int fi = static_cast<int>(f.getIndex());
        auto [e1, e2] = faceOrthonormalBasis(geom, f);
        Vector3 u = e1 * std::cos(alpha(fi)) + e2 * std::sin(alpha(fi));
        out(fi, 0) = u.x;
        out(fi, 1) = u.y;
        out(fi, 2) = u.z;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────
// Build per-vertex direction vectors from per-vertex angles.
// For each vertex v, the direction vector is:
//   u = cos(α(v)) e₁(v) + sin(α(v)) e₂(v)
// where {e₁, e₂} is the vertex tangent basis.
// ─────────────────────────────────────────────────────────────

static Eigen::MatrixXd buildVertexVectorsFromAngles(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& alpha
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();
    int nV = surface.nV();

    geom.requireVertexTangentBasis();

    Eigen::MatrixXd out(nV, 3);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        auto basis = geom.vertexTangentBasis[v];
        Vector3 u = basis[0] * std::cos(alpha(vi)) + basis[1] * std::sin(alpha(vi));
        out(vi, 0) = u.x;
        out(vi, 1) = u.y;
        out(vi, 2) = u.z;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────
// trivial — public entry point
// ─────────────────────────────────────────────────────────────

DirectionFieldResult trivial(
    const SurfaceArrays& surface,
    const operators::DECOperators& dec,
    const nxr::operators::CholeskyFactor& exactFactor,
    const std::map<int, double>& singularityMap
) {
    geometry::IntrinsicSurface gs(surface);
    DirectionFieldResult result;
    result.eulerCharacteristic = 0.0;
    result.gaussBonnetSatisfied = false;

    // 1. Trivial connection φ = δβ
    result.connections = computeTrivialConnection(surface, dec, exactFactor, singularityMap);

    // 2. Euler characteristic from Gauss-Bonnet
    auto& geom = gs.embedded();
    geom.requireVertexGaussianCurvatures();
    double totalK = 0.0;
    for (Vertex v : gs.inputMesh().vertices())
        totalK += geom.vertexGaussianCurvatures[v];
    result.eulerCharacteristic = totalK / (2.0 * M_PI);

    // 3. Gauss-Bonnet check
    double sumSing = 0.0;
    for (const auto& [_, s] : singularityMap) sumSing += s;
    result.gaussBonnetSatisfied =
        std::abs(sumSing - result.eulerCharacteristic) < 1e-3;

    // 4. Propagate angles via dual spanning tree
    Eigen::VectorXd alpha = propagateAngles(surface, result.connections, dec);

    // 5. Build direction field vectors
    result.directionVectors = buildFaceVectorsFromAngles(surface, alpha);

    // 6. Orthogonal field: rotate by 90° on each face (α + π/2)
    Eigen::VectorXd alphaOrth = alpha.array() + M_PI / 2.0;
    result.orthogonalVectors = buildFaceVectorsFromAngles(surface, alphaOrth);

    // 7. Propagate angles on primal graph → per-vertex angles
    Eigen::VectorXd alphaV = propagateAnglesVertex(surface, result.connections);

    // 8. Build vertex direction field vectors
    result.vertexVectors = buildVertexVectorsFromAngles(surface, alphaV);
    Eigen::VectorXd alphaVOrth = alphaV.array() + M_PI / 2.0;
    result.vertexOrthogonalVectors = buildVertexVectorsFromAngles(surface, alphaVOrth);

    std::cout << "[direction_field] χ=" << result.eulerCharacteristic
              << ", Σσ=" << sumSing
              << ", Gauss-Bonnet "
              << (result.gaussBonnetSatisfied ? "OK" : "VIOLATED")
              << std::endl;

    return result;
}

} // namespace nxr::operators::connection


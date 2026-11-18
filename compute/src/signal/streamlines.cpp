// Streamline tracing through a face vector field (port of geometry-processing-js projects/direction-field-design/streamlines.js).
#include "nxr/geometry/mesh.h"
#include "nxr/signal/surface_calculus.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace nxr::signal {

using namespace geometrycentral;
using namespace geometrycentral::surface;
using geometry::SurfaceArrays;

namespace {
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
} // namespace



using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

// Barycentric coordinates of point p in triangle (p0, p1, p2).
// Returns (u, v, w) with u + v + w = 1.
struct Bary { double u, v, w; };
Bary barycentric(Vector3 p, Vector3 p0, Vector3 p1, Vector3 p2) {
    Vector3 e1 = p1 - p0;
    Vector3 e2 = p2 - p0;
    Vector3 ep = p - p0;
    double d00 = dot(e1, e1);
    double d01 = dot(e1, e2);
    double d11 = dot(e2, e2);
    double d20 = dot(ep, e1);
    double d21 = dot(ep, e2);
    double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-12) return {0, 0, 1};
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    return {1.0 - v - w, v, w};
}

// Line segment p1→p2 vs triangle edge pa→pb (in 2D face-local coords).
// Returns (crosses, t, intersection3D) where t ∈ [0,1] along p1→p2.
struct IntersectResult {
    bool crosses;
    Vector3 intersection;
};

IntersectResult lineIntersectsEdge2D(
    Vector3 p1, Vector3 p2, Vector3 pa, Vector3 pb,
    Vector3 basis1, Vector3 basis2
) {
    auto to2D = [&](Vector3 p) -> std::array<double, 2> {
        return {dot(p, basis1), dot(p, basis2)};
    };
    auto [p1x, p1y] = to2D(p1);
    auto [p2x, p2y] = to2D(p2);
    auto [pax, pay] = to2D(pa);
    auto [pbx, pby] = to2D(pb);

    double det = (pbx - pax) * (p1y - p2y) - (p1x - p2x) * (pby - pay);
    if (std::abs(det) < 1e-12) return {false, {}};

    double t = ((p1x - pax) * (p1y - p2y) - (p1y - pay) * (p1x - p2x)) / det;
    double s = ((pbx - pax) * (p1y - pay) - (pby - pay) * (p1x - pax)) / det;

    if (t >= 0 && t <= 1 && s >= 0 && s <= 1) {
        Vector3 inter = pa + (pb - pa) * t;
        return {true, inter};
    }
    return {false, {}};
}

} // namespace (anonymous)

StreamlineResult streamline(
    const geometry::SurfaceArrays& surface,
    const Eigen::MatrixXd& faceField,
    int numSeeds,
    double stepCoef,
    int maxSteps
) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();
    auto& geom = gs.embedded();

    geom.requireFaceNormals();
    geom.requireEdgeLengths();

    // Mean edge length
    double meanEdge = 0.0;
    int nE = static_cast<int>(gs.inputMesh().nEdges());
    for (Edge e : mesh.edges()) meanEdge += geom.edgeLengths[e];
    meanEdge /= std::max(1, nE);
    double stepSize = meanEdge * stepCoef;

    // Seed faces — quasi-uniform across mesh
    int nF = surface.nF();
    int seedStep = std::max(1, nF / (numSeeds * numSeeds));

    std::vector<double> outPts;  // flat [x,y,z, x,y,z, ...] endpoint pairs

    // Trace a single streamline forward from (face, position).
    // Returns a list of points along the streamline.
    auto trace = [&](Face seedFace, Vector3 seedPos, bool negate)
        -> std::vector<Vector3>
    {
        std::vector<Vector3> curve;
        curve.push_back(seedPos);
        Face currentFace = seedFace;
        Vector3 currentPos = seedPos;

        for (int step = 0; step < maxSteps; step++) {
            int fi = static_cast<int>(currentFace.getIndex());
            Vector3 dir{faceField(fi, 0), faceField(fi, 1), faceField(fi, 2)};
            if (negate) dir = -dir;
            double mag = dir.norm();
            if (mag < 1e-12) break;
            dir /= mag;

            Vector3 nextPos = currentPos + dir * stepSize;

            // Try to stay in current face
            Vector3 p0, p1, p2;
            Halfedge hf = currentFace.halfedge();
            p0 = geom.inputVertexPositions[hf.tailVertex()];
            p1 = geom.inputVertexPositions[hf.next().tailVertex()];
            p2 = geom.inputVertexPositions[hf.next().next().tailVertex()];
            Bary b = barycentric(nextPos, p0, p1, p2);
            double eps = -0.001;
            if (b.u >= eps && b.v >= eps && b.w >= eps) {
                currentPos = nextPos;
                curve.push_back(currentPos);
                continue;
            }

            // Need to cross an edge — find which one
            auto [basis1, basis2] = faceOrthonormalBasis(geom, currentFace);
            bool crossed = false;
            for (Halfedge h : currentFace.adjacentHalfedges()) {
                Vector3 pa = geom.inputVertexPositions[h.tailVertex()];
                Vector3 pb = geom.inputVertexPositions[h.tipVertex()];
                auto result = lineIntersectsEdge2D(
                    currentPos, nextPos, pa, pb, basis1, basis2);
                if (result.crosses) {
                    if (!h.twin().isInterior()) {
                        // Hit boundary, stop
                        break;
                    }
                    currentFace = h.twin().face();
                    currentPos = result.intersection + (nextPos - result.intersection) * 0.5;
                    // Just use intersection point to avoid overshooting into wrong face
                    currentPos = result.intersection;
                    curve.push_back(currentPos);
                    crossed = true;
                    break;
                }
            }
            if (!crossed) break;
        }
        return curve;
    };

    int nCurves = 0;
    for (int si = 0; si < nF; si += seedStep) {
        Face f = mesh.face(si);
        // Centroid
        Halfedge h = f.halfedge();
        Vector3 p0 = geom.inputVertexPositions[h.tailVertex()];
        Vector3 p1 = geom.inputVertexPositions[h.next().tailVertex()];
        Vector3 p2 = geom.inputVertexPositions[h.next().next().tailVertex()];
        Vector3 centroid = (p0 + p1 + p2) / 3.0;

        auto forward = trace(f, centroid, false);
        auto backward = trace(f, centroid, true);

        // Combine: reversed backward + forward (skip duplicate start)
        std::vector<Vector3> full(backward.rbegin(), backward.rend());
        for (size_t i = 1; i < forward.size(); i++) full.push_back(forward[i]);

        // Emit line segments
        for (size_t i = 0; i + 1 < full.size(); i++) {
            outPts.push_back(full[i].x);
            outPts.push_back(full[i].y);
            outPts.push_back(full[i].z);
            outPts.push_back(full[i + 1].x);
            outPts.push_back(full[i + 1].y);
            outPts.push_back(full[i + 1].z);
        }
        if (full.size() >= 2) nCurves++;
    }

    StreamlineResult result;
    int segCount = static_cast<int>(outPts.size() / 6);
    result.segmentCount = segCount;
    result.positions.resize(segCount * 2, 3);
    for (int i = 0; i < segCount * 2; i++) {
        result.positions(i, 0) = outPts[i * 3 + 0];
        result.positions(i, 1) = outPts[i * 3 + 1];
        result.positions(i, 2) = outPts[i * 3 + 2];
    }

    std::cout << "[streamlines] " << nCurves << " curves, "
              << segCount << " total segments" << std::endl;

    return result;
}

} // namespace nxr::signal

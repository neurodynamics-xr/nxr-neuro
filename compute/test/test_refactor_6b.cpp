// The gate of the toolbox refactor's geometry / operators / flow rows that merged duplicates (REFACTOR-PLAN 1.4 rows 19, 20, 24). Each REFERENCE below is the old
// function, copied verbatim from the code before the merge (the generator of the golden); the surviving function must give the same result on the same synthetic
// input: bitwise where the row says so, otherwise the difference is measured and bounded.
//   row 20  flow's float32 face gradient (std::hypot, no FMA flag) vs the forms port it now calls (jsHypot, -ffp-contract=off): the float32 values
//   row 24  the nearest-vertex search of select::track_crossings (its own k-d tree) vs geometry::nearest (the surviving k-d tree) and its brute force: the vertex chosen, ties to the lowest index
//   row 18  signal::isoline (now geometry-central's marching triangles) vs the Studio port it replaced: the segment SET (unordered, endpoints to 1e-12)
//   row 19  FaceOperators::vertexGradient (applied per face, then averaged) vs signal::vertexGradient (the assembled [V x V] operator): NOT bitwise, so both stay (named by how they differ)
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/mesh.h"
#include "nxr/signal/surface_calculus.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "nxr/geometry/surface_maps.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/select/track_crossings.h"
#include "nxr/signal/forms.h"
#include "nxr/signal/opticalflow_connection.h"
#include "nxr/signal/phase_gradient.h"

// REFERENCE (row 20): opticalflow_connection.cpp's assembleFaceGradient before the merge.
static std::vector<float> oldFlowFaceGradient(const float* position, const int32_t* index, int nF) {
    std::vector<float> grads(9 * static_cast<size_t>(nF), 0.0f);
    for (int f = 0; f < nF; ++f) {
        const int i0 = index[3 * f], i1 = index[3 * f + 1], i2 = index[3 * f + 2];
        const double ax = position[3 * i0], ay = position[3 * i0 + 1], az = position[3 * i0 + 2];
        const double bx = position[3 * i1], by = position[3 * i1 + 1], bz = position[3 * i1 + 2];
        const double px = position[3 * i2], py = position[3 * i2 + 1], pz = position[3 * i2 + 2];
        const double e1x = bx - ax, e1y = by - ay, e1z = bz - az, e2x = px - ax, e2y = py - ay, e2z = pz - az;
        const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const double twoA = std::hypot(nx, ny, nz);
        if (!(twoA > 1e-12)) continue;
        const double ux = nx / twoA, uy = ny / twoA, uz = nz / twoA;
        const size_t o = 9 * static_cast<size_t>(f);
        auto put = [&](int slot, double ex, double ey, double ez) {
            grads[o + 3 * static_cast<size_t>(slot)] = static_cast<float>((uy * ez - uz * ey) / twoA);
            grads[o + 3 * static_cast<size_t>(slot) + 1] = static_cast<float>((uz * ex - ux * ez) / twoA);
            grads[o + 3 * static_cast<size_t>(slot) + 2] = static_cast<float>((ux * ey - uy * ex) / twoA);
        };
        put(0, px - bx, py - by, pz - bz);
        put(1, ax - px, ay - py, az - pz);
        put(2, bx - ax, by - ay, bz - az);
    }
    return grads;
}


// REFERENCE (row 18): the isoline of signal/isolines.cpp before the switch to marching triangles (the Studio's isoline.ts port).
static std::vector<double> oldIsoline(nxr::geometry::Manifold& m, const Eigen::VectorXd& field, int numLevels, double lo, double hi) {
    using namespace geometrycentral; using namespace geometrycentral::surface;
    auto& mesh = m.mesh(); auto& geom = m.geometry();
    const double spacing = (hi - lo) / numLevels;
    std::vector<double> pts;
    for (Face f : mesh.faces()) {
        Halfedge h = f.halfedge();
        Vertex v0 = h.tailVertex(), v1 = h.next().tailVertex(), v2 = h.next().next().tailVertex();
        const double d[3] = {field(static_cast<int>(v0.getIndex())), field(static_cast<int>(v1.getIndex())), field(static_cast<int>(v2.getIndex()))};
        const Vector3 p[3] = {geom.inputVertexPositions[v0], geom.inputVertexPositions[v1], geom.inputVertexPositions[v2]};
        for (int level = 1; level < numLevels; level++) {
            const double lv = lo + level * spacing;
            Vector3 c[2]; int n = 0;
            for (int e = 0; e < 3; e++) {
                const int a = e, b = (e + 1) % 3;
                if (lv > std::min(d[a], d[b]) && lv < std::max(d[a], d[b])) {
                    const double t = (lv - d[a]) / (d[b] - d[a]);
                    const Vector3 q = p[a] + (p[b] - p[a]) * t;
                    if (n < 2) c[n] = q;
                    n++;
                }
            }
            if (n == 2) for (int k = 0; k < 2; ++k) { pts.push_back(c[k].x); pts.push_back(c[k].y); pts.push_back(c[k].z); }
        }
    }
    return pts;
}
// the segments as a sorted list of (endpoint, endpoint) with the lexicographically smaller endpoint first
static std::vector<std::array<double, 6>> segmentSet(const std::vector<double>& pts) {
    std::vector<std::array<double, 6>> out;
    for (size_t i = 0; i + 5 < pts.size(); i += 6) {
        std::array<double, 6> s{pts[i], pts[i + 1], pts[i + 2], pts[i + 3], pts[i + 4], pts[i + 5]};
        if (std::lexicographical_compare(s.begin() + 3, s.end(), s.begin(), s.begin() + 3)) { std::swap(s[0], s[3]); std::swap(s[1], s[4]); std::swap(s[2], s[5]); }
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {   // by the rounded start point first, so 1e-15 noise cannot reorder
        for (int k = 0; k < 6; ++k) { const double x = std::round(a[static_cast<size_t>(k)] * 1e9), y = std::round(b[static_cast<size_t>(k)] * 1e9); if (x != y) return x < y; }
        return false; });
    return out;
}

int main() {
    std::vector<double> V; std::vector<int32_t> F;
    icosphere(3, V, F);
    std::mt19937 gen(11); std::normal_distribution<double> nd;
    for (double& x : V) x *= 1.0 + 0.05 * nd(gen);            // a bumpy sphere: no two faces alike
    const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);


    std::cout << "=== row 18: isoline on geometry-central's marching triangles ===\n";
    {
        nxr::geometry::Manifold m(V.data(), nV, F.data(), nF);
        Eigen::VectorXd field(nV);
        for (int i = 0; i < nV; ++i) field[i] = std::sin(4 * V[3 * i]) + 0.7 * V[3 * i + 1] * V[3 * i + 2] + 0.01 * nd(gen);
        const int levels = 9;
        const double lo = field.minCoeff(), hi = field.maxCoeff();
        const nxr::signal::IsolineResult now = nxr::signal::isoline(m, field, levels, lo, hi);
        const std::vector<double> old = oldIsoline(m, field, levels, lo, hi);
        std::vector<double> nowPts(static_cast<size_t>(now.positions.rows() * 3));
        for (Eigen::Index i = 0; i < now.positions.rows(); ++i) for (int c = 0; c < 3; ++c) nowPts[static_cast<size_t>(3 * i + c)] = now.positions(i, c);
        const auto a = segmentSet(old), b = segmentSet(nowPts);
        std::cout << "  segments: old " << a.size() << ", new " << b.size() << "\n";
        bool same = a.size() == b.size(); double worst = 0;
        for (size_t i = 0; same && i < a.size(); ++i) for (size_t k = 0; k < 6; ++k) worst = std::max(worst, std::abs(a[i][k] - b[i][k]));
        EXPECT(same, "the same number of segments");
        EXPECT(same && worst < 1e-12, "the same segment set to 1e-12 (worst " << worst << ")");
    }

    std::cout << "=== row 20: the flow's face gradient is the forms port's ===\n";
    {
        std::vector<float> P(V.begin(), V.end());
        const std::vector<float> old = oldFlowFaceGradient(P.data(), F.data(), nF);
        const std::vector<float> now = nxr::signal::connection::assembleFaceGradient(P.data(), F.data(), nV, nF).grads;
        size_t differ = 0; double worst = 0;
        for (size_t i = 0; i < old.size(); ++i) { if (old[i] != now[i]) ++differ; worst = std::max(worst, std::abs(static_cast<double>(old[i]) - now[i]) / std::max(1e-30, std::abs(static_cast<double>(old[i])))); }
        std::cout << "  float32 entries that differ: " << differ << " of " << old.size() << ", worst relative " << worst << "\n";
        EXPECT(now.size() == old.size(), "same size");
        EXPECT(worst < 2e-7, "equal to float32 rounding (the last bit of a float32 at most)");
        const std::vector<float> forms = nxr::signal::assembleFaceGradient(P.data(), F.data(), nV, nF).grads;
        EXPECT(forms == now, "the flow's grads are the forms port's, bit for bit");
    }

    std::cout << "=== row 24: one nearest-vertex search ===\n";
    {
        Eigen::MatrixXd pos(nV, 3);
        for (int i = 0; i < nV; ++i) for (int c = 0; c < 3; ++c) pos(i, c) = V[3 * i + c];
        Eigen::MatrixXd q(400, 3);
        for (int i = 0; i < 400; ++i) for (int c = 0; c < 3; ++c) q(i, c) = 1.2 * nd(gen);
        // ties: queries at the lattice's centre of two coincident vertices and on exact vertices
        const std::vector<std::int32_t> got = nxr::select::track_crossings::nearestVertex(pos, q);
        std::vector<double> pf(V), qf(static_cast<size_t>(q.size()));
        for (int i = 0; i < 400; ++i) for (int c = 0; c < 3; ++c) qf[static_cast<size_t>(3 * i + c)] = q(i, c);
        const auto brute = nxr::geometry::nearest::nearestVerticesBrute<double>(pf, qf);
        const auto tree = nxr::geometry::nearest::nearestVertices<double>(pf, qf);
        EXPECT(got == brute.vertex, "track_crossings::nearestVertex == the brute force");
        EXPECT(tree.vertex == brute.vertex, "the k-d tree == the brute force");
        Eigen::MatrixXd tie(2, 3); tie << 0, 0, 0, 0.1, 0.2, 0.3;
        Eigen::MatrixXd dup(3, 3); dup << 1, 1, 1, 1, 1, 1, 5, 5, 5;
        EXPECT(nxr::select::track_crossings::nearestVertex(dup, tie.topRows(1)).front() == 0, "two vertices at one place: the lowest index");
    }

    std::cout << "=== row 19: the two vertex gradients are not bitwise equal; both kept ===\n";
    {
        Eigen::MatrixXd P(nV, 3);
        for (int i = 0; i < nV; ++i) for (int c = 0; c < 3; ++c) P(i, c) = V[3 * i + c];
        nxr::operators::FaceMat Fm = Eigen::Map<const nxr::operators::FaceMat>(F.data(), nF, 3);
        nxr::operators::RowMat3 Pr = P;
        Eigen::MatrixXd x(nV, 2);
        for (int i = 0; i < nV; ++i) { x(i, 0) = std::sin(3 * P(i, 0)) + P(i, 2); x(i, 1) = P(i, 1) * P(i, 1); }
        nxr::operators::FaceOperators ops(Pr, Fm, false);
        Eigen::MatrixXd g[3];
        ops.meanFaceGradient(x, g);
        const nxr::signal::Gradient G = nxr::signal::vertexGradient(P, Fm);
        double worst = 0, scale = 0; bool bitwise = true;
        for (int d = 0; d < 3; ++d) {
            const Eigen::MatrixXd h = G[static_cast<size_t>(d)] * x;
            worst = std::max(worst, (g[d] - h).cwiseAbs().maxCoeff()); scale = std::max(scale, h.cwiseAbs().maxCoeff());
            if ((g[d] - h).cwiseAbs().maxCoeff() != 0.0) bitwise = false;
        }
        std::cout << "  max |difference| " << worst << " (scale " << scale << "), bitwise equal: " << (bitwise ? "yes" : "no") << "\n";
        EXPECT(worst / scale < 1e-12, "the two agree to rounding");
    }
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}

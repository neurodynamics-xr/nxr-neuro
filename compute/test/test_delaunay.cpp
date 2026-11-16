// The one Delaunay construction (nxr/delaunay.h): invariants on synthetic meshes, and the operators built on it
// (Manifold default = intrinsic mollified Delaunay + mixed-Voronoi mass; the DEC follows the mesh option).
// Parity with the Studio's intrinsicDelaunay.ts and nxr's `prepare` is in test/parity/test_delaunay_parity.py.
#include "nxr/compute.h"
#include "nxr/facets.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace nxr::manifold;

static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

// A subdivided octahedron (level `lev`), optionally stretched and jittered so that many edges are not Delaunay.
static void sphere(int lev, double sx, double sy, double sz, double jitter, unsigned seed, std::vector<double>& V, std::vector<int32_t>& F) {
    V = {1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1};
    F = {0,2,4, 2,1,4, 1,3,4, 3,0,4, 2,0,5, 1,2,5, 3,1,5, 0,3,5};
    for (int l = 0; l < lev; ++l) {
        std::vector<int32_t> G;
        std::vector<std::pair<std::pair<int,int>, int>> cache;
        auto mid = [&](int a, int b) {
            if (a > b) std::swap(a, b);
            for (auto& c : cache) if (c.first.first == a && c.first.second == b) return c.second;
            const int id = static_cast<int>(V.size() / 3);
            double x = (V[3*a] + V[3*b]) / 2, y = (V[3*a+1] + V[3*b+1]) / 2, z = (V[3*a+2] + V[3*b+2]) / 2;
            const double n = std::sqrt(x*x + y*y + z*z);
            V.push_back(x / n); V.push_back(y / n); V.push_back(z / n);
            cache.push_back({{a, b}, id});
            return id;
        };
        for (size_t f = 0; f < F.size(); f += 3) {
            const int a = F[f], b = F[f+1], c = F[f+2], ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            for (int t : {a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca}) G.push_back(t);
        }
        F = G;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (size_t i = 0; i < V.size(); i += 3) {
        V[i] *= sx; V[i+1] *= sy; V[i+2] *= sz;
        V[i] += jitter * u(gen); V[i+1] += jitter * u(gen); V[i+2] += jitter * u(gen);
    }
}

static double areaOf(const delaunay::Mesh& m) { return delaunay::faceAreas(m).sum(); }

int main() {
    std::cout << "=== delaunay: stretched, jittered sphere ===\n";
    {
        std::vector<double> V; std::vector<int32_t> F;
        sphere(3, 1.0, 1.0, 6.0, 0.03, 7, V, F);
        const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
        const delaunay::Mesh raw = delaunay::prepare(V.data(), nV, F.data(), nF, {}, /*intrinsic=*/false);
        const delaunay::Mesh m = delaunay::prepare(V.data(), nV, F.data(), nF);
        CHECK(delaunay::nonDelaunayEdges(raw) > 20, "the input has many non-Delaunay edges (" + std::to_string(delaunay::nonDelaunayEdges(raw)) + ")");
        CHECK(m.record.flips > 20, "flips were made (" + std::to_string(m.record.flips) + ")");
        CHECK(delaunay::nonDelaunayEdges(m) == 0, "no edge below -flipEps remains");
        CHECK(m.record.mollify == 1e-6 && m.record.flipEps == 1e-6, "options recorded");
        double s = 0; for (double x : m.mass) s += x;
        CHECK(std::abs(s - areaOf(raw)) < 1e-9 * s, "mixed-Voronoi mass of the input sums to its area");
        CHECK(std::abs(areaOf(m) - areaOf(raw)) < 1e-9 * s, "flips keep the total area");
        bool mpos = true; for (double x : m.mass) mpos = mpos && x > 0;
        CHECK(mpos, "the recorded mass is positive");
        bool twins = true;
        for (int h = 0; h < 3 * m.nF; ++h) { const int t = m.twin[h]; twins = twins && t >= 0 && m.twin[t] == h && m.edgeOf[t] == m.edgeOf[h]; }
        CHECK(twins, "twins are consistent after the flips");
        bool ang = true;
        for (int v = 0; v < nV; ++v) ang = ang && std::abs(m.angleSum[v] - raw.angleSum[v]) < 1e-9 * 7;
        CHECK(ang, "flips keep every cone angle");
        bool inrange = true;
        for (int h = 0; h < 3 * m.nF; ++h) inrange = inrange && m.angle[h] >= 0 && m.angle[h] < m.angleSum[m.F[h]] + 1e-12;
        CHECK(inrange, "signposts stay in [0, cone angle)");
    }
    std::cout << "=== delaunay: mollification of a degenerate triangle ===\n";
    {
        std::vector<double> V; std::vector<int32_t> F;
        sphere(2, 1, 1, 1, 0.0, 1, V, F);
        // put vertex 6 (of the first face) exactly on the segment between the face's other two vertices: a zero-area triangle
        const int a = F[0], b = F[1], c = F[2];
        for (int k = 0; k < 3; ++k) V[3*c+k] = 0.5 * (V[3*a+k] + V[3*b+k]);
        const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
        const delaunay::Mesh m = delaunay::prepare(V.data(), nV, F.data(), nF);
        CHECK(m.record.epsilon > 0, "mollification added epsilon (" + std::to_string(m.record.epsilon) + ")");
        const delaunay::Mesh m0 = delaunay::prepare(V.data(), nV, F.data(), nF, delaunay::DelaunayOptions{1e-3, 1e-6});
        CHECK(m0.record.epsilon > m.record.epsilon, "a larger mollify margin gives a larger epsilon");
        bool fin = true; for (double x : m.mass) fin = fin && std::isfinite(x);
        CHECK(fin, "mass finite");
    }
    std::cout << "=== delaunay: embedded option ===\n";
    {
        std::vector<double> V; std::vector<int32_t> F;
        sphere(2, 1, 1, 4, 0.02, 3, V, F);
        const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
        const delaunay::Mesh e = delaunay::prepare(V.data(), nV, F.data(), nF, {}, false);
        CHECK(e.record.flips == 0 && e.record.epsilon == 0, "embedded: no flips, no mollification");
        bool same = true; for (int i = 0; i < 3 * nF; ++i) same = same && e.F[i] == F[i];
        CHECK(same, "embedded: faces unchanged");
    }
    std::cout << "=== Manifold: default is the intrinsic mesh; DEC follows it ===\n";
    {
        std::vector<double> V; std::vector<int32_t> F;
        sphere(3, 1.0, 1.0, 5.0, 0.02, 11, V, F);
        const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
        Manifold m(V.data(), nV, F.data(), nF);
        CHECK(m.isIntrinsicDelaunay(), "default Manifold is intrinsic");
        CHECK(m.operatorMesh().record.flips > 0, "default Manifold flipped");
        auto dec = m.operators().dec();
        Eigen::SparseMatrix<double> L = dec.d0.transpose() * dec.hodge1 * dec.d0;
        const auto& C = m.operators().laplacian().cotan();
        CHECK((L - C).norm() < 1e-9 * C.norm(), "d0' *1 d0 == cotan Laplacian");
        const auto& M = m.operators().mass().mixedVoronoi();
        CHECK((dec.hodge0 - M).norm() < 1e-12 * M.norm(), "hodge0 == the mixed-Voronoi mass");
        auto ops = ops::assembleManifoldOperators(m);
        CHECK(ops.massVariant == ops::MassMatrixVariant::MixedVoronoi, "default mass variant is mixedVoronoi");
        CHECK((ops.mass - M).norm() < 1e-14 * M.norm(), "assembleManifoldOperators default mass == the mixed-Voronoi mass");
        Manifold e(V.data(), nV, F.data(), nF, /*intrinsicDelaunay=*/false);
        CHECK(!e.isIntrinsicDelaunay(), "flag false = the embedded option");
        auto dece = e.operators().dec();
        Eigen::SparseMatrix<double> Le = dece.d0.transpose() * dece.hodge1 * dece.d0;
        CHECK((Le - e.operators().laplacian().cotan()).norm() < 1e-9 * Le.norm(), "embedded: d0' *1 d0 == cotan Laplacian");
        // the embedded DEC (the operator mesh unflipped) agrees edge by edge with geometry-central's
        auto& g = e.decEmbedded();
        CHECK((g.d0 - dece.d0).norm() == 0, "embedded DEC: d0 equals geometry-central's (same edge order and orientation)");
        CHECK((g.hodge1 - dece.hodge1).norm() < 1e-9 * g.hodge1.norm(), "embedded DEC: star1 equals geometry-central's");
    }
    std::cout << "=== Hodge-Dirac and connection Laplacian use the same construction ===\n";
    {
        std::vector<double> V; std::vector<int32_t> F;
        sphere(2, 1, 1, 1, 0.0, 1, V, F);
        const int a = F[0], b = F[1], c = F[2];
        for (int k = 0; k < 3; ++k) V[3*c+k] = 0.5 * (V[3*a+k] + V[3*b+k]);      // a zero-area triangle: needs mollification
        const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
        const delaunay::Mesh ref = delaunay::prepare(V.data(), nV, F.data(), nF);
        Manifold m(V.data(), nV, F.data(), nF);
        const auto h = ops::hodgedirac::assemble(m);
        CHECK(h.record.epsilon == ref.record.epsilon && h.mollifyEps > 0, "Hodge-Dirac mollifies exactly as the construction (epsilon " + std::to_string(h.mollifyEps) + ")");
        CHECK(h.record.flips == ref.record.flips && h.record.flipEps == 1e-6, "Hodge-Dirac flips as the construction (" + std::to_string(h.record.flips) + " flips)");
        bool same = true; for (int i = 0; i < nV; ++i) same = same && h.star0(i) == ref.mass[i];
        CHECK(same, "Hodge-Dirac star0 is the mixed-Voronoi mass of the mollified input, recorded before the flips");
        // another mollify: its own construction, recorded
        const auto h2 = ops::hodgedirac::assemble(m, 0.0, 1e-3, 1e-9);
        CHECK(h2.record.mollify == 1e-3 && h2.record.flipEps == 1e-9 && h2.mollifyEps > h.mollifyEps, "Hodge-Dirac takes mollify and flipEps as recorded options");
        // the vertex connection Laplacian on the signposts: Hermitian, positive semidefinite on the Delaunay mesh
        namespace cl = ops::laplacian::connection;
        cl::ConnectionLaplacianOptions o; o.domain = cl::ConnectionDomain::Vertex; o.nSym = 1; o.format = cl::ConnectionLaplacianFormat::Complex;
        const Eigen::MatrixXcd K = Eigen::MatrixXcd(cl::assembleConnectionLaplacian(m, o).K_complex);
        CHECK((K - K.adjoint()).norm() < 1e-12 * K.norm(), "connection Laplacian is Hermitian");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(K);
        CHECK(es.eigenvalues()(0) > -1e-9, "connection Laplacian is positive semidefinite on the Delaunay mesh (min " + std::to_string(es.eigenvalues()(0)) + ")");
        // its vertex frames are the signposts' gauge: e1 along the reference neighbour projected on the tangent plane, e2 = n x e1, orthonormal
        const geometry::VertexFrames vf = geometry::vertexFrames(m);
        double err = 0; for (int v = 0; v < nV; ++v) {
            Eigen::Vector3d e1 = vf.e1.row(v), e2 = vf.e2.row(v), n = vf.normals.row(v);
            err = std::max({err, std::abs(e1.norm() - 1), std::abs(e2.norm() - 1), std::abs(e1.dot(e2)), std::abs(e1.dot(n)), (e1.cross(e2) - n).norm()});
        }
        CHECK(err < 1e-12, "vertex frames are orthonormal and right-handed");
    }
    std::cout << (failures ? "\nFAILED\n" : "\nALL PASSED\n");
    return failures ? 1 : 0;
}

// THE ONE-TIME COMPARISON of the geometry-central IntrinsicSurface with the old line port (nxr/delaunay.h, nxr/chart.h):
// docs/specs/2026-11-17-geometry-central-switch.md section 4 and its appendix B. It builds both on the same synthetic meshes, prints every number, and asserts the
// differences the spec explains: the gauge rotation, the triangulation, the boundary vertices, the masses, the frames and the charts. A test kept until the port is
// deleted (G7). Plus the poor-mesh study: eigenvalue error and positive semidefiniteness, intrinsic against embedded, on needle and obtuse meshes.
#include "chart_port.h"
#include "nxr/compute.h"
#include "nxr/delaunay.h"
#include "nxr/intrinsic_surface.h"
#include "synthetic_meshes.h"

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace nxr::manifold;
namespace gcs = geometrycentral::surface;
namespace chart = nxr::chart_port;
namespace delaunay = nxr::manifold::delaunay;
using C = std::complex<double>;

static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)
static std::string f(double x) { char b[40]; std::snprintf(b, sizeof b, "%.3g", x); return b; }

static double q(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }

// the gauge rotation of every vertex: the angle of the port's reference halfedge in geometry-central's gauge
static std::vector<double> rotation(IntrinsicSurface& s, const delaunay::Mesh& pm) {
    auto& G = s.inputGeometry(); G.requireHalfedgeVectorsInVertex();
    std::vector<double> th(static_cast<size_t>(pm.nV), 0.0);
    for (gcs::Vertex v : s.inputMesh().vertices())
        for (gcs::Halfedge he : v.outgoingHalfedges()) if (static_cast<int>(he.tipVertex().getIndex()) == pm.ref[v.getIndex()]) th[v.getIndex()] = G.halfedgeVectorsInVertex[he].arg();
    return th;
}

static void compare(const std::string& name, const Mesh& m, bool strictCharts = true) {
    std::cout << name << " (" << m.nV() << " vertices)\n";
    const int nV = m.nV();
    // flipEps = 1e-12 stands for 0 (geometry-central flips at <= -eps, so an exact 0 would flip ties for ever)
    delaunay::DelaunayOptions po; po.flipEps = 1e-12;
    IntrinsicOptions o; o.flipEps = 1e-12;
    delaunay::Mesh pm = delaunay::prepare(m.V.data(), nV, m.F.data(), m.nF(), po, true);
    IntrinsicSurface s(m.V.data(), nV, m.F.data(), m.nF(), o);
    std::cout << "      flips: port " << pm.record.flips << ", geometry-central " << s.record().flips << "; eps " << pm.record.epsilon << " / " << s.record().epsilon << "\n";
    const std::vector<double> th = rotation(s, pm);
    // 1 + 2: the gauge rotation and the triangulation
    auto bnd = [&](int v) { return pm.boundary[static_cast<size_t>(v)] != 0; };
    const Eigen::MatrixXcd Kp = Eigen::MatrixXcd(delaunay::connectionLaplacian(pm, 1)), Kg = Eigen::MatrixXcd(s.connectionLaplacian(1));
    Eigen::VectorXcd d(nV); for (int v = 0; v < nV; ++v) d[v] = std::polar(1.0, th[static_cast<size_t>(v)]);
    double best[2] = {0, 0}, onB[2] = {0, 0};
    for (int sgn = 0; sgn < 2; ++sgn) {
        Eigen::VectorXcd dd = sgn == 0 ? d : Eigen::VectorXcd(d.conjugate());
        const Eigen::MatrixXcd R = dd.asDiagonal() * Kp * dd.conjugate().asDiagonal();
        for (int i = 0; i < nV; ++i) for (int j = 0; j < nV; ++j) { const double e = std::abs(Kg(i, j) - R(i, j)); if (bnd(i) || bnd(j)) onB[sgn] = std::max(onB[sgn], e); else best[sgn] = std::max(best[sgn], e); }
    }
    const int sgn = best[0] <= best[1] ? 0 : 1;
    std::cout << "      K_gc = D K_port D^H" << (sgn ? " with D conjugated" : "") << ": max interior difference " << f(best[sgn]) << " (other convention " << f(best[1 - sgn]) << "); entries touching a boundary vertex " << f(onB[sgn]) << "\n";
    CHECK(best[sgn] < 1e-9, name + ": the connection Laplacians differ by the gauge rotation only (interior entries, " + f(best[sgn]) + ")");
    const Eigen::MatrixXd Lp = Eigen::MatrixXd(delaunay::cotanLaplacian(pm)), Lg = Eigen::MatrixXd(s.cotanLaplacian());
    const double dl = (Lp - Lg).cwiseAbs().maxCoeff();
    CHECK(dl < 1e-9 && pm.record.flips == s.record().flips, name + ": equal triangulation (" + std::to_string(s.record().flips) + " flips): the cotan Laplacians agree to " + f(dl));
    // the default flipEps 1e-6 may leave different near-Delaunay edges unflipped
    { delaunay::DelaunayOptions p6; IntrinsicOptions o6; delaunay::Mesh pm6 = delaunay::prepare(m.V.data(), nV, m.F.data(), m.nF(), p6, true); IntrinsicSurface s6(m.V.data(), nV, m.F.data(), m.nF(), o6);
      const double d6 = (Eigen::MatrixXd(delaunay::cotanLaplacian(pm6)) - Eigen::MatrixXd(s6.cotanLaplacian())).cwiseAbs().maxCoeff();
      std::cout << "      at flipEps 1e-6: flips " << pm6.record.flips << " / " << s6.record().flips << ", cotan Laplacian difference " << f(d6) << "\n";
      CHECK(d6 < 1e-4, name + ": flipEps 1e-6: the Laplacians differ by O(flipEps) at most (" + f(d6) + ")"); }
    // 3 boundary vertices
    std::cout << "      boundary vertices: port " << std::count(pm.boundary.begin(), pm.boundary.end(), 1) << ", geometry-central " << s.record().boundaryVertices << "\n";
    // 4 masses
    double dm = 0; { const auto g = s.massDiagonal(); for (int v = 0; v < nV; ++v) dm = std::max(dm, std::abs(g[static_cast<size_t>(v)] - pm.mass[static_cast<size_t>(v)])); }
    IntrinsicOptions ob = o; ob.mass = MassKind::Barycentric; IntrinsicSurface sb(m.V.data(), nV, m.F.data(), m.nF(), ob);
    IntrinsicOptions og = o; og.mass = MassKind::Galerkin; IntrinsicSurface sg(m.V.data(), nV, m.F.data(), m.nF(), og);
    double db = 0; { const auto g = sb.massDiagonal(); const auto p = delaunay::barycentricMass(pm); for (int v = 0; v < nV; ++v) db = std::max(db, std::abs(g[static_cast<size_t>(v)] - p[static_cast<size_t>(v)])); }
    const double dg = (Eigen::MatrixXd(sg.massMatrix()) - Eigen::MatrixXd(delaunay::galerkinMass(pm))).cwiseAbs().maxCoeff();
    CHECK(dm < 1e-12 && db < 1e-12 && dg < 1e-12, name + ": masses equal to rounding: mixed Voronoi " + f(dm) + ", barycentric " + f(db) + ", galerkin " + f(dg));
    // 5 frames: the port's e1 against geometry-central's, after removing the rotation
    const chart::Basis pb = chart::tangentBasis(pm, m.V.data(), m.F.data(), m.nF()); const TangentBasis gb = s.tangentBasis();
    std::vector<double> ang;
    for (int v = 0; v < nV; ++v) {
        Eigen::Vector3d n(gb.n[3*v], gb.n[3*v+1], gb.n[3*v+2]), e1(gb.e1[3*v], gb.e1[3*v+1], gb.e1[3*v+2]), e2(gb.e2[3*v], gb.e2[3*v+1], gb.e2[3*v+2]);
        const Eigen::Vector3d g = std::cos(th[v]) * e1 + std::sin(th[v]) * e2;   // the port's reference direction in geometry-central's frame
        const Eigen::Vector3d p(pb.e1[3*v], pb.e1[3*v+1], pb.e1[3*v+2]);
        ang.push_back(std::atan2(p.cross(g).dot(n), p.dot(g)));
    }
    std::vector<double> aa; for (double a : ang) aa.push_back(std::abs(a));
    std::cout << "      frame angle, port e1 vs gc e1 rotated by theta: median " << f(q(aa, .5)) << ", 99% " << f(q(aa, .99)) << ", max " << f(q(aa, 1)) << " rad\n";
    CHECK(q(aa, .5) < 0.05, name + ": the frames differ by the 3D realisation only (median " + f(q(aa, .5)) + " rad; the tail is at vertices with a cone angle far from 2 pi)");
    // 6 charts
    chart::ChartKit kit(m.V.data(), nV, m.F.data(), m.nF(), po);
    gcs::HeatMethodDistanceSolver hm(s.geometry()); gcs::VectorHeatMethodSolver vh(s.geometry());
    const int src = nV / 3;
    const Eigen::VectorXd dp = kit.heatDistance(src); const auto dgc = hm.computeDistance(s.inputMesh().vertex(src));
    double dd = 0, dn = 0; for (int v = 0; v < nV; ++v) { dd = std::max(dd, std::abs(dp[v] - dgc[v])); dn = std::max(dn, std::abs(dp[v])); }
    std::cout << "      heat distance: max difference " << f(dd) << " (max distance " << f(dn) << ")\n";
    const chart::LogMap lp = kit.logMap(src); const auto lg = vh.computeLogMap(s.inputMesh().vertex(src));
    double sg0 = 0, sg1 = 0; std::vector<double> e0, e1;
    // inside the injectivity radius, off the source
    for (int v = 0; v < nV; ++v) { if (!(lp.r[v] > 0.15) || !(lp.r[v] < 1.2) || !std::isfinite(lp.theta[v])) continue; e0.push_back(std::abs(std::remainder(lp.theta[v] - (lg[v].arg() + th[src]), 2 * M_PI))); e1.push_back(std::abs(std::remainder(lp.theta[v] - (lg[v].arg() - th[src]), 2 * M_PI))); }
    sg0 = q(e0, .5); sg1 = q(e1, .5);
    const std::vector<double>& el = sg0 <= sg1 ? e0 : e1;
    std::cout << "      log map angle after the rotation by theta (" << (sg0 <= sg1 ? "+" : "-") << "): median " << f(q(el, .5)) << ", 90% " << f(q(el, .9)) << " rad\n";
    if (s.record().boundaryVertices == 0 && strictCharts) CHECK(q(el, .5) < 0.1, name + ": log map angles agree after the rotation (median " + f(q(el, .5)) + " rad)");
    if (s.record().boundaryVertices == 0) CHECK(dd < 1e-3 * dn, name + ": heat distance agrees (relative " + f(dd / dn) + ")");
    else std::cout << "      (a boundary: the port takes the Dirichlet average at a boundary, geometry-central does not; the distances differ there, reported only)\n";
}

// a UV sphere with every ring jittered in longitude and latitude: skewed needle and obtuse triangles on the unit sphere (the connectivity is kept)
static Mesh needleJittered(int nLat, int nLon, double frac, unsigned seed) {
    Mesh m = uvSphere(nLat, nLon); std::mt19937 g(seed);
    for (int v = 1; v < m.nV() - 1; ++v) {
        double* x = &m.V[3 * v]; double th = std::acos(x[2]), ph = std::atan2(x[1], x[0]);
        ph += frac * (2 * M_PI / nLon) * (2 * u01(g) - 1); th += frac * 0.5 * (M_PI / nLat) * (2 * u01(g) - 1);
        x[0] = std::sin(th) * std::cos(ph); x[1] = std::sin(th) * std::sin(ph); x[2] = std::cos(th);
    }
    return m;
}

// the poor-mesh study: eigenvalue error and positive semidefiniteness, intrinsic against embedded
static void poor(const std::string& name, const Mesh& m) {
    IntrinsicOptions oi, oe; oe.intrinsic = false;
    double res[2], lmin[2], neg[2], worstW[2], hd[2]; int flips = 0;
    for (int k = 0; k < 2; ++k) {
        IntrinsicSurface s(m.V.data(), m.nV(), m.F.data(), m.nF(), k == 0 ? oi : oe);
        if (k == 0) flips = s.record().flips;
        const Eigen::MatrixXd L = Eigen::MatrixXd(s.cotanLaplacian()); const auto md = s.massDiagonal();
        Eigen::VectorXd mi(m.nV()); for (int i = 0; i < m.nV(); ++i) mi[i] = 1 / std::sqrt(md[i]);
        const Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(mi.asDiagonal() * L * mi.asDiagonal()).eigenvalues();
        // the first clusters l <= 3: mean relative error
        double e = 0; int at = 1; for (int l = 1; l <= 3; ++l) { double mean = 0; for (int i = 0; i < 2 * l + 1; ++i) mean += ev[at + i] / (2 * l + 1); e += std::abs(mean - l * (l + 1.0)) / (l * (l + 1.0)) / 3; at += 2 * l + 1; }
        res[k] = e; lmin[k] = ev[0];
        { gcs::HeatMethodDistanceSolver hm(s.geometry()); int pole = 0; const auto D = hm.computeDistance(s.inputMesh().vertex(pole)); double err = 0; int cnt = 0;
          for (int v = 1; v < m.nV(); ++v) { err += std::abs(D[v] - std::acos(std::max(-1.0, std::min(1.0, m.V[3*v]*m.V[0] + m.V[3*v+1]*m.V[1] + m.V[3*v+2]*m.V[2])))); ++cnt; } hd[k] = err / cnt; }
        int n = 0, tot = 0; double w = 0; for (int i = 0; i < m.nV(); ++i) for (int j = i + 1; j < m.nV(); ++j) if (L(i, j) != 0) { ++tot; if (-L(i, j) < 0) { ++n; w = std::min(w, -L(i, j)); } }
        neg[k] = tot ? double(n) / tot : 0; worstW[k] = w;
    }
    std::cout << "    " << name << " (" << m.nV() << " vertices, " << flips << " flips): eigenvalue error intrinsic " << f(res[0]) << " / embedded " << f(res[1])
              << "; heat distance error from vertex 0 intrinsic " << f(hd[0]) << " / embedded " << f(hd[1]) << "; smallest eigenvalue of L intrinsic " << f(lmin[0]) << " / embedded " << f(lmin[1]) << "; negative cotan weights intrinsic " << f(100 * neg[0]) << "% (worst " << f(worstW[0]) << ") / embedded "
              << f(100 * neg[1]) << "% (worst " << f(worstW[1]) << ")\n";
    CHECK(lmin[0] > -1e-9, name + ": the intrinsic cotan Laplacian is positive semidefinite (smallest eigenvalue " + f(lmin[0]) + ")");
    CHECK(lmin[1] > -1e-9, name + ": so is the embedded one (the P1 stiffness matrix is positive semidefinite for any valid triangulation; what negative weights break is the maximum principle)");
    CHECK(worstW[0] > -1e-9, name + ": intrinsic: no negative cotan weight (the discrete maximum principle holds; the worst is " + f(worstW[0]) + ")");
    CHECK(neg[1] > 0.01, name + ": the embedded Laplacian has negative weights here (" + f(100 * neg[1]) + "% of edges)");
}

int main(int argc, char** argv) {
    auto want = [&](const char* n) { if (argc < 2) return true; for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == n) return true; return false; };
    if (want("compare")) {
        std::cout << "THE ONE-TIME COMPARISON with the port\n";
        compare("icosphere 2", icosphere(2));
        compare("jittered icosphere 2", jitterSphere(icosphere(2), 0.35, 5), false);   // a log map is only a chart where the mesh resolves it: reported, not asserted
        compare("uv sphere 8x12 (poles)", uvSphere(8, 12));
        compare("torus 12x8", torus(12, 8, 3, 1));
        compare("planar disc (boundary)", disc(5, 0.25, 9));
    }
    if (want("poor")) {
        std::cout << "POOR MESHES: intrinsic against embedded\n";
        poor("needle sphere 6x96, skewed 0.45", needleJittered(6, 96, 0.45, 1));
        poor("needle sphere 8x128, skewed 0.45", needleJittered(8, 128, 0.45, 2));
        poor("jittered icosphere 3 (0.7)", jitterSphere(icosphere(3), 0.7, 7));
    }
    std::cout << (failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}

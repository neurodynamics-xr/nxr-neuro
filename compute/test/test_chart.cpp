// nxr::chart: the Cortical Flow Studio's charts, ported line for line (heat method on the intrinsic mesh, vector heat transport, log map, gauge heading, disc).
// (1) PARITY with the Studio's own code (golden/gen_chart_golden.mts runs SurfaceKit: distance, transported, logMap, headingOf, disc, gradient; tangentBasis) on a sphere, an irregular
//     (stretched, jittered) mesh and an open patch;
// (2) the sphere's analytic log map (geodesic polar coordinates), read in the signpost gauge and in a shared north gauge; the gauge round trip; the transport is rotation-linear;
// (3) the geodesic default is the intrinsic mesh (HeatMesh::Intrinsic == the chart's heat method; HeatMesh::Input stays as the named option); disc refuses a radius past maxRadius.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/chart.h"
#include "nxr/compute.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

static double wrap(double a) { return std::remainder(a, 2 * M_PI); }
// max |a-b| over entries; NaN / inf must agree
static double maxDiff(const MatrixXd& a, const MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e9;
    double d = 0;
    for (Eigen::Index j = 0; j < a.cols(); ++j) for (Eigen::Index i = 0; i < a.rows(); ++i) {
        const double x = a(i, j), y = b(i, j);
        if (std::isnan(x) || std::isnan(y)) { if (std::isnan(x) != std::isnan(y)) d = 1e9; continue; }
        if (std::isinf(x) || std::isinf(y)) { if (x != y) d = 1e9; continue; }
        d = std::max(d, std::abs(x - y));
    }
    return d;
}
// a flat golden block [rows x cols] row-major as one column of rows*cols
static MatrixXd flat(const MatrixXd& m) { MatrixXd o(m.size(), 1); Eigen::Index k = 0; for (Eigen::Index i = 0; i < m.rows(); ++i) for (Eigen::Index j = 0; j < m.cols(); ++j) o(k++, 0) = m(i, j); return o; }
template <class T> static MatrixXd asCol(const std::vector<T>& v) { MatrixXd m(static_cast<Eigen::Index>(v.size()), 1); for (size_t k = 0; k < v.size(); ++k) m(static_cast<Eigen::Index>(k), 0) = static_cast<double>(v[k]); return m; }

static void parity(Golden& g) {
    if (g.tok() != "mesh") { std::cout << "  [FAIL] golden: expected mesh\n"; std::exit(1); }
    const std::string name = g.tok();
    const int n = static_cast<int>(g.num()), nSeeds = static_cast<int>(g.num());
    const double radius = g.num();
    const MatrixXd Pm = g.mat("positions"), Fm = g.mat("faces");
    std::vector<double> P(static_cast<size_t>(Pm.size())); std::vector<int32_t> F(static_cast<size_t>(Fm.size()));
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) P[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = Pm(i, k);
    for (int i = 0; i < Fm.rows(); ++i) for (int k = 0; k < 3; ++k) F[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = static_cast<int32_t>(Fm(i, k));
    nxr::chart::ChartKit kit(P.data(), n, F.data(), static_cast<int>(Fm.rows()));
    const MatrixXd e1 = g.mat("e1"), e2 = g.mat("e2"), nr = g.mat("nrm");
    const nxr::chart::Basis b = kit.basis();
    std::cout << "  " << name << ": tangent basis max |diff| e1 " << sci(maxDiff(asCol(b.e1), flat(e1))) << ", e2 " << sci(maxDiff(asCol(b.e2), flat(e2))) << ", n " << sci(maxDiff(asCol(b.n), flat(nr))) << "\n";
    // e1 and n bit for bit; e2 = n x e1 may differ in an entry that cancels to ~1e-17 (hypot's last bit)
    EXPECT(maxDiff(asCol(b.e1), flat(e1)) == 0 && maxDiff(asCol(b.e2), flat(e2)) < 1e-15 && maxDiff(asCol(b.n), flat(nr)) == 0, name + ": tangent basis equals the Studio's (float32; e1, n bit-identical)");
    double dD = 0, dX = 0, dX7 = 0, dR = 0, dT = 0, dH = 0, dG = 0; bool discOk = true;
    const double dir[3] = {0.3, -0.5, 0.8};
    for (int s = 0; s < nSeeds; ++s) {
        if (g.tok() != "seed") { std::cout << "  [FAIL] golden: expected seed\n"; std::exit(1); }
        const int v = static_cast<int>(g.num());
        const MatrixXd gd = g.mat("dist"), gX = g.mat("X"), gX7 = g.mat("X07"), gr = g.mat("r"), gt = g.mat("theta"), gh = g.mat("heading"), gdisc = g.mat("disc"), gg = g.mat("grad");
        const VectorXd d = kit.heatDistance(v);
        dD = std::max(dD, maxDiff(d, gd));
        dX = std::max(dX, maxDiff(kit.transported(v, 0.0), flat(gX)));
        dX7 = std::max(dX7, maxDiff(kit.transported(v, 0.7), flat(gX7)));
        const nxr::chart::LogMap lm = kit.logMap(v);
        dR = std::max(dR, maxDiff(lm.r, gr));
        // theta is undefined where the radial field has no direction (the seed's cut locus: a sphere's antipode): the last 5% of r is not compared
        const double rmax = gr.maxCoeff();
        for (int i = 0; i < n; ++i) { if (std::isnan(lm.theta[i]) != std::isnan(gt(i, 0))) dT = 1e9; else if (!std::isnan(lm.theta[i]) && gr(i, 0) < 0.95 * rmax) dT = std::max(dT, std::abs(wrap(lm.theta[i] - gt(i, 0)))); }
        dH = std::max(dH, std::abs(kit.gaugeHeading(v, dir) - gh(0, 0)));
        discOk = discOk && static_cast<double>(kit.disc(v, radius).size()) == gdisc(0, 0);
        VectorXd dz = d; for (Eigen::Index i = 0; i < dz.size(); ++i) if (!std::isfinite(dz[i])) dz[i] = 0;
        dG = std::max(dG, maxDiff(kit.gradient(dz), flat(gg)));
    }
    std::cout << "  " << name << ": max |diff| distance " << sci(dD) << ", transported " << sci(dX) << " / " << sci(dX7) << ", r " << sci(dR) << ", theta " << sci(dT) << ", heading " << sci(dH) << ", gradient " << sci(dG) << "\n";
    EXPECT(dD < 1e-10 && dR < 1e-10, name + ": heat distance equals the Studio's (< 1e-10)");
    EXPECT(dX < 1e-10 && dX7 < 1e-10, name + ": vector heat transport equals the Studio's (< 1e-10)");
    EXPECT(dG < 1e-10, name + ": surface gradient equals the Studio's");
    EXPECT(dT < 1e-8, name + ": log map theta equals the Studio's (< 1e-8)");
    EXPECT(dH < 1e-12, name + ": gauge heading equals the Studio's");
    EXPECT(discOk, name + ": disc has the Studio's vertices");
}

int main() {
    std::cout << "=== chart (the Studio's charts) ===\n";
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/chart_golden.txt");
    if (!g.good()) { std::cout << "  [FAIL] chart_golden.txt missing\n"; return 1; }
    parity(g); parity(g); parity(g);

    // (2) the sphere: geodesic polar coordinates
    std::vector<double> V; std::vector<int32_t> F;
    icosphere(3, V, F, 1.0);
    const int n = static_cast<int>(V.size() / 3);
    nxr::chart::ChartKit kit(V.data(), n, F.data(), static_cast<int>(F.size() / 3));
    const int seed = 7;
    const Eigen::Vector3d p(V[3 * seed], V[3 * seed + 1], V[3 * seed + 2]);
    const nxr::chart::Basis bs = kit.basis();
    const Eigen::Vector3d E1(bs.e1[3 * seed], bs.e1[3 * seed + 1], bs.e1[3 * seed + 2]), E2(bs.e2[3 * seed], bs.e2[3 * seed + 1], bs.e2[3 * seed + 2]);
    const nxr::chart::LogMap lm = kit.logMap(seed);
    const Eigen::Vector3d z(0, 0, 1), N = (z - z.dot(p) * p).normalized();   // the shared north: z on the tangent plane
    const double N3[3] = {N[0], N[1], N[2]};
    const double hN = kit.gaugeHeading(seed, N3);
    double eR = 0, eT = 0, eN = 0; int used = 0;
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3d x(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
        const double r = std::acos(std::min(1.0, std::max(-1.0, x.dot(p))));
        if (i == seed || r > 1.0 || r < 0.25) continue;
        const Eigen::Vector3d t = x - x.dot(p) * p;   // the leaving direction at the seed
        eR = std::max(eR, std::abs(lm.r[i] - r));
        eT = std::max(eT, std::abs(wrap(lm.theta[i] - std::atan2(t.dot(E2), t.dot(E1)))));
        eN = std::max(eN, std::abs(wrap((lm.theta[i] - hN) - std::atan2(t.dot(p.cross(N)), t.dot(N)))));   // theta read from north: the compass bearing
        ++used;
    }
    std::cout << "  sphere (icosphere 3, " << used << " vertices, 0.25 < r < 1): max |r - arccos| " << sci(eR) << ", max |theta - leaving angle| " << sci(eT) << ", from north " << sci(eN) << "\n";
    EXPECT(eR < 0.03, "log map r is the geodesic distance on the sphere");
    EXPECT(eT < 0.1, "log map theta is the geodesic polar angle on the sphere (against the seed's reference edge)");
    EXPECT(eN < 0.1, "theta - gaugeHeading(north) is the compass bearing (consistent under the gauge)");
    EXPECT(lm.theta[seed] == 0.0 && lm.r[seed] == 0.0, "theta = 0 and r = 0 at the seed");

    double eh = 0;
    for (double psi : {-3.0, -1.0, 0.0, 0.4, 2.5}) { double w[3]; kit.headingVector(seed, psi, w); eh = std::max(eh, std::abs(wrap(kit.gaugeHeading(seed, w) - psi))); }
    EXPECT(eh < 1e-6, "gaugeHeading(headingVector(psi)) = psi (float32 basis)");
    const VectorXd X0 = kit.transported(seed, 0.0), X1 = kit.transported(seed, 1.1);
    double el = 0;
    for (int i = 0; i < n; ++i) { const std::complex<double> a(X0[2 * i], X0[2 * i + 1]), b(X1[2 * i], X1[2 * i + 1]); el = std::max(el, std::abs(b - a * std::polar(1.0, 1.1))); }
    EXPECT(el < 1e-10, "transported(psi) = e^{i psi} transported(0)");
    bool refused = false;
    try { kit.disc(seed, 0.5, 0.3); } catch (const std::exception&) { refused = true; }
    EXPECT(refused && !kit.disc(seed, 0.5, 0.6).empty(), "disc refuses a radius past the validity radius");

    // (3) the geodesic default
    nxr::manifold::Manifold m(V.data(), n, F.data(), static_cast<int>(F.size() / 3));
    nxr::manifold::solve::HeatGeodesicSolver sDefault(m), sInput(m, 1.0, nxr::manifold::solve::HeatMesh::Input);
    const VectorXd dDefault = nxr::manifold::solve::heat(sDefault, {seed}), dInput = nxr::manifold::solve::heat(sInput, {seed}), dChart = kit.heatDistance(seed);
    EXPECT((dDefault - dChart).cwiseAbs().maxCoeff() < 1e-10, "the geodesic default (solve::heat) is the chart's intrinsic heat method");
    EXPECT(dInput.size() == dDefault.size() && (dInput - dDefault).cwiseAbs().maxCoeff() < 0.05, "HeatMesh::Input stays as the named option (geometry-central, input mesh)");
    std::cout << "  geodesic default vs input-mesh option on the sphere: max |diff| " << sci((dInput - dDefault).cwiseAbs().maxCoeff()) << "\n";
    std::cout << (g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}

// nxr::chart: the charts on geometry-central (ChartKit, a thin wrapper of HeatMethodDistanceSolver and VectorHeatMethodSolver on one IntrinsicSurface per connected piece, G4 of
// docs/specs/2026-11-17-geometry-central-switch.md). The values are geometry-central's, in THE gauge (its vertex gauge of the input mesh), so the checks are
// (1) the wrapper IS geometry-central: the kit's distance, transport, log map and basis equal the solvers' and IntrinsicSurface's own, to rounding (the port's 1e-4 of G3 is gone);
// (2) analytic results: the sphere's geodesic polar coordinates, read in the gauge and in a shared north; the gauge round trip; the transport is rotation-linear; a plane's gradient;
// (3) the kit's pieces: two components (NaN off the seed's, Infinity for distance, a vertex in no face is refused), the geodesic default is the same solver, disc refuses past maxRadius.
// The Studio's chart golden (chart_golden.txt and its generator) is deleted: the Studio's gauge and Dirichlet boundary are not geometry-central's, so it is replaced by the analytic checks here.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/chart.h"
#include "nxr/compute.h"
#include "nxr/intrinsic_surface.h"

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace gcs = geometrycentral::surface;

static double wrap(double a) { return std::remainder(a, 2 * M_PI); }

int main() {
    std::cout << "=== chart (geometry-central) ===\n";

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
    EXPECT(eh < 1e-12, "gaugeHeading(headingVector(psi)) = psi");
    const VectorXd X0 = kit.transported(seed, 0.0), X1 = kit.transported(seed, 1.1);
    double el = 0;
    // (the transport's magnitude falls like exp(-d / sqrt t): past d = 1.5 on this sphere its direction is solver noise, which geometry-central normalises away, so it is compared within 1.5)
    const VectorXd dseed = kit.heatDistance(seed);
    for (int i = 0; i < n; ++i) { if (dseed[i] > 1.5) continue; const std::complex<double> a(X0[2 * i], X0[2 * i + 1]), b(X1[2 * i], X1[2 * i + 1]); el = std::max(el, std::abs(b - a * std::polar(1.0, 1.1))); }
    std::cout << "  transport rotation-linearity: max |diff| " << sci(el) << "\n";
    EXPECT(el < 1e-6, "transported(psi) = e^{i psi} transported(0) (within 1.5 of the source)");
    bool refused = false;
    try { kit.disc(seed, 0.5, 0.3); } catch (const std::exception&) { refused = true; }
    EXPECT(refused && !kit.disc(seed, 0.5, 0.6).empty(), "disc refuses a radius past the validity radius");

    // (1) the wrapper IS geometry-central: the same solvers on the same IntrinsicSurface, to rounding
    {
        nxr::manifold::IntrinsicSurface& surf = kit.surfaceOf(seed);
        gcs::HeatMethodDistanceSolver hm(surf.geometry());
        gcs::VectorHeatMethodSolver vh(surf.geometry());
        const gcs::VertexData<double> dg = hm.computeDistance(surf.mesh().vertex(static_cast<size_t>(seed)));
        const VectorXd dk = kit.heatDistance(seed);
        double eD = 0, eX = 0, eL = 0, eB = 0;
        for (int i = 0; i < n; ++i) eD = std::max(eD, std::abs(dg[static_cast<size_t>(i)] - dk[i]));
        const auto xg = vh.transportTangentVector(surf.mesh().vertex(static_cast<size_t>(seed)), geometrycentral::Vector2{std::cos(0.7), std::sin(0.7)});
        const VectorXd xk = kit.transported(seed, 0.7);
        const auto lg = vh.computeLogMap(surf.mesh().vertex(static_cast<size_t>(seed)), gcs::LogMapStrategy::VectorHeat);
        for (int i = 0; i < n; ++i) {
            eX = std::max({eX, std::abs(xg[static_cast<size_t>(i)].x - xk[2 * i]), std::abs(xg[static_cast<size_t>(i)].y - xk[2 * i + 1])});
            if (i != seed) eL = std::max(eL, std::abs(wrap(std::atan2(lg[static_cast<size_t>(i)].y, lg[static_cast<size_t>(i)].x) - lm.theta[i])));
            eD = std::max(eD, std::abs(lm.r[i] - dk[i]));
        }
        // the basis is geometry-central's vertexTangentBasis (nothing is mollified on a regular icosphere: epsilon = 0), orthonormal, right-handed
        gcs::ManifoldSurfaceMesh& im = surf.inputMesh();
        gcs::VertexPositionGeometry& eg = surf.embedded();
        eg.requireVertexTangentBasis(); eg.requireVertexNormals();
        for (int i = 0; i < n; ++i) {
            const auto B = eg.vertexTangentBasis[static_cast<size_t>(i)];
            const auto N = eg.vertexNormals[static_cast<size_t>(i)];
            const Eigen::Vector3d a(bs.e1[3 * i], bs.e1[3 * i + 1], bs.e1[3 * i + 2]), b(bs.e2[3 * i], bs.e2[3 * i + 1], bs.e2[3 * i + 2]), nn(bs.n[3 * i], bs.n[3 * i + 1], bs.n[3 * i + 2]);
            eB = std::max({eB, (a - Eigen::Vector3d(B[0].x, B[0].y, B[0].z)).norm(), (b - Eigen::Vector3d(B[1].x, B[1].y, B[1].z)).norm(), (nn - Eigen::Vector3d(N.x, N.y, N.z)).norm(),
                           std::abs(a.dot(b)), std::abs(a.norm() - 1), std::abs(a.cross(b).dot(nn) - 1)});
        }
        (void)im;
        std::cout << "  kit vs geometry-central: distance " << sci(eD) << ", transport " << sci(eX) << ", log map " << sci(eL) << ", basis vs vertexTangentBasis " << sci(eB) << "\n";
        EXPECT(eD < 1e-12, "the kit's heat distance is HeatMethodDistanceSolver's on the IntrinsicSurface");
        EXPECT(eX < 1e-12, "the kit's transport is VectorHeatMethodSolver::transportTangentVector's");
        EXPECT(eL < 1e-12, "the kit's log map is VectorHeatMethodSolver::computeLogMap's angle, with the heat distance as r");
        EXPECT(eB < 1e-12, "the kit's basis is geometry-central's vertexTangentBasis (e1 at the gauge's angle 0), orthonormal and right-handed");
        EXPECT(std::abs(xk[2 * seed] - std::cos(0.7)) < 1e-12 && std::abs(xk[2 * seed + 1] - std::sin(0.7)) < 1e-12, "the transported vector at its source is the source vector, in the gauge");
    }

    // (2) a plane's gradient is exact; the transport is the identity on a flat patch (the gauge of a plane's vertices is the same everywhere up to each vertex's own reference edge)
    {
        const int g = 24;
        std::vector<double> Pp; std::vector<int32_t> Fp;
        for (int j = 0; j <= g; ++j) for (int i = 0; i <= g; ++i) { Pp.push_back(i * 0.1); Pp.push_back(j * 0.1 + 0.004 * i); Pp.push_back(0.0); }
        for (int j = 0; j < g; ++j) for (int i = 0; i < g; ++i) {
            const int a = j * (g + 1) + i, b = a + 1, c = a + g + 1, d = c + 1;
            for (int x : {a, b, d, a, d, c}) Fp.push_back(x);
        }
        nxr::chart::ChartKit pk(Pp.data(), (g + 1) * (g + 1), Fp.data(), static_cast<int>(Fp.size() / 3));
        VectorXd fx((g + 1) * (g + 1));
        for (int v = 0; v < fx.size(); ++v) fx[v] = 2.0 * Pp[3 * v] - 3.0 * Pp[3 * v + 1];
        const VectorXd gr = pk.gradient(fx);
        double eG = 0;
        for (int v = 0; v < fx.size(); ++v) eG = std::max({eG, std::abs(gr[3 * v] - 2.0), std::abs(gr[3 * v + 1] + 3.0), std::abs(gr[3 * v + 2])});
        EXPECT(eG < 1e-12, "the surface gradient of a linear function on a plane is its slope, exactly");
        const int c0 = (g / 2) * (g + 1) + g / 2;
        const VectorXd dp = pk.heatDistance(c0);
        double eP = 0; int m = 0;
        for (int v = 0; v < fx.size(); ++v) { const double r = std::hypot(Pp[3 * v] - Pp[3 * c0], Pp[3 * v + 1] - Pp[3 * c0 + 1]); if (r > 0.5 && r < 1.0) { eP = std::max(eP, std::abs(dp[v] - r) / r); ++m; } }
        std::cout << "  plane: heat distance from the centre, relative error (" << m << " vertices) " << sci(eP) << "\n";
        EXPECT(eP < 0.08, "heat distance on a plane is the euclidean distance (within the heat method's accuracy on this mesh)");
    }

    // (3) pieces: two spheres in one mesh, plus a vertex in no face
    {
        std::vector<double> V2 = V; std::vector<int32_t> F2 = F;
        for (int i = 0; i < n; ++i) { V2.push_back(V[3 * i] + 5.0); V2.push_back(V[3 * i + 1]); V2.push_back(V[3 * i + 2]); }
        for (int32_t f : F) F2.push_back(f + n);
        V2.push_back(0); V2.push_back(0); V2.push_back(9);   // a vertex in no face
        const int n2 = 2 * n + 1;
        nxr::chart::ChartKit k2(V2.data(), n2, F2.data(), static_cast<int>(F2.size() / 3));
        const VectorXd d = k2.heatDistance(seed);
        bool inf = true, finite = true;
        for (int i = 0; i < n; ++i) finite = finite && std::isfinite(d[i]);
        for (int i = n; i < 2 * n; ++i) inf = inf && std::isinf(d[i]);
        EXPECT(finite && inf, "distance is finite on the seed's piece and Infinity on the other");
        const nxr::chart::LogMap l2 = k2.logMap(seed);
        bool nan = true, same = true;
        for (int i = n; i < n2; ++i) nan = nan && std::isnan(l2.r[i]) && std::isnan(l2.theta[i]);
        for (int i = 0; i < n; ++i) same = same && std::abs(l2.r[i] - lm.r[i]) < 1e-12;
        EXPECT(nan && same, "the log map is NaN off the seed's piece, and the piece is solved by itself (equal to the single sphere's)");
        const VectorXd x2 = k2.transported(seed, 0.0);
        EXPECT(std::isnan(x2[2 * n]) && std::isnan(x2[2 * (n2 - 1)]), "the transport is NaN off the seed's piece");
        const nxr::chart::Basis b2 = k2.basis();
        EXPECT(std::isnan(b2.e1[3 * (n2 - 1)]) && std::abs(b2.e1[3 * seed] - bs.e1[3 * seed]) < 1e-15, "the basis is NaN at a vertex in no face and per piece elsewhere");
        bool refusedFree = false;
        try { k2.heatDistance(n2 - 1); } catch (const std::exception&) { refusedFree = true; }
        EXPECT(refusedFree, "a vertex in no face is refused");
        const VectorXd ds = k2.heatDistance(std::vector<int>{seed, n + 3});
        EXPECT(std::isfinite(ds[0]) && std::isnan(ds[n2 - 1]), "a source set is solved on the first source's piece (NaN elsewhere)");
    }

    // (4) the geodesic default (solve::heat) is the same solver on the same surface: exact again (it was 1e-4 against the port in G3)
    nxr::manifold::Manifold m(V.data(), n, F.data(), static_cast<int>(F.size() / 3));
    nxr::manifold::solve::HeatGeodesicSolver sDefault(m), sInput(m, 1.0, nxr::manifold::solve::HeatMesh::Input);
    const VectorXd dDefault = nxr::manifold::solve::heat(sDefault, {seed}), dInput = nxr::manifold::solve::heat(sInput, {seed}), dChart = kit.heatDistance(seed);
    EXPECT((dDefault - dChart).cwiseAbs().maxCoeff() < 1e-10, "the geodesic default (solve::heat) and the chart's heat distance are the same solver on the same surface (exact)");
    EXPECT(dInput.size() == dDefault.size() && (dInput - dDefault).cwiseAbs().maxCoeff() < 0.05, "HeatMesh::Input stays as the named option (geometry-central, input mesh)");
    std::cout << "  geodesic default vs chart: max |diff| " << sci((dDefault - dChart).cwiseAbs().maxCoeff()) << "; vs input-mesh option on the sphere: " << sci((dInput - dDefault).cwiseAbs().maxCoeff()) << "\n";
    std::cout << (g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}

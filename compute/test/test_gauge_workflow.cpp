#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/assembled.h"
#include <cmath>
#include <iostream>
#include "surface_of.h"
using namespace nxr::geometry;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

// There is no connection state: the connection is each function's argument, checked by geometry::checkSingularities (Gauss–Bonnet for the trivial one).
static void testDefaultsAndValidation() {
    std::cout << "\n=== connection: defaults + Gauss-Bonnet ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);
    EXPECT(nxr::geometry::eulerCharacteristic(m) == 2, "chi = 2");

    // valid singularities (sum == chi == 2)
    bool ok = true;
    try { nxr::geometry::checkSingularities(m, ConnectionType::Trivial, {{0,1.0},{3,1.0}}); }
    catch (...) { ok = false; }
    EXPECT(ok, "trivial with sum = 2 accepted");

    // invalid sum throws
    bool threw = false;
    try { nxr::geometry::checkSingularities(m, ConnectionType::Trivial, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "Gauss-Bonnet violation (sum=1 != chi=2) throws");

    // empty singularities throws
    threw = false;
    try { nxr::geometry::checkSingularities(m, ConnectionType::Trivial, {}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "trivial with empty singularities throws");

    // non-trivial connection with singularities throws (fail loud, not silent discard)
    threw = false;
    try { nxr::geometry::checkSingularities(m, ConnectionType::LeviCivita, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "non-trivial connection with singularities throws");
    ok = true;
    try { nxr::geometry::checkSingularities(m, ConnectionType::LeviCivita, {}); } catch (...) { ok = false; }
    EXPECT(ok, "Levi-Civita with no singularities accepted");

    // the operators check the connection they are given
    threw = false;
    try { (void)nxr::operators::vertexConnectionLaplacian(m, ConnectionType::Trivial, {{0,1.0}}); } catch (const Error&) { threw = true; }
    EXPECT(threw, "the connection Laplacian validates Gauss-Bonnet");
}

static void testGridRealization() {
    std::cout << "\n=== connection: grid realization ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);

    // Levi-Civita: grid == raw vertexGrid
    EXPECT((nxr::operators::connectionGrid(m, ConnectionType::LeviCivita, {}) - nxr::geometry::vertexGrid(m)).cwiseAbs().maxCoeff() < 1e-12, "LC grid == vertexGrid");

    // trivial
    std::map<int,double> sing{{0,1.0},{3,1.0}};
    Eigen::MatrixXcd triGrid = nxr::operators::connectionGrid(m, ConnectionType::Trivial, sing);
    EXPECT(triGrid.rows() == 12 && triGrid.cols() == 3, "trivial grid [12,3]");
    // realized trivial frame stays unit-tangent (|c row| preserved vs LC up to rotation)
    EXPECT(std::abs(triGrid.row(0).norm() - nxr::geometry::vertexGrid(m).row(0).norm()) < 1e-9,
           "trivial rotation preserves frame magnitude");

    // the same request on the same surface gives the same value (nothing is held between calls)
    const auto m2 = surfaceOf(V.data(), 12, F.data(), 20);
    EXPECT((nxr::operators::connectionGrid(m2, ConnectionType::Trivial, sing) - triGrid).cwiseAbs().maxCoeff() == 0.0,
           "the trivial grid is a function of its arguments");
}

int main() {
    testDefaultsAndValidation();
    testGridRealization();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}

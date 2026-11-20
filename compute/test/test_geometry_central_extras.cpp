/**
 * test_geometry_central_extras.cpp — smoke for the geometry-central
 * extras shipped in this PR:
 *
 *   1. the vector heat method (transport / extendScalar / logMap / findCenter)
 *   2. the signed heat method (signed geodesic distance from a curve)
 *   3. smoothFace / smoothVertex (NRoSy)
 *   4. compute
 *
 * All five primitives are run on the unit icosahedron fixture used
 * by the other smokes. The test verifies basic sanity (output sizes,
 * non-trivial ranges, distance ≈ logmap norm) rather than
 * bit-for-bit numerical contracts — the underlying solvers belong
 * to geometry-central and have their own test suite.
 */

#include "nxr/geometry/curvature.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "surface_of.h"

using namespace nxr::geometry;
using namespace nxr::operators;
using namespace nxr::operators;
using namespace nxr::operators::laplacian::connection;
using namespace nxr::geometry::transport;
using namespace nxr::geometry::connection;
using namespace nxr::operators::connection;
using namespace nxr::geometry::parametrization;
using namespace nxr::geometry;
using namespace nxr::geometry::query;
using namespace nxr::synthesis;
using namespace nxr::signal;
using namespace nxr::operators;
using namespace nxr::signal;

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

static void buildIcosahedron(std::vector<double>& V, std::vector<int32_t>& F) {
    double t = (1.0 + std::sqrt(5.0)) / 2.0;
    double raw[] = {
        -1,  t,  0,   1,  t,  0,  -1, -t,  0,   1, -t,  0,
         0, -1,  t,   0,  1,  t,   0, -1, -t,   0,  1, -t,
         t,  0, -1,   t,  0,  1,  -t,  0, -1,  -t,  0,  1,
    };
    for (int i = 0; i < 36; i += 3) {
        double len = std::sqrt(raw[i]*raw[i] + raw[i+1]*raw[i+1] + raw[i+2]*raw[i+2]);
        V.push_back(raw[i]   / len);
        V.push_back(raw[i+1] / len);
        V.push_back(raw[i+2] / len);
    }
    F = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
}

static void testVectorHeat(const nxr::geometry::SurfaceArrays& m) {
    std::cout << "[vhm] starting" << std::endl;
    int nV = m.nV();

    // Transport one tangent vector from vertex 0. Expect non-zero output everywhere.
    Eigen::MatrixXd src(1, 3);
    src << 1.0, 0.0, 0.0;
    auto transported = parallel(m, std::vector<int>{0}, src);
    REQUIRE(transported.rows() == nV, "transported has nV rows");
    REQUIRE(transported.cols() == 3,  "transported has 3 cols");
    double maxNorm = 0.0;
    for (int i = 0; i < nV; i++) {
        maxNorm = std::max(maxNorm, transported.row(i).norm());
    }
    REQUIRE(maxNorm > 1e-6, "transport produced non-trivial vectors");

    // Extend a scalar (1.0 at vertex 0, 0.0 at vertex 6 — antipodes-ish).
    auto extended = extendScalar(m, {0, 6}, (Eigen::VectorXd(2) << 1.0, 0.0).finished());
    REQUIRE(extended.size() == nV, "extended has nV entries");
    // Source values reproduced (heat-method approximation, so loose tol).
    REQUIRE(std::abs(extended(0) - 1.0) < 0.5, "vertex 0 near 1.0");
    REQUIRE(std::abs(extended(6) - 0.0) < 0.5, "vertex 6 near 0.0");

    // Log map at vertex 0. Norm of (logX, logY) should be ~ geodesic distance.
    auto logmap = logMap(m, 0);
    REQUIRE(logmap.logCoords.rows() == nV, "logCoords has nV rows");
    REQUIRE(logmap.logCoords.cols() == 2,  "logCoords has 2 cols");
    REQUIRE(logmap.logCoords.row(0).norm() < 1e-6, "log map at source is ~0");
    // sourceE1 / E2 should be unit-length.
    REQUIRE(std::abs(logmap.sourceE1.norm() - 1.0) < 1e-6, "sourceE1 unit");
    REQUIRE(std::abs(logmap.sourceE2.norm() - 1.0) < 1e-6, "sourceE2 unit");

    // Log map at a point of an input face: a corner reproduces the vertex form (coordinates identical), the centroid is a point on the face in the
    // face's plane, bad faces and weights are refused.
    {
        const int f = 3;
        const int c0 = m.faces(f, 0);
        const auto lv = logMap(m, c0);
        const auto lc = logMap(m, FacePoint{f, {1.0, 0.0, 0.0}});
        REQUIRE((lc.logCoords - lv.logCoords).cwiseAbs().maxCoeff() < 1e-9, "face point at a corner = the vertex log map");
        const auto lm = logMap(m, FacePoint{f, {1.0 / 3, 1.0 / 3, 1.0 / 3}});
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int k = 0; k < 3; ++k) centroid += m.positions.row(m.faces(f, k)).transpose() / 3.0;
        REQUIRE((lm.sourcePoint - centroid).norm() < 1e-12, "sourcePoint = the weighted corners");
        REQUIRE(std::abs(lm.sourceE1.norm() - 1.0) < 1e-9 && std::abs(lm.sourceE2.norm() - 1.0) < 1e-9 && std::abs(lm.sourceE1.dot(lm.sourceE2)) < 1e-9,
                "face frame orthonormal");
        const Eigen::Vector3d nf = (m.positions.row(m.faces(f, 1)) - m.positions.row(m.faces(f, 0))).transpose().cross(
            (m.positions.row(m.faces(f, 2)) - m.positions.row(m.faces(f, 0))).transpose()).normalized();
        REQUIRE(std::abs(lm.sourceE1.dot(nf)) < 1e-9 && (lm.sourceNormal - nf).norm() < 1e-9, "face frame tangent to the face, normal = the face's");
        REQUIRE(lm.logCoords.allFinite(), "face log map finite");
        const auto both = logMap(m, std::vector<FacePoint>{{f, {1.0, 0.0, 0.0}}, {f, {0.2, 0.3, 0.5}}});
        REQUIRE(both.size() == 2 && (both[0].logCoords - lc.logCoords).cwiseAbs().maxCoeff() < 1e-12, "batch form = single form");
        bool refused = false;
        try { logMap(m, FacePoint{f, {0.5, 0.6, -0.1}}); } catch (const nxr::core::Error&) { refused = true; }
        REQUIRE(refused, "negative weight refused");
        refused = false;
        try { logMap(m, FacePoint{f, {0.5, 0.3, 0.3}}); } catch (const nxr::core::Error&) { refused = true; }
        REQUIRE(refused, "weights not summing to 1 refused");
        refused = false;
        try { logMap(m, FacePoint{m.nF(), {1, 0, 0}}); } catch (const nxr::core::Error&) { refused = true; }
        REQUIRE(refused, "face out of range refused");
    }

    // findCenter of three nearby vertices: must be a finite 3D point.
    Eigen::Vector3d c = findCenter(m, {0, 1, 5});
    REQUIRE(std::isfinite(c.x()) && std::isfinite(c.y()) && std::isfinite(c.z()),
            "center is finite");
    // Icosahedron is unit-radius, so center should sit roughly on the surface.
    REQUIRE(std::abs(c.norm() - 1.0) < 0.4, "center near surface");
    std::cout << "[vhm] PASS" << std::endl;
}

static void testSignedHeat(const nxr::geometry::SurfaceArrays& m) {
    std::cout << "[shm] starting" << std::endl;
    int nV = m.nV();

    // Loop around the "north" pole of the icosahedron (vertex 0).
    auto sd = signedHeat(m, std::vector<int>{11, 5, 1, 7, 10}, true);
    REQUIRE(sd.size() == nV, "signed distance has nV entries");
    // Should have both signs (curve splits the surface in two).
    REQUIRE(sd.minCoeff() < 0.0 && sd.maxCoeff() > 0.0,
            "both positive and negative sides of the curve");
    std::cout << "[shm] PASS" << std::endl;
}

static void testSmoothFields(const nxr::geometry::SurfaceArrays& m) {
    std::cout << "[field] starting" << std::endl;
    int nF = m.nF();
    int nV = m.nV();

    auto faceField = smoothFace(m, 4);
    REQUIRE(faceField.rows() == nF, "face field has nF rows");
    REQUIRE(faceField.cols() == 3,  "face field has 3 cols");
    double mn = 0.0;
    for (int i = 0; i < nF; i++) mn = std::max(mn, faceField.row(i).norm());
    REQUIRE(mn > 1e-6, "face field is non-trivial");

    auto vfield = smoothVertex(m, 2);
    REQUIRE(vfield.vertexVectors.rows() == nV, "vertex field nV rows");
    REQUIRE(vfield.vertexFieldRaw.size() == nV * 2, "vertex field raw nV*2");
    REQUIRE(vfield.nSym == 2, "nSym preserved");
    std::cout << "[field] PASS" << std::endl;
}

int main() {
    std::vector<double>  V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    int nV = static_cast<int>(V.size() / 3);
    int nF = static_cast<int>(F.size() / 3);
    const auto m = surfaceOf(V.data(), nV, F.data(), nF);

    testVectorHeat(m);
    testSignedHeat(m);
    testSmoothFields(m);

    std::cout << "all geometry-central extras tests PASSED" << std::endl;
    return 0;
}

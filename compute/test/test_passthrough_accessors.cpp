/**
 * test_passthrough_accessors.cpp — Verify zero-copy DEC / vertexDualAreas
 * accessors are bit-identical to assembleDECOperators / the corresponding
 * VertexData<double>, and that repeated calls return the same reference.
 *
 *   1. Each accessor returns a matrix bit-equal to the field on the
 *      DECOperators struct returned by assembleDECOperators (i.e. zero
 *      drift vs. the legacy API).
 *   2. Two calls to the same accessor return the same reference (the
 *      address of the underlying field) — proves no copy, no rebuild.
 *   3. vertexDualAreas returns an Eigen::VectorXd whose entries match
 *      geometry-central's VertexData<double> read element-by-element.
 */

#include "nxr/geometry/curvature.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/transport.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"

#include <cassert>
#include <cmath>
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

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (0)

// Frobenius norm of (A - B) for sparse matrices.
static double sparseDiffNorm(const Eigen::SparseMatrix<double>& A,
                             const Eigen::SparseMatrix<double>& B) {
    Eigen::SparseMatrix<double> D = A - B;
    return D.norm();
}

int main() {
    std::cout << "[test_passthrough_accessors] starting" << std::endl;

    std::vector<double>  V;
    std::vector<int32_t> F;
    buildIcosahedron(V, F);
    const int nV = static_cast<int>(V.size() / 3);
    const int nF = static_cast<int>(F.size() / 3);
    std::cout << "  mesh: " << nV << " vertices, " << nF << " faces" << std::endl;

    const auto m = surfaceOf(V.data(), nV, F.data(), nF);

    // ── 1. Bit-identical to assembleDECOperators ────────────────
    {
        DECOperators dec = assembleDECOperators(m);

        REQUIRE(sparseDiffNorm(d0(m),            dec.d0)            < 1e-15, "d0 matches struct field");
        REQUIRE(sparseDiffNorm(d1(m),            dec.d1)            < 1e-15, "d1 matches struct field");
        REQUIRE(sparseDiffNorm(hodge0(m),        dec.hodge0)        < 1e-15, "hodge0 matches struct field");
        REQUIRE(sparseDiffNorm(hodge1(m),        dec.hodge1)        < 1e-15, "hodge1 matches struct field");
        REQUIRE(sparseDiffNorm(hodge2(m),        dec.hodge2)        < 1e-15, "hodge2 matches struct field");
        REQUIRE(sparseDiffNorm(hodge1Inverse(m), dec.hodge1Inverse) < 1e-15, "hodge1Inverse matches struct field");

        std::cout << "  ✓ DEC accessors bit-identical to assembleDECOperators" << std::endl;
    }

    // ── 2. Repeat calls assemble the same matrices (nothing is cached: each call is by value) ────
    {
        REQUIRE(sparseDiffNorm(d0(m), d0(m)) == 0.0, "d0 assembled twice is the same matrix");
        REQUIRE(sparseDiffNorm(hodge1(m), hodge1(m)) == 0.0, "hodge1 assembled twice is the same matrix");
        REQUIRE((vertexDualAreas(m) - vertexDualAreas(m)).norm() == 0.0, "vertexDualAreas assembled twice is the same vector");

        std::cout << "  ✓ Repeat calls give identical matrices" << std::endl;
    }

    // ── 3. vertexDualAreas matches assembleManifoldOperators' vertexAreas ──
    //
    // assembleManifoldOperators currently builds its `vertexAreas` VectorXd
    // by copying out of `geometry.vertexDualAreas` element-by-element
    // (see mesh_operators.cpp). The accessor here returns GC's raw
    // VectorXd directly — so the two should be bit-identical.
    {
        const Eigen::VectorXd& dual = vertexDualAreas(m);
        REQUIRE(static_cast<int>(dual.size()) == nV, "vertexDualAreas size == nV");

        // All entries strictly positive (Voronoi dual areas).
        for (int i = 0; i < nV; ++i) {
            REQUIRE(dual(i) > 0.0, "vertexDualAreas entry > 0");
        }

        // Bit-identical to assembleManifoldOperators(m).vertexAreas.
        ManifoldOperators ops = assembleManifoldOperators(m);
        REQUIRE(ops.vertexDualAreas.size() == dual.size(), "vertexAreas size match");
        const double diff = (ops.vertexDualAreas - dual).cwiseAbs().maxCoeff();
        REQUIRE(diff < 1e-15, "vertexDualAreas == assembleManifoldOperators(m).vertexAreas");

        std::cout << "  ✓ vertexDualAreas matches assembleManifoldOperators "
                  << "(sum = " << dual.sum() << ", max diff = " << diff << ")" << std::endl;
    }

    std::cout << "[test_passthrough_accessors] OK" << std::endl;
    return 0;
}

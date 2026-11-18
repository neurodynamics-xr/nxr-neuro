// Rows 22/23 of the toolbox refactor: the frames and normals of compute against geometry-central's (vertexNormals, faceTangentBasis), bit for bit.
// What is bit-equal is taken from geometry-central; the rest stays, named for the convention it keeps (printed: the largest difference).
#include "nxr/geometry/surface_arrays.h"
#include "nxr/geometry/frames.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <cstdio>
#include <iostream>
#include <random>

static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

int main() {
    for (const Mesh& m : {jitterSphere(icosphere(4), 0.3, 3), disc(10, 0.3, 5)}) {
        const auto s = nxr::geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF());
        nxr::geometry::IntrinsicSurface gs(s);
        auto& E = gs.embedded();
        E.requireVertexNormals(); E.requireFaceTangentBasis(); E.requireFaceNormals();
        const Eigen::MatrixXd N = nxr::geometry::normals(s, nxr::geometry::NormalType::AngleWeighted);
        double dn = 0;
        for (int v = 0; v < m.nV(); ++v) for (int c = 0; c < 3; ++c) dn = std::max(dn, std::abs(N(v, c) - E.vertexNormals[v][c]));
        std::printf("  angle-weighted normals vs geometry-central vertexNormals: %.3g\n", dn);
        CHECK(dn > 0.0 && dn < 1e-12, "angle-weighted normals: geometry-central's vertexNormals within 1e-12 but NOT bitwise (unit() vs /norm): ours kept");
        const nxr::geometry::FaceFrames F = nxr::geometry::frames(s);
        double de1 = 0, de2 = 0, dfn = 0;
        for (int f = 0; f < m.nF(); ++f) for (int c = 0; c < 3; ++c) {
            de1 = std::max(de1, std::abs(F.e1(f, c) - E.faceTangentBasis[f][0][c]));
            de2 = std::max(de2, std::abs(F.e2(f, c) - E.faceTangentBasis[f][1][c]));
            dfn = std::max(dfn, std::abs(F.normals(f, c) - E.faceNormals[f][c]));
        }
        std::printf("  face frames vs geometry-central faceTangentBasis: e1 %.3g, e2 %.3g, normal %.3g\n", de1, de2, dfn);
        CHECK(dfn == 0.0, "face normals ARE geometry-central's faceNormals (bitwise)");
        CHECK(de1 < 1e-12 && de2 < 1e-12 && (de1 > 0 || de2 > 0), "face frames: geometry-central's faceTangentBasis within 1e-12 but NOT bitwise (removeComponent): ours kept");
    }
    std::cout << (failures ? "FAILED " : "PASSED ") << failures << "\n";
    return failures ? 1 : 0;
}

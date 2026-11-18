// The intrinsic triangulation as a DATA PRODUCT (nxr/geometry/surface_arrays.h): exported to arrays and rebuilt from them, every geometry-central quantity and
// every operator is BIT-IDENTICAL to the one built on the original triangulation (same numbering, lengths, gauge). Meshes with flips, a boundary, the
// embedded option and each mass.
#include "nxr/geometry/surface_arrays.h"

#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"

#include <Eigen/Sparse>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace nxr::geometry;
namespace gcs = geometrycentral::surface;

static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

template <class S> static bool same(const Eigen::SparseMatrix<S>& a, const Eigen::SparseMatrix<S>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || a.nonZeros() != b.nonZeros()) return false;
    Eigen::SparseMatrix<S> A = a, B = b;
    A.makeCompressed(); B.makeCompressed();
    for (Eigen::Index i = 0; i < A.nonZeros(); ++i)
        if (A.valuePtr()[i] != B.valuePtr()[i] || A.innerIndexPtr()[i] != B.innerIndexPtr()[i]) return false;
    for (Eigen::Index j = 0; j <= A.outerSize(); ++j) if (A.outerIndexPtr()[j] != B.outerIndexPtr()[j]) return false;
    return true;
}
template <class T> static bool same(const std::vector<T>& a, const std::vector<T>& b) { return a == b; }

static void compare(const std::string& name, const Mesh& m, const IntrinsicOptions& o) {
    std::cout << name << "\n";
    IntrinsicSurface A(m.V.data(), m.nV(), m.F.data(), m.nF(), o);
    const SurfaceArrays s = surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF(), o);
    IntrinsicSurface B(s);
    if (o.intrinsic) CHECK(A.record().flips > 0 || name.find("noflip") != std::string::npos, "the construction flipped edges (" + std::to_string(A.record().flips) + ")");
    CHECK(same(A.cotanLaplacian(), B.cotanLaplacian()), "cotan Laplacian");
    CHECK(same(A.massMatrix(), B.massMatrix()), "mass");
    CHECK(same(A.galerkinMassMatrix(), B.galerkinMassMatrix()), "Galerkin mass");
    CHECK(same(A.barycentricMass(), B.barycentricMass()), "barycentric mass");
    CHECK(same(A.connectionLaplacian(1), B.connectionLaplacian(1)), "connection Laplacian n = 1");
    CHECK(same(A.connectionLaplacian(2), B.connectionLaplacian(2)), "connection Laplacian n = 2");
    const TangentBasis ta = A.tangentBasis(), tb = B.tangentBasis();
    CHECK(ta.e1 == tb.e1 && ta.e2 == tb.e2 && ta.n == tb.n, "tangent basis");
    const auto ca = A.complex(), cb = B.complex();
    CHECK(ca.faces == cb.faces && ca.edges == cb.edges && ca.faceEdges == cb.faceEdges && ca.length == cb.length && ca.angle == cb.angle
          && ca.angleSum == cb.angleSum && ca.reference == cb.reference, "complex (combinatorics, lengths, signposts)");
    const auto ia = A.inputCorners(), ib = B.inputCorners();
    CHECK(ia.signpost == ib.signpost && ia.angle == ib.angle && ia.angleSum == ib.angleSum && ia.boundary == ib.boundary, "input corners");
    auto& ga = A.geometry();
    auto& gb = B.geometry();
    ga.requireDECOperators(); gb.requireDECOperators();
    CHECK(same(ga.d0, gb.d0) && same(ga.d1, gb.d1) && same(ga.hodge0, gb.hodge0) && same(ga.hodge1, gb.hodge1) && same(ga.hodge2, gb.hodge2), "DEC operators");
    ga.requireTransportVectorsAlongHalfedge(); gb.requireTransportVectorsAlongHalfedge();
    ga.requireHalfedgeVectorsInFace(); gb.requireHalfedgeVectorsInFace();
    ga.requireVertexGaussianCurvatures(); gb.requireVertexGaussianCurvatures();
    bool okT = true, okF = true, okK = true;
    for (size_t h = 0; h < A.mesh().nHalfedges(); ++h) {
        if (!(ga.transportVectorsAlongHalfedge[h] == gb.transportVectorsAlongHalfedge[h])) { okT = false; std::printf("      transport h=%zu interior=%d\n", h, (int)A.mesh().halfedge(h).isInterior()); }
        if (A.mesh().halfedge(h).isInterior()) okF = okF && ga.halfedgeVectorsInFace[h] == gb.halfedgeVectorsInFace[h];   // an exterior halfedge has no face: geometry-central leaves its entry unset
    }
    for (size_t v = 0; v < A.mesh().nVertices(); ++v) okK = okK && ga.vertexGaussianCurvatures[v] == gb.vertexGaussianCurvatures[v];
    CHECK(okT, "transport along halfedges");
    CHECK(okF, "halfedge vectors in face (interior halfedges)");
    CHECK(okK, "Gaussian curvature");
    // the arrays of the rebuilt triangulation are the arrays it was rebuilt from
    const IntrinsicTriangulation t2 = B.triangulation();
    const IntrinsicTriangulation& t1 = s.triangulation;
    CHECK(t1.halfedgeNext == t2.halfedgeNext && t1.edgeLength == t2.edgeLength && t1.signpost == t2.signpost && t1.halfedgeVectorInVertex == t2.halfedgeVectorInVertex
          && t1.vertexAngleSum == t2.vertexAngleSum && t1.inputEdgeLength == t2.inputEdgeLength && t1.mass == t2.mass, "export of the rebuilt = the arrays");
}

int main() {
    IntrinsicOptions o;
    compare("jittered sphere (flips)", jitterSphere(icosphere(3), 0.35, 7), o);
    compare("uv sphere (slivers, noflip)", uvSphere(14, 22), o);
    compare("jittered uv sphere (flips)", jitterSphere(uvSphere(14, 22), 0.4, 5), o);
    compare("disc (boundary, flips)", disc(8, 0.3, 3), o);
    IntrinsicOptions bary = o; bary.mass = MassKind::Barycentric;
    compare("jittered sphere, barycentric mass", jitterSphere(icosphere(3), 0.35, 11), bary);
    IntrinsicOptions gal = o; gal.mass = MassKind::Galerkin;
    compare("jittered sphere, Galerkin mass", jitterSphere(icosphere(3), 0.35, 13), gal);
    IntrinsicOptions emb = o; emb.intrinsic = false;
    compare("embedded option (noflip)", jitterSphere(icosphere(3), 0.35, 7), emb);
    std::cout << (failures ? "FAILED " : "PASSED ") << failures << "\n";
    return failures ? 1 : 0;
}

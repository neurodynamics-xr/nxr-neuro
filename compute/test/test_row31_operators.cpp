// What the optical flow and the intrinsic forms read is BUILT FROM the operators module (decMatrices, cotanLaplacian, vertexMass, vertexConnectionLaplacian), not
// re-assembled. Each is bit-identical to what the geometry-central objects give directly (the construction the functions had before): the complex's d0, d1 and stars, the
// embedded cotan Laplacian and Galerkin mass, and the real connection Laplacian.
#include "nxr/geometry/surface_arrays.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "nxr/operators/dec.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/opticalflow.h"
#include "nxr/signal/opticalflow_connection.h"

#include <iostream>

using namespace nxr;
static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

template <class S> static bool same(const Eigen::SparseMatrix<S>& a, const Eigen::SparseMatrix<S>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || a.nonZeros() != b.nonZeros()) return false;
    Eigen::SparseMatrix<S> A = a, B = b;
    A.makeCompressed(); B.makeCompressed();
    for (Eigen::Index i = 0; i < A.nonZeros(); ++i) if (A.valuePtr()[i] != B.valuePtr()[i] || A.innerIndexPtr()[i] != B.innerIndexPtr()[i]) return false;
    for (Eigen::Index j = 0; j <= A.outerSize(); ++j) if (A.outerIndexPtr()[j] != B.outerIndexPtr()[j]) return false;
    return true;
}

int main() {
    std::cout << "test_row31_operators\n";
    const Mesh m = jitterSphere(icosphere(3), 0.35, 7);
    const geometry::SurfaceArrays s = geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF());
    geometry::IntrinsicSurface gs(s);

    // complexOf: d0, d1 from the signed face-edge incidence, the stars from geometry-central's DEC and the recorded mass
    const signal::Complex c = signal::complexOf(s);
    const auto K = gs.complex();
    std::vector<Eigen::Triplet<double>> t0, t1;
    for (int e = 0; e < c.nE; ++e) { t0.emplace_back(e, K.edges[2 * e], -1.0); t0.emplace_back(e, K.edges[2 * e + 1], 1.0); }
    for (int h = 0; h < 3 * c.nF; ++h) { const int se = K.faceEdges[h]; t1.emplace_back(h / 3, std::abs(se) - 1, se > 0 ? 1.0 : -1.0); }
    Eigen::SparseMatrix<double> d0(c.nE, c.nV), d1(c.nF, c.nE);
    d0.setFromTriplets(t0.begin(), t0.end()); d1.setFromTriplets(t1.begin(), t1.end());
    CHECK(same(c.d0, d0) && same(c.d1, d1), "complexOf d0, d1 = the signed incidence");
    auto& g = gs.geometry();
    g.requireDECOperators();
    const Eigen::VectorXd s1 = g.hodge1.diagonal(), s2 = g.hodge2.diagonal();
    const std::vector<double> mass = gs.massDiagonal();
    CHECK(c.star1 == s1 && c.star2 == s2, "complexOf star1, star2 = geometry-central's DEC");
    CHECK(c.star0 == Eigen::Map<const Eigen::VectorXd>(mass.data(), c.nV), "complexOf star0 = the recorded mass");

    // the embedded cotan Laplacian and Galerkin mass
    geometry::IntrinsicOptions io; io.intrinsic = false;
    geometry::IntrinsicSurface emb(m.V.data(), m.nV(), m.F.data(), m.nF(), io);
    Eigen::SparseMatrix<double> Kold = emb.cotanLaplacian();
    Kold = (0.5 * (Kold + Eigen::SparseMatrix<double>(Kold.transpose()))).eval();
    const Eigen::SparseMatrix<double> Mold = emb.galerkinMassMatrix();
    const Eigen::VectorXd massOld = Mold * Eigen::VectorXd::Ones(Mold.cols());
    Eigen::SparseMatrix<double> Knew; Eigen::VectorXd massNew;
    signal::embeddedLaplacian(geometry::RowMat3(Eigen::Map<const geometry::RowMat3>(m.V.data(), m.nV(), 3)), geometry::FaceMat(Eigen::Map<const geometry::FaceMat>(m.F.data(), m.nF(), 3)), Knew, massNew);
    CHECK(same(Knew, Kold) && massNew == massOld, "embeddedLaplacian = the embedded cotan Laplacian and Galerkin mass");

    // the real connection Laplacian
    const Eigen::SparseMatrix<std::complex<double>> Kc = gs.connectionLaplacian(1);
    std::vector<Eigen::Triplet<double>> t;
    for (int k = 0; k < Kc.outerSize(); ++k)
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(Kc, k); it; ++it) {
            const int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
            const double a = it.value().real(), b = it.value().imag();
            t.emplace_back(2 * i, 2 * j, a); t.emplace_back(2 * i, 2 * j + 1, -b); t.emplace_back(2 * i + 1, 2 * j, b); t.emplace_back(2 * i + 1, 2 * j + 1, a);
        }
    Eigen::SparseMatrix<double> C(2 * s.nV(), 2 * s.nV());
    C.setFromTriplets(t.begin(), t.end()); C.makeCompressed();
    CHECK(same(signal::connection::realConnectionLaplacian(s), C), "realConnectionLaplacian = the real form of geometry-central's connection Laplacian");

    std::cout << (failures ? "FAIL" : "PASS") << " test_row31_operators (" << failures << " failures)\n";
    return failures ? 1 : 0;
}

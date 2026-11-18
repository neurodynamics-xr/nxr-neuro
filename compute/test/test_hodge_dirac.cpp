// Hodge–Dirac d+δ on the intrinsic Delaunay mesh: structure, ±μ symmetry, the exact/coexact split against the dense
// pencil, spectrum slicing, the explicit-shift indefinite solve, the ⋆1 floor.
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "mesh_helpers.h"
#include "nxr/operators/registry/operator_registry.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
using namespace nxr::geometry;
using ops::hodgedirac::HodgeDirac;

static Eigen::MatrixXd dense(const Eigen::SparseMatrix<double>& S) { return Eigen::MatrixXd(S); }

static void testStructure() {
    std::cout << "\n=== hodgedirac::assemble: structure on the unit icosphere (level 2) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(2, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    EXPECT(h.nV == 162 && h.nE == 480 && h.nF == 320 && h.K.rows() == 962, "V,E,F = 162,480,320; N = 962");
    EXPECT((h.K - Eigen::SparseMatrix<double>(h.K.transpose())).norm() < 1e-14, "K symmetric");
    EXPECT(h.M.minCoeff() > 0, "M > 0");
    EXPECT((h.d1 * h.d0).norm() < 1e-14, "d1 d0 = 0");
    EXPECT(std::abs(h.star0.sum() - h.star2.cwiseInverse().sum()) < 1e-10, "sum star0 = total area = sum 1/star2");
    EXPECT(std::abs(h.star0.sum() - 4 * M_PI) / (4 * M_PI) < 0.02, "total area ~ 4 pi (inscribed polyhedron)");
    EXPECT((h.d0.transpose() * h.star1.asDiagonal() * h.d0 * Eigen::VectorXd::Ones(h.nV)).norm() < 1e-12, "cotan Laplacian kills constants");
    // K blocks: ⋆1 d0 under the 1-form/0-form block, ⋆2 d1 under 2/1
    Eigen::SparseMatrix<double> A = h.star1.asDiagonal() * h.d0;
    EXPECT((h.K.block(h.nV, 0, h.nE, h.nV) - A).norm() < 1e-14, "K[1-forms, 0-forms] = star1 d0");
    ops::hodgedirac::HodgeDirac hc = ops::hodgedirac::assemble(m, 10.0);   // floor above every weight
    EXPECT(hc.star1Clamped == hc.nE && hc.star1.minCoeff() >= 10.0, "star1 floor clamps (all edges at floor 10)");
}

static void testSpectrumAndSplit() {
    std::cout << "\n=== dense pencil: +-mu symmetry, zero modes, D^2 = Hodge Laplacians, the split ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(2, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    Eigen::MatrixXd Kd = dense(h.K), Md = dense(ops::hodgedirac::massMatrix(h));
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd, Md);
    Eigen::VectorXd mu = es.eigenvalues();
    int zeros = 0; std::vector<double> pos, neg;
    for (int i = 0; i < mu.size(); ++i) { if (std::abs(mu(i)) < 1e-8) ++zeros; else if (mu(i) > 0) pos.push_back(mu(i)); else neg.push_back(-mu(i)); }
    std::sort(pos.begin(), pos.end()); std::sort(neg.begin(), neg.end());
    EXPECT(zeros == 2, "2 zero modes on a sphere (constant 0-form, constant 2-form), got " + std::to_string(zeros));
    double sym = 0; if (pos.size() == neg.size()) for (size_t i = 0; i < pos.size(); ++i) sym = std::max(sym, std::abs(pos[i] - neg[i]) / pos[i]);
    EXPECT(pos.size() == neg.size() && sym < 1e-9, "spectrum symmetric +-mu to " + std::to_string(sym));
    EXPECT(std::abs(pos[0] * pos[0] - 2.0) < 0.1, "lowest mu^2 ~ 2 (l(l+1), l=1), got " + std::to_string(pos[0] * pos[0]));

    const int k = 40;
    nxr::operators::HodgeSolveOptions lim; lim.limit = k;
    nxr::operators::EigenResult r = nxr::operators::hodgeDiracModes(h, 2 * k + 8, false, lim);   // the reference holds ABOUT count modes (the single λ cut): ask more, keep the k lowest
    EXPECT(r.k == k && r.kind.size() == k, "40 modes with kinds");
    double dmu = 0; for (int i = 0; i < k; ++i) dmu = std::max(dmu, std::abs(r.eigenvalues(i) - pos[i]) / pos[i]);
    EXPECT(dmu < 1e-8, "split mu == dense pencil mu (lowest 40 positive), max rel diff " + std::to_string(dmu));
    Eigen::MatrixXd X = r.eigenvectors;
    double res = (Kd * X - Md * X * r.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-9, "K x = mu M x residual " + std::to_string(res));
    double orth = (X.transpose() * Md * X - Eigen::MatrixXd::Identity(k, k)).cwiseAbs().maxCoeff();
    EXPECT(orth < 1e-9, "modes M-orthonormal " + std::to_string(orth));
    int nEx = (r.kind.array() == 0).count(), nCo = (r.kind.array() == 2).count();
    EXPECT(nEx > 0 && nCo > 0 && nEx + nCo == k, "both kinds present: exact " + std::to_string(nEx) + ", coexact " + std::to_string(nCo));
    nxr::operators::EigenResult rs = nxr::operators::hodgeDiracModes(h, 28, /*bothSigns=*/true, [] { nxr::operators::HodgeSolveOptions o; o.limit = 10; return o; }());
    double mres = (Kd * rs.eigenvectors - Md * rs.eigenvectors * rs.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(rs.k == 20 && rs.eigenvalues(10) == -rs.eigenvalues(9) && std::is_sorted(rs.eigenvalues.data(), rs.eigenvalues.data() + 20) && mres < 1e-9, "bothSigns: -mu mirror (1-form block negated) is an eigenvector, residual " + std::to_string(mres));
    // the kind from the dense eigenvectors agrees with the split's kind
    Eigen::VectorXd dm(k); Eigen::MatrixXd dX(h.K.rows(), k); int c = 0;
    for (int i = 0; i < mu.size() && c < k; ++i) if (mu(i) > 1e-8) { dm(c) = mu(i); dX.col(c) = es.eigenvectors().col(i); ++c; }
    Eigen::VectorXi dk = nxr::operators::classifyHodgeModes(h, dX, dm);
    int agree = 0; for (int i = 0; i < k; ++i) agree += (dk(i) == r.kind(i));
    EXPECT(agree >= k - 4, "classifyHodgeModes on the dense eigenvectors matches the split's kinds (" + std::to_string(agree) + "/40; mixed inside degenerate pairs)");
}

static void testSlicingAndShift() {
    std::cout << "\n=== spectrum slicing and the explicit-shift indefinite solve (level 3, 642 vertices) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(3, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    Eigen::SparseMatrix<double> L0 = h.d0.transpose() * h.star1.asDiagonal() * h.d0, M0(h.nV, h.nV);
    M0 = h.star0.asDiagonal();
    auto one = nxr::operators::eigenSliced(L0, M0, 150, 400);          // one window
    auto sl  = nxr::operators::eigenSliced(L0, M0, 150, 30);           // 5+ windows
    double d = (one.eigenvalues - sl.eigenvalues).cwiseAbs().maxCoeff();
    EXPECT(sl.k == 150 && d < 1e-7, "sliced (30/window) == single window, 150 modes, max diff " + std::to_string(d));
    double res = (L0 * sl.eigenvectors - M0 * sl.eigenvectors * sl.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-7, "sliced residual " + std::to_string(res));
    double orth = (sl.eigenvectors.transpose() * M0 * sl.eigenvectors - Eigen::MatrixXd::Identity(150, 150)).cwiseAbs().maxCoeff();
    EXPECT(orth < 1e-6, "sliced modes M-orthonormal " + std::to_string(orth));
    // sphere: lambda = l(l+1) with multiplicity 2l+1 -> the first 3 nonconstant are ~2
    EXPECT(std::abs(sl.eigenvalues(1) - 2.0) < 0.05 && std::abs(sl.eigenvalues(3) - 2.0) < 0.05 && std::abs(sl.eigenvalues(4) - 6.0) < 0.2, "lambda_2..4 ~ 2, lambda_5 ~ 6");

    // thousands of modes through the split: ask for 300 positive modes
    nxr::operators::EigenResult r = nxr::operators::hodgeDiracModes(h, 620, false, [] { nxr::operators::HodgeSolveOptions o; o.limit = 300; o.slice = 60; o.normalize = true; return o; }());
    EXPECT(r.k == 300 && std::is_sorted(r.eigenvalues.data(), r.eigenvalues.data() + 300), "300 +mu modes, sorted");
    Eigen::SparseMatrix<double> M = ops::hodgedirac::massMatrix(h);
    double res2 = (h.K * r.eigenvectors - M * r.eigenvectors * r.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(res2 < 1e-6, "300-mode residual " + std::to_string(res2));
    {
        Eigen::SparseMatrix<double> s1(h.nE, h.nE), s1i(h.nE, h.nE);
        s1 = h.star1.asDiagonal(); s1i = h.star1.cwiseInverse().asDiagonal();
        Eigen::SparseMatrix<double> Bt = h.star2.asDiagonal() * h.d1, S2 = Bt * s1i * Bt.transpose(), M2(h.nF, h.nF);
        M2 = h.star2.asDiagonal();
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> e0(dense(L0), dense(M0)), e2(dense(S2), dense(M2));
        std::vector<double> all;
        for (int i = 0; i < e0.eigenvalues().size(); ++i) if (e0.eigenvalues()(i) > 1e-6) all.push_back(std::sqrt(e0.eigenvalues()(i)));
        for (int i = 0; i < e2.eigenvalues().size(); ++i) if (e2.eigenvalues()(i) > 1e-6) all.push_back(std::sqrt(e2.eigenvalues()(i)));
        std::sort(all.begin(), all.end());
        double dmu = 0; for (int i = 0; i < 300; ++i) dmu = std::max(dmu, std::abs(r.eigenvalues(i) - all[i]));
        EXPECT(dmu < 1e-7, "300 +mu modes == dense ladders' sqrt(lambda) (none lost, none repeated), max diff " + std::to_string(dmu));
        Eigen::MatrixXd G = r.eigenvectors.transpose() * M * r.eigenvectors - Eigen::MatrixXd::Identity(300, 300);
        EXPECT(G.cwiseAbs().maxCoeff() < 1e-8, "300 modes M-orthonormal (no duplicates: no off-diagonal near 1), max |G - I| " + std::to_string(G.cwiseAbs().maxCoeff()));
    }

    // harder: windows of 12 on a sphere whose multiplets (2l+1, split by the icosahedral symmetry) straddle many window edges;
    // every eigenvalue against the dense solve (multiplicities included), M-orthonormal, no duplicate vectors.
    {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> dn(dense(L0), dense(M0));
        for (int slice : {12, 17, 40}) {
            const int cnt = 220;
            auto t = nxr::operators::eigenSliced(L0, M0, cnt, slice);
            double dv = (t.eigenvalues - dn.eigenvalues().head(cnt)).cwiseAbs().maxCoeff();
            Eigen::MatrixXd G = t.eigenvectors.transpose() * M0 * t.eigenvectors - Eigen::MatrixXd::Identity(cnt, cnt);
            double dr = (L0 * t.eigenvectors - M0 * t.eigenvectors * t.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
            EXPECT(t.k == cnt && dv < 1e-7, "sliced " + std::to_string(slice) + "/window, 220 modes == dense eigenvalues (multiplets complete), max diff " + std::to_string(dv));
            EXPECT(G.cwiseAbs().maxCoeff() < 1e-8 && dr < 1e-7, "sliced " + std::to_string(slice) + "/window M-orthonormal " + std::to_string(G.cwiseAbs().maxCoeff()) + ", residual " + std::to_string(dr));
        }
    }

    // explicit shift on the indefinite pencil: 8 nearest 3.0, both signs
    nxr::operators::EigenOperatorSpec spec{nxr::operators::EigenOperator::HodgeDirac};
    nxr::operators::EigenResult s = nxr::operators::eigenOperator(m, spec, 8, 0.01, true);
    double rs = (h.K * s.eigenvectors - M * s.eigenvectors * s.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    EXPECT(s.k == 8 && s.kind.size() == 8 && rs < 1e-7, "eigenOperator(HodgeDirac, sigma=0.01): 8 pairs, kinds, residual " + std::to_string(rs));
    bool hasZero = false, hasNeg = false;
    for (int i = 0; i < s.k; ++i) { hasZero |= std::abs(s.eigenvalues(i)) < 1e-6; hasNeg |= s.eigenvalues(i) < -1e-6; }
    EXPECT(hasZero && hasNeg, "window around 0 holds the two harmonic zero modes and negative mu");
}

static void testRemoveDCAndRegistry() {
    std::cout << "\n=== removeDC removes the whole null space; the registry says what hodgeDirac is ===\n";
    // two components (two zero modes) and a 4-fold zero (relative Dirac); kinds kept in step
    nxr::operators::EigenResult r;
    r.k = 7; r.nConverged = 7;
    r.eigenvalues.resize(7); r.eigenvalues << -3e-15, 1e-15, 2e-14, 0.0, 3.0, 5.0, 9.0;
    r.eigenvectors = Eigen::MatrixXd::Identity(7, 7);
    r.kind.resize(7); r.kind << 1, 1, 1, 1, 0, 2, 0;
    int nrem = -1;
    nxr::operators::EigenResult o = nxr::operators::removeDC(r, &nrem);
    EXPECT(nrem == 4 && o.k == 3 && o.eigenvalues(0) == 3.0 && o.kind(1) == 2 && o.eigenvectors.cols() == 3, "removeDC drops all 4 null modes, reports 4, keeps kinds");
    int n0 = -1;
    nxr::operators::EigenResult o2 = nxr::operators::removeDC(o, &n0);
    EXPECT(n0 == 0 && o2.k == 3, "removeDC on a result with no null mode removes nothing");
    const auto* hd = nxr::operators::registry::operatorById("hodgeDirac");
    EXPECT(hd && hd->domain == nxr::operators::registry::Domain::mixed, "registry: hodgeDirac domain is mixed (V+E+F forms)");
}

static void testReferenceMethod() {
    std::cout << "\n=== the reference method (nxr, Rheome) and its named options (level 3, 642 vertices) ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(3, V, F);
    Manifold m(V.data(), (int)V.size() / 3, F.data(), (int)F.size() / 3);
    HodgeDirac h = ops::hodgedirac::assemble(m);
    EXPECT(h.star1Clamped == 0, "no *1 floor by default (nxr), clamped " + std::to_string(h.star1Clamped));
    const int count = 100;
    nxr::operators::EigenResult ref = nxr::operators::hodgeDiracModes(h, count);
    const int want = count / 2 + 1;
    EXPECT(ref.k >= count - 12 && ref.k <= 2 * want, "about count modes under the single lambda cut: " + std::to_string(ref.k) + " for " + std::to_string(count));
    nxr::operators::HodgeSolveOptions rel; rel.shift = nxr::operators::HodgeShift::RelativeLadder; rel.normalize = true;
    nxr::operators::EigenResult rr = nxr::operators::hodgeDiracModes(h, count, false, rel);
    nxr::operators::HodgeSolveOptions sl; sl.slice = 40; sl.normalize = true;
    nxr::operators::EigenResult rs = nxr::operators::hodgeDiracModes(h, count, false, sl);
    EXPECT(rr.k == ref.k && rs.k == ref.k, "relative and sliced options keep the same modes (" + std::to_string(rr.k) + ", " + std::to_string(rs.k) + ")");
    // the dense ladders are the truth: every option against them
    Eigen::SparseMatrix<double> s1(h.nE, h.nE), s1i(h.nE, h.nE), M0(h.nV, h.nV), M2(h.nF, h.nF);
    s1 = h.star1.asDiagonal(); s1i = h.star1.cwiseInverse().asDiagonal(); M0 = h.star0.asDiagonal(); M2 = h.star2.asDiagonal();
    Eigen::SparseMatrix<double> L0 = h.d0.transpose() * s1 * h.d0, Bt = h.star2.asDiagonal() * h.d1, S2 = Bt * s1i * Bt.transpose();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> e0(dense(L0), dense(M0)), e2(dense(S2), dense(M2));
    std::vector<double> all;
    for (int i = 0; i < e0.eigenvalues().size(); ++i) if (e0.eigenvalues()(i) > 1e-6) all.push_back(std::sqrt(e0.eigenvalues()(i)));
    for (int i = 0; i < e2.eigenvalues().size(); ++i) if (e2.eigenvalues()(i) > 1e-6) all.push_back(std::sqrt(e2.eigenvalues()(i)));
    std::sort(all.begin(), all.end());
    double d0 = 0, d1 = 0, d2 = 0;
    for (int i = 0; i < ref.k; ++i) d0 = std::max(d0, std::abs(ref.eigenvalues(i) - all[i]));
    for (int i = 0; i < std::min(rr.k, rs.k); ++i) { d1 = std::max(d1, std::abs(rr.eigenvalues(i) - all[i])); d2 = std::max(d2, std::abs(rs.eigenvalues(i) - all[i])); }
    EXPECT(d0 < 1e-9, "reference mu == dense ladders (none lost, none repeated), max diff " + std::to_string(d0));
    EXPECT(d2 < 1e-8, "sliced option mu == dense ladders, max diff " + std::to_string(d2));
    std::cout << "  (relative ladder, one solve, vs dense: max diff " << d1 << ")\n";
    Eigen::SparseMatrix<double> M = ops::hodgedirac::massMatrix(h);
    double res = (h.K * ref.eigenvectors - M * ref.eigenvectors * ref.eigenvalues.asDiagonal()).cwiseAbs().maxCoeff();
    double orth = (ref.eigenvectors.transpose() * M * ref.eigenvectors - Eigen::MatrixXd::Identity(ref.k, ref.k)).cwiseAbs().maxCoeff();
    EXPECT(res < 1e-9, "reference residual " + std::to_string(res));
    EXPECT(orth < 1e-8, "reference modes M-orthonormal without whitening " + std::to_string(orth));
}

int main() {
    testStructure();
    testSpectrumAndSplit();
    testSlicingAndShift();
    testRemoveDCAndRegistry();
    testReferenceMethod();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}

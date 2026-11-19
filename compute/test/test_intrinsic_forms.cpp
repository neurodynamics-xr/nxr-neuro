// nxr::signal on the INTRINSIC complex of a sphere (IntrinsicSurface of an icosphere whose vertices are jittered, so edges flip):
//   d1·d0 = 0; ⋆1 ≥ 0 (Delaunay); an unflipped edge's direction at each end is the mesh edge's (the signposts embedded through the input's edges);
//   flat of a GRADIENT field ∇f (f = z, its tangential part) ≈ d0 f, the exact edge differences (the field integrated along each edge);
//   sharp(flat(J)) ≈ J for smooth tangent fields (a gradient and a rotation n × ẑ), away from nothing (a closed sphere);
//   area2form / vertexDensity: a constant density integrates to the area and comes back constant;
//   on an OPEN grid: flat and sharp exact for a constant field at every edge and vertex, the boundary included.
#include <cmath>
#include <iostream>
#include <complex>
#include <random>

#include <Eigen/Eigenvalues>

#include "mesh_helpers.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
    std::cout << "test_intrinsic_forms\n";
    std::vector<double> V; std::vector<int32_t> F;
    icosphere(4, V, F, 1.0);
    std::mt19937 rng(5); std::normal_distribution<double> nd(0, 0.15);
    const int nV = static_cast<int>(V.size() / 3);
    // jitter along the sphere (tangentially, then back onto it): thin and obtuse triangles, so the construction flips edges
    for (int v = 0; v < nV; ++v) {
        Eigen::Vector3d p(V[3 * v], V[3 * v + 1], V[3 * v + 2]);
        p += Eigen::Vector3d(nd(rng), nd(rng), nd(rng)) * 0.06;
        p.normalize();
        for (int k = 0; k < 3; ++k) V[3 * v + k] = p(k);
    }
    nxr::geometry::IntrinsicSurface m(V.data(), nV, F.data(), static_cast<int>(F.size() / 3));
    const nxr::signal::IntrinsicComplex c = nxr::signal::intrinsicComplex(nxr::geometry::surfaceArrays(V.data(), nV, F.data(), static_cast<int>(F.size() / 3)));
    std::printf("  %d vertices, %d edges, %d faces; %d flips\n", c.nV, c.nE, c.nF, m.record().flips);
    EXPECT(m.record().flips > 0, "the jittered sphere needed flips");
    EXPECT(Eigen::SparseMatrix<double>(c.d1 * c.d0).norm() == 0.0, "d1 d0 = 0");
    EXPECT(c.star1.minCoeff() >= -1e-9, "every cotan weight >= 0 (Delaunay)");
    {
        double worst0 = 0, worst1 = 0; int n = 0;
        for (int e = 0; e < c.nE; ++e) {
            const int a = c.edges[2 * e], b = c.edges[2 * e + 1];
            Eigen::Vector3d ch = c.positions.row(b) - c.positions.row(a);
            if (std::abs(ch.norm() - c.length(e)) > 1e-6 * c.length(e)) continue;   // a flipped edge: its chord is not its path
            const Eigen::Vector3d na = c.normals.row(a), nb = c.normals.row(b);
            const Eigen::Vector3d ca = (ch - ch.dot(na) * na).normalized(), cb = (ch - ch.dot(nb) * nb).normalized();
            worst0 = std::max(worst0, 1 - nxr::signal::tangentAt(c, e, false).dot(ca));
            worst1 = std::max(worst1, 1 - nxr::signal::tangentAt(c, e, true).dot(cb));
            ++n;
        }
        EXPECT(worst0 < 1e-4 && worst1 < 1e-4, "an unflipped edge's direction at each end is the mesh edge's (" << n << " edges; worst 1 - cos " << std::max(worst0, worst1) << ")");
    }
    // fields: f = z; its tangential gradient; a rotation about z (n × ẑ)
    Eigen::VectorXd f(nV);
    Eigen::MatrixXd G(3 * nV, 1), R(3 * nV, 1);
    for (int v = 0; v < nV; ++v) {
        const Eigen::Vector3d n = c.normals.row(v), z(0, 0, 1);
        f(v) = c.positions(v, 2);
        G.middleRows(3 * v, 3) = z - z.dot(n) * n;
        R.middleRows(3 * v, 3) = n.cross(z);
    }
    const Eigen::VectorXd w = nxr::signal::flat(c, G).col(0), df = c.d0 * f;
    const double eFlat = (w - df).norm() / df.norm();
    EXPECT(eFlat < 0.03, "flat(∇z) = d0 z, the exact edge differences (relative error " << eFlat << ")");
    auto roundTrip = [&](const Eigen::MatrixXd& J) {
        const Eigen::MatrixXd back = nxr::signal::sharp(c, nxr::signal::flat(c, J));
        double num = 0, den = 0;
        for (int v = 0; v < nV; ++v) { num += c.star0(v) * (back.middleRows(3 * v, 3) - J.middleRows(3 * v, 3)).squaredNorm(); den += c.star0(v) * J.middleRows(3 * v, 3).squaredNorm(); }
        return std::sqrt(num / den);
    };
    const double eG = roundTrip(G), eR = roundTrip(R);
    EXPECT(eG < 0.02 && eR < 0.02, "sharp(flat(J)) = J for a gradient (" << eG << ") and a rotation (" << eR << ")");
    const Eigen::MatrixXd one = Eigen::MatrixXd::Ones(nV, 1);
    const Eigen::MatrixXd s = nxr::signal::area2form(c, one), back = nxr::signal::vertexDensity(c, s);
    EXPECT(std::abs(s.sum() - c.star2.cwiseInverse().sum()) < 1e-9 && (back.array() - 1).abs().maxCoeff() < 1e-12, "a constant density: integrates to the area, comes back constant");
    // THE WEIGHTED POTENTIAL FITS: a smooth potential's exact edge differences, with large errors on 5 % of the edges weighted down: both fits recover it;
    // weighting every edge alike does not. The modes fit on a smaller sphere whose exact modes (the intrinsic Laplacian's) are solved densely here.
    {
        std::vector<double> V3; std::vector<int32_t> F3;
        icosphere(3, V3, F3, 1.0);
        const int n3 = static_cast<int>(V3.size() / 3);
        for (int v = 0; v < n3; ++v) {
            Eigen::Vector3d p(V3[3 * v], V3[3 * v + 1], V3[3 * v + 2]);
            p += Eigen::Vector3d(nd(rng), nd(rng), nd(rng)) * 0.06; p.normalize();
            for (int k = 0; k < 3; ++k) V3[3 * v + k] = p(k);
        }
        nxr::geometry::IntrinsicSurface m3(V3.data(), n3, F3.data(), static_cast<int>(F3.size() / 3));
        const nxr::signal::IntrinsicComplex c3 = nxr::signal::intrinsicComplex(nxr::geometry::surfaceArrays(V3.data(), n3, F3.data(), static_cast<int>(F3.size() / 3)));
        const Eigen::MatrixXd Lap = Eigen::MatrixXd(c3.d0.transpose() * c3.star1.asDiagonal() * c3.d0);
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Lap, Eigen::MatrixXd(c3.star0.asDiagonal()));
        const int K = 40;
        nxr::signal::Modes md;
        md.P = ges.eigenvectors().middleCols(1, K) / std::sqrt(2.0);          // ψ/√2, the constant left out
        md.X2 = Eigen::MatrixXd::Zero(c3.nF, K);
        md.mu = ges.eigenvalues().segment(1, K).cwiseSqrt();
        md.kind.assign(K, 0);
        Eigen::VectorXd a = Eigen::VectorXd::Zero(K);
        for (int k = 0; k < 12; ++k) a(k) = nd(rng);
        const Eigen::VectorXd f0 = md.P * a;
        Eigen::VectorXd om = c3.d0 * f0, wgt = Eigen::VectorXd::Ones(c3.nE);
        std::uniform_real_distribution<double> ud(0, 1);
        for (int e = 0; e < c3.nE; ++e) if (ud(rng) < 0.05) { om(e) += 3.0 * nd(rng) / 0.15; wgt(e) = 1e-4; }
        auto corr = [&](const Eigen::VectorXd& x) { const Eigen::VectorXd p = x.array() - x.mean(), q = f0.array() - f0.mean(); return p.dot(q) / (p.norm() * q.norm()); };
        const double rMesh = corr(nxr::signal::weightedPotentialMesh(c3, wgt, om));
        const double rModes = corr(nxr::signal::weightedPotentialModes(c3, md, std::vector<char>(K, 1), wgt, om));
        const double rEqual = corr(nxr::signal::weightedPotentialMesh(c3, Eigen::VectorXd::Ones(c3.nE), om));
        std::printf("  weighted fits with 5 %% of edges corrupted and weighted down: mesh r %.5f, modes r %.5f; every edge alike r %.3f\n", rMesh, rModes, rEqual);
        EXPECT(rMesh > 0.9999 && rModes > 0.9999, "the weighted fits recover the potential (mesh " << rMesh << ", modes " << rModes << ")");
        EXPECT(rEqual < 0.99, "weighting every edge alike does not (" << rEqual << ")");
    }
    // patternFields on a small sphere's intrinsic Laplacian modes (solved densely here): a coexact mode's tangent coefficient c gives S = c P_k / μ_k, an
    // exact one's gives A, a normal coefficient gives its density; a normal row of an exact mode is refused
    {
        std::vector<double> V2; std::vector<int32_t> F2;
        icosphere(2, V2, F2, 1.0);
        const int n2 = static_cast<int>(V2.size() / 3);
        nxr::geometry::IntrinsicSurface m2(V2.data(), n2, F2.data(), static_cast<int>(F2.size() / 3));
        const nxr::signal::IntrinsicComplex c2 = nxr::signal::intrinsicComplex(nxr::geometry::surfaceArrays(V2.data(), n2, F2.data(), static_cast<int>(F2.size() / 3)));
        const Eigen::MatrixXd Lap = Eigen::MatrixXd(c2.d0.transpose() * c2.star1.asDiagonal() * c2.d0);
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(Lap, Eigen::MatrixXd(c2.star0.asDiagonal()));
        const int K = 12;
        nxr::signal::Modes mp;
        mp.P = ges.eigenvectors().middleCols(1, K) / std::sqrt(2.0);
        mp.mu = ges.eigenvalues().segment(1, K).cwiseSqrt();
        mp.kind.assign(K, 0);
        mp.kind[3] = 2;                                                        // treated as coexact: its 2-form part a density's area form
        mp.X2 = Eigen::MatrixXd::Zero(c2.nF, K);
        mp.X2.col(3) = nxr::signal::area2form(c2, Eigen::MatrixXd(mp.P.col(5))).col(0);
        const std::complex<double> cA(0.4, -1.2), cB(2.0, 0.5), cN(-0.3, 0.8);
        const auto pf = nxr::signal::patternFields(c2, mp, {{3, 1}, {7, 1}, {3, 2}}, Eigen::Vector3cd(cA, cB, cN));
        const Eigen::MatrixXd d3 = nxr::signal::vertexDensity(c2, Eigen::MatrixXd(mp.X2.col(3)));
        const double eS = (pf.S - cA * mp.P.col(3).cast<std::complex<double>>() / mp.mu(3)).norm(), eA = (pf.A - cB * mp.P.col(7).cast<std::complex<double>>() / mp.mu(7)).norm();
        const double eN = (pf.rho - cN * d3.col(0).cast<std::complex<double>>()).norm();
        EXPECT(eS < 1e-12 && eA < 1e-12 && eN < 1e-12, "patternFields: S from a coexact tangent coefficient, A from an exact one, rho from a normal one");
        bool bad = false;
        try { nxr::signal::patternFields(c2, mp, {{7, 2}}, Eigen::VectorXcd::Ones(1)); } catch (const std::invalid_argument&) { bad = true; }
        EXPECT(bad, "patternFields refuses a normal row of an exact mode");
    }
    {   // the Hodge–Dirac modes read through modesOf: projection and synthesis invert each other on the modes' span, d0 of the exact potential is the
        // exact part, coexactStream is splitPhase's stream, phaseCurrent is |z_tail||z_tip| sin of the phase step; mismatched modes are refused
        std::vector<double> V2; std::vector<int32_t> F2;
        icosphere(2, V2, F2, 1.0);
        const int n2 = static_cast<int>(V2.size() / 3), f2 = static_cast<int>(F2.size() / 3);
        const auto s2 = nxr::geometry::surfaceArrays(V2.data(), n2, F2.data(), f2);
        const auto h = nxr::operators::hodgedirac::assemble(s2);
        nxr::operators::HodgeSolveOptions o; o.limit = 30;
        const auto r = nxr::operators::hodgeDiracModes(h, 40, false, o);
        const nxr::signal::Modes md = nxr::signal::modesOf(r, h.nV, h.nE, h.nF);
        const nxr::signal::IntrinsicComplex c2 = nxr::signal::intrinsicComplex(s2);
        EXPECT((c2.star1 - h.star1).cwiseAbs().maxCoeff() == 0.0, "the complex's star1 is the pencil's");
        const int K = static_cast<int>(md.mu.size());
        int nEx = 0; for (int k : md.kind) nEx += k == 0;
        EXPECT(K == 30 && md.P.rows() == n2 && md.X2.rows() == c2.nF && nEx > 0 && nEx < K, "modesOf: 30 modes, both kinds, the 0- and 2-form rows");
        Eigen::MatrixXd coef = Eigen::MatrixXd::Zero(K, 2);
        for (int k = 0; k < K; ++k) { coef(k, 0) = nd(rng); coef(k, 1) = nd(rng); }
        const Eigen::MatrixXd w = nxr::signal::synthesizeOneForm(c2, md, coef);
        const double eRT = (nxr::signal::projectOneForm(c2, md, w) - coef).cwiseAbs().maxCoeff();
        EXPECT(eRT < 1e-8, "projectOneForm(synthesizeOneForm(c)) = c (" << eRT << ")");
        Eigen::MatrixXd ce = coef;
        for (int k = 0; k < K; ++k) if (md.kind[k] != 0) ce.row(k).setZero();
        const double eP = (c2.d0 * nxr::signal::exactPotential(md, coef) - nxr::signal::synthesizeOneForm(c2, md, ce)).cwiseAbs().maxCoeff();
        EXPECT(eP < 1e-10, "d0 exactPotential = the exact part (" << eP << ")");
        const auto sp = nxr::signal::splitPhase(c2, md, w.col(0));
        EXPECT((nxr::signal::coexactStream(c2, md, coef).col(0) - sp.stream).cwiseAbs().maxCoeff() < 1e-12, "coexactStream = splitPhase's stream");
        Eigen::MatrixXcd z(n2, 2);
        for (int v = 0; v < n2; ++v) {
            z(v, 0) = std::polar(1.0 + 0.5 * V2[3 * v + 2], 3.0 * V2[3 * v]);
            z(v, 1) = std::polar(2.0, -V2[3 * v + 1]);
        }
        const Eigen::MatrixXd J = nxr::signal::phaseCurrent(c2, z);
        double eJ = 0;
        for (int col = 0; col < 2; ++col) {
            const Eigen::VectorXd dphi = nxr::signal::phaseGradient(c2, Eigen::MatrixXcd(z.col(col))).col(0);
            for (int e = 0; e < c2.nE; ++e) {
                const double want = std::abs(z(c2.edges[2 * e], col)) * std::abs(z(c2.edges[2 * e + 1], col)) * std::sin(dphi(e));
                eJ = std::max(eJ, std::abs(J(e, col) - want));
            }
        }
        EXPECT(J.rows() == c2.nE && J.cols() == 2 && eJ < 1e-12, "phaseCurrent = |z_tail||z_tip| sin(phase step), one column a frame (" << eJ << ")");
        bool refused = false;
        try { (void)nxr::signal::modesOf(nxr::operators::hodgeDiracModes(h, 40, true, o), h.nV, h.nE, h.nF); } catch (const std::invalid_argument&) { refused = true; }
        EXPECT(refused, "modesOf refuses a both-signs solve (negative mu)");
        std::vector<double> V1; std::vector<int32_t> F1;
        icosphere(1, V1, F1, 1.0);
        const nxr::signal::IntrinsicComplex c1 = nxr::signal::intrinsicComplex(nxr::geometry::surfaceArrays(V1.data(), static_cast<int>(V1.size() / 3), F1.data(), static_cast<int>(F1.size() / 3)));
        refused = false;
        try { (void)nxr::signal::projectOneForm(c1, md, Eigen::MatrixXd::Zero(c1.nE, 1)); } catch (const std::invalid_argument&) { refused = true; }
        EXPECT(refused, "modes of another triangulation are refused");
    }
    {   // an OPEN surface (a flat grid, one boundary loop): every edge's direction at both ends, the boundary edges included, is the mesh edge's,
        // so flat of a constant field is the exact edge difference and sharp recovers it at every vertex, the boundary vertices included
        const int n = 7;
        std::vector<double> P; std::vector<int32_t> T;
        // positions as j * (1 / (n - 1)) (NumPy's linspace), not j / (n - 1): at some rim vertices the input's angle sum then lands one ulp under the
        // intrinsic cone angle, the case where a boundary edge's direction once wrapped round through the exterior
        const double step = 1.0 / (n - 1);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { P.push_back(j * step); P.push_back(i * step); P.push_back(0.0); }
        for (int i = 0; i + 1 < n; ++i) for (int j = 0; j + 1 < n; ++j) { const int a = i * n + j, b = a + 1, cc = a + n, d = cc + 1; T.insert(T.end(), {a, b, d, a, d, cc}); }
        const auto sg = nxr::geometry::surfaceArrays(P.data(), n * n, T.data(), static_cast<int>(T.size() / 3));
        const nxr::signal::IntrinsicComplex g = nxr::signal::intrinsicComplex(sg);
        {   // the grid's boundary edges face right angles: a raw weight of 0 (or -1e-16); the complex holds the pencil's raised weights, so the coexact
            // parts of the modes are finite (a raw 0 gave inf, then NaN, on one platform)
            const auto hg = nxr::operators::hodgedirac::assemble(sg);
            EXPECT(hg.star1Clamped > 0 && (g.star1 - hg.star1).cwiseAbs().maxCoeff() == 0.0 && g.star1.minCoeff() > 0, "open grid: the complex's star1 is the pencil's, raised above 0");
        }
        const Eigen::Vector3d u(0.3, -0.7, 0.0);
        Eigen::MatrixXd J(3 * g.nV, 1);
        for (int v = 0; v < g.nV; ++v) J.middleRows(3 * v, 3) = u;
        const Eigen::MatrixXd w = nxr::signal::flat(g, J);
        double worstFlat = 0;
        for (int e = 0; e < g.nE; ++e) {
            const Eigen::Vector3d ch = g.positions.row(g.edges[2 * e + 1]) - g.positions.row(g.edges[2 * e]);
            worstFlat = std::max(worstFlat, std::abs(w(e, 0) - ch.dot(u)));
        }
        EXPECT(worstFlat < 1e-6, "open grid: flat of a constant field = the edge differences, boundary edges included (worst " << worstFlat << ")");
        const Eigen::MatrixXd back = nxr::signal::sharp(g, w);
        EXPECT((back - J).cwiseAbs().maxCoeff() < 1e-6, "open grid: sharp recovers the constant field at every vertex, boundary included");
    }
    std::cout << (failures ? "FAIL" : "PASS") << " test_intrinsic_forms (" << failures << " failures)\n";
    return failures ? 1 : 0;
}

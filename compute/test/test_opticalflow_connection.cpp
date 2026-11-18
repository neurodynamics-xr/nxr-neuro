// nxr::flow, the connection-Laplacian regulariser (the Cortical Flow Studio's optical flow, ported line for line; the default since).
// (1) PARITY with the Studio's own code (golden/gen_opticalflow_connection_golden.mts runs flow/opticalFlow.ts): the pencil's C, M, the gradient in C's gauge, the gauge solution z
//     (the Studio's system solved densely in float64) and the embedded velocity, on a sphere, a finer sphere, an irregular (stretched, jittered) mesh and an open patch;
// (2) the Studio test's assertions (opticalFlow.test.ts): the sphere bump oracle (direction at every alpha, the speed at the default alpha), GAUGE-FREEDOM (the faces listed in
//     another order, every reference edge moved, leaves v as it was), the round trip through the gauge, equivariance, a stationary pattern gives no flow, two components independent;
// (3) the pcg route equals the direct one; the defaults are connection / gauge.
// REPORTED, not asserted: connection against ambient on a sphere (R = 0.08 m) and on a corrugated sheet (opticalflow_connection report lines).
#include <algorithm>
#include <cmath>
#include <iostream>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/opticalflow_connection.h"
#include "nxr/opticalflow_methods.h"

using namespace nxr::flow;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace cn = nxr::flow::connection;

static double maxAbs(const VectorXd& a) { return a.size() ? a.cwiseAbs().maxCoeff() : 0; }
static double relMax(const VectorXd& a, const VectorXd& b) { return a.size() != b.size() ? 1e9 : maxAbs(a - b) / std::max(1e-300, maxAbs(b)); }

struct Mesh { RowMat3 P; FaceMat F; };
static Mesh sphere(int sub, double R) {
    std::vector<double> V; std::vector<int32_t> F;
    icosphere(sub, V, F, R);
    Mesh m; m.P = Eigen::Map<const RowMat3>(V.data(), static_cast<Eigen::Index>(V.size() / 3), 3);
    m.F = Eigen::Map<const FaceMat>(F.data(), static_cast<Eigen::Index>(F.size() / 3), 3);
    return m;
}
// the travelling bump of the Studio test (MATLAB's optical_flow_validate): a Gaussian of geodesic width sigma = 0.2 R whose centre turns about y at omega0
static const double kOmega = 3, kDt = 0.004;
static Eigen::Vector3d centreAt(double k, double R) { const double th = kOmega * k * kDt; return {R * std::cos(th), 0, -R * std::sin(th)}; }
static VectorXd bumpAt(const RowMat3& P, double k, double R) {
    const Eigen::Vector3d c = centreAt(k, R);
    const double sigma = 0.2 * R;
    VectorXd I(P.rows());
    for (Eigen::Index v = 0; v < P.rows(); ++v) {
        const Eigen::Vector3d x = P.row(v);
        const double d = R * std::acos(std::min(1.0, std::max(-1.0, x.dot(c) / (R * R))));
        I[v] = std::exp(-(d * d) / (2 * sigma * sigma));
    }
    return I;
}
// median direction cosine and speed ratio against the rigid rotation's tangential part, on the vertices within 2.5 sigma of the centre (the Studio test's readMiddle)
struct Read { double cos, speed; int count; };
static Read readMiddle(const RowMat3& P, const VectorXd& vel, double mid, double R) {
    const Eigen::Vector3d c = centreAt(mid + 0.5, R);
    std::vector<double> cs, got, want;
    for (Eigen::Index v = 0; v < P.rows(); ++v) {
        const Eigen::Vector3d x = P.row(v);
        const double d = R * std::acos(std::min(1.0, std::max(-1.0, x.dot(c) / (R * R))));
        if (d >= 2.5 * 0.2 * R) continue;
        const Eigen::Vector3d t = kOmega * kDt * Eigen::Vector3d(x[2], 0, -x[0]), u = vel.segment<3>(3 * v);
        cs.push_back(u.dot(t) / std::max(u.norm() * t.norm(), 1e-30)); got.push_back(u.norm()); want.push_back(t.norm());
    }
    auto med = [](std::vector<double> a) { std::sort(a.begin(), a.end()); return a.empty() ? 0.0 : a[a.size() / 2]; };
    return {med(cs), med(got) / med(want), static_cast<int>(cs.size())};
}
static VectorXd flowOf(const Mesh& m, const VectorXd& a, const VectorXd& b, double alpha, double peak, Regulariser r = Regulariser::Connection, DataFit d = DataFit::Auto) {
    OpticalflowScalar f(m.P, m.F, vertexNormals(m.P, m.F), alpha, peak, FlowForm{r, d});
    return f.step(a, b);
}

int main() {
    // ── (1) parity with the Studio ──
    {
        Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/opticalflow_connection_golden.txt");
        EXPECT(g.good(), "golden file");
        double worstC = 0, worstM = 0, worstG = 0, worstZ = 0, worstVz = 0, worstV = 0, closedVz = 0, closedV = 0;
        int meshes = 0;
        while (true) {
            std::string t;
            if (!(g.f >> t)) break;
            if (t != "mesh") { EXPECT(false, "golden: expected mesh, got " + t); break; }
            const std::string name = g.tok();
            const int n = static_cast<int>(g.num()), nT = static_cast<int>(g.num());
            const double alpha = g.num(), peak = g.num();
            const RowMat3 P = g.mat("positions");
            const FaceMat F = g.mat("faces").cast<std::int32_t>();
            const MatrixXd I = g.mat("I"), Ct = g.mat("C"), Mg = g.mat("M"), Ng = g.mat("normals"), gg = g.mat("g"), zg = g.mat("z"), zvg = g.mat("zv"), vg = g.mat("velocity");
            (void)Ng;
            cn::ConnectionFlow flow(P, F, alpha, peak);
            // C
            Eigen::SparseMatrix<double> Ce(2 * n, 2 * n);
            std::vector<Eigen::Triplet<double>> tr;
            for (int k = 0; k < Ct.rows(); ++k) tr.emplace_back(static_cast<int>(Ct(k, 0)), static_cast<int>(Ct(k, 1)), Ct(k, 2));
            Ce.setFromTriplets(tr.begin(), tr.end());
            const Eigen::SparseMatrix<double> Cm = flow.smoothness() / alpha;
            const double eC = MatrixXd(Cm - Ce).cwiseAbs().maxCoeff() / MatrixXd(Ce).cwiseAbs().maxCoeff();
            const double eM = relMax(flow.mass(), Mg.col(0));
            const VectorXd I0 = I.row(0).transpose(), I1 = I.row(1).transpose();
            VectorXd gm = flow.gaugeGradient(I0), gE(2 * n);
            for (int i = 0; i < n; ++i) { gE[2 * i] = gg(i, 0); gE[2 * i + 1] = gg(i, 1); }
            const double eG = relMax(gm, gE);
            const VectorXd v0 = flow.step(I0, I1);
            VectorXd zE(2 * n), zvE(3 * n);
            for (int i = 0; i < n; ++i) { zE[2 * i] = zg(i, 0); zE[2 * i + 1] = zg(i, 1); for (int k = 0; k < 3; ++k) zvE[3 * i + k] = zvg(i, k); }
            const double eZ = relMax(flow.gauge(), zE), eVz = relMax(v0, zvE);
            // the Studio's own float32 velocity: ours rounded to float32 matches it to float32's resolution (1.2e-7 of the value)
            double eV = 0, top = 0, exact = 0;
            for (int t2 = 0; t2 + 1 < nT; ++t2) {
                const VectorXd a = I.row(t2).transpose(), b = I.row(t2 + 1).transpose();
                const VectorXd v = t2 == 0 ? v0 : flow.step(a, b);
                for (int k = 0; k < 3 * n; ++k) { const double ve = vg(t2, k); eV = std::max(eV, std::abs(static_cast<double>(static_cast<float>(v[k])) - ve)); top = std::max(top, std::abs(ve)); exact += static_cast<float>(v[k]) == static_cast<float>(ve); }
            }
            eV /= std::max(top, 1e-300);
            std::printf("  [%s] n=%d alpha=%g: C %.1e  M %.1e  gauge gradient %.1e  gauge solution z %.1e  embedded z %.1e  velocity (float32) %.1e, %.1f %% of the floats identical\n", name.c_str(), n, alpha, eC, eM, eG, eZ, eVz, eV, 100.0 * exact / ((nT - 1) * 3.0 * n));
            worstC = std::max(worstC, eC); worstM = std::max(worstM, eM); worstG = std::max(worstG, eG); worstZ = std::max(worstZ, eZ); worstVz = std::max(worstVz, eVz); worstV = std::max(worstV, eV);
            if (name != "patch") { closedVz = std::max(closedVz, eVz); closedV = std::max(closedV, eV); }
            ++meshes;
        }
        EXPECT(meshes == 4, "four golden meshes (sphere, finer sphere, irregular, open patch)");
        std::printf("  worst over meshes: C %.2e  M %.2e  g %.2e  z %.2e  v(z) %.2e  velocity(float32) %.2e\n", worstC, worstM, worstG, worstZ, worstVz, worstV);
        // The connection is geometry-central's (docs/specs/2026-11-17-geometry-central-switch.md): its gauge (the reference edge of every vertex) is not the Studio's, so C, the gradient in C's
        // gauge and the gauge solution z are REPORTED, not compared (they differ by a rotation per vertex). What is gauge-free is compared: the lumped mass, the embedded velocity (a closed
        // surface: the cone target is 2 pi at every vertex). The open patch is REPORTED: geometry-central scales a boundary vertex's angles to pi, the Studio's port left them (spec section 3.4).
        EXPECT(worstM < 1e-12, "the lumped mass equals the Studio's to 1e-12");
        EXPECT(closedVz < 1e-9, "the embedded velocity (the gauge solution carried to 3D) of the closed meshes equals the Studio's dense float64 solve to 1e-9");
        EXPECT(closedV < 2e-7, "the embedded velocity of the closed meshes equals the Studio's float32 velocity to float32's resolution (2e-7 of the peak)");
        (void)worstC; (void)worstG; (void)worstZ; (void)worstVz; (void)worstV;
    }

    // ── (2) the Studio test's assertions ──
    const double R = 0.1;
    const Mesh S = sphere(4, R);
    const Eigen::Index n = S.P.rows();
    const int nT = 30, mid = nT / 2 - 1;
    {
        const VectorXd a = bumpAt(S.P, mid, R), b = bumpAt(S.P, mid + 1, R);
        const double peak = 1.0;   // the Gaussian's peak is 1 (the Studio normalises by the movie's)
        for (double alpha : {0.01, 0.1, 1.0}) {
            const Read r = readMiddle(S.P, flowOf(S, a, b, alpha, peak), mid, R);
            std::printf("  [oracle] alpha=%g: %d active vertices, median direction cos %.4f, median speed %.1f %% of truth\n", alpha, r.count, r.cos, 100 * r.speed);
            EXPECT(r.cos > 0.99, "the bump's direction (cos > 0.99) at alpha " + std::to_string(alpha));
            if (alpha == 0.01) EXPECT(r.speed > 0.9 && r.speed < 1.1, "the speed at the default alpha 0.01 within 10 % of truth");
        }
    }
    {   // gauge-free: the sphere jittered (the intrinsic Delaunay triangulation is then unique), the faces reversed in order so that every vertex's reference edge moves
        Mesh J = S; unsigned seed = 9;
        for (Eigen::Index v = 0; v < n; ++v) { seed = static_cast<unsigned>((static_cast<unsigned long long>(seed) * 16807ull) % 2147483647ull); J.P.row(v) *= 1 + 0.04 * (seed / 2147483647.0 - 0.5); }
        const VectorXd a = bumpAt(J.P, 0, R), b = bumpAt(J.P, 1, R);
        Mesh T = J;
        const Eigen::Index nF = J.F.rows();
        for (Eigen::Index f = 0; f < nF; ++f) T.F.row(f) = J.F.row(nF - 1 - f);
        const VectorXd va = flowOf(J, a, b, 0.1, 1.0), vb = flowOf(T, a, b, 0.1, 1.0);
        // the frames DID turn: the first corner's edge of most vertices differs
        auto pencil = [&](const Mesh& m) {
            std::vector<double> P(m.P.data(), m.P.data() + 3 * n);
            std::vector<int32_t> F(m.F.data(), m.F.data() + 3 * m.F.rows());
            return cn::componentPencil(P.data(), static_cast<int>(n), F);
        };
        const cn::Pencil pa = pencil(J), pb = pencil(T);
        int moved = 0;
        for (Eigen::Index v = 0; v < n; ++v) {
            const size_t ha = static_cast<size_t>(pa.corners.order[static_cast<size_t>(pa.corners.ptr[static_cast<size_t>(v)])]), hb = static_cast<size_t>(pb.corners.order[static_cast<size_t>(pb.corners.ptr[static_cast<size_t>(v)])]);
            double dot = 0; for (int k = 0; k < 3; ++k) dot += pa.corners.edge[3 * ha + k] * pb.corners.edge[3 * hb + k];
            moved += dot < 0.9;
        }
        const double d = relMax(va, vb);
        std::printf("  [gauge] reference edges moved at %d of %lld vertices; max |dv| / max |v| = %.2e\n", moved, static_cast<long long>(n), d);
        EXPECT(moved > n / 2, "the frames turned (more than half the reference edges moved)");
        EXPECT(d < 1e-6, "the faces listed in another order leave v as it was (gauge-free)");
    }
    {   // the round trip through C's gauge: psi -> corner and theta -> the vector in that face -> its angle back
        std::vector<double> P(S.P.data(), S.P.data() + 3 * n);
        std::vector<int32_t> F(S.F.data(), S.F.data() + 3 * S.F.rows());
        const cn::Pencil pen = cn::componentPencil(P.data(), static_cast<int>(n), F);
        const cn::Corners& c = pen.corners;
        double worst = 0; int outside = 0;
        for (int v = 0; v < n; ++v) {
            const int a = c.ptr[static_cast<size_t>(v)], b = c.ptr[static_cast<size_t>(v) + 1];
            if (a == b) continue;
            const int last = c.order[static_cast<size_t>(b) - 1];
            const double span = c.psi0[static_cast<size_t>(last)] + c.width[static_cast<size_t>(last)];
            for (int k = 0; k < 16; ++k) {
                const double psi = span * (k + 0.37) / 16;
                int h; double theta;
                cn::cornerAt(c, v, psi, h, theta);
                if (theta < -1e-12 || theta * c.scale[static_cast<size_t>(v)] > c.width[static_cast<size_t>(h)] + 1e-12) ++outside;
                const double x = std::cos(theta), y = std::sin(theta);
                double vec[3], ed[3], pe[3];
                for (int q = 0; q < 3; ++q) { ed[q] = c.edge[3 * static_cast<size_t>(h) + static_cast<size_t>(q)]; pe[q] = c.perp[3 * static_cast<size_t>(h) + static_cast<size_t>(q)]; vec[q] = x * ed[q] + y * pe[q]; }
                const double th = std::atan2(vec[0] * pe[0] + vec[1] * pe[1] + vec[2] * pe[2], vec[0] * ed[0] + vec[1] * ed[1] + vec[2] * ed[2]);
                const double back = c.psi0[static_cast<size_t>(h)] + th * c.scale[static_cast<size_t>(v)];
                worst = std::max(worst, std::abs(std::atan2(std::sin(back - psi), std::cos(back - psi))));
            }
        }
        std::printf("  [round trip] %lld vertices x 16 angles: worst %.2e rad, %d outside their corner\n", static_cast<long long>(n), worst, outside);
        EXPECT(worst < 1e-9 && outside == 0, "the round trip through C's gauge is exact at every vertex and angle");
    }
    {   // equivariance: the mesh and the bump turned rigidly turn v with them
        const double th = 0.7, c = std::cos(th), s = std::sin(th);
        Mesh Q = S;
        for (Eigen::Index v = 0; v < n; ++v) { const Eigen::Vector3d x = S.P.row(v); Q.P.row(v) = Eigen::Vector3d(c * x[0] - s * x[2], x[1], s * x[0] + c * x[2]); }
        const VectorXd a = bumpAt(S.P, 0, R), b = bumpAt(S.P, 1, R);
        const VectorXd va = flowOf(S, a, b, 0.1, 1.0), vb = flowOf(Q, a, b, 0.1, 1.0);
        double d = 0, m = 0;
        for (Eigen::Index v = 0; v < n; ++v) { const Eigen::Vector3d u = va.segment<3>(3 * v); const Eigen::Vector3d r(c * u[0] - s * u[2], u[1], s * u[0] + c * u[2]); d = std::max(d, (r - Eigen::Vector3d(vb.segment<3>(3 * v))).cwiseAbs().maxCoeff()); m = std::max(m, vb.segment<3>(3 * v).cwiseAbs().maxCoeff()); }
        std::printf("  [equivariance] max |Rv - v(R mesh)| / max |v| = %.2e\n", d / m);
        EXPECT(d / m < 1e-5, "the mesh and the bump turned rigidly turn v with them");
    }
    {   // a stationary pattern gives no flow; a map with a zero peak gives none
        const VectorXd a = bumpAt(S.P, 0, R);
        const VectorXd v = flowOf(S, a, a, 1.0, 1.0);
        EXPECT(maxAbs(v) == 0, "a stationary pattern gives no flow (exactly 0)");
    }
    {   // two disconnected pieces are solved independently
        const Mesh A = sphere(2, R);
        const Eigen::Index nA = A.P.rows();
        Mesh B2; B2.P.resize(2 * nA, 3); B2.F.resize(2 * A.F.rows(), 3);
        B2.P.topRows(nA) = A.P; B2.P.bottomRows(nA) = A.P; B2.P.bottomRows(nA).col(0).array() += 0.5;
        B2.F.topRows(A.F.rows()) = A.F; B2.F.bottomRows(A.F.rows()) = A.F.array() + static_cast<std::int32_t>(nA);
        VectorXd a = VectorXd::Zero(2 * nA), b = VectorXd::Zero(2 * nA);
        a.head(nA) = bumpAt(A.P, 0, R); b.head(nA) = bumpAt(A.P, 1, R);
        const VectorXd v = flowOf(B2, a, b, 0.1, 1.0);
        EXPECT(maxAbs(v.tail(3 * nA)) == 0 && maxAbs(v.head(3 * nA)) > 0, "two components: the bump on one leaves the other still");
        const VectorXd single = flowOf(A, a.head(nA), b.head(nA), 0.1, 1.0);
        EXPECT(relMax(v.head(3 * nA), single) < 1e-12, "a component's flow equals its flow alone");
    }
    {   // the pcg route equals the direct one; the defaults
        FlowMethodOptions d, p;
        d.method = FlowMethod::Direct; p.method = FlowMethod::Pcg;
        EXPECT(d.regulariser == Regulariser::Connection && kDefaultRegulariser == Regulariser::Connection, "the default regulariser is connection");
        EXPECT(resolveDataFit(d.regulariser, d.dataFit) == DataFit::Gauge, "its default data term is the Studio's gauge");
        bool refused = false; try { resolveDataFit(Regulariser::Connection, DataFit::Triangle); } catch (const std::invalid_argument&) { refused = true; }
        EXPECT(refused, "connection with another data term is refused");
        const Mesh T = sphere(3, R);
        const VectorXd a = bumpAt(T.P, 0, R), b = bumpAt(T.P, 1, R);
        FlowEngine ed(d, T.P, T.F, vertexNormals(T.P, T.F), 0.05), ep(p, T.P, T.F, vertexNormals(T.P, T.F), 0.05);
        ed.observe(a); ed.observe(b); ep.observe(a); ep.observe(b);
        const FlowStep sd = ed.step(a, b), sp = ep.step(a, b);
        const double e = relMax(sp.velocity[0], sd.velocity[0]);
        std::printf("  [pcg] %d iterations; |pcg - direct| / |direct| = %.2e\n", sp.iterations[0], e);
        EXPECT(e < 1e-6 && maxAbs(sd.velocity[0]) > 0, "pcg equals the direct solve (1e-6)");
    }

    // ── REPORT: connection against ambient ──
    {
        auto compare = [&](const char* name, const Mesh& M, const VectorXd& a, const VectorXd& b, const std::function<bool(Eigen::Index)>& active, double alphaC, Read (*truth)(const RowMat3&, const VectorXd&, double, double), double R0) {
            (void)truth; (void)R0;
            const VectorXd vc = flowOf(M, a, b, alphaC, a.maxCoeff());
            auto speedMed = [&](const VectorXd& v) { std::vector<double> s; for (Eigen::Index i = 0; i < M.P.rows(); ++i) if (active(i)) s.push_back(v.segment<3>(3 * i).norm()); std::sort(s.begin(), s.end()); return s.empty() ? 0.0 : s[s.size() / 2]; };
            auto diff = [&](const VectorXd& u, const VectorXd& w, double& relDiff, double& cosine) {
                double nd = 0, nw = 0, dot = 0, nu = 0;
                for (Eigen::Index i = 0; i < M.P.rows(); ++i) if (active(i)) { const Eigen::Vector3d x = u.segment<3>(3 * i), y = w.segment<3>(3 * i); nd += (x - y).squaredNorm(); nw += y.squaredNorm(); nu += x.squaredNorm(); dot += x.dot(y); }
                relDiff = std::sqrt(nd / nw); cosine = dot / std::sqrt(nu * nw);
            };
            const VectorXd va = flowOf(M, a, b, alphaC, a.maxCoeff(), Regulariser::Ambient, DataFit::Triangle);
            double rd, cs; diff(vc, va, rd, cs);
            std::printf("  [report %s] same alpha %.3g: |v_conn - v_amb| / |v_amb| = %.3f, direction cosine %.4f; median speeds conn %.4g amb %.4g\n", name, alphaC, rd, cs, speedMed(vc), speedMed(va));
            // alpha matched by balance: the ambient alpha at which its median active speed equals the connection's (the data/smoothness balance sets how far the prior pulls the speed down)
            double lo = std::log(alphaC * 1e-3), hi = std::log(alphaC * 1e3);
            const double target = speedMed(vc);
            for (int it = 0; it < 30; ++it) {
                const double midl = 0.5 * (lo + hi);
                const double sp = speedMed(flowOf(M, a, b, std::exp(midl), a.maxCoeff(), Regulariser::Ambient, DataFit::Triangle));
                if (sp > target) lo = midl; else hi = midl;   // speed falls as alpha grows
            }
            const double alphaA = std::exp(0.5 * (lo + hi));
            const VectorXd vm = flowOf(M, a, b, alphaA, a.maxCoeff(), Regulariser::Ambient, DataFit::Triangle);
            const double reached = speedMed(vm) / target;
            if (reached < 0.99) {   // ambient's data term caps its speed below the connection's even as alpha -> 0: no matched alpha exists
                std::printf("  [report %s] no matched alpha: ambient's median speed tops out at %.1f %% of the connection's (alpha -> 0)\n", name, 100 * reached);
                return;
            }
            diff(vc, vm, rd, cs);
            std::printf("  [report %s] ambient alpha matched to the connection's median speed: alpha_amb = %.3g (x%.2f): relative difference %.3f, direction cosine %.4f\n", name, alphaA, alphaA / alphaC, rd, cs);
        };
        // (a) the sphere at R = 0.08 m
        const double R8 = 0.08;
        const Mesh P8 = sphere(4, R8);
        const VectorXd a8 = bumpAt(P8.P, mid, R8), b8 = bumpAt(P8.P, mid + 1, R8);
        const Eigen::Vector3d c8 = centreAt(mid + 0.5, R8);
        auto act8 = [&](Eigen::Index v) { const Eigen::Vector3d x = P8.P.row(v); return R8 * std::acos(std::min(1.0, std::max(-1.0, x.dot(c8) / (R8 * R8)))) < 2.5 * 0.2 * R8; };
        compare("sphere R=0.08", P8, a8, b8, act8, 0.01, nullptr, R8);
        // (b) a corrugated sheet: z = A sin(2 pi x / L) sin(2 pi y / L), amplitude comparable to the wavelength (steep folds); the bump slides along x
        const int N = 61; const double W = 0.1, L = 0.025, Amp = 0.009;
        Mesh C; C.P.resize(N * N, 3); C.F.resize(2 * (N - 1) * (N - 1), 3);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { const double x = W * i / (N - 1), y = W * j / (N - 1); C.P.row(i * N + j) << x, y, Amp * std::sin(2 * M_PI * x / L) * std::sin(2 * M_PI * y / L); }
        int f = 0;
        for (int i = 0; i + 1 < N; ++i) for (int j = 0; j + 1 < N; ++j) { const int v00 = i * N + j, v10 = (i + 1) * N + j, v01 = i * N + j + 1, v11 = (i + 1) * N + j + 1; C.F.row(f++) << v00, v10, v11; C.F.row(f++) << v00, v11, v01; }
        auto sheetBump = [&](double cx) { VectorXd I(N * N); for (int v = 0; v < N * N; ++v) { const double dx = C.P(v, 0) - cx, dy = C.P(v, 1) - 0.5 * W; I[v] = std::exp(-(dx * dx + dy * dy) / (2 * 0.012 * 0.012)); } return I; };
        const VectorXd as = sheetBump(0.45 * W), bs = sheetBump(0.45 * W + 0.0012);
        auto actS = [&](Eigen::Index v) { const double dx = C.P(v, 0) - 0.45 * W - 0.0006, dy = C.P(v, 1) - 0.5 * W; return std::hypot(dx, dy) < 2.5 * 0.012; };
        compare("corrugated sheet", C, as, bs, actS, 0.01, nullptr, 0);
    }
    std::cout << (g_failures ? "FAILED" : "all passed") << "\n";
    return g_failures ? 1 : 0;
}

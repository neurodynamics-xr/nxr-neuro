// nxr::quadrature: (1) parity with nxr.spectral.lagged on small synthetic inputs (golden/quadrature_golden.txt, generator beside it); (2) zero-delay mixing adds no quadrature;
// (3) waves on an icosphere: impropriety, the phase-gradient direction and alignment, the standing surrogate; (4) a two-source quarter-cycle lag: the pair and who leads, the rotary
// spectrum, the time lag; (5) refusals.
#include "nxr/core/algebra.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/geometry/mesh.h"
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/stats/circular.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/signal/spectral.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "golden_io.h"
#include "mesh_helpers.h"
#include <random>
using namespace nxr::core;
using namespace nxr::signal;
using namespace nxr::stats;
using namespace nxr::measurement;
using namespace nxr::geometry;
using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Eigen::VectorXcd;
using cd = std::complex<double>;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
static MatrixXd form(const nxr::signal::QuadraturePair& p) { return p.p * p.q.transpose() - p.q * p.p.transpose(); }
static VectorXcd col(const MatrixXcd& M, int j) { return M.col(j); }

static void parity() {
    std::cout << "=== parity with nxr.spectral.lagged (golden/quadrature_golden.txt) ===\n";
    Golden g(kDir + "/quadrature_golden.txt");
    EXPECT(g.good(), "quadrature golden opens");
    const MatrixXcd F = g.cmat("F_re", "F_im");
    const MatrixXcd C = F * F.adjoint();
    auto close = [&](const char* name, double got, double want, double tol) {
        const double e = std::abs(got - want) / std::max(1.0, std::abs(want));
        EXPECT(e < tol, std::string(name) + ": " + std::to_string(got) + " vs nxr " + std::to_string(want) + " (" + sci(e) + ")");
    };
    close("quadratureFraction (lagged_fraction)", nxr::signal::quadratureFraction(C), g.mat("lagged_fraction")(0, 0), 1e-12);
    close("quadratureFractionFromFactor (im_fraction)", nxr::signal::quadratureFractionFromFactor(F), g.mat("im_fraction")(0, 0), 1e-12);
    close("quadratureEnergy summed over rows (im_rows)", quadratureEnergy(F.topRows(5), F).sum(), g.mat("im_rows")(0, 0), 1e-11);
    const MatrixXd A = g.mat("A_normalised");
    const double nIm = g.mat("norm_imC")(0, 0);
    const QuadratureFactor f = quadratureFactor(F);
    EXPECT(relErr(f.Q * f.S * f.Q.transpose(), MatrixXd(C.imag())) < 1e-12, "Im C = Q S Q^T, exactly (" + sci(relErr(f.Q * f.S * f.Q.transpose(), MatrixXd(C.imag()))) + ")");
    EXPECT(relErr(f.Q * f.S * f.Q.transpose() / f.S.norm(), A) < 1e-12, "quadratureFactor normalised = nxr lagged(F): " + sci(relErr(f.Q * f.S * f.Q.transpose() / f.S.norm(), A)));
    EXPECT(std::abs(f.S.norm() - nIm) < 1e-10, "||S||_F = ||Im C||_F");
    EXPECT((f.Q.transpose() * f.Q - MatrixXd::Identity(f.Q.cols(), f.Q.cols())).cwiseAbs().maxCoeff() < 1e-12, "Q orthonormal");
    const auto P = nxr::signal::planes(f, 3);
    EXPECT(P.size() == 3, "three pairs");
    const MatrixXd sg = g.mat("sigma");
    for (int i = 0; i < 3; ++i) {
        EXPECT(std::abs(P[i].sigma / nIm - sg(0, i)) < 1e-10, "sigma " + std::to_string(i) + " / ||Im C|| = nxr's: " + sci(std::abs(P[i].sigma / nIm - sg(0, i))));
        const MatrixXd want = g.mat("form" + std::to_string(i));
        EXPECT(relErr(form(P[i]), want) < 1e-9, "pair " + std::to_string(i) + ": p q^T - q p^T = nxr's x y^T - y x^T (so p = x leads q = y): " + sci(relErr(form(P[i]), want)));
        EXPECT(std::abs(P[i].p.norm() - 1) < 1e-12 && std::abs(P[i].q.norm() - 1) < 1e-12 && std::abs(P[i].p.dot(P[i].q)) < 1e-12, "pair " + std::to_string(i) + " orthonormal");
    }
    // the gauge: the strongest row has q = 0, p > 0
    {
        Eigen::Index i; (P[0].p.array().square() + P[0].q.array().square()).maxCoeff(&i);
        EXPECT(std::abs(P[0].q[i]) < 1e-12 && P[0].p[i] > 0, "gauge: q = 0 and p > 0 at the pair's strongest row");
    }
    close("lagged_share (explained)", lagged_share(f, P), g.mat("explained")(0, 0), 1e-10);
    double s2 = 0; for (const auto& p : P) s2 += 2 * p.sigma * p.sigma;
    close("share = sum 2 sigma^2 / ||Im C||^2", lagged_share(f, P), s2 / (nIm * nIm), 1e-10);
    close("all pairs explain everything", lagged_share(f, nxr::signal::planes(f, 100)), 1.0, 1e-10);
    EXPECT(relErr(form(nxr::signal::planes(F, 1)[0]), g.mat("top_form")) < 1e-9, "planes(F, 1) = nxr top_plane(F)");
    const nxr::signal::RotaryPower rp = nxr::signal::rotaryPower(F, P[0].p, P[0].q);
    EXPECT(rp.sense == 1 && std::abs((rp.positive - rp.negative) - 4 * P[0].sigma) < 1e-9 * (1 + rp.positive), "rotary power: P+ - P- = 4 sigma, sense +1 (p leads q)");
    // impropriety = 1 - travel index
    {
        const VectorXcd u = VectorXcd(g.cmat("u_re", "u_im").row(0).transpose());
        const VectorXcd us = VectorXcd(g.cmat("u_standing_re", "u_standing_im").row(0).transpose());
        close("impropriety = 1 - nxr travel_index", nxr::signal::impropriety(u), 1 - g.mat("travel_index")(0, 0), 1e-12);
        close("impropriety of a standing pattern = 1 - travel_index = 1", nxr::signal::impropriety(us), 1 - g.mat("travel_index_standing")(0, 0), 1e-12);
    }
    // rotary spectrum
    const double rfs = g.mat("rotary_fs")(0, 0);
    const MatrixXd fr = g.mat("rotary_freqs"), zr = g.mat("rotary_z_re"), zi = g.mat("rotary_z_im");
    VectorXcd z(zr.cols());
    for (int i = 0; i < zr.cols(); ++i) z[i] = cd(zr(0, i), zi(0, i));
    std::vector<double> fq; for (int i = 0; i < fr.cols(); ++i) fq.push_back(fr(0, i));
    const nxr::signal::RotarySpectrum rs = nxr::signal::rotarySpectrum(z, rfs, fq, 7.0);
    const MatrixXd wantPos = g.mat("rotary_pos"), wantNeg = g.mat("rotary_neg");
    EXPECT(relErr(rs.positive, wantPos) < 1e-9 && relErr(rs.negative, wantNeg) < 1e-9, "rotarySpectrum = nxr rotary_power: " + sci(relErr(rs.positive, wantPos)) + ", " + sci(relErr(rs.negative, wantNeg)));
    // phase gradient on the icosphere
    const MatrixXd Vg = g.mat("positions"), Fg = g.mat("faces"), win = g.mat("window"), Zr = g.mat("Z_re"), Zi = g.mat("Z_im");
    nxr::operators::RowMat3 pos = Vg;
    nxr::operators::FaceMat fc(Fg.rows(), 3);
    for (int i = 0; i < Fg.rows(); ++i) for (int k = 0; k < 3; ++k) fc(i, k) = static_cast<std::int32_t>(Fg(i, k));
    nxr::operators::FaceOperators ops(pos, fc);
    std::vector<int> w; for (int i = 0; i < win.cols(); ++i) w.push_back(static_cast<int>(win(0, i)));
    VectorXcd Z(Zr.cols());
    for (int i = 0; i < Zr.cols(); ++i) Z[i] = cd(Zr(0, i), Zi(0, i));
    const VectorXd area = faceAreas(pos, fc);
    // (the golden's phase_gradient_flux angle and strength summed raw 3-D vectors over a window with no transport: wrong on a curved sheet; no longer a parity target)
    g.mat("flux_angle"); g.mat("flux_strength");   // read past: the old raw-sum targets
    const MatrixXd Sr = g.mat("Zs_re"), Si = g.mat("Zs_im");
    const VectorXcd sur = nxr::signal::standingSurrogate(Z);
    double e = 0; for (int i = 0; i < Sr.cols(); ++i) e = std::max(e, std::abs(sur[i] - cd(Sr(0, i), Si(0, i))));
    EXPECT(e < 1e-12, "standingSurrogate = nxr standing_surrogate: " + sci(e));
    g.mat("flux_strength_surrogate");
    const MatrixXd an = g.mat("angles");
    VectorXd av(an.cols()); for (int i = 0; i < an.cols(); ++i) av[i] = an(0, i);
    const nxr::stats::Rayleigh ry = nxr::stats::circ_rtest(av);
    close("circ_r (R)", nxr::stats::circ_r(av), g.mat("dir_R")(0, 0), 1e-12);
    close("circ_rtest p", ry.p, g.mat("dir_p")(0, 0), 1e-12);
    close("mean direction", ry.meanDirection, g.mat("dir_mean")(0, 0), 1e-12);
}

static void zeroDelay() {
    std::cout << "=== zero-delay mixing adds no quadrature ===\n";
    std::mt19937 rng(3);
    std::normal_distribution<double> nd;
    const int M = 8, r = 3, N = 30;
    MatrixXd G(M, r), K(N, M);
    for (auto& v : G.reshaped()) v = nd(rng);
    for (auto& v : K.reshaped()) v = nd(rng);
    MatrixXcd F0(M, r);   // sensors all in phase: each independent source one phase for every channel
    for (int k = 0; k < r; ++k) F0.col(k) = G.col(k).cast<cd>() * std::polar(1.0, 0.3 + k);
    const MatrixXcd F = K.cast<cd>() * F0;
    EXPECT(nxr::signal::quadratureFractionFromFactor(F0) < 1e-7, "in-phase sensors: quadratureFraction 0 (" + sci(nxr::signal::quadratureFractionFromFactor(F0)) + ")");
    EXPECT(nxr::signal::quadratureFractionFromFactor(F) < 1e-7, "after a real mixing (an inverse): still 0 (" + sci(nxr::signal::quadratureFractionFromFactor(F)) + ")");
    EXPECT(quadratureEnergy(F, F).maxCoeff() < 1e-12 * (1 + (F * F.adjoint()).real().cwiseAbs().maxCoeff()), "quadratureEnergy 0 on every row");
    EXPECT(nxr::signal::planes(F, 3).empty(), "no quadrature pair");
    // a lagged pair of sensors: mixing keeps Im C = K (Im C0) K^T
    MatrixXcd L0(M, 2);   // two independent patterns, each a real pattern plus i times another: a lag within each
    L0.col(0) = G.col(0).cast<cd>() + cd(0, 1) * G.col(1).cast<cd>();
    L0.col(1) = G.col(2).cast<cd>() + cd(0, 0.5) * G.col(0).cast<cd>();
    const MatrixXcd C0 = L0 * L0.adjoint(), C = K.cast<cd>() * C0 * K.transpose().cast<cd>();
    EXPECT(relErr(nxr::signal::quadspectrum(C), MatrixXd(K * nxr::signal::quadspectrum(C0) * K.transpose())) < 1e-12, "Im(K C K^T) = K (Im C) K^T");
    EXPECT(relErr(nxr::signal::cospectrum(C), MatrixXd(K * nxr::signal::cospectrum(C0) * K.transpose())) < 1e-12, "Re(K C K^T) = K (Re C) K^T");
    EXPECT(std::abs(nxr::signal::quadratureFractionFromFactor(K.cast<cd>() * L0) - nxr::signal::quadratureFraction(C)) < 1e-10, "quadratureFraction from the factor = from C");
    // per-row energy against the N x N matrix
    const MatrixXd Im = nxr::signal::quadspectrum(C);
    const VectorXd e = quadratureEnergy(K.cast<cd>() * L0, K.cast<cd>() * L0);
    EXPECT((e - Im.array().square().matrix().rowwise().sum()).cwiseAbs().maxCoeff() < 1e-9 * Im.cwiseAbs().maxCoeff() * Im.cwiseAbs().maxCoeff(), "quadratureEnergy_i = sum_j (Im C_ij)^2");
}

static void waves() {
    std::cout << "=== waves on an icosphere ===\n";
    std::vector<double> V; std::vector<int32_t> Fv;
    icosphere(3, V, Fv);
    const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(Fv.size() / 3);
    nxr::operators::RowMat3 pos(nV, 3); nxr::operators::FaceMat fc(nF, 3);
    for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) pos(i, k) = V[3 * i + k];
    for (int i = 0; i < nF; ++i) for (int k = 0; k < 3; ++k) fc(i, k) = Fv[3 * i + k];
    nxr::operators::FaceOperators ops(pos, fc);
    const VectorXd area = faceAreas(pos, fc);
    EXPECT(std::abs(area.sum() - 4 * kPi) < 0.1, "the faces' areas sum to about 4 pi (" + std::to_string(area.sum()) + ")");
    // a log-map disc round the north pole (25 degrees)
    nxr::geometry::ChartKit kit(pos.data(), nV, Fv.data(), nF);
    int pole = 0;
    for (int i = 1; i < nV; ++i) if (pos(i, 2) > pos(pole, 2)) pole = i;
    const double rad = 25 * kPi / 180;
    const Eigen::Vector3d east(1, 0, 0);
    // the anatomical frame (SCS axes: anterior +x) on the signpost basis, to read the direction from anterior
    const nxr::geometry::Basis cb = kit.basis();
    nxr::geometry::Mat3 fe1(nV, 3), fe2(nV, 3), fn(nV, 3);
    for (int i = 0; i < nV; ++i) for (int d = 0; d < 3; ++d) { fe1(i, d) = cb.e1[3 * i + d]; fe2(i, d) = cb.e2[3 * i + d]; fn(i, d) = cb.n[3 * i + d]; }
    nxr::geometry::AnatomicalOptions ao; ao.side = nxr::geometry::Side::Left;
    const nxr::geometry::Anatomical anat = nxr::geometry::anatomicalFrame(V.data(), nV, Fv.data(), nF, fn, fe1, fe2, ao);
    const nxr::signal::AnatomicalFrame af{&anat, &fe1, &fe2};
    EXPECT(nxr::geometry::disc(kit, pole, rad).size() > 10, "the disc holds vertices");
    for (double deg : {0.0, 60.0, -120.0}) {
        const double a = deg * kPi / 180;
        const Eigen::Vector3d k = 3.0 * Eigen::Vector3d(std::cos(a), std::sin(a), 0);
        VectorXcd z(nV), st(nV);
        for (int i = 0; i < nV; ++i) {
            const double ph = k.dot(Eigen::Vector3d(pos.row(i)));
            z[i] = std::polar(1.0, -ph);
            st[i] = std::polar(1.0, 0.4) * std::cos(ph);
        }
        const nxr::signal::PhaseGradient pg = nxr::signal::weightedPhaseGradient(z, ops, fc, area, kit, pole, rad);
        const double tv[3] = {std::cos(a), std::sin(a), 0};
        double d = std::remainder(pg.direction - kit.gaugeHeading(pole, tv), 2 * kPi);
        const nxr::signal::PhaseGradient pa = weightedPhaseGradient(z, ops, fc, area, kit, pole, rad, &af);
        EXPECT(std::abs(std::remainder(pa.direction - pg.direction, 2 * kPi)) < 1e-12, "the gauge angle is unchanged by asking for the anatomical one");
        EXPECT(std::abs(std::remainder(pa.directionFromAnterior - std::atan2(std::sin(a), std::cos(a)), 2 * kPi)) < 5 * kPi / 180 || std::abs(std::remainder(pa.directionFromAnterior + a, 2 * kPi)) < 5 * kPi / 180, "direction from anterior is the wave's angle from +x (up to the second axis' sign)");
        std::printf("  wave toward %6.1f deg: direction %.2f deg, alignment %.4f\n", deg, pg.direction * 180 / kPi, pg.phaseGradientAlignment);
        EXPECT(std::abs(d) < 5 * kPi / 180, "weightedPhaseGradient direction = the wave's (" + std::to_string(deg) + " deg)");
        EXPECT(pg.phaseGradientAlignment > 0.95, "phaseGradientAlignment near 1 for a plane wave");
        const nxr::signal::PhaseGradient ss = nxr::signal::weightedPhaseGradient(st, ops, fc, area, kit, pole, rad);
        EXPECT(ss.phaseGradientAlignment < 1e-9, "a standing pattern: alignment 0 (" + sci(ss.phaseGradientAlignment) + ")");
        const nxr::signal::PhaseGradient sg = nxr::signal::weightedPhaseGradient(nxr::signal::standingSurrogate(z), ops, fc, area, kit, pole, rad);
        EXPECT(sg.phaseGradientAlignment < 1e-9, "the standing surrogate of the wave: alignment 0 (" + sci(sg.phaseGradientAlignment) + ")");
        // impropriety over the whole sphere with a short wavelength
        VectorXcd zw(nV), sw(nV);
        for (int i = 0; i < nV; ++i) { const double ph = 8.0 * pos(i, 0); zw[i] = std::polar(1.0, -ph); sw[i] = std::polar(1.0, 0.4) * std::cos(ph); }
        if (deg == 0.0) {
            EXPECT(nxr::signal::impropriety(zw) < 0.1, "travelling wave: impropriety near 0 (" + std::to_string(nxr::signal::impropriety(zw)) + ")");
            EXPECT(std::abs(impropriety(sw) - 1) < 1e-12, "standing pattern: impropriety 1 (" + std::to_string(impropriety(sw)) + ")");
            EXPECT(std::abs(impropriety(standingSurrogate(zw)) - 1) < 1e-12, "standing surrogate: impropriety 1");
            VectorXcd rot(2); rot << cd(1, 0), cd(0, 1);       // a rotating pattern: circular
            EXPECT(impropriety(rot) < 1e-14, "(1, i): circular, impropriety 0");
        }
    }
    EXPECT(throws([&] { weightedPhaseGradient(VectorXcd::Zero(nV - 1), ops, fc, area, kit, pole, rad); }), "z of the wrong size refused");
    EXPECT(throws([&] { weightedPhaseGradient(VectorXcd::Zero(nV), ops, fc, area, kit, nV, rad); }), "a window face outside the mesh refused");
}

// A plane wave ACROSS a corrugated sheet (z = A sin(kc x); phase linear in arc length along x; the sheet is developable, so the transported direction is the unrolled one):
// raw 3-D sums lose the steep faces' share, the transported sum keeps T near 1 and the direction.
static void corrugated() {
    std::cout << "=== a plane wave across a corrugated sheet ===\n";
    const int nx = 61, ny = 41;
    const double Lx = 6.0, Ly = 4.0, A = 0.5, kc = 2 * kPi / 1.5, k = 3.0;
    const int nV = nx * ny, nF = 2 * (nx - 1) * (ny - 1);
    nxr::operators::RowMat3 pos(nV, 3); nxr::operators::FaceMat fc(nF, 3);
    std::vector<double> V(3 * nV); std::vector<int32_t> Fv(3 * nF);
    VectorXd s(nx);                                  // arc length along x
    s[0] = 0;
    auto zOf = [&](double x) { return A * std::sin(kc * x); };
    for (int i = 1; i < nx; ++i) { const double x0 = Lx * (i - 1) / (nx - 1), x1 = Lx * i / (nx - 1); s[i] = s[i - 1] + std::hypot(x1 - x0, zOf(x1) - zOf(x0)); }
    VectorXcd z(nV);
    for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const int v = j * nx + i, x = Lx * i / (nx - 1);
        pos(v, 0) = Lx * i / (nx - 1); pos(v, 1) = Ly * j / (ny - 1); pos(v, 2) = zOf(pos(v, 0));
        for (int d = 0; d < 3; ++d) V[3 * v + d] = pos(v, d);
        z[v] = std::polar(1.0, -k * s[i]); (void)x;
    }
    int f = 0;
    for (int j = 0; j + 1 < ny; ++j) for (int i = 0; i + 1 < nx; ++i) {
        const int a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
        const int t[2][3] = {{a, b, d}, {a, d, c}};
        for (auto& tri : t) { for (int m = 0; m < 3; ++m) { fc(f, m) = tri[m]; Fv[3 * f + m] = tri[m]; } ++f; }
    }
    nxr::operators::FaceOperators ops(pos, fc);
    const VectorXd area = faceAreas(pos, fc);
    nxr::geometry::ChartKit kit(pos.data(), nV, Fv.data(), nF);
    const int anchor = (ny / 2) * nx + nx / 2;
    const double x0 = pos(anchor, 0), slope = A * kc * std::cos(kc * x0);
    const Eigen::Vector3d along = Eigen::Vector3d(1, 0, slope).normalized();
    const double rad = 1.6;                           // more than one corrugation period each side
    const PhaseGradient pg = weightedPhaseGradient(z, ops, fc, area, kit, anchor, rad);
    // the raw sum, for the record: |sum P| / sum |P| over the same faces (the old reading's strength)
    double rawS[3] = {0, 0, 0}, rawD = 0;
    {
        VectorXd gr[3], gi[3];
        for (int d = 0; d < 3; ++d) { gr[d] = ops.grad[d] * VectorXd(z.real()); gi[d] = ops.grad[d] * VectorXd(z.imag()); }
        const std::vector<int> disc = nxr::geometry::disc(kit, anchor, rad);
        std::vector<char> in(nV, 0); for (int v : disc) in[v] = 1;
        for (int q = 0; q < nF; ++q) {
            if (!(in[fc(q, 0)] && in[fc(q, 1)] && in[fc(q, 2)])) continue;
            const cd zf = (z[fc(q, 0)] + z[fc(q, 1)] + z[fc(q, 2)]) / 3.0;
            double n2 = 0;
            for (int d = 0; d < 3; ++d) { const cd g(gr[d][q], gi[d][q]); const double P = (std::conj(zf) * g).imag(); rawS[d] += area[q] * P; n2 += P * P; }
            rawD += area[q] * std::sqrt(n2);
        }
    }
    const double rawT = std::sqrt(rawS[0] * rawS[0] + rawS[1] * rawS[1] + rawS[2] * rawS[2]) / rawD;
    std::printf("  corrugated sheet: transported T %.4f, direction %.2f deg (truth 0); the raw 3-D sum would give T %.4f\n", pg.phaseGradientAlignment, pg.direction * 180 / kPi, rawT);
    EXPECT(pg.phaseGradientAlignment > 0.97, "corrugated sheet: alignment near 1 (" + std::to_string(pg.phaseGradientAlignment) + ")");
    EXPECT(std::abs(std::remainder(pg.direction - kit.gaugeHeading(anchor, along.data()), 2 * kPi)) < 5 * kPi / 180, "corrugated sheet: the direction is the wave's");
    EXPECT(pg.phaseGradientAlignment > rawT + 0.05, "the raw 3-D sum loses strength across the corrugation (" + std::to_string(rawT) + ")");
    const PhaseGradient ss = weightedPhaseGradient(standingSurrogate(z), ops, fc, area, kit, anchor, rad);
    EXPECT(ss.phaseGradientAlignment < 1e-9, "corrugated sheet: the standing surrogate has T = 0 (" + sci(ss.phaseGradientAlignment) + ")");
}

static void lag() {
    std::cout << "=== two sources, a quarter cycle apart; a planted delay ===\n";
    using namespace nxr::signal::spectral;
    const double fs = 100, f0 = 10;
    const int T = 2000;
    std::mt19937 rng(8);
    std::normal_distribution<double> nd;
    MatrixXd x(2, T);
    for (int t = 0; t < T; ++t) {
        x(0, t) = std::cos(2 * kPi * f0 * t / fs) + 0.01 * nd(rng);
        x(1, t) = std::cos(2 * kPi * f0 * t / fs - kPi / 2) + 0.01 * nd(rng);        // delayed by a quarter cycle: source 0 leads
    }
    WelchOptions wo = scipyWelchOptions(fs, 400);
    const std::size_t bin = static_cast<std::size_t>(f0 * 400 / fs);
    CrossWelch cw(2, wo, {{bin, bin, true}}, true);
    cw.add(x);
    const CrossWelchResult r = cw.result();
    EXPECT(r.Sq.size() == 1 && (r.Sq[0] + r.Sq[0].transpose()).cwiseAbs().maxCoeff() < 1e-12, "CrossWelch's Im C is antisymmetric");
    MatrixXcd S = r.S[0].cast<cd>() + cd(0, 1) * r.Sq[0].cast<cd>();
    EXPECT(r.Sq[0](0, 1) > 0, "Im C_01 > 0: source 0 leads source 1");
    EXPECT(std::abs(quadratureFraction(S) - std::sqrt(0.5)) < 1e-3, "quadratureFraction = 1/sqrt(2) (the two powers are in phase, the cross term is all quadrature): " + std::to_string(quadratureFraction(S)));
    CrossWelch plain(2, wo, {{bin, bin, true}});
    plain.add(x);
    EXPECT(plain.result().Sq.empty() && relErr(plain.result().S[0], r.S[0]) == 0, "without the option CrossWelch is as before");
    const MatrixXcd L = hermitianFactor(S);
    EXPECT(relErr(L * L.adjoint(), S) < 1e-12, "hermitianFactor: S = L L^H");
    const auto P = planes(L, 3);
    EXPECT(P.size() == 1, "one quadrature pair");
    if (!P.empty()) {
        std::printf("  pair: p = (%.3f, %.3f), q = (%.3f, %.3f), sigma %.4g\n", P[0].p[0], P[0].p[1], P[0].q[0], P[0].q[1], P[0].sigma);
        EXPECT(P[0].p[0] > 0.99 && std::abs(P[0].q[1]) > 0.99 && P[0].q[1] > 0, "p = source 0 (the leader), q = source 1");
        EXPECT(lagged_share(quadratureFactor(L), P) > 0.9999, "the pair explains all of Im C");
        EXPECT(quadratureEnergy(L, L)[0] > 0, "energy in row 0");
    }
    EXPECT(std::abs(timeLag(S(0, 1), f0) - 0.025) < 5e-4, "timeLag(C_01) = a quarter cycle, 25 ms (" + std::to_string(timeLag(S(0, 1), f0)) + ")");
    EXPECT(std::abs(timeLag(S(1, 0), f0) + 0.025) < 5e-4, "timeLag(C_10) = -25 ms");
    EXPECT(std::abs(phaseLag(S(0, 1)) - kPi / 2) < 0.02, "phaseLag = pi/2");
    // the rotary spectrum of z = p.x + i q.x: all at +f
    VectorXcd z(T);
    for (int t = 0; t < T; ++t) z[t] = cd(P[0].p.dot(x.col(t)), P[0].q.dot(x.col(t)));
    const RotarySpectrum rs = rotarySpectrum(z, fs, {f0}, 7.0);
    const double pp = rs.positive.middleCols(300, 1400).mean(), nn = rs.negative.middleCols(300, 1400).mean();
    EXPECT(pp > 100 * nn, "rotarySpectrum: z = p.x + i q.x turns at +f (" + sci(pp) + " vs " + sci(nn) + ")");
    // sensor filters
    const QuadraturePairFilters pf = quadraturePairFilters(MatrixXd::Identity(2, 2), S);
    EXPECT(pf.sense == 1 && pf.fx[0] > 0.99 && pf.fy[1] > 0.99 && pf.positive > 100 * pf.negative, "quadraturePairFilters: fx = leader, fy = follower, sense +1");
    // a planted delay (not a quarter cycle)
    for (double d : {0.013, -0.031}) {
        MatrixXd y(2, T);
        for (int t = 0; t < T; ++t) {
            y(0, t) = std::cos(2 * kPi * f0 * t / fs) + 0.01 * nd(rng);
            y(1, t) = std::cos(2 * kPi * f0 * (t / fs - d)) + 0.01 * nd(rng);
        }
        CrossWelch c2(2, wo, {{bin, bin, true}}, true);
        c2.add(y);
        const auto q = c2.result();
        const MatrixXcd Sd = q.S[0].cast<cd>() + cd(0, 1) * q.Sq[0].cast<cd>();
        EXPECT(std::abs(timeLag(Sd(0, 1), f0) - d) < 6e-4, "timeLag recovers the planted delay " + std::to_string(d) + " s (" + std::to_string(timeLag(Sd(0, 1), f0)) + ")");
    }
}

static void stats() {
    std::cout << "=== circular statistics ===\n";
    VectorXd same = VectorXd::Constant(20, 0.7);
    EXPECT(std::abs(circ_r(same) - 1) < 1e-14 && std::abs(circ_rtest(same).meanDirection - 0.7) < 1e-14 && circ_rtest(same).p < 1e-8, "equal directions: R = 1, small p");
    VectorXd uni(16); for (int i = 0; i < 16; ++i) uni[i] = 2 * kPi * i / 16;
    EXPECT(circ_r(uni) < 1e-14 && circ_rtest(uni).p > 0.99, "uniform directions: R = 0, p = 1");
    EXPECT(throws([&] { hermitianFactor(MatrixXcd::Zero(2, 3)); }), "a non-square cross-spectrum refused");
    EXPECT(throws([&] { rotarySpectrum(VectorXcd::Zero(10), 0.0, {10.0}); }), "no sampling frequency refused");
}

static void congruenceEnergy() {
    std::cout << "=== the energy of B C B^T without forming it ===\n";
    std::mt19937 rng(11);
    std::normal_distribution<double> nd;
    const int n = 17, m = 6, r = 3;
    MatrixXd A(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) A(i, j) = nd(rng);
    MatrixXcd Fc(m, r);
    for (int i = 0; i < m; ++i) for (int j = 0; j < r; ++j) Fc(i, j) = cd(nd(rng), nd(rng));
    const MatrixXcd C = Fc * Fc.adjoint();                                   // Hermitian, a rank-3 cross-spectrum with quadrature
    const MatrixXcd T = A * C * A.transpose();                               // the dense form
    const double total = T.squaredNorm(), quad = T.imag().squaredNorm();
    const CongruenceNorms c = congruenceNorms(A, C);
    EXPECT(std::abs(c.total - total) < 1e-10 * total && std::abs(c.quadrature - quad) < 1e-10 * quad, "||T||^2 and ||Im T||^2 equal the dense T's to round-off (" + std::to_string(c.total / total - 1) + ", " + std::to_string(c.quadrature / quad - 1) + ")");
    EXPECT(quad > 0.05 * total, "the planted cross-spectrum has quadrature");
    // grouped rows (groups 0 and 1, row 3 in none), matrices (one empty): each group's block, NaN for the empty matrix and the empty group
    std::vector<int> g(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) g[static_cast<std::size_t>(i)] = i == 3 ? -1 : (i < 9 ? 0 : 1);
    const MatrixXcd C2 = MatrixXcd::Identity(m, m) * cd(2.0, 0);
    const auto gn = groupedCongruenceNorms(A, g, 3, {C, MatrixXcd(), C2});
    std::vector<Eigen::Index> rows0;
    for (int i = 0; i < n; ++i) if (g[static_cast<std::size_t>(i)] == 0) rows0.push_back(i);
    MatrixXd B0(static_cast<Eigen::Index>(rows0.size()), m);
    for (std::size_t k = 0; k < rows0.size(); ++k) B0.row(static_cast<Eigen::Index>(k)) = A.row(rows0[k]);
    const MatrixXcd T0 = B0 * C * B0.transpose();
    EXPECT(std::abs(gn.total(0, 0) - T0.squaredNorm()) < 1e-10 * T0.squaredNorm() && std::abs(gn.quadrature(0, 0) - T0.imag().squaredNorm()) < 1e-10 * T0.imag().squaredNorm(), "a group's rows are its own block");
    const MatrixXcd T2 = B0 * C2 * B0.transpose();
    EXPECT(std::abs(gn.total(0, 2) - T2.squaredNorm()) < 1e-10 * T2.squaredNorm() && std::abs(gn.quadrature(0, 2)) < 1e-12 * T2.squaredNorm(), "an in-phase (real) cross-spectrum has no quadrature");
    EXPECT(std::isnan(gn.total(0, 1)) && std::isnan(gn.total(2, 0)) && std::isnan(gn.quadrature(2, 2)), "an empty matrix and an empty group read NaN");
    // a standing pattern (one real pattern times a phase) has none, a rank-one travelling one at most 1/sqrt(2)
    VectorXd p = VectorXd::Random(m), q = VectorXd::Random(m);
    const VectorXcd standing = cd(0.6, 0.8) * p;
    const MatrixXcd Cs = standing * standing.adjoint();
    const CongruenceNorms cs = congruenceNorms(A, Cs);
    EXPECT(cs.quadrature < 1e-12 * cs.total, "a standing pattern: no quadrature");
    const VectorXcd rot = p.cast<cd>() + cd(0, 1) * q;
    const CongruenceNorms cr = congruenceNorms(A, rot * rot.adjoint());
    EXPECT(std::sqrt(cr.quadrature / cr.total) <= 1 / std::sqrt(2.0) + 1e-12, "a rank-one pattern: the fraction is at most 1/sqrt(2)");
    EXPECT(throws([&] { congruenceNorms(A, MatrixXcd::Zero(m + 1, m + 1)); }) && throws([&] { groupedCongruenceNorms(A, std::vector<int>(3, 0), 1, {C}); }) && throws([&] { groupedCongruenceNorms(A, std::vector<int>(static_cast<std::size_t>(n), 5), 2, {C}); }), "shape and group mismatches refused");
}

// the area-weighted ingredients and the blocks between groups, against the dense forms; the weighted value does not move when a region is sampled more finely, the unweighted one does
static void weightedAndPairs() {
    std::cout << "=== area weights and the blocks between groups ===\n";
    std::mt19937 rng(23);
    std::normal_distribution<double> nd;
    const int n = 20, m = 5, r = 3;
    MatrixXd A(n, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) A(i, j) = nd(rng);
    VectorXd w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.2 + std::abs(nd(rng));
    MatrixXcd F(m, r), F2(m, r);
    for (int i = 0; i < m; ++i) for (int j = 0; j < r; ++j) { F(i, j) = cd(nd(rng), nd(rng)); F2(i, j) = cd(nd(rng), nd(rng)); }
    const MatrixXcd C = F * F.adjoint(), C2 = F2 * F2.adjoint();
    std::vector<int> g(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) g[static_cast<std::size_t>(i)] = i % 4 == 3 ? -1 : (i < 7 ? 0 : (i < 14 ? 1 : 2));
    auto rowsOf = [&](int k) { std::vector<Eigen::Index> v; for (int i = 0; i < n; ++i) if (g[static_cast<std::size_t>(i)] == k) v.push_back(i); return v; };
    // dense weighted block of (a, b): Σ w_i w_j |T_ij|² (T = B_a C B_bᵀ)
    auto dense = [&](int a, int b, const MatrixXcd& Cm, bool weighted) {
        const auto ra = rowsOf(a), rb = rowsOf(b);
        MatrixXd Ba(static_cast<Eigen::Index>(ra.size()), m), Bb(static_cast<Eigen::Index>(rb.size()), m);
        for (std::size_t k = 0; k < ra.size(); ++k) Ba.row(static_cast<Eigen::Index>(k)) = A.row(ra[k]);
        for (std::size_t k = 0; k < rb.size(); ++k) Bb.row(static_cast<Eigen::Index>(k)) = A.row(rb[k]);
        const MatrixXcd T = Ba * Cm * Bb.transpose();
        double tot = 0, qd = 0;
        for (std::size_t i = 0; i < ra.size(); ++i) for (std::size_t j = 0; j < rb.size(); ++j) {
            const double ww = weighted ? w[ra[i]] * w[rb[j]] : 1.0;
            tot += ww * std::norm(T(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
            qd += ww * T(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).imag() * T(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).imag();
        }
        return std::make_pair(tot, qd);
    };
    const std::vector<std::pair<int, int>> pairs = {{0, 1}, {1, 2}, {0, 2}};
    const std::vector<MatrixXcd> Cs = {C, MatrixXcd(), C2};
    const auto pn = pairCongruenceNorms(A, g, 4, pairs, Cs, w);
    const auto gw = groupedCongruenceNorms(A, g, 4, Cs, w);
    double worst = 0;
    for (int k = 0; k < 3; ++k)
        for (int t : {0, 2}) {
            const auto d = dense(pairs[static_cast<std::size_t>(k)].first, pairs[static_cast<std::size_t>(k)].second, Cs[static_cast<std::size_t>(t)], true);
            worst = std::max({worst, std::abs(pn.pairTotal(k, t) - d.first) / d.first, std::abs(pn.pairQuadrature(k, t) - d.second) / d.second});
        }
    EXPECT(worst < 1e-10, "the weighted pair ingredients equal the dense weighted block (worst relative error " + std::to_string(worst) + ")");
    double wt = 0;
    for (int a = 0; a < 3; ++a) for (int t : {0, 2}) {
        const auto d = dense(a, a, Cs[static_cast<std::size_t>(t)], true);
        wt = std::max({wt, std::abs(gw.total(a, t) - d.first) / d.first, std::abs(pn.tiles.total(a, t) - d.first) / d.first, std::abs(pn.tiles.quadrature(a, t) - d.second) / d.second});
    }
    EXPECT(wt < 1e-10, "the weighted tile ingredients (grouped and in the pair node) equal the dense weighted form (" + std::to_string(wt) + ")");
    EXPECT(std::isnan(pn.pairTotal(0, 1)) && std::isnan(pn.tiles.total(3, 0)) && std::isnan(pn.pairTotal(0, 1)), "an empty matrix reads NaN in the pairs, and an empty group in the tiles");
    // no weights = all ones = the unweighted dense form
    const auto pu = pairCongruenceNorms(A, g, 4, pairs, Cs);
    const auto du = dense(0, 1, C, false);
    EXPECT(std::abs(pu.pairTotal(0, 0) - du.first) < 1e-10 * du.first && std::abs(pu.pairQuadrature(0, 0) - du.second) < 1e-10 * du.second, "no weights: the unweighted dense block");
    EXPECT(std::abs(pn.pairTotal(0, 0) - du.first) > 1e-3 * du.first, "the weights change the value");
    EXPECT(throws([&] { pairCongruenceNorms(A, g, 4, {{0, 9}}, Cs, w); }) && throws([&] { pairCongruenceNorms(A, g, 4, pairs, Cs, VectorXd::Ones(3)); }), "a pair outside the groups and a wrong number of weights are refused");
    // a pair of groups is symmetric: (a, b) and (b, a) are the same energy
    const auto sw = pairCongruenceNorms(A, g, 4, {{1, 0}}, {C}, w);
    EXPECT(std::abs(sw.pairTotal(0, 0) - pn.pairTotal(0, 0)) < 1e-10 * pn.pairTotal(0, 0) && std::abs(sw.pairQuadrature(0, 0) - pn.pairQuadrature(0, 0)) < 1e-10 * pn.pairQuadrature(0, 0), "T_ba is T_ab conjugate-transposed: the same norms");
    // REFINING A REGION: points on a line, B_i = φ(x_i) a smooth map of x into the channels, the area a_i the spacing. Group 0 = x < 0.5, group 1 = x >= 0.5; refining group 1 (4x the points)
    // leaves the weighted value near its integral and multiplies the unweighted by the count
    auto build = [&](int n0, int n1, MatrixXd& B, VectorXd& a, std::vector<int>& grp) {
        B.resize(n0 + n1, m); a.resize(n0 + n1); grp.assign(static_cast<std::size_t>(n0 + n1), 0);
        for (int i = 0; i < n0 + n1; ++i) {
            const bool second = i >= n0;
            const double x = second ? 0.5 + 0.5 * (i - n0 + 0.5) / n1 : 0.5 * (i + 0.5) / n0;
            a[i] = second ? 0.5 / n1 : 0.5 / n0;
            for (int j = 0; j < m; ++j) B(i, j) = std::cos((j + 1) * x * 3.0) + 0.3 * j * x;
            grp[static_cast<std::size_t>(i)] = second ? 1 : 0;
        }
    };
    MatrixXd Bc, Bf; VectorXd ac, af; std::vector<int> gc, gf;
    build(40, 40, Bc, ac, gc); build(40, 160, Bf, af, gf);
    const auto coarseW = pairCongruenceNorms(Bc, gc, 2, {{0, 1}}, {C}, ac), fineW = pairCongruenceNorms(Bf, gf, 2, {{0, 1}}, {C}, af);
    const auto coarseU = pairCongruenceNorms(Bc, gc, 2, {{0, 1}}, {C}), fineU = pairCongruenceNorms(Bf, gf, 2, {{0, 1}}, {C});
    const double dW = std::abs(fineW.pairTotal(0, 0) / coarseW.pairTotal(0, 0) - 1), rU = fineU.pairTotal(0, 0) / coarseU.pairTotal(0, 0);
    std::cout << "  refining one region 4x: weighted pair energy changes by " << dW << ", unweighted multiplies by " << rU << "\n";
    EXPECT(dW < 0.01, "area-weighted: the pair's energy is the same when one region is sampled four times more finely");
    EXPECT(rU > 3.5 && rU < 4.5, "unweighted: the same energy scales with the number of points");
}

int main() {
    parity();
    zeroDelay();
    waves();
    corrugated();
    lag();
    stats();
    congruenceEnergy();
    weightedAndPairs();
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}

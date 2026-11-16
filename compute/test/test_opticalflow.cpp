// nxr::flow: (1) parity with Rheome (golden: tangent_basis, flow_readout, activation on a tiny irregular sphere);
// (2) Horn–Schunck on a bump rotating on a sphere gives a velocity in the bump's direction;
// (3) the Hodge–Dirac reading of a velocity field: a pure-divergence flow (-grad of an exact mode) reads only exact coefficients, a pure rotational one
//     (n x grad of a coexact mode) only coexact ones, and the divergence / curl readouts agree in sign.
#include <cmath>
#include <iostream>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/opticalflow.h"

using namespace nxr::flow;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); }

int main() {
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/opticalflow_golden.txt");
    EXPECT(g.good(), "golden file");
    const MatrixXd Vg = g.mat("positions"), Fg = g.mat("faces"), N = g.mat("normals"), A = g.mat("A"), e1 = g.mat("e1"), e2 = g.mat("e2"), v1 = g.mat("v1"), v2 = g.mat("v2"),
                   dv = g.mat("div"), cu = g.mat("curl"), sp = g.mat("speed"), J = g.mat("J"), an = g.mat("act_analytic"), nm = g.mat("act_norm");
    const RowMat3 pos = Vg;
    const FaceMat faces = Fg.cast<std::int32_t>();
    const RowMat3 nrm = N;
    {
        const TangentBasis tb = tangentBasis(nrm);
        std::printf("  tangent basis vs Rheome: %.2e\n", std::max(rel(tb.e1, e1), rel(tb.e2, e2)));
        EXPECT(std::max(rel(tb.e1, e1), rel(tb.e2, e2)) < 1e-12, "tangentBasis matches Rheome");
        EXPECT(rel(vertexNormals(pos, faces), N) < 1e-12, "vertexNormals are Rheome's area-weighted normals");
    }
    const FlowReadout r = flowReadout(v1, pos, faces);
    std::printf("  flowReadout vs Rheome: div %.2e curl %.2e speed %.2e\n", rel(r.divergence, dv), rel(r.vorticity, cu), rel(r.speed, sp));
    EXPECT(rel(r.divergence, dv) < 1e-10 && rel(r.vorticity, cu) < 1e-10 && rel(r.speed, sp) < 1e-10, "flowReadout matches Rheome to 1e-10");
    EXPECT(rel(activation(J.cast<std::complex<double>>(), ActivationMethod::Analytic, false), an) < 1e-10, "activation (analytic envelope) matches Rheome");
    EXPECT(rel(activation(J.cast<std::complex<double>>(), ActivationMethod::Norm), nm) < 1e-12, "activation (norm) matches Rheome");
    const Apparent ap = apparent(A, pos, faces, {300.0, 1.0});
    EXPECT(rel(ap.speed, ap.speedPerFrame * 300.0) < 1e-12 && ap.velocity.cols() == A.cols() - 1, "apparent: speed = per-frame speed x rate");

    // (2) a bump rotating about z on an icosphere: the velocity points the way it moves
    std::vector<double> Vv; std::vector<int32_t> Ff;
    icosphere(3, Vv, Ff);
    RowMat3 P = Eigen::Map<RowMat3>(Vv.data(), Vv.size() / 3, 3);
    FaceMat Fm = Eigen::Map<FaceMat>(Ff.data(), Ff.size() / 3, 3);
    const double theta = 0.06;
    MatrixXd bump(P.rows(), 3);
    const Eigen::Vector3d c0 = Eigen::Vector3d(1, 0.3, 0.5).normalized();
    for (int t = 0; t < 3; ++t) {
        const Eigen::Vector3d c = Eigen::AngleAxisd(theta * t, Eigen::Vector3d::UnitZ()) * c0;
        for (Eigen::Index i = 0; i < P.rows(); ++i) bump(i, t) = 0.1 + std::exp(-(P.row(i).transpose() - c).squaredNorm() / 0.3);
    }
    const MatrixXd v = opticalflowScalar(bump, P, Fm, vertexNormals(P, Fm), 0.01);
    double num = 0, den = 0, cosSum = 0, wsum = 0;
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
        const Eigen::Vector3d x = P.row(i), truth = Eigen::Vector3d::UnitZ().cross(x) * theta, est = v.block<3, 1>(3 * i, 0);
        const double wgt = (bump(i, 1) - bump(i, 0)) * (bump(i, 1) - bump(i, 0));   // where the pattern changes
        if (truth.norm() < 1e-9 || est.norm() < 1e-12) continue;
        num += wgt * est.dot(truth); den += wgt * truth.squaredNorm();
        cosSum += wgt * est.dot(truth) / (est.norm() * truth.norm()); wsum += wgt;
    }
    std::printf("  bump: weighted cosine %.3f, speed ratio %.3f\n", cosSum / wsum, num / den);
    EXPECT(cosSum / wsum > 0.8, "Horn-Schunck velocity of a moving bump points in the bump's direction");
    EXPECT(num / den > 0.2 && num / den < 2.0, "and has the order of its speed (aperture-limited)");

    // (3) the Hodge–Dirac reading, on the glued icosphere of the hodge-weights golden
    Golden h(std::string(NXR_TEST_GOLDEN_DIR) + "/hodge_weights_golden.txt");
    nxr::hodge::Basis b;
    const MatrixXd V2 = h.mat("positions"), F2 = h.mat("faces"), P2 = h.mat("P"), mu = h.mat("mu"), kind = h.mat("kind"), mass = h.mat("mass"), keep = h.mat("keep");
    b.P = P2; b.positions = V2; b.faces = F2.cast<std::int32_t>(); b.mu = mu.row(0).transpose(); b.mass = mass.row(0).transpose();
    for (int i = 0; i < kind.cols(); ++i) b.kind.push_back(static_cast<int>(kind(0, i)));
    for (int i = 0; i < keep.cols(); ++i) b.keep.push_back(static_cast<std::int64_t>(keep(0, i)));
    const nxr::hodge::FaceOperators ops(b.positions, b.faces);
    const int K = static_cast<int>(P2.rows()), V = static_cast<int>(P2.cols());
    int ex = -1, co = -1, exs = 0, cos_ = 0;
    for (int k = 0; k < K; ++k) {
        if (b.kind[k] == 2) { if (co < 0 && b.mu[k] > 1e-8) co = k; ++cos_; } else { if (ex < 0 && b.mu[k] > 1e-8) ex = k; ++exs; }
    }
    std::printf("  basis: %d modes (%d exact, %d coexact); using exact %d, coexact %d\n", K, exs, cos_, ex, co);
    auto flowOf = [&](int k, bool rotational) {
        MatrixXd g[3];
        ops.vertexGradient(P2.row(k).transpose(), g);
        MatrixXd vel(3 * V, 1);
        for (int i = 0; i < V; ++i) {
            const Eigen::Vector3d gr(g[0](i, 0), g[1](i, 0), g[2](i, 0)), n = ops.normals.row(i), f = rotational ? Eigen::Vector3d(n.cross(gr)) : Eigen::Vector3d(-gr);
            vel.block<3, 1>(3 * i, 0) = f;
        }
        return vel;
    };
    for (int pass = 0; pass < 2; ++pass) {
        const int k = pass == 0 ? ex : co;
        const MatrixXd c = flowCoefficients(b, flowOf(k, pass == 1));
        double same = 0, other = 0;
        for (int j = 0; j < K; ++j) { const bool isCo = b.kind[j] == 2; if ((pass == 1) == isCo) same += c(j, 0) * c(j, 0); else other += c(j, 0) * c(j, 0); }
        std::printf("  %s flow: power on its own kind %.3e, on the other %.3e, mode %d coefficient %.3f\n", pass == 0 ? "pure divergence" : "pure rotation", same, other, k, c(k, 0));
        EXPECT(other < 1e-12 * same, (pass == 0 ? "a pure-divergence flow reads only exact coefficients" : "a pure rotational flow reads only coexact coefficients"));
        EXPECT(std::abs(std::abs(c(k, 0)) - 1) < 1e-8, "and reads the mode it was made of (unit coefficient)");
    }
    // (4) the streaming forms equal the all-at-once ones: OpticalflowScalar::step, FlowReader::read, WeightsReader::flow
    {
        OpticalflowScalar st(P, Fm, vertexNormals(P, Fm), 0.01, bump.cwiseAbs().maxCoeff());
        const FlowReader fr(P, Fm);
        const nxr::hodge::WeightsReader wr(b);
        const FlowReadout full = flowReadout(v, P, Fm);
        double e1 = 0, e2 = 0, e3 = 0;
        const MatrixXd rnd = MatrixXd::Random(3 * V, 3);
        const MatrixXd all = nxr::hodge::hodgeWeights(b, rnd);
        for (int t = 0; t < 2; ++t) {
            const VectorXd vt = st.step(bump.col(t), bump.col(t + 1));
            e1 = std::max(e1, (vt - v.col(t)).cwiseAbs().maxCoeff());
            VectorXd d, c, s;
            fr.read(vt, d, c, s);
            e2 = std::max({e2, (d - full.divergence.col(t)).cwiseAbs().maxCoeff(), (c - full.vorticity.col(t)).cwiseAbs().maxCoeff(), (s - full.speed.col(t)).cwiseAbs().maxCoeff()});
        }
        for (int j = 0; j < 3; ++j) e3 = std::max(e3, (wr.flow(rnd.col(j)).col(0) - all.col(j).tail(K)).cwiseAbs().maxCoeff());
        std::printf("  streaming vs all at once: step %.1e, readout %.1e, flow weights %.1e\n", e1, e2, e3);
        EXPECT(e1 < 1e-12 && e2 < 1e-12 && e3 < 1e-12, "the streaming steps equal the all-at-once versions");
    }
    std::cout << (g_failures ? "FAILED\n" : "ok\n");
    return g_failures ? 1 : 0;
}

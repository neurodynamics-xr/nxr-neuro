// Hodge–Dirac fused weights: (1) parity with nxr.inverse.fused (golden: gluedKernel, fused_weights, synthesis on a tiny synthetic glued icosphere);
// (2) reading back: hodgeWeights(hodgeSynthesis(w)) == w; (3) mode power from the cross-spectrum == Welch run on B x directly.
#include <cmath>
#include <iostream>
#include <random>

#include "golden_io.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/measurement/mode_power.h"
#include "nxr/signal/spectral.h"

using namespace nxr::operators;
using Eigen::MatrixXd;
static int g_failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)
static double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }

int main() {
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/hodge_weights_golden.txt");
    EXPECT(g.good(), "golden file");
    Basis b;
    const MatrixXd V = g.mat("positions"), F = g.mat("faces"), P = g.mat("P"), mu = g.mat("mu"), kind = g.mat("kind"), mass = g.mat("mass"), keep = g.mat("keep"),
                   Vin = g.mat("input_positions"), W = g.mat("W"), Wg = g.mat("Wg"), Bg = g.mat("B"), X = g.mat("X"), Jg = g.mat("J");
    b.P = P; b.mu = mu.col(0); b.mass = mass.col(0);
    b.positions = V; b.faces = F.cast<std::int32_t>();
    for (int i = 0; i < kind.cols(); ++i) b.kind.push_back(static_cast<int>(kind(0, i)));
    for (int i = 0; i < keep.cols(); ++i) b.keep.push_back(static_cast<std::int64_t>(keep(0, i)));
    // the golden stores vectors as 1 x n rows
    b.mu = mu.row(0).transpose(); b.mass = mass.row(0).transpose();
    const MatrixXd Wc = gluedKernel(W, b, Vin);
    const double eg = rel(Wc, Wg);
    std::printf("  glued kernel vs nxr: %.2e\n", eg);
    EXPECT(eg < 1e-12, "gluedKernel matches nxr glued_kernel");
    const MatrixXd B = hodgeWeights(b, Wc);
    const double eb = rel(B, Bg);
    std::printf("  fused weights vs nxr: %.2e\n", eb);
    EXPECT(eb < 1e-8, "hodgeWeights matches nxr fused_weights");
    const Synthesis s = hodgeSynthesis(b, Bg * X);
    const double ej = rel(s.J, Jg);
    std::printf("  synthesis vs nxr: %.2e\n", ej);
    EXPECT(ej < 1e-8, "hodgeSynthesis matches nxr synthesis");
    // (2) the read-back: a synthesised current gives its weights back, whatever they are
    std::mt19937 rng(3);
    std::normal_distribution<double> nd;
    MatrixXd w(B.rows(), 5);
    for (auto& v : w.reshaped()) v = nd(rng);
    for (Eigen::Index k = 0; k < B.rows() / 2; ++k) if (b.kind[static_cast<std::size_t>(k)] != 2) w.row(k).setZero();   // an exact mode has no normal-current weight
    const MatrixXd back = hodgeWeights(b, hodgeSynthesis(b, w).J);
    std::printf("  read-back: %.2e\n", rel(back, w));
    EXPECT(rel(back, w) < 1e-8, "hodgeWeights(hodgeSynthesis(w)) == w");
    // (3) mode power by the cross-spectrum == Welch on B x
    const int C = static_cast<int>(B.cols()), T = 4000;
    MatrixXd x(C, T);
    for (auto& v : x.reshaped()) v = nd(rng);
    auto wo = nxr::signal::spectral::brainstormPsdOptions(250.0, 4.0);
    const std::size_t bins = wo.segmentSamples / 2 + 1;
    const std::vector<nxr::signal::spectral::BinGroup> groups = {{8, 24, true}, {30, 60, true}};
    nxr::signal::spectral::CrossWelch cw(C, wo, groups);
    cw.add(x);
    const auto r = cw.result();
    const MatrixXd got = nxr::measurement::modePower(B, r.S);
    const auto direct = nxr::signal::spectral::welch(B * x, wo);
    MatrixXd want(B.rows(), 2);
    for (int i = 0; i < 2; ++i) want.col(i) = direct.psd.middleCols(groups[i].first, groups[i].last - groups[i].first + 1).rowwise().mean();
    (void)bins;
    std::printf("  mode power CSD vs Welch on B x: %.2e\n", rel(got, want));
    EXPECT(rel(got, want) < 1e-10, "modePower == Welch of B x");
    std::cout << (g_failures ? "FAILED\n" : "ok\n");
    return g_failures ? 1 : 0;
}

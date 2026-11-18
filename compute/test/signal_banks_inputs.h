// The synthetic inputs of the banks' golden (gen_signal_banks_golden.cpp made it from the code before the refactor of the transforms into wt / iwt / gwt / igwt / jwt / ijwt;
// test_signal_banks_golden.cpp compares the new code to it): signals, banks, a path-graph operator with its modes. Synthetic only.
#pragma once
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/jointfilterbank.h"
#include "nxr/signal/timefilterbank.h"
#include <random>
namespace sbi {
namespace tb = nxr::signal::timefilterbank;
namespace fb = nxr::signal::graphfilterbank;
using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::MatrixXcd;
constexpr double kFs = 250;
inline MatrixXd noise(int r, int c, unsigned seed) { std::mt19937 g(seed); std::normal_distribution<double> N; MatrixXd x(r, c); for (int j = 0; j < c; ++j) for (int i = 0; i < r; ++i) x(i, j) = N(g); return x; }
inline MatrixXd tones(int n, int C, unsigned seed, double fmax) {   // [n x C]: random tones below fmax
    std::mt19937 g(seed); std::uniform_real_distribution<double> U(0, 1); MatrixXd x = MatrixXd::Zero(n, C);
    for (int c = 0; c < C; ++c) for (int k = 0; k < 30; ++k) { const double f = 0.5 + U(g) * (fmax - 0.5), a = U(g), ph = 6.283185307179586 * U(g); for (int i = 0; i < n; ++i) x(i, c) += a * std::cos(6.283185307179586 * f * i / kFs + ph); }
    return x;
}
inline tb::TimeFilterbank itersine(std::size_t nfft = 256) { tb::Design d(tb::Family::Itersine, 2.0, tb::Sides::Analytic); d.voices = 2; d.limits = std::make_pair(2.0, 100.0); return tb::design(d, nfft, kFs); }
inline tb::TimeFilterbank meyer(std::size_t nfft = 256) { tb::Design d(tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.limits = std::make_pair(2.0, 60.0); return tb::design(d, nfft, kFs); }
inline tb::TimeFilterbank morlet(std::size_t nfft = 256) { tb::Design d(tb::Family::Morlet, 2.0, tb::Sides::Analytic); d.centres = {4, 8, 16, 32}; d.cycles = 6; return tb::design(d, nfft, kFs); }
inline tb::TimeFilterbank twoSided(std::size_t nfft = 128) { tb::Design d(tb::Family::Itersine, 1.0, tb::Sides::TwoSided); d.voices = 2; d.limits = std::make_pair(2.0, 100.0); return tb::design(d, nfft, kFs); }
struct Graph { MatrixXd Ld, Md; Eigen::SparseMatrix<double> L, M; fb::EigenTransform T; VectorXd lam; int n = 0; };
inline Graph pathGraph(int n = 24) {
    Graph g; g.n = n; g.Ld = MatrixXd::Zero(n, n); g.Md = MatrixXd::Zero(n, n);
    for (int i = 0; i + 1 < n; ++i) { const double w = 1.0 + 0.3 * std::sin(i); g.Ld(i, i) += w; g.Ld(i + 1, i + 1) += w; g.Ld(i, i + 1) -= w; g.Ld(i + 1, i) -= w; }
    for (int i = 0; i < n; ++i) g.Md(i, i) = 1.0 + 0.2 * std::cos(i);
    g.L = g.Ld.sparseView(); g.M = g.Md.sparseView();
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(g.Ld, g.Md);
    g.T.phi = es.eigenvectors(); g.T.lambda = es.eigenvalues(); g.T.mass = g.M;
    for (int i = 0; i < g.T.lambda.size(); ++i) if (std::abs(g.T.lambda(i)) < 1e-10) g.T.lambda(i) = 0;
    g.lam = g.T.lambda; return g;
}
inline fb::GraphFilterbank graphBank(const Graph& g, fb::Family f = fb::Family::Itersine) { return fb::design(f, fb::Range::fromSpectrum(g.lam)); }
}  // namespace sbi

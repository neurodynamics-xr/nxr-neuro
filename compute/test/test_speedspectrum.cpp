// nxr::speedspectrum on synthetic mode series (no data): a travelling wave (a degenerate mode pair in quadrature at f, wavenumber mu) puts the speed
// histogram's peak at 2 pi f / mu and its tile's quadrature fraction high; a standing oscillation (the same modes in phase) has the same tile energy
// and quadrature fraction ~0; the tiles of a tight bank sum to the power. The mode series go through the real CrossWelch (one-bin groups).
#include "nxr/speedspectrum.h"
#include "nxr/welch.h"
#include "mesh_helpers.h"
#include <random>
namespace ss = nxr::speedspectrum;
using Eigen::MatrixXd;
using Eigen::VectorXd;
static constexpr double kPi = 3.14159265358979323846;

struct Case { ss::Result r; double c = 0, power = 0; };

static Case run(bool travelling) {
    const double fs = 100, f = 30 * fs / 256;       // on bin 30
    const int n = 4096, K = 12;
    VectorXd mu(K);
    std::vector<int> kind(K, 2);
    for (int k = 0; k < K; ++k) mu(k) = std::exp(std::log(1.0) + (std::log(9.0) / (K - 1)) * k);
    mu(4) = mu(5) = 3.0;                             // the degenerate pair that carries the wave
    kind[0] = kind[1] = 0;                           // two exact modes with noise only
    MatrixXd x(K, n);
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0.0, 0.02);
    for (int k = 0; k < K; ++k) for (int t = 0; t < n; ++t) x(k, t) = N(rng);
    for (int t = 0; t < n; ++t) {
        const double ph = 2 * kPi * f * t / fs;
        x(4, t) += std::cos(ph);
        x(5, t) += travelling ? std::sin(ph) : std::cos(ph);
    }
    nxr::welch::WelchOptions wo = nxr::welch::scipyWelchOptions(fs, 256);
    wo.scaling = nxr::welch::Scaling::Spectrum;
    std::vector<nxr::welch::BinGroup> groups;
    for (std::size_t b = 1; b < 128; ++b) groups.push_back({b, b, true});
    nxr::welch::CrossWelch w(K, wo, groups, true);
    w.add(x);
    const auto cw = w.result();
    ss::ModeSpectra sp;
    sp.frequencies.resize(127);
    for (int b = 1; b < 128; ++b) sp.frequencies(b - 1) = b * fs / 256;
    sp.S = cw.S; sp.Sq = cw.Sq;
    const auto graph = nxr::filterbank::design(nxr::filterbank::Family::LogItersine, nxr::filterbank::Range::fromSpectrum(mu.array().square()));
    nxr::timebank::Design td(nxr::timebank::Family::Itersine, 1.0, nxr::timebank::Sides::Analytic);
    const auto time = nxr::timebank::design(td, 256, fs);
    ss::Options o; o.speedBins = 60;
    Case c;
    c.r = ss::compute(graph, time, mu, kind, sp, o);
    c.c = 2 * kPi * f / 3.0;
    for (const auto& p : c.r.parts) c.power += p.power;
    return c;
}

static const ss::Part& part(const Case& c, int kind) { for (const auto& p : c.r.parts) if (p.kind == kind) return p; throw std::runtime_error("no part"); }

int main() {
    const Case tr = run(true), st = run(false);
    std::cout << "=== speed spectrum: a travelling wave and a standing oscillation at the same (mu, f) ===\n";
    EXPECT(tr.r.parts.size() == 2 && tr.r.parts[0].kind == 0 && tr.r.parts[1].kind == 2, "exact and coexact modes are kept apart");
    const ss::Part& p = part(tr, 2);
    // histogram, summed over the spatial scales: the peak bin holds 2 pi f / mu
    Eigen::Index pk = 0;
    const VectorXd h = p.histogram.colwise().sum().transpose();
    h.maxCoeff(&pk);
    const double lo = tr.r.speedEdges(pk), hi = tr.r.speedEdges(pk + 1);
    EXPECT(lo <= tr.c && tr.c < hi, "histogram peak bin [" + std::to_string(lo) + ", " + std::to_string(hi) + ") holds c = 2 pi f / mu = " + std::to_string(tr.c));
    EXPECT(h(pk) > 0.9 * h.sum(), "the peak bin holds " + std::to_string(h(pk) / h.sum()) + " of the coexact energy");
    // the energy tile and its quadrature
    Eigen::Index gi = 0, ti = 0;
    p.tileEnergy.maxCoeff(&gi, &ti);
    const ss::Part& q = part(st, 2);
    Eigen::Index gs = 0, ts = 0;
    q.tileEnergy.maxCoeff(&gs, &ts);
    EXPECT(gi == gs && ti == ts, "the same tile carries the energy in both cases");
    EXPECT(std::abs(p.tileEnergy(gi, ti) - q.tileEnergy(gs, ts)) < 0.05 * p.tileEnergy(gi, ti), "the same tile energy (" + std::to_string(p.tileEnergy(gi, ti)) + " vs " + std::to_string(q.tileEnergy(gs, ts)) + ")");
    EXPECT(p.quadrature(gi, ti) > 0.65, "travelling: quadrature fraction " + std::to_string(p.quadrature(gi, ti)) + " (two modes in exact quadrature: 0.707)");
    EXPECT(q.quadrature(gs, ts) < 0.05, "standing: quadrature fraction " + std::to_string(q.quadrature(gs, ts)) + " ~ 0");
    EXPECT(std::abs(p.tileSpeed(gi, ti) - tr.c) / tr.c < 0.6, "the tile's centre speed " + std::to_string(p.tileSpeed(gi, ti)) + " is near c (the tile is a band)");
    // frame bounds: a tight bank conserves the power
    double sum = 0;
    for (const auto& pt : tr.r.parts) sum += pt.tileEnergy.sum();
    EXPECT(std::abs(sum - tr.power) / tr.power < 1e-6, "tile energies sum to the total power: " + std::to_string(sum) + " vs " + std::to_string(tr.power));
    double hsum = 0;
    for (const auto& pt : tr.r.parts) hsum += pt.histogram.sum() + pt.outside.sum();
    EXPECT(std::abs(hsum - tr.power) / tr.power < 1e-6, "the speed histograms (and what lies outside) sum to the total power");
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}

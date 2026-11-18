// Shared by golden/gen_welch_golden_bits.cpp (the OLD Welch classes) and test_welch_bits.cpp (the NEW stft + estimate composition): one deterministic synthetic recording (sines, no random generator), a mask with
// bad samples, and the runs both make. The old results are recorded as hashes of the raw doubles (bitwise).
#pragma once
#include "golden_hash.h"
#include "nxr/signal/spectral.h"
#include <cmath>
namespace wi {
using Eigen::MatrixXd; using Eigen::MatrixXcd;
namespace sp = nxr::signal::spectral;
constexpr int kC = 4, kN = 3000; constexpr double kFs = 200.0;
inline MatrixXd recording() {
    MatrixXd x(kC, kN);
    for (int c = 0; c < kC; ++c) for (int i = 0; i < kN; ++i) {
        double v = 0; for (int k = 0; k < 9; ++k) v += std::cos(2 * M_PI * (1.3 + 4.1 * k + 0.7 * c) * i / kFs + 0.9 * k + c) / (1.0 + 0.2 * k);
        x(c, i) = v + 0.2 * std::sin(0.37 * i * (1 + c));
    }
    return x;
}
inline std::vector<std::uint8_t> mask() { std::vector<std::uint8_t> g(kN, 1); for (int i = 700; i < 760; ++i) g[i] = 0; for (int i = 2100; i < 2105; ++i) g[i] = 0; return g; }
inline sp::WelchOptions options(int which) {
    if (which == 0) return sp::brainstormPsdOptions(kFs, 1.0);       // symmetric Hamming, density
    if (which == 1) return sp::nxrChainWelchOptions(kFs, 1.0);       // chain scaling, no detrend
    sp::WelchOptions o = sp::scipyWelchOptions(kFs, 128); o.detrend = sp::Detrend::Linear; o.scaling = sp::Scaling::Spectrum; o.overlap = 0.75; return o;
}
// every run the golden records: name -> matrices, the record fed in chunks of `chunk` samples (0: whole)
template <class Put> void runs(Put&& put, int chunk) {
    const MatrixXd x = recording(); const auto good = mask();
    MatrixXd map(6, kC); for (int r = 0; r < 6; ++r) for (int c = 0; c < kC; ++c) map(r, c) = std::sin(1.0 + r * 1.7 + c * 0.6);
    MatrixXcd z(2, kN); for (int i = 0; i < kN; ++i) { z(0, i) = {x(0, i), x(1, i)}; z(1, i) = {x(2, i), -x(3, i)}; }
    auto feed = [&](auto& acc, const auto& data, bool masked) {
        const int step = chunk ? chunk : kN;
        for (int a = 0; a < kN; a += step) { const int m = std::min(step, kN - a); const auto part = data.middleCols(a, m).eval();
            if (masked) acc.add(part, std::vector<std::uint8_t>(good.begin() + a, good.begin() + a + m)); else acc.add(part); }
    };
    for (int w = 0; w < 3; ++w) {
        const std::string p = "opt" + std::to_string(w) + ".";
        const sp::WelchOptions o = options(w);
        { sp::Welch a(kC, o); feed(a, x, true); const auto r = a.result(); put(p + "psd", r.psd); put(p + "psdSeg", double(r.segments)); put(p + "psdSkip", double(r.segmentsSkipped)); }
        { sp::Welch a(kC, o); feed(a, x, false); put(p + "psdClean", a.result().psd); }
        { sp::ComplexWelch a(2, o); feed(a, z, true); const auto r = a.result(); put(p + "cpsd", r.psd); put(p + "cpsdSeg", double(r.segments)); }
        const std::size_t bins = o.segmentSamples / 2 + 1;
        const std::vector<sp::BinGroup> groups = {{1, 5, true}, {3, bins - 1, false}, {0, bins - 1, true}};
        { sp::CrossWelch a(kC, o, groups, true); feed(a, x, true); const auto r = a.result(); for (std::size_t g = 0; g < groups.size(); ++g) { put(p + "xS" + std::to_string(g), r.S[g]); put(p + "xSq" + std::to_string(g), r.Sq[g]); } }
        { sp::CrossWelch a(kC, o, groups, false); feed(a, x, true); const auto r = a.result(); put(p + "xplain", r.S[0]); }
        { sp::LinearWelch a(map, 3, o, 2, bins - 2); feed(a, x, true); const auto r = a.result(); put(p + "lin3", r.psd); }
        { sp::LinearWelch a(map, 1, o, 0, bins - 1); feed(a, x, true); put(p + "lin1", a.result().psd); }
        put(p + "fnPsd", sp::psd(x, o, &good).psd); put(p + "fnXpsd", sp::xpsd(x, o, groups, true, &good).Sq[1]); put(p + "fnCpsd", sp::psd(z, o, &good).psd);
    }
}
}  // namespace wi

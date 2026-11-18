// Generator of frames_wt_golden.txt: the OLD frames path (the band's FIR taps, Morlet and Meyer, then the Kaiser sinc of half width 32 at the frames), compiled against the code of 2026-11-18 BEFORE the
// frames step was fused with the analytic step (it needs nxr/signal/resample.h, removed in that change; to remake it, check out the parent of that commit). Not built by CMake.
// A line: name, count, then count values (re, im interleaved per channel then instant), %.17g.
#include "../frames_wt_inputs.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/resample.h"
#include <cstdio>
int main(int argc, char** argv) {
    FILE* f = std::fopen(argc > 1 ? argv[1] : "frames_wt_golden.txt", "w");
    const Eigen::MatrixXd x = fwi::recording();
    std::vector<float> xf(static_cast<std::size_t>(fwi::kC * fwi::kN));
    for (int c = 0; c < fwi::kC; ++c) for (int i = 0; i < fwi::kN; ++i) xf[static_cast<std::size_t>(c * fwi::kN + i)] = static_cast<float>(x(c, i));
    const std::vector<double> pos = fwi::instants();
    for (const char* shape : {"morlet", "meyer"}) {
        const auto gain = [&](double f_) { return std::string(shape) == "morlet" ? nxr::signal::graphfilterbank::morletBand(f_, 8.0, 12.0) : nxr::signal::graphfilterbank::meyerBand(f_, 8.0, 12.0, 1.0 / 3.0); };
        const auto h = nxr::signal::gainTaps(gain, fwi::kFs, false), q = nxr::signal::gainTaps(gain, fwi::kFs, true);
        const auto re = nxr::signal::convolveRows(xf.data(), fwi::kC, fwi::kN, h, 0, fwi::kN, false), im = nxr::signal::convolveRows(xf.data(), fwi::kC, fwi::kN, q, 0, fwi::kN, true);
        const auto yr = nxr::signal::sincInterpolate(re.data(), fwi::kC, fwi::kN, pos, 32), yi = nxr::signal::sincInterpolate(im.data(), fwi::kC, fwi::kN, pos, 32);
        std::fprintf(f, "%s %zu", shape, yr.size() * 2);
        for (std::size_t i = 0; i < yr.size(); ++i) std::fprintf(f, " %.17g %.17g", static_cast<double>(yr[i]), static_cast<double>(yi[i]));
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

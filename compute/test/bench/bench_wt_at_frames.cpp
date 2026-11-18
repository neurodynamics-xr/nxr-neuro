// Timing, not a test: the analytic signal of a band read at a frame grid's instants, two ways, on a synthetic recording (10 Hz in noise, 1200 Hz, 36 frames per cycle = a frame every 3.33 samples):
//   OLD  the band's analytic signal on the whole block (wt of a Meyer band on the block's grid, FFT) then sincInterpolate (Kaiser-windowed sinc, half width H samples) at the frames;
//   NEW  wt(bank, member, window, instants): the member's bins summed at the instants directly (exact, no interpolation).
// The block is 2^15 samples of 4 channels (the chunk length), repeated; best of 5. Prints the times and the largest difference between the two (the sinc's own error).
#include "nxr/signal/resample.h"
#include "nxr/signal/wavelet.h"
#include <chrono>
#include <iostream>
#include <random>
using namespace nxr::signal; namespace tb = nxr::signal::timefilterbank;
int main() {
    const double fs = 1200, f0 = 10; const int C = 4, n = 1 << 15; const double hop = fs / f0 / 36;
    std::mt19937 g(1); std::normal_distribution<double> N;
    Eigen::MatrixXd x(C, n);
    for (int c = 0; c < C; ++c) for (int i = 0; i < n; ++i) x(c, i) = std::cos(2 * M_PI * f0 * i / fs + c) + 0.5 * N(g);
    tb::Design d(tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.bands = {{8, 12}}; d.edgeBins = tb::EdgeBins::Drop;
    const tb::TimeFilterbank bank = tb::design(d, n, fs);
    Eigen::VectorXd at; { std::vector<double> v; for (double t = 64; t < n - 64; t += hop) v.push_back(t); at = Eigen::Map<Eigen::VectorXd>(v.data(), (long)v.size()); }
    std::vector<double> pos(at.data(), at.data() + at.size());
    std::cout << "member support: bins " << bank.support[0].first << " .. " << bank.support[0].second << " of " << n / 2 + 1 << "\n";
    using clk = std::chrono::steady_clock;
    for (int H : {32, 64}) {
        double tOld = 1e9, tNew = 1e9; Eigen::MatrixXcd A, B;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = clk::now();
            const auto W = wt(bank, Eigen::MatrixXd(x.transpose()));                           // the analytic band of the block [n x C]
            std::vector<float> re(C * n), im(C * n);
            for (int c = 0; c < C; ++c) for (int i = 0; i < n; ++i) { re[c * n + i] = (float)W[0](i, c).real(); im[c * n + i] = (float)W[0](i, c).imag(); }
            const auto yr = sincInterpolate(re.data(), C, n, pos, H), yi = sincInterpolate(im.data(), C, n, pos, H);
            auto t1 = clk::now();
            tOld = std::min(tOld, std::chrono::duration<double, std::milli>(t1 - t0).count());
            A.resize(C, at.size()); for (int c = 0; c < C; ++c) for (Eigen::Index j = 0; j < at.size(); ++j) A(c, j) = {yr[c * at.size() + j], yi[c * at.size() + j]};
            t0 = clk::now(); B = wt(bank, 0, x, at); t1 = clk::now();
            tNew = std::min(tNew, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::cout << "sinc half width " << H << ": old (band analytic + sinc) " << tOld << " ms, new (wt at the instants) " << tNew << " ms, " << at.size() << " frames x " << C << " channels; max |old - new| = "
                  << (A - B).cwiseAbs().maxCoeff() << " (signal peak " << B.cwiseAbs().maxCoeff() << ")\n";
    }
}

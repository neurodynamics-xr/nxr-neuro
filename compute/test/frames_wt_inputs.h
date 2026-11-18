// The synthetic input of test_analytic_frames_step: one deterministic synthetic
// recording (sums of sines at fixed frequencies and phases, no random generator, so every platform has the same numbers), the bands and the frame instants.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
namespace fwi {
constexpr double kFs = 1200.0;
constexpr int kN = 4096, kC = 3;
constexpr double kHop = kFs / (10.954451150103322 * 36.0);      // 36 frames to a cycle of the 8-12 Hz band's centre (the geometric mean of its ends), in samples
inline Eigen::MatrixXd recording() {         // [channels x samples]
    Eigen::MatrixXd x(kC, kN);
    for (int c = 0; c < kC; ++c)
        for (int i = 0; i < kN; ++i) {
            double v = 0;
            for (int k = 0; k < 14; ++k) v += std::cos(2 * M_PI * (3.1 + 1.7 * k + 0.13 * c) * i / kFs + 0.7 * k + 1.1 * c) / (1.0 + 0.1 * k);
            for (int k = 0; k < 6; ++k) v += 0.3 * std::sin(2 * M_PI * (23.3 + 17.9 * k) * i / kFs + 0.4 * k * c);
            x(c, i) = v;
        }
    return x;
}
inline std::vector<double> instants() {       // every 7th frame, far enough from the ends that the taps' reflection does not reach them
    std::vector<double> t;
    for (int j = 0;; j += 7) { const double p = 400.0 + j * kHop; if (p > kN - 400.0) break; t.push_back(p); }
    return t;
}
}  // namespace fwi

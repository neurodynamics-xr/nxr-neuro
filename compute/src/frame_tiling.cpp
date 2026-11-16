#include "nxr/frame_tiling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::frametiling {

namespace {
double besselI0(double x) {
    double sum = 1, term = 1;
    for (int k = 1; k < 60; ++k) { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; if (term < 1e-17 * sum) break; }
    return sum;
}
}  // namespace

std::vector<float> sincInterpolate(const float* x, std::int64_t rows, std::int64_t width, const std::vector<double>& positions, int half_width, double kaiser_beta) {
    if (half_width < 1) throw std::invalid_argument("frames: half_width is at least 1 sample");
    const std::int64_t m = static_cast<std::int64_t>(positions.size());
    std::vector<float> out(static_cast<std::size_t>(rows * m), 0.0f);
    const double norm = besselI0(kaiser_beta);
    std::vector<double> w(static_cast<std::size_t>(2 * half_width));
    for (std::int64_t k = 0; k < m; ++k) {
        const double p = positions[static_cast<std::size_t>(k)];
        const std::int64_t i0 = static_cast<std::int64_t>(std::floor(p)) - half_width + 1;          // taps i0 … i0 + 2 half_width − 1 around p
        for (int a = 0; a < 2 * half_width; ++a) {
            const double u = static_cast<double>(i0 + a) - p;
            const double r = u / half_width;
            const double win = std::fabs(r) < 1 ? besselI0(kaiser_beta * std::sqrt(1 - r * r)) / norm : 0.0;
            const double s = std::fabs(u) < 1e-12 ? 1.0 : std::sin(M_PI * u) / (M_PI * u);
            w[static_cast<std::size_t>(a)] = s * win;
        }
        for (std::int64_t r = 0; r < rows; ++r) {
            double acc = 0;
            for (int a = 0; a < 2 * half_width; ++a) {
                const std::int64_t i = i0 + a;
                if (i < 0 || i >= width) continue;
                acc += w[static_cast<std::size_t>(a)] * static_cast<double>(x[static_cast<std::size_t>(r * width + i)]);
            }
            out[static_cast<std::size_t>(r * m + k)] = static_cast<float>(acc);
        }
    }
    return out;
}

}  // namespace nxr::frametiling

// nxr::taps — zero-phase taps and the row convolution (taps.h): nxr's filters/taps.py, line for line.
#include "nxr/signal/filters_fir.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <stdexcept>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
// numpy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src, pairwise_sum): fewer than 8 values summed in order; up to 128 summed into 8
// partial sums ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), the rest added in order; more split in two at a multiple of 8, each half recursively
double pairwiseSum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}
}  // namespace

std::vector<float> gainTaps(const std::function<double(double)>& gain, double fd, bool odd) {
    using cd = std::complex<double>;
    const std::size_t half = kGrid / 2;
    std::vector<double> G(half + 1);
    for (std::size_t j = 0; j <= half; ++j) G[j] = gain(static_cast<double>(j) * fd / static_cast<double>(kGrid));
    std::vector<cd> spec(kGrid, cd(0, 0)), out(kGrid);
    if (!odd) {
        for (std::size_t j = 0; j <= half; ++j) spec[j] = G[j];
        for (std::size_t j = 1; j < half; ++j) spec[kGrid - j] = G[j];
    } else {
        for (std::size_t j = 1; j < half; ++j) { spec[j] = cd(0, -G[j]); spec[kGrid - j] = cd(0, G[j]); }
    }
    c2c(spec.data(), out.data(), {kGrid}, 0, false);          // numpy's ifft (Backward: 1/N on the inverse)
    std::vector<double> h(kMaxTaps + 1);
    for (std::size_t k = 0; k <= kMaxTaps; ++k) h[k] = out[k].real();
    std::vector<double> sq(kMaxTaps);
    for (std::size_t k = 1; k <= kMaxTaps; ++k) sq[k - 1] = h[k] * h[k];
    const double total = h[0] * h[0] + 2 * pairwiseSum(sq.data(), sq.size());   // numpy: h[0]**2 + 2*float(np.sum(h[1:]**2))
    std::size_t K = kMaxTaps;
    double tail = 0;
    while (K > 0 && tail + 2 * (h[K] * h[K]) <= kTailEnergy * total) { tail += 2 * (h[K] * h[K]); --K; }   // numpy: 2 * h[K] ** 2
    if (!odd) {                                                          // the DC after truncation: Σ h = G(0) exactly
        h[0] += G[0] - (h[0] + 2 * pairwiseSum(h.data() + 1, K));        // numpy: np.sum(h[1:K + 1])
    }
    std::vector<float> t(K + 1);
    for (std::size_t k = 0; k <= K; ++k) t[k] = static_cast<float>(h[k]);
    return t;
}

std::vector<float> convolveRows(const float* x, std::size_t C, std::size_t Wr, const std::vector<float>& taps, std::size_t left, std::size_t width, bool odd) {
    if (Wr == 0 || left + width > Wr) throw std::invalid_argument("convolveRows: the columns [left, left + width) are not inside the block");
    if (taps.empty()) throw std::invalid_argument("convolveRows: no taps");
    const long last = static_cast<long>(Wr) - 1;
    auto at = [last](long t) {                                           // an index past either end reflected evenly about it (nxr's `at`)
        long i = std::labs(t);
        if (i > last) i = 2 * last - i;
        return std::min(std::max(i, 0L), last);
    };
    const std::size_t K = taps.size() - 1;
    std::vector<float> y(C * width);
    for (std::size_t c = 0; c < C; ++c) {
        const float* r = x + c * Wr;
        for (std::size_t t = 0; t < width; ++t) {
            const long u = static_cast<long>(left + t);
            double acc = static_cast<double>(taps[0]) * static_cast<double>(r[u]);
            for (std::size_t k = 1; k <= K; ++k) {
                const float a = r[at(u - static_cast<long>(k))], b = r[at(u + static_cast<long>(k))];
                // numpy: (a ± b) in float32, times the float32 tap, added to the float64 accumulator
                const float s = odd ? a - b : a + b;
                acc += static_cast<double>(taps[k] * s);
            }
            y[c * width + t] = static_cast<float>(acc);
        }
    }
    return y;
}

}  // namespace nxr::signal

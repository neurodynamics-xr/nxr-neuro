// The frames step through wt: the band's analytic signal read at the frame instants by wt(bank, member, window, instants), against the OLD path recorded in golden/frames_wt_golden.txt
// (gen_frames_wt_golden.cpp: the band's FIR taps, Morlet and Meyer, then the Kaiser sinc of half width 32 at the same instants, on the same synthetic recording).
// Measured (relative to the peak): Morlet 1.9e-3 (the taps' truncated tails and the sinc), Meyer 2.0e-2 (its sharp edges cut by the taps). By decision (2026-11-18) the frame values CHANGE: from the taps' truncated filter and the sinc approximation to the exact analytic signal of the band. The test holds the difference to what was measured.
#include "frames_wt_inputs.h"
#include "mesh_helpers.h"
#include "nxr/signal/wavelet.h"
#include <fstream>
#include <sstream>
namespace tb = nxr::signal::timefilterbank;
int main() {
    std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/frames_wt_golden.txt");
    EXPECT(in.good(), "the golden is there");
    const Eigen::MatrixXd x = fwi::recording();
    const std::vector<double> pos = fwi::instants();
    const Eigen::VectorXd at = Eigen::Map<const Eigen::VectorXd>(pos.data(), static_cast<long>(pos.size()));
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line); std::string shape; std::size_t n; s >> shape >> n;
        std::vector<double> v(n); for (auto& e : v) s >> e;
        tb::Design d(shape == "morlet" ? tb::Family::Morlet : tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.bands = {{8.0, 12.0}}; d.edgeBins = tb::EdgeBins::Drop;
        const tb::TimeFilterbank bank = tb::design(d, static_cast<std::size_t>(fwi::kN), fwi::kFs);
        const Eigen::MatrixXcd a = nxr::signal::wt(bank, 0, x, at);      // [channels x instants]
        double e = 0, peak = 0;
        const std::size_t J = pos.size();
        for (int c = 0; c < fwi::kC; ++c)
            for (std::size_t j = 0; j < J; ++j) {
                const std::size_t k = (static_cast<std::size_t>(c) * J + j) * 2;
                e = std::max(e, std::abs(a(c, static_cast<long>(j)) - std::complex<double>(v[k], v[k + 1])));
                peak = std::max(peak, std::abs(a(c, static_cast<long>(j))));
            }
        std::cout << "  " << shape << ": max |new - old| = " << e << " (peak " << peak << ", relative " << e / peak << ")\n";
        EXPECT(e < (shape == "morlet" ? 3e-3 : 3e-2) * peak, shape << ": the exact analytic signal differs from the old taps + sinc by " << e / peak << " of the peak");
    }
    std::cout << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}

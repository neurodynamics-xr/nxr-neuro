// wt at frame instants: one band's analytic signal of a 2^15-sample chunk (64 channels) read at 512 fractional instants. Synthetic data, fixed seed.
#include "bench.h"
#include "nxr/signal/wavelet.h"
#include <random>

int main() {
  namespace tb = nxr::signal::timefilterbank;
  const double fs = 1000;
  const int n = 1 << 15, channels = 64, instants = 512;
  tb::Design d(tb::Family::Meyer, 2.0, tb::Sides::Analytic);
  d.bands = {{4, 30}};
  d.edgeBins = tb::EdgeBins::Drop;
  const tb::TimeFilterbank bank = tb::design(d, n, fs);
  std::mt19937 g(1);
  std::normal_distribution<double> N;
  Eigen::MatrixXd x(channels, n);
  for (int c = 0; c < channels; ++c) for (int i = 0; i < n; ++i) x(c, i) = N(g);
  Eigen::VectorXd at(instants);
  for (int j = 0; j < instants; ++j) at(j) = 100.25 + j * (n - 300.0) / instants;
  nxr_bench::run("wt_at_frame_instants", "chunk=32768 channels=64 band=4-30Hz instants=512", 5, [&] {
    const Eigen::MatrixXcd a = nxr::signal::wt(bank, 0, x, at);
    nxr_bench::keep(std::abs(a(0, 0)));
  });
  return 0;
}

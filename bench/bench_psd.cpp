// psd of a 64-channel chunk (2^15 samples at 1000 Hz), Brainstorm's default options. Synthetic data, fixed seed.
#include "bench.h"
#include "nxr/signal/spectral.h"
#include <random>

int main() {
  const double fs = 1000;
  const int n = 1 << 15, channels = 64;
  std::mt19937 g(2);
  std::normal_distribution<double> N;
  Eigen::MatrixXd x(channels, n);
  for (int c = 0; c < channels; ++c) for (int i = 0; i < n; ++i) x(c, i) = N(g);
  const auto options = nxr::signal::spectral::brainstormPsdOptions(fs);
  nxr_bench::run("psd_chunk", "chunk=32768 channels=64 fs=1000 options=brainstorm", 5, [&] {
    const auto r = nxr::signal::spectral::psd(x, options);
    nxr_bench::keep(r.psd(0, 0));
  });
  return 0;
}

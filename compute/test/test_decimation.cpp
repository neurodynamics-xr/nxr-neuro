// nxr::signal::decimate on synthetic signals: the decimated blocks of a chunked signal, each read with its halo, laid end to end equal the whole signal decimated (any
// block size, float64 and float32); a band below the new Nyquist frequency passes with no delay (zero phase); a band above it is removed (no alias); q = 1 is the identity;
// the refusals.
#include "nxr/signal/decimation.h"
#include "mesh_helpers.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using nxr::signal::decimate;
using nxr::signal::decimationHalo;

static constexpr double kPi = 3.14159265358979323846;
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

// two channels at fs: a 5 Hz sinusoid plus a 200 Hz one (above the new Nyquist frequency at q = 4), and a 12 Hz cosine
static MatrixXd signal(double fs, Index n) {
  MatrixXd x(2, n);
  for (Index t = 0; t < n; ++t) {
    const double s = static_cast<double>(t) / fs;
    x(0, t) = std::sin(2 * kPi * 5.0 * s) + 0.5 * std::sin(2 * kPi * 200.0 * s);
    x(1, t) = std::cos(2 * kPi * 12.0 * s);
  }
  return x;
}

// the signal cut into blocks of `width` samples (the last partial), each handed over with `halo` samples of context on each side (clipped at the ends), decimated and joined
template <class M>
static M chunked(const M& x, int q, Index width, Index halo) {
  const Index n = x.cols();
  std::vector<M> parts;
  Index total = 0;
  for (Index first = 0; first < n; first += width) {
    const Index w = std::min(width, n - first), a = std::max<Index>(0, first - halo), b = std::min(n, first + w + halo);
    const M block = x.middleCols(a, b - a);
    const Index offset = (q - first % q) % q;
    parts.push_back(decimate(block, q, first - a, w, offset));
    total += parts.back().cols();
  }
  M out(x.rows(), total);
  Index at = 0;
  for (const M& p : parts) { out.middleCols(at, p.cols()) = p; at += p.cols(); }
  return out;
}

int main() {
  const double fs = 1200.0;
  const int q = 4;
  const Index n = 6000;
  const MatrixXd x = signal(fs, n);
  const MatrixXd whole = decimate(x, q);
  EXPECT(whole.rows() == 2 && whole.cols() == (n + q - 1) / q, "the whole signal decimated: [channels x ceil(n / q)]");

  std::cout << "=== chunk-size independence ===\n";
  const Index halo = decimationHalo(q);
  EXPECT(halo == 40, "the halo at q = 4 is the order's half (20 q / 2)");
  for (const Index width : {Index(997), Index(1024), Index(1500), Index(4096)}) {
    const MatrixXd c = chunked(x, q, width, halo);
    EXPECT(c.cols() == whole.cols() && (c - whole).cwiseAbs().maxCoeff() < 1e-12, "blocks of " << width << " samples with the halo give the whole signal's output");
  }
  const MatrixXf xf = x.cast<float>();
  const MatrixXf wf = decimate(xf, q);
  const MatrixXf cf = chunked(xf, q, 1001, halo);
  EXPECT(cf.cols() == wf.cols() && (cf - wf).cwiseAbs().maxCoeff() < 1e-6f, "float32: the same, in float32");
  EXPECT((wf.cast<double>() - whole).cwiseAbs().maxCoeff() < 1e-5, "float32 and float64 agree to float32's rounding");

  std::cout << "=== the response ===\n";
  // away from the ends (the filter's half-length), the output is the low band alone, at the kept instants: no delay, no alias of 200 Hz
  double err0 = 0, err1 = 0;
  for (Index k = halo; k < whole.cols() - halo; ++k) {
    const double s = static_cast<double>(k * q) / fs;
    err0 = std::max(err0, std::abs(whole(0, k) - std::sin(2 * kPi * 5.0 * s)));
    err1 = std::max(err1, std::abs(whole(1, k) - std::cos(2 * kPi * 12.0 * s)));
  }
  EXPECT(err0 < 5e-3, "5 Hz passes, 200 Hz (above the new Nyquist frequency) is removed, with no delay (max error " << err0 << ")");
  EXPECT(err1 < 5e-3, "12 Hz passes with zero phase (max error " << err1 << ")");

  std::cout << "=== edges and refusals ===\n";
  EXPECT((decimate(x, 1, 10, 50, 0) - x.middleCols(10, 50)).cwiseAbs().maxCoeff() == 0.0, "q = 1 keeps the own columns as they are");
  EXPECT(decimate(x, q, 0, 2, 3).cols() == 0, "an offset past the own columns keeps none");
  EXPECT(throws([&] { decimate(x, 0); }), "q = 0 is refused");
  EXPECT(throws([&] { decimate(x, q, 0, -1, q); }), "an offset of q or more is refused");
  EXPECT(throws([&] { decimate(x, q, n - 5, 10); }), "own columns past the block's end are refused");
  EXPECT(throws([&] { decimate(x, q, 0, -1, 0, 7); }), "an odd order is refused");

  std::cout << (g_failures ? "FAILED\n" : "all passed\n");
  return g_failures ? 1 : 0;
}

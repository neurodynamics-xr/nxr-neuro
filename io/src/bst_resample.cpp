// nxr::bst — scipy.signal.resample_poly (and its helpers firwin, the Kaiser window, upfirdn) in float64, and the stored-recording resample of nxr.convert.timeseries.
#include <algorithm>
#include <cmath>
#include <numeric>

#include "np.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/session.h"
#include "nxr/io/layout.h"

#ifndef NXR_FMA_I0
#define NXR_FMA_I0 0
#endif
#ifndef NXR_FMA_ACC
#define NXR_FMA_ACC 0
#endif

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

// Cephes' I0 (scipy.special.i0 / numpy.i0): Chebyshev series in exp(x) * ...
constexpr double kI0A[30] = {
  -4.4153416464793395e-18,
  3.3307945188222384e-17,
  -2.431279846547955e-16,
  1.715391285555133e-15,
  -1.1685332877993451e-14,
  7.676185498604936e-14,
  -4.856446783111929e-13,
  2.95505266312964e-12,
  -1.726826291441556e-11,
  9.675809035373237e-11,
  -5.189795601635263e-10,
  2.6598237246823866e-09,
  -1.300025009986248e-08,
  6.046995022541919e-08,
  -2.670793853940612e-07,
  1.1173875391201037e-06,
  -4.4167383584587505e-06,
  1.6448448070728896e-05,
  -5.754195010082104e-05,
  0.00018850288509584165,
  -0.0005763755745385824,
  0.0016394756169413357,
  -0.004324309995050576,
  0.010546460394594998,
  -0.02373741480589947,
  0.04930528423967071,
  -0.09490109704804764,
  0.17162090152220877,
  -0.3046826723431984,
  0.6767952744094761};
constexpr double kI0B[25] = {
  -7.233180487874754e-18,
  -4.830504485944182e-18,
  4.46562142029676e-17,
  3.461222867697461e-17,
  -2.8276239805165836e-16,
  -3.425485619677219e-16,
  1.7725601330565263e-15,
  3.8116806693526224e-15,
  -9.554846698828307e-15,
  -4.150569347287222e-14,
  1.54008621752141e-14,
  3.8527783827421426e-13,
  7.180124451383666e-13,
  -1.7941785315068062e-12,
  -1.3215811840447713e-11,
  -3.1499165279632416e-11,
  1.1889147107846439e-11,
  4.94060238822497e-10,
  3.3962320257083865e-09,
  2.266668990498178e-08,
  2.0489185894690638e-07,
  2.8913705208347567e-06,
  6.889758346916825e-05,
  0.0033691164782556943,
  0.8044904110141088};

double chbevl(double x, const double* v, int n) {
  double b0 = v[0], b1 = 0.0, b2 = 0.0;
  for (int i = 1; i < n; ++i) {
    b2 = b1;
    b1 = b0;
#if NXR_FMA_I0
    b0 = std::fma(x, b1, -b2) + v[i];
#else
    b0 = x * b1 - b2 + v[i];
#endif
  }
  return 0.5 * (b0 - b2);
}
double i0(double x) {
  x = std::fabs(x);
  if (x <= 8.0) return std::exp(x) * chbevl(x / 2.0 - 2.0, kI0A, 30);
  return std::exp(x) * chbevl(32.0 / x - 2.0, kI0B, 25) / std::sqrt(x);
}

// scipy.signal.windows.kaiser(M, beta, sym=True)
std::vector<double> kaiser(std::int64_t M, double beta) {
  std::vector<double> w(static_cast<std::size_t>(M));
  const double alpha = (static_cast<double>(M) - 1) / 2.0;
  const double d = i0(beta);
  for (std::int64_t n = 0; n < M; ++n) {
    const double r = (static_cast<double>(n) - alpha) / alpha;
    w[static_cast<std::size_t>(n)] = i0(beta * std::sqrt(1 - r * r)) / d;
  }
  return w;
}

double np_sinc(double x) {
  const double y = M_PI * (x == 0 ? 1.0e-20 : x);
  return std::sin(y) / y;
}

// scipy.signal.firwin(numtaps, cutoff, window=('kaiser', 5.0)) for a lowpass with cutoff relative to Nyquist
std::vector<double> firwin_lowpass(std::int64_t numtaps, double cutoff) {
  const double alpha = 0.5 * static_cast<double>(numtaps - 1);
  std::vector<double> h(static_cast<std::size_t>(numtaps));
  const std::vector<double> win = kaiser(numtaps, 5.0);
  for (std::int64_t i = 0; i < numtaps; ++i) {
    const double m = static_cast<double>(i) - alpha;
    double v = 0.0;
    v += cutoff * np_sinc(cutoff * m);
    v -= 0.0 * np_sinc(0.0 * m);
    h[static_cast<std::size_t>(i)] = v * win[static_cast<std::size_t>(i)];
  }
  // scale to unit gain at DC: h /= sum(h * cos(pi * m * 0)) = sum(h)
  const double s = np_sum(h.data(), h.size());
  for (auto& x : h) x /= s;
  return h;
}

// y = resample_poly(x, up, down) of one row (padtype 'constant', zero padded): scipy's design, polyphase upfirdn, the accumulation order of its C loop
void resample_poly_row(const double* x, std::int64_t n_in, std::int64_t up, std::int64_t down, const std::vector<double>& h0, std::int64_t half_len, double* out,
                       std::int64_t n_out) {
  std::vector<double> h = h0;
  for (auto& v : h) v *= static_cast<double>(up);
  const std::int64_t n_pre_pad = down - half_len % down;
  std::int64_t n_post_pad = 0;
  const std::int64_t n_pre_remove = (half_len + n_pre_pad) / down;
  auto output_len = [&](std::int64_t len_h) { return ((n_in - 1) * up + len_h - 1) / down + 1; };
  while (output_len(static_cast<std::int64_t>(h.size()) + n_pre_pad + n_post_pad) < n_out + n_pre_remove) ++n_post_pad;
  std::vector<double> hp(static_cast<std::size_t>(n_pre_pad), 0.0);
  hp.insert(hp.end(), h.begin(), h.end());
  hp.insert(hp.end(), static_cast<std::size_t>(n_post_pad), 0.0);
  while (hp.size() % static_cast<std::size_t>(up)) hp.push_back(0.0);   // upfirdn pads the filter to a multiple of `up`
  const std::int64_t hpp = static_cast<std::int64_t>(hp.size()) / up;   // taps per phase
  for (std::int64_t k = 0; k < n_out; ++k) {
    const std::int64_t y = n_pre_remove + k;
    const std::int64_t t = (y * down) % up, xi = (y * down) / up;
    double acc = 0.0;
    for (std::int64_t xc = std::max<std::int64_t>(0, xi - hpp + 1); xc <= std::min(xi, n_in - 1); ++xc) {
      const double hv = hp[static_cast<std::size_t>(t + (xi - xc) * up)];
#if NXR_FMA_ACC
      acc = std::fma(x[xc], hv, acc);
#else
      acc += x[xc] * hv;
#endif
    }
    out[k] = acc;
  }
}

}  // namespace

std::vector<float> resample_rows(const std::function<std::vector<double>(std::int64_t, std::int64_t)>& read, std::int64_t n_chan, std::int64_t n_t, double fs, double to,
                                 std::int64_t& n_new, std::int64_t step) {
  // Fraction(round(to * 1000), round(fs * 1000))
  std::int64_t num = static_cast<std::int64_t>(std::nearbyint(to * 1000)), den = static_cast<std::int64_t>(std::nearbyint(fs * 1000));
  const std::int64_t g = std::gcd(num, den);
  num /= g;
  den /= g;
  if (den > 1000 || num > 1000)
    value_error("cannot resample " + float_repr(fs) + " Hz to " + float_repr(to) + " Hz: the ratio " + std::to_string(num) + (den == 1 ? "" : "/" + std::to_string(den)) +
                " is not a small fraction");
  n_new = -((-n_t * num) / den);   // ceil(n_t * num / den)
  std::vector<float> out(static_cast<std::size_t>(n_chan * n_new));
  const std::int64_t max_rate = std::max(num, den);
  const std::int64_t half_len = 10 * max_rate;
  const std::vector<double> h = firwin_lowpass(2 * half_len + 1, 1.0 / static_cast<double>(max_rate));
  std::vector<double> row(static_cast<std::size_t>(n_new));
  for (std::int64_t c0 = 0; c0 < n_chan; c0 += step) {
    const std::int64_t c1 = std::min(c0 + step, n_chan);
    const std::vector<double> x = read(c0, c1);
    for (std::int64_t c = c0; c < c1; ++c) {
      const double* xr = x.data() + (c - c0) * n_t;
      if (num == 1 && den == 1) std::copy(xr, xr + n_t, row.begin());
      else resample_poly_row(xr, n_t, num, den, h, half_len, row.data(), n_new);
      for (std::int64_t k = 0; k < n_new; ++k) out[static_cast<std::size_t>(c * n_new + k)] = static_cast<float>(row[static_cast<std::size_t>(k)]);
    }
  }
  return out;
}

}  // namespace nxr::bst

// nxr::signal — one Gabor atom fitted to a burst (gabor.h). The maths of the lab's staging detector (gabor.py: direct, refine, is_burst; spacetime.py: fit_r2), written here.
#include "nxr/signal/gabor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

namespace {
constexpr double kPi = 3.14159265358979323846;
const double kC4 = std::pow(kPi, 0.25);
constexpr double kFwhm = 2.355;                        // the reference's FWHM / σ (2 √(2 ln 2) = 2.3548)
}  // namespace

std::complex<double> gaborCoefficient(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, double t0, double f, double sigma) {
  if (!(fs > 0) || !(sigma > 0)) throw std::invalid_argument("gaborCoefficient: fs > 0 and σ > 0");
  const Eigen::Index n = x.size();
  // the reference's int(): truncation toward zero
  const Eigen::Index i0 = std::max<Eigen::Index>(0, static_cast<Eigen::Index>(std::trunc((t0 - 5 * sigma) * fs)));
  const Eigen::Index i1 = std::min<Eigen::Index>(n, static_cast<Eigen::Index>(std::trunc((t0 + 5 * sigma) * fs)) + 1);
  const double norm = std::pow(kPi * sigma * sigma, -0.25);
  double re = 0, im = 0;
  for (Eigen::Index i = i0; i < i1; ++i) {
    const double tt = static_cast<double>(i) / fs - t0, g = x(i) * norm * std::exp(-tt * tt / (2 * sigma * sigma)), ph = -2 * kPi * f * tt;
    re += g * std::cos(ph);
    im += g * std::sin(ph);
  }
  return {re / fs, im / fs};
}

GaborAtom gaborFit(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, double t0, double f, double sigma, int steps) {
  if (steps < 0) throw std::invalid_argument("gaborFit: steps is at least 0");
  double s = sigma;
  for (int k = 0; k < steps; ++k)
    for (int ax = 0; ax < 2; ++ax) {
      const double d = ax == 0 ? 1 / (4 * kPi * s) : std::log(2.0) / 8;
      double v[3];
      for (int o = -1; o <= 1; ++o)
        v[o + 1] = std::log(std::abs(ax == 0 ? gaborCoefficient(x, fs, t0, f + o * d, s) : gaborCoefficient(x, fs, t0, f, s * std::exp(o * d))));
      const double den = v[0] - 2 * v[1] + v[2];
      const double o = den < 0 ? std::clamp(0.5 * (v[0] - v[2]) / den, -1.0, 1.0) : 0.0;
      if (ax == 0) f += o * d; else s *= std::exp(o * d);
    }
  const std::complex<double> W = gaborCoefficient(x, fs, t0, f, s);
  GaborAtom a;
  a.t0 = t0; a.f = f; a.sigma = s; a.fwhm = kFwhm * s; a.cycles = kFwhm * s * f;
  a.coefficient = std::abs(W);
  a.amplitude = 2 * a.coefficient / (kC4 * std::sqrt(s));
  a.phase = std::arg(W);
  return a;
}

bool gaborWidthPeak(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, const GaborAtom& a, double sigmaLow, double sigmaHigh, double step) {
  if (!(step > 1)) throw std::invalid_argument("gaborWidthPeak: the ladder's step is above 1");
  if (!(sigmaLow * std::sqrt(step) < a.sigma && a.sigma < sigmaHigh / std::sqrt(step))) return false;
  for (const double m : {step, 1 / step})
    if (std::abs(gaborCoefficient(x, fs, a.t0, a.f, a.sigma * m)) > a.coefficient) return false;
  return true;
}

double gaborR2(const Eigen::Ref<const Eigen::VectorXd>& y, double fs, const GaborAtom& a) {
  const Eigen::Index T = y.size();
  const Eigen::Index i0 = std::max<Eigen::Index>(0, static_cast<Eigen::Index>(std::ceil((a.t0 - a.fwhm / 2) * fs)));
  const Eigen::Index i1 = std::min<Eigen::Index>(T, static_cast<Eigen::Index>(std::trunc((a.t0 + a.fwhm / 2) * fs)) + 1);
  double res = 0, tot = 0;
  for (Eigen::Index i = i0; i < i1; ++i) {
    const double tl = static_cast<double>(i) / fs - a.t0;
    const double fit = a.amplitude * std::exp(-tl * tl / (2 * a.sigma * a.sigma)) * std::cos(2 * kPi * a.f * tl + a.phase);
    res += (y(i) - fit) * (y(i) - fit);
    tot += y(i) * y(i);
  }
  return tot > 0 ? 1 - res / tot : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace nxr::signal

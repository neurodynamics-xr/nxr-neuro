// fir1: scipy.signal.firwin's route (the ideal band-pass pieces as sinc differences, the window, the passband-reference scaling).
#include "nxr/signal/filters_fir_design.h"
#include "nxr/signal/spectral.h"   // windowOf

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace nxr::signal {
namespace {
double sinc(double x) { const double pi = std::acos(-1.0); return x == 0.0 ? 1.0 : std::sin(pi * x) / (pi * x); }
}

Eigen::VectorXd fir1(int order, const std::vector<double>& Wn, Band type, const Eigen::VectorXd& window, bool scale) {
  if (order < 1) throw std::invalid_argument("fir1: the order must be at least 1, got " + std::to_string(order));
  const std::size_t want = (type == Band::Bandpass || type == Band::Bandstop) ? 2 : 1;
  if (Wn.size() != want) throw std::invalid_argument("fir1: this type takes " + std::to_string(want) + " edge(s) in units of Nyquist, got " + std::to_string(Wn.size()));
  for (double w : Wn) if (!(w > 0.0 && w < 1.0)) throw std::invalid_argument("fir1: an edge must lie strictly between 0 and 1 (units of Nyquist), got " + std::to_string(w));
  if (want == 2 && !(Wn[0] < Wn[1])) throw std::invalid_argument("fir1: the two edges must be ascending");
  const std::size_t taps = static_cast<std::size_t>(order) + 1;
  const bool passesNyquist = type == Band::Highpass || type == Band::Bandstop;
  if (passesNyquist && taps % 2 == 0) throw std::invalid_argument("fir1: a filter that passes the Nyquist frequency needs an even order (an odd number of taps), got order " + std::to_string(order));
  if (window.size() != 0 && static_cast<std::size_t>(window.size()) != taps) throw std::invalid_argument("fir1: the window must have order + 1 = " + std::to_string(taps) + " samples, got " + std::to_string(window.size()));

  // the passbands [left, right] in Nyquist units
  std::vector<std::pair<double, double>> bands;
  switch (type) {
    case Band::Lowpass: bands = {{0.0, Wn[0]}}; break;
    case Band::Highpass: bands = {{Wn[0], 1.0}}; break;
    case Band::Bandpass: bands = {{Wn[0], Wn[1]}}; break;
    case Band::Bandstop: bands = {{0.0, Wn[0]}, {Wn[1], 1.0}}; break;
  }
  const Eigen::VectorXd win = window.size() ? window : spectral::windowOf(taps, spectral::Window::Hamming, false);
  Eigen::VectorXd h = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(taps));
  const double alpha = 0.5 * (static_cast<double>(taps) - 1.0);
  for (std::size_t n = 0; n < taps; ++n) {
    const double m = static_cast<double>(n) - alpha;
    double v = 0.0;
    for (const auto& b : bands) v += b.second * sinc(b.second * m) - b.first * sinc(b.first * m);
    h(static_cast<Eigen::Index>(n)) = v * win(static_cast<Eigen::Index>(n));
  }
  if (scale) {
    const double pi = std::acos(-1.0);
    const double f = (type == Band::Lowpass || type == Band::Bandstop) ? 0.0 : type == Band::Highpass ? 1.0 : 0.5 * (Wn[0] + Wn[1]);
    double s = 0.0;
    for (std::size_t n = 0; n < taps; ++n) s += h(static_cast<Eigen::Index>(n)) * std::cos(pi * (static_cast<double>(n) - alpha) * f);
    h /= s;
  }
  return h;
}

}  // namespace nxr::signal

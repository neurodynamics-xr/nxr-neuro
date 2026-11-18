// butter: scipy.signal.butter's route (buttap, lp2xx_zpk on the pre-warped edges, bilinear_zpk, zpk2tf), digital, fs = 2 (edges in Nyquist units).
#include "nxr/signal/filters_iir.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

namespace nxr::signal {
namespace {
using cd = std::complex<double>;
using Roots = std::vector<cd>;

cd productOf(const Roots& v, cd shift_sign, cd offset) {   // prod(offset + shift_sign * v)
  cd p = 1.0;
  for (const cd& r : v) p *= offset + shift_sign * r;
  return p;
}
Roots poly(const Roots& roots) {   // coefficients of prod (x - r), descending powers (numpy.poly)
  Roots c{1.0};
  for (const cd& r : roots) {
    Roots n(c.size() + 1, 0.0);
    for (std::size_t i = 0; i < c.size(); ++i) { n[i] += c[i]; n[i + 1] -= r * c[i]; }
    c = n;
  }
  return c;
}
Eigen::VectorXd realPart(const Roots& c, double gain) {
  Eigen::VectorXd v(c.size());
  for (std::size_t i = 0; i < c.size(); ++i) v(static_cast<Eigen::Index>(i)) = gain * c[i].real();
  return v;
}
}  // namespace

TransferFunction butter(int order, const std::vector<double>& Wn, Band type) {
  if (order < 1) throw std::invalid_argument("butter: the order must be at least 1, got " + std::to_string(order));
  const std::size_t want = (type == Band::Bandpass || type == Band::Bandstop) ? 2 : 1;
  if (Wn.size() != want) throw std::invalid_argument("butter: this type takes " + std::to_string(want) + " edge(s) in units of Nyquist, got " + std::to_string(Wn.size()));
  for (double w : Wn) if (!(w > 0.0 && w < 1.0)) throw std::invalid_argument("butter: an edge must lie strictly between 0 and 1 (units of Nyquist), got " + std::to_string(w));
  if (want == 2 && !(Wn[0] < Wn[1])) throw std::invalid_argument("butter: the two edges must be ascending");

  const double fs = 2.0, pi = std::acos(-1.0);
  // the analogue prototype: poles on the left half of the unit circle, no zeros, gain 1
  Roots z, p;
  for (int m = -order + 1; m < order; m += 2) p.push_back(-std::exp(cd(0, pi * m / (2.0 * order))));
  double k = 1.0;
  const std::size_t proto = p.size();   // the prototype's degree N

  Roots wc;
  for (double w : Wn) wc.push_back(2.0 * fs * std::tan(pi * w / fs));   // the pre-warped edges, rad/s
  const double wo1 = wc[0].real();

  switch (type) {
    case Band::Lowpass:   // lp2lp_zpk: p *= wo; k *= wo^N
      for (cd& r : p) r *= wo1;
      k *= std::pow(wo1, static_cast<double>(proto));
      break;
    case Band::Highpass: {   // lp2hp_zpk: p = wo / p, a zero at 0 for each pole; k *= real(prod(-z) / prod(-p))
      k *= (productOf(z, -1.0, 0.0) / productOf(p, -1.0, 0.0)).real();
      for (cd& r : p) r = wo1 / r;
      z.assign(proto, cd(0, 0));
      break;
    }
    case Band::Bandpass: {   // lp2bp_zpk: each pole -> two, p +- sqrt(p^2 - wo^2), N zeros at 0
      const double bw = wc[1].real() - wc[0].real(), wo = std::sqrt(wc[0].real() * wc[1].real());
      Roots pb;
      for (const cd& r : p) { const cd pl = r * bw / 2.0, d = std::sqrt(pl * pl - wo * wo); pb.push_back(pl + d); pb.push_back(pl - d); }
      p = pb; z.assign(proto, cd(0, 0));
      k *= std::pow(bw, static_cast<double>(proto));
      break;
    }
    case Band::Bandstop: {   // lp2bs_zpk: p_hp = (bw/2) / p, each -> two; zeros at +- j wo (N each); k *= real(prod(-z) / prod(-p))
      const double bw = wc[1].real() - wc[0].real(), wo = std::sqrt(wc[0].real() * wc[1].real());
      k *= (productOf(z, -1.0, 0.0) / productOf(p, -1.0, 0.0)).real();
      Roots pb;
      for (const cd& r : p) { const cd ph = (bw / 2.0) / r, d = std::sqrt(ph * ph - wo * wo); pb.push_back(ph + d); pb.push_back(ph - d); }
      p = pb; z.clear();
      for (std::size_t i = 0; i < proto; ++i) { z.push_back(cd(0, wo)); z.push_back(cd(0, -wo)); }
      break;
    }
  }

  // bilinear_zpk with fs = 2: s -> 2 fs (z - 1) / (z + 1); the missing zeros go to z = -1
  const double fs2 = 2.0 * fs;
  const std::size_t degree = p.size() - z.size();
  const cd kz = k * productOf(z, -1.0, fs2) / productOf(p, -1.0, fs2);
  Roots zz, pz;
  for (const cd& r : z) zz.push_back((fs2 + r) / (fs2 - r));
  for (const cd& r : p) pz.push_back((fs2 + r) / (fs2 - r));
  for (std::size_t i = 0; i < degree; ++i) zz.push_back(cd(-1, 0));
  return {realPart(poly(zz), kz.real()), realPart(poly(pz), 1.0)};
}

}  // namespace nxr::signal

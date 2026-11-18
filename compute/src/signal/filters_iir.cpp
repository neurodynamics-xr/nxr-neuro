// butter: scipy.signal.butter's route (buttap, lp2xx_zpk on the pre-warped edges, bilinear_zpk, zpk2tf), digital, fs = 2 (edges in Nyquist units).
#include "nxr/signal/filters_iir.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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
// The square root of the band-pass shift: for a REAL prototype pole (odd order) scipy's p**2 carries a rounding-sized imaginary part of negative sign (numpy's complex power), which puts the root
// with the NEGATIVE imaginary part first; a complex pole's root is the principal one.
cd rootOf(cd v, const cd& prototypePole) {
  const cd d = std::sqrt(v);
  return prototypePole.imag() == 0.0 && v.imag() == 0.0 && d.imag() > 0.0 ? std::conj(d) : d;
}
// The band-stop shift's root: scipy's lp2bs puts the root with the POSITIVE imaginary part first for a real prototype pole (the sign of a zero imaginary part is not left to the rounding)
cd rootOfBandstop(cd v, const cd& prototypePole) {
  const cd d = std::sqrt(v);
  return prototypePole.imag() == 0.0 && v.imag() == 0.0 && d.imag() < 0.0 ? std::conj(d) : d;
}
Eigen::VectorXd realPart(const Roots& c, double gain) {
  Eigen::VectorXd v(c.size());
  for (std::size_t i = 0; i < c.size(); ++i) v(static_cast<Eigen::Index>(i)) = gain * c[i].real();
  return v;
}
}  // namespace

ZeroPoleGain butterZpk(int order, const std::vector<double>& Wn, Band type) {
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
      Roots pb, pm;   // scipy's order: the "+" roots of every pole, then the "-" roots
      for (const cd& r : p) { const cd pl = r * bw / 2.0, d = rootOf(pl * pl - wo * wo, r); pb.push_back(pl + d); pm.push_back(pl - d); }
      pb.insert(pb.end(), pm.begin(), pm.end());
      p = pb; z.assign(proto, cd(0, 0));
      k *= std::pow(bw, static_cast<double>(proto));
      break;
    }
    case Band::Bandstop: {   // lp2bs_zpk: p_hp = (bw/2) / p, each -> two; zeros at +- j wo (N each); k *= real(prod(-z) / prod(-p))
      const double bw = wc[1].real() - wc[0].real(), wo = std::sqrt(wc[0].real() * wc[1].real());
      k *= (productOf(z, -1.0, 0.0) / productOf(p, -1.0, 0.0)).real();
      Roots pb, pm;   // scipy's order: the "+" roots of every pole, then the "-" roots; the zeros at +j wo, then at -j wo
      for (const cd& r : p) { const cd ph = (bw / 2.0) / r, d = rootOfBandstop(ph * ph - wo * wo, r); pb.push_back(ph + d); pm.push_back(ph - d); }
      pb.insert(pb.end(), pm.begin(), pm.end());
      p = pb; z.clear();
      for (std::size_t i = 0; i < proto; ++i) z.push_back(cd(0, wo));
      for (std::size_t i = 0; i < proto; ++i) z.push_back(cd(0, -wo));
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
  return {zz, pz, kz.real()};
}

TransferFunction butter(int order, const std::vector<double>& Wn, Band type) {
  const ZeroPoleGain f = butterZpk(order, Wn, type);
  return {realPart(poly(f.z), f.k), realPart(poly(f.p), 1.0)};
}

namespace {
// scipy.signal._cplxreal: the complex roots as ONE of each conjugate pair (positive imaginary part, sorted by real part, then by imaginary part), and the real roots sorted
void cplxreal(const Roots& in, Roots& pairs, std::vector<double>& reals) {
  const double tol = 100.0 * std::numeric_limits<double>::epsilon();
  Roots z = in;
  std::stable_sort(z.begin(), z.end(), [](const cd& a, const cd& b) { return a.real() != b.real() ? a.real() < b.real() : std::abs(a.imag()) < std::abs(b.imag()); });
  Roots zp, zn;
  for (const cd& v : z) {
    if (std::abs(v.imag()) <= tol * std::abs(v)) reals.push_back(v.real());
    else if (v.imag() > 0) zp.push_back(v);
    else zn.push_back(v);
  }
  if (zp.size() != zn.size()) throw std::invalid_argument("zpkToSos: a complex root has no matching conjugate");
  // each run of (nearly) the same real part is ordered by the imaginary part, as the pairs' two halves must match
  for (std::size_t i = 0; i < zp.size();) {
    std::size_t j = i + 1;
    while (j < zp.size() && zp[j].real() - zp[j - 1].real() <= tol * std::abs(zp[j - 1])) ++j;
    auto byImag = [](const cd& a, const cd& b) { return std::abs(a.imag()) < std::abs(b.imag()); };
    std::stable_sort(zp.begin() + static_cast<std::ptrdiff_t>(i), zp.begin() + static_cast<std::ptrdiff_t>(j), byImag);
    std::stable_sort(zn.begin() + static_cast<std::ptrdiff_t>(i), zn.begin() + static_cast<std::ptrdiff_t>(j), byImag);
    i = j;
  }
  for (std::size_t i = 0; i < zp.size(); ++i) {
    if (std::abs(zp[i] - std::conj(zn[i])) > tol * std::abs(zn[i])) throw std::invalid_argument("zpkToSos: a complex root has no matching conjugate");
    pairs.push_back((zp[i] + std::conj(zn[i])) / 2.0);
  }
}
bool isReal(const cd& v) { return v.imag() == 0.0; }
std::size_t countReal(const Roots& v) { std::size_t n = 0; for (const cd& r : v) n += isReal(r); return n; }
std::size_t nearest(const Roots& from, const cd& to, int which) {   // 0 real, 1 complex, 2 any: the closest element of the kind (scipy _nearest_real_complex_idx)
  std::vector<std::size_t> order(from.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return std::abs(from[a] - to) < std::abs(from[b] - to); });
  for (const std::size_t i : order) if (which == 2 || (which == 0) == isReal(from[i])) return i;
  throw std::invalid_argument("zpkToSos: no root of the wanted kind to pair with");
}
void erase(Roots& v, std::size_t i) { v.erase(v.begin() + static_cast<std::ptrdiff_t>(i)); }
// one section of up to two zeros and two poles (scipy _single_zpksos): b and a as coefficients of z^-0, z^-1, z^-2 (a missing root is a factor 1)
void section(const Roots& zs, const Roots& ps, double* row) {
  auto coeff = [](const Roots& r, double* out) {   // prod (1 - r z^-1), up to degree two
    cd c[3] = {1.0, 0.0, 0.0};
    for (const cd& x : r) { for (int i = 2; i >= 1; --i) c[i] = c[i] - x * c[i - 1]; }
    for (int i = 0; i < 3; ++i) out[i] = c[i].real();
  };
  // scipy's zpk2tf right-aligns the numerator and denominator in the section: b has len(z)+1 coefficients, so a section with fewer zeros than two puts its coefficients at the END of [b0 b1 b2]
  double b[3], a[3];
  coeff(zs, b); coeff(ps, a);
  const std::size_t nb = zs.size() + 1, na = ps.size() + 1;
  for (int i = 0; i < 3; ++i) row[i] = 0.0;
  for (int i = 0; i < 3; ++i) row[3 + i] = 0.0;
  for (std::size_t i = 0; i < nb; ++i) row[3 - nb + i] = b[i];
  for (std::size_t i = 0; i < na; ++i) row[6 - na + i] = a[i];
}
}  // namespace

Eigen::MatrixXd zpkToSos(const std::vector<std::complex<double>>& zeros, const std::vector<std::complex<double>>& poles, double gain) {
  if (zeros.empty() && poles.empty()) { Eigen::MatrixXd s(1, 6); s << gain, 0, 0, 1, 0, 0; return s; }
  Roots p = poles, z = zeros;
  p.resize(std::max(p.size(), z.size()), 0.0);
  z.resize(p.size(), 0.0);
  const std::size_t nSections = (std::max(p.size(), z.size()) + 1) / 2;
  if (p.size() % 2 == 1) { p.push_back(0.0); z.push_back(0.0); }
  {
    Roots pc, zc; std::vector<double> pr, zr;
    cplxreal(p, pc, pr); cplxreal(z, zc, zr);
    p = pc; for (const double r : pr) p.push_back(r);
    z = zc; for (const double r : zr) z.push_back(r);
  }
  auto worst = [](const Roots& v) { std::size_t best = 0; for (std::size_t i = 1; i < v.size(); ++i) if (std::abs(1.0 - std::abs(v[i])) < std::abs(1.0 - std::abs(v[best]))) best = i; return best; };
  Eigen::MatrixXd sos = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nSections), 6);
  for (std::size_t si = nSections; si-- > 0;) {
    const std::size_t i1 = worst(p);
    const cd p1 = p[i1];
    erase(p, i1);
    double row[6];
    if (isReal(p1) && countReal(p) == 0) {                        // the last real pole: with the nearest real zero
      const std::size_t zi = nearest(z, p1, 0);
      const cd z1 = z[zi]; erase(z, zi);
      section({z1, 0.0}, {p1, 0.0}, row);
    } else if (p.size() + 1 == z.size() && !isReal(p1) && countReal(p) == 1 && countReal(z) == 1) {   // one real pole and one real zero left: this pole takes a complex zero
      const std::size_t zi = nearest(z, p1, 1);
      const cd z1 = z[zi]; erase(z, zi);
      section({z1, std::conj(z1)}, {p1, std::conj(p1)}, row);
    } else {
      cd p2;
      if (isReal(p1)) {
        Roots real; std::vector<std::size_t> at;
        for (std::size_t i = 0; i < p.size(); ++i) if (isReal(p[i])) { real.push_back(p[i]); at.push_back(i); }
        const std::size_t k = at[worst(real)];
        p2 = p[k]; erase(p, k);
      } else p2 = std::conj(p1);
      if (!z.empty()) {
        const std::size_t zi = nearest(z, p1, 2);
        const cd z1 = z[zi]; erase(z, zi);
        if (!isReal(z1)) section({z1, std::conj(z1)}, {p1, p2}, row);
        else if (!z.empty()) {
          const std::size_t z2i = nearest(z, p1, 0);
          const cd z2 = z[z2i]; erase(z, z2i);
          section({z1, z2}, {p1, p2}, row);
        } else section({z1}, {p1, p2}, row);
      } else section({}, {p1, p2}, row);
    }
    for (int j = 0; j < 6; ++j) sos(static_cast<Eigen::Index>(si), j) = row[j];
  }
  sos.row(0).head(3) *= gain;
  return sos;
}

Eigen::MatrixXd butterSos(int order, const std::vector<double>& Wn, Band type) {
  const ZeroPoleGain f = butterZpk(order, Wn, type);
  return zpkToSos(f.z, f.p, f.k);
}

}  // namespace nxr::signal

// nxr::spectral — the scaling of a spectral quantity, chosen at read (spectral_scaling.h).
#include "nxr/spectral_scaling.h"

#include <stdexcept>

namespace nxr::spectral {

Target parseTarget(const std::string& n) {
  if (n == "stored") return Target::Stored;
  if (n == "density") return Target::Density;
  if (n == "power") return Target::Power;
  if (n == "amplitude") return Target::Amplitude;
  if (n == "amplitude-density") return Target::AmplitudeDensity;
  if (n == "db") return Target::Decibel;
  if (n == "db-density") return Target::DecibelDensity;
  throw std::invalid_argument("scaling '" + n + "': one of stored, density, power, amplitude, amplitude-density, db, db-density");
}
std::string targetName(Target t) {
  switch (t) {
    case Target::Stored: return "stored";
    case Target::Density: return "density";
    case Target::Power: return "power";
    case Target::Amplitude: return "amplitude";
    case Target::AmplitudeDensity: return "amplitude-density";
    case Target::Decibel: return "db";
    default: return "db-density";
  }
}
Stored parseStored(const std::string& n) {
  if (n == "density") return Stored::Density;
  if (n == "spectrum") return Stored::Spectrum;
  if (n == "band-power") return Stored::BandPower;
  throw std::invalid_argument("stored scaling '" + n + "': one of density, spectrum, band-power");
}
std::string storedName(Stored s) { return s == Stored::Density ? "density" : s == Stored::Spectrum ? "spectrum" : "band-power"; }
bool isLinear(Target t) { return t == Target::Stored || t == Target::Density || t == Target::Power; }

namespace {
double need(double x, const char* what) {
  if (!(std::isfinite(x) && x > 0)) throw std::invalid_argument(std::string("spectral scaling: the conversion needs ") + what + ", which this quantity's provenance does not give");
  return x;
}
// the factor from the stored value to a density and to a power (degree 1)
double toDensity(const Scaling& s) {
  switch (s.stored) {
    case Stored::Density: return 1.0;
    case Stored::Spectrum: return 1.0 / need(s.binEnbwHz, "the window's noise bandwidth per bin");
    default: return 1.0 / need(s.bandWidthHz, "the band's bandwidth");
  }
}
double toPower(const Scaling& s) {
  switch (s.stored) {
    case Stored::BandPower: return 1.0;
    default: return toDensity(s) * need(s.bandWidthHz, "the band's bandwidth");
  }
}
}  // namespace

double linearFactor(const Scaling& s, Target t) {
  if (s.degree < 1) throw std::invalid_argument("spectral scaling: the degree is at least 1");
  double f = 1.0;
  if (t == Target::Density) f = toDensity(s);
  else if (t == Target::Power) f = toPower(s);
  else if (t != Target::Stored) throw std::invalid_argument("spectral scaling: " + targetName(t) + " is not linear (not a factor): it applies to a real power, not to a cross-spectrum");
  return std::pow(f, s.degree);
}

double convert(double v, const Scaling& s, Target t) {
  if (isLinear(t)) return v * linearFactor(s, t);
  if (s.degree != 1) throw std::invalid_argument("spectral scaling: " + targetName(t) + " is defined for a spectrum (degree 1), not for a product of spectra");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  switch (t) {
    case Target::Amplitude: { const double p = v * toPower(s); return p >= 0 ? std::sqrt(p) : nan; }
    case Target::AmplitudeDensity: { const double d = v * toDensity(s); return d >= 0 ? std::sqrt(d) : nan; }
    case Target::Decibel: { const double p = v * toPower(s); return p > 0 ? 10 * std::log10(p) : nan; }
    default: { const double d = v * toDensity(s); return d > 0 ? 10 * std::log10(d) : nan; }
  }
}

std::string unitOf(const std::string& pu, const Scaling& s, Target t) {
  auto deg = [&](const std::string& u) { return s.degree == 1 ? u : "(" + u + ")^" + std::to_string(s.degree); };
  switch (t) {
    case Target::Stored: return deg(s.stored == Stored::Density ? pu + "/Hz" : pu);
    case Target::Density: return deg(pu + "/Hz");
    case Target::Power: return deg(pu);
    case Target::Amplitude: return "\xe2\x88\x9a(" + pu + ")";
    case Target::AmplitudeDensity: return "\xe2\x88\x9a(" + pu + ")/\xe2\x88\x9aHz";
    case Target::Decibel: return "dB re 1 " + pu;
    default: return "dB re 1 " + pu + "/Hz";
  }
}

}  // namespace nxr::spectral

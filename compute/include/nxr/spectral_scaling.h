#pragma once
// nxr::spectral — THE SCALING OF A SPECTRAL QUANTITY, chosen at READ. PURE MATHS: numbers in, numbers out.
//
// What is stored (`Stored`):
//   Density    power per Hz (units²/Hz): Welch with scipy's "density" scaling (Brainstorm's physical units); a band's value is the MEAN over its bins
//   Spectrum   power per bin (units²): Welch with scipy's "spectrum" scaling, the power of a tone at the bin; density = spectrum / the window's ENBW per bin
//   BandPower  the power the band-pass passes (units²): the variance of the band-passed signal (the analytic estimator: ⟨|z|²⟩ / 2, Re z the band-pass)
// The normalisers (`Scaling`): the bandwidth the value spans (a Welch band: its bins × the resolution; one bin: the resolution; the analytic band: the
// noise bandwidth of its band-pass, timebank::noiseBandwidth), the window's equivalent noise bandwidth per bin (Welch), and the DEGREE (1 for a spectrum,
// 2 for a product of two, e.g. quadrature's |Im C|²: every factor enters squared).
//
// Targets (`Target`, by name):
//   stored            as stored
//   density           units²/Hz                  stored → Density: Density as is; Spectrum / binEnbw; BandPower / bandWidth
//   power             units² in the band         Density × bandWidth; Spectrum / binEnbw × bandWidth; BandPower as is (the broadband reading of a
//                                                Welch band: density × its width; a tone's power is the Spectrum value itself)
//   amplitude         units, √power              (degree 1 only; a negative value gives NaN)
//   amplitude-density units/√Hz, √density        (degree 1 only)
//   db                10 log10(power), dB re 1 units²            (degree 1 only)
//   db-density        10 log10(density), dB re 1 units²/Hz       (degree 1 only)
// A LINEAR target (stored, density, power) is a factor (raised to the degree): it applies to a complex value (a cross-spectrum) and commutes with sums,
// means, maxima. The others are pointwise non-linear: they apply to a non-negative real value (a power, a statistic of one), never to a cross-spectrum.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace nxr::spectral {

enum class Stored { Density, Spectrum, BandPower };
enum class Target { Stored, Density, Power, Amplitude, AmplitudeDensity, Decibel, DecibelDensity };

struct Scaling {
  Stored stored = Stored::Density;
  int degree = 1;
  double bandWidthHz = std::numeric_limits<double>::quiet_NaN();   // the bandwidth the value spans
  double binEnbwHz = std::numeric_limits<double>::quiet_NaN();     // Welch: fs Σw² / (Σw)² (needed for Stored::Spectrum)
};

/** "stored" · "density" · "power" · "amplitude" · "amplitude-density" · "db" · "db-density"; anything else throws std::invalid_argument naming them. */
Target parseTarget(const std::string& name);
std::string targetName(Target t);
/** "density" · "spectrum" · "band-power"; anything else throws. */
Stored parseStored(const std::string& name);
std::string storedName(Stored s);
bool isLinear(Target t);
/** The factor a LINEAR target multiplies the stored value by (already raised to the degree). Throws std::invalid_argument for a non-linear target, or
 *  when a normaliser it needs is missing (not finite, or ≤ 0). */
double linearFactor(const Scaling& s, Target t);
/** A real stored value converted to the target (linear or not). A non-linear target with degree ≠ 1 throws; a negative power gives NaN. */
double convert(double value, const Scaling& s, Target t);
/** The unit of the target, from the power unit (e.g. "A²·m²"): density "A²·m²/Hz", amplitude "√(A²·m²)", db "dB re 1 A²·m²", … ; degree 2 squares it. */
std::string unitOf(const std::string& powerUnit, const Scaling& s, Target t);

}  // namespace nxr::spectral

#pragma once
// nxr::signal — ONE GABOR ATOM fitted to a burst (spec docs/specs/2026-11-20-gabor-bursts.md, pieces 4, 5 and 7; ported from the reference detector: direct, refine, is_burst,
// and fit_r2).
//
// PURE MATHS: a series (any: a sensor channel, a source; the functions never know which), its rate, an atom (t0, f, σ) in, numbers out. The bank of atoms on a grid is the time filter bank's Gabor
// family (timefilterbank.h, Family::Gabor); here one atom at a CONTINUOUS (t0, f, σ), by the inner product itself.
//
//   ψ(t; t0, f, σ) = (πσ²)^(−1/4) e^(−(t−t0)²/2σ²) e^(i2πf(t−t0))      (unit energy)        W(t0, f, σ) = ∫ x ψ* dt
//
// For a Gaussian burst A e^(−(t−t0)²/2s²) cos(2πf(t−t0) + φ), |W| is largest at σ = s, where |W| = (A/2) π^(1/4) √σ and arg W = φ: so the amplitude is A = 2|W| / (π^(1/4) √σ), the duration
// (full width at half maximum) 2.355 σ, the cycles at half maximum 2.355 σ f (the reference's constant 2.355).
//
// What it cannot resolve: the time and frequency of an atom are known to its own resolution (σ and σ_f = 1/(2πσ)); two bursts closer than that are one atom. The fit refines f and σ, never t0
// (the grid's time is kept, as the reference). A burst that is not Gaussian in its envelope is matched by the atom of the nearest width; R² says how much of the band's signal one atom explains.
#include <complex>
#include <Eigen/Dense>

namespace nxr::signal {

/** W(t0, f, σ) = ∫ x ψ* dt by the inner product over t0 ± 5σ (the reference's `direct`): x [n] sampled at fs, sample i at time i/fs (s); the sum is over the samples inside [0, n), divided by fs. */
std::complex<double> gaborCoefficient(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, double t0, double f, double sigma);

/** A MATCHED ATOM: its centre (s, the series' clock), frequency (Hz), width σ (s), full width at half maximum 2.355 σ (s), cycles at half maximum 2.355 σ f, amplitude 2|W| / (π^(1/4) √σ) (the series'
 *  unit), phase arg W (rad, of cos(2πf(t − t0) + φ)), and |W| (the series' unit · √s). */
struct GaborAtom {
  double t0 = 0, f = 0, sigma = 0, fwhm = 0, cycles = 0, amplitude = 0, phase = 0, coefficient = 0;
};

/** THE FIT (the reference's `refine`): from a grid atom (t0, f, σ), a parabola through log|W| at three points in frequency (f ± σ_f/2, σ_f = 1/(2πσ)), then in log width (σ · 2^(±1/8)), the vertex
 *  clipped to the outer points and taken only where the parabola opens downwards; `steps` rounds (2); then the atom's numbers at the refined (f, σ). t0 is kept. */
GaborAtom gaborFit(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, double t0, double f, double sigma, int steps = 2);

/** THE WIDTH TEST (the reference's `is_burst`, without its cycles test, which is a keep rule of its own): the atom's width lies inside the ladder (sigmaLow · √step < σ < sigmaHigh / √step) and |W| at
 *  σ · step and at σ / step is at most the atom's |W|: the coefficient peaks along width too. `step` is the ladder's (2^(1/perOctave)). */
bool gaborWidthPeak(const Eigen::Ref<const Eigen::VectorXd>& x, double fs, const GaborAtom& atom, double sigmaLow, double sigmaHigh, double step);

/** THE FIT TEST (the reference's `fit_r2`): R² = 1 − Σ (y − fit)² / Σ y² over the atom's half-maximum window [t0 − fwhm/2, t0 + fwhm/2], y the band's signal (the caller's: a band-passed series on the
 *  same clock as the atom), fit = A e^(−(t−t0)²/2σ²) cos(2πf(t − t0) + φ): the share of the band's signal there that one atom explains. NaN when the window holds no signal. */
double gaborR2(const Eigen::Ref<const Eigen::VectorXd>& y, double fs, const GaborAtom& atom);

}  // namespace nxr::signal

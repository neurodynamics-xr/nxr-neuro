#pragma once
// nxr::signal: the halves of a cross-spectral matrix, the quadrature fraction and share, rotary power and spectrum, phase and time lag.
// Moved from quadrature.h (toolbox refactor): declarations unchanged.
// The conventions (spectrum, lead, quadrature) are in quadrature.h.
#include <Eigen/Dense>
#include <complex>
#include <vector>

#include "nxr/signal/quadrature_pending.h"
#include "nxr/signal/crossspectral_decomposition.h"

namespace nxr::signal {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

// ── the two halves of a cross-spectral matrix ───────────────────────────────────────────────────────────────────────

/** Re C. */
MatrixXd coSpectrum(const MatrixXcd& C);
/** Im C = (C - C^H)/(2i) for a Hermitian C: antisymmetric. */
MatrixXd quadratureSpectrum(const MatrixXcd& C);

/** ||Im C||_F / ||C||_F of C = F F^H, from the small Gram matrices: ||C||^2 = ||F^H F||^2, ||Im C||^2 = (||C||^2 - Re tr(G conj(G)))/2 with G = F^T F. nxr's `im_fraction`. */
double quadratureFractionFromFactor(const MatrixXcd& F);
/** ||Im C||_F / ||C||_F of a stored Hermitian matrix C. nxr's `lagged_fraction`. */
double quadratureFraction(const MatrixXcd& C);

/** The share of ||Im C||_F^2 the pairs explain (see the header). */
double quadratureShare(const QuadratureFactor& f, const std::vector<QuadraturePair>& pairs);

/** The rotary power of a pair in the factor's patterns: P+ = ||F^T w||^2 and P- = ||F^H w||^2 with w = p + iq (the power of z = p.x + i q.x at +f and -f, summed over the factor's columns),
 *  and the sense (+1 when P+ >= P-). For a pair of Im(F F^H) as quadraturePairs gives it, P+ - P- = 4 sigma > 0: the sense is always +1; the numbers are the check. */
struct RotaryPower { double positive = 0, negative = 0; int sense = 1; };
RotaryPower rotaryPower(const MatrixXcd& F, const VectorXd& p, const VectorXd& q);

/** The rotary spectrum by complex Morlet transform: power(f, t) = |(z * w_{+f})(t)|^2 and |(z * w_{-f})(t)|^2, w_{+-f}(s) = exp(-s^2/(2 sd^2)) e^{+-2 pi i f s} / sum exp(-s^2/(2 sd^2)),
 *  sd = ncycles / (2 pi |f|), s in [-4 sd, 4 sd) at the sampling step ('same' length and centering as scipy.signal.fftconvolve). A standing pattern is symmetric. nxr's `rotary_power`. */
struct RotarySpectrum {
    MatrixXd positive;  // [freqs x T] at +f: z turning p -> q
    MatrixXd negative;  // [freqs x T] at -f
};
RotarySpectrum rotarySpectrum(const VectorXcd& z, double sfreq, const std::vector<double>& freqs, double ncycles = 7.0);

// ── delays ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** arg C_ij in (-pi, pi], rad, for every entry: positive where row i leads row j. */
MatrixXd phaseLag(const MatrixXcd& C);
double phaseLag(std::complex<double> cij);
/** arg C_ij / (2 pi f), seconds: row i leads row j by this time (negative: lags). Unambiguous within half a cycle (|delay| < 1 / (2 f)). */
MatrixXd timeLag(const MatrixXcd& C, double frequencyHz);
double timeLag(std::complex<double> cij, double frequencyHz);

}  // namespace nxr::signal

#pragma once
// nxr::signal: the halves of a cross-spectral matrix, the quadrature fraction and share, rotary power and spectrum, phase and time lag.
// Moved from quadrature.h (toolbox refactor): declarations unchanged.
// The conventions (spectrum, lead, quadrature) are in quadrature.h.
#include <Eigen/Dense>
#include <complex>
#include <vector>

#include "nxr/core/algebra.h"
#include "nxr/signal/crossspectral_decomposition.h"

namespace nxr::signal {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

// ── the two halves of a cross-spectral matrix ───────────────────────────────────────────────────────────────────────

/** Re C. */
MatrixXd cospectrum(const MatrixXcd& C);
/** Im C = (C - C^H)/(2i) for a Hermitian C: antisymmetric. */
MatrixXd quadspectrum(const MatrixXcd& C);

/** ||Im C||_F / ||C||_F of C = F F^H, from the small Gram matrices: ||C||^2 = ||F^H F||^2, ||Im C||^2 = (||C||^2 - Re tr(G conj(G)))/2 with G = F^T F. nxr's `im_fraction`. */
double quadratureFractionFromFactor(const MatrixXcd& F);
/** ||Im C||_F / ||C||_F of a stored Hermitian matrix C. nxr's `lagged_fraction`. */
double quadratureFraction(const MatrixXcd& C);

/** The share of ||Im C||_F^2 the pairs explain (see the header). */
double lagged_share(const core::QuadratureFactor& f, const std::vector<QuadraturePair>& pairs);

/** The rotary power of a pair in the factor's patterns: P+ = ||F^T w||^2 and P- = ||F^H w||^2 with w = p + iq (the power of z = p.x + i q.x at +f and -f, summed over the factor's columns),
 *  and the sense (+1 when P+ >= P-). For a pair of Im(F F^H) as planes gives it, P+ - P- = 4 sigma > 0: the sense is always +1; the numbers are the check. */
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

// ── readings of a band's cross-spectrum kept bin by bin (from meeg/crossspectrum.h; spec docs/specs/2026-11-15-band-resolved-cross-spectrum.md) ──
// Pure functions of matrices: C(f) [M x M] complex per bin, C_ab = <X_a conj(X_b)> (Hermitian), and the bins' frequencies in Hz. SIGN (the stored phase's): arg C_ab > 0 when channel a LEADS
// channel b (X_a = X_b e^{+i 2 pi f tau}, tau > 0), so a delay tau and a phase slope d(arg C_ab)/df = 2 pi tau have the same sign.
/** Coherency per bin: C_ab(f) / sqrt(C_aa(f) C_bb(f)) (NaN where a channel has no power at the bin); its modulus is the coherence, its imaginary part Nolte's. */
std::vector<Eigen::MatrixXcd> coherency(const std::vector<Eigen::MatrixXcd>& C);

struct PhaseSlopeOptions {
  /** "coherence": least squares weighted by |coherency|^2 per bin (the bins where the pair is coupled count); "none": every bin counts the same. */
  enum class Weights { Coherence, None } weights = Weights::Coherence;
  // the phase is unwrapped along frequency by adjacent bins (each step taken within +-pi of the last): the only unwrap there is
};
struct PhaseSlope {
  Eigen::MatrixXd delayMs;      // [M x M] tau = (1 / 2 pi) d phi / df in ms: the group delay of the pair over the bins; > 0 when the row channel leads. NaN with fewer than 2 bins or weight 0
  Eigen::MatrixXd residual;     // [M x M] the fit's weighted root-mean-square residual, radians: how far the unwrapped phase is from a straight line in f
};
/** Weighted least squares of the unwrapped phase of C_ab(f) on f (a line with a free intercept). `frequencies` in Hz, one per bin, increasing. */
PhaseSlope phaseSlope(const std::vector<Eigen::MatrixXcd>& C, const Eigen::VectorXd& frequencies, const PhaseSlopeOptions& options = {});

/** The phase-slope index of Nolte et al. 2008 (PRL 100, 234101): Psi_ab = Im sum_f conj(coherency_ab(f)) coherency_ab(f + df) over adjacent bins, written from the paper.
 *  [M x M], antisymmetric; > 0 when the row channel leads (sends to) the column channel. Not normalised (Nolte's Psi / std needs the segments' C: a later option). */
Eigen::MatrixXd phase_slope_index(const std::vector<Eigen::MatrixXcd>& C);

}  // namespace nxr::signal

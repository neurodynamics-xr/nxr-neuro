// nxr::meeg — readings of a band's cross-spectrum kept bin by bin (`crossspectrum compute --resolve bins`; spec docs/specs/2026-11-15-band-resolved-cross-spectrum.md).
// Pure functions of matrices: C(f) [M x M] complex per bin, C_ab = <X_a conj(X_b)> (Hermitian), and the bins' frequencies in Hz. They know nothing of where C came from.
//
// SIGN (the stored phase's): arg C_ab > 0 when channel a LEADS channel b (X_a = X_b e^{+i 2 pi f tau}, tau > 0). So a delay tau and a phase slope d(arg C_ab)/df = 2 pi tau
// have the same sign: tau > 0 and Psi > 0 when the ROW channel leads the column channel.
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace nxr::meeg {

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
Eigen::MatrixXd phaseSlopeIndex(const std::vector<Eigen::MatrixXcd>& C);

}  // namespace nxr::meeg

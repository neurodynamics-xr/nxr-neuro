#pragma once
// nxr::signal — amplitude and phase of complex coefficients (MATLAB's names: abs, angle, unwrap, instfreq), and the carrier phase of wt's
// baseband coefficients. PURE MATHS: arrays in, arrays out.
//
// LAYOUT: time runs down the ROWS, one series per column (as MATLAB's matrices and wt::Block::coef: rows = instants, columns = channels).
//
// AMPLITUDE: with a filterbank's gain G = 2 (timebank's analytic convention), |c| of a real tone of amplitude A in a member's passband is A.
// PHASE: angle(c) in (-pi, pi], the phase of cos: a tone A cos(2 pi f t + phi) has analytic signal A e^{i(2 pi f t + phi)}.
//
// THE CARRIER (wt only). wt stores member m shifted to baseband by its lowest support bin, c_m(t) = a_m(t) e^{-2 pi i f_lo t}, f_lo = support.first * fs / nfft,
// so angle(c) is the phase RELATIVE to f_lo, and members (or a member and the FFT) cannot be compared until it is undone: carrier() returns
// a_m(t_k) = c_m(t_k) e^{+2 pi i f_lo t_k} at the block's instants t_k = k p/q samples (k global), the factor's phase computed exactly in integers
// (no precision lost hours into a recording).
#include "nxr/signal/wavelet.h"

#include <Eigen/Dense>

namespace nxr::signal {

/** |c|, elementwise. */
Eigen::MatrixXd abs(const Eigen::MatrixXcd& c);
/** arg(c) in (-pi, pi], elementwise. */
Eigen::MatrixXd angle(const Eigen::MatrixXcd& c);
/** MATLAB unwrap along each column: wherever consecutive phases jump by more than `tolerance` (default pi), add the multiple of 2 pi that brings
 *  the jump into (-pi, pi]. */
Eigen::MatrixXd unwrap(const Eigen::MatrixXd& phase, double tolerance = 3.14159265358979323846);
/** The instantaneous frequency in Hz of each column of an analytic series sampled at `rate` Hz: (rate / 2 pi) dphi/dt, dphi between neighbours as
 *  arg(z_{k+1} conj z_k) (no unwrapping needed); at the interior samples the mean of the two neighbouring steps, at the ends the one step.
 *  A series of one sample gives 0. For wt coefficients pass carrier()'s output (or add f_lo to the baseband result) and the member's rate. */
Eigen::MatrixXd instfreq(const Eigen::MatrixXcd& z, double rate);

/** wt: member m's shift frequency f_lo in Hz. */
double carrierFrequency(const signal::Plan& plan, int member);
/** wt: the block's coefficients with the baseband shift undone, a_m(t_k) = c e^{+2 pi i f_lo t_k} (rows = instants first, first+1, ...). */
Eigen::MatrixXcd carrier(const signal::Plan& plan, int member, const signal::Block& block);
/** wt: the times of the block's instants, in seconds from sample 0 of the record. */
Eigen::VectorXd instants(const signal::Plan& plan, int member, const signal::Block& block);

// Moved from quadrature.h (toolbox refactor): declarations unchanged.
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

// ── standing or circular ───────────────────────────────────────────────────────────────────────────────────────────

/** rho(u) = |u^T u| / (u^H u) in [0, 1]; NaN for the zero pattern. */
double impropriety(const VectorXcd& u);

/** Z rotated so that sum Z^2 is real and positive, its real part kept, then rotated back: the standing pattern with the same real part up to its best phase (alignment 0 by construction). */
VectorXcd standingSurrogate(const VectorXcd& z);

}  // namespace nxr::signal

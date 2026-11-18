#pragma once
// nxr::signal jwt / ijwt — the joint time-vertex wavelet transform (Rheome's wt / iwt of @jointfilterbank, GSPBox's gsp_jtv_filter_analysis): on a spectrum, on modes x time, on vertices x time (exact through
// an EigenTransform, or by a Chebyshev series for a separable bank). A joint bank only DESIGNS (nxr/signal/jointfilterbank.h).
#include "nxr/signal/graphwavelet.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/jointfilterbank.h"

namespace nxr::signal {
using jointfilterbank::JointFilterbank;
using graphfilterbank::Synthesis;
using JointCoefficients = jointfilterbank::Coefficients;

// ── the spectrum domain (Rheome: wt / iwt, jtv.analysis / jtv.synthesis) ────────────────────────────────────────────────

/** Y_i = C ⊙ W_i  [K × F] each. */
std::vector<Eigen::MatrixXcd> jwtSpectrum(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const Eigen::MatrixXcd& C);
/** Dual: Σ conj(W_i/S) ⊙ Y_i — recovers C exactly wherever S > 0. Tight: Σ conj(W_i) ⊙ Y_i = S ⊙ C (Rheome 'tight'). */
Eigen::MatrixXcd ijwtSpectrum(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const std::vector<Eigen::MatrixXcd>& Y, Synthesis mode = Synthesis::Dual);

// ── modes × time ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Analytic bank: a real series per mode, C [K × T]. */
JointCoefficients jwt(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const Eigen::MatrixXd& C);
/** Two-sided bank: complex (or real) series. An analytic bank REFUSES this overload. */
JointCoefficients jwt(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const Eigen::MatrixXcd& C);
/** → the real series [K × n] (analytic bank). Dual is exact where the bank covers; Tight needs both factors tight and is exact then. */
Eigen::MatrixXd ijwt(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);
/** → the complex series [K × n] (two-sided bank). */
Eigen::MatrixXcd ijwtComplex(const JointFilterbank& bank, const Eigen::VectorXd& lambda, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);

// ── vertices × time ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Exact: W_i = Φ [ g(Λ) ⊙ time-filter(Φᵀ B F) ] — any kernel. F [n × T] real (analytic bank) or complex; blocks [n × nfft]. */
JointCoefficients jwt(const JointFilterbank& bank, const graphtransforms::EigenTransform& t, const Eigen::MatrixXd& F);
JointCoefficients jwt(const JointFilterbank& bank, const graphtransforms::EigenTransform& t, const Eigen::MatrixXcd& F);
Eigen::MatrixXd ijwt(const JointFilterbank& bank, const graphtransforms::EigenTransform& t, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);
Eigen::MatrixXcd ijwtComplex(const JointFilterbank& bank, const graphtransforms::EigenTransform& t, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);
/** Polynomial: SEPARABLE banks only (a kernel couples λ and f, which a series in the operator cannot see; refused with a clear error). */
JointCoefficients jwt(const JointFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const Eigen::MatrixXd& F);
JointCoefficients jwt(const JointFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const Eigen::MatrixXcd& F);
Eigen::MatrixXd ijwt(const JointFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);
Eigen::MatrixXcd ijwtComplex(const JointFilterbank& bank, const graphtransforms::ChebyshevOperator& t, const JointCoefficients& W, std::size_t n, Synthesis mode = Synthesis::Dual);

// the energy of each member, sum |W_i|^2 over vertices (or modes) and time, is a measurement: nxr::measurement::memberTotals (nxr/measurement/band_energy.h).

}  // namespace nxr::signal

#pragma once
// nxr::signal — the space–time kernels of Rheome's +filters, as PURE kernel functions of (λ, f) or (λ, t) or λ.
//
// Every parameter is an argument (no defaults: a kernel's parameters are choices, as a bank's family is). Frequencies are in HERTZ and SIGNED, in EVERY
// (λ, f) kernel; the grid of an FFT is turned into signed frequencies with `signedFrequencies(grid, Half)` (the half stated, never guessed) or
// timebank/fft's fftfreq. Outputs are [K × N]: modes by frequencies (or lags). Units are PHYSICAL: λ in 1/length², c in length/s, t in s, f in Hz.
// A (λ, f) kernel multiplies a joint spectrum. A (λ, t) kernel (diffusion, wave, dampedwave, kleingordon, schrodinger) acts on a time LAG — the response to
// an impulse as time goes by — and CANNOT multiply a joint (λ, f) spectrum; it must be transformed along time first. The joint bank refuses it
// (jointbank::LagKernel) and says why. The propagators are the CONTINUUM forms; the Chebyshev-angle form is the named Variant::Chebyshev.
// The ambiguities found in Rheome's definitions and how they were ruled are in docs/specs/2026-11-10-filterbanks.md (§ kernels: RESOLVED).
#include "nxr/signal/graphfilterbank.h"
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace nxr::signal {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/** Which half of the spectrum a frequency grid covers: Full = bins 0 … N−1 at df (an FFT grid: the upper half is negative); Positive = bins 0 … fs/2 (a
 *  real signal's one-sided grid: every bin non-negative, nothing wraps). The grid alone cannot say which (both are 0, df, 2df, …), so it is stated. */
enum class Half { Full, Positive };
/** The grid as signed frequencies (Hz). Full: w − Fs·(w ≥ Fs/2), Fs = N·df (+Fs/2 maps to −Fs/2, as fftfreq); Positive: unchanged. Rheome's signedfrequencies. */
VectorXd signedFrequencies(const VectorXd& grid, Half half);

// ── (λ, f) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/** A spatiotemporal Gabor packet: Gaussian in k = √λ about k0 (width k0/3) × Gaussian in f about ±f0 (width sf). */
MatrixXd gabor(const VectorXd& lambda, const VectorXd& f, double k0, double f0, double sf);
/** A travelling-wave dispersion ridge, with a SENSE. f is SIGNED, in Hz; c in length/s, λ in 1/length², width in Hz.
 *  sense = +1: exp(−(f − c√λ/2π)² / 2·width²) (positive frequencies only); −1: the ridge at −c√λ/2π; 0: both, a function of |f| (the choice for the even
 *  spectrum of a real coefficient). On a complex (rotary) coefficient the sign of f is the sense of travel, so a wave going one way and the same wave going
 *  the other sit on opposite sides of f = 0 and +1 / −1 separate them. Throws unless sense ∈ {−1, 0, +1}. */
MatrixXd travwave(const VectorXd& lambda, const VectorXd& f, double c, double width, int sense);
/** A damped harmonic resonator at f0, the same for every mode: f0² / (f0² − f² + i·f·f0/Q) (complex). */
MatrixXcd resonator(const VectorXd& lambda, const VectorXd& f, double f0, double Q);
/** The spatiotemporal 1/f (Whittle–Matérn) background: (κ² + λ + (f/v)²)^(−ν). */
MatrixXd stmatern(const VectorXd& lambda, const VectorXd& f, double kappa, double nu, double v);
/** A Gaussian temporal band-pass on |f| (both signs): exp(−(|f| − f0)² / 2 bw²). */
VectorXd temporalBandpass(const VectorXd& f, double f0, double bw);
/** Raised-cosine temporal band (Rheome psitime / jointfilterbank.band): 0 outside (f1, f2), cos² edges of `width` Hz, 1 inside; `omega` is 2πf (rad/s) there. */
VectorXd raisedCosineBand(const VectorXd& f, double f1, double f2, double width);

// ── (λ, t): time-LAG propagators, physical units ───────────────────────────────────────────────────────────────────────────
/** Which form of the wave family: Continuum (the definition: ω = c√λ, ω² = c²λ + m², in physical units, nothing normalised by λmax) or Chebyshev (the earlier
 *  discrete-time form: θ = acos(1 − 2λ/λmax), "frequency" c·θ bounded and relative to λmax; for Klein–Gordon the mass shifts λ inside the acos). Never the default. */
enum class Variant { Continuum, Chebyshev };
/** exp(−D·λ·|t|): λ in 1/length², D (diffusivity) in length²/s, t in s. Not normalised by λmax. */
MatrixXd diffusion(const VectorXd& lambda, const VectorXd& t, double D);
/** cos(t·c·√λ) — each mode at ω = c√λ rad/s (c in length/s). On the unit sphere of radius R: ω_l = c√(l(l+1))/R. */
MatrixXd wave(const VectorXd& lambda, const VectorXd& t, double c);
/** Chebyshev variant: cos(c·t·θ_k), θ_k = acos(clamp(1 − 2λ/λmax)); λmax ≤ 0 → max λ (the earlier Rheome `wave(λ, t, α, lmax)`, numbers unchanged). */
MatrixXd wave(const VectorXd& lambda, const VectorXd& t, double c, Variant variant, double lmax);
/** exp(−β|t|)·cos(t·c·√λ): β (1/s) is a given amplitude decay rate. */
MatrixXd dampedwave(const VectorXd& lambda, const VectorXd& t, double c, double beta);
MatrixXd dampedwave(const VectorXd& lambda, const VectorXd& t, double c, double beta, Variant variant, double lmax);   // exp(−β|t|)·cos(c·t·θ_k)
/** cos(t·√(c²λ + m²)): m the rest frequency (rad/s); m enters only through ω², even the λ = 0 mode rings at m. */
MatrixXd kleingordon(const VectorXd& lambda, const VectorXd& t, double c, double m);
/** Chebyshev variant: cos(c·t·acos(clamp(1 − 2(λ + μ)/λmax))) with μ the mass shift in eigenvalue units (a shifted Laplacian, not a mass term). */
MatrixXd kleingordon(const VectorXd& lambda, const VectorXd& t, double c, double mu, Variant variant, double lmax);
/** The free propagator exp(−i λ t / ħ): unitary, |G| = 1 (complex); each mode turns at λ/ħ rad/s. Not normalised by λmax. */
MatrixXcd schrodinger(const VectorXd& lambda, const VectorXd& t, double hbar);

// ── λ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/** e^(−λ t) (λ divided by λmax first when λmax > 0), one column per t. */
MatrixXd heat(const VectorXd& lambda, const VectorXd& t, double lmax);
/** tλ·e^(−tλ), one column per t. */
MatrixXd mexhat(const VectorXd& lambda, const VectorXd& t);
/** e^(−λ t1) − e^(−λ t2), t1 < t2 elementwise. */
MatrixXd diffgauss(const VectorXd& lambda, const VectorXd& t1, const VectorXd& t2);
/** Analytic response of a rotation-invariant filter on the unit sphere: Σ_l g_l (2l+1)/(4π) P_l(cos γ), g_l the gain on degree l. */
VectorXd sphere_kernel(const VectorXd& gamma, const VectorXd& gl);

// ── quaternion fields ──────────────────────────────────────────────────────────────────────────────────────────────────────
/** Steer a full-quaternion field [4n × m] (w x y z per vertex) by RIGHT multiplication with a unit quaternion q0 (normalised here). One global rotation. */
MatrixXd steer(const MatrixXd& W, const Eigen::Vector4d& q0);
/** Rotate a vector field V [n × 3] by a position-dependent quaternion, by conjugation q V q̄, with q = (cos θ/2, sin θ/2 · axis): the FULL angle θ. */
MatrixXd spin(const MatrixXd& V, const MatrixXd& axis, const VectorXd& theta);

// ── graph wavelet scattering (Rheome filters.scatter; Mallat 2012) ───────────────────────────────────────────────────────

struct Scatter {
    MatrixXcd S0;                                  // <F>, [nR × nT]: the pooled field
    std::vector<MatrixXd> S1;                      // [M] of [nR × nT]: <|g_m1(L) F|>
    std::vector<std::pair<int, int>> pairs;        // the (m1, m2) computed: sigma(m2) > sigma(m1) (Mallat's coarser-second rule), neither a low-pass
    std::vector<MatrixXd> S2;                      // one [nR × nT] per pair: <|g_m2(L) |g_m1(L) F||>
    VectorXd sigma;                                // member scale 1/(centre wavenumber), every family; NaN for a low-pass member (it has none and is not paired)
};
/** Low-pass members: Role::Scaling, or (not for a heat bank, whose members are a scale-space of low-passes) a member whose gain at λ = 0 is at least half its peak over [0, λmax]. */
std::vector<bool> lowPass(const signal::graphfilterbank::GraphFilterbank& bank);
/** |W| cascaded through the bank in the exact eigenbasis, then pooled. The second member must be coarser (smaller centre wavenumber); the low-pass takes no part. `order` 1 or 2. `pool` [n × nR]: pooling weights (empty: the mass-weighted global mean).
 *  F real or complex ([n × nT]; complex is the strongly preferred input: |·| of a real graph-wavelet coefficient rectifies it). */
Scatter scatter(const signal::graphfilterbank::GraphFilterbank& bank, const signal::graphfilterbank::EigenTransform& t, const MatrixXcd& F, int order, const MatrixXd& pool = MatrixXd());

}  // namespace nxr::signal

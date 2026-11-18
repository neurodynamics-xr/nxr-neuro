#pragma once
// TEMPORARY FORWARDER (toolbox refactor): deleted at the end of batch B5, see tools/refactor.py forwarders.
// nxr::quadrature — the part of a cross-spectrum that no zero-delay mixing can make, and the measures built on it. PURE MATHS (arrays in, arrays out). A port of nxr's
// `nxr.spectral.lagged` (MIT) with exact names.
//
// CONVENTIONS (one place, used by every function below)
//   Spectrum   a signal x(t) has the Fourier coefficient X(f) = sum_t x(t) e^{-2 pi i f t} (numpy's rfft; nxr::fft::r2c). A component cos(2 pi f t + phi) has X ∝ e^{i phi}.
//   Cross-spectral matrix  C_ij = E[X_i conj(X_j)], C = F F^H for a complex factor F [N x r] (its columns are r independent complex patterns). C is Hermitian.
//   Lead       arg C_ij = phi_i - phi_j. Im C_ij > 0 means row i LEADS row j (it peaks earlier, by phaseLag / (2 pi f) seconds).
//
//   coSpectrum(C)          Re C   the in-phase (co-) spectrum: symmetric
//   quadratureSpectrum(C)  Im C   the quadrature spectrum: ANTISYMMETRIC. Zero-delay mixing (a head model, an inverse, any real matrix K applied to the sensors) turns C into K C K^T, so Im C becomes
//                                 K (Im C) K^T: it adds no quadrature of its own. Sensors in phase (Im C = 0) stay in phase at every source; a lag read at the sources was in the sensors.
//   quadratureFactor(F)    (Q, S) with Im C = Q S Q^T, Q [N x m] orthonormal columns, S [m x m] antisymmetric (m <= 2r): through the QR of [Re F, Im F]; never an N x N matrix.
//   quadratureFraction     ||Im C||_F / ||C||_F  in [0, 1]; 0 for any in-phase mixture.
//   quadratureEnergy       per row i, sum_j (Im C_ij)^2 = ||Im(F_i F^H)||^2: the quadrature spectrum's energy in row i (nxr's `im_rows` is the sum of this over its rows).
//   quadraturePairs        the canonical decomposition of the antisymmetric Im C into eigenvalue pairs +- i sigma_k: Im C = sum_k sigma_k (p_k q_k^T - q_k p_k^T), p_k, q_k orthonormal real
//                          patterns, sigma_k > 0 the pair's quadrature amplitude, strongest first. BY THE LEAD CONVENTION ABOVE p LEADS q BY A QUARTER CYCLE: the activity is
//                          p cos(wt) + q sin(wt): p -> q -> -p -> -q, counter-clockwise in the (p, q) plane (the rotary spectrum's +f). (nxr's x = p, y = q.)
//                          The term sigma (p q^T - q p^T) is invariant under a rotation of (p, q) in their plane, so the pair is a plane with an orientation; we fix the gauge:
//                          the row where p^2 + q^2 is largest has q = 0 and p > 0.
//   quadratureShare        the share of ||Im C||_F^2 the pairs explain: ||B^T (Im C) B||_F^2 / ||Im C||_F^2, B the pairs' orthonormal basis (= sum 2 sigma_k^2 / ||Im C||_F^2).
//   impropriety            rho(u) = |u^T u| / u^H u (Schreier & Scharf's degree of impropriety) of a complex pattern: 1 a standing pattern (one real pattern times one phase),
//                          0 a circular one (a pure travelling or rotating pattern). 1 - impropriety is what nxr called the travel index.
//   rotarySpectrum         the rotary spectrum (Gonella 1972) of z = p.x + i q.x: the power at +f (p -> q) and at -f, by complex Morlet transform in time (nxr's `rotary_power`).
//   quadraturePairFilters  the top quadrature pair of a read-out of the sensors in a band, as two sensor filters and its rotary sense.
//   weightedPhaseGradient  P = Im(conj(z) grad z) = |z|^2 grad arg z, area-weighted over a log-map disc, each face's vector transported to the anchor: the direction and phaseGradientAlignment.
//   standingSurrogate      the best standing approximation of a complex map.
//   meanResultantLength · rayleighTest   circular statistics of directions.
//   phaseLag · timeLag     arg C_ij (rad) and arg C_ij / (2 pi f) (s): positive when row i leads row j.
#include "nxr/signal/quadrature_pending.h"
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/stats/circular.h"
#include "nxr/measurement/congruence_norms.h"
namespace nxr::quadrature { using namespace nxr::signal; using namespace nxr::stats; using namespace nxr::measurement; }

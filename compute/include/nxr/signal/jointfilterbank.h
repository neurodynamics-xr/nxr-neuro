#pragma once
// nxr::signal::jointfilterbank — the JOINT time–vertex filterbank on the (λ, f) plane (Rheome's @jointfilterbank + +jtv; GSPBox's gsp_jtv_design_dgw).
//
// PURE MATHS. A member is the product of three factors,   W_i(λ, f) = K_k(λ, f) · g_g(λ) · h_t(f),   one for each combination (ig, it, ik):
// Nf = Ng · Nt · Nk, the index running ig fastest (Rheome's ndgrid order). g_g are the members of a GRAPH bank (nxr::signal::graphfilterbank), h_t those of a TIME
// bank (nxr::signal::timefilterbank) on its grid — SIGNED f when the time bank is two-sided — and K an optional joint kernel (a speed-selective ridge, a dispersion
// curve, ...). Without kernels the bank is SEPARABLE; then its frame operator is the product S(λ, f) = S_g(λ) · S_t(f) and its bounds are the products
// of the factors' framebounds (A = A_g A_t, B = B_g B_t): tight iff both factors are. The canonical dual is g/S either way.
//
// THE TIME SIDE decides what the joint bank is for. Analytic (one-sided f ≥ 0): a REAL series per mode, complex coefficients (envelope + phase);
// Two-sided (signed f): a COMPLEX series per mode — complex eigenmode coefficients, rotating patterns (+f counter-clockwise, −f clockwise), x + iy
// fields — and speed-selective ridges with a SENSE (f = ± v√λ/2π). An analytic bank given a complex series is refused (timebank.h).
//
// Joint kernels K(λ, f) are called with f in HERTZ, SIGNED (signed on a two-sided bank), always — the same unit in Rheome's jtv.bank and jointfilterbank
// since the 2026-11-10 ruling (they used to pass ω = 2π f); a kernel that needs rad/s converts itself (`speedkernel`: ω = 2π f, so a speed stays in length/s).
// A (λ, t) LAG kernel (diffusion, wave, ...: nxr::signal) acts on a time lag and cannot multiply a joint (λ, f) spectrum; `make` refuses it (LagKernel).
//
// Domains. Spectrum: C [K × F] on the time bank's grid (Rheome's wt/iwt: an elementwise multiply). Modes: a [K × T] series of mode coefficients (time along
// the columns, as everywhere in nxr::signal::graphfilterbank) — FFT along time, multiply, inverse FFT. Vertices: a field [n × T] through an EigenTransform (exact,
// any kernel) or a ChebyshevOperator (separable banks only: time filter, then the graph series, no eigenvectors).
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/timefilterbank.h"
#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace nxr::signal::jointfilterbank {

using cd = std::complex<double>;
using Kernel = std::function<cd(double /*lambda*/, double /*f, Hz*/)>;
using nxr::signal::graphfilterbank::Synthesis;

struct Index { std::size_t g = 0, t = 0, k = 0; };

struct JointFilterbank {
    signal::graphfilterbank::GraphFilterbank graph;
    signal::timefilterbank::TimeFilterbank time;
    std::vector<Kernel> kernels;      // empty: separable (K = 1)
    std::vector<Index> index;         // [Nf], ig fastest
    int size() const { return static_cast<int>(index.size()); }
    bool separable() const { return kernels.empty(); }
};

/** Every combination of the graph bank's members, the time bank's members and the kernels (none = separable). */
JointFilterbank make(const signal::graphfilterbank::GraphFilterbank& graph, const signal::timefilterbank::TimeFilterbank& time, std::vector<Kernel> kernels = {});
/** The two factors on the plane, computed once for many members: graph gains Gg [K × Ng], time gains Gt [F × Nt], the time bank's frequencies f. */
struct Plane { Eigen::MatrixXd Gg, Gt; Eigen::VectorXd f; };
Plane plane(const JointFilterbank& bank, const Eigen::VectorXd& lambda);
Eigen::MatrixXcd jointfilters(const JointFilterbank& bank, const Plane& plane, const Eigen::VectorXd& lambda, int m);
Eigen::MatrixXd frameOperator(const JointFilterbank& bank, const Plane& plane, const Eigen::VectorXd& lambda);
/** A kernel of (λ, t): a propagator in time LAG. It is not a joint-plane kernel. */
struct LagKernel { std::function<cd(double /*lambda*/, double /*t, s*/)> g; };
/** Always throws std::invalid_argument: a (λ, t) kernel acts on a time lag (the response to an impulse as time goes by); it cannot multiply a joint (λ, f)
 *  spectrum and must first be transformed along the time axis. (Rheome: jtv.bank type 'ts'.) */
JointFilterbank make(const signal::graphfilterbank::GraphFilterbank& graph, const signal::timefilterbank::TimeFilterbank& time, const std::vector<LagKernel>& lagKernels);

/** Member m on the plane at the graph spectrum `lambda` and the time bank's grid → [K × F] (the DC / Nyquist rule of the time bank applies). */
Eigen::MatrixXcd jointfilters(const JointFilterbank& bank, int m, const Eigen::VectorXd& lambda);
/** S(λ, f) = Σ_i |W_i|²  → [K × F]. */
Eigen::MatrixXd frameOperator(const JointFilterbank& bank, const Eigen::VectorXd& lambda);
/** How the frame bounds are taken: over the whole plane (the operator S(lambda, f) on the grid) or, for a separable bank, as the PRODUCT of the factors' bounds (A_g A_t, B_g B_t; throws on a bank with kernels). Equal on the same grid. */
enum class BoundsBy { Plane, Product };
/** Frame bounds (analytic: DC and Nyquist bins excluded, as timefilterbank::framebounds). */
signal::graphfilterbank::Bounds framebounds(const JointFilterbank& bank, const Eigen::VectorXd& lambda, BoundsBy by = BoundsBy::Plane);
bool isframetight(const JointFilterbank& bank, const Eigen::VectorXd& lambda, double tol = 1e-6);

// THE TRANSFORMS (jwt, ijwt: on the spectrum, on modes x time, on vertices x time) are nxr/signal/jointwavelet.h; a joint bank only designs.
/** One [K × nfft] complex block per member (modes x time) or [n × nfft] (vertices x time). */
using Coefficients = std::vector<Eigen::MatrixXcd>;

// ── the axes (Rheome jtv.axes / jtv.compatible) ───────────────────────────────────────────────────────────────────────────

/** The axes a joint representation lives on, carried with it: Rheome's .lambda .f .half .boundary .fs .nfft. `half` is Analytic (positive half, complex) or
 *  TwoSided (signed). `boundary`: Ring (a plain FFT: periodic) | Path (a padded record). Two representations are combinable only if these agree. */
enum class Boundary { Ring, Path };
struct Axes { Eigen::VectorXd lambda, f; double fs = 0; std::size_t nfft = 0; signal::timefilterbank::Sides half = signal::timefilterbank::Sides::Analytic; Boundary boundary = Boundary::Ring; };
/** The axes of a bank on the graph spectrum `lambda`; `padded` marks a Path boundary (a record zero-padded beyond its length). */
Axes axes(const JointFilterbank& bank, const Eigen::VectorXd& lambda, bool padded = false);
/** Do two representations live on the same axes? Empty on success, else every mismatch found (sizes agreeing proves nothing: values, half and boundary too). */
std::vector<std::string> iscompatible(const Axes& a, const Axes& b, double tol = 1e-10);

// ── kernels worth naming ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Rheome's speedkernel(c, d): exp(−(2π f − c√λ)² / 2d²) — a ridge on the dispersion curve ω = 2π f = c√λ (positive f only); f in Hz, c in length/s, d the ANGULAR width (rad/s). */
Kernel speedkernel(double c, double d);
/** nxr's ridge_filter: exp(−½((f − sense·v·μ/2π)/width)²), μ = √λ (rad per unit length), v in length/s, `width` in Hz, sense ±1 (needs a two-sided bank). */
Kernel ridgeKernel(double v, double width, int sense);

}  // namespace nxr::signal::jointfilterbank

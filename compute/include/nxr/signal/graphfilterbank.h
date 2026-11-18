#pragma once
// nxr::signal::graphfilterbank — spectral filterbanks on a self-adjoint operator (the graph filterbank).
//
// PURE MATHS: gains are functions of an eigenvalue λ; a field is filtered either through the operator's EIGENBASIS (exact:
// coefficients g(λ_k)·⟨φ_k, f⟩_M) or by a CHEBYSHEV series in the PENCIL (L, M) (no eigenvectors). Arrays and operators in,
// arrays out; no file I/O, no mesh. Matches Rheome's @graphfilterbank + +graphtransform and nxr's filters.frames.graph_bank
// (the four Rheome families), and the Studio's / nxr's Meyer dyadic frame (wavelets/dyadicFrame.ts).
// Design note and where the three implementations differ: docs/specs/2026-11-10-filterbanks.md.
//
// A BANK is a list of M members g_m(λ). Its FRAME OPERATOR is S(λ) = Σ_m g_m(λ)²; the frame bounds are A = min S, B = max S, and
// the bank is TIGHT when A = B (then synthesis∘analysis = A·identity). Analysis: W_m = g_m(L) f. Synthesis: f = Σ_m w_m(L) W_m with
// w_m = g_m / S (the canonical DUAL: exact for any bank that covers the spectrum) or w_m = g_m (TIGHT: exact when S = const).
//
// Part 2 (time / continuous-wavelet and joint time–vertex banks) reuses GraphFilterbank, Member, gains/bounds and the Coefficients layout:
// a joint bank is a product of a time GraphFilterbank (gains on FFT bins, nxr::signal) and a graph GraphFilterbank.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/signal/graphtransforms.h"

namespace nxr::signal::graphfilterbank {

// ── the bank ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

enum class Family {
    MexHat,        // GSPBox's quartic scaling function + Mexican hats tλ·e^(−tλ) at log-spaced t.   Rheome · nxr   (not tight)
    Heat,          // low-pass e^(−tλ) at log-spaced t, coarse → fine (a diffusion scale-space).     Rheome · nxr   (not tight)
    Itersine,      // half-cosine windows, LINEAR in λ, Nf members.                                  Rheome · nxr   (tight in the interior)
    LogItersine,   // itersine windows on the log-wavenumber axis, low- and high-pass completing it. Rheome · nxr   (tight: Σg² = 1)
};
enum class FineLimit { Basis, Usable };   // t_min = 1/λmax (tiles the whole spectrum) or 4.744/λmax (every member responds fully)

/** The spectral range a bank is built on: λmax; [λmin, λmax]; or the whole λ vector (keeps it for exact frame bounds). */
struct Range {
    double lmax = 0;
    double lmin = std::numeric_limits<double>::quiet_NaN();   // NaN: unknown (then λmax / lpFactor is used)
    Eigen::VectorXd lambda;                                   // the exact spectrum; empty unless given
    static Range fromLmax(double lmax);
    static Range fromLimits(double lmin, double lmax);
    static Range fromSpectrum(const Eigen::VectorXd& lambda);   // λmin = smallest λ > λmax·1e-12
};

struct DesignOptions {
    int numFilters = 0;                  // wavelet count; 0 derives it from `voices` (Rheome's NumFilters = [])
    int voices = 3;                      // members per octave when derived (VoicesPerOctave)
    double lpFactor = 20;                // λmin = λmax / lpFactor when λmin is unknown (GSPBox's constant)
    FineLimit fineLimit = FineLimit::Basis;
    std::optional<std::pair<double, double>> scaleLimits;   // (t_min, t_max) overrides the policy above
};

enum class Role { Member, Scaling, Wavelet, Top };

struct Member {
    Role role = Role::Member;
    double scale = std::numeric_limits<double>::quiet_NaN();   // t (Mexican hat / heat); NaN where the family has none
    double lo = std::numeric_limits<double>::quiet_NaN();      // nominal band (Meyer edges); NaN otherwise
    double hi = std::numeric_limits<double>::quiet_NaN();
    double level = std::numeric_limits<double>::quiet_NaN();   // Meyer: the tiling level (Weyl: j)
    graphtransforms::Gain gain;
};

struct GraphFilterbank {
    std::string kind;                    // "mexhat" · "heat" · "itersine" · "logitersine" · "meyer"
    std::vector<Member> members;
    double total = std::numeric_limits<double>::quiet_NaN();   // Σ g² where the design makes it constant (1), else NaN
    double lmax = 0;                     // the spectral range the bank was designed on (Meyer: 0 — no range)
    double lmin = std::numeric_limits<double>::quiet_NaN();   // λmin as supplied (NaN: unknown)
    double lpFactor = 20;
    Eigen::VectorXd lambda;              // the exact spectrum when one was given
    double halfWidth = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> edges;           // Meyer
    int size() const { return static_cast<int>(members.size()); }
};

/** Rheome's `designbank` on a range: Family::MexHat · Heat · Itersine · LogItersine. Throws on a signed or empty range. */
GraphFilterbank design(Family family, const Range& range, const DesignOptions& options = {});

/** Meyer's auxiliary ν: 0 → 1 on [0, 1], ν(x) + ν(1 − x) = 1, four derivatives flat at the ends. */
double meyerNu(double x);
/** A band with Meyer edges: sin of the rise at `lo`, cos of the fall at `hi`, each over ±halfWidth octaves. lo ≤ 0: no rise; hi = ∞: no fall. */
double meyerBand(double x, double lo, double hi, double halfWidth = 1.0 / 3.0);
/** A band's MORLET gain: a Gaussian in frequency, UNIT peak, centred on the
 *  band's ARITHMETIC centre (lo + hi)/2, the band's width hi − lo its FULL WIDTH AT HALF MAXIMUM, so σ = (hi − lo)/(2√(2 ln 2)). Even in x (the real band; the analytic signal's
 *  one-sided gain is twice it, as the taps route's odd taps give). Not a tight frame: neighbouring bands overlap and do not add up (Meyer's do). lo < hi required. */
double morletBand(double x, double lo, double hi);
/** The MEXICAN HAT of a spatial level: g(λ) = λs²·e^(−λs²/2) / max with s² = 2/centre, i.e. g = (λ/c)·e^(1 − λ/c): UNIT peak at λ = `centre`
 *  (> 0), 0 at λ = 0, a long tail e^(−λ/c) (no compact support: neighbouring levels overlap and do not add up; Meyer's do). It is the same shape as the bank's MexHat family tλe^(−tλ), t = 1/c, normalised. */
double mexicanHat(double lambda, double centre);
/** The λ above the peak where the hat's gain falls to `floor` (default 1e-3): how far the modes must reach for the eigen route to stand for the hat within `floor`. */
double mexicanHatReach(double centre, double floor = 1e-3);
/** The Studio's / nxr's DYADIC (Meyer) frame over ascending edges: scaling [0,e₀], a wavelet per [e_{k−1},e_k], top [e_last,∞); Σ g² = 1. */
GraphFilterbank dyadic(const std::vector<double>& edges, double halfWidth = 1.0 / 3.0);
/** The surface's frame by Weyl's law: edges 4π·2^j / area, j = 0 … levels − 1 (`area` in the units of 1/λ). */
GraphFilterbank weyl(double area, int levels, double halfWidth = 1.0 / 3.0);
/** A basis Weyl does not fit (the relative Dirac): edges 2^j + ½ on the RANK coordinate (`rankCoordinates`), never on λ. */
GraphFilterbank rank(int levels, double halfWidth = 1.0 / 3.0);
/** Each mode's pattern index (1-based) among |λ|, counted in groups of `mult` (4: a quaternion quadruplet), in the modes' own order. */
Eigen::VectorXd rankCoordinates(const Eigen::VectorXd& lambdas, int mult = 1);

/** A bank DESCRIBED IN DATA (by a binding, a definition): the family by name and the settings it takes.
 *    "mexhat" · "heat" · "itersine" · "logitersine"  → design(family, range, options)         (range: lmax, or lmin and lmax, or the spectrum lambda)
 *    "meyer"                                          → dyadic(edges, halfWidth)               (edges ascending, in λ)
 *    "weyl"                                           → weyl(area, levels, halfWidth)          (one octave a level: edges 4π·2^j / area)
 *    "rank"                                           → rank(levels, halfWidth)
 *  Every other name is refused, naming the valid ones; a setting the family needs and lacks is refused by the design function. */
struct NamedBank {
    std::string family;
    Range range;                         // mexhat · heat · itersine · logitersine
    DesignOptions options;               // the same four
    std::vector<double> edges;           // meyer
    double area = 0;                     // weyl, in the units of 1/λ
    int levels = 0;                      // weyl · rank
    double halfWidth = 1.0 / 3.0;        // meyer · weyl · rank: the transition half-width in octaves
};
GraphFilterbank graphBank(const NamedBank& spec);
/** "mexhat" | "heat" | "itersine" | "logitersine" → Family; throws naming the valid ones. */
Family familyOf(const std::string& name);

// ── the frame ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Every member's gain at each λ → [K × M]. */
Eigen::MatrixXd gains(const GraphFilterbank& bank, const Eigen::VectorXd& lambda);
/** The canonical dual's gains g_m / S → [K × M]; 0 where S ≤ 1e-12·max S (uncovered). */
Eigen::MatrixXd dualGains(const GraphFilterbank& bank, const Eigen::VectorXd& lambda);

struct Bounds {
    double A = 0, B = 0;       // min and max of S(λ) = Σ g²
    double tightness = 0;      // B / max(A, eps); 1 for a tight frame
    double uncovered = 0;      // fraction of the λ with S ≈ 0
};
/** Frame bounds on `lambda` (the spectrum the transform will see). */
Bounds framebounds(const GraphFilterbank& bank, const Eigen::VectorXd& lambda);
/** Frame bounds on the bank's own spectrum if it has one, else on 512 points of [0, λmax] (the continuum's, not the transform's). */
Bounds framebounds(const GraphFilterbank& bank);
/** |B/A − 1| ≤ tol (Rheome's isframetight, tol 1e-6). */
bool isframetight(const GraphFilterbank& bank, double tol = 1e-6);

/** Member descriptors (Rheome: scales, widths = √(2t), centerWavenumbers, MassLost, Usable). NaN where a family has none. */
Eigen::VectorXd scales(const GraphFilterbank& bank);
Eigen::VectorXd widths(const GraphFilterbank& bank);
Eigen::VectorXd centerWavenumbers(const GraphFilterbank& bank);   // gain-weighted centroid of √λ over [λmin, λmax], 512 points
/** Fraction of each member's spectral mass beyond λmax (the coi analogue): mexhat (1+u)e^(−u), u = tλmax; heat e^(−u); itersine 0. */
Eigen::VectorXd massLost(const GraphFilterbank& bank);
inline bool usable(double massLost) { return massLost <= 0.05; }

// ── coefficients and synthesis weights (the transforms are gwt / igwt: nxr/signal/graphwavelet.h) ──

/** One [n × nT] block per member. Trailing time/trial columns pass through untouched. */
using Coefficients = std::vector<Eigen::MatrixXd>;
using CoefficientsC = std::vector<Eigen::MatrixXcd>;
/** Synthesis weights: the canonical Dual g/S (exact wherever the bank covers) or the Tight g (the frame operator: A·f). */
enum class Synthesis { Dual, Tight };

// ── a bank straight from an operator (Rheome's graphfilterbank.fromOperator) ──

/** A bank and the Chebyshev transform built with it. */
struct OperatorBank {
    GraphFilterbank bank;
    graphtransforms::ChebyshevOperator transform;
};
/** Build a bank straight from the operator: λmax by Lanczos (one eigenvalue, no decomposition; the series' bound, with the margin ChebyshevOperator keeps), the family designed on [0, λmax],
 *  and the Chebyshev transform of the pencil (L, mass) of degree `order`: GSPBox's ergonomics, and the safe path (the polynomial's λmax bounds the operator's spectrum by construction).
 *  Family::Itersine and the others as `design`; throws as design and ChebyshevOperator do. */
OperatorBank fromOperator(const Eigen::SparseMatrix<double>& L, Family family, const Eigen::SparseMatrix<double>& mass = {}, int order = 40, const DesignOptions& options = {});

// THE ENERGIES (scaleEnergy, memberEnergy, bandMoments, bandPower) are measurements: nxr/measurement/band_energy.h.

}  // namespace nxr::signal::graphfilterbank

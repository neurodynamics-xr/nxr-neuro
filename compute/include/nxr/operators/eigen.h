#pragma once
// eigensolvers and the Poisson solver (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"

namespace nxr::operators {
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;
} // namespace nxr::operators

namespace nxr::operators {

// ── Eigensolver ──────────────────────────────────────────────

struct EigenResult {
    Eigen::MatrixXd eigenvectors;  // [nV, k] columns are eigenvectors
    Eigen::VectorXd eigenvalues;   // [k] sorted ascending (hodgeDiracModes with bothSigns: ascending in signed μ)
    int k;
    int nConverged;
    Eigen::VectorXi kind;          // Hodge–Dirac only: per mode 0 exact · 1 harmonic · 2 coexact (empty otherwise)
};

/**
 * THE DEFAULT SHIFT: none. `kShiftLadder` (NaN) means "solve by the scale-aware SHIFT LADDER" (`eigenSmallest`:
 * σ = −1e-7·λmax, then −1e-4·λmax, then a Cholesky-mode last resort), never a fixed shift on the null space.
 *
 * The old default, a FIXED σ = −1e-8, sits on the operator's null space (the Laplacian's constant, the relative Dirac's four
 * constant quaternions) whatever the spectrum's scale: K − σM is nearly singular and the shift-invert Lanczos returns the low
 * spectrum wrong. Measured 2026-09-29 on a real cortex hemisphere (10 242 vertices): the cotan Laplacian's λ₃ came back 337.4
 * for 317.2, low modes mixed, every mode's residual 0.4–1.6; the relative Dirac's first vector per quartet at 2e-4 — while
 * the ladder gives residuals ~1e-11 on the same pencils. An EXPLICIT σ is still honoured as a fixed shift. Every binding
 * (MEX, Node, wasm) defaults to this.
 */
inline constexpr double kShiftLadder = std::numeric_limits<double>::quiet_NaN();

/**
 * Controls of the shift-invert Lanczos run (an explicit σ). The defaults are the library's long-standing ones (ncv = 2k+1, 1000 restarts,
 * tol 1e-10, a partial result accepted); the Hodge–Dirac reference solve asks for machine precision and for every pair (requireAll).
 */
struct EigenControls {
    int ncv = 0;               // Krylov dimension; 0 = min(max(2k+1, 20), N) (SciPy's eigsh default)
    int maxit = 1000;          // restarts
    double tol = 1e-12;        // Spectra's relative tolerance on the Ritz estimates (ARPACK's tol = 0 means machine precision)
    bool requireAll = false;   // throw EigensolveNotConverged unless all k pairs converged (ARPACK raises as well)
};

/**
 * Solve the generalized eigenvalue problem K φ = λ M φ for k eigenpairs (ascending in λ).
 * Matches MATLAB: eigs(K, M, k, 'SM') for the default shift.
 *
 * @param sigma      Shift. `kShiftLadder` (NaN, the default) = the scale-aware shift ladder, see above. A finite value is a
 *                   FIXED shift-invert centre: the k eigenvalues nearest σ are returned. (K − σM) is factored once and each
 *                   IRAM iteration is a forward/back substitution, so a shift near the wanted end is much faster than
 *                   standard mode on large sparse systems. Never put a fixed σ on the null space.
 * @param normalize  If true, M-orthonormalize eigenvectors so U' * M * U = I (Cholesky whitening). Default false: the
 *                   vectors are returned as Spectra produces them. Set true for downstream spectral synthesis.
 * @param removeDC   If true, strip the null space (every mode with |λ| within a relative tolerance of 0, see `removeDC`).
 *                   Default false.
 * @param cancel     Optional cancellation token, polled once per Spectra perform_op; on true, throws Error(Cancelled).
 * @param progress   Optional observer of solver progress.
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N - 1.
 * Throws Error(Cancelled)          on cancel.requested().
 * Throws Error(EigensolveNotConverged) on Spectra failure with no partial result.
 */
EigenResult eigen(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma                       = kShiftLadder,
    bool normalize                     = false,
    bool removeDC                      = false,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {},
    const EigenControls& controls      = {}
);

/**
 * The k smallest generalized eigenpairs of a pre-assembled symmetric pencil
 * (A, B) with B SPD, ported from Brainstorm's bst_eigs_smallest: estimate the
 * spectrum scale lmax (largest-magnitude, Cholesky mode on B), then a relative
 * shift ladder — σ=-1e-7·lmax → σ=-1e-4·lmax → last-resort Cholesky-mode
 * SmallestMagn. Returns M-orthonormal (if normalize), ascending. Reuses eigen()
 * per shift rung; the last resort factors only the SPD B (never the singular A).
 *
 * Throws Error(EigensolveInvalidK) on k < 1 or k > N-1.
 * Throws Error(EigensolveNotConverged) only if every rung fails.
 */
EigenResult eigenSmallest(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B,
    int k,
    bool normalize                     = true,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {}
);

/**
 * M-orthonormalize an eigenvector basis: U' * M * U = I.
 * Cholesky-whitening fast path with eigen-decomposition fallback.
 * Typically called via `eigen(..., normalize=true)`; this standalone
 * entry point is for transforming pre-computed bases (e.g. arriving
 * from MATLAB / Python).
 */
Eigen::MatrixXd normalize(
    const Eigen::MatrixXd& U,
    const Eigen::SparseMatrix<double>& M
);

/**
 * Strip the null space from an EigenResult: EVERY mode with |λ| <= relTol · max|λ| of the result (default 1e-8), not just one.
 * The null space is one constant per connected component for a Laplacian (two hemispheres: two), and four constant quaternions
 * for the relative Dirac. If no mode is that small (a window away from zero) nothing is removed. `kind`, when present, is kept
 * in step. `*nRemoved` (optional) receives how many modes were removed.
 * Caveat: the tolerance is relative to the largest |λ| in the result, so a result holding only near-zero values keeps them.
 * Typically called via `eigen(..., removeDC=true)`; this standalone entry point strips a pre-computed EigenResult.
 */
EigenResult removeDC(const EigenResult& result, int* nRemoved = nullptr, double relTol = 1e-8);


// ── Named-operator eigensolve ────────────────────────────────
//
// Single source of truth for "named operator → generalized eigenproblem".
// Each operator has a NATURAL generalized mass (the metric its eigenproblem is
// posed against); that pairing is library knowledge, not something every
// consumer should re-derive. `eigenProblemFor` centralizes it; `eigenOperator`
// assembles + eigensolves (+ optional ℍ-multiplet reconstruction / dense path).
// Design: docs/archive/compute/superpowers/specs/2026-06-15-operator-eigensolve-design.md.

enum class EigenOperator {
    LaplacianCotan,   // (cotanL, mass)
    LaplacianGraph,   // (graphL, I)           — standard eigenproblem
    Dirac,            // (dirac(τ) 4V, M_galerkin ⊗ I₄)   — vertex-interleaved 4v+c
    DiracFace,        // (diracFace(τ) 4F, diag(faceArea) ⊗ I₄)
    HodgeDirac,       // (d+δ on 0/1/2-forms of the intrinsic Delaunay mesh, diag(⋆0,⋆1,⋆2)), blockSize 1, N = V+E+F
    SpinExtrinsic,    // spin Diracs on face spinors, solved on vertex spinors (4V) by Ye et al.'s quotient
    SpinIntrinsicHY,
    SpinIntrinsicYe
    // Covariant (3N) deferred: it is component-major (c·N+v, mass = kron(I₃, M),
    // a different expansion than the Dirac's interleaved kron(M, I₄)) and its
    // ambient spectrum is just the triply-degenerate Laplace–Beltrami spectrum.
};

struct EigenOperatorSpec {
    EigenOperator op;
    double tau = 0.0;                                       // Dirac / DiracFace blend ∈ [0,1]
    operators::MassMatrixVariant mass = nxr::operators::MassMatrixVariant::MixedVoronoi;  // vertex-domain mass choice
    bool graphDegreeMass = false;   // LaplacianGraph: mass = diag(degree) (L φ = λ D φ) instead of identity
    double star1Floor = 0.0;        // HodgeDirac: floor of ⋆1, 0 = off (default, as nxr)
    double mollify    = 1e-6;       // HodgeDirac: the mollification margin relative to the mean edge length (shared name `mollify`)
    double flipEps    = 1e-6;       // HodgeDirac: an edge is flipped while its cotan weight is below −flipEps (shared name `flipEps`)
    bool bothSigns    = false;      // HodgeDirac, default shift: also return the −μ mirror of every +μ mode
};

struct EigenProblem {
    Eigen::SparseMatrix<double> K;   // the operator
    Eigen::SparseMatrix<double> M;   // its natural generalized mass (block-expanded)
    double sigma;                    // shift for shift-invert
    int blockSize;                   // 1 scalar / 3 covariant / 4 Dirac
};

// Build the (K, M, σ, blockSize) generalized eigenproblem for a named operator.
EigenProblem eigenProblemFor(const geometry::SurfaceArrays& surface, const EigenOperatorSpec& spec);

// blockKron(M, b): expand a [n×n] sparse to [bn×bn] block-diagonal (kron(M, I_b),
// entry (i,j,v) → (b·i+c, b·j+c, v) for c=0..b−1). Exposed for callers that build
// their own block masses; used internally by eigenProblemFor.
Eigen::SparseMatrix<double> blockKron(const Eigen::SparseMatrix<double>& M, int b);

// Assemble + eigensolve a named operator. With reconstructMultiplets and a
// blockSize-4 (Dirac) operator, the eigenspaces are closed under right-quaternion
// multiplication to recover exact 4-fold multiplets (Spectra miscounts the
// degenerate clusters). With dense=true, uses Eigen's dense generalized solver
// (exact eigenvalues + multiplicities; small meshes / verification only — throws
// EigensolveInvalidK if the dense size exceeds the cap unless the mesh is small).
EigenResult eigenOperator(
    const geometry::SurfaceArrays& surface,
    const EigenOperatorSpec& spec,
    int k,
    double sigma                       = kShiftLadder,
    bool normalize                     = true,
    bool reconstructMultiplets         = false,
    bool dense                         = false,
    const CancellationToken& cancel    = {},
    const ProgressObserver& progress   = {}
);

// Spectrum slicing: the `count` lowest eigenvalues of a symmetric pencil (A SPSD, B SPD) in windows of `slice`
// shift-invert solves, so thousands of modes need only a (slice)-column Krylov basis. The first window is the ladder
// solve (eigenSmallest); each next window is centred at the top of the covered range and keeps the eigenvalues above it.
// Throws EigensolveInvalidK when count ≥ N.
// eigenLargestMagn: the k largest-|λ| generalized eigenpairs (Cholesky mode, B SPD, A symmetric) — the inverse-spectrum solve of the spin quotient.
EigenResult eigenLargestMagn(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, int k, bool normalize = true);
EigenResult eigenSliced(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                        int count, int slice = 400, bool normalize = true,
                        const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

// Hodge–Dirac mode kinds from M-orthonormal pencil eigenvectors X (columns): 0 exact (0-form block dominates),
// 2 coexact (2-form block dominates), 1 harmonic (|μ| ≤ tol·max|μ|).
Eigen::VectorXi classifyHodgeModes(const operators::hodgedirac::HodgeDirac& h, const Eigen::MatrixXd& X,
                                   const Eigen::VectorXd& mu, double tol = 1e-8);

// ── The Hodge–Dirac mode solve: the REFERENCE method, and its named options ──
// The reference is the method of nxr (`nxr.eigen.hodge_dirac.solve_hodge`, `nxr.pipelines.hodge.solved_package`) and of Rheome
// (`rheome.eigen.hodge_dirac`, `hodge_package`); the maintainers' decision of 2026-11-10 is that nxr-neuro uses exactly it, by default:
//   * two ladders: exact Δ0 = d0ᵀ⋆1 d0 against ⋆0, coexact Δ2 = (⋆2d1) ⋆1⁻¹ (⋆2d1)ᵀ against ⋆2;
//   * want = count/2 + 1; each ladder asks for min(want + 1, n − 2) pairs (n = nV for Δ0, nF for Δ2);
//   * ONE shift-invert solve per ladder at the absolute σ = −1e-3 (generalised symmetric), machine-precision tolerance, no
//     re-orthonormalisation afterwards; ascending, λ = max(λ, 0);
//   * the constants dropped per ladder (λ > 1e-6 · max λ of the ladder), then ONE λ cut: top = min(max kept λ0, max kept λ2),
//     both ladders truncated at it; μ = √λ; merged by μ (stable: exact before coexact on a tie); kinds 0 exact · 2 coexact;
//     modes with μ ≤ 1e-6 · max μ left out.
// The methods this library used before stay as named options (never defaults):
//   shift     Absolute (σ, default −1e-3)  |  RelativeLadder (eigenSmallest: σ = −1e-7·λmax, then −1e-4·λmax, then a Cholesky last resort)
//   slice     0 = none (one solve per ladder)  |  a window size (spectrum slicing, eigenSliced; the windows use their own shifts)
//   normalize false  |  true (Cholesky whitening: M-orthonormalise afterwards)
// The options are recorded in the stored result's provenance (nxr-pipelines), and a different option is a different cached result.
enum class HodgeShift { Absolute, RelativeLadder };

struct HodgeSolveOptions {
    HodgeShift shift = HodgeShift::Absolute;
    double sigma = -1e-3;      // Absolute: the shift (absolute, not scaled by the spectrum)
    int slice = 0;             // 0 = none
    bool normalize = false;
    double tol = 1e-14;        // Absolute, no slice: Spectra's tolerance (ARPACK's tol = 0 is machine precision; Spectra measures it relative to the Ritz value)
    int ncv = 0;               // Absolute, no slice: Krylov dimension, 0 = min(max(2k+1, 20), n) (SciPy's eigsh default)
    int maxit = 100000;        // Absolute, no slice: restarts allowed (generous: a large k converges in one solve)
    int limit = 0;             // hodgeDiracModes only: keep at most this many of the lowest modes (0 = all under the λ cut, about `count`, as the reference)
};

// One ladder: the k smallest eigenpairs of (S, M), ascending, λ clamped to ≥ 0, by the options' method.
EigenResult hodgeLadder(const Eigen::SparseMatrix<double>& S, const Eigen::SparseMatrix<double>& M, int k, const HodgeSolveOptions& opt = {},
                        const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

struct HodgeLadders {
    EigenResult exact, coexact;        // the whole ladders as solved (ψ [nV×k0], χ [nF×k2]; λ ascending)
    double top = 0;                    // the single λ cut
    struct Mode { double mu; int kind; int col; };   // kind 0 exact · 2 coexact; col indexes the ladder's columns
    std::vector<Mode> modes;           // the kept modes, merged by μ ascending (the constants and the μ ≤ 1e-6·max out)
    int nExact = 0, nCoexact = 0;      // how many of `modes` are of each kind
};

// The reference split: both ladders (`count` as above), the null cut, the single λ cut and the merge. S2 = (⋆2d1)⋆1⁻¹(⋆2d1)ᵀ.
HodgeLadders hodgeLadders(const Eigen::SparseMatrix<double>& L0, const Eigen::SparseMatrix<double>& M0,
                          const Eigen::SparseMatrix<double>& S2, const Eigen::SparseMatrix<double>& M2,
                          int count, const HodgeSolveOptions& opt = {},
                          const CancellationToken& cancel = {}, const ProgressObserver& progress = {});

// The Hodge–Dirac eigenpairs with POSITIVE μ by the exact/coexact split (no indefinite solve), the reference method above:
//   exact    (ψ, dψ/μ, 0)/√2     from Δ0 ψ = λ ψ  (d0ᵀ⋆1 d0, mass ⋆0),                  μ = √λ
//   coexact  (0, δχ/μ, χ)/√2     from Δ2 χ = λ χ  (⋆2d1 ⋆1⁻¹ (⋆2d1)ᵀ, mass ⋆2),         μ = √λ
// merged by μ; the constant modes (μ = 0) are dropped, harmonic 1-forms are NOT produced (use the pencil solve for them).
// The result holds ABOUT `count` modes (all under the single λ cut; options.limit keeps at most that many of the lowest).
// bothSigns adds the −μ mirror (1-form block negated). Result: eigenvalues μ ascending (signed if bothSigns:
// −μ_max … −μ_min, then +μ_min … +μ_max), eigenvectors [N×k] (M-orthonormal when the solve is), kind per column.
EigenResult hodgeDiracModes(const operators::hodgedirac::HodgeDirac& h, int count, bool bothSigns = false,
                            const HodgeSolveOptions& options = {}, const CancellationToken& cancel = {},
                            const ProgressObserver& progress = {});

// Close each eigenvalue cluster of a 4-interleaved quaternion eigenbasis under right-ℍ multiplication and return an
// M-orthonormal basis of the true (multiple-of-4) eigenspaces; recovers multiplets Lanczos under-counts. M is the inner product.
EigenResult reconstructDiracMultiplets(const EigenResult& r, const Eigen::SparseMatrix<double>& M);

// The k spin-Dirac eigenpairs nearest 0 as Ye et al. (dirac_eigen.m): on vertex spinors ψ, AᵀDM⁻¹DAψ = μ AᵀDAψ,
// solved as AᵀDAψ = ν (AᵀDM⁻¹DA + ε AᵀMA) ψ for the k largest |ν|, μ = 1/ν. M = area ⊗ I₄. Returns μ ascending (signed),
// ψ [4V×k] normalised to ψᵀ AᵀMA ψ = 1. A = covariant vertexToFace for the intrinsic kinds, plain for the extrinsic.
EigenResult eigenSpin(const geometry::SurfaceArrays& surface, operators::spin::Kind kind, int k,
                      const CancellationToken& cancel = {}, const ProgressObserver& progress = {});


// ── Poisson Solver ───────────────────────────────────────────

/**
 * Solve the scalar Poisson equation: K * φ = -M * (ρ - ρ̄)
 * where ρ̄ is the M-weighted mean of ρ.
 *
 * Solves with the caller's factor of (K + ε·I) (operators::laplacianFactor(K)), so
 * repeated calls with the same factor skip the factorization step.
 *
 * Agnostic in K and M: works on cotangent Laplacians + lumped mass
 * (mesh case), graph Laplacians + node-weight diagonals (graph case),
 * or any SPD pair. The total mass `Σᵢⱼ Mᵢⱼ` plays the role of the
 * "totalArea" normalization for the weighted mean.
 *
 * @param K           SPD stiffness matrix (Laplacian)
 * @param M           SPD mass / inner-product matrix (typically diagonal)
 * @param laplacianFactor  the factor of (K + εI), made once by the caller
 * @param densityMap  sparse density: index → density value
 * @return            harmonic potential φ
 */
Eigen::VectorXd poisson(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    const nxr::operators::CholeskyFactor& laplacianFactor,
    const std::map<int, double>& densityMap
);

/** Convenience overload: forwards to (K, M) using ops.cotanLaplacian, ops.mass. */
inline Eigen::VectorXd poisson(
    const operators::ManifoldOperators& ops,
    const nxr::operators::CholeskyFactor& laplacianFactor,
    const std::map<int, double>& densityMap
) {
    return poisson(ops.cotanLaplacian, ops.mass, laplacianFactor, densityMap);
}
} // namespace nxr::operators
#include "nxr/operators/laplacians.h"

namespace nxr::operators {
struct Eigenpairs {
  Eigen::VectorXd values;        // [k] ascending
  Eigen::MatrixXcd vectors;      // [n x k], column i the i-th eigenvector (unit norm, its largest component real and positive)
  int converged = 0;             // how many the iterative solver converged (k for the dense oracle)
};

/** the lowest k eigenpairs: shift-invert at `sigma` (below the spectrum: L − σI positive definite for a PSD L), `tol` the solver's relative tolerance.
 *  Throws nxr::core::Error (EigensolveInvalidK) for k outside [1, n − 1], (EigensolveNotConverged) when fewer than k converge. */
Eigenpairs hermitianEigs(const Hermitian& L, int k, double sigma = -1e-6, double tol = 1e-10);
/** the dense oracle (n ≤ 4096) */
Eigenpairs hermitianEigsDense(const Hermitian& L, int k);
}  // namespace nxr::operators

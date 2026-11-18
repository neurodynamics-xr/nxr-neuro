#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseCholesky.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <limits>

namespace nxr::operators {

static EigenResult polishPairs(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, EigenResult r, double sigma);

namespace {

// ── CancelProgressOp — wraps a Spectra OpType to add cancel/progress ──
//
// Spectra's iteration calls `op.perform_op(x_in, y_out)` once per inner
// Lanczos step. Wrapping the OpType lets us intercept every step and:
//   1. Check the cancellation token. If set, throw Error(Cancelled);
//      Spectra's compute() unwinds cleanly because all its state is
//      Eigen members (RAII, no manual allocation).
//   2. Increment a progress counter and report it to the observer.
//
// The wrapper inherits the Inner type's interface (rows/cols/Scalar
// typedef, plus set_shift for shift-invert ops) and forwards
// perform_op to the inner. SymShiftInvert is the AOp here; SparseSymMatProd
// is the BOp and isn't wrapped (one wrap is enough — both ops fire each
// iteration so AOp gets the cancel/progress signal).
template <typename Inner>
class CancelProgressOp {
public:
    using Scalar = typename Inner::Scalar;

    CancelProgressOp(Inner& inner,
                     const CancellationToken& cancel,
                     const ProgressObserver& progress,
                     int totalEstimate)
        : inner_(inner), cancel_(cancel), progress_(progress),
          totalEstimate_(totalEstimate) {}

    Eigen::Index rows() const { return inner_.rows(); }
    Eigen::Index cols() const { return inner_.cols(); }

    // Required by SymShiftInvert (the shift-invert AOp specifically).
    void set_shift(const Scalar& sigma) { inner_.set_shift(sigma); }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        if (cancel_.requested()) {
            throw Error(ErrorCode::Cancelled,
                "Eigensolve cancelled by request");
        }
        inner_.perform_op(x_in, y_out);
        ++count_;
        progress_.update(count_, totalEstimate_, 0.0);
    }

private:
    Inner&                     inner_;
    const CancellationToken&   cancel_;
    const ProgressObserver&    progress_;
    int                        totalEstimate_;
    mutable int                count_ = 0;
};

} // namespace

EigenResult eigen(
    const Eigen::SparseMatrix<double>& K,
    const Eigen::SparseMatrix<double>& M,
    int k,
    double sigma,
    bool normalize,
    bool removeDC,
    const CancellationToken& cancel,
    const ProgressObserver& progress,
    const EigenControls& ctl
) {
    /* NO SHIFT GIVEN (kShiftLadder): the scale-aware ladder, never a fixed shift on the null space (compute.h) */
    if (std::isnan(sigma)) {
        EigenResult r = eigenSmallest(K, M, k, normalize, cancel, progress);
        return removeDC ? ::nxr::operators::removeDC(r) : r;
    }
    int n = static_cast<int>(K.rows());

    // Validate inputs — caught before any expensive factorization.
    if (K.rows() != K.cols() || M.rows() != M.cols() || K.rows() != M.rows()) {
        throw Error(ErrorCode::InvalidInput,
            "eigen: K and M must be square and same size");
    }
    if (k < 1) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigen: k must be >= 1");
    }
    if (k > n - 1) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "eigen: k must be < N (matrix size)");
    }
    // K ceiling — THE WASM BUILD ONLY (2026-09-29). Spectra's Krylov
    // basis is `ncv = 2k+1` doubles per matrix row (capped at n). At
    // k=1000 on a 10k-vertex cortical mesh the basis is ≈ 160 MB; at
    // k=5000 it approaches an n×n dense matrix and the WASM linear-memory
    // cap (2 GB) is hit before convergence. Native builds (the Node addon,
    // MEX, the CLI) have the headroom and take any k < n — a Dirac basis
    // of 1600 modes (+ its 4-fold guard) is a native solve.
#ifdef __EMSCRIPTEN__
    constexpr int kMaxK = 1000;
    if (k > kMaxK) {
        throw Error(ErrorCode::EigensolveInvalidK,
            "solve: k must be <= 1000",
            "Browser/WASM Krylov basis ceiling — higher k requires "
            "a native build (addon, MEX) or a memory64 build. See "
            "docs/eigensolve-cap.md.");
    }
#endif

    // Convergence parameter: ncv must be > k and <= n.
    // Larger ncv costs more memory and per-iteration work but
    // converges in fewer iterations. 2*k+1 is Spectra's usual
    // recommendation; we clamp to n.
    int ncv = ctl.ncv > 0 ? std::min(std::max(ctl.ncv, k + 1), n) : std::min(std::max(2 * k + 1, 20), n);

    // High-water bound on perform_op invocations for the progress
    // counter denominator. Spectra runs up to `maxit` outer
    // iterations, each performing roughly `ncv` SpMVs. The bound
    // is loose by design — most solves converge in a small fraction
    // of this — and consumers display a normalized fraction.
    const int kMaxIter = ctl.maxit > 0 ? ctl.maxit : 1000;
    int totalEstimate = kMaxIter * ncv;

    auto t0 = std::chrono::steady_clock::now();

    // Cancellation point before the (potentially several-second)
    // factorization. Catches "user clicked cancel before the solve
    // started" without paying for the LLT.
    if (cancel.requested()) {
        throw Error(ErrorCode::Cancelled,
            "Eigensolve cancelled before factorization");
    }

    // ─── Shift-invert mode ──────────────────────────────────
    //
    // Generalized problem: K φ = λ M φ
    //
    // Shift-invert transforms to:
    //   (K - σM)^{-1} M φ = ν φ   where ν = 1/(λ - σ)
    //
    // Eigenvalues near σ become the LARGEST of ν — which is
    // exactly what IRAM converges to fastest. For our use case
    // (smallest λ of the Laplacian, near 0), σ ≈ 0 gives
    // dramatically faster convergence than standard mode.
    //
    // The inner factorization of (K - σM) is done once by
    // SymShiftInvert's constructor; each IRAM iteration then
    // does one triangular forward/back-solve (O(V)) instead
    // of a full SpMV (O(nnz)).

    // SymShiftInvert factors (K - σM) using Eigen's SimplicialLLT.
    // Per CONTRIBUTING.md §10, nxr-compute is constrained to Eigen + Spectra +
    // geometry-central; CHOLMOD's supernodal factorization would
    // be 3-10× faster on x86 but breaks the WASM and MEX targets,
    // so we don't pull it in.
    using ShiftInvertOp = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    ShiftInvertOp innerOp(K, M);
    CancelProgressOp<ShiftInvertOp> opShift(innerOp, cancel, progress, totalEstimate);

    // B operator for inner products: just M * x (matrix-vector product).
    // This is different from Cholesky mode where B needs to be decomposed.
    Spectra::SparseSymMatProd<double> opM(M);

    auto tFactor = std::chrono::steady_clock::now();
    double factorMs = std::chrono::duration<double, std::milli>(tFactor - t0).count();
    std::cout << "[eigensolver] Pre-factored (K - " << sigma << " M) in "
              << factorMs << " ms" << std::endl;

    Spectra::SymGEigsShiftSolver<
        CancelProgressOp<ShiftInvertOp>,
        Spectra::SparseSymMatProd<double>,
        Spectra::GEigsMode::ShiftInvert
    > solver(opShift, opM, k, ncv, sigma);

    solver.init();

    // compute() runs the IRAM loop. If the wrapped op throws
    // Error(Cancelled) mid-iteration, the exception propagates
    // out and Spectra's internal Eigen-typed members destruct
    // cleanly (no manual allocations to leak).
    int nconv = solver.compute(Spectra::SortRule::LargestMagn, kMaxIter, ctl.tol);

    auto tSolve = std::chrono::steady_clock::now();
    double solveMs = std::chrono::duration<double, std::milli>(tSolve - tFactor).count();
    double totalMs = std::chrono::duration<double, std::milli>(tSolve - t0).count();

    EigenResult result;

    if (solver.info() == Spectra::CompInfo::Successful ||
        solver.info() == Spectra::CompInfo::NotConverging) {
        // Extract results (may be partial if not all converged, unless the caller requires all)
        if (ctl.requireAll && nconv < k) {
            throw Error(ErrorCode::EigensolveNotConverged,
                "eigen: " + std::to_string(nconv) + " of " + std::to_string(k) + " eigenpairs converged (tol " + std::to_string(ctl.tol) +
                ", ncv " + std::to_string(ncv) + ", maxit " + std::to_string(kMaxIter) + ")");
        }
        Eigen::VectorXd evals = solver.eigenvalues();
        Eigen::MatrixXd evecs = solver.eigenvectors();

        // Sort ascending by eigenvalue (Spectra returns them in order of 1/(λ−σ), already ascending for σ below the spectrum:
        // then the vectors are moved, not copied)
        std::vector<int> idx(evals.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return evals(a) < evals(b);
        });
        const bool inOrder = std::is_sorted(idx.begin(), idx.end());
        if (inOrder) {
            result.eigenvalues = std::move(evals);
            result.eigenvectors = std::move(evecs);
        } else {
            result.eigenvalues.resize(evals.size());
            result.eigenvectors.resize(evecs.rows(), evecs.cols());
            for (int i = 0; i < static_cast<int>(evals.size()); i++) {
                result.eigenvalues(i) = evals(idx[i]);
                result.eigenvectors.col(i) = evecs.col(idx[i]);
            }
        }

        result.k = static_cast<int>(result.eigenvalues.size());
        result.nConverged = nconv;

        std::cout << "[eigensolver] Converged: " << nconv << "/" << k
                  << " eigenvalues in " << totalMs << " ms"
                  << " (factor: " << factorMs << " ms, IRAM: " << solveMs << " ms)"
                  << ". Range: [" << result.eigenvalues(0)
                  << ", " << result.eigenvalues(result.k - 1) << "]" << std::endl;
    } else {
        std::cerr << "[eigensolver] FAILED: Spectra returned error code "
                  << static_cast<int>(solver.info())
                  << " after " << totalMs << " ms" << std::endl;
        throw Error(ErrorCode::EigensolveNotConverged,
            "Spectra eigensolver returned a numerical-issue error code");
    }

    // polished only for a window that starts above its shift (the k smallest above sigma: K positive semi-definite, sigma below the spectrum); a window
    // around an interior shift (an indefinite operator, the Hodge-Dirac's) is left as the solver gave it
    if (result.k > 0 && sigma < result.eigenvalues(0) && result.eigenvalues(0) > -1e-6 * std::max(1.0, std::abs(result.eigenvalues(result.k - 1))))
        result = polishPairs(K, M, std::move(result), sigma);
    if (removeDC) {
        result = ::nxr::operators::removeDC(result);
    }
    if (normalize) {
        result.eigenvectors = ::nxr::operators::normalize(result.eigenvectors, M);
    }

    return result;
}

namespace {

// Coarse largest-magnitude estimate via Cholesky mode (factors only SPD B).
// Returns 0 on any failure so the caller falls through to the smallest path.
double estimateLmax(const Eigen::SparseMatrix<double>& A,
                    const Eigen::SparseMatrix<double>& B) {
    try {
        int n = static_cast<int>(A.rows());
        int ncv = std::min(std::max(2 * 1 + 1, 20), n);
        Spectra::SparseSymMatProd<double> opA(A);
        Spectra::SparseCholesky<double>   opB(B);
        Spectra::SymGEigsSolver<
            Spectra::SparseSymMatProd<double>,
            Spectra::SparseCholesky<double>,
            Spectra::GEigsMode::Cholesky> est(opA, opB, 1, ncv);
        est.init();
        est.compute(Spectra::SortRule::LargestMagn, 300, 1e-2);
        if (est.info() == Spectra::CompInfo::Successful)
            return std::abs(est.eigenvalues()(0));
    } catch (...) { /* fall through */ }
    return 0.0;
}

// Last-resort smallest-magnitude solve via Cholesky mode (no A factorization).
// Mirrors eigen()'s sort-ascending + optional M-normalize extraction.
EigenResult eigenCholesky(const Eigen::SparseMatrix<double>& A,
                          const Eigen::SparseMatrix<double>& B,
                          int k, bool normalize, Spectra::SortRule rule) {
    int n = static_cast<int>(A.rows());
    int ncv = std::min(std::max(2 * k + 1, 20), n);
    Spectra::SparseSymMatProd<double> opA(A);
    Spectra::SparseCholesky<double>   opB(B);
    Spectra::SymGEigsSolver<
        Spectra::SparseSymMatProd<double>,
        Spectra::SparseCholesky<double>,
        Spectra::GEigsMode::Cholesky> solver(opA, opB, k, ncv);
    solver.init();
    int nconv = solver.compute(rule, 1000, 1e-10);
    if (solver.info() != Spectra::CompInfo::Successful &&
        solver.info() != Spectra::CompInfo::NotConverging)
        throw Error(ErrorCode::EigensolveNotConverged,
            "eigenSmallest: Cholesky-mode last resort failed");

    Eigen::VectorXd evals = solver.eigenvalues();
    Eigen::MatrixXd evecs = solver.eigenvectors();
    std::vector<int> idx(evals.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals(a) < evals(b); });

    EigenResult r;
    r.eigenvalues.resize(evals.size());
    r.eigenvectors.resize(evecs.rows(), evecs.cols());
    for (int i = 0; i < static_cast<int>(evals.size()); i++) {
        r.eigenvalues(i) = evals(idx[i]);
        r.eigenvectors.col(i) = evecs.col(idx[i]);
    }
    r.k = static_cast<int>(evals.size());
    r.nConverged = nconv;
    if (normalize)
        r.eigenvectors = ::nxr::operators::normalize(r.eigenvectors, B);
    return r;
}

} // namespace

EigenResult eigenLargestMagn(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                             int k, bool normalize) {
    return eigenCholesky(A, B, k, normalize, Spectra::SortRule::LargestMagn);
}

// The k eigenpairs of the ladder's window, made complete: see completeLowest (below).
static EigenResult completeLowest(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, EigenResult r, double sigma,
                                  const CancellationToken& cancel);

EigenResult eigenSmallest(
    const Eigen::SparseMatrix<double>& A_in,
    const Eigen::SparseMatrix<double>& B_in,
    int k, bool normalize,
    const CancellationToken& cancel, const ProgressObserver& progress
) {
    // Symmetrize (no-op to ~1e-16 for symmetric operators; forces the Lanczos path).
    Eigen::SparseMatrix<double> A = (A_in + Eigen::SparseMatrix<double>(A_in.transpose())) * 0.5;
    Eigen::SparseMatrix<double> B = (B_in + Eigen::SparseMatrix<double>(B_in.transpose())) * 0.5;

    const double lmax = estimateLmax(A, B);
    if (std::isfinite(lmax) && lmax > 0.0) {
        const double sigmas[2] = { -1e-7 * lmax, -1e-4 * lmax };
        for (double sigma : sigmas) {
            try {
                // The window is solved, completed and polished on the solver's own vectors; `normalize` is applied once, to the final result.
                // (Normalising before the polish changed which basis of a multiple eigenvalue the polish started from, and an eigenvalue of the
                // cut multiplet came back 6.5e-8 off: the result must not depend on the flag.)
                EigenResult r = completeLowest(A, B, eigen(A, B, k, sigma, /*normalize=*/false, /*removeDC=*/false, cancel, progress), sigma, cancel);
                if (normalize) r.eigenvectors = ::nxr::operators::normalize(r.eigenvectors, B);
                return r;
            } catch (const Error& e) {
                if (e.code() != ErrorCode::EigensolveNotConverged) throw;  // k-range etc: don't retry
            }
        }
    }
    return eigenCholesky(A, B, k, normalize, Spectra::SortRule::SmallestMagn);
}

} // namespace nxr::operators
namespace nxr::operators {

namespace {
// The eigenpairs of a window that Lanczos converged: sorted by distance from sigma, the prefix whose relative residual
// ‖Ax − λBx‖ / (‖Bx‖ · max|λ|) is below 1e-8. Returns the indices and the radius they cover. The farthest pairs of a
// shift-invert window are the least converged (Spectra reports NotConverging as success), so they are not trusted.
std::vector<int> convergedPrefix(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                                 const EigenResult& w, double sigma, double& radius) {
    std::vector<int> idx(w.k);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return std::abs(w.eigenvalues(a) - sigma) < std::abs(w.eigenvalues(b) - sigma); });
    std::vector<int> good;
    radius = 0;
    const double scale = std::max(1e-300, w.eigenvalues.cwiseAbs().maxCoeff());   // the window's spectral scale (λ ≈ 0 must not blow the ratio up)
    for (int c : idx) {
        const Eigen::VectorXd x = w.eigenvectors.col(c);
        const Eigen::VectorXd ax = A * x, bx = B * x;
        const double lam = w.eigenvalues(c);
        const double res = (ax - lam * bx).norm() / (bx.norm() * scale + 1e-300);
        if (res > 1e-8) break;
        good.push_back(c);
        radius = std::abs(lam - sigma);
    }
    return good;
}
} // namespace

namespace {
// (K − σM)^{-1}M with the already-held vectors projected out of the result: y ← y − X Xᵀ M y (X M-orthonormal). The vectors of X
// become eigenvalue 0 of the operator (ν = 0, i.e. λ = ∞), so a Lanczos run on it returns what is *left* near σ. The shift is
// refactored only when it changes, so several deflated runs at one shift share one factorisation.
class DeflatedShiftInvertOp {
public:
    using Scalar = double;
    using Inner = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    DeflatedShiftInvertOp(Inner& inner, const Eigen::SparseMatrix<double>& M, const CancellationToken& cancel)
        : inner_(inner), M_(M), cancel_(cancel) {}
    void setDeflation(const Eigen::MatrixXd* X) { X_ = X; }
    Eigen::Index rows() const { return inner_.rows(); }
    Eigen::Index cols() const { return inner_.cols(); }
    void set_shift(const double& sigma) {
        if (!haveShift_ || sigma != shift_) { inner_.set_shift(sigma); shift_ = sigma; haveShift_ = true; }
    }
    void perform_op(const double* x_in, double* y_out) const {
        if (cancel_.requested()) throw Error(ErrorCode::Cancelled, "Eigensolve cancelled by request");
        inner_.perform_op(x_in, y_out);
        if (X_ && X_->cols() > 0) {
            Eigen::Map<Eigen::VectorXd> y(y_out, rows());
            const Eigen::VectorXd t = X_->transpose() * (M_ * y);
            y -= (*X_) * t;
        }
    }
private:
    Inner& inner_;
    const Eigen::SparseMatrix<double>& M_;
    const CancellationToken& cancel_;
    const Eigen::MatrixXd* X_ = nullptr;
    double shift_ = 0; bool haveShift_ = false;
};
} // namespace

// COMPLETENESS OF THE SMALLEST k. A single-vector Lanczos run finds ONE vector of a multiple eigenvalue in exact arithmetic (a sphere's 2l+1, any
// symmetry, a diagonal operator with repeats): the others appear only through rounding, and some never do, so the window can hold an eigenvalue
// above one it missed (an l = 5 value in place of the last l = 4 one). Sylvester's law of inertia counts them: the number of negative pivots of
// the LDL' factorisation of K − τM is the number of eigenvalues below τ (M positive definite). With τ just under the largest value held, the held
// values below τ must be all of the spectrum below τ; if not, the missing ones are searched for with the held vectors projected out of the
// operator (DeflatedShiftInvertOp), at the same shift, and merged, until the count is right. Deterministic: no random start decides what is found.
// When the factorisation cannot be trusted (a zero or non-finite pivot) the window is returned as it is.
static EigenResult completeCore(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, EigenResult r, double sigma,
                                const CancellationToken& cancel) {
    const int n = static_cast<int>(A.rows());
    const int k = r.k;
    if (k < 1 || k >= n - 1) return r;
    using Inner = DeflatedShiftInvertOp::Inner;
    for (int round = 0; round < 25; ++round) {
        const double lk = r.eigenvalues(k - 1);
        const double tau = lk - 1e-6 * std::max(1.0, std::abs(lk));
        int foundBelow = 0;
        for (int i = 0; i < r.k; ++i) if (r.eigenvalues(i) < tau) ++foundBelow;
        Eigen::SparseMatrix<double> S = A - tau * B;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.compute(S);
        if (ldlt.info() != Eigen::Success) return r;
        const Eigen::VectorXd D = ldlt.vectorD();
        int below = 0;
        for (Eigen::Index i = 0; i < D.size(); ++i) {
            if (!std::isfinite(D(i)) || D(i) == 0.0) return r;
            if (D(i) < 0.0) ++below;
        }
        const int missing = below - foundBelow;
        if (missing <= 0) return r;

        Eigen::MatrixXd X = ::nxr::operators::normalize(r.eigenvectors, B);
        Inner inner(A, B);
        DeflatedShiftInvertOp op(inner, B, cancel);
        op.setDeflation(&X);
        const int k2 = std::min(missing + 4, n - 1 - r.k);
        if (k2 < 1) return r;
        const int ncv = std::min(std::max(2 * k2 + 1, 20), n);
        Spectra::SparseSymMatProd<double> opM(B);
        Spectra::SymGEigsShiftSolver<DeflatedShiftInvertOp, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> solver(op, opM, k2, ncv, sigma);
        solver.init();
        solver.compute(Spectra::SortRule::LargestMagn, 1000, EigenControls().tol);
        if (solver.info() != Spectra::CompInfo::Successful && solver.info() != Spectra::CompInfo::NotConverging) return r;
        const Eigen::VectorXd lam = solver.eigenvalues();
        const Eigen::MatrixXd vec = ::nxr::operators::normalize(solver.eigenvectors(), B);

        std::vector<double> L(r.eigenvalues.data(), r.eigenvalues.data() + r.k);
        std::vector<Eigen::VectorXd> V;
        for (int c = 0; c < r.k; ++c) V.push_back(r.eigenvectors.col(c));
        Eigen::MatrixXd held = X;
        int added = 0;
        for (int c = 0; c < vec.cols(); ++c) {
            Eigen::VectorXd x = vec.col(c);
            Eigen::VectorXd bx = B * x;
            for (int pass = 0; pass < 2; ++pass) {
                const Eigen::VectorXd coef = held.transpose() * bx;
                x -= held * coef;
                bx = B * x;
            }
            const double nrm = std::sqrt(std::max(0.0, x.dot(bx)));
            if (nrm < 0.1) continue;                      // already spanned by the held vectors
            x /= nrm;
            held.conservativeResize(Eigen::NoChange, held.cols() + 1);
            held.col(held.cols() - 1) = x;
            L.push_back(lam(c)); V.push_back(x); ++added;
        }
        if (added == 0) return r;
        std::vector<int> ord(L.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return L[a] < L[b]; });
        EigenResult out;
        out.k = k; out.nConverged = k;
        out.eigenvalues.resize(k);
        out.eigenvectors.resize(n, k);
        for (int c = 0; c < k; ++c) { out.eigenvalues(c) = L[ord[c]]; out.eigenvectors.col(c) = V[ord[c]]; }
        r = std::move(out);
    }
    return r;
}


// Rayleigh quotients of the members of a multiple eigenvalue agree to rounding and can come in any order: the result is returned ascending.
static EigenResult sortAscending(EigenResult r) {
    std::vector<int> ord(r.k);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return r.eigenvalues(a) < r.eigenvalues(b); });
    Eigen::VectorXd l(r.k);
    Eigen::MatrixXd V(r.eigenvectors.rows(), r.k);
    for (int c = 0; c < r.k; ++c) { l(c) = r.eigenvalues(ord[c]); V.col(c) = r.eigenvectors.col(ord[c]); }
    r.eigenvalues = std::move(l); r.eigenvectors = std::move(V);
    return r;
}

// POLISH. A member of a multiple eigenvalue converges through rounding, so its Ritz estimate is not the residual that Spectra tests: its eigenvalue can be
// off by 1e-9 where the others are exact. A few steps of subspace iteration with Rayleigh–Ritz (one factorisation at the shift, the held vectors and their
// shift-inverted images as the trial space) bring every pair to the tolerance. The pairs' order and the eigenvectors' scale are kept (M-normalised
// vectors are returned as M-normalised, the solver's own as they were).
static EigenResult polishPairs(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, EigenResult r, double sigma) {
    const int n = static_cast<int>(A.rows()), k = r.k;
    if (k < 1 || 2 * k > n) return r;
    using Inner = DeflatedShiftInvertOp::Inner;
    Inner inner(A, B);
    inner.set_shift(sigma);
    Eigen::MatrixXd X = ::nxr::operators::normalize(r.eigenvectors, B);
    Eigen::VectorXd lam = r.eigenvalues, bestLam = lam;
    Eigen::MatrixXd bestX = X;
    double bestResid = std::numeric_limits<double>::infinity();
    int worse = 0;
    {   // the pairs as held are a candidate too: a step that only adds rounding must not replace them
        Eigen::VectorXd l0(k);
        for (int j = 0; j < k; ++j) l0(j) = X.col(j).dot(A * X.col(j));
        const Eigen::MatrixXd R0 = A * X - B * X * l0.asDiagonal();
        bestResid = R0.colwise().norm().maxCoeff() / std::max(1.0, l0.cwiseAbs().maxCoeff());
        bestLam = l0;
        if (bestResid < 1e-12) { r.eigenvalues = bestLam; r.eigenvectors = bestX; return sortAscending(std::move(r)); }
    }
    for (int step = 0; step < 40; ++step) {
        Eigen::MatrixXd Y(n, k);
        const Eigen::MatrixXd BX = B * X;
        for (int j = 0; j < k; ++j) {
            Eigen::VectorXd y(n);
            inner.perform_op(BX.col(j).data(), y.data());
            Y.col(j) = y;
        }
        std::vector<double> before(k);
        for (int j = 0; j < k; ++j) before[j] = std::sqrt(std::max(0.0, Y.col(j).dot(B * Y.col(j))));
        Y -= X * (X.transpose() * (B * Y));                      // what the step adds beyond the held vectors
        // A column that is only rounding (nothing left once the held vectors are projected out: the pair is already converged) is dropped:
        // normalised, it would be noise of unit size in the trial space, and the Rayleigh–Ritz step would degrade the pair it was meant to polish.
        std::vector<int> useful;
        for (int j = 0; j < k; ++j) {
            const double nj = std::sqrt(std::max(0.0, Y.col(j).dot(B * Y.col(j))));
            if (nj > 1e-12 * before[j]) { Y.col(j) /= nj; useful.push_back(j); }
        }
        if (useful.empty()) break;
        const int nw = k + static_cast<int>(useful.size());
        Eigen::MatrixXd W(n, nw);
        W.leftCols(k) = X;
        for (size_t c = 0; c < useful.size(); ++c) W.col(k + static_cast<int>(c)) = Y.col(useful[c]);
        const Eigen::MatrixXd BW = B * W;
        Eigen::MatrixXd Ma = W.transpose() * BW;
        Eigen::MatrixXd Ka = W.transpose() * (A * W);
        Ma = 0.5 * (Ma + Ma.transpose()); Ka = 0.5 * (Ka + Ka.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> em(Ma);
        const double smax = em.eigenvalues().maxCoeff();
        std::vector<int> keep;
        for (int i = 0; i < em.eigenvalues().size(); ++i) if (em.eigenvalues()(i) > 1e-13 * smax) keep.push_back(i);
        if (static_cast<int>(keep.size()) < k) return r;
        Eigen::MatrixXd T(nw, keep.size());
        for (size_t c = 0; c < keep.size(); ++c) T.col(c) = em.eigenvectors().col(keep[c]) / std::sqrt(em.eigenvalues()(keep[c]));
        Eigen::MatrixXd Kr = T.transpose() * Ka * T;
        Kr = 0.5 * (Kr + Kr.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> er(Kr);
        std::vector<int> pick(er.eigenvalues().size());          // the k Ritz values nearest the shift (the window), ascending
        std::iota(pick.begin(), pick.end(), 0);
        std::sort(pick.begin(), pick.end(), [&](int a, int b) { return std::abs(er.eigenvalues()(a) - sigma) < std::abs(er.eigenvalues()(b) - sigma); });
        pick.resize(k);
        std::sort(pick.begin(), pick.end());
        Eigen::MatrixXd Cm(er.eigenvectors().rows(), k);
        for (int c = 0; c < k; ++c) { Cm.col(c) = er.eigenvectors().col(pick[c]); lam(c) = er.eigenvalues()(pick[c]); }
        X = W * (T * Cm);
        // stop when the residuals ‖K x − λ M x‖ are down to 1e-12 of the spectral scale (vectors converge slower than the values: the values alone say "done" early)
        const Eigen::MatrixXd R = A * X - B * X * lam.asDiagonal();
        const double resid = R.colwise().norm().maxCoeff() / std::max(1.0, lam.cwiseAbs().maxCoeff());
        if (resid < bestResid) { bestResid = resid; bestX = X; bestLam = lam; worse = 0; }
        else if (++worse >= 3) break;                            // rounding in the shift-inverted images now outweighs what a step adds: keep the best
        if (resid < 1e-12) break;
    }
    if (bestResid < std::numeric_limits<double>::infinity()) { r.eigenvalues = bestLam; r.eigenvectors = bestX; }     // M-orthonormal
    return sortAscending(std::move(r));
}

static EigenResult completeLowest(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, EigenResult r, double sigma,
                                  const CancellationToken& cancel) {
    return polishPairs(A, B, completeCore(A, B, std::move(r), sigma, cancel), sigma);
}

// What was wrong with the first version (2026-11-10): it assumed a shift-invert window finds *every* eigenvalue within the radius
// it converged to. It does not: a single-vector Lanczos run returns some members of a multiplet and not others (a sphere's
// 2l+1, any symmetry), so the window "covered" a range while missing members of it. Those members were never re-tried, a cut
// that fell inside a multiplet made it worse, and members of one eigenspace taken from two windows were two different bases of it
// (not M-orthogonal). Now a window is made complete *within its ball* by deflation, and every vector is merged by eigenvector:
//  (1) a window's converged pairs go through accept(), which orthogonalises against the held vectors of nearby eigenvalue (M inner
//      product) and drops what is already spanned;
//  (2) the ball is then re-searched with the held vectors projected out of the operator (DeflatedShiftInvertOp) until a run finds
//      nothing new inside it, so missing multiplet members appear, exactly M-orthogonal to the rest;
//  (3) the next shift is put in a gap of the spectrum near the top of the ball (never on an eigenvalue).
// The result is always M-orthonormal (windows are M-normalised internally whatever `normalize` says).
EigenResult eigenSliced(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                        int count, int slice, bool normalize,
                        const CancellationToken& cancel, const ProgressObserver& progress) {
    const int n = static_cast<int>(A.rows());
    if (count < 1 || count > n - 1)
        throw Error(ErrorCode::EigensolveInvalidK, "eigenSliced: count must be in [1, N-1]");
    slice = std::max(2, std::min(slice, n - 1));
    EigenResult first = eigenSmallest(A, B, std::min(count, slice), normalize, cancel, progress);
    if (count <= slice) return first;

    std::vector<double> lam;
    std::vector<Eigen::VectorXd> vec, Bvec;           // the held pairs and B·vector
    // Merge x (eigenvalue l) with the held set: remove its components along held vectors of nearby eigenvalue; keep what remains
    // if it is substantial (a new direction), otherwise it was a duplicate. Returns whether something was added.
    auto accept = [&](double l, Eigen::VectorXd x, double band) {
        Eigen::VectorXd bx = B * x;
        const double n0 = std::sqrt(std::max(0.0, x.dot(bx)));
        if (!(n0 > 0)) return false;
        x /= n0; bx /= n0;
        for (int pass = 0; pass < 2; ++pass)           // Gram–Schmidt twice
            for (size_t h = 0; h < lam.size(); ++h) {
                if (std::abs(lam[h] - l) > band) continue;
                const double c = vec[h].dot(bx);
                if (std::abs(c) > 1e-13) { x -= c * vec[h]; bx -= c * Bvec[h]; }
            }
        const double nrm = std::sqrt(std::max(0.0, x.dot(bx)));
        if (nrm < 0.1) return false;                    // already spanned by held vectors
        lam.push_back(l); vec.push_back(x / nrm); Bvec.push_back(bx / nrm);
        return true;
    };

    using Inner = DeflatedShiftInvertOp::Inner;
    auto runPass = [&](DeflatedShiftInvertOp& op, double sg, int k) {       // k eigenpairs nearest sg of the (deflated) pencil
        const int ncv = std::min(std::max(2 * k + 1, 20), n);
        Spectra::SparseSymMatProd<double> opM(B);
        Spectra::SymGEigsShiftSolver<DeflatedShiftInvertOp, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> solver(op, opM, k, ncv, sg);
        solver.init();
        solver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10);
        if (solver.info() != Spectra::CompInfo::Successful && solver.info() != Spectra::CompInfo::NotConverging)
            throw Error(ErrorCode::EigensolveNotConverged, "eigenSliced: window solve failed");
        EigenResult w;
        w.eigenvalues = solver.eigenvalues();
        w.eigenvectors = solver.eigenvectors();
        w.k = static_cast<int>(w.eigenvalues.size());
        w.eigenvectors = ::nxr::operators::normalize(w.eigenvectors, B);
        return w;
    };

    const int kDefl = std::min(std::max(8, slice / 2), n - 1);
    double covered = -std::numeric_limits<double>::infinity();     // every eigenvalue <= covered is held
    double nextSigma = 0;
    // Merge a window's converged pairs (distance from `centre`), complete its ball by deflation at the factored shift `sg`.
    auto takeWindow = [&](const EigenResult& w0, double centre, Inner& inner, double sg) {
        double r;
        const std::vector<int> good = convergedPrefix(A, B, w0, centre, r);
        const double rho = 0.98 * r;                   // all eigenvalues in (centre−ρ, centre+ρ) are found; 2 % margin
        const double scale = std::max(1e-300, w0.eigenvalues.cwiseAbs().maxCoeff());
        const double band = 1e-3 * std::max(1.0, scale);
        for (int c : good)
            if (std::abs(w0.eigenvalues(c) - centre) <= rho) accept(w0.eigenvalues(c), w0.eigenvectors.col(c), band);
        DeflatedShiftInvertOp op(inner, B, cancel);
        Eigen::MatrixXd X;
        for (int pass = 0; pass < 50; ++pass) {
            std::vector<int> inBall;
            for (size_t h = 0; h < lam.size(); ++h) if (std::abs(lam[h] - centre) <= rho) inBall.push_back(static_cast<int>(h));
            if (static_cast<int>(inBall.size()) >= n - 1) break;
            X.resize(n, static_cast<int>(inBall.size()));
            for (size_t j = 0; j < inBall.size(); ++j) X.col(j) = vec[inBall[j]];
            op.setDeflation(&X);
            const int k = std::min(kDefl, n - 1 - static_cast<int>(inBall.size()));
            if (k < 1) break;
            EigenResult d = runPass(op, sg, k);
            double rd;
            const std::vector<int> dgood = convergedPrefix(A, B, d, centre, rd);
            int added = 0;
            for (int c : dgood)
                if (std::abs(d.eigenvalues(c) - centre) <= rho && accept(d.eigenvalues(c), d.eigenvectors.col(c), band)) ++added;
            if (added == 0) break;                     // nothing new inside the ball: it is complete
        }
        covered = std::max(covered, centre + rho);
        // The next shift: in a gap between consecutive held eigenvalues, in the top quarter of the ball (never on an eigenvalue).
        std::vector<double> held;
        for (double l : lam) if (l > centre + 0.75 * rho && l <= centre + rho) held.push_back(l);
        std::sort(held.begin(), held.end());
        const double gapTol = 1e-5 * scale;
        nextSigma = centre + rho;
        for (int i = static_cast<int>(held.size()) - 2; i >= 0; --i)
            if (held[i + 1] - held[i] > gapTol) { nextSigma = 0.5 * (held[i] + held[i + 1]); break; }
    };
    auto heldBelow = [&](double u) { int c = 0; for (double l : lam) if (l <= u) ++c; return c; };

    {   // window 0: the ladder's (smallest eigenpairs); completed at the ladder's own first shift, −1e-7·λmax
        Eigen::SparseMatrix<double> As = (A + Eigen::SparseMatrix<double>(A.transpose())) * 0.5, Bs = (B + Eigen::SparseMatrix<double>(B.transpose())) * 0.5;
        Inner inner(As, Bs);
        const double lmax0 = std::max(1e-300, first.eigenvalues.cwiseAbs().maxCoeff());
        takeWindow(first, 0.0, inner, -1e-7 * lmax0);
    }
    int stalled = 0;
    for (int it = 0; it < 10000 && heldBelow(covered) < count; ++it) {
        double sg = nextSigma;
        const double centre0 = nextSigma;
        Inner inner(A, B);
        DeflatedShiftInvertOp op(inner, B, cancel);
        EigenResult w;
        for (int attempt = 0;; ++attempt) {      // a shift exactly on an eigenvalue makes K − σM singular: nudge it
            try {
                w = runPass(op, sg, std::min(slice, n - 1));
                break;
            } catch (const Error& e) {
                if (e.code() != ErrorCode::EigensolveNotConverged || attempt >= 3) throw;
                sg = centre0 + 1e-6 * (1 + std::abs(centre0)) * std::pow(10.0, attempt);
            } catch (const std::runtime_error&) {        // Spectra's TridiagEigen failure: the same singular-shift symptom
                if (attempt >= 3) throw;
                sg = centre0 + 1e-6 * (1 + std::abs(centre0)) * std::pow(10.0, attempt);
            }
        }
        const double before = covered;
        takeWindow(w, sg, inner, sg);
        if (covered > before) stalled = 0;
        else if (++stalled >= 3) throw Error(ErrorCode::EigensolveNotConverged, "eigenSliced: window did not advance");
        else nextSigma = covered + 1e-6 * (1 + std::abs(covered)) * std::pow(10.0, stalled);   // a window narrower than the overlap: shift to the edge of what is covered
    }
    const int k = std::min(count, static_cast<int>(lam.size()));
    std::vector<int> ord(lam.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return lam[a] < lam[b]; });
    EigenResult out;
    out.eigenvalues.resize(k);
    out.eigenvectors.resize(n, k);
    for (int c = 0; c < k; ++c) { out.eigenvalues(c) = lam[ord[c]]; out.eigenvectors.col(c) = vec[ord[c]]; }
    out.k = k; out.nConverged = k;
    return out;
}

} // namespace nxr::operators

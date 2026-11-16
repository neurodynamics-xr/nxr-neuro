#include "nxr/compute.h"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/MatOp/SparseCholesky.h>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <limits>

namespace nxr::manifold::solve {

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
        return removeDC ? ::nxr::manifold::solve::removeDC(r) : r;
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

    if (removeDC) {
        result = ::nxr::manifold::solve::removeDC(result);
    }
    if (normalize) {
        result.eigenvectors = ::nxr::manifold::solve::normalize(result.eigenvectors, M);
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
        r.eigenvectors = ::nxr::manifold::solve::normalize(r.eigenvectors, B);
    return r;
}

} // namespace

EigenResult eigenLargestMagn(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B,
                             int k, bool normalize) {
    return eigenCholesky(A, B, k, normalize, Spectra::SortRule::LargestMagn);
}

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
                return eigen(A, B, k, sigma, normalize, /*removeDC=*/false, cancel, progress);
            } catch (const Error& e) {
                if (e.code() != ErrorCode::EigensolveNotConverged) throw;  // k-range etc: don't retry
            }
        }
    }
    return eigenCholesky(A, B, k, normalize, Spectra::SortRule::SmallestMagn);
}

} // namespace nxr::manifold::solve
namespace nxr::manifold::solve {

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
        w.eigenvectors = ::nxr::manifold::solve::normalize(w.eigenvectors, B);
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

} // namespace nxr::manifold::solve

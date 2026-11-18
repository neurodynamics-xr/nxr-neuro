#pragma once
// the factor cache (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/dec.h"

namespace nxr::operators {
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;
} // namespace nxr::operators

namespace nxr::operators {
// ── Factor Cache ─────────────────────────────────────────────
//
// Pre-factored Cholesky / LU decompositions of mesh-derived
// matrices, cached across compute calls. Each factor is built
// lazily on first request and reused on every subsequent call
// for the lifetime of the owning geometry::Manifold.
//
// Why this exists: mesh-derived sparse factorizations cost
// O(n^1.5) to build but only O(nnz) per back-substitution. Each
// solver call (Poisson, Hodge α-solve, direction-field) hits
// the same matrix; without caching we redo that O(n^1.5) work
// on every click and on every frame of any future per-frame
// pipeline. Both geometry-processing-js and geometry-central
// pre-factor and reuse for this reason.
//
// Bit-for-bit contract: each cached factor is built by the same
// matrix-assembly + factorization sequence the previous inline
// solvers used (same regularization ε, same coefficient order,
// same Eigen solver type). Refactored callers must produce
// numerically identical output to the pre-cache implementation.
//
// Lifetime: owned by ContextHolder in the addon; destroyed when
// the JavaScript handle releases (i.e. when the mesh changes).
//
// Not cached: the eigensolver's (K - σM) factor lives inside
// Spectra::SymShiftInvert and is discarded per call. Caching
// it would require replacing Spectra's ShiftInvertOp with a
// custom one — out of scope for v1.
//
// Perf options compatible with the modularity rule (CONTRIBUTING.md §10):
// AVX2/AVX-512 compile flags (the build enables these on x86_64),
// Eigen's separate analyzePattern() + factorize() for matrices
// that share sparsity pattern, multi-RHS batched solves, and GPU
// triangular back-substitution via TSL for hot per-frame paths.
// Not on the table: CHOLMOD / SuiteSparse / MKL / PARDISO — they
// break the WASM and MEX targets.

class CholeskyCache {
public:
    CholeskyCache() = default;
    CholeskyCache(const CholeskyCache&) = delete;
    CholeskyCache& operator=(const CholeskyCache&) = delete;

    /** Diagonal regularization added to every factored matrix.
     *  Identical to the constant the pre-cache inline solvers used. */
    static constexpr double kRegularization = 1e-8;

    /** LLT of (K + ε·I) for any symmetric PSD K. The cache is keyed
     *  by lifetime, not content — the *first* matrix passed in is the
     *  one factored, and subsequent calls return that factor regardless
     *  of what's passed. Use a fresh CholeskyCache per matrix. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const Eigen::SparseMatrix<double>& K);

    /** Convenience overload: factors ops.cotanLaplacian. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& laplacian(
        const ManifoldOperators& ops) {
        return laplacian(ops.cotanLaplacian);
    }

    /** LLT of (d0ᵀ ★₁ d0 + ε·I).
     *  Used by both Hodge α-solve and direction-field. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& hodgeExact(
        const DECOperators& dec);

    /** LU of (d1 ★₁⁻¹ d1ᵀ + ε·I), used by Hodge β-solve.
     *  B is symmetric PSD after regularization, so SimplicialLLT
     *  would also work; SparseLU preserved for bit-for-bit
     *  equivalence with the pre-cache implementation. */
    const Eigen::SparseLU<Eigen::SparseMatrix<double>>& hodgeCoExact(
        const DECOperators& dec);

private:
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> laplacian_;
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> hodgeExact_;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> hodgeCoExact_;
};
} // namespace nxr::operators


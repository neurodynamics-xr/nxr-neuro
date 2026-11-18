#pragma once
// The factorizations of the mesh's systems: DATA PRODUCTS, made once by the caller and passed in. Nothing in compute caches a factor.
//
// A factorization costs O(n^1.5) to make and O(nnz) per solve: whoever solves the same system many times (the pipeline executor, a binding, a test)
// makes the factor once with these functions and passes it to every solve.
//   - (d1 ⋆1⁻¹ d1ᵀ + εI): geometry-central's SquareSolver (Eigen's SparseLU without SuiteSparse): bit-identical to the SparseLU used before
//     (test_factor_solvers).
//   - (K + εI) and (d0ᵀ ⋆1 d0 + εI): Eigen's SimplicialLLT, the factor every golden was made with. geometry-central's PositiveDefiniteSolver
//     (SimplicialLDLT) solves the same systems to ~1e-10 relative but not bit for bit (test_factor_solvers), so it is not substituted.
// ε = 1e-8 on the diagonal, the constant of the solvers these replace. Not available: CHOLMOD / SuiteSparse / MKL (they break the wasm and MEX targets).
#include <memory>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "geometrycentral/numerical/linear_solvers.h"
#include "nxr/core/errors.h"
#include "nxr/operators/dec.h"

namespace nxr::operators {

/** The Cholesky factor of a regularised symmetric positive semidefinite system. */
using CholeskyFactor = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>;
/** geometry-central's LU solver of a square system. */
using SquareSolver = geometrycentral::SquareSolver<double>;

/** The diagonal regularization ε added to every factored matrix. */
constexpr double kFactorRegularization = 1e-8;

/** The factor of (K + εI) for a symmetric PSD K (a Laplacian). Throws Error(GeometryDegenerate) when the factorization fails. */
std::unique_ptr<CholeskyFactor> laplacianFactor(const Eigen::SparseMatrix<double>& K);
/** The factor of (d0ᵀ ⋆1 d0 + εI): the exact part of the Hodge decomposition and the trivial connection. */
std::unique_ptr<CholeskyFactor> exactFormFactor(const DECOperators& dec);
/** The solver of (d1 ⋆1⁻¹ d1ᵀ + εI): the co-exact part of the Hodge decomposition. */
std::unique_ptr<SquareSolver> coexactFormSolver(const DECOperators& dec);

} // namespace nxr::operators

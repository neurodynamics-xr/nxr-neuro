// nxr::optimisation — preconditioned conjugate gradients for a symmetric positive definite sparse system A x = b.
// Pure: a matrix, a right side, a start and a preconditioner in, the solution and its record out. Eigen only. The one solver of the optical flow's `pcg`
// method (both its regularisers); the stopping rule is the relative residual |r| / |b| <= tolerance, at most maxIterations.
#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

namespace nxr::optimisation {

struct PcgOptions {
    enum class Preconditioner { Ldlt, IncompleteCholesky };
    double tolerance = 1e-8;
    int maxIterations = 500;
    Preconditioner preconditioner = Preconditioner::Ldlt;   // Ldlt: the exact factor of the matrix the preconditioner is built from; IncompleteCholesky: Eigen's, of the same matrix
};
struct PcgInfo { int iterations = 0; double residual = 0; bool converged = true; };

/** A factorisation of a symmetric positive definite matrix used as the preconditioner (z = M^-1 r): the exact LDLT, or Eigen's incomplete Cholesky. */
class PcgPreconditioner {
public:
    /** Factors `matrix` (built once; for a fixed sparsity pattern analyse and factorise are separate steps of Eigen's LDLT, as before). */
    void build(const Eigen::SparseMatrix<double>& matrix, PcgOptions::Preconditioner kind);
    bool built() const { return built_; }
    Eigen::VectorXd solve(const Eigen::VectorXd& r) const;
private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    Eigen::IncompleteCholesky<double> ic_;
    PcgOptions::Preconditioner kind_ = PcgOptions::Preconditioner::Ldlt;
    bool built_ = false;
};

/** Solves A x = b from the start `x0` (any other size than b's starts from zero). b = 0 gives x = 0 and residual 0. `info` receives the iterations, the final relative residual and whether it is within tolerance. */
Eigen::VectorXd pcg(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0, const PcgPreconditioner& preconditioner,
                    const PcgOptions& options = {}, PcgInfo* info = nullptr);

}  // namespace nxr::optimisation

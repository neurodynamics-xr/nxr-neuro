#include "nxr/optimisation/pcg.h"

namespace nxr::optimisation {

using Eigen::VectorXd;

void PcgPreconditioner::build(const Eigen::SparseMatrix<double>& matrix, PcgOptions::Preconditioner kind) {
    kind_ = kind;
    if (kind_ == PcgOptions::Preconditioner::Ldlt) { ldlt_.analyzePattern(matrix); ldlt_.factorize(matrix); }
    else ic_.compute(matrix);
    built_ = true;
}

VectorXd PcgPreconditioner::solve(const VectorXd& r) const {
    return kind_ == PcgOptions::Preconditioner::Ldlt ? VectorXd(ldlt_.solve(r)) : VectorXd(ic_.solve(r));
}

VectorXd pcg(const Eigen::SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const PcgPreconditioner& precond, const PcgOptions& o, PcgInfo* info) {
    VectorXd x = x0.size() == b.size() ? x0 : VectorXd::Zero(b.size());
    const double bn = b.norm();
    PcgInfo pi;
    if (!(bn > 0)) { x.setZero(); pi.residual = 0; }
    else {
        VectorXd r = b - A * x, z = precond.solve(r), p = z;
        double rz = r.dot(z);
        while (r.norm() > o.tolerance * bn && pi.iterations < o.maxIterations) {
            const VectorXd Ap = A * p;
            const double pAp = p.dot(Ap);
            if (!(pAp > 0)) break;
            const double a = rz / pAp;
            x += a * p; r -= a * Ap;
            z = precond.solve(r);
            const double rz2 = r.dot(z);
            p = z + (rz2 / rz) * p; rz = rz2;
            ++pi.iterations;
        }
        pi.residual = r.norm() / bn;
        pi.converged = pi.residual <= o.tolerance;
    }
    if (info) *info = pi;
    return x;
}

}  // namespace nxr::optimisation

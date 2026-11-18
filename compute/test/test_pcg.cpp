// nxr::optimisation::pcg against the two loops it replaced (the optical flow's scalar and connection-Laplacian solves, whose bodies were the same line for line).
// REFERENCE below is that loop, kept here as the generator of the golden: the result must equal it bit for bit, iteration counts included, on a synthetic SPD
// system (a lattice Laplacian plus a mass), for both preconditioners and for a zero right side.
#include <cmath>
#include <iostream>
#include <random>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "nxr/optimisation/pcg.h"

using Eigen::SparseMatrix;
using Eigen::VectorXd;
namespace opt = nxr::optimisation;

static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { std::cout << "FAIL: " << msg << "\n"; ++g_fail; } } while (0)

static SparseMatrix<double> lattice(int n) {
    std::vector<Eigen::Triplet<double>> T;
    const int N = n * n;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        const int v = i * n + j; double d = 0.3 + 0.01 * v;
        if (i > 0) { T.emplace_back(v, v - n, -1.0); d += 1; }
        if (i < n - 1) { T.emplace_back(v, v + n, -1.0); d += 1; }
        if (j > 0) { T.emplace_back(v, v - 1, -1.0); d += 1; }
        if (j < n - 1) { T.emplace_back(v, v + 1, -1.0); d += 1; }
        T.emplace_back(v, v, d);
    }
    SparseMatrix<double> A(N, N); A.setFromTriplets(T.begin(), T.end()); A.makeCompressed();
    return A;
}

// REFERENCE: the loop of OpticalflowScalar::stepPcg / ConnectionFlow::stepPcg before the merge.
template <class Precond>
static VectorXd referencePcg(const SparseMatrix<double>& system, const VectorXd& rhs, VectorXd x, const Precond& precond, double tol, int maxIt, int& iterations, double& residual) {
    const double bn = rhs.norm();
    iterations = 0; residual = 0;
    if (!(bn > 0)) { x.setZero(); return x; }
    VectorXd r = rhs - system * x, z = precond(r), p = z;
    double rz = r.dot(z);
    while (r.norm() > tol * bn && iterations < maxIt) {
        const VectorXd Ap = system * p;
        const double pAp = p.dot(Ap);
        if (!(pAp > 0)) break;
        const double a = rz / pAp;
        x += a * p; r -= a * Ap;
        z = precond(r);
        const double rz2 = r.dot(z);
        p = z + (rz2 / rz) * p; rz = rz2;
        ++iterations;
    }
    residual = r.norm() / bn;
    return x;
}

int main() {
    const SparseMatrix<double> A = lattice(14);
    std::mt19937 gen(7); std::normal_distribution<double> nd;
    VectorXd b(A.rows()), warm(A.rows());
    for (int i = 0; i < b.size(); ++i) { b[i] = nd(gen); warm[i] = 0.01 * nd(gen); }
    for (int kind = 0; kind < 2; ++kind) {
        opt::PcgOptions o; o.tolerance = 1e-10; o.maxIterations = 500;
        o.preconditioner = kind == 0 ? opt::PcgOptions::Preconditioner::Ldlt : opt::PcgOptions::Preconditioner::IncompleteCholesky;
        // the preconditioner matrix: the system plus a diagonal, as the flow builds it
        SparseMatrix<double> pre = A; for (int i = 0; i < pre.rows(); ++i) pre.coeffRef(i, i) += 0.5;
        opt::PcgPreconditioner P; P.build(pre, o.preconditioner);
        Eigen::SimplicialLDLT<SparseMatrix<double>> ldlt; Eigen::IncompleteCholesky<double> ic;
        if (kind == 0) { ldlt.analyzePattern(pre); ldlt.factorize(pre); } else ic.compute(pre);
        auto precond = [&](const VectorXd& r) { return kind == 0 ? VectorXd(ldlt.solve(r)) : VectorXd(ic.solve(r)); };
        for (int start = 0; start < 2; ++start) {
            const VectorXd x0 = start == 0 ? VectorXd() : warm;      // any other size starts from zero
            int it = 0; double res = 0;
            const VectorXd ref = referencePcg(A, b, x0.size() == b.size() ? x0 : VectorXd::Zero(b.size()), precond, o.tolerance, o.maxIterations, it, res);
            opt::PcgInfo info;
            const VectorXd got = opt::pcg(A, b, x0, P, o, &info);
            CHECK(got.size() == ref.size() && (got - ref).cwiseAbs().maxCoeff() == 0.0, "pcg equals the reference bit for bit (kind " << kind << ", start " << start << ")");
            CHECK(info.iterations == it && info.residual == res, "iterations and residual equal (kind " << kind << ", start " << start << ")");
            CHECK(info.converged && info.iterations > 0, "converged");
        }
    }
    opt::PcgPreconditioner P; P.build(A, opt::PcgOptions::Preconditioner::Ldlt);
    opt::PcgInfo info;
    const VectorXd z = opt::pcg(A, VectorXd::Zero(A.rows()), VectorXd::Ones(A.rows()), P, {}, &info);
    CHECK(z.norm() == 0 && info.iterations == 0 && info.residual == 0, "zero right side gives zero");
    std::cout << (g_fail ? "FAILURES\n" : "ALL PASSED\n");
    return g_fail ? 1 : 0;
}

// The Hodge decomposition of a field: see nxr/signal/hodge_decomposition.h.
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/signal/surface_calculus.h"

#include <iostream>

namespace nxr::signal {

HodgeResult hodge(
    const geometry::SurfaceArrays& surface,
    const operators::DECOperators& dec,
    const operators::CholeskyFactor& exactFactor,
    operators::SquareSolver& coexactSolver,
    const Eigen::VectorXd& omega
) {
    const auto& d0 = dec.d0;
    const auto& d1 = dec.d1;
    const auto& hodge1 = dec.hodge1;
    const auto& hodge1Inv = dec.hodge1Inverse;

    int nV = surface.nV();
    int nE = static_cast<int>(omega.size());
    int nF = surface.nF();

    HodgeResult result;
    result.omega = omega;
    result.exactPotential = Eigen::VectorXd::Zero(nV);
    result.coExactPotentialV = Eigen::VectorXd::Zero(nV);
    result.combinedPotential = Eigen::VectorXd::Zero(nV);
    result.dAlpha = Eigen::VectorXd::Zero(nE);
    result.deltaBeta = Eigen::VectorXd::Zero(nE);
    result.gamma = Eigen::VectorXd::Zero(nE);

    // ── Exact component: α and dα ────────────────────────────
    // Solve A α = d0ᵀ ★₁ ω  where  A = d0ᵀ ★₁ d0 (regularized).
    Eigen::VectorXd rhsAlpha = d0.transpose() * (hodge1 * omega);
    result.exactPotential = exactFactor.solve(rhsAlpha);
    result.dAlpha = d0 * result.exactPotential;

    // ── Co-exact component: β (on faces) and δβ ──────────────
    // Solve B β̃ = d₁ ω  where  B = d₁ ★₁⁻¹ d₁ᵀ (regularized).
    // Then δβ = ★₁⁻¹ d₁ᵀ β̃  (the co-exact 1-form on edges).
    Eigen::VectorXd rhsBeta = d1 * omega;
    result.coExactPotentialF = coexactSolver.solve(rhsBeta);
    result.deltaBeta = hodge1Inv * (d1.transpose() * result.coExactPotentialF);

    // ── Harmonic component: γ = ω - dα - δβ ──────────────────
    result.gamma = omega - result.dAlpha - result.deltaBeta;

    // Average face β values to vertices (for visualization)
    Eigen::VectorXd vertSum = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXi vertCount = Eigen::VectorXi::Zero(nV);
    for (int fi = 0; fi < nF; ++fi) {
        const double val = result.coExactPotentialF(fi);
        for (int k = 0; k < 3; ++k) {
            const int vi = surface.faces(fi, k);
            vertSum(vi) += val;
            vertCount(vi)++;
        }
    }
    for (int i = 0; i < nV; i++) {
        if (vertCount(i) > 0) result.coExactPotentialV(i) = vertSum(i) / vertCount(i);
    }

    result.combinedPotential = result.exactPotential + result.coExactPotentialV;

    // ── Face-centered vector fields via Whitney interpolation ─
    result.omegaVectors     = whitney(surface, dec, result.omega);
    result.dAlphaVectors    = whitney(surface, dec, result.dAlpha);
    result.deltaBetaVectors = whitney(surface, dec, result.deltaBeta);
    result.gammaVectors     = whitney(surface, dec, result.gamma);

    std::cout << "[hodge] Decomposed 1-form. ||dα|| = " << result.dAlpha.norm()
              << ", ||δβ|| = " << result.deltaBeta.norm()
              << ", ||γ|| = " << result.gamma.norm() << std::endl;

    return result;
}

} // namespace nxr::signal

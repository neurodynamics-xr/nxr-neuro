#pragma once
// The Hodge decomposition of a FIELD (a 1-form on the edges): a pure function of the operators' matrices (the DEC) and their factorizations, made by
// the caller (operators/factorization.h) and passed in. The operators (assembly, potentials' systems) are the operators toolbox's; the decomposition
// of a field is signal's.
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/factorization.h"

namespace nxr::signal {

struct HodgeResult {
    // Scalar potentials (per vertex)
    Eigen::VectorXd exactPotential;      // α
    Eigen::VectorXd coExactPotentialF;   // β (per face)
    Eigen::VectorXd coExactPotentialV;   // β averaged to vertices
    Eigen::VectorXd combinedPotential;   // α + β
    Eigen::VectorXd omega;               // input 1-form on edges

    // 1-form components of the decomposition (all [nE])
    Eigen::VectorXd dAlpha;              // exact 1-form: d(α)
    Eigen::VectorXd deltaBeta;           // co-exact 1-form: δ(β)
    Eigen::VectorXd gamma;               // harmonic 1-form: ω - dα - δβ

    // Face-centered vector fields via Whitney interpolation, each [nF, 3]
    Eigen::MatrixXd omegaVectors;
    Eigen::MatrixXd dAlphaVectors;
    Eigen::MatrixXd deltaBetaVectors;
    Eigen::MatrixXd gammaVectors;
};

/**
 * Hodge/Helmholtz decomposition of a 1-form ω:  ω = dα + δβ + γ
 *   - α: scalar potential (vertex-based) — gradient part, from (d0ᵀ ⋆1 d0 + εI) α = d0ᵀ ⋆1 ω
 *   - β: scalar potential (face-based, averaged to vertices) — curl part, from (d1 ⋆1⁻¹ d1ᵀ + εI) β = d1 ω
 *
 * @param surface        the mesh arrays (the face-to-vertex average and the Whitney vectors)
 * @param dec            DEC operators (d0, d1, hodge1, hodge1Inverse) of the INPUT mesh (operators::embeddedDEC)
 * @param exactFactor    operators::exactFormFactor(dec)
 * @param coexactSolver  operators::coexactFormSolver(dec)
 * @param omega          input 1-form on edges
 */
HodgeResult hodge(
    const geometry::SurfaceArrays& surface,
    const operators::DECOperators& dec,
    const operators::CholeskyFactor& exactFactor,
    operators::SquareSolver& coexactSolver,
    const Eigen::VectorXd& omega
);

} // namespace nxr::signal

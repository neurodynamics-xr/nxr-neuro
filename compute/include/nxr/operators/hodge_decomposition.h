#pragma once
// Hodge decomposition of a field on the surface (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"

namespace nxr::operators {
namespace ops = nxr::operators;
using nxr::geometry::Manifold;
using nxr::geometry::IntrinsicSurface;
using nxr::geometry::IntrinsicRecord;
using nxr::geometry::IntrinsicOptions;
using nxr::geometry::DelaunayOptions;
using nxr::geometry::MassKind;
using nxr::geometry::TangentBasis;
using nxr::geometry::ConnectionType;
using nxr::geometry::OperatorId;
using nxr::geometry::MeshKind;
using nxr::geometry::ManifoldOptions;
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;
} // namespace nxr::operators

namespace nxr::operators {
// ── Hodge Decomposition ──────────────────────────────────────

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
 * Hodge/Helmholtz decomposition of a 1-form ω:
 *   ω = dα + δβ + γ
 *
 * Given random ω, returns:
 *   - α: scalar potential (vertex-based) — gradient part
 *   - β: scalar potential (face-based, averaged to vertices) — curl part
 *
 * @param dec    DEC operators (d0, d1, hodge1, hodge1Inverse)
 * @param omega  input 1-form on edges (random or user-supplied)
 */
HodgeResult hodge(
    Manifold& m,
    const ops::DECOperators& dec,
    ops::CholeskyCache& cache,
    const Eigen::VectorXd& omega
);
} // namespace nxr::operators


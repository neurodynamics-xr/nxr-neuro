#pragma once
// trivial-connection direction fields (toolbox refactor P1: moved from compute.h, declarations unchanged).
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

namespace nxr::geometry::connection {

// ── Direction Field Design (Trivial Connections) ─────────────

struct DirectionFieldResult {
    Eigen::VectorXd connections;            // φ per edge (1-form angles)
    Eigen::MatrixXd directionVectors;       // [nF, 3] smooth direction field on faces
    Eigen::MatrixXd orthogonalVectors;      // [nF, 3] 90° rotated direction field on faces
    Eigen::MatrixXd vertexVectors;          // [nV, 3] smooth direction field on vertices
    Eigen::MatrixXd vertexOrthogonalVectors;// [nV, 3] 90° rotated direction field on vertices
    double eulerCharacteristic;
    bool gaussBonnetSatisfied;
};

// ── Smooth Direction Fields (NRoSy, Knöppel-Crane) ───────────
//
// Smoothest unit-norm direction field of order n via the
// Knöppel-Crane formulation — minimizes Dirichlet energy on
// the connection Laplacian, no prescribed singularities (they
// emerge automatically). Distinct from `trivial`
// above (which solves trivial connections with user-prescribed
// singularities).
//
// nSym selects the symmetry order:
//    1 → vector field (one direction per face)
//    2 → line field (two opposite directions per face)
//    4 → cross field (four directions, useful for quad meshing
//                     and stripe patterns aligned to two axes)
//
// `alignToCurvature` returns the principal-curvature-aligned
// field instead of pure smoothest (only meaningful for nSym ≥ 2).

/** Result of `smoothVertex`. The face-based variant
 *  returns a plain Eigen::MatrixXd directly, since face stripes
 *  aren't a downstream consumer of its raw form. */
struct SmoothVertexFieldResult {
    Eigen::MatrixXd vertexVectors;   // [nV, 3] world-space, per vertex
    Eigen::VectorXd vertexFieldRaw;  // [nV * 2] raw Vector2 in vertex tangent basis
    int nSym;
};

/**
 * Compute the trivial connection 1-form φ = δβ (per-edge rotation angles).
 *
 * Solves the Poisson system A β = -K + 2π σ where:
 *   A   = d0ᵀ ★₁ d0 (cotangent Laplacian, via operators::CholeskyCache::hodgeExact)
 *   K   = per-vertex Gaussian curvature (angle defect)
 *   σ   = singularity index at each vertex (0 for non-singular)
 *
 * Returns φ = ★₁ d₀ β as an Eigen::VectorXd of length nEdges.
 * Sign convention: φ(e) is positive in the direction of e.halfedge().
 *
 * Gauss-Bonnet requirement: Σ singularityMap values must equal χ(mesh).
 * This is NOT checked here — the caller is responsible (directionField
 * checks it; assembleTrivialConnectionLaplacian documents it).
 *
 * For sphere topology (genus 0, χ=2): two vertices with σ=1.0 each.
 * For n-RoSy fields: use σ = (desired holonomy) / (2π).
 *   e.g. nSym=4 cross field on sphere: two vertices with σ=0.5.
 */
Eigen::VectorXd computeTrivialConnection(
    Manifold& m,
    const operators::DECOperators& dec,
    operators::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

// ── Trivial-gauge rotations (vertex + face) ──────────────────
//
// Integrates the trivial-connection 1-form φ (from computeTrivialConnection)
// into unit-complex gauge rotations relative to the Levi-Civita frames:
//   vertex — BFS over the primal vertex graph (transportVectorsAlongHalfedge)
//   face   — BFS over the dual face graph (same transport + sign conventions
//            as propagateAngles in direction_field.cpp)
//
// Convention (the COMBING multiplier): the stored rotation is the gauge
// transform that realizes the combed (trivially-parallel) frame by complex
// multiplication —
//   combed vertex frame = vertex(v) .* vertexGridEmbedded.row(v)   (geometry-central's frame, the light Geometry bundle's `vertexGrid`)
//   combed face frame   = face(f)   .* faceGrid.row(f)
// Equivalently real(vertex .* vertexGridEmbedded) IS the trivial-connection parallel
// field (DirectionFieldResult::vertexVectors), and likewise on faces
// (directionVectors). Note this is the CONJUGATE of the accumulated transport
// coefficient e^{iα}: actively rotating a frame by α multiplies the complex
// grid c = e1 + i·e2 by e^{-iα}. Verified numerically against the trivial
// direction field (test_geometry_bundle.cpp).
//
// Anchors: vertex 0 and face 0 carry the identity rotation. The two combs are
// each globally parallel, but their relative phase (vertex comb vs face comb)
// is anchor-dependent and carries no meaning.
//
// Gauss-Bonnet: Σ singularityMap values must equal χ(mesh) (the caller's
// responsibility — not checked here).
//
// Connectivity contract: BFS starts at vertex 0 / face 0 and visits only that
// connected component; on a disconnected mesh other components receive
// 0+0i. Intended usage is one connected component (e.g. a single
// cortical hemisphere) at a time.
struct GaugeRotations {
    Eigen::VectorXcd vertex;   // [nV] combing rotation, |r_v| = 1
    Eigen::VectorXcd face;     // [nF] combing rotation, |r_f| = 1
};
GaugeRotations integrateTrivialGaugeRotations(
    Manifold& m,
    const operators::DECOperators& dec,
    operators::CholeskyCache& cache,
    const std::map<int, double>& singularityMap);

/**
 * Compute a smooth direction field with prescribed singularities.
 * Uses the trivial connections algorithm (Crane, de Goes, Desbrun).
 *
 * Input: singularityMap — vertex index → singularity index (typically ±1 or ±0.5)
 * The sum of indices MUST equal the mesh Euler characteristic (Gauss-Bonnet).
 *
 * Output: φ = δβ + γ (per-edge 1-form angles), plus the resulting
 * direction and orthogonal vector fields on faces.
 */
DirectionFieldResult trivial(
    Manifold& m,
    const operators::DECOperators& dec,
    operators::CholeskyCache& cache,
    const std::map<int, double>& singularityMap
);

/** Compute the smoothest face-based direction field.
 *  Returns [nF, 3] world-space vectors (principal nSym-RoSy
 *  representative; the other nSym-1 directions are obtained by
 *  rotating in the face tangent plane). */
Eigen::MatrixXd smoothFace(
    Manifold& m,
    int nSym = 4,
    bool alignToCurvature = false
);

/** Compute the smoothest vertex-based direction field (needed
 *  as input to stripe patterns).
 *  Returns the raw [nV * 2] Vector2 form alongside [nV, 3] world
 *  vectors, since stripes consume the Vector2 form internally. */
SmoothVertexFieldResult smoothVertex(
    Manifold& m,
    int nSym = 2,
    bool alignToCurvature = false
);
} // namespace nxr::geometry::connection


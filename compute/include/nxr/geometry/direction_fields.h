#pragma once
// smooth direction fields (NRoSy, Knöppel-Crane) on the geometry-central surface.
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"

namespace nxr::geometry::connection {


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
//    4 → cross field (four directions, useful for quad meshing)
//
// `alignToCurvature` returns the principal-curvature-aligned
// field instead of pure smoothest (only meaningful for nSym ≥ 2).

/** Result of `smoothVertex`. The face-based variant
 *  returns a plain Eigen::MatrixXd directly. */
struct SmoothVertexFieldResult {
    Eigen::MatrixXd vertexVectors;   // [nV, 3] world-space, per vertex
    Eigen::VectorXd vertexFieldRaw;  // [nV * 2] raw Vector2 in vertex tangent basis
    int nSym;
};

/** Compute the smoothest face-based direction field.
 *  Returns [nF, 3] world-space vectors (principal nSym-RoSy
 *  representative; the other nSym-1 directions are obtained by
 *  rotating in the face tangent plane). */
Eigen::MatrixXd smoothFace(
    const SurfaceArrays& surface,
    int nSym = 4,
    bool alignToCurvature = false
);

/** Compute the smoothest vertex-based direction field .
 *  Returns the raw [nV * 2] Vector2 form alongside [nV, 3] world
 *  vectors. */
SmoothVertexFieldResult smoothVertex(
    const SurfaceArrays& surface,
    int nSym = 2,
    bool alignToCurvature = false
);
} // namespace nxr::geometry::connection


#pragma once
// BFF parametrization and stripe patterns (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"

namespace nxr::geometry::parametrization {

// ── Surface Parametrization ──────────────────────────────────

/** Boundary First Flattening (Sawhney & Crane 2017) — produces a
 *  conformal-by-default planar parametrization of the surface.
 *
 *  Returns a [V, 2] matrix of UV coordinates (u, v) per vertex.
 *  Mesh must have at least one boundary loop; for closed meshes
 *  the user must cut the surface first. The mesh in nxr-compute is
 *  ManifoldSurfaceMesh so boundary detection works automatically.
 *
 *  Used by texture-based visualizations (LIC, isoline rendering
 *  in UV space, anisotropic shaders). Lower angular distortion
 *  than LSCM, especially near complex boundaries. Uses
 *  geometry-central's bundled BFF implementation — no extra
 *  dependency.
 *
 *  Throws if the mesh has no boundary. */
Eigen::MatrixXd bff(const SurfaceArrays& surface);
} // namespace nxr::geometry::parametrization

namespace nxr::geometry::parametrization::stripes {

// ── Stripe Patterns (Knöppel-Crane SIGGRAPH 2015) ────────────
//
// Procedural sinusoidal stripes aligned to a 2-RoSy direction
// field. Output is a list of 3D polyline segments (the zero
// level set of the per-corner phase function), suitable for
// drawing as three.js LineSegments.
//
// Frequencies control stripe spacing in oscillations per unit
// edge length. A constant uniformFrequency is the common case;
// pass per-vertex frequencies if you want curvature-modulated
// striping.

struct StripePatternResult {
    Eigen::MatrixXd positions;   // [2 * segmentCount, 3] endpoint pairs
    int segmentCount;
};

/** Stripe pattern with a uniform target frequency. The vertex
 *  field must be 2-RoSy (use `smoothVertex(m, 2)`
 *  or pass an existing line field). */
StripePatternResult compute(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2] in vertex tangent basis
    double uniformFrequency,
    bool connectOnSingularities = true
);

/** Stripe pattern with per-vertex target frequencies. */
StripePatternResult computeFreq(
    const SurfaceArrays& surface,
    const Eigen::VectorXd& vertexFieldRaw,  // [nV * 2]
    const Eigen::VectorXd& frequencies,     // [nV]
    bool connectOnSingularities = true
);
} // namespace nxr::geometry::parametrization::stripes


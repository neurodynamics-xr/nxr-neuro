#pragma once
// Whitney interpolation, isolines and streamlines on the surface (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/dec.h"

namespace nxr::signal {

// ── Vector Field Operations ──────────────────────────────────

/** Whitney interpolation: edge 1-form → face-centered 3D vectors.
 *  Formula: V(f) = (N × (a + b + c)) / (6A)
 *  Returns [nF, 3] matrix (row-major vectors). */
Eigen::MatrixXd whitney(
    const geometry::SurfaceArrays& surface,
    const operators::DECOperators& dec,
    const Eigen::VectorXd& oneForm
);


// ── Isolines ─────────────────────────────────────────────────

struct IsolineResult {
    Eigen::MatrixXd positions;  // [segmentCount * 2, 3] endpoints of line segments
    int segmentCount;
};

/** How the edge crossings are found; both give the same segments (the same crossing points), in a different order.
 *    PerTriangle (DEFAULT): the Studio's isoline.ts, face by face, every level: one pass over the faces (timed on a 164k-vertex mesh: test_isoline_methods).
 *    MarchingTriangles: geometry-central's marchingTriangles, level by level, the segments in the order of the curves (its pass grows faster than linearly
 *    in a level's crossings). */
enum class IsolineMethod { PerTriangle, MarchingTriangles };

/** Extract isoline (contours) from a per-vertex scalar field.
 *  numLevels evenly-spaced contour values in [minValue, maxValue].
 *  If minValue == maxValue, auto-detects range from the data. */
IsolineResult isoline(
    const geometry::SurfaceArrays& surface,
    const Eigen::VectorXd& scalarField,
    int numLevels = 20,
    double minValue = 0.0,
    double maxValue = 0.0,
    IsolineMethod method = IsolineMethod::PerTriangle
);

// ── Streamlines ──────────────────────────────────────────────

struct StreamlineResult {
    // Flat [2 * N, 3] array of line segment endpoints (pair-wise)
    Eigen::MatrixXd positions;
    int segmentCount;
};

/**
 * Trace streamlines through a face vector field using forward Euler
 * with face-crossing detection.
 *
 * @param faceField  [nF, 3] vector field on face centroids
 * @param numSeeds   target number of seed streamlines (quasi-uniform)
 * @param stepCoef   step size = stepCoef * meanEdgeLength
 * @param maxSteps   max integration steps per streamline
 */
StreamlineResult streamline(
    const geometry::SurfaceArrays& surface,
    const Eigen::MatrixXd& faceField,
    int numSeeds = 15,
    double stepCoef = 0.15,
    int maxSteps = 1000
);


} // namespace nxr::signal

#pragma once
// curvatures (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"

namespace nxr::geometry {

// ── Curvatures ───────────────────────────────────────────────

struct CurvatureResult {
    Eigen::VectorXd gaussian;        // K per vertex (angle defect)
    Eigen::VectorXd mean;            // H per vertex
    Eigen::VectorXd kMin;            // κ_min per vertex
    Eigen::VectorXd kMax;            // κ_max per vertex
    Eigen::MatrixXd principalDirMax; // [nV, 3] max principal direction lifted to 3D
};

CurvatureResult curvatures(Manifold& m);
// ── Curvature as a 2-RoSy complex + mean ─────────────────────
//
// The shape operator (3 real DOF) split into its trace and deviatoric
// parts. `deviatoric` is the traceless symmetric part encoded as a
// 2-RoSy complex number q: arg(q)/2 is the max principal direction
// angle in the vertex tangent basis, |q| = (κmax − κmin)/2. `mean`
// is (κmax + κmin)/2. See bundle design spec §5.2. Everything else
// (principal curvatures/directions, extrinsic Gaussian) derives from
// these; intrinsic Gaussian comes from the angle-defect (2π − angleSum).
// If geometry-central principal directions are unavailable at a vertex, the
// deviatoric phase falls back to 0; only |q| (anisotropy) and mean (H) are
// reliable there.
struct VertexCurvature2RoSy {
    Eigen::VectorXcd deviatoric;  // [nV] q
    Eigen::VectorXd  mean;        // [nV] H
};
VertexCurvature2RoSy vertexCurvature(Manifold& m);
} // namespace nxr::geometry


#pragma once
// nxr::transport — THE GEOMETRY TRANSPORT: carrying a vector along the cortex as a current NORMAL to it would be carried (a rotation R_uv per edge, SO(3)).
// It ENCODES THE PRIOR that currents are normal to the cortex: a named option for that model, NOT the
// default and NOT the direction route. Direction analysis makes no orientation assumption: the joint data-block magnetic Laplacian
// (magnetic.h, blockMagneticLaplacian) takes the relation between two currents from the data. With a measured lag θ, e^{iθ} R_uv is the transport of
// magnetic.h's transportLaplacian (the normal-current model).
//
//   THE TRANSPORT (a named option; the rotations are world-coordinate, R_uv J_v is v's vector carried to u):
//     levi_civita   the minimal rotation taking v's normal to u's (about n_v × n_u): the surface's own parallel transport along the edge, from the normals
//                   alone (no gauge); tangent vectors keep their angle to the edge. It has holonomy where the surface is curved (Gaussian curvature).
//     frames        F_u F_vᵀ with F = [e1 | e2 | n] the vertices' frames: compute's flat full-frame transport (geometry::differential::frameTransport,
//                   P = F_uᵀ F_v in frame coordinates) in world coordinates; path-independent (no holonomy); its tangential part depends on how each
//                   vertex's e1 was chosen (geometry::vertexFrames takes it from a reference edge).
//   compare(T, R, Tuu, Tvv)   per edge the transported cross-spectrum tr(T_uv R_uvᵀ) = Σ_a ⟨J_u^a conj((R_uv J_v)^a)⟩ (T_uv = ⟨J_u J_vᴴ⟩, a 3 x 3 block:
//                   algebra::blockCongruence), its lag θ = arg and its strength |·| / √(tr T_uu tr T_vv) (a coherency, 0 … 1).
//
// AN ASSUMPTION, NOT A DEFAULT: the geometry transport ENCODES the prior that currents are normal to the cortex (it carries a vector as
// a normal current would be carried). Nothing requires an estimate to obey it; direction analysis makes no orientation assumption (the magnetic
// Laplacian spec, "direction analysis makes no orientation assumption": the joint data-block route). On MEG a free minimum-norm estimate had ~1/3 of its
// power along the normal, and transport rotated its nearly equal world vectors apart. And `compare` reduces a 3 x 3 block to its
// trace (like components only): on MEG most of the lag content sat in the CROSS-COMPONENT terms, which the trace discards. Read lag content from the
// full block (‖Im T‖ / ‖T‖) before trusting a trace.
//
// PURE MATHS: arrays in, arrays out.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::geometry {

using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
enum class Transport { LeviCivita, Frames };
/** the option by name ("levi_civita", "frames"); std::invalid_argument naming the valid ones otherwise */
Transport transportOf(const std::string& name);

/** the minimal rotations R_uv (n_v → n_u) for edges (u, v); `normals` [n x 3] unit. Antiparallel normals: a half-turn about an axis perpendicular to both. */
std::vector<Eigen::Matrix3d> leviCivita(const Eigen::MatrixXd& normals, const Edges& edges);
/** R_uv = F_u F_vᵀ, F_v = [e1_v | e2_v | n_v] (each [n x 3], orthonormal a vertex) */
std::vector<Eigen::Matrix3d> frames(const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& normals, const Edges& edges);

struct Compared {
  std::vector<std::complex<double>> S;   // tr(T_uv R_uvᵀ) per edge
  Eigen::VectorXd lag;                   // arg S, rad
  Eigen::VectorXd strength;              // |S| / √(tr T_uu tr T_vv)
};
/** T [E] the edges' 3 x 3 blocks ⟨J_u J_vᴴ⟩, R [E] their rotations, Tuu / Tvv [E] the ends' powers (tr of their own blocks) */
Compared compare(const std::vector<Eigen::MatrixXcd>& T, const std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& Tuu, const Eigen::VectorXd& Tvv);

}  // namespace nxr::transport
#include "nxr/geometry/mesh.h"

namespace nxr::geometry::transport {

// ── Vector Heat Method (Sharp, Soliman, Crane 2019) ──────────
//
// Stateful solver around a single (Laplacian, vector heat, Poisson)
// factor triple. Owned by the binding shell's per-mesh holder
// (ContextHolder / ContextWrapper) and lazily constructed on first
// use, mirroring operators::CholeskyCache's lifetime model.
//
// The solver is opaque to JS — three free functions below operate
// on a long-lived solver passed by reference. The shells expose
// these as methods on their stateful context wrapper so the
// internal `VectorHeatMethodSolver*` never crosses the boundary.
//
// All tangent-vector inputs/outputs are V × 3 world-space vectors.
// Internally we project to / lift from each vertex's intrinsic
// tangent basis (geometry-central's `vertexTangentBasis`). This
// is a §11 layout extension: V × 2 tangent-frame data is lifted
// to V × 3 at the C++/JS boundary so callers never see Vector2.

class VectorHeatSolverImpl;
class VectorHeatSolver {
public:
    explicit VectorHeatSolver(Manifold& m, double tCoef = 1.0);
    ~VectorHeatSolver();
    VectorHeatSolver(const VectorHeatSolver&) = delete;
    VectorHeatSolver& operator=(const VectorHeatSolver&) = delete;

    // Internal — bindings should not call these directly.
    VectorHeatSolverImpl& impl();
    Manifold& m();

private:
    std::unique_ptr<VectorHeatSolverImpl> impl_;
};

/** Parallel-transport tangent vectors from a sparse set of sources
 *  to every vertex via the vector heat method.
 *
 *  Each source is (vertexIdx, worldVec3). The solver projects
 *  worldVec3 onto the source vertex's tangent basis, runs the
 *  vector heat solve, then lifts each destination vertex's
 *  Vector2 result to a world-space V × 3 vector using its own
 *  vertex tangent basis. Vectors at vertices very near a source
 *  reproduce the source vector almost exactly; vectors far away
 *  are the parallel transport of those sources along geodesics.
 *
 *  @return [nV, 3] row-major world-space tangent vectors. */
Eigen::MatrixXd parallel(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::MatrixXd& sourceVectors  // [nSources, 3]
);

/** Extend a sparse scalar field from the given source vertices to
 *  the entire mesh via the scalar heat method. Behaves like a
 *  geodesic Voronoi: each destination vertex takes the value of
 *  its nearest source, smoothly interpolated.
 *
 *  @return [nV] per-vertex scalars. */
Eigen::VectorXd extendScalar(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    const Eigen::VectorXd& sourceValues   // [nSources]
);

/** Logarithmic map at a source vertex: per-vertex 2D coordinates
 *  (logX, logY) in the *source* vertex's tangent frame.
 *
 *  The norm √(logX² + logY²) is the geodesic distance from source
 *  to that vertex; the angle atan2(logY, logX) is the direction
 *  in source's frame. Equivalent to the exponential-map inverse
 *  flattened around the source, useful for picking, brushing, and
 *  any "distance + bearing from p" UI.
 *
 *  Strategy defaults to AffineLocal (Affine Heat Method, faster
 *  and more accurate near source than the original VectorHeat
 *  approach; the inner solver pre-factors and caches its
 *  connection-Laplacian so repeated calls on different sources
 *  amortize the factorization).
 *
 *  Returns:
 *    - logCoords: [nV, 2] row-major (logX, logY)
 *    - sourceE1, sourceE2: 3D world vectors of the source frame
 *      so consumers can reconstruct world positions if needed
 *      via  `position(v) ≈ source + logX·e1 + logY·e2`. */
struct LogMapResult {
    Eigen::MatrixXd logCoords;   // [nV, 2]
    Eigen::Vector3d sourceE1;    // source vertex's tangent basis e1 in world
    Eigen::Vector3d sourceE2;    // source vertex's tangent basis e2 in world
};

enum class LogMapStrategy {
    VectorHeat = 0,      // original VHM paper (no prefactoring)
    AffineLocal = 1,     // Affine Heat Method, prefactored, default
    AffineAdaptive = 2,  // Affine Heat Method, no prefactor, most accurate
};

LogMapResult logMap(
    VectorHeatSolver& solver,
    int sourceVertex,
    LogMapStrategy strategy = LogMapStrategy::AffineLocal
);

/** Karcher mean / surface center of a set of source vertices.
 *  Returns the 3D world position of the SurfacePoint that
 *  minimizes ∑ d(p, source_i)^p (default p=2 → Karcher mean). */
Eigen::Vector3d findCenter(
    VectorHeatSolver& solver,
    const std::vector<int>& sourceVertices,
    int p = 2
);
} // namespace nxr::geometry::transport


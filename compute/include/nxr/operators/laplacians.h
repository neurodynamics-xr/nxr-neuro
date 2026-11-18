#pragma once
// graph and connection Laplacians (toolbox refactor P1: moved from compute.h, declarations unchanged).
#include <complex>
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/mesh.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"

namespace nxr::operators {
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;
} // namespace nxr::operators

namespace nxr::operators {
// Graph (combinatorial) Laplacian L = D − A, built as d0ᵀ d0 from the
// metric-free exterior derivative. Pure topology — independent of vertex
// positions. Symmetric PSD; diagonal = vertex degree,
// off-diagonal = −1 for adjacent vertices, 0 otherwise.
Eigen::SparseMatrix<double> graphLaplacian(const geometry::SurfaceArrays& surface);
} // namespace nxr::operators

namespace nxr::operators::laplacian::connection {

// ── Connection Laplacian ─────────────────────────────────────
//
// Connection Laplacian on a chosen domain (vertex / face / edge).
// The operator that drives smoothest n-RoSy direction fields,
// parallel-transport energies, and tangent-bundle eigendecompositions.
//
// Mathematically the operator is complex Hermitian on the chosen
// domain (V, F, or E entries), built from cotangent weights and
// Levi-Civita transport rotations raised to the nSym power. The
// `Real2N` format (default) lowers each complex entry a + bi to the
// 2×2 real block [[a, -b], [b, a]], producing a 2N×2N symmetric real
// matrix that drops directly into nxr-compute's existing real-only
// eigensolver (`solve`) when paired with a block-diagonal
// real mass matrix `blkdiag(M, M)`. The smallest eigenpair then
// reproduces the smoothest n-direction field that
// `computeSmoothest{Vertex,Face}DirectionField` returns internally.
//
// `Complex` format leaves the matrix as complex Hermitian; bindings
// surface real and imaginary triplet arrays separately. Useful when
// callers want to apply a complex-Hermitian eigensolver themselves.
//
// MATLAB reference: none in `+bct` today (the existing direction-field
// reference uses trivial connections; connection-Laplacian-based
// smoothest fields delegate to geometry-central). When a MATLAB
// oracle harness lands, mirror it under
// `+bct.+manifold.+operator.connectionLaplacian`.

enum class ConnectionDomain {
    Vertex,                  // V × V (intrinsic vertex CL)
    Face,                    // F × F (intrinsic face CL — geometry-central
                             // has FIXME on face weights, propagated as a
                             // doc caveat below)
    EdgeCrouzeixRaviart      // E × E (intrinsic Crouzeix-Raviart CL)
};

enum class ConnectionLaplacianFormat {
    Real2N,                  // 2N × 2N symmetric real (default)
    Complex                  // N × N complex Hermitian, surfaced as
                             // parallel real/imag triplet arrays at
                             // the binding edge
};

/** Options bag for `assembleConnectionLaplacian`. Each field is
 *  defaulted; a value-initialized struct produces the default
 *  vertex / nSym=1 / regularization=1e-8 / Real2N variant. */
struct ConnectionLaplacianOptions {
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int    nSym                      = 1;       // 1=vector, 2=line, 4=cross
    double regularization            = 1e-8;    // additive ε·I shift
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Result of `assembleConnectionLaplacian`. Exactly one of `K_real`
 *  / `K_complex` is populated, selected by `options.format`. */
struct ConnectionLaplacian {
    Eigen::SparseMatrix<double>                K_real;     // populated iff format == Real2N
    Eigen::SparseMatrix<std::complex<double>>  K_complex;  // populated iff format == Complex
    Eigen::MatrixXd  frameE1;   // [baseDim, 3] tangent frame e1 per vertex/face (gauge)
    Eigen::MatrixXd  frameE2;   // [baseDim, 3] tangent frame e2 per vertex/face (gauge)
                                // Empty (0 rows) for EdgeCrouzeixRaviart domain.
    int  baseDim   = 0;                                    // V / F / E
    int  outputDim = 0;                                    // 2 * baseDim (Real2N) or baseDim (Complex)
    ConnectionDomain domain          = ConnectionDomain::Vertex;
    int  nSym                        = 1;
    double regularization            = 1e-8;
    ConnectionLaplacianFormat format = ConnectionLaplacianFormat::Real2N;
};

/** Assemble the connection Laplacian on the chosen domain.
 *
 *  Walks halfedges / faces / edges directly using
 *  `geometry.edgeCotanWeights`, `geometry.transportVectorsAlongHalfedge`,
 *  `geometry.transportVectorsAcrossHalfedge`, and applies `.pow(nSym)` to
 *  each transport rotation (mirrors the file-static helpers in
 *  `geometry-central/src/surface/direction_fields.cpp` that drive
 *  `computeSmoothest{Vertex,Face}DirectionField`).
 *
 *  Throws `Error(InvalidInput)` for `nSym <= 0`. The Crouzeix-Raviart
 *  variant requires triangle meshes (geometry-central's own assertion).
 *
 *  Note: the face variant inherits geometry-central's FIXME on face
 *  weights (uniform 1.0). The numerical answer is still well-defined
 *  for nRoSy energy minimization; revisit if a more principled face
 *  weight emerges upstream. */
ConnectionLaplacian assembleConnectionLaplacian(
    const geometry::SurfaceArrays& surface,
    const ConnectionLaplacianOptions& opts = {}
);

/** First-order connection gradient d^∇ : tangentVertex [V] (complex) → tangentEdge [E]
 *  (complex). Per edge e=(i→j) via e.halfedge(): row e has +1 at column j and
 *  −transportVectorsAlongHalfedge[he].pow(nSym) at column i. Built in the active gauge,
 *  so (d^∇)ᴴ diag(edgeCotanWeight) d^∇ == the connection Laplacian for the same (nSym,
 *  gauge). Vertex domain (v1).
 *
 *  Throws Error(InvalidInput) for nSym <= 0. */
Eigen::SparseMatrix<std::complex<double>>
assembleConnectionGradient(const geometry::SurfaceArrays& surface, int nSym = 1);

/** String → enum helpers for binding shells (WASM, addon).
 *  Domain accepts "vertex", "face", "edge". Format accepts "real2N"
 *  / "real" / "complex". Throws Error(InvalidInput) on unknown. */
ConnectionDomain          parseConnectionDomain(const std::string& name);
ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name);

/**
 * Assemble the connection Laplacian using the trivial connection transport.
 *
 * The trivial connection for the given singularity map is computed first
 * (via computeTrivialConnection), then used to modulate the Levi-Civita
 * transport on each halfedge:
 *
 *   ρ^TC[he] = ρ^LC[he] · exp(i · sign(he) · φ[edge(he)])
 *
 * where sign(he) = +1 if he == edge.halfedge(), -1 otherwise, and
 * φ is the per-edge trivial connection 1-form (Eigen::VectorXd, nEdges).
 *
 * The off-diagonal entry K[i,j] for halfedge he (tail=i, tip=j) is then:
 *
 *   K[i,j] += -w_ij · (ρ^TC[he.twin()])^nSym
 *
 * Gauss-Bonnet: Σ singularityMap values must equal χ(mesh). Not checked
 * internally — throws nothing if violated, but φ will be incorrect.
 *
 * Only ConnectionDomain::Vertex is currently supported. Passing Face or
 * EdgeCrouzeixRaviart throws Error(InvalidInput).
 *
 * Output format follows opts.format (Real2N or Complex), same as
 * assembleConnectionLaplacian.
 */
ConnectionLaplacian assembleTrivialConnectionLaplacian(
    const geometry::SurfaceArrays& surface,
    const std::map<int, double>& singularityMap,
    const operators::DECOperators& dec,
    const nxr::operators::CholeskyFactor& exactFactor,
    const ConnectionLaplacianOptions& opts = {}
);

// ── Covariant (3-frame) Laplacian ─────────────────────────────
//
// 3N×3N real symmetric sparse Laplacian on the full 3-frame
// [a; b; c] = [Re z; Im z; normal], component-major block layout.
// Two couplings: Product (blkdiag) and Ambient (frame-conjugate).
// See spec docs/superpowers/specs/2026-06-08-vector-laplacian-3d-design.md §3–4.

// Coupling for the 3-frame covariant Laplacian (see covariant-laplacian spec).
enum class CovariantCoupling {
    Product,   // blkdiag(real-expand(K), cotanL) — tangent ⊕ normal decoupled
    Ambient    // frame-conjugate of kron(I3, cotanL): L3[i,j] = cotanL[i,j]·(Fiᵀ Fj)
};

// 3N×3N real symmetric Laplacian on the 3-frame [a;b;c] = [Re z; Im z; normal],
// component-major block layout. Pure linear algebra over pieces the caller holds:
//   K        = gauge connection Laplacian (V×V complex Hermitian) — used by Product
//              (and equals the Ambient tangent block for the matching gauge)
//   gaugeGrid= realized gauge frame, V×3 complex c = e1 + i·e2 (normal = Re×Im) — used by Ambient
//   cotanL   = scalar cotan Laplacian (V×V real)
// See spec §3–4. Symmetric PSD.
Eigen::SparseMatrix<double> assembleCovariantLaplacian(
    CovariantCoupling coupling,
    const Eigen::SparseMatrix<std::complex<double>>& K,
    const Eigen::MatrixXcd& gaugeGrid,
    const Eigen::SparseMatrix<double>& cotanL);
} // namespace nxr::operators::laplacian::connection

// ---- the magnetic Laplacians (former magnetic.h)
namespace nxr::operators {

using Complex = std::complex<double>;
using Hermitian = Eigen::SparseMatrix<Complex>;
using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;   // each undirected edge once: (u, v)
using Faces = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

enum class Normalise { None, Symmetric };

/** the normalisation by its name ("none", "symmetric"); std::invalid_argument naming the valid ones otherwise */
Normalise normaliseOf(const std::string& name);

/** L on n vertices from the edges' weights (W[e] = W_uv for edges(e, 0) = u, edges(e, 1) = v; W_vu = conj). Throws std::invalid_argument on a shape
 *  mismatch, an edge outside [0, n), a self-loop, an edge given twice, or a non-finite weight. With `symmetric`, a vertex of no weight (D_uu = 0) keeps a
 *  zero row (it is in no edge's fit). */
Hermitian magneticLaplacian(int n, const Edges& edges, const std::vector<Complex>& W, Normalise normalise);

/** THE TRANSPORTED MAGNETIC LAPLACIAN (a connection Laplacian whose transport is the surface's rotation times the measured lag): on n vertices with a
 *  b-vector a vertex (b = 3: a current), W_uv = w_uv e^{iθ_uv} R_uv (R_uv [b x b] orthogonal, carrying v's vector to u: nxr::geometry), W_vu = W_uvᴴ,
 *  D_uu = Σ_v w_uv I_b; L = D − W [b·n x b·n], Hermitian, PSD, its quadratic form Σ_edges w_uv |x_u − e^{iθ_uv} R_uv x_v|². Its lowest eigenvector is ONE
 *  vector map x (b complex numbers a vertex) whose transported lags fit the measured ones: no per-vertex sign is chosen; only a global phase remains.
 *  `normalise` symmetric: D^{-1/2} L D^{-1/2}. Throws std::invalid_argument on mismatched sizes, a self-loop, an edge twice, a negative or non-finite
 *  weight, or a rotation of the wrong shape. */
Hermitian transportLaplacian(int n, const Edges& edges, const Eigen::VectorXd& w, const Eigen::VectorXd& theta, const std::vector<Eigen::MatrixXd>& R, Normalise normalise);

/** THE BLOCK MAGNETIC LAPLACIAN: b complex numbers a vertex, ARBITRARY complex b x b edge blocks W_uv (blocks[e] for edges(e) = (u, v); W_vu = W_uvᴴ),
 *  degree D_uu = Σ_v ‖W_uv‖₂ I_b (the blocks' spectral norms: PSD for any blocks, since |x_uᴴ W x_v| ≤ ‖W‖ |x_u| |x_v|); L = D − W [b·n x b·n]. b = 1 is
 *  magneticLaplacian; blocks w e^{iθ} R (R orthogonal) transportLaplacian. THE JOINT DATA-BLOCK ROUTE: with
 *  each edge's normalised cross-spectrum block (T_uv / √(tr T_uu tr T_vv)) its lowest eigenvector is one complex b-vector a vertex fitting every pair's
 *  relation (orientation and lag together) consistently over the graph; only a global phase is free. `normalise` symmetric: D^{-1/2} L D^{-1/2}.
 *  Throws std::invalid_argument on mismatched sizes, a self-loop, an edge twice, blocks of unequal or wrong shape, or a non-finite entry. */
Hermitian blockMagneticLaplacian(int n, const Edges& edges, const std::vector<Eigen::MatrixXcd>& blocks, Normalise normalise);


/** the winding number of arg x round each face (faces CCW): an integer per face */
std::vector<int> winding(const Eigen::VectorXcd& x, const Faces& faces);

}  // namespace nxr::operators

namespace nxr::operators::differential {
// ── Covariant differential operators on 3-vector cortical fields ──────────────
//
// Artifact-free transport + differentiation of a 3-vector field (e.g. a leadfield)
// expressed in per-vertex cortical frames Fv = [e1 | e2 | n] (columns), sourced from
// geometry::vertexFrames. The connection is the flat full-frame transport
// P_ij = Fj^T Fi — it accounts for the tangent rotation AND the normal tilt, and is
// path-independent (zero holonomy), which is exactly what removes the frame-rotation
// artifact (e.g. Cartesian-parallel leadfields on opposite sulcal walls reading as
// antiparallel in local frames). See the design doc.

// 3x3 transport of a frame-local vector from vertex i's frame to vertex j's frame,
// P_ij = Fj^T Fi. Flat => exact for ANY pair (i, j) (adjacent or not). 0-based indices.
Eigen::Matrix3d frameTransport(const geometry::SurfaceArrays& surface, int i, int j);

// Per-vertex frames Fv stacked as [nV, 9], row v = Fv flattened ROW-major
// (Fv(0,0),Fv(0,1),Fv(0,2), Fv(1,0),...). Columns of Fv are [e1 | e2 | n].
Eigen::MatrixXd vertexFrameMatrices(const geometry::SurfaceArrays& surface);

// Lift a frame-local field to world (Cartesian): world[v] = Fv * Lloc[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToWorld(const geometry::SurfaceArrays& surface, const Eigen::MatrixXd& Lloc);

// Express a world (Cartesian) field in frames: Lloc[v] = Fv^T * Lworld[v]. [nV,3] -> [nV,3].
Eigen::MatrixXd liftToFrame(const geometry::SurfaceArrays& surface, const Eigen::MatrixXd& Lworld);

// Covariant gradient operator G : 3N -> 3E (component-major [a;b;c] blocks). For each
// oriented edge e: i->j, the covariant difference is delta_e = L_j - P_ij L_i. G has a +I3
// block at vertex j and a -P_ij block at vertex i. A Cartesian-constant field has G*L = 0.
Eigen::SparseMatrix<double> covariantGradient(const geometry::SurfaceArrays& surface);

// Extrinsic vector Weitzenböck Δ₃ + W_extrinsic on the ambient (ℝ³) bundle, a
// [3V×3V] real symmetric sparse matrix in WORLD-frame component-major layout
// (row/col index = c·N + v, c = world x/y/z). It is, by construction, the
// imaginary (vector) block of the immersion SQUARED Dirac
//   S = 2·dirac(0.5) = dirac(0) + dirac(1) = Δ₄ + D_N   (the [4V×4V] real
// vertex-interleaved 4v+c operator), extracting the quaternion components
// c ∈ {1,2,3} and dropping the scalar (c=0) border. This is the principled,
// geometrically-meaningful construction: the imaginary quaternion axes (x,y,z)
// ARE the ambient world axes the Dirac normals (Gauss map) are expressed in, so
// no per-vertex basis change is needed. Its "flat part" (the component-diagonal
// piece coming from dirac(0)=cotanL⊗I₄) is exactly kron(I₃, cotanLaplacian); the
// extra coupling is the extrinsic curvature endomorphism W_ext (the imag block
// of D_N), which vanishes on a developable/flat region (constant normals) and is
// nonzero on curved regions. Like the first-order Dirac, S annihilates GLOBALLY-
// constant world fields (D kills constant quaternions), so W_ext is detected as a
// nonzero MATRIX difference from the flat anchor, not as a constant-field response.
Eigen::SparseMatrix<double> assembleExtrinsicWeitzenbock(const geometry::SurfaceArrays& surface);
} // namespace nxr::operators::differential

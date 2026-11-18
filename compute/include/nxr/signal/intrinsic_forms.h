#pragma once
// nxr::intrinsic — fields on the INTRINSIC complex the Hodge–Dirac modes are solved on, and their reading in the modes' FULL FORMS (spec
// docs/specs/2026-11-12-direction-timing.md;: one triangulation for the modes and everything applied to them). PURE MATHS: arrays in, arrays out.
//
//   Complex      the intrinsic complex as stored by `eigen solve`: its edges (tail, tip), lengths, d0 [E x V], d1 [F x E], the stars, the signposts (each edge's
//                angle at its tail and at its tip, each vertex's cone angle and reference neighbour), and, for what is AMBIENT, the input's vertex positions and
//                normals (the same vertices: the embedding)
//   tangentAt    an intrinsic edge's direction at one of its ends, as an ambient unit vector in the vertex's tangent plane: its signpost angle placed
//                between the two INPUT edges whose angles bracket it, their actual directions interpolated (exact on the input's own edges)
//   flat         an ambient vertex vector field (x y z a vertex) → a 1-form on the intrinsic edges: ω_e = ½ L_e (J_tail · t_tail + J_tip · t_tip), t the edge's
//                direction at each end (pointing tail → tip): the field integrated along the edge, trapezoid rule
//   sharp        a 1-form → a vertex vector field in the tangent planes: at each vertex the least-squares tangent vector v with v · t_e ≈ ω_e / L_e over its edges
//   area2form    a vertex density → a 2-form on the intrinsic faces (the density integrated: area × the mean of the face's vertices); vertexDensity: back (each
//                vertex the area-weighted mean of its faces' densities)
//   Modes        the full forms: P [V x K] the 0-form parts ψ/√2 (the eigenvectors), X2 [F x K] the 2-form parts χ/√2, μ, kind (0 exact, 2 coexact)
//   read1        a 1-form → the coefficients c [K x m] in the modes' 1-form parts φ_k (dψ/(μ√2) exact, ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2) coexact), orthonormal in ⋆1 to 1/2:
//                c_k = 2 ⟨ω, φ_k⟩⋆1 — inner products, no solve
//   synth1       coefficients → the 1-form Σ c_k φ_k
//   read2        a 2-form → the coefficients a [K x m] in the coexact 2-form parts (⋆2-orthonormal to 1/2): a_k = 2 ⟨σ, χ_k/√2⟩⋆2; synth2: back
//   potential    the exact part's potential: Σ_exact c_k P_k / μ_k, whose d0 IS the exact part of Σ c_k φ_k (a timing map, from a phase 1-form)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "nxr/intrinsic_surface.h"

namespace nxr::signal {

struct Complex {
    int nV = 0, nE = 0, nF = 0;
    std::vector<std::int32_t> edges;           // [nE x 2] (tail, tip)
    Eigen::VectorXd length;                    // [nE]
    Eigen::SparseMatrix<double> d0, d1;        // [nE x nV], [nF x nE]
    Eigen::VectorXd star0, star1, star2;       // [nV], [nE], [nF]
    Eigen::VectorXd angleTail, angleTip;       // [nE] the signposts at each end (rad, in [0, cone angle))
    Eigen::VectorXd coneAngle;                 // [nV]
    std::vector<std::int32_t> reference;       // [nV] the reference neighbour
    Eigen::MatrixXd positions, normals;        // [nV x 3] the embedding: positions and unit outward vertex normals
    // the embedding of the signposts: per vertex its INPUT edges' signpost angles (ascending, the same coordinate as the intrinsic edges': flips keep the
    // reference and the cone angle) and their ambient unit directions (the input edge projected on the tangent plane); `attachEmbedding` fills them
    std::vector<std::vector<std::pair<double, Eigen::Vector3d>>> fans;
};

/** The embedding of the signposts from the INPUT mesh (`positions` [3 nV], `inputFaces` [3 nF_in], the same vertices; its own signposts, the input left
 *  as it is): normals (area-weighted) and, per vertex, its input edges' angles and directions. An intrinsic edge's direction is then interpolated between
 *  the two input edges whose angles bracket its own (exact on every input edge; no rescaling of the cone angle, which distorts on a rough mesh). */
void attachEmbedding(Complex& c, const double* positions, const std::int32_t* inputFaces, int nFin);

/** The embedding of the signposts from an IntrinsicSurface (normals: geometry-central's corner-angle-weighted; per vertex its input edges' raw signpost angles
 *  and directions). `complexOf` calls it. */
void attachEmbedding(Complex& c, nxr::manifold::IntrinsicSurface& surface);

/** The complex of an IntrinsicSurface (geometry-central's intrinsic Delaunay triangulation, with signposts): its edges, lengths, d0, d1, the stars (⋆0 the mesh's recorded mass, ⋆1 the
 *  cotan weights, ⋆2 = 1 / area), the signposts, and the embedding (`positions` [3 nV], the INPUT `faces` [3 nF_in] for the area-weighted vertex normals).
 *  What `eigen solve` stores, built in memory (tests; a caller with no stored package). */
Complex complexOf(nxr::manifold::IntrinsicSurface& surface);

/** The direction of edge e at its tail (atTip false) or tip (atTip true), pointing ALONG the edge (tail → tip), as an ambient unit vector. */
Eigen::Vector3d tangentAt(const Complex& c, int e, bool atTip);

/** An ambient vertex field J [3V x m] (rows x y z per vertex) → a 1-form [nE x m] on the intrinsic edges. */
Eigen::MatrixXd flat(const Complex& c, const Eigen::MatrixXd& J);
/** A 1-form [nE x m] → the vertex tangent vectors [3V x m] (least squares over each vertex's edges). */
Eigen::MatrixXd sharp(const Complex& c, const Eigen::MatrixXd& w);

/** THE POINTWISE FORMS of the exact tile integrals of a 1-form on the complex (the discrete Stokes and divergence theorems, additive up the tile tree). A tile's value is the SUM of these over
 *  its vertices: the measures `outflow` and `circulation`, `nxr::measurement::reduce` with measure sum over the tile's vertices (a reduction is never a function here).
 *  vertexOutflow: the OUTFLOW of the vector field whose 1-form is u (u = d0 f for ∇f) out of each vertex's barycentric region = −(d0ᵀ ⋆1 u)_v (the interior edges cancel in a sum over a tile); a tile's mean divergence is its sum over the area.
 *  vertexCirculation: the circulation of w, each face's (d1 w)_f shared equally among its vertices (n/3 of it to each of the 3), so a tile's sum is Σ over faces of (d1 w)_f × (the share of the face's corners in the tile: n/3); a winding number × 2π round a phase defect.
 *  The extrinsic form for a vertex field (a face-constant tangent vector a face, nxr measure/integrals) is nxr::integrals. */
Eigen::VectorXd vertexOutflow(const Complex& c, const Eigen::VectorXd& u);
Eigen::VectorXd vertexCirculation(const Complex& c, const Eigen::VectorXd& w);

/** A vertex density [V x m] → a 2-form [nF x m] (area × the mean of the face's vertices); and back (the area-weighted mean of a vertex's faces). */
Eigen::MatrixXd area2form(const Complex& c, const Eigen::MatrixXd& rho);
Eigen::MatrixXd vertexDensity(const Complex& c, const Eigen::MatrixXd& sigma);

struct Modes {
    Eigen::MatrixXd P;                         // [V x K] the 0-form parts ψ/√2
    Eigen::MatrixXd X2;                        // [F x K] the 2-form parts χ/√2 (0 on an exact mode)
    Eigen::VectorXd mu;                        // [K]
    std::vector<int> kind;                     // [K] 0 exact · 2 coexact
};

Eigen::MatrixXd read1(const Complex& c, const Modes& m, const Eigen::MatrixXd& w);       // [K x m]; 0 on nothing (every mode has a 1-form part)
Eigen::MatrixXd synth1(const Complex& c, const Modes& m, const Eigen::MatrixXd& coef);   // [nE x m]
Eigen::MatrixXd read2(const Complex& c, const Modes& m, const Eigen::MatrixXd& sigma);   // [K x m]; 0 on exact modes
Eigen::MatrixXd synth2(const Complex& c, const Modes& m, const Eigen::MatrixXd& coef);   // [nF x m]
Eigen::MatrixXd potential(const Modes& m, const Eigen::MatrixXd& coef);                  // [V x m] Σ_exact c_k P_k / μ_k

/** THE WEIGHTED POTENTIAL FIT of a 1-form ω on the intrinsic edges (the linearised magnetic Laplacian: each edge's lag counted by its weight):
 *  φ minimising Σ_e ⋆1_e w_e (φ_tip − φ_tail − ω_e)², w [nE] ≥ 0 (e.g. each edge's cross-power, or coherence).
 *  On the whole complex: the sparse solve (d0ᵀ W d0) φ = d0ᵀ W ω, W = ⋆1 w, its constant fixed by a ridge ε ⋆0 (ε = 1e-10 × the mean diagonal / the mean
 *  ⋆0: φ's mean over the mass is ~0). Returns φ [nV]. */
Eigen::VectorXd weightedPotentialMesh(const Complex& c, const Eigen::VectorXd& w, const Eigen::VectorXd& omega);
/** The same fit restricted to EXACT modes: φ = Σ_k a_k P_k over the modes with `use[k]` (exact ones only), the normal equations (Aᵀ W A) a = Aᵀ W ω with
 *  A = d0 P_use [nE x n_use], dense. A band-limited potential. Returns φ [nV]. */
Eigen::VectorXd weightedPotentialModes(const Complex& c, const Modes& m, const std::vector<char>& use, const Eigen::VectorXd& w, const Eigen::VectorXd& omega);
/** A PATTERN in a band's coefficients on the cortex (spec docs/specs/2026-11-13-cycle-pattern.md). `rows` names each coefficient: (mode k, part), part 2 the
 *  normal current's 2-form coefficient (a coexact mode), part 1 the tangent current's 1-form coefficient (any mode); `m` [rows] complex. Returns three
 *  complex vertex fields [V]: S, the stream function of the tangent part's divergence-free (coexact) piece, Σ (c_k / μ_k) P_k; A, the potential of its
 *  curl-free (exact) piece, Σ (c_k / μ_k) P_k; ρ, the normal current, vertexDensity(synth2(a)). Throws std::invalid_argument on a row outside the modes
 *  or a normal row of an exact mode. */
struct PatternFields { Eigen::VectorXcd S, A, rho; };
PatternFields patternFields(const Complex& c, const Modes& m, const std::vector<std::pair<int, int>>& rows, const Eigen::VectorXcd& coef);

/** THE SPLIT OF A PHASE 1-FORM w (each intrinsic edge's lag φ(tip) − φ(tail), rad; e.g. arg T of a cross-spectrum, or arg(x_tip conj x_tail) of a complex
 *  pattern) by inner products with the modes' 1-forms: the exact part (= d of a timing potential: one order of arrival), the coexact part (lags that
 *  circulate round loops), the residual (outside the modes). Shares are of the ⋆1 energy. `potential` φ [V] with d0 φ = the exact part (rad; timing =
 *  φ / ω, a larger value leads for z ∝ e^{+iωt}); `propagation` −∇φ at the vertices [3V] (rad per length unit, sharpened); `stream` the coexact part's
 *  2-form potential at the vertices; `inconsistency` per vertex: over its edges, the share of ⋆1 |w|² the exact part leaves. */
struct PhaseSplit {
  Eigen::VectorXd coef, exact, coexact;      // [K] the coefficients; [nE] the exact and coexact parts
  double shareTiming = 0, shareRotation = 0, shareResidual = 0;
  Eigen::VectorXd potential, stream, inconsistency;   // [V]
  Eigen::MatrixXd propagation;               // [3V x 1]
};
PhaseSplit splitPhase(const Complex& c, const Modes& m, const Eigen::VectorXd& w);
/** The phase 1-form of a complex vertex field x [V]: w_e = arg(x_tip conj x_tail). */
Eigen::VectorXd phaseOneForm(const Complex& c, const Eigen::VectorXcd& x);

}  // namespace nxr::signal

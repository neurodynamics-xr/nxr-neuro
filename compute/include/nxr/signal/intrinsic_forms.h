#pragma once
// nxr::signal — fields on the INTRINSIC complex the Hodge–Dirac modes are solved on, and their reading in the modes' FULL FORMS (spec
// docs/specs/2026-11-12-direction-timing.md;: one triangulation for the modes and everything applied to them). PURE MATHS: arrays in, arrays out.
//
//   IntrinsicComplex      the intrinsic complex as stored by `eigen solve`: its edges (tail, tip), lengths, d0 [E x V], d1 [F x E], the stars, the signposts (each edge's
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
//   projectOneForm        a 1-form → the coefficients c [K x m] in the modes' 1-form parts φ_k (dψ/(μ√2) exact, ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2) coexact), orthonormal in ⋆1 to 1/2:
//                c_k = 2 ⟨ω, φ_k⟩⋆1 — inner products, no solve
//   synthesizeOneForm       coefficients → the 1-form Σ c_k φ_k
//   projectTwoForm        a 2-form → the coefficients a [K x m] in the coexact 2-form parts (⋆2-orthonormal to 1/2): a_k = 2 ⟨σ, χ_k/√2⟩⋆2; synthesizeTwoForm: back
//   exactPotential  the exact part's potential: Σ_exact c_k P_k / μ_k, whose d0 IS the exact part of Σ c_k φ_k (a timing map, from a phase 1-form)
//   coexactStream   the coexact part's stream function: the 2-form potential Σ_coexact c_k X2_k / μ_k at the vertices (vertexDensity)
//   modesOf      the Hodge–Dirac mode solve's result (operators::hodgeDiracModes: rows 0-forms · 1-forms · 2-forms) as Modes
//   phaseCurrent  the phase current Im(z̄ dz) on the edges, J_e = Im(conj(z_tail) z_tip); phaseGradient: the phase step dφ on the edges. The same quantities as
//                 phase_gradient.h's (vectors at the vertices, by a vertex gradient): the discretisation is the operator the caller passes, never the function's
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/eigen.h"

namespace nxr::signal {

struct IntrinsicComplex {
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
void attachEmbedding(IntrinsicComplex& c, const double* positions, const std::int32_t* inputFaces, int nFin);

/** The embedding of the signposts from an geometry::IntrinsicSurface (normals: geometry-central's corner-angle-weighted; per vertex its input edges' raw signpost angles
 *  and directions). `intrinsicComplex` calls it. */
void attachEmbedding(IntrinsicComplex& c, nxr::geometry::IntrinsicSurface& surface);

/** The complex of a surface's arrays (geometry-central's intrinsic Delaunay triangulation, with signposts): its edges, lengths, d0, d1, the stars (⋆0 the mesh's recorded mass, ⋆1 the
 *  cotan weights, ⋆2 = 1 / area), the signposts, and the embedding (`positions` [3 nV], the INPUT `faces` [3 nF_in] for the area-weighted vertex normals).
 *  What `eigen solve` stores, built in memory (tests; a caller with no stored package). */
IntrinsicComplex intrinsicComplex(const nxr::geometry::SurfaceArrays& surface);

/** The direction of edge e at its tail (atTip false) or tip (atTip true), pointing ALONG the edge (tail → tip), as an ambient unit vector. */
Eigen::Vector3d tangentAt(const IntrinsicComplex& c, int e, bool atTip);

/** An ambient vertex field J [3V x m] (rows x y z per vertex) → a 1-form [nE x m] on the intrinsic edges. */
Eigen::MatrixXd flat(const IntrinsicComplex& c, const Eigen::MatrixXd& J);
/** A 1-form [nE x m] → the vertex tangent vectors [3V x m] (least squares over each vertex's edges). */
Eigen::MatrixXd sharp(const IntrinsicComplex& c, const Eigen::MatrixXd& w);

/** THE POINTWISE FORMS of the exact tile integrals of a 1-form on the complex (the discrete Stokes and divergence theorems, additive up the tile tree). A tile's value is the SUM of these over
 *  its vertices: the measures `outflow` and `circulation`, `nxr::measurement::reduce` with measure sum over the tile's vertices (a reduction is never a function here).
 *  vertexOutflow: the OUTFLOW of the vector field whose 1-form is u (u = d0 f for ∇f) out of each vertex's barycentric region = −(d0ᵀ ⋆1 u)_v (the interior edges cancel in a sum over a tile); a tile's mean divergence is its sum over the area.
 *  vertexCirculation: the circulation of w, each face's (d1 w)_f shared equally among its vertices (n/3 of it to each of the 3), so a tile's sum is Σ over faces of (d1 w)_f × (the share of the face's corners in the tile: n/3); a winding number × 2π round a phase defect.
 *  The extrinsic form for a vertex field (a face-constant tangent vector a face, nxr measure/integrals) is nxr::integrals. */
Eigen::VectorXd vertexOutflow(const IntrinsicComplex& c, const Eigen::VectorXd& u);
Eigen::VectorXd vertexCirculation(const IntrinsicComplex& c, const Eigen::VectorXd& w);
/** THE POINTWISE FORM of the ⋆1 ENERGY of a 1-form, Σ_e ⋆1_e w_e² (the squared norm the modes' 1-forms are orthonormal in): each edge's ⋆1_e w_e² shared equally between its two
 *  ends, so the whole complex's energy, and a tile's, is the SUM of its vertices' values (the measure sum over any partition, `nxr::measurement::reduce`; a share of energy is a
 *  ratio of such sums, derived at read). w [nE x m], one column a frame → [V x m]. */
Eigen::MatrixXd vertexEnergy(const IntrinsicComplex& c, const Eigen::MatrixXd& w);

/** A vertex density [V x m] → a 2-form [nF x m] (area × the mean of the face's vertices); and back (the area-weighted mean of a vertex's faces). */
Eigen::MatrixXd area2form(const IntrinsicComplex& c, const Eigen::MatrixXd& rho);
Eigen::MatrixXd vertexDensity(const IntrinsicComplex& c, const Eigen::MatrixXd& sigma);

struct Modes {
    Eigen::MatrixXd P;                         // [V x K] the 0-form parts ψ/√2
    Eigen::MatrixXd X2;                        // [F x K] the 2-form parts χ/√2 (0 on an exact mode)
    Eigen::VectorXd mu;                        // [K]
    std::vector<int> kind;                     // [K] 0 exact · 2 coexact
};

Eigen::MatrixXd projectOneForm(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& w);       // [K x m]; 0 on nothing (every mode has a 1-form part)
Eigen::MatrixXd synthesizeOneForm(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& coef);   // [nE x m]
Eigen::MatrixXd projectTwoForm(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& sigma);   // [K x m]; 0 on exact modes
Eigen::MatrixXd synthesizeTwoForm(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& coef);   // [nF x m]
Eigen::MatrixXd exactPotential(const Modes& m, const Eigen::MatrixXd& coef);                  // [V x m] Σ_exact c_k P_k / μ_k
Eigen::MatrixXd coexactStream(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& coef);  // [V x m] vertexDensity(Σ_coexact c_k X2_k / μ_k)
/** The modes of a Hodge–Dirac solve (positive μ, kinds 0 exact and 2 coexact; nV, nE, nF the pencil's): P the 0-form rows, X2 the 2-form rows.
 *  Throws std::invalid_argument on a result whose rows are not nV + nE + nF, a negative μ (a `bothSigns` solve) or a harmonic mode. */
Modes modesOf(const nxr::operators::EigenResult& r, int nV, int nE, int nF);

/** THE WEIGHTED POTENTIAL FIT of a 1-form ω on the intrinsic edges (the linearised magnetic Laplacian: each edge's lag counted by its weight):
 *  φ minimising Σ_e ⋆1_e w_e (φ_tip − φ_tail − ω_e)², w [nE] ≥ 0 (e.g. each edge's cross-power, or coherence).
 *  On the whole complex: the sparse solve (d0ᵀ W d0) φ = d0ᵀ W ω, W = ⋆1 w, its constant fixed by a ridge ε ⋆0 (ε = 1e-10 × the mean diagonal / the mean
 *  ⋆0: φ's mean over the mass is ~0). Returns φ [nV]. */
Eigen::VectorXd weightedPotentialMesh(const IntrinsicComplex& c, const Eigen::VectorXd& w, const Eigen::VectorXd& omega);
/** The same fit restricted to EXACT modes: φ = Σ_k a_k P_k over the modes with `use[k]` (exact ones only), the normal equations (Aᵀ W A) a = Aᵀ W ω with
 *  A = d0 P_use [nE x n_use], dense. A band-limited potential. Returns φ [nV]. */
Eigen::VectorXd weightedPotentialModes(const IntrinsicComplex& c, const Modes& m, const std::vector<char>& use, const Eigen::VectorXd& w, const Eigen::VectorXd& omega);
/** A PATTERN in a band's coefficients on the cortex (spec docs/specs/2026-11-13-cycle-pattern.md). `rows` names each coefficient: (mode k, part), part 2 the
 *  normal current's 2-form coefficient (a coexact mode), part 1 the tangent current's 1-form coefficient (any mode); `m` [rows] complex. Returns three
 *  complex vertex fields [V]: S, the stream function of the tangent part's divergence-free (coexact) piece, Σ (c_k / μ_k) P_k; A, the potential of its
 *  curl-free (exact) piece, Σ (c_k / μ_k) P_k; ρ, the normal current, vertexDensity(synthesizeTwoForm(a)). Throws std::invalid_argument on a row outside the modes
 *  or a normal row of an exact mode. */
struct PatternFields { Eigen::VectorXcd S, A, rho; };
PatternFields patternFields(const IntrinsicComplex& c, const Modes& m, const std::vector<std::pair<int, int>>& rows, const Eigen::VectorXcd& coef);

/** THE SPLIT OF A PHASE 1-FORM w (each intrinsic edge's lag φ(tip) − φ(tail), rad; e.g. arg T of a cross-spectrum, or `phaseGradient(complex, x)` of a complex
 *  pattern) by inner products with the modes' 1-forms: the exact part (= d of a timing potential: one order of arrival), the coexact part (lags that
 *  circulate round loops); the residual (outside the modes) is w − exact − coexact. POINTWISE only: `potential` φ [V] with d0 φ = the exact part (rad;
 *  timing = φ / ω, a larger value leads for z ∝ e^{+iωt}); `propagation` −∇φ at the vertices [3V] (rad per length unit, sharpened); `stream` the coexact
 *  part's 2-form potential at the vertices. The shares of the ⋆1 energy and the inconsistency are ratios of reductions: `vertexEnergy` of w, of the exact
 *  part, of the coexact part and of w − exact, summed over a partition's members (measures phase_lag_energy, timing_energy, rotation_energy,
 *  timing_misfit_energy), and the derived measures timing_share, rotation_share, residual_share and timing_inconsistency read from those sums. */
struct PhaseSplit {
  Eigen::VectorXd coef, exact, coexact;      // [K] the coefficients; [nE] the exact and coexact parts
  Eigen::VectorXd potential, stream;         // [V]
  Eigen::MatrixXd propagation;               // [3V x 1]
};
PhaseSplit splitPhase(const IntrinsicComplex& c, const Modes& m, const Eigen::VectorXd& w);
/** THE PHASE GRADIENT on the intrinsic edges: the phase step w_e = arg(x_tip conj(x_tail)) (rad, in (−π, π]) of complex vertex fields x [V x m], one column a frame:
 *  dφ integrated along each edge. The same quantity as the vertex `phaseGradient(j, p)` (phase_gradient.h), discretised on the edges of the complex given. [nE x m] */
Eigen::MatrixXd phaseGradient(const IntrinsicComplex& c, const Eigen::MatrixXcd& x);
/** THE PHASE CURRENT on the intrinsic edges: J_e = Im(conj(z_tail) z_tip) = |z_tail| |z_tip| sin(arg z_tip − arg z_tail) of complex vertex fields z [V x m], one
 *  column a frame: Im(z̄ dz) integrated along each edge (tail → tip). The same quantity as the vertex `phaseCurrent(Z, components, G, N)` (phase_gradient.h),
 *  discretised on the edges of the complex given; the two agree through `flat` / `sharp` to second order in the phase step. Which discretisation is the
 *  caller's: the operator it passes. [nE x m] */
Eigen::MatrixXd phaseCurrent(const IntrinsicComplex& c, const Eigen::MatrixXcd& z);

}  // namespace nxr::signal

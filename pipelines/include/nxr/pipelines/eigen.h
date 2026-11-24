// nxr-pipelines — a subject's Hodge–Dirac eigensolve, stored in the four classes (Manifold, Field, Selection, Operator) on the schema's chain-complex design (D92, D93).
// THE VERBS ARE DEFINITIONS: `nxr-neuro run intrinsic` (definitions/intrinsic.yaml, the step `geometry.intrinsic`) makes a surface's INTRINSIC TRIANGULATION, and `nxr-neuro run eigenmodes`
// (definitions/eigenmodes.yaml, the step `operators.hodge_dirac`) solves on it (the runner makes the triangulation first when it is not stored). What stays here: the pure functions the steps
// call (`intrinsic_triangulation`, `solve_hodge_dirac`, which tests call too) and `eigen_solve`, the library's thin caller of the definitions.
// The twin of `nxr eigen solve` (nxr.pipelines.hodge) for a surface of one closed piece (a cortex of two hemispheres is glued first by `run glue`, and the glued surface named): intrinsic Delaunay,
// the split solve (the Laplace–Beltrami ladder d0^T *1 d0 with *0: exact modes; the 2-form ladder *2 d1 *1^-1 d1^T *2 with *2: coexact modes), cut at ONE lambda, the constants out.
//
// THE INTRINSIC TRIANGULATION (its own step, on the surface; a manifold with the surface's geometry and its OWN topology: the flipped edges and faces are its cells):
//     source/<surface>_intrinsic             Manifold  surface    the surface's positions (the same array, not copied), its own edges [nE x 2] (rank 1) and faces [nF x 3] (rank 2)
//     <intrinsic>_d0, _d1                    Operator  analytic   method 'dec': d0 vertices -> edges (cell 0 -> 1), d1 edges -> faces (cell 1 -> 2): no bytes (D93), evaluated from the triangulation
//     <intrinsic>_star0, _star1, _star2      Operator  analytic   method 'dec': the diagonal Hodge stars (cell k -> k*): *0 the mixed-Voronoi mass of the mollified input, *1 the cotan weights, *2 = 1/area
//     <intrinsic>_length, _angle_tail, _angle_tip   Field (cell 1)   the edge lengths (the triangulation's metric) and the signposts at each end
//     <intrinsic>_angle_sum, _reference      Field (cell 0)        cone angles, reference neighbours
//
// THE EIGENSOLVE (stem `hodgeDirac`, or `hodgeDirac__<surface>` for a surface that is not the primary one):
//     eigenvalue/<stem>                      Manifold  points [K]    the EIGENVALUE LINE: mu = sqrt(lambda) > 0, ascending
//     <stem>_eigenvalues                     Field     [K]           lambda = mu^2, a field of the operator (`of_operator`)
//     <stem>_kinds                           Selection partition     the LABELLING of the eigenvalue line by the Hodge part: 0 exact (a 0-form potential) · 2 coexact (a 2-form potential)
//     <stem>_eigenvectors                    Operator  float32 [(V + E + F) x K]   SYNTHESIS: the eigenvalue line (cell 0) -> the intrinsic triangulation (cell key '0+1+2'): each mode's
//                                                                    MIXED FORM, rows stacked 0-forms [0, V) · 1-forms [V, V+E) · 2-forms [V+E, V+E+F) (the Studio's Hodge–Dirac order):
//                                                                    the 0-form part psi/sqrt2 (0 for a coexact mode), the 1-form part d psi/(mu sqrt2) or *1^-1 d1^T *2 chi/(mu sqrt2), the 2-form part chi/sqrt2
//                                                                    (orthonormal under the stars *0, *1, *2). A coexact mode's stream function at the vertices is derived from its 2-form part
//                                                                    (stored_vertex_potentials, input.h), never stored
// One mode a chunk of V rows, so a vertex reader reads the first row chunk of each mode. The analysis (line <- triangulation) is this operator's D106 pair, never stored twice.
// The 1-form parts are *1-orthogonal with energy 1/2 each (test_pipelines), so a 1-form on the intrinsic edges is read into the modes by inner products.
// A solve stored in the earlier form (three operators _eigenvectors, _forms1, _forms2, a kind field, stored d0/d1/stars) is converted in place by `nxr-neuro migrate eigensolve`.
#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/operators/factorization.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/input.h"

namespace nxr::pipelines {

// The solve's result, with no I/O: about `modes` positive modes.
struct HodgeDirac {
  std::vector<double> mu;                    // [K] ascending, 1/length
  std::vector<std::uint8_t> kind;            // [K] 0 exact · 2 coexact
  std::vector<float> eigenvectors;           // [n_vertices x K], C order: each mode's 0-form part psi/sqrt2 (0 for a coexact mode: its stream function is derived from forms2, stored_vertex_potentials)
  std::int64_t n_vertices = 0;
  std::int64_t n_exact = 0, n_coexact = 0;
  double seconds = 0;                        // the solve
  double top = 0;                            // the single lambda cut
  std::int64_t star1_clamped = 0;            // edges whose *1 was raised to the floor (0 without a floor)
  double mollify_epsilon = 0;                // the length mollification added to every edge (0 = none needed)
  std::int64_t flips = 0;                    // edge flips of the intrinsic Delaunay construction
  nxr::io::Json gauge;                       // the triangulation's gauge (G6): {gauge, gauge_record: {mollify, epsilon, flipEps, mass, boundary_vertices, fingerprint}}
  // THE FULL FORMS: the intrinsic complex the modes are solved on, and each mode's 2-form. With the 0-form part (the vertex potential
  // ψ/√2 above) they give both 1-form parts exactly: dψ/(μ√2) (exact) and ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2) (coexact), with d0, d1 read from `edges`, `face_edges`.
  std::vector<std::int32_t> edges;           // [nE x 2] each intrinsic edge's (tail, tip) vertices: d0's orientation (−1 at the tail, +1 at the tip)
  std::vector<double> lengths;               // [nE] the intrinsic edge lengths (input units)
  std::vector<std::int32_t> faces;           // [nF x 3] the intrinsic faces' vertices (two edges may join one pair of vertices: read faces by face_edges)
  std::vector<std::int32_t> face_edges;      // [nF x 3] each face's edges in corner order (corner k → k+1), signed: e + 1 along the edge, −(e + 1) against (d1's signs)
  std::vector<double> star0, star1, star2;   // [nV], [nE], [nF] the Hodge stars the pencil used (⋆0 the mixed-Voronoi mass; ⋆1 after any floor; ⋆2 = 1/area)
  std::vector<double> angle;                 // [nF x 3] the signposts: each halfedge's angle at its tail (corner order), in [0, angle_sum)
  std::vector<double> angle_sum;             // [nV] each vertex's cone angle
  std::vector<std::int32_t> reference;       // [nV] each vertex's reference neighbour (the head of its angle-0 halfedge, an input edge)
  std::vector<float> forms2;                 // [nF x K] each mode's 2-form part χ/√2 on the intrinsic faces (0 for an exact mode)
  std::vector<float> forms1;                 // [nE x K] each mode's 1-form part dψ/(μ√2) (exact) or ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2) (coexact) on the intrinsic edges: the solve's own 1-form rows
  std::int64_t n_modes() const { return static_cast<std::int64_t>(mu.size()); }
};

// THE METHOD. By default the REFERENCE method of nxr (`nxr.eigen.hodge_dirac.solve_hodge`, `nxr.pipelines.hodge.solved_package`) and Rheome
// (`rheome.eigen.hodge_dirac`, `hodge_package`): see nxr-compute's `HodgeSolveOptions` (compute.h) for the rule in full. The methods used before are named
// options, never defaults: shift (absolute sigma | relative ladder), slice (none | window size), normalize (false | true). The *1 floor is a named option too:
// nxr has none, Rheome floors at 1e-6 (the maintainers has not decided); unset here = none. Every option is recorded in the stored result's provenance and
// belongs to the cache key (another option, another cached result).
// The triangulation is THE one Delaunay construction (nxr-compute's delaunay.h, a line port of the Studio's intrinsicDelaunay.ts), with the two numerical
// constants as named, recorded options of the shared names: `mollify` (the mollification margin relative to the mean edge length) and `flip_eps` (an edge is
// flipped while its cotan weight is below -flip_eps); both in the cache key and the provenance.
struct SolveMethod {
  nxr::operators::HodgeSolveOptions solve;   // shift, sigma, slice, normalize (+ convergence controls tol, ncv, maxit)
  std::optional<double> star1_floor;               // floor of *1 (the cotan weights); unset = none
  double mollify = 1e-6;                           // delta = mollify x the mean edge length (the Studio's `mollify`)
  double flip_eps = 1e-6;                          // the Studio's `flipEps`
};

// THE INTRINSIC TRIANGULATION of a surface (no rows): THE one Delaunay construction (mollify, flip to Delaunay) the solve is built on, its cells and their signposts.
struct IntrinsicTriangulation {
  std::int64_t n_vertices = 0;
  std::vector<std::int32_t> edges;           // [nE x 2] each edge's (tail, tip): d0's orientation (−1 at the tail, +1 at the tip)
  std::vector<std::int32_t> faces;           // [nF x 3] the faces' vertices (two edges may join one pair of vertices: read the incidence from face_edges)
  std::vector<std::int32_t> face_edges;      // [nF x 3] each face's edges in corner order, signed: e + 1 along the edge, −(e + 1) against (d1's signs)
  std::vector<double> lengths;               // [nE] the edge lengths (input units): the triangulation's metric
  std::vector<double> angle;                 // [nF x 3] the signposts: each halfedge's angle at its tail (corner order)
  std::vector<double> angle_sum;             // [nV] each vertex's cone angle
  std::vector<std::int32_t> reference;       // [nV] each vertex's reference neighbour (the head of its angle-0 halfedge, an input edge)
  double mollify = 1e-6, flip_eps = 1e-6;    // the construction's options
  double mollify_epsilon = 0;                // the length mollification added to every edge
  std::int64_t flips = 0;
  nxr::io::Json gauge;                       // {gauge, gauge_record}
  std::int64_t n_edges() const { return static_cast<std::int64_t>(lengths.size()); }
  std::int64_t n_faces() const { return static_cast<std::int64_t>(faces.size() / 3); }
};
IntrinsicTriangulation intrinsic_triangulation(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, double mollify = 1e-6, double flip_eps = 1e-6);

// nxr-compute, no rows: the surface (vertices [nV x 3] float64, faces [nF x 3]) of one closed piece -> its Hodge–Dirac modes.
HodgeDirac solve_hodge_dirac(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, int modes, const SolveMethod& method = {});

struct EigenSolveOptions {
  int modes = 3000;                          // about this many positive modes, half exact and half coexact
  std::optional<std::string> surface;        // a surface by name (default: the primary)
  SolveMethod method;                        // the reference method unless an option says otherwise
  bool replace = false;                      // on a miss with a package already stored: replace it (its arrays moved aside, never deleted)
};

struct EigenSolveResult {
  bool hit = false;                          // the stored result was reused
  double seconds = 0;                        // what the call took (the load, or the solve and the store)
  nxr::io::Json info;                        // nxr's `solve_and_store` record (axis, surface, rows, n_modes, n_exact, n_coexact, mu, support, operator_id …)
  HodgeDirac modes;                          // mu, the kinds and the eigenvectors' vertex rows (each mode's 0-form part) as stored
  std::vector<float> vertex_potentials;      // [n_vertices x K]: what a vertex reader draws (stored_vertex_potentials): the 0-form part of an exact mode, the stream function of a coexact one
};

// The caller of the definition `eigenmodes` (`glued` chosen from the surface's pieces): its run finds the stored package whose producer record equals the job's key (the surface's mesh key and
// positions, the options, nxr-compute's version) and reuses it, or solves and stores it (rows and arrays together). The result reads the stored mu, kind and eigenvectors back (`modes`; the
// complex and the 2-form parts are stored, not returned). Progress lines (`{"stage": ...}`) go to `progress`.
EigenSolveResult eigen_solve(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const EigenSolveOptions& opt = {},
                             const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

// nxr-pipelines — a subject's Hodge–Dirac eigenmodes. THE VERB IS A DEFINITION (definitions/eigenmodes.yaml: `nxr-neuro run eigenmodes`, the registered step `hodge_dirac` in eigen_steps.cpp, stored by
// io's declared outputs: a mode axis, cell complexes, sparse operators, fields on cells). What stays here: `solve_hodge_dirac` (the pure solve the step calls, and tests call) and `eigen_solve`, the library's
// thin caller of the definition (the spatial bands make missing eigenmodes with it, `dataset create --modes` the default subject's). The stored layout below is unchanged.
// Before the conversion: read from its stored surface (nxr-io), solved (nxr-compute), stored (nxr-io), CACHED.
// The twin of `nxr eigen solve` (nxr.pipelines.hodge: solve / solved_package, nxr.io.eigen: store_package) for a surface of one closed piece, or a cortex glued by anatomy (below):
// intrinsic Delaunay, the split solve — the Laplace–Beltrami ladder (d0^T *1 d0, *0: exact modes) and the 2-form ladder (*2 d1 *1^-1 d1^T *2, *2:
// coexact modes), cut at ONE lambda, the constants out — and the package stored in nxr's layout (see `nxr/io/eigen.py`):
//
//     eigenvalue/hodgeDirac           Manifold  points [K]       mu = sqrt(lambda) > 0, ascending
//     eigenvalue/hodgeDirac_kind      Field     uint8  [K]       0 exact · 2 coexact
//     hodgeDirac_eigenvectors         Operator  float32 [V x K]  axis -> the surface: per mode ONE vertex potential (psi/sqrt2, or <chi/sqrt2, w_v>)
//     hodgeDirac_support              Selection indices          the surface's vertices the solve covers (all of them for one piece)
//
// (another surface of the subject: the same names with `__<surface>` after `hodgeDirac`). A surface of TWO pieces (a cortex's hemispheres) is GLUED BY
// ANATOMY first (nxr.pipelines.hodge.solve; the Studio's glue.ts, nxr-compute's glueMesh): each medial wall (its stored Desikan-Killiany atlas'
// `unassigned`) cut as a disk, a bridge grown between the rims; the solve is on the glued surface, stored as nxr stores one:
//
//     source/<surface>_glued          Manifold  surface           the kept vertices (in support order), then the bridge's
//     source/<surface>_glued_bridge   Selection indices           the bridge (not cortex)
//     hodgeDirac_eigenvectors         Operator  [V_glued x K]     onto the glued surface; hodgeDirac_support the input's kept vertices
//
// The medial wall belongs to the cache key. Three or more pieces are refused.
//
// THE FULL FORMS are stored with the package: the INTRINSIC COMPLEX the modes are solved on (the operator mesh: the same vertices, its own
// edges and faces; two edges may join one pair of vertices, so it is read by its incidence, never by vertex pairs) and each mode's 2-form:
//
//     source/<surface>[_glued]_intrinsic          Manifold  surface          its vertices, edges [nE x 2] (rank-1 cells) and faces [nF x 3] (rank-2 cells)
//     <intrinsic>_d0, <intrinsic>_d1               Operator  csr              d0: vertices → edges (cell 0 → 1), d1: edges → faces (cell 1 → 2)
//     <intrinsic>_length, _star1, _angle_tail, _angle_tip   Field (cell 1)    edge lengths, ⋆1 (cotan weights, ≥ 0), the signposts at each end
//     <intrinsic>_star0, _angle_sum, _reference   Field (cell 0)    ⋆0 (the mixed-Voronoi mass), cone angles, reference neighbours
//     <intrinsic>_star2                            Field (cell 2)    ⋆2 = 1 / area
//     hodgeDirac_forms2                            Operator  [nF x K]         axis → the faces (cell 2): each mode's 2-form part χ/√2 (0 on an exact mode)
//
// With the 0-form parts (hodgeDirac_eigenvectors: ψ/√2) both 1-form parts follow exactly: dψ/(μ√2) and ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2); they are ⋆1-orthogonal with
// energy 1/2 each (test_pipelines), so a 1-form on the intrinsic edges is read into the modes by inner products.
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
  std::vector<float> eigenvectors;           // [n_vertices x K], C order: the vertex potentials
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
  HodgeDirac modes;
};

// The caller of the definition `eigenmodes` (`glued` chosen from the surface's pieces): its run finds the stored package whose producer record equals the job's key (the surface's mesh key and
// positions, the options, nxr-compute's version) and reuses it, or solves and stores it (rows and arrays together). The result reads the stored mu, kind and eigenvectors back (`modes`; the
// complex and the 2-form parts are stored, not returned). Progress lines (`{"stage": ...}`) go to `progress`.
EigenSolveResult eigen_solve(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const EigenSolveOptions& opt = {},
                             const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

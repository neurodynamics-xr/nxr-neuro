// nxr-pipelines — THE SPATIAL BAND OF A TILE LEVEL, shared by `patches compute` (a static map) and the `apply_kernel` step (a kernel, phase 2a): the level l of a surface's tile ladder
// IS its Weyl band, member l of `weyl(mean piece area, l + 1)`, its λ range the one the ladder records for the level (checked on every use: a ladder
// made with another area is refused). The band is a linear map F_l of the surface's fields, by the filterbank's transforms.
//
// THE TRANSFORM: "default spatial transform is stored eigenmodes when they exist, chebyshev fallback but not silently. if for some reason eigenmodes are not there, we can
// compute them and there should be some sort of message. we are eigenmode first, chebyshev second". So `auto` (the default):
//   1. the subject's STORED Hodge–Dirac package of the surface reaches the band (its largest exact eigenvalue ≥ the band's top, the Meyer edge included; for the Mexican hat, where its gain falls under 1e-3): F_l = Φ g(λ) Φᵀ ⋆0 from the stored
//      exact (Laplace–Beltrami) modes, on the package's glued surface (the bridge carries 0; the cut medial wall's vertices get 0);
//   2. no package: the modes are COMPUTED (`eigen_solve`: enough for the band by Weyl's law), STORED as `eigen solve` stores them, and said so (progress and provenance);
//   3. a package that does not reach the band (a fine level), or modes that cannot be computed: CHEBYSHEV, announced (progress, `warnings` of the provenance), never silent.
// `eigen` forces 1 or 2 and refuses (with the reason) what cannot reach; `chebyshev` is the series in the cotangent pencil of the intrinsic Delaunay mesh (no eigenvectors); `exact` is the dense
// eigen-decomposition of the input mesh itself (at most 4096 vertices): the reference the tests compare against, not a way to run a cortex.
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/signal/graphfilterbank.h"
#include "nxr/io.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/task.h"

namespace nxr::pipelines {

/** What the stored ladder says of level l of a surface: the tiles of every vertex, their areas, the tile graph, and the Weyl member that is the level's band (paired and checked). */
struct SurfaceLevel {
  task::Node ladder, level;                  // the ladder Selection and the level's `set` member
  nxr::io::Json record;                      // the level's params_json
  std::string surface, tiling;               // the surface's name; "<surface> tiles L<l>"
  int l = 0;
  std::int64_t nodes = 0;                    // tiles of the level (pieces × 2^l)
  std::vector<std::int32_t> tile;            // [nV] the tile of every vertex (-1: none)
  std::vector<double> tile_area;             // [nodes] m²
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;   // the tiles joined by a mesh edge (a < b)
  double piece_area = 0;                     // the mean connected piece's area (m²): Weyl's A
  nxr::signal::graphfilterbank::GraphFilterbank bank;                // weyl(piece_area, max(1, l + 1))
  int member = 0;                            // = l: the level's band
};
/** Reads and checks (refusing with the reason, as the pipelines do): a ladder of the surface exists, the level exists, its arrays fit the surface, its band is Weyl member l. */
SurfaceLevel read_surface_level(nxr::io::SubjectWriter& sub, const SurfaceInput& surface, int level);

struct SpatialBandOptions {
  std::string spatial_shape = "meyer";      // the level's gain: meyer = the level IS its Weyl band (compact edges, the frame: levels add up) | mexhat = the Mexican hat g = (λ/c)e^(1−λ/c), unit peak at c, the
                                             // GEOMETRIC centre sqrt(lambda_min · lambda_max) of the level's Weyl band (level 0, whose band starts at 0: hi/√2, as if it began one octave below), nxr `vortex_cores.hat_operator`;
                                             // it has a long tail: the eigen route needs modes up to where the gain is under 1e-3 (`reach`), else the Chebyshev series, announced. The library default stays meyer (patches, cellquadrature);
                                             // the `apply_kernel` step defaults to mexhat.
  std::string transform = "auto";           // auto (eigenmode first, chebyshev an announced fallback) | eigen | chebyshev | exact (dense modes of the input mesh, at most 4096 vertices)
  int order = 0;                             // chebyshev degree; 0: the smallest of 40, 80, … 5120 whose series is within `tolerance` of the band's gain
  double tolerance = 0.01;                   // the largest |gain − series| accepted (a band the series cannot follow is refused, never used)
  int modes = 0;                             // when the package is COMPUTED: its size (about half exact, half coexact); 0: the standard 3000, or more when the band needs it by Weyl's law
  bool riesz = false;                        // also build the band's RIESZ part R_l = ∇ Δ^{-1/2} F_l (the spatial analytic signal, nxr/signal/riesz.h): the same transform with the gain g/√λ, then the mesh gradient; eigenmode first, Chebyshev second, as the band
  bool glued = false;                        // (one cortex): ALSO evaluate the band on EVERY vertex of the glued surface (the kept vertices and the bridge, the corpus callosum included), `apply_glued`. The input is still the
                                             // native vertices (zero on the bridge before the band: the band fills it); eigenmodes first (the glued modes, as stored), Chebyshev or the dense modes on the glued mesh as the announced fallback
                                             // (never on the native mesh: that has no bridge). Needs the glued surface stored (`eigen solve`), or a one-piece surface. Not with `riesz`
  int max_modes = 8000;                      // the largest package `auto` computes (memory: glued vertices × modes × 4 bytes stored and read); a band needing more is Chebyshev, announced
};
struct SpatialBand {
  nxr::io::Json info;                        // level, member, role, λ range, wavelength, transform, source, the modes (package, count, λmax) or the degree, warnings, messages: what the stored result records
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> apply;   // F_l on columns [nV × n] → [nV × n]
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> apply_glued;   // only with `glued`: F_l on the native columns [nV × n] → every glued vertex [Vg × n] (the kept vertices first, in `support` order, then the bridge's); `apply` is its read-back on the kept vertices
  std::string glued_surface;                 // only with `glued`: the surface `apply_glued` lives on (its name; the input itself for one closed piece)
  std::int64_t glued_vertices = 0;           // and its vertex count
  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> apply_riesz;   // only with `riesz`: R_l on columns [nV × n] → [3 nV × n] (x y z a vertex: the tangent vector field ∇ Δ^{-1/2} F_l f, per column); 0 at the constant mode
  std::vector<std::string> warnings, messages;   // a fallback and a computation, said in words (also in `info`)
};

/** WHAT THE TRANSFORM WILL BE, decided from the request and what is stored, reading only (no eigenvector array, nothing written): the same on the same state, so it can enter a cache key before any
 *  work. Refuses a forced `eigen` that cannot reach the band, with the reason. */
struct SpatialBandPlan {
  std::string transform;                     // eigen | chebyshev | exact: what will be used
  std::string source;                        // eigen: stored | computed
  std::string package;                       // eigen: the package's stem
  nxr::io::Json key;                         // what the cache key takes: the transform, and the package's name or the series' request (order, tolerance)
  std::int64_t compute_modes = 0;            // eigen, computed: the package's size to solve
  double reach = 0;                          // the band's top λ with its Meyer edge, or where the hat's gain is under 1e-3 (1/m²)
  std::vector<std::string> warnings, messages;
};
SpatialBandPlan plan_spatial_band(nxr::io::SubjectWriter& sub, const SurfaceInput& surface, const SurfaceLevel& level, const SpatialBandOptions& opt);

/** Makes the package a plan says it will compute (`plan.source == "computed"`): the `eigenmodes` pipeline run on its own, with its own provenance and cache key. Gives {package, modes, exact, coexact, seconds, hit}. */
nxr::io::Json make_eigenmodes(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& surface, const SpatialBandPlan& plan, const std::function<void(const nxr::io::Json&)>& progress = {});
/** Builds the band: plans it, computes and stores the package when the plan says so (said in `progress` and in the result), builds the operators once. */
SpatialBand make_spatial_band(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurfaceInput& surface, const SurfaceLevel& level, const SpatialBandOptions& opt,
                              const std::function<void(const nxr::io::Json&)>& progress = {});
/** The two transforms that need no subject (chebyshev, exact); `auto` and `eigen` are refused (they read and may write the subject's package). */
SpatialBand make_spatial_band(const SurfaceInput& surface, const SurfaceLevel& level, const SpatialBandOptions& opt);

}  // namespace nxr::pipelines

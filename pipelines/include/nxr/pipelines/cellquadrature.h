// nxr-pipelines — QUADRATURE PER CELL AND PER BURST: is the activity of a joint cell, or of a global burst, travelling or standing? (spec docs/specs/2026-11-13-cortical-events.md, "Phase 2d").
// From STORED nodes only, no recording pass: the analytic cross-spectrum's per-time-tile sensor Σ z zᴴ (`crossspectrum compute --estimator analytic`), the inverse kernel, the joint Selection of the pair
// (spatial level, time level) and, for bursts, a spans Selection of them.
//
// THE GRAPH (the paramount architecture: loading · preparation and type checks · a graph of pure nodes · readback):
//   loading      the cross-spectrum's tiles on the kernel's channels, in its order (io, by name)                                      ℂ [tiles × m × m]
//   preparation  type checks (the cross-spectrum's tile level IS the joint pair's time level; same recording; the kernel's channels; the surface), then M_l = (F_l ⊗ I) K bound ONCE (spatial_band.h)
//   graph        congruence_norms    per spatial tile s and time tile t: (‖T‖², ‖Im T‖²), T = B_s C_t B_sᵀ (B_s the tile's rows of M_l, ALL components), without forming T
//                sum_over_members    per burst: C_b = Σ_{t in the burst} C_t  (products add: exact), then congruence_norms per spatial tile
//   readback     measurements over the joint cells and over the bursts: the ingredients `cross_spectrum_norm2` and `cross_spectrum_quadrature_norm2`; the quadrature fraction is DERIVED at read
//
// WHAT IS STORED (the ingredients are SUMS of products; the fraction ‖Im T‖/‖T‖ is their ratio, derived at read, never stored, never reduced):
//   over the JOINT cells (rows of the recording on the joint Selection): an array beside the recording [spatial tiles × time tiles × 2] float64 (the two ingredients), NaN where a time tile is not stored
//   over the BURSTS (rows of the recording on the bursts Selection): [spatial tiles × bursts × 2] float64, NaN for a burst whose time tiles are not all stored
//   `computed_by` = the pipeline and a hash of its key; the recipe (every option, the inputs, what the spatial band resolved) is an entry of the dataset's `recipes` dictionary.
// THE TILE PAIRS (tile_pairs, default on): for each pair (s, s') of tiles of the level joined by a mesh edge (tileAdjacency; the order is the tiles' pair list, stored beside as int32 [pairs × 2]), the ingredients
//   of T_ss' = B_s C B_s'ᵀ, `tile_pair_cross_spectrum_norm2` and `tile_pair_cross_spectrum_quadrature_norm2`: an array beside the recording [pairs × time tiles × 2] over the joint cells and [pairs × bursts × 2] over the
//   bursts (rows = pairs, members = the Selection's), same NaN and per-burst rules; the pair's quadrature fraction is DERIVED at read (`tile_pair_quadrature_fraction`). T_s's = T_ss'ᴴ: a pair is listed once (s < s').
// AREA WEIGHTS (`weights`, default area): every sum is Σ a_i a_j |T_ij|² with the barycentric vertex areas (pairs: a_i of tile s, a_j of s'); `none` is the unweighted behaviour (named option, not a default).
// ROLL-UPS: over time the burst's quantity is made from the SUMMED C, never from the cells' ingredients (Σ_t ‖T_t‖² ≠ ‖T of Σ_t C_t‖²); over space the ingredients of a union of tiles are NOT the sum of the
// tiles' (the cross-tile blocks are missing). A re-run with the same inputs and options is a hit; other rows of the same Selection and measure are refused unless `replace`.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct CellQuadratureOptions {
  std::string cross_spectrum;                // REQUIRED: a stored analytic cross-spectrum `<stem>_<band>_windows` (the tiles of Σ z zᴴ, `crossspectrum compute --estimator analytic`)
  int space_level = -1;                      // REQUIRED: the surface ladder's level; its Weyl band is the spatial band, its tiles the spatial partition
  std::optional<std::string> kernel;         // the inverse kernel (free or fixed orientation) by name; default: the subject's only one
  std::optional<std::string> joint;          // the joint Selection by name; default: the one of the cross-spectrum's recording, this space level and the cross-spectrum's tile level (refused if there is none)
  std::optional<std::string> bursts;         // a spans Selection of bursts on the recording's time line (`run global_bursts`'s `bursts`): also made per burst (default: the cells only)
  // AREA WEIGHTS (2026-11-13): "area" (default) = a tile's vertices count by the area they stand for (the barycentric vertex areas the tiles' areas are sums of):
  // Σ_ij a_i a_j |T_ij|², so the value does not change when a region is sampled more finely; "none" = every vertex counts equally (the behaviour before this option; its rows and cache key are the old ones).
  std::string weights = "area";
  // TILE PAIRS: also the cross-tile blocks T_ss' = B_s C B_s'ᵀ of the level's touching tiles (nxr::ladder::tileAdjacency): a wave running from one tile into the next is there, not in the within-tile blocks.
  // Stored as `tile_pair_cross_spectrum_norm2` and `tile_pair_cross_spectrum_quadrature_norm2`; off = the within-tile ingredients only.
  bool tile_pairs = true;
  // THE SPATIAL TRANSFORM (spatial_band.h;: eigenmodes first, Chebyshev second)
  std::string transform = "auto";
  int modes = 0;
  int order = 0;
  double tolerance = 0.01;
  // MEMORY: M_l [vertices × components × channels] float64 (a 20k-vertex free cortex and 300 channels: 144 MB) beside the kernel as stored; the cross-spectrum's tiles [tiles × m × m] complex (600 tiles and
  // 270 channels: 700 MB); one channels × channels Gram per spatial tile at a time. TIME: two channels × channels products per (spatial tile, time tile).
  bool replace = false;                      // other rows of the same measures are moved aside (deleted rows; the arrays are rewritten)
};
struct CellQuadratureResult {
  nxr::io::Json info;
  bool cells_hit = false, bursts_hit = false;
  bool cell_pairs_hit = false, burst_pairs_hit = false;   // only meaningful when tile_pairs
};
CellQuadratureResult cellquadrature_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const CellQuadratureOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

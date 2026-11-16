// nxr-pipelines — PATCHES on a static cortical map: the spatial equivalent of the bursts of a band (spec docs/specs/2026-11-13-cortical-events.md, phase 1).
// Where a map (a PET map, any scalar Field on a surface) stands out from its surroundings at the size of a level of the surface's tile ladder.
//
// THE GRAPH (the paramount architecture: loading · preparation and type checks · a graph of pure nodes · readback). Preparation reads the map, the surface and
// the level's tiling from their stored rows (a refusal, with its reason, before any maths), binds the mesh operators and the level's tile graph once; the nodes are
// pure functions of matrices (compute/); the results go back to io:
//
//   map (ℝ per vertex)
//     → spatial_band    the level's Weyl band: member l of nxr::filterbank::weyl, paired with level l by construction (the member's band IS the ladder's
//                       lambda_min … lambda_max of the level, checked)               ℝ per vertex → ℝ per vertex     MATERIALIZED: the Field `<map>_band_L<l>`
//     → measure         per tile of the level, the area-weighted mean, max, min, abs_max (nxr::partition, `measure compute`)           → ℝ per tile    REDUCED: measurements of
//                       the band Field over the level's tiling `<surface> tiles L<l>`
//     → summarize       the baseline across the level's tiles: median and the robust SD (1.4826 MAD)                    → a statistic          REDUCED: `<map>_band_L<l>_baseline`
//     → sign            positive (x) · negative (−x) · magnitude (|x|): the caller's, applied before the thresholds (nxr::regions::applySign)
//     → detect_regions  hysteresis on the tile graph (nxr::regions::detectRegions): tiles ≥ median + exit_sd·SD form connected regions over the pairs of tiles joined
//                       by a mesh edge (the level's stored adjacency); a region is kept when a tile reaches median + enter_sd·SD and its area is within the limits
//                       → a labelling of the tiles                                                   REDUCED: the `set` Selection `<map> patches L<l> <sign>`
//     → measure         each patch's mean, max, min, abs_max over the patches Selection (the exact roll-up of its tiles: a union of tiles measured as one member), its
//                       area and vertex count (geometry), n_tiles and centre_vertex (the tile count and the vertex nearest its area-weighted centroid)
//     → summarize       the statistics across the patches: count, sum, median, mean, sd, percentiles of area, n_tiles, mean and peak
//
// WHAT IS STORED (every name and key below is in the rows):
//   `<map>_band_L<l>`            Field (kind map, float64, the map's unit): the band-passed map. The tile and patch measurements are OF this Field, never of the map, so a plain
//                                `measure compute` of the map over the same tiling (its own rows, keyed by the map) is never overwritten.
//   rows on `<surface> tiles L<l>`   mean · max · min · abs_max per tile, and the tiles' area and n_elements, by the recipe `patches-barycentric` (weights: barycentric areas)
//   `<map>_band_L<l>_baseline`   statistic (median, mad) of the tile means
//   `<map> patches L<l> <sign>`   Selection `set` on the surface: a code per vertex = its patch's ordinal (by its lowest tile), -1 in none; dictionary "patch k" with
//                                attributes (tiles, centre vertex); params_json: pipeline, level, band (the member, its λ range, the transform), sign, thresholds, the baseline, the cache key
//   rows on the patches          mean · max · min · abs_max (of the band Field), area · n_elements, and n_tiles · centre_vertex (nxr-neuro's own measures)
//   `<map>_patches_L<l>_<sign>_<what>`  statistics across the patches of area, n_tiles, mean, peak (max for positive, min for negative, abs_max for magnitude)
//
// A re-run with the same inputs and options is a cache hit at every stage; another choice at a stage whose result exists is refused unless `replace`.
// What it does NOT do: it builds no instrument limit into the detection (a resolution, a leakage, a smallest size: the caller's `min_area`, named); it does not merge
// patches that do not touch; it finds events of ONE level (a coarser level is its own run).
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct PatchesComputeOptions {
  std::string map;                           // REQUIRED: the stored Field to find patches in (a scalar per vertex of a surface); no non-finite value
  int level = -1;                            // REQUIRED: the ladder level (tiles 2^l per piece); its Weyl band is the map's filter. The ladder (`tiles compute`) must exist
  std::string sign = "positive";             // positive (bumps: high values) · negative (dips) · magnitude (|x|): applied to the tile means before the thresholds
  double enter_sd = 1.5;                     // a region must reach median + enter_sd robust SDs (Rheome's labelalpha hiSD, as band_bursts)
  double exit_sd = 1.0;                      // and extends over tiles ≥ median + exit_sd robust SDs (loSD)
  std::optional<double> min_area;            // m²: the smallest patch kept (default none); the caller's size, never an instrument's
  std::optional<double> max_area;            // m²: the largest kept (default none)
  // THE SPATIAL TRANSFORM: spatial_band.h
  std::string transform = "auto";            // auto (stored eigenmodes; else computed and stored, said; else chebyshev, announced) | eigen | chebyshev (the series in the pencil) | exact (dense modes of the mesh, at most 4096 vertices)
  int modes = 0;                             // when auto/eigen COMPUTES the package: its size; 0: the standard 3000, or more when the band needs it
  int order = 0;                             // chebyshev: the series degree; 0: the smallest of 40, 80, … 5120 whose series is within `tolerance` of the band's gain
  double tolerance = 0.01;                   // chebyshev: the largest |gain − series| accepted (a band the series cannot follow is refused, never used)
  std::vector<std::string> summaries = {"count", "sum", "median", "mean", "sd", "percentile:25", "percentile:75", "percentile:90"};   // across the patches (band_bursts' list)
  bool replace = false;                      // another choice at a stage whose result exists is moved aside and rewritten
};
struct PatchesComputeResult {
  nxr::io::Json info;
  bool hit = false;                          // the patches Selection was found by its key
};
PatchesComputeResult patches_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PatchesComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

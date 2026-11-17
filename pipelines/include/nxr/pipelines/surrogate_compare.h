// nxr-pipelines — THE SURROGATE COMPARISONS (spec docs/specs/2026-11-17-surrogates.md, sections 3 and 7): `run surrogate_lagfree_vortex_compare` and `run surrogate_lagfree_spatial_events_compare`.
// ORCHESTRATION ONLY: for i = 0, 1, … the draw definition is run with `seed: i` through the runner (each draw a typed, cached run of its own), the draw's small statistic is read through io and checked
// against the data's (the SAME selection, the same generator), and folded into a tally by nxr::nulltest; after each batch the stopping rule is asked (nxr::nulltest::undecidedMembers): BATCHES with EARLY
// STOPPING (Besag and Clifford 1991: batches of 20, at least 20, at most 100, h = 10 exceedances). Every number is compute's (nxr::nulltest, nxr::crossings); the loop over draws is the only thing here.
//
// surrogate_lagfree_vortex_compare (the draws of `surrogate_lagfree_vortex`; the data's crossings of `run vortex_cores`), per tile of the level the crossings were measured over, the two senses pooled, the
// stage asked for (a tile counts with at least `min_crossings` crossings in the data):
//   consistency   the unbiased rho2 = (|S|^2 - n) / (n (n - 1)) of the tile's pooled sum S of n crossings (never R), one-sided: the draws at least as large as the data's
//   repeat        with two recordings: the dot product of the two runs' mean resultants, draw i of one run against draw i of the other, and the count-weighted mean cosine over the tiles
// STORED (statistics of the first recording's Field, OVER the level's Selection: partition_selection_id):
//   surrogate_lagfree_vortex_compare.<rec>.consistency_p   [tiles x (p, q, exceedances)]   measure null_p, method besag_clifford
//   surrogate_lagfree_vortex_compare.<rec>.crossing_ratio  [tiles]                         the data's crossings over the surrogate's mean number, measure crossing_ratio
//   surrogate_lagfree_vortex_compare.<rec>.resultant       [tiles x (x, y, z, mean crossings)]   the surrogate's mean resultant per tile (the group subtraction), measure null_resultant
//   vortex_cores.run_repeat.<a>.<b>.p                      [tiles x (p, q, exceedances)]   and .whole_p [(p, exceedances, draws)] for the count-weighted cosine
// SEEDS: the white noise is a function of the seed, so the two runs' draws are made independent: run A's draw i is seed i, run B's seed i + 100000.
//
// surrogate_lagfree_spatial_events_compare (the draws of `surrogate_lagfree_spatial_events`; the data's `spatial_events.home.<rec>` and `.summary.<rec>`):
//   home occupancy   per glued vertex, the fraction of the burst frames it lies in a patch, one-sided (a vertex the data never puts in a patch has no test)
//   size             per entry of the patch summary: the share of the patches at each characteristic LEVEL and in each AREA bin, and the patches per frame: two-sided
// STORED: surrogate_lagfree_spatial_events_compare.<rec>.home_p [vertices x (p, q, exceedances)], .home_null [vertices], .size_null [entries x (data, null mean, p_upper, p_lower, q_upper, q_lower)].
//
// WHAT IT CANNOT SAY: the surrogate has the data's zero-lag covariance, spectrum and envelope and nothing else: a member beyond it is beyond what mixing and amplitude modulation produce under the same
// threshold, burst frames and baseline (the data's), never by itself proof of a travelling wave. Draws that exist are reused (a draw is its own cached node), so a rerun with a larger `max_draws`
// continues where the last stopped.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct SurrogateCompareResult { nxr::io::Json info; bool hit = false; };

struct SurrogateVortexCompareOptions {
  std::string recording_a;                   // REQUIRED: a recording whose `tile_crossings` (`run vortex_cores`) are compared
  std::optional<std::string> recording_b;    // a second recording: the run-to-run repeat's comparison as well (draws in step)
  std::optional<std::string> over;           // the tile level's Selection (`<surface> tiles L<l>`) the crossings were measured over; default: the one the recordings have
  int stage = 0;                             // the stage by index (rows 2s, 2s+1 of the stored crossings) ...
  std::optional<std::string> stage_name;     // ... or by name (burst, onset, within, offset): each run's own rows name them
  nxr::io::Json set = nxr::io::Json::object();   // the `surrogate_lagfree_vortex` parameters of every draw (`{recording}` stands for each recording's name): space_level, cross_spectrum, band, transform, bursts, ...
  std::optional<std::string> chunks;         // what each draw spans: chunk "k", chunks "a-b" or "all" ...
  std::optional<double> start, stop;         // ... or [start, stop) seconds (both). REQUIRED: the caller states the span; none is an error
  int batch = 20, min_draws = 20, max_draws = 100;   // the draws are made in batches of `batch` (a member of the draw line's grid); at least `min_draws`, at most `max_draws`
  int h = 10;                                // Besag-Clifford: a tile is decided at h exceedances
  int min_crossings = 10;                    // a tile counts with at least this many crossings
  bool replace = false;                      // replace the stored results, and any draw stored with other parameters (moved aside, never deleted); a draw that is the same is a hit and kept
};
SurrogateCompareResult surrogate_lagfree_vortex_compare(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurrogateVortexCompareOptions& opt,
                                                        const std::function<void(const nxr::io::Json&)>& progress = {});

struct SurrogateSpatialEventsCompareOptions {
  std::string recording;                     // REQUIRED: the recording whose stored `spatial_events.home` and `.summary` (`run spatial_events`) are compared
  std::string events = "spatial_events";     // the pipeline the data's statistics are named by (`<events>.home.<recording>`, `<events>.summary.<recording>`)
  nxr::io::Json set = nxr::io::Json::object();   // the `surrogate_lagfree_spatial_events` parameters of every draw (the data's: levels, reference_level, threshold, bursts, cross_spectrum, band, ...)
  std::optional<std::string> chunks;         // what each draw spans: chunk "k", chunks "a-b" or "all" ...
  std::optional<double> start, stop;         // ... or [start, stop) seconds (both). REQUIRED: the caller states the span; none is an error
  int batch = 20, min_draws = 20, max_draws = 100;
  int h = 10;
  bool replace = false;                      // replace the stored results, and any draw stored with other parameters (moved aside, never deleted)
};
SurrogateCompareResult surrogate_lagfree_spatial_events_compare(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurrogateSpatialEventsCompareOptions& opt,
                                                                const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

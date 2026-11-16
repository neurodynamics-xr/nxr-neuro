// nxr-pipelines — THE LAG-FREE NULL OF THE SPATIAL EVENTS (`spatialnull compute`; spec docs/specs/2026-11-14-spatial-events.md, "Statistics and nulls"; vortex-cores spec, piece 10). Orchestration only, as `null compute`: runs the draws of
// `spatial_events_null` (the spatial-events graph on a lag-free surrogate of the recording, one run per seed; the baseline, threshold and burst frames the data's), reads each draw's small statistics (its home map and its patch summary) and tests the data's against them,
// in BATCHES with EARLY STOPPING (Besag and Clifford 1991: batches of 20, at least 20, at most 100, h = 10 exceedances).
//
// THE TESTS:
//   home occupancy   per glued vertex, the fraction of the burst frames it lies in a patch (the data's `spatial_events.home.<rec>`) against each draw's: one-sided, the draws at least as large as the data's (a vertex with no occupancy in the data has no test)
//   size             per entry of the patch summary: the share of the patches at each characteristic LEVEL and in each AREA bin (the size distribution), and the patches per frame: two-sided, the draws at least as large and at most as large as the data's
// Benjamini-Hochberg over the vertices (home) and over the entries (size). A draw that makes no patch has no shares (NaN) and is not counted for them.
// STORED (statistics of the recording's Field):
//   spatial_events.null.<rec>.home_p       [vertices x (p, q, exceedances)] on the glued surface   measure null_p, method besag_clifford
//   spatial_events.null.<rec>.home_null    [vertices]                       the null's mean occupancy fraction (what a home map subtracts), measure null_fraction
//   spatial_events.null.<rec>.size_null    [entries x (data, null mean, p_upper, p_lower, q_upper, q_lower)]   measure null_size: the shares and the rate named by the summary's entries
// WHAT IT CANNOT SAY: the null has the data's zero-lag covariance, spectrum and envelope and nothing else: a vertex beyond it is occupied more than mixing and amplitude modulation under the same threshold make it, not that a travelling wave is there. The instrument's blur
// (the point sources) is a separate reference. A rerun is a hit; draws that exist are kept (each is its own cached node).
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct SpatialNullOptions {
  std::string recording;                     // REQUIRED: the recording whose stored `spatial_events.home` and `.summary` (`run spatial_events`) are tested
  std::string events = "spatial_events";     // the pipeline the data's statistics are named by (`<events>.home.<recording>`, `<events>.summary.<recording>`)
  nxr::io::Json set = nxr::io::Json::object();   // the `spatial_events_null` parameters of every draw (the data's: levels, reference_level, threshold, bursts, cross_spectrum, band, transform, ...)
  std::optional<std::string> chunks;         // what each draw reads: chunk "k", chunks "a-b" or "all" ...
  std::optional<double> start, stop;         // ... or [start, stop) seconds (both). REQUIRED: the caller states the span; none is an error
  int batch = 20, min_draws = 20, max_draws = 100;
  int h = 10;
  bool replace = false;                      // replace the stored results, and any draw stored with other parameters (moved aside, never deleted)
};
struct SpatialNullResult { nxr::io::Json info; bool hit = false; };
SpatialNullResult spatial_null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SpatialNullOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

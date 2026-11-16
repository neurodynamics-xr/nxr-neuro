// nxr-pipelines — SCALE SELECTION: each event's OWN size and rhythm across a declared set of (spatial level, time level) pairs (spec docs/specs/2026-11-13-cortical-events.md, "Phase 2e"). From STORED rows only, no
// recording pass: what `run cortical_event_pairs` stored for every pair of the product `space_levels` × `time_levels` (the cells' amplitude on the pair's joint Selection, the cells' per-tile baseline, the labelling of
// the pair's events on the cells).
//
// THE GRAPH (the paramount architecture: loading · preparation and type checks · a graph of pure nodes · readback):
//   loading      per pair: the joint Selection, the labelling [tiles × time tiles] int32, the cells' amplitude [tiles × time tiles] and its baseline (median, mad per tile)          (io, by name)
//   preparation  type checks (one recording and surface; every pair stored, naming EVERY missing one; each labelling, cell array and baseline of that pair's joint partition; the labellings made by `source`);
//                z = (amplitude − median) / SD per tile, the standardization the extents were found on (`nxr::extent::standardize`)
//   graph        event_facts → link_events → select_scale (`nxr::scale`): events of different pairs are linked by the ladders' exact nesting in space and overlap in time; the connected components
//                are the cross-pair events; each one's response at every pair (its footprint), best pair, the response there
//   readback     a `spans` Selection of the cross-pair events on the recording's time line (their span in time; one event per cross-pair event), measurements over it, and over each pair's labelling the
//                cross-pair event of each of its events
//
// WHAT IS STORED (registry 1030 … 1039; nothing rolls up):
//   over the cross-pair events [1 × events × 8] float64: scale_space_level, scale_time_level, scale_peak, scale_tile_area (m²), scale_band_lo, scale_band_hi (Hz), scale_pairs, scale_edge (a bit mask)
//   the footprint scale_response [pairs × events × 1] (rows = the pairs in the Selection's declared order; NaN where the event has no member at that pair)
//   over each pair's labelling scale_event [1 × events of the pair × 1]: the cross-pair event it belongs to
//   `computed_by` = the pipeline and a hash of its key; the recipe (every option, the pair table, the inputs) an entry of the dataset's `recipes` dictionary and the spans Selection's params_json.
// A rerun with the same inputs and options is a hit; other rows of the same Selection are refused unless `replace`.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct ScaleSelectionOptions {
  std::string recording;                     // REQUIRED: the recording whose pairs are selected over
  std::vector<int> space_levels;             // REQUIRED: the surface ladder's levels (the pair set is the PRODUCT space_levels × time_levels)
  std::vector<int> time_levels;              // REQUIRED: the recording's constant-Q levels (each one's band is the pair's rhythm)
  std::string source = "cortical_event_pairs";   // the pipeline whose stored labellings, cells and baselines are read (a labelling made by another is not used)
  std::string response = "peak";             // peak: the largest standardized cell amplitude of an event (blob-detection scale selection) | mean: their mean
  int min_link = 1;                          // events of two pairs are the same cross-pair event when at least this many pairs of their cells nest in space and overlap in time
  std::optional<std::string> name;           // the cross-pair events' Selection; default `<recording>_scale_selection`
  std::optional<std::string> surface;        // the subject's surface (default: the joint partitions')
  // MEMORY: every pair's labelling, cells and z: Σ over the pairs of tiles × time tiles × (4 + 8 + 8) bytes (levels 3 … 7, 5 bands, an hour: tens of MB); the links are events × events sample-bound tests.
  bool replace = false;                      // other rows of the same Selection and measures are moved aside (deleted rows; the arrays are rewritten)
};
struct ScaleSelectionResult {
  nxr::io::Json info;
  bool hit = false;
};
ScaleSelectionResult scaleselection_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const ScaleSelectionOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

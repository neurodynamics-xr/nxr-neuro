// nxr-pipelines — WHAT SCALE SELECTION READS (reading only): for every pair (spatial level, time level) of the product of two lists, what `run cortical_event_pairs` stored: the pair's joint Selection,
// the labelling of its events on the cells, the cells' amplitude and the cells' per-tile baseline (median, mad). Every pair is found and checked here, naming EVERY missing one. The runner hands it to
// the scale-selection step through the bound services (bound.h); the step computes from it and stores nothing.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct ScalePairInput {
  int space_level = 0, time_level = 0;
  std::string joint_id, joint_name, labelling_id, labelling_name, baseline_name;
  nxr::io::Json joint;                                     // the joint Selection's record (params_json.joint)
  std::int64_t tiles = 0, frames = 0, window = 0, events = 0;
  std::vector<std::int32_t> labels;                        // [tiles x frames]: the event of each cell (-1: none)
  std::vector<double> amplitude;                           // [tiles x frames]: the cells' amplitude
  std::vector<double> median, sd;                          // [tiles]: the baseline's median and mad
};
struct ScaleInputs {
  std::string surface, manifold_id;                        // the surface the joint Selections are on, and the product they lie on
  std::vector<ScalePairInput> pairs;                       // space level ascending in the order given, then time level
};

/** the pairs' stored rows, found and checked (invalid_argument naming every missing pair and what makes it); `surface` stated is checked against the joint Selections' */
ScaleInputs read_scale_inputs(nxr::io::SubjectWriter& sub, const std::string& recording_id, const std::string& recording_name, const std::vector<int>& space_levels,
                              const std::vector<int>& time_levels, const std::string& source, const std::optional<std::string>& surface);

}  // namespace nxr::pipelines

// nxr-pipelines — STATISTIC SUMMARIES (spec docs/specs/2026-11-13-statistic-summaries.md): a stored measurement summarised over its partition's members into a
// STATISTIC (field -> selection/measurement -> statistic, the maintainers' flow; across the members, never within one): per row (a channel, a vertex), pooled, or of the
// mean over rows; the summaries of nxr::stats. The measurement is read through io (either stored form), never the recording.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct StatisticComputeOptions {
  std::string over;                          // the partition, by name: a Selection whose members were measured (a parcellation, a level of a time ladder)
  std::string of;                            // what was measured: a Field by name, or a band of a map statistic (`<statistic>_<band>`)
  std::string measure;                       // the stored measure (amplitude, mean, mean_square, …); a derived one is refused
  std::vector<std::string> summaries;        // median · mad · percentile:p · mean · sd · peak · kurtosis · skewness · min · max (nxr::stats)
  bool own = false;                          // the measure is the members' OWN (area, n_elements, n_tiles: rows of no Field, e.g. the areas of the patches); `of` is then its parent
  std::string combine = "each";              // each (per row) · pool (every value of every row) · mean (of the mean over rows per member)
  int ddof = 1;                              // sd's divisor n - ddof (1: MATLAB's std)
  std::optional<std::string> name;           // default `<of>_<measure>_<over>`
  bool replace = false;
};
struct StatisticComputeResult { nxr::io::Json info; bool hit = false; };
StatisticComputeResult statistic_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const StatisticComputeOptions& opt,
                                         const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

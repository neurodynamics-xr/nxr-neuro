// nxr-pipelines — MEASUREMENTS of stored Fields over a PARTITION: each member of a labelling (a `set` Selection with one code per vertex and a
// dictionary, D48: Desikan-Killiany, a level of a tile ladder, D85) gets the Field's measures (nxr/partition.h, the Studio's measureOverPartition), written
// as the Studio writes them (`selection_measurement`, D68: selection = the labelling, code = the member, of_field_id = the Field, `weight` = the member's
// weight so a mean rolls up exactly). How they were made is a RECIPE (the Studio's recipes, docs/recipes.md): an entry of the dataset's `recipes`
// dictionary, kind `reduction`, carrying every choice; every row names it in `computed_by` (`recipe <name>@<version> · nxr-neuro <version>`).
//
//   per Field and member   mean · max · min · mean_square · abs_max (the reference's defaults; `measures` to choose)
//   per member, once       n_elements, and `area` (m²) when the weights are areas
//
// The WEIGHT is stated, never assumed: none (1 a vertex — Brainstorm's scout mean, which is the plain vertex mean), barycentric (a
// third of each incident triangle — nxr's atlas, Rheome, the Studio's tile ladder), voronoi (the mixed Voronoi area of the mollified input — the
// Studio's `lumpedMass`, nxr-neuro's operators' mass). On a real cortex they differ by tens of percent per region (2026-11-11 audit).
// One recipe's rows at a time per (labelling, Field, measure) (the table's key has no recipe): a re-run of the same recipe is a cache hit; another
// recipe's rows are refused unless `replace`.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct MeasureComputeOptions {
  std::string labelling;                     // the partition, by name: a `set` Selection with per-vertex codes and a dictionary (a ladder's level is one)
  std::string fields;                        // an ECMAScript regex matched against the whole name of each Field on the labelling's surface (a scalar per vertex)
  std::string weight;                        // REQUIRED: none · barycentric · voronoi (no default: it changes the numbers)
  std::vector<std::string> measures;         // default: mean, max, min, mean_square, abs_max (the Studio's); also sum
  std::optional<std::string> recipe;         // the recipe's name (default `partition-<weight>`); a new version when its choices change
  bool replace = false;                      // another recipe's rows for the same (labelling, Field, measure) are replaced
  bool gradient = false;                     // also the GRADIENT of each scalar Field (nxr measure.vectors.vertex_gradient: the area-weighted face gradients projected
                                             // on the vertex's tangent plane), measured as the registry's gradient_* sums in the canonical frames (needs `frames compute`)
};
struct MeasureComputeResult { nxr::io::Json info; };

MeasureComputeResult measure_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const MeasureComputeOptions& opt,
                                     const std::function<void(const nxr::io::Json&)>& progress = {});

// MEASUREMENT OVER TIME — a time-resolved Field (a recording, a source series: [rows x samples] on a product whose time factor is the tiling's line)
// measured over the windows of a TIME PARTITION (the cardinal rule: time is partitioned as space is): a level of a constant-Q time ladder
// (`constantq compute`; `over` = the ladder, `level` = its level) or a closed-form `spans` tiling such as the chunk grid (`over` = `<recording>_chunks`).
// The Studio's design (the envelope, D96): per Field ONE ARRAY beside it, [rows x windows x components], and one `selection_measurement` row a measure
// (selection = the tiling's member, of_field_id = the Field, array_path, component), `count` (the samples a value is over: the roll-up weight) always the
// last component. The arithmetic is the Studio's `reduceTiles` (nxr/partition.h), a row at a time, folded window by window across the chunks the chunk
// service hands over (the span: one chunk by default; windows outside it are NaN). Windows never straddle a stored chunk on a real recording (the ladder's
// windows are powers of two, chunks 2^15), so the fold is the whole-recording arithmetic; a window split by a chunk edge is folded in float (rounding).
// Bad segments are not left out.
struct MeasureOverTimeOptions {
  std::string over;                          // the time tiling by name: a constant-Q ladder (with `level`) or a closed-form `spans` Selection
  std::optional<int> level;                  // the ladder's level (required for a ladder)
  std::string fields;                        // an ECMAScript regex matched against the whole name of each time-resolved Field on the tiling's time line
  std::vector<std::string> measures;         // default mean, mean_square, max, min (also abs_max); `count` is always stored
  std::optional<std::string> chunks;         // the span read: "k", "a-b", "all"; or start/stop seconds; neither: chunk 0
  std::optional<double> start, stop;
  bool replace = false;                      // rows of the same (window member, Field, measure) are replaced; else refused
};
MeasureComputeResult measure_over_time(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const MeasureOverTimeOptions& opt,
                                       const std::function<void(const nxr::io::Json&)>& progress = {});

// THE EXPORT — the stored measurements of a dataset as a MEASUREMENT TABLE for statistics, a study release's layout
// (nxr docs/specs/2026-11-09-group-analysis-design.md §8): READ ONLY on the dataset; every value read through the measurement lookup
// (query_measurements: a stored measure as stored, a derived one — sd, variance, rms — from its stored ingredients).
//
//   <out>/<table>_<condition>_<statistic>.csv   (split, the default) one row a subject, one column a member: e.g. meg_dk_alpha_mean.csv, column
//                           precuneus_L_mean; the members' own statistics (area, n_elements) in <table>_<statistic>.csv; empty where a subject lacks it
//   <out>/<table>.csv       (--layout wide) one table, one column a (condition × member × statistic): e.g. alpha_precuneus_L_mean
//   <out>/<table>.json      the provenance: nxr-neuro version, the request, the dataset, each recipe (its choices), per subject the labelling and the Fields read
//   <out>/dictionary.csv    one row a column of every table in the folder (this table's rows replaced, the others kept): the release's columns
//                           index · data_table · column_number · variable · description · units · possible_data_values · value_type · notes,
//                           then labelling · member · condition · statistic · field_kind · recipe
//
// The CONDITION of a Field is what makes it the same measurement in every subject: the first capture group of `fields` (e.g. '.*_psd_(.*)' → the band);
// with no group, the Field's kind. A subject with two Fields of one condition is refused (the request is ambiguous). MEMBERS are joined across subjects
// by their dictionary NAME (codes may differ between subjects' labellings). `area` and `n_elements` are the members' own (no condition).
struct MeasureExportOptions {
  std::string labelling;                     // the partition, by name, in every subject
  std::string fields;                        // an ECMAScript regex over Field names; its first capture group names the condition
  std::vector<std::string> statistics = {"mean"};   // stored or derived measures: mean · sd · variance · rms · mean_square · max · min · abs_max · sum · area · n_elements
  std::string table;                         // the table's name (data_table, and its files' names)
  std::vector<std::string> subjects;         // default: every subject of the dataset (kind `subject`), by name
  std::optional<std::string> recipe;         // only measurements made by this recipe (`name` or `name@version`); a subject whose rows another made is refused
  std::vector<std::string> exclude = {"unassigned"};   // members left out of the table, by dictionary name; empty keeps all
  std::string layout = "split";              // split: a table a condition × statistic, a column a member (the release's layout); wide: one table
  std::string out;                           // the folder written (made if absent)
  // THE SCALING chosen at read: stored (default) · density · power · amplitude · amplitude-density · db · db-density, applied
  // to each Field's values from its producer's `spectral_scaling` record (a Field without one is refused any but stored). Linear targets scale every
  // statistic of the Field (mean_square and variance by the factor squared; ratios such as normal_share unchanged); the others apply to mean, max, min
  // only (of the stored statistic: the amplitude of the member's mean power).
  std::string scaling = "stored";
};
nxr::io::Json measure_export(const std::string& dataset_folder, const MeasureExportOptions& opt);

}  // namespace nxr::pipelines

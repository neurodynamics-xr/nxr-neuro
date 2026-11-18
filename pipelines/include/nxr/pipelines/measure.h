// nxr-pipelines — MEASUREMENTS of stored Fields over a PARTITION: each member of a labelling (a `set` Selection with one code per vertex and a
// dictionary, D48: Desikan-Killiany, a level of a tile ladder, D85) gets the Field's measures (nxr/measurement/reduce.h, the Studio's measureOverPartition), written
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

// MEASURING a Field over a labelling is the definition `measure` (pipelines/definitions/measure.yaml; run_shipped.h): the registry measures reduced over the members, the weight stated.



// MEASUREMENT OVER TIME is the definition `measure_over_time` (pipelines/definitions/, the step `window_measures` of steps_measure.cpp): a time-resolved field measured over the windows of a time partition.

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

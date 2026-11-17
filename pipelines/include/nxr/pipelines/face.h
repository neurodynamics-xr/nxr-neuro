// nxr-pipelines — WHAT A FACE ASKS (the Python wheel and the MATLAB binding; docs/specs/2026-11-17-compute-and-run.md, section 7): the pipeline level in the calls a binding makes,
// as the command line makes the same calls. A face converts arguments and results and calls these; they hold no maths and no SQL, only the library's own calls.
//
//   run         a pipeline on a subject (`nxr-neuro run`): the same library calls, so a result made from any face is the same cached result
//   pipelines   what `run` takes: the built-in `psd` and every shipped definition, with each parameter's meaning, default and unit
//   datasets · subjects · results      `nxr-neuro list datasets | subjects | recordings | kernels | results`
//   read        a stored field, statistic, operator or selection (and a selection's measurements) as arrays plus metadata
//
// A dataset is a folder: a catalogue of subject packages, or a one-database dataset. The time range of a run is REQUIRED: from and to (seconds) or all.
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"

namespace nxr::pipelines::face {

using nxr::io::Json;
using Progress = std::function<void(const Json&)>;

/** the producer stamp of what a face makes ("nxr-neuro <version> / C++17", as the command line's) */
std::string stamp();

/** what `run` takes through a face: [{name, form ("built-in" | "definition"), summary, parameters: [{name, default, meaning, unit, choices}]}], by name */
Json pipelines();

struct RunRequest {
  std::string pipeline;                    // a built-in's name, a shipped definition's name, or a .yaml path
  Json parameters = Json::object();        // the pipeline's parameters by name (a built-in's options with the dashes as underscores: `segment_seconds`)
  std::optional<double> from, to;          // the time range in seconds on the recording's clock (both)
  bool all = false;                        // or the whole recording (still read one chunk at a time)
  bool replace = false;                    // another job's outputs of the same names are moved aside (never deleted) and replaced
  std::string preset;                      // a named parameter set of the definition to start from (`run measure --set cortical_band_pyramid`); provenance only, the cache key is the resolved parameters
  bool dry_run = false;                    // plan only: every run's plan as text, nothing read or stored
};
/** What a face asks of a GROUP run (a definition of scope group: `roll_up` across subjects): the member subjects are the caller's (no default), the result is stored in the group's default subject of the
 *  dataset (`into`, the group package named by the caller in a catalogue; the dataset's default subject otherwise). */
struct GroupRunRequest {
  std::string pipeline;
  Json parameters = Json::object();
  std::vector<std::string> subjects;       // the members, by name, in the order of the stack
  std::string into;                        // the group package (a catalogue dataset) or the default subject's name (a one-database dataset)
  bool replace = false;
  std::string preset;
  bool dry_run = false;
};
/** Runs the group definition over its members; the same report as `run` ({pipeline, cache, stored, runs, seconds}, `subjects` listing the members). */
Json run_group(const std::filesystem::path& dataset, const GroupRunRequest& request, const Progress& progress = {});
/** Runs the pipeline on the subject. The range is REQUIRED (`from` and `to`, or `all`; one of them: anything else is an Error invalid_argument naming it). Returns
 *  {pipeline, subject, cache ("hit" when every run was found stored, else "computed"), stored: [{output, name, id, path}], runs: [...each run as the library reports it], seconds}.
 *  Progress lines are the command line's JSON lines. */
Json run(const std::filesystem::path& dataset, const std::string& subject, const RunRequest& request, const Progress& progress = {});

/** the datasets of a datastore: {id, datasets: [...]} */
Json datasets(const std::filesystem::path& datastore);
/** the subjects of a dataset: {dataset, subjects: [...]}; in a catalogue, `names` (by name or id) or `where` (the participants filter language, version 1) select, not both */
Json subjects(const std::filesystem::path& dataset, const std::vector<std::string>& names = {}, const std::optional<std::string>& where = std::nullopt);
/** what a subject holds, by kind: "results" (default: every other complete field and statistic), "recordings" or "kernels": [{name, kind, table}] */
Json results(const std::filesystem::path& dataset, const std::string& subject, const std::string& what = "results");

/** an array handed back: C order, `dtype` as the store names it (float32, float64, int32, ...), its bytes little-endian */
struct StoredArray {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};
struct Stored {
  Json meta;                               // class, name, id, and what the node records about itself (unit, kind, axes, options, producer, ...)
  std::vector<StoredArray> arrays;         // a field, operator or selection: its array (a sparse operator: data, indices, indptr); a statistic: values; a measurement table: code, value, weight, members
};
/** THE MEASUREMENT QUERY (consolidation spec 2.3; rule 35): a stored measurement over a selection, read at the level of a QUANTITY (a tile size, a duration, `whole`) and, for a group analysis, across subjects.
 *  Roll-ups live in two places by the PI's decision, with ONE definition: each measure's merge rule (sum, max, min, mean via sum and count, ratios derived at read) is DATA in the measures registry, applied by io in SQL
 *  (`where: "io"`, for filters and group queries over many subjects) and by compute in memory (`where: "compute"`, from the finest ingredients and the grid's closed form); a parity test holds the two equal. */
struct MeasurementQuery {
  std::string selection;                   // the selection (a grid, a product or an event set) the measurement is over, by name or id
  std::string measure;                     // a stored or derived measure
  std::optional<std::string> of_field;     // the field it is of, when several are measured over the selection
  std::optional<std::string> to;           // the level asked, a quantity ("tile 20 mm", "wavelength 40 mm", "2 s", "whole"); none: as stored (the finest)
  std::optional<std::string> factor;       // for a product: the factor the roll-up runs along (its name); none: the only grid factor
  std::vector<std::string> subjects;       // across subjects (the same members in each): ingredients summed code for code before the roll-up; empty: the one subject
  std::string where = "compute";           // "io" (the SQL query) or "compute" (in memory)
  // the FILTERS of io's query (rule 35: filters are SQL, in io's `query::members`): rows whose (rolled-up) value is in [min_value, max_value], the first `limit` of them, ordered by value
  std::optional<double> min_value, max_value;
  std::int64_t limit = 0;                  // 0: all
  std::string order;                       // "asc" | "desc" by value; empty: by member
};
/** {measure, selection, level, snapped: {the level's real size in its unit}, rows, members, values: float64 [rows x members] row-major as `StoredArray`-like {shape, data}, weight, unit, where}. Nothing is written. */
Json measurement_query(const std::filesystem::path& dataset, const std::string& subject, const MeasurementQuery& query);
/** The stored thing `name` of the subject (a statistic, field, operator or selection, by name or id), read whole. With `measure`, `name` is a selection and the answer is its measurement table
 *  (what the measure carries on each member, `of_field` naming the field when several are measured). Nothing is written (the dataset is opened for reading). */
Stored read(const std::filesystem::path& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure = std::nullopt,
            const std::optional<std::string>& of_field = std::nullopt);

/** A small one-database dataset for tests and demos, made in `datastore` (named `name`, subject `subject`): `channels` channels of known signals (sinusoid, noise; the last bad) stored as the
 *  recording `recording` in chunks of 700 samples at `sfreq`, `seconds` long, with one bad span. Returns the dataset's folder. */
std::filesystem::path synthetic_dataset(const std::filesystem::path& datastore, const std::string& name = "demo", const std::string& subject = "S1", const std::string& recording = "rest",
                                        int channels = 3, double sfreq = 100.0, double seconds = 40.0, unsigned seed = 21);

}  // namespace nxr::pipelines::face

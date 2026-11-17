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
};
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
/** The stored thing `name` of the subject (a statistic, field, operator or selection, by name or id), read whole. With `measure`, `name` is a selection and the answer is its measurement table
 *  (what the measure carries on each member, `of_field` naming the field when several are measured). Nothing is written (the dataset is opened for reading). */
Stored read(const std::filesystem::path& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure = std::nullopt,
            const std::optional<std::string>& of_field = std::nullopt);

/** A small one-database dataset for tests and demos, made in `datastore` (named `name`, subject `subject`): `channels` channels of known signals (sinusoid, noise; the last bad) stored as the
 *  recording `recording` in chunks of 700 samples at `sfreq`, `seconds` long, with one bad span. Returns the dataset's folder. */
std::filesystem::path synthetic_dataset(const std::filesystem::path& datastore, const std::string& name = "demo", const std::string& subject = "S1", const std::string& recording = "rest",
                                        int channels = 3, double sfreq = 100.0, double seconds = 40.0, unsigned seed = 21);

}  // namespace nxr::pipelines::face

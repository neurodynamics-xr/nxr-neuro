// nxr-pipelines — WHAT A FACE ASKS (the Python wheel and the MATLAB binding; docs/specs/2026-11-17-compute-and-run.md, section 7): the pipeline level in the calls a binding makes,
// as the command line makes the same calls. A face converts arguments and results and calls these; they hold no maths and no SQL, only the library's own calls.
//
//   run         a pipeline on a subject (`nxr-neuro run`): the same library calls, so a result made from any face is the same cached result
//   pipelines   what `run` takes: the built-in `psd` and every shipped definition, with each parameter's meaning, default and unit
//   datasets · subjects · results      `nxr-neuro list datasets | subjects | recordings | kernels | results`
//   read        a stored field, statistic, operator, selection or manifold (a selection's measurements, a recording's span and rows, a manifold's labelling) as arrays plus metadata
//
// A dataset is a folder: a catalogue of subject packages, or a one-database dataset. The time range of a run is REQUIRED: from and to (seconds) or all.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/task/commands.h"

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
/** What a face asks of a GROUP run (a definition of scope group: `roll_up` across subjects): the member subjects are the caller's (no default), the result is stored in the dataset's
 *  template (format 56: the template holds the cohort; a catalogue's one member of kind template, a one-database dataset's default subject). */
struct GroupRunRequest {
  std::string pipeline;
  Json parameters = Json::object();
  std::vector<std::string> subjects;       // the members, by name, in the order of the stack
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

/** What `import subject` takes: a Brainstorm protocol's subject into a dataset, then its sensor meshes (`sensor_mesh`) unless the caller says not to. */
struct ImportSubjectRequest {
  std::string protocol, subject, dataset;
  std::optional<double> resample;          // resample every recording to this rate (Hz) when imported; none: the original rate
  bool sensor_mesh = true;                 // false (--no-sensor-mesh): no sensor mesh is made, and the result says so
};
/** Imports the subject (io's import, unchanged), then runs `sensor_mesh` once for each channels manifold it holds (a points manifold with a `type` labelling) and each
 *  sensor type among its channels (MEG: MEG, MEG MAG, MEG GRAD; EEG). Returns io's result with `sensor_meshes` [{channels, sensor_type, surface}]; a mesh that is refused
 *  (fewer than four good channels, no quality labelling, a mesher refusal) is added to `skipped` as {what: "sensor mesh <channels> <type>", reason} and emitted as a
 *  `skipped` progress line, never silent; the import itself stands. */
Json import_subject(const ImportSubjectRequest& request, const Progress& progress = {});

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
/** THE MEASUREMENT QUERY (consolidation spec 2.3): a stored measurement over a selection, read at the level of a QUANTITY (a tile size, a duration, `whole`) and, for a group analysis, across subjects.
 *  Roll-ups live in two places, with ONE definition: each measure's merge rule (sum, max, min, mean via sum and count, ratios derived at read) is DATA in the measures registry, applied by io in SQL
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
/** What `read` is asked beyond the name. Every member is optional; an option that does not apply to what `name` is (a span of a manifold, a family of a field) is refused. */
struct ReadRequest {
  std::optional<std::string> measure;      // `name` is a selection: its measurement table of this measure
  std::optional<std::string> of_field;     // ... of this field, when several are measured over the selection
  // A TIME-RESOLVED FIELD (a recording, a source series: [rows x samples] on its time line) is read through io's chunk service, one chunk at a time:
  std::optional<std::int64_t> start, stop; // the samples [start, stop) of its time axis (both, or neither: the whole recording, still read chunk by chunk)
  std::vector<std::int32_t> rows;          // its rows (channels), in the order given; empty: every row
  std::string projectors = "active";       // the chunk service's option: "active" (the recording's active projectors applied, as every run reads it), "none" (the samples as stored), or
                                           // "NAME[,NAME...]" (exactly those projectors, whatever their stored status; an unknown name is refused)
  std::string reading = "samples";         // "samples", or "shape": the description only (rate, origin, samples, channels, unit, projection, provenance), no sample read
  std::optional<double> from, to;          // or the time range in seconds on the field's clock (both; not with start and stop), read chunk by chunk
  bool all = false;                        // or the whole field, read chunk by chunk
  bool range_required = false;             // a time-resolved field needs its range stated (from and to, start and stop, or all): the command line's rule (`read field`); a Refused otherwise
  // A MANIFOLD (by name or id: its positions, its cells, its axes, its element names) or, with `family`, one LABELLING of it:
  std::optional<std::string> family;       // the labelling of the manifold of that name (type, unit, quality, hemisphere, lobe, ...): its code per element and its dictionary
  std::string cls;                         // only a node of this class: field | statistic | operator | selection | manifold (empty: the first of statistic, field, operator, selection, manifold)
  // A PART of a node with no time line: [start, stop) per leading axis of EACH array the read gives (empty: whole); a stop past an array's extent is cut to it, and the result's `regions`
  // say what each array read. A time-resolved field is read by rows and a time range instead, a sparse operator or field whole.
  std::vector<std::array<std::int64_t, 2>> region;
  double budget = 1e9;                     // bytes: a read whose arrays would hold more is refused before any is read, naming the size (default 1 GB)
  // CARRIED ONTO A GLUED CORTEX (an operator, or a field, whose rows are the vertices of a two-hemisphere surface): its rows SELECTED onto the glued cortex `onto` (by name or id) by the glue's rule
  // (input.h glue_relation): each glued vertex the rows of its original vertex, a bridge vertex's rows zero. Vertex-major rows (k consecutive rows a vertex: a free orientation's 3). Not with region or rows.
  std::optional<std::string> onto;
};
/** The stored thing `name` of the subject (a statistic, field, operator, selection or manifold, by name or id). Nothing is written (the dataset is opened for reading).
 *  - a statistic, an operator, a field with no time line: read whole (`values`, in the stored precision; a sparse operator `data`, `indices`, `indptr`); a field's meta names its
 *    manifold (`manifold`, `manifold_id`) and the cells its values sit on (`cell`: "0" vertices, "1" edges, "2" faces);
 *  - a time-resolved field: `values` [rows x samples] in its stored precision, the span and rows of `request`, through the chunk service (the active projectors applied unless
 *    `projectors` is none); meta: rate (Hz), origin (s), n_samples, n_channels, rows, the channels' manifold, the span read, the projection applied, unit, producer;
 *  - a selection with an array: `values`, its per-element array; a selection of SPANS (events, bad segments): `start`, `stop` (s, on its line's clock), `code` (-1: none) and meta
 *    `dictionary` [{code, name}]; with `measure`, the measurement table (`code`, `value`, `weight`, `members`);
 *  - a manifold: `positions` [n x 3] as stored (meta `coordinate_unit` says their unit), `cells_<rank>` when it has cells; meta: type, n, axes, `names` (its `<name>_names` labelling, one
 *    name per element, when stored), its coordinate system (`coordinate_system` {name, unit, space}) and the selections that lie on it; with `family`: `codes` (int64, one per element,
 *    -1: none) and meta `dictionary` [{code, name}]. */
Stored read(const std::filesystem::path& dataset, const std::string& subject, const std::string& name, const ReadRequest& request);
/** the same with only a measure (the form the MATLAB binding calls) */
Stored read(const std::filesystem::path& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure = std::nullopt,
            const std::optional<std::string>& of_field = std::nullopt);

/** A small one-database dataset for tests and demos, made in `datastore` (named `name`, subject `subject`): `channels` channels of known signals (sinusoid, noise; the last bad) stored as the
 *  recording `recording` in chunks of 700 samples at `sfreq`, `seconds` long, on the channel manifold `sensors` (positions, the names C0, C1, ... and the labellings type, unit and
 *  quality), with one bad span (`<recording>_bad_segments`, type BAD). Returns the dataset's folder. */
std::filesystem::path synthetic_dataset(const std::filesystem::path& datastore, const std::string& name = "demo", const std::string& subject = "S1", const std::string& recording = "rest",
                                        int channels = 3, double sfreq = 100.0, double seconds = 40.0, unsigned seed = 21);

// ---- `read <class>` (docs/specs/2026-11-19-io-service.md, 4.1): one stored node by name, as `read` above, the class named by the command. The result is the node's meta with
// `arrays` [{name, dtype, shape}] (the arrays themselves in CommandResult::arrays); with `out`, the arrays are written as .npy files and `files` {name: path} says where:
// one array to `out` itself, several (a sparse operator's data, indices and indptr; a manifold's positions and cells; spans' start, stop and code) to `<out stem>.<name>.npy`.
using nxr::io::task::CommandResult;
struct ReadNodeRequest {
  std::string dataset, subject, name;
  std::optional<double> from, to;          // field: the time range in seconds (both), or
  bool all = false;                        // field: the whole field (a time-resolved field needs one of the two; read one chunk at a time)
  std::vector<std::int32_t> rows;          // field: its rows, in order (empty: all)
  std::string projectors = "active";       // field: active | none | NAME[,NAME...]
  bool shape_only = false;                 // field: its description only, no sample read
  std::optional<std::string> measure, field;   // selection: its measurement table of this measure (of this field)
  std::optional<std::string> family;       // manifold: one labelling of it
  std::optional<std::string> out;          // the .npy file (or stem) the arrays are written to
 
  std::vector<std::array<std::int64_t, 2>> region;   // [start, stop) per leading axis of each array (`--region`; not with a time-resolved field: its rows and time range)
  double budget = 1e9;                     // bytes (`--budget`): a larger read is refused before anything is read, naming its size and suggesting `--region`
  std::optional<std::string> onto;         // operator, field: its rows carried onto this glued cortex (ReadRequest::onto)
};
CommandResult read_field(const ReadNodeRequest& request);
CommandResult read_statistic(const ReadNodeRequest& request);
CommandResult read_operator(const ReadNodeRequest& request);
CommandResult read_selection(const ReadNodeRequest& request);
CommandResult read_manifold(const ReadNodeRequest& request);

/** `read subject` and `list <class>` (nxr/io/task/commands.h) with each node's TYPE (`type`: the JSON `compile` gives a value, a manifold's its term): io reads and pages the rows, the type
 *  system types them. The command line, Python and JavaScript call these. */
CommandResult read_subject(const nxr::io::task::ReadSubjectRequest& request, const Progress& progress = {});
CommandResult list_class(const nxr::io::task::ListClassRequest& request, const Progress& progress = {});

// ---- THE COMMANDS THAT NEED COMPUTE (docs/specs/2026-11-19-io-service.md, section 5): one function a command, the command line calling it as the faces will; the io-only
// commands are nxr/io/task/commands.h. Each takes the command's options as a request and returns its result object (without "stage") and its arrays.
using nxr::io::task::CommandResult;

/** `create dataset`: io's (task::create_dataset), and with `templates` the one-database dataset with its default subject (its frames, group tree and Levi-Civita connection):
 *  both or neither. Its result is the library's report ({dataset, ...}: the command line prints it with "stage" last). */
CommandResult create_dataset(const nxr::io::task::CreateDatasetRequest& request, const Progress& progress = {});

/** `create registration`: a registration package between two templates. Result: the registration's info, then `package` and `content_hash`. */
struct CreateRegistrationRequest {
  std::string package, name, from, to, from_sphere, to_sphere;
  std::optional<std::string> method;
  std::optional<int> neighbours;
  bool one_direction = false;
  std::vector<std::string> templates;
  std::optional<std::string> home_dataset, source, licence, attribution;
  std::optional<std::string> redistributable;   // "yes" | "no"
  std::string created_by;
};
CommandResult create_registration(const CreateRegistrationRequest& request, const Progress& progress = {});

/** `list recordings` · `list kernels` · `list results`: {subject, <what>: [{name, kind, table}]} */
struct ListNodesRequest { std::string dataset, subject; };
CommandResult list_recordings(const ListNodesRequest& request, const Progress& progress = {});
CommandResult list_kernels(const ListNodesRequest& request, const Progress& progress = {});
CommandResult list_results(const ListNodesRequest& request, const Progress& progress = {});

/** `read crossspectrum`: readings derived from a stored `<stem>_<band>_bins` statistic, for one pair of recording channels (row A, column B) or a seed against every channel.
 *  `reading`: coherency (re, im, abs per bin) | imag | phase (wrapped, unwrapped) | delay (phase slope, ms, with its residual; `weights` coherence | none) | psi. */
struct ReadCrossspectrumRequest {
  std::string dataset, subject, name;
  std::optional<std::array<std::int32_t, 2>> pair;
  std::optional<std::int32_t> seed;
  std::string reading = "coherency";
  std::string weights = "coherence";
 
  std::string created_by;
};
CommandResult read_crossspectrum(const ReadCrossspectrumRequest& request, const Progress& progress = {});

/** `query measurements`: `measurement_query` with the command's options; one of band, spatial_scale, tile is the level asked. Result {selection, measure, result}. */
struct QueryMeasurementsRequest {
  std::string dataset, subject, selection, measure;
  std::optional<std::string> field, band, spatial_scale, tile, factor, where;
  std::optional<std::vector<std::string>> subjects;
  std::optional<double> min_value, max_value;
  std::optional<std::int64_t> limit;
  std::optional<std::string> order;
};
CommandResult query_measurements(const QueryMeasurementsRequest& request);

/** `export measurements`: the stored measurements as a measurement table (one row a subject) with its data dictionary and provenance, a study release's layout.
 *  Result: the folder, the tables, the dictionary and the provenance written. */
struct ExportMeasurementsRequest {
  std::string dataset, labelling, fields, table, out;
  std::optional<std::vector<std::string>> statistics, subjects, exclude;
  std::optional<std::string> recipe, layout, scaling;
  bool keep_all = false;                   // no member excluded (`exclude` cleared)
};
CommandResult export_measurements(const ExportMeasurementsRequest& request);

/** `compile`: a pipeline definition expanded and type-checked WITHOUT running it; the COMPILED GRAPH handed back, so another program (a renderer) builds on nxr-neuro's types.
 *  Nothing is read beyond metadata and nothing is written. Each plan of the definition (one per match of its `each:` input and per combination of its `plans:` lists) is compiled:
 *  - with `dataset` and `subject` (both, read-only): the inputs are found as `run` finds them and typed from their stored records, and the graph is checked as `run` checks it
 *    (every node's plan and signature: the halo and the span grid are known); `bind` names the stored node each input that is not streamed is bound to, `streams` the recording read;
 *  - without them: each input's type is the caller's (`inputs`: name -> a type in the structured JSON form or the printed form; an optional input left out is absent) and the
 *    graph is checked by the node signatures alone (no node's plan runs: `halo` and `align` are null).
 *  `signatures` declares nodes nxr-neuro does not implement (nxr/graph/type_json.h `external_nodes`), checked with the rest; a node neither registered nor declared is refused.
 *  Result: {definition, subject (or null), types: "stored" | "given", plans: [{parameters, anchor, graph (nxr::graph::compiled_json), bind, streams}]}. A refusal is an Error
 *  invalid_argument carrying the checker's message. */
struct CompileRequest {
  std::string definition;                  // a shipped definition's name or a .yaml path
  std::optional<std::string> dataset, subject;   // both or neither
  Json parameters = Json::object();        // the definition's parameters by name (the command line's --set NAME=VALUE, the value read as YAML)
  std::string preset;                      // a named parameter set of the definition (`--set NAME`)
  Json signatures;                         // the external nodes' signatures ({"nodes": [...]} or a list); null: none
  Json inputs = Json::object();            // without a dataset: each input's type by name
};
CommandResult compile(const CompileRequest& request);

}  // namespace nxr::pipelines::face

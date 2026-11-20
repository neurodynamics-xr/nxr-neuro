// nxr-io — THE COMMANDS (docs/specs/2026-11-19-io-service.md, section 5): one library function a command of the command line's table (cli/command_table.cpp), the same
// function the command line, the Python binding and the WebAssembly binding call. Each takes a request whose members are the command's options (the long option with its
// dashes as underscores: `--registered-by` is `registered_by`; a C++ keyword takes a trailing underscore: `--default` is `default_`) and returns the command's result.
//
//   CommandResult r = task::show_dataset({"/data/cohort"});       // r.result: what `nxr-neuro show dataset --dataset /data/cohort` prints as its result line (without "stage")
//
// The io-only commands are here (datasets, subjects, packages, catalogues, `read dataset`); those that need compute (`create dataset --templates`, `create registration`,
// `read crossspectrum`, `query measurements`, `list recordings | kernels | results`) are in nxr/pipelines/face.h. A command is one unit: everything it stores, or nothing.
//
// Errors: a request the command refuses as given (an option's value, two options that exclude each other) is a `Refused` (the command line's usage error, exit 2);
// a data problem is an `nxr::io::Error` with its code. Progress goes to the optional `Progress` callback: the command line's JSON lines, and `{"stage": "note", "note": TEXT}`
// for a note the command line prints on stderr as `note: TEXT`.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

using Progress = std::function<void(const Json&)>;

/** a request refused as given; `bare`: the message stands alone (the command line prints `error: ...` without the command's name) */
struct Refused : std::invalid_argument {
  explicit Refused(const std::string& message, bool bare = false) : std::invalid_argument(message), bare(bare) {}
  bool bare = false;
};

/** an array handed back (where the command line writes an array to a file or prints its checksum): C order, `dtype` as the store names it, its bytes little-endian */
struct CommandArray {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};
/** what a command gives back: its result object (the command line's `"stage": "result"` line without `stage`) and its arrays */
struct CommandResult {
  Json result = Json::object();
  std::vector<CommandArray> arrays;
};

/** An array a command takes: a .npy file by path (the command line's form), or an array in memory (the faces': Python and JavaScript hand their arrays over directly, with no
 *  temporary file). The name the producer records for it is the file's name, or the in-memory array's `name` (e.g. "values.npy"), never a path. A string converts to the file form. */
struct ArrayInput {
  std::string path;                      // the .npy file (when not in memory)
  bool in_memory = false;
  std::string name;                      // in memory: the name recorded as the input
  std::string dtype;                     // in memory: as the store names it (float32, float64, int8 to int64, uint8 to uint64, bool)
  std::vector<std::int64_t> shape;       // in memory: C order
  std::vector<std::uint8_t> bytes;       // in memory: the values of `dtype`, little-endian

  ArrayInput() = default;
  ArrayInput(std::string file) : path(std::move(file)) {}
  ArrayInput(const char* file) : path(file) {}
  static ArrayInput memory(std::string name, std::string dtype, std::vector<std::int64_t> shape, std::vector<std::uint8_t> bytes) {
    ArrayInput a;
    a.in_memory = true;
    a.name = std::move(name);
    a.dtype = std::move(dtype);
    a.shape = std::move(shape);
    a.bytes = std::move(bytes);
    return a;
  }
  /** the name a message and the producer record give it: the file's name (never its path), or the in-memory array's name */
  std::string label() const { return in_memory ? name : std::filesystem::path(path).filename().string(); }
};

// ---- datasets ----

/** `create dataset`: a catalogue of subject packages at `dataset` (the default form), or (`datastore` and `name`) the one-database dataset DATASTORE/NAME.
 *  The default subject of `templates` needs compute: nxr::pipelines::face::create_dataset. */
struct CreateDatasetRequest {
  std::optional<std::string> dataset, datastore, name, source_tool, source_protocol, source_path;
  std::optional<std::string> templates, default_, atlas_finest;   // the default subject (one-database form only)
  std::optional<std::vector<std::string>> lc_scales;               // quantities ("parts 16", "20 mm")
  std::optional<int> modes;
  std::optional<std::string> store_format;                         // "51" | "52"
  std::string created_by;                                           // the producer stamp
};
CommandResult create_dataset(const CreateDatasetRequest& request, const Progress& progress = {});
/** the one-database form's checks (a name; the default subject's options need `templates`; the store format) and its options: shared by create_dataset and the face's
 *  templated form (`datastore` given, `dataset` not) */
CreateOptions one_database_options(const CreateDatasetRequest& request, const Progress& progress = {});

struct ListDatasetsRequest { std::string datastore; };
CommandResult list_datasets(const ListDatasetsRequest& request);

struct ShowDatasetRequest { std::string dataset; };
CommandResult show_dataset(const ShowDatasetRequest& request, const Progress& progress = {});

struct DeleteDatasetRequest { std::string datastore, name; };
CommandResult delete_dataset(const DeleteDatasetRequest& request);

struct RecoverDatasetRequest { std::string dataset; };
CommandResult recover_dataset(const RecoverDatasetRequest& request);

struct CompactDatasetRequest {
  std::string dataset;
  std::optional<std::string> subject;
  bool dry_run = false, no_shard = false, no_small_files = false;
};
CommandResult compact_dataset(const CompactDatasetRequest& request);

struct SplitDatasetRequest { std::string dataset, registered_by; bool dry_run = false; };
CommandResult split_dataset(const SplitDatasetRequest& request, const Progress& progress = {});

struct JoinDatasetRequest { std::string dataset; };
CommandResult join_dataset(const JoinDatasetRequest& request);

struct MigrateMeasurementsRequest { std::string dataset; bool dry_run = false, remove_statistics = false; };
CommandResult migrate_measurements(const MigrateMeasurementsRequest& request, const Progress& progress = {});

// ---- subjects ----

struct CreateSubjectRequest {
  std::string dataset, name;
  std::string kind = "subject";            // subject | template | group
  std::string created_by;
};
CommandResult create_subject(const CreateSubjectRequest& request, const Progress& progress = {});

struct ListSubjectsRequest {
  std::string dataset;
  std::optional<std::vector<std::string>> subjects;   // by name or id (a catalogue)
  std::optional<std::string> where;                   // the participants filter (a catalogue)
};
CommandResult list_subjects(const ListSubjectsRequest& request, const Progress& progress = {});

struct ShowSubjectRequest { std::string dataset, subject; std::vector<std::string> templates; };
CommandResult show_subject(const ShowSubjectRequest& request, const Progress& progress = {});

struct DeleteSubjectRequest {
  std::string dataset, subject;
  std::optional<std::string> node;         // one node (a store path, or a name unique in the subject) instead of the subject
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult delete_subject(const DeleteSubjectRequest& request);

/** `read subject` (docs/specs/2026-11-20-data-layer-queries.md, 3.1): every node's DESCRIPTION, no arrays, as nxr-neuro's typed model: {schema: "nxr-neuro-subject@1", subject: {id, name, kind, template},
 *  manifolds, fields, selections, operators, statistics, dictionaries, spaces, acquisitions: [description...]} (only the `classes` asked for), each node named and nested, never a table row (docs/reference.md,
 *  "Queries over a dataset"); with `related`, `related`: [{subject, kind, relation, description}] for the subjects this one refers to (its template, the owners of the manifolds its operators map between
 *  and its fields lie on: a registration's source and target) */
struct ReadSubjectRequest {
  std::string dataset, subject;
  std::optional<std::vector<std::string>> classes;   // manifolds | fields | selections | operators | statistics | dictionaries | spaces | acquisitions; default: all
  bool related = false;
  std::vector<std::string> templates;
};
CommandResult read_subject(const ReadSubjectRequest& request, const Progress& progress = {});

/** `list <class>` (3.2): the nodes of one class of a subject (`subject` is required for a catalogue; a one-database dataset lists every subject's, each row naming its own), `cls` one of
 *  manifolds | fields | selections | operators | statistics | dictionaries | spaces | acquisitions | jobs. Each row is the class's description (as in `read subject`) with its `subject` first, or only the
 *  `columns` asked for. `name`: the nodes whose name contains TEXT (case-insensitive); `kind`: a field's kind, a manifold's, selection's or operator's type or kind, a statistic's measure, a dictionary's scope, a space's
 *  geometry, a job's kind; `on`: the nodes that lie on the manifold of that name (a field on it or on a product of it, a selection on it, an operator from or to it, a statistic with an axis along it, the manifolds built on it).
 *  Ordered by name; `limit` rows a page, the `next` cursor (opaque, null at the end) given back as `after`. Result: {schema: "nxr-neuro-list@1", class, subject, rows, next}. */
struct ListClassRequest {
  std::string cls, dataset;
  std::optional<std::string> subject, name, kind, on, after;
  std::optional<std::int64_t> limit;
  std::optional<std::vector<std::string>> columns;
  std::vector<std::string> templates;
};
CommandResult list_class(const ListClassRequest& request, const Progress& progress = {});

struct AddSurfaceRequest {
  std::string dataset, subject;
  std::string file;                        // the mesh file (the positional FILE); or empty and
  std::optional<ArrayInput> vertices, faces;   // the surface from arrays: vertices [n x 3] and faces [m x 3] (0-based), .npy files or in memory (both; then `name` is required)
  std::optional<std::string> name;
  double scale = 1.0;
  bool primary = false, no_primary = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_surface(const AddSurfaceRequest& request, const Progress& progress = {});

// ---- `add <class>`: what a user computed, stored as ONE unit (docs/specs/2026-11-19-io-service.md, 4.2) ----
// Each reads its arrays from .npy files or from memory (ArrayInput; C order, little-endian), keeps their precision (float32 stays float32), checks them against the manifold or selection they lie on
// (a refusal says what was expected), and stores one node through io's typed writes (task::Unit) with a producer record naming the command, the program, the input FILES'
// NAMES (never their paths) and the user's `description`. A name already used in the subject is refused unless `replace`: the old node is then moved aside, never deleted.
// Result: {class, name, id, path, (replaced: where the old one went)} and what the command checked (n, shape, dtype).

/** `add field`: values [n] or [n x components...] on a manifold (n its vertices, or its edges or faces by `cell`), or with `time` [n x samples x components...] on the
 *  manifold and a time line (a time-resolved field, read back with `read field --from/--to | --all`) */
struct AddFieldRequest {
  std::string dataset, subject, name, manifold;
  ArrayInput values;                       // a .npy file, or in memory
  std::optional<std::string> time;         // a time line (by name): the field is time-resolved
  std::string value_type = "scalar";       // scalar · integer · complex · vector2 · vector3 · matrix3 · quaternion · hermitian3 (its components trail the element axes)
  std::string kind = "added";              // what its values are
  std::optional<std::string> unit, cell, derived_from, description;   // cell: "0" vertices (default), "1" edges, "2" faces
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_field(const AddFieldRequest& request, const Progress& progress = {});

/** `add selection`: a partition or set (a label per element, -1: none; `members`), a pick of elements in order (`indices`), or spans on a time line (`spans`: [k x 2] start,
 *  stop or [k x 3] start, stop, code; in the line's unit); `entries` name the labels (codes) 0, 1, ... */
struct AddSelectionRequest {
  std::string dataset, subject, name, manifold, type;   // type: partition | set | indices | spans
  std::optional<ArrayInput> members, indices, spans;   // the array of its type: a .npy file, or in memory
  std::vector<std::string> entries;
  std::optional<std::string> description;
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_selection(const AddSelectionRequest& request, const Progress& progress = {});

/** `add statistic`: values on named axes, each running along a manifold (`axes`: NAME:MANIFOLD, one an axis, in order), of a field (`of_field`) or computed elsewhere (detached) */
struct AddStatisticRequest {
  std::string dataset, subject, name, measure, method;
  ArrayInput values;
  std::vector<std::string> axes;           // NAME:MANIFOLD
  std::string value_type = "scalar";
  std::optional<std::string> of_field, partition, unit, description;
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_statistic(const AddStatisticRequest& request, const Progress& progress = {});

/** `add measurements`: one value a member of a selection for each measure ([members], in member order; stored as measurement rows, float64), of a field (or the members' own) */
struct AddMeasurementsRequest {
  std::string dataset, subject, selection;
  std::optional<std::string> field, unit;
  std::vector<std::string> measure;        // MEASURE=FILE.npy
  std::vector<std::pair<std::string, ArrayInput>> measure_arrays;   // MEASURE and its values given as arrays (in memory, or a file): after `measure`
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_measurements(const AddMeasurementsRequest& request, const Progress& progress = {});

/** `add operator`: a linear map from one manifold to another, dense (`values` [to x from]) or sparse CSR (`data`, `indices`, `indptr`: scipy.sparse's arrays, rows the
 *  `to` manifold's elements, columns the `from` manifold's) */
struct AddOperatorRequest {
  std::string dataset, subject, name, from_manifold, to_manifold;
  std::string kind = "added";              // what the operator is
  std::optional<ArrayInput> values, data, indices, indptr;
  std::optional<std::string> description;
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_operator(const AddOperatorRequest& request, const Progress& progress = {});

/** `add points`: a manifold of points, positions [n x 3] (in `coordinate_system`, default scs) or one coordinate each [n] (default: a line of its own name) */
struct AddPointsRequest {
  std::string dataset, subject, name;
  ArrayInput positions;
  std::optional<std::string> coordinate_system, description;
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_points(const AddPointsRequest& request, const Progress& progress = {});

/** `add line`: a closed-form line, `n` points `spacing` apart from `origin`, in `unit`; `dimension` time (a clock in s: a time line fields can be time-resolved on), frequency, ... */
struct AddLineRequest {
  std::string dataset, subject, name;
  std::int64_t n = 0;
  double spacing = 1, origin = 0;
  std::string unit, dimension = "time";
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_line(const AddLineRequest& request, const Progress& progress = {});

// ---- subject packages and the catalogue ----

struct CreatePackageRequest { std::string package, name, home_dataset; std::string kind = "subject"; std::string created_by; };
/** {package, subject_id, content_hash} (the command line prints the first two, then the content hash, as two result lines) */
CommandResult create_package(const CreatePackageRequest& request);

struct VerifyPackageRequest { std::string package; };
/** {ok, content_hash, problems}; the command line exits 1 when not ok */
CommandResult verify_package(const VerifyPackageRequest& request);

struct ShowPackageRequest { std::string package; };
CommandResult show_package(const ShowPackageRequest& request);

struct RegisterPackageRequest {
  std::string dataset, registered_by;
  std::vector<std::string> package;        // the packages, or `scan`
  bool scan = false, replace = false;
  std::optional<std::vector<std::string>> templates;
};
/** {results: [{package, name, outcome, reason}]}; the command line exits 1 when one is refused */
CommandResult register_package(const RegisterPackageRequest& request);

struct ListMembersRequest { std::string dataset; };
CommandResult list_members(const ListMembersRequest& request);

struct CreateIndexRequest { std::string dataset; bool force = false; };
CommandResult create_index(const CreateIndexRequest& request);

struct ReleaseTemplateRequest { std::string package, name, template_version, released_by, into; };
CommandResult release_template(const ReleaseTemplateRequest& request);

struct ReleaseRedistributableRequest { std::string package, into, released_by; };
CommandResult release_redistributable(const ReleaseRedistributableRequest& request);

struct ListTemplatesRequest { std::vector<std::string> templates; };
CommandResult list_templates(const ListTemplatesRequest& request);

struct CreateTemplateRequest {
  std::string recipe, sources, package;
  std::optional<std::string> name, home_dataset, hcp_trk, hcp_end_surface;
  std::optional<double> max_end_distance;
  std::optional<int> hcp_points;
  bool cortex_mask = false, scalp = false, accept_unlisted_sources = false;
  std::optional<std::vector<std::string>> surface, parcellation;
  std::string created_by;
};
CommandResult create_template(const CreateTemplateRequest& request);

// ---- reading ----

/** `read dataset`: without `array`, the subjects and every array the rows name (shape, dtype, CRC-32 of the decoded little-endian bytes); with it, that array (or `region`
 *  of it, [lo, hi) per axis): its shape, dtype and checksum in the result and its bytes in `arrays`; `out` writes the bytes to that file as the command does. */
struct ReadDatasetRequest {
  std::string folder;                                        // the dataset folder (the positional FOLDER)
  std::optional<std::array<std::string, 2>> array;           // the subject's name and the array's path inside the subject
  std::vector<std::array<std::int64_t, 2>> region;           // per axis [lo, hi); empty: whole
  std::optional<std::string> out;
};
/** without `array`: {subjects: [{name, path, kind}], arrays: [{subject, record, path, shape, dtype, checksum}]}; with it: {shape, dtype, checksum} and the array */
CommandResult read_dataset(const ReadDatasetRequest& request);

// ---- import (Brainstorm protocols; files in the field's formats into a subject package; BIDS; participants) ----

/** `import list PROTOCOL`: {root, subjects} (the command line prints it as its `"stage": "protocol"` line) */
struct ImportListRequest { std::string protocol; };
CommandResult import_list(const ImportListRequest& request);

/** `import scan PROTOCOL`: {kind, root, [wrote], the counts, view}; `summary`: the view left out (the counts only); `out` writes the view as JSON (`prefer`: protocol.mat | walk) */
struct ImportScanRequest { std::string protocol; std::optional<std::string> prefer, out; bool summary = false; };
CommandResult import_scan(const ImportScanRequest& request);

/** `import subject` · `import template` · `import group` · `import templates`: a Brainstorm subject, template or group, or the templates folder, into a one-database dataset */
struct ImportSubjectRequest { std::string protocol, subject, dataset; std::optional<double> resample; std::string created_by; };
CommandResult import_subject(const ImportSubjectRequest& request, const Progress& progress = {});
struct ImportTemplateRequest { std::string protocol, dataset; std::optional<std::vector<std::string>> surface; std::string created_by; };
CommandResult import_template(const ImportTemplateRequest& request, const Progress& progress = {});
struct ImportGroupRequest { std::string protocol, dataset; std::string created_by; };
CommandResult import_group(const ImportGroupRequest& request, const Progress& progress = {});
struct ImportTemplatesRequest { std::string source, dataset; std::optional<std::string> subject, trk, surface; std::string created_by; };
CommandResult import_templates(const ImportTemplatesRequest& request);

/** `import condition PROTOCOL`: one Brainstorm condition as a session (the subject made if absent) */
struct ImportConditionRequest {
  std::string protocol, subject, condition, dataset;
  std::optional<std::string> surface, recording, channel_file;
  std::optional<double> resample;
  bool no_gain = false, all_surfaces = false, include_imported = false, no_raw = false;
  std::optional<std::vector<std::string>> extra_surface;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult import_condition(const ImportConditionRequest& request, const Progress& progress = {});

/** `import surface` (tess_*.mat files; progress: a `surface` line each) · `import mri` · `import inverse` · `import map` (a `map` line each) · `import resample` ·
 *  `import badsegments` · `import events`: into a subject of a dataset */
struct ImportSurfaceRequest { std::string dataset, subject; std::vector<std::string> file; std::optional<std::string> bst_root; std::vector<std::string> templates; std::string created_by; };
CommandResult import_surface(const ImportSurfaceRequest& request, const Progress& progress = {});
struct ImportMriRequest { std::string dataset, subject, anat; std::optional<std::vector<std::string>> only; std::vector<std::string> templates; std::string created_by; };
CommandResult import_mri(const ImportMriRequest& request, const Progress& progress = {});
struct ImportInverseRequest {
  std::string dataset, subject, file, acquisition, bst_root;
  std::optional<std::string> channels_id, surface_id, forward_id, name, channel_file;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult import_inverse(const ImportInverseRequest& request, const Progress& progress = {});
struct ImportMapRequest {
  std::string dataset, subject, bst_root;
  std::vector<std::string> file;
  std::optional<std::string> acquisition, provenance;   // provenance: a JSON file recorded with each map
  bool replace = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult import_map(const ImportMapRequest& request, const Progress& progress = {});
struct ImportResampleRequest { std::string dataset, subject, recording; double to = 0; std::optional<std::string> name; std::vector<std::string> templates; std::string created_by; };
CommandResult import_resample(const ImportResampleRequest& request, const Progress& progress = {});
struct ImportBadsegmentsRequest { std::string dataset, subject, from; bool replace = false; std::vector<std::string> templates; std::string created_by; };
CommandResult import_badsegments(const ImportBadsegmentsRequest& request, const Progress& progress = {});
using ImportEventsRequest = ImportBadsegmentsRequest;   // `from`: the protocol's data folder of the recording
CommandResult import_events(const ImportEventsRequest& request, const Progress& progress = {});

/** the origin of an imported file, as every file importer records it */
struct FileOrigin {
  std::optional<std::string> source_root;            // the recorded source is relative to it (default: the file's folder), never an absolute path
  std::optional<std::string> licence, attribution;   // default "unknown"
  bool anonymize = false;
};
/** `import recording` (FIF, EDF/BDF, BrainVision, EEGLAB, CTF, KIT; copied block by block, or linked `in_place`) into a subject package */
struct ImportRecordingRequest {
  std::string from, package;
  std::optional<std::string> name, acquisition, format, montage;
  FileOrigin origin;
  bool in_place = false;
};
CommandResult import_recording(const ImportRecordingRequest& request, const Progress& progress = {});
/** `import fif` (an MNE object file) · `import transform` (a head-to-MRI transform) */
struct ImportFifRequest {
  std::string from, package;
  std::optional<std::string> kind, name, acquisition, channels, recording, surface;
  FileOrigin origin;
};
CommandResult import_fif(const ImportFifRequest& request, const Progress& progress = {});
struct ImportTransformRequest { std::string from, package; std::optional<std::string> name, channels; };
CommandResult import_transform(const ImportTransformRequest& request, const Progress& progress = {});
/** `import nifti` · `import freesurfer` · `import gifti` · `import cifti` */
struct ImportNiftiRequest {
  std::string from, package;
  std::optional<std::string> name, kind, parcellation, label_names, field_kind, unit;
  FileOrigin origin;
};
CommandResult import_nifti(const ImportNiftiRequest& request, const Progress& progress = {});
struct ImportFreesurferRequest {
  std::string from, package;
  std::optional<std::string> name, orig, field_kind;
  std::optional<std::vector<std::string>> surfaces, annotations, overlays;
  FileOrigin origin;
};
CommandResult import_freesurfer(const ImportFreesurferRequest& request, const Progress& progress = {});
struct ImportGiftiRequest {
  std::string from, package;
  std::optional<std::string> name, kind, surface, parcellation, field_kind, unit;
  FileOrigin origin;
};
CommandResult import_gifti(const ImportGiftiRequest& request, const Progress& progress = {});
struct ImportCiftiRequest {
  std::string from, package;
  std::optional<std::string> name, left, right, parcellation, field_kind, unit;
  FileOrigin origin;
};
CommandResult import_cifti(const ImportCiftiRequest& request, const Progress& progress = {});

/** `create recording --synthetic`: sine waves plus noise from a seed, into a subject package through the recording importer */
struct CreateRecordingRequest {
  std::string package;
  bool synthetic = false;
  std::optional<std::string> name, acquisition;
  std::optional<std::int64_t> channels;
  std::optional<double> sfreq, duration, amplitude, noise;
  std::optional<std::uint64_t> seed;
  std::optional<std::vector<double>> frequency;
};
CommandResult create_recording(const CreateRecordingRequest& request, const Progress& progress = {});

/** `import bids`: a BIDS dataset (MEG, EEG, iEEG) into a new dataset of subject packages, with the participant variables */
struct ImportBidsRequest {
  std::string from, dataset;
  std::vector<std::string> subject;        // labels, with or without `sub-`; empty: every subject
  bool in_place = false, anonymize = false, no_participants = false;
  std::optional<std::string> registered_by;
  std::string created_by;                  // the default of registered_by
};
CommandResult import_bids(const ImportBidsRequest& request, const Progress& progress = {});

/** `import participants` · `export participants`: the participant variables of a catalogue dataset from or to a TSV (and its JSON sidecar) */
struct ParticipantsRequest { std::string dataset, tsv; std::optional<std::string> sidecar; };
CommandResult import_participants(const ParticipantsRequest& request);
CommandResult export_participants(const ParticipantsRequest& request);

// ---- export (stored results as the field's standard files; the files are the command's purpose: the result names them) ----

/** what an export records of the command that made it (its provenance): the program's version, the command as given, its options */
struct ExportProvenance { std::string version, command; Json options = Json::object(); };
/** a series' time range, in seconds on the recording's clock: `from` and `to`, or `all`; none for a map */
struct ExportRange { std::optional<double> from, to; bool all = false; };

struct ExportGiftiRequest {
  std::string package, out;
  std::optional<std::string> field, labelling, kind, surface_unit;
  bool with_surface = false, hemispheres = false;
  ExportRange range;
  ExportProvenance provenance;
};
CommandResult export_gifti(const ExportGiftiRequest& request, const Progress& progress = {});
struct ExportTsvRequest {
  std::string package, out;
  std::optional<std::string> statistic, partition, field;
  std::optional<std::vector<std::string>> measures;
  ExportProvenance provenance;
};
CommandResult export_tsv(const ExportTsvRequest& request, const Progress& progress = {});
/** a source estimate (`field`: -lh.stc, -rh.stc) or a covariance statistic (`covariance`: -cov.fif); `forward` and `inverse` are refused, saying why */
struct ExportFifRequest {
  std::string package, out;
  std::optional<std::string> field, covariance, forward, inverse;
  ExportRange range;
  ExportProvenance provenance;
};
CommandResult export_fif(const ExportFifRequest& request, const Progress& progress = {});
struct ExportBidsDerivativesRequest {
  std::string package, out, results;
  std::optional<std::string> space, session, task, run, measurements;
  bool with_surface = false, stc = false;
  ExportRange range;
  ExportProvenance provenance;
};
CommandResult export_bids_derivatives(const ExportBidsDerivativesRequest& request, const Progress& progress = {});

}  // namespace nxr::io::task

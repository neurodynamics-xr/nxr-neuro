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

struct AddSurfaceRequest {
  std::string dataset, subject;
  std::string file;                        // the mesh file (the positional FILE)
  std::optional<std::string> name;
  double scale = 1.0;
  bool primary = false, no_primary = false;
  std::vector<std::string> templates;
  std::string created_by;
};
CommandResult add_surface(const AddSurfaceRequest& request, const Progress& progress = {});

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

}  // namespace nxr::io::task

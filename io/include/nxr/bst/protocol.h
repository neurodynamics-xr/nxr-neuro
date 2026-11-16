// nxr-io — a Brainstorm protocol, read as nxr's Python does: `nxr.convert.protocol` (what the Data page's Import stage lists), `nxr.convert.naming`
// (Brainstorm's own file stems as node names) and `nxr.convert.sources` (the protocol as a SOURCE VIEW: `data/protocol.mat`, else the walk).
//
//   Json subjects = list_subjects("proto");                 // [{subject, conditions: [{name, recordings, kernels, ...}], surfaces}]
//   Json view = read_source("proto");                       // nxr.source@1.0: root, kind, subjects, db_version, scanned_at, ...
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"

namespace nxr::bst {

using Json = nxr::io::Json;

// ── paths as Python prints them ──
// str(Path(p)): the separators doubled and the "." components dropped, the trailing slash removed
std::string py_path(const std::string& p);

// ── names (nxr.convert.naming) ──
// The last component of a path as Python's PurePosixPath.name gives it ("a/b/" -> "b").
std::string name_of(const std::string& path);
// `tess_cortex_pial_low.mat` -> `cortex_pial_low`
std::string surface_node_name(const std::filesystem::path& surface_file);

// ── the protocol (nxr.convert.protocol) ──
// What a protocol holds, from the folders: every subject with a condition worth importing, its conditions (a condition is a base name with up to two homes,
// the plain folder and `@raw<base>`) and its surfaces. FileNotFound when there is no `data/`.
Json list_subjects(const std::filesystem::path& root);

// ── the source view (nxr.convert.sources) ──
enum class Reader { Auto, ProtocolMat, Walk };
// `nxr.source@1.0` as a JSON document, in the field order nxr's dataclasses give (root, kind, subjects, db_version, scanned_at, index_mtime, protocol_comment, schema).
// FileNotFound when there is no `data/`. Reads the variables of `data/protocol.mat` and, of a raw link or a kernel, one field.
Json read_source(const std::filesystem::path& root, Reader prefer = Reader::Auto);
// subjects · conditions · recordings · kernels · importable
Json source_counts(const Json& view);
// `data/protocol.mat`'s view only when the file exists and reads; the cortex file `iCortex` names for `subject`, or nullopt (nxr.convert.subject.index_cortex)
std::optional<std::string> index_cortex(const std::filesystem::path& root, const std::string& subject);

// The entries of a folder whose names fit `prefix*suffix` (and contain `middle` between them, when given), sorted, as Python's Path.glob("prefix*middle*suffix") finds them.
std::vector<std::string> glob_names(const std::filesystem::path& dir, const std::string& prefix, const std::string& suffix, const std::string& middle = "");

}  // namespace nxr::bst

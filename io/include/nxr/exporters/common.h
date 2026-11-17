// nxr-io — EXPORTERS (docs/specs/2026-11-17-interoperability.md, sections 3.3 and 5): what the package holds, written as the field's standard files, by the pure writers of
// io/formats/. An exporter finds stored nodes through the task API (`find`), opens their arrays and writes what is stored, in the units stored, or refuses; it does no maths.
// This header is what every exporter shares: the context a command hands over, the PROVENANCE record that travels with every export, and the licence terms carried from the
// inputs' `origin` records.
//
// Every export writes a sidecar `<file>.json` (never lost by a format with no room for metadata) with:
//   producer    the program and its version; command: the command line; options: every option of the export
//   span        the time range read, when the export reads a recording-like series (always stated by the caller; never defaulted)
//   package     the package's name and content hash
//   inputs      each stored node read: its class, name, kind, unit, producer record, origin and the digest of its arrays (the keys' count, bytes and one SHA-256 over their
//               recorded digests)
//   terms       the licence and attribution carried from the inputs' origin records: a ShareAlike licence stays ShareAlike; `redistributable` only when every input is
//   outputs     each file written with its SHA-256
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/task.h"

namespace nxr::exporters {

using nxr::io::Json;
namespace task = nxr::io::task;

// what a command hands every exporter
struct Context {
  std::string tool = "nxr-neuro", version, command;   // the program, its version, the command line that made the export
  Json options = Json::object();                      // every option of the export, as the command line states it
  std::function<void(const Json&)> progress;          // one JSON line per stage (may be empty)
};

// The licence and attribution of what an export carries, from its inputs' origin records.
struct Terms {
  std::string licence = "unknown";                    // the licence of the output; "unknown" when no input states one
  std::string attribution = "unknown";
  bool redistributable = false;                       // true only when every input's record says so
  bool share_alike = false;                           // a ShareAlike licence among the inputs: the output carries it too
  Json origins = Json::array();                       // each input's own record: {name, class, source, licence, attribution, redistributable}
  Json to_json() const;
};
// Whether a licence text is a ShareAlike licence (CC BY-SA, ODbL, ShareAlike in words).
bool is_share_alike(const std::string& licence);
// The terms of an export of `inputs`. Two different ShareAlike licences cannot be combined: refused (`invalid_argument`), naming both.
Terms terms_of(const task::Subject& subject, const std::vector<task::Node>& inputs);

// The digest of what an input holds in the package: its arrays' keys (count, bytes, one SHA-256 over the recorded `key:size:sha256` lines).
Json input_digest(const task::Subject& subject, const task::Node& node);

// The provenance record of an export. `span`: the range read (null when the export reads no series); `extra`: what the exporter adds (e.g. the conversions it made).
Json provenance(const nxr::io::SubjectPackage& package, const task::Subject& subject, const Context& ctx, const std::vector<task::Node>& inputs, const Json& span,
                const Json& extra = Json::object());

// The files an export wrote: each is recorded in the sidecar with its SHA-256, then the sidecar is written next to the FIRST file as `<file>.json` (or at `sidecar_path`); returns that path.
std::string write_sidecar(const std::string& first_file, Json provenance, const std::vector<std::string>& files, const std::string& sidecar_path = "");

// Writes a JSON document as the sidecar writers do (indent 2, a final newline).
void write_json(const std::string& path, const Json& j);

std::string file_sha256(const std::string& path);

// The span a series export states: [from, to) seconds or the whole recording, resolved to the samples read.
struct SeriesRange {
  bool all = false;
  double from = 0, to = 0;            // seconds on the recording's clock, when !all
};

// An array's values widened to float64 (every numeric type the store holds; complex is refused with the reason).
std::vector<double> widen(const std::vector<std::uint8_t>& raw, nxr::io::DType dtype);

}  // namespace nxr::exporters

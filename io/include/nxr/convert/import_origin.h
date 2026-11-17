// nxr-io — the ORIGIN of what an importer writes (docs/audits/2026-11-17-format-mapping-and-coordinates.md): one record on every item the import made (source, licence,
// attribution, whether it may be redistributed), with the SHA-256 of the file on its source.
#pragma once
#include <string>

#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct ImportOrigin {
  std::string format;                     // `nifti`, `gifti`, `freesurfer`, `cifti`, ...: the prefix of the recorded source
  std::string source;                     // the file's path as it is recorded (relative); empty: the file's own name
  std::string source_file;                // the file to hash; empty: no hash
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
};

// The record, with the file's hash in its source when `source_file` is a file.
nxr::io::Json origin_record(const ImportOrigin& o, const std::string& path);
// Set the record on every item of the package that has none yet; returns {origin, origin_items}.
nxr::io::Json stamp_origin(nxr::io::DatasetWriter& w, const ImportOrigin& o, const std::string& path);

}  // namespace nxr::convert

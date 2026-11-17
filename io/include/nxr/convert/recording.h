// nxr-io — IMPORTERS of continuous recordings (docs/specs/2026-11-17-interoperability.md, sections 3.2 and 4): a pure format reader's recording becomes a recording of a
// subject PACKAGE, stored exactly as the Brainstorm importer stores one (the chunk and shard grid, the envelope, the events and bad segments as Selections on the time line,
// the channel flags, the sensor manifold), with an `origin` record on every item and the file's own header in the recording's producer record.
//
//   auto w = nxr::io::open_package_writer(package);
//   auto sub = nxr::io::SubjectWriter::open(w, name);
//   auto reader = nxr::convert::open_recording("fif", "sub-01_task-rest_meg.fif");
//   nxr::convert::RecordingImport opt;  opt.source = "sub-01/meg/sub-01_task-rest_meg.fif";  opt.source_file = path;
//   nxr::convert::import_recording(w, sub, *reader, opt);
//
// The recording is COPIED, block by block (2^17 samples, all channels; one block in memory), never loaded whole. Identifying header fields are kept as the file has them
// unless `anonymize` is set, which drops them (the file's start time, its subject, experimenter and description, the patient and recording identification of EDF, ...).
// Units are SI; the file's own units stay in the producer record. Epoched and evoked sources are refused (the format readers refuse them).
#pragma once
#include <memory>
#include <string>

#include "nxr/bst/session.h"
#include "nxr/formats/recording.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct RecordingImport {
  std::string name = "recording";       // the recording's name in the subject
  std::string acquisition = "rest";     // the session it is stored under
  std::string format;                   // "fif", "edf", ... (recorded in `source` and the producer record)
  std::string source;                   // the file's path as it is recorded: relative to the root the caller names, never absolute
  std::string source_file;              // the file to hash for the origin record (empty: no hash)
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;               // drop the identifying header fields
  nxr::io::Json extra = nullptr;        // what the format adds to the producer record (transforms, digitisation, projectors), an object or null
  nxr::bst::Emit progress;
};

// The formats `open_recording` reads: fif, edf, bdf, brainvision, eeglab, ctf, kit. `format_of` names the one a path is (by its extension), or throws a refusal naming the choices.
std::string format_of(const std::string& path);
std::unique_ptr<nxr::formats::RecordingReader> open_recording(const std::string& format, const std::string& path);
// What the reader knows beyond the common header and the importer keeps: FIF's transforms and digitisation, CTF's compensation grade, KIT's system.
nxr::io::Json format_record(const std::string& format, const nxr::formats::RecordingReader& reader, bool anonymize);

// Copy the recording into the subject `sub` of the package `w` writes; returns {name, id, n_channels, n_samples, sfreq, events, bad_segments, channels, origin}.
nxr::io::Json import_recording(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const nxr::formats::RecordingReader& reader, const RecordingImport& opt);

}  // namespace nxr::convert

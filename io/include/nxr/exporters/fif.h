// nxr-io — EXPORT MNE OBJECTS (docs/specs/2026-11-17-interoperability.md, 5.3): what MNE-Python reads, written by the pure writers of io/formats/.
//
//   a source map or a source series on the two hemispheres of a cortex  ->  <stem>-lh.stc and <stem>-rh.stc   (`mne.read_source_estimate`)
//
// The estimate lies on the hemispheres the glued cortex stores as components: left first, right second, each hemisphere's vertices numbered from 0 in its own surface (the
// source space is every vertex of the surface). A map is one time point (start 0, step 1 s, recorded in the sidecar); a series is read chunk by chunk through the chunk service
// over the range the caller states (required for a series, refused for a map; never defaulted) and written frame by frame, so memory is one frame whatever the range.
// An .stc file has no room for metadata: the sidecar `<stem>-lh.stc.json` holds the provenance (nxr/exporters/common.h) and the field's unit. Values are written as float32
// (`details.converted` says when the stored field was float64). Only scalar fields are written.
//
// Covariances, forward solutions and inverse operators are not written yet (their mapping to MNE's objects is under review; the FIF writers for them are to follow).
#pragma once
#include <filesystem>
#include <optional>
#include <string>

#include "nxr/exporters/common.h"

namespace nxr::exporters {

struct StcExport {
  std::string field;                       // a Field on a glued cortex (a map), or on cortex x time line (a series), by name
  std::string out;                         // the path stem: <out>-lh.stc and <out>-rh.stc (the folder must exist)
  std::optional<SeriesRange> range;        // a series: the time range read (REQUIRED); a map: must be unset
  Context ctx;
};

// Writes the two files and the sidecar; returns {files: [lh, rh], sidecar, frames, ...}.
Json export_stc(const std::filesystem::path& package, const StcExport& options);

}  // namespace nxr::exporters

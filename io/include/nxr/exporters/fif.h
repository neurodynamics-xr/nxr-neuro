// nxr-io — EXPORT MNE OBJECTS (docs/specs/2026-11-17-interoperability.md, 5.3): what MNE-Python reads, written by the pure writers of io/formats/.
//
//   a source map or a source series on the two hemispheres of a cortex  ->  <stem>-lh.stc and <stem>-rh.stc   (`mne.read_source_estimate`)
//
// The estimate lies on the hemispheres the glued cortex stores as components: left first, right second, each hemisphere's vertices numbered from 0 in its own surface (the
// source space is every vertex of the surface). A map is one time point (start 0, step 1 s, recorded in the sidecar); a series is read chunk by chunk through the chunk service
// over the range the caller states (required for a series, refused for a map; never defaulted) and written frame by frame, so memory is one frame whatever the range.
// An .stc file has no room for metadata: the sidecar `<stem>-lh.stc.json` holds the provenance (nxr/exporters/common.h) and the field's unit. Values are written as float32
// (`details.converted` says when the stored field was float64). Only scalar fields are written.
//   a covariance STATISTIC (measure `covariance`, [M x M] on its channels' Selection)  ->  <stem>-cov.fif   (`mne.read_cov`)
//
// The covariance is written as the noise covariance (kind 1) with the channel names the statistic's channel Selection stands for (the names labelling of the channel manifold), its
// degrees of freedom as `nfree` (when stored), its values and unit as stored (the sidecar records the unit and the span it was computed over; MNE's covariances are in the units of the
// channels: T^2, (T/m)^2, V^2). Only a real covariance is written; its projectors and bads (an imported one keeps them in its options) are not carried.
// Forward solutions and inverse operators are NOT written (an MNE forward needs each channel's FIF info: kind, coil type and position, which a package does not keep; an MNE inverse
// operator is the eigen-decomposition of the whitened forward, which a stored kernel does not hold): they are refused with that reason.
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

struct CovarianceExport {
  std::string statistic;                   // the covariance statistic, by name
  std::string out;                         // the path stem: <out>-cov.fif
  Context ctx;
};
// Writes <out>-cov.fif and the sidecar; returns {files, sidecar, channels, details}.
Json export_covariance(const std::filesystem::path& package, const CovarianceExport& options);

// Writes the two files and the sidecar; returns {files: [lh, rh], sidecar, frames, ...}.
Json export_stc(const std::filesystem::path& package, const StcExport& options);

}  // namespace nxr::exporters

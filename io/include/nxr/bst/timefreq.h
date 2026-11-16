// nxr-io — Brainstorm `timefreq_*.mat` results (SOURCE POWER by band, per sleep stage) as Fields: `nxr.convert.timefreq` (register D55/O30/O31, D158).
//
// A subject's `timefreq_srcpow_<stage>__*.mat` (in its `@raw<condition>` folder) and the group's `timefreq_average_*.mat` (`data/Group_analysis/@intra`) have one
// shape: `TF [nSources x nBands]` on the cortex the kernel indexes (`SurfaceFile`), `Freqs` a list of bands `[name, 'lo, hi', function]`, and, for an average,
// a `History` naming the averaged files and `nAvg`. ONE FIELD PER BAND on its SURFACE, `source/<stem>_<band>`, the band recorded in its `producer_json`.
//
//   auto info = bst::export_timefreq(sub, "timefreq_srcpow_N2__x.mat", "rest");    // nullopt when the file is not source-level
#pragma once
#include <filesystem>
#include <optional>
#include <string>

#include "nxr/bst/subject.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

// One results-level timefreq file -> `source/<stem>_<band>` [nSources] float32 per band, Fields on its surface (this subject's, else the dataset's template's);
// a group average's contributions are the members' fields. Returns nxr's info object (name, id, fields, n_sources, n_bands, sleep_stage, group), or nullopt
// when the file is not source-level (a sensor-level timefreq is never loaded).
std::optional<Json> export_timefreq(nxr::io::SubjectWriter& sub, const std::filesystem::path& mat_path, const std::optional<std::string>& acquisition, bool group = false);

// SESSION HOOK. `convert_condition`'s step "SOURCE POWER BY BAND": every `timefreq_*.mat` of the condition's folder and of its `@raw<condition>` folder
// (`cond_dir.parent / "@raw" + condition`), each exported on the cortex the kernel indexes; each one written is announced as `{"stage": "field", ...info}`.
// The session port calls this after the condition's surfaces are written. Returns how many files were exported.
int export_condition_timefreqs(nxr::io::SubjectWriter& sub, const std::filesystem::path& cond_dir, const std::string& condition, const std::optional<std::string>& acquisition,
                               const Emit& emit);

}  // namespace nxr::bst

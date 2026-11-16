// nxr-pipelines — THE JOINT PARTITION of space × time (spec docs/specs/2026-11-13-cortical-events.md, phase 2a): a `set` Selection on the product (surface × the recording's time line) whose
// cells are (a spatial tile of level l of the surface's tile ladder) × (a time tile of level m of the recording's constant-Q ladder). The Cortical Flow Studio atlas's layout, ported (audit
// docs/audits/2026-11-13-atlas-joint-cells.md 1.2): code = (tile << B) | frame, B = bitsFor(F) = max(1, ceil(log2(max(2, F)))), F the time tiles. CLOSED FORM: nothing stored per cell; `n_members`
// = tiles × F and the layout is in params_json.joint {space, time, space_level, time_level, band [lo, hi] Hz (the time level's `resolves_hz`: the level IS the band, D84), bits, tiles, frames,
// layout, recording, surface}. ONE per pair (l, m): the pair names both bands, and the measurements over it (`run cortical_cells`) are of the recording, as the time tiles' are.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct JointComputeOptions {
  std::string recording;                     // REQUIRED: its time line's constant-Q ladder (`constantq compute`) gives the time level
  std::optional<std::string> surface;        // the subject's surface (default: the primary one); its tile ladder (`tiles compute`) gives the spatial level
  int space_level = -1;                      // REQUIRED: the surface ladder's level (2^l tiles a piece)
  int time_level = -1;                       // REQUIRED: the constant-Q ladder's level (its member ordinal; the band it resolves is the level's)
  bool replace = false;                      // an existing joint Selection of the pair is removed (with the measurements over it) and rewritten
};
struct JointComputeResult {
  nxr::io::Json info;
  bool existed = false;                      // the stored one was kept
};
JointComputeResult joint_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const JointComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});
/** the bits of the time index: max(1, ceil(log2(max(2, frames)))) (the atlas's `bitsFor`) */
int joint_bits(std::int64_t frames);

}  // namespace nxr::pipelines

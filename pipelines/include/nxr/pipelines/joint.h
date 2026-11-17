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

/** the bits of the time index: max(1, ceil(log2(max(2, frames)))) (the atlas's `bitsFor`) */
int joint_bits(std::int64_t frames);

}  // namespace nxr::pipelines

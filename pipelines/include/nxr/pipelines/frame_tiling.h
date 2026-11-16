// nxr-pipelines — THE FRAME TILING of a recording, on the TOWER (D135;, measurement-pyramid port step 1; nxr `pipelines/atlas/build.py`, `tower.py`): the frames are cycles of ONE tower level
// (86 400 · 2^L s; frame k = code `frame_code0 + k`, a level-m tile = code >> m), so every frame has a dyadic parent and the tiling is the measurement pyramid's own. A `spans` Selection on the recording's time
// line, stored as nxr stores it: name `<recording> frames <s> s`, `params_json.tiling` {origin, width, hop, count} and `params_json.atlas` {frame_s, frame_level, frame_code0, anchor, start_s, made}. The level
// is named (`frame_level`) or derived from a band: the coarsest tower level whose frame is at or under 1/(centre × frames_per_cycle) (the project's 10° step is 36). A frame's TIME is its tile's START
// (tower.h `FrameTiling::position`), as the Studio's and nxr's resampled frames are; the `frames` step samples a chunk's field there. Replaces the per-band tiling of PR #144 (not kept: it broke the
// dyadic nesting of time). ONE per (recording, level).
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct FrameTilingOptions {
  std::string recording;                     // REQUIRED
  std::optional<int> frame_level;            // the tower level of a frame (0 = the day, −18 = 0.33 s, −25 = 2.57 ms); or derive it from a band:
  std::optional<double> centre_hz;           //   the band's centre (Hz): the middle of `band` when that is given
  std::optional<std::pair<double, double>> band;   //   [lo, hi] Hz: the centre is (lo + hi) / 2, and the band is recorded
  int frames_per_cycle = 36;                 //   frames to a cycle of the centre (the project's 10° step); the level is the coarsest whose frame is at or under 1/(centre × frames_per_cycle)
  std::optional<double> start_s;             // the recording's first sample as seconds from local midnight: anchor "day" (default: the anchor is the recording)
  bool replace = false;                      // an existing tiling of the same name is removed and rewritten
};
struct FrameTilingResult {
  nxr::io::Json info;
  bool existed = false;
};
FrameTilingResult frame_tiling_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const FrameTilingOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});
/** the name the tiling is stored under, nxr's: `<recording> frames <frame_s:g> s` (frame_s = 86 400 · 2^level) */
std::string frame_tiling_name(const std::string& recording, int frame_level);

}  // namespace nxr::pipelines

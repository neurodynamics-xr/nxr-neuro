#pragma once
// nxr::tower — THE FRAME GRID: a recording's tower frames (tower.h `FrameTiling`) as a GRID of its time line, made by the selection system's own maker (a tiling whose hop is its width), so every
// reduction over frames is a measurement over a grid's members, and its hierarchy is the tower's closed form (frame k is the tile `code0 + k`, its level-m tile `>> m`).
//   frameGrid       the tiling `selection::tiled`: member k = [origin + k frame, origin + (k + 1) frame) seconds from the recording's origin, `origin` = `FrameTiling::origin_s()` (before the first sample
//                   when the first frame is partial), the unit s. The frame node (`frame_tiling`) describes this tiling; io stores it.
//   frameIntervals  that grid resolved on the recording's time line (`selection::resolve_intervals`): one interval of samples a frame, a sample in the frame of its tower tile (floor((offset + i) / (frame fs)));
//                   what `pyramid::frameOfSamples` and `addSampleCounts` take.
// Pure functions of numbers (the rate and the length are the caller's): no file, no chunk, no recording. std::invalid_argument for a rate or length that cannot be tiled.
#include <cstdint>

#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"
#include "nxr/tower.h"

namespace nxr::tower {

/** the frames of `t` as a time grid in seconds from the recording's own origin: `count` members of `frame_s()`, a new one every `frame_s()` from `origin_s()` */
nxr::selection::Selection frameGrid(const FrameTiling& t);
/** the grid on the line of `t.samples` samples at `t.sfreq` Hz (origin 0, in seconds): one interval of samples a frame (a partial first or last frame clipped to the line) */
nxr::selection::Intervals frameIntervals(const FrameTiling& t);

}  // namespace nxr::tower

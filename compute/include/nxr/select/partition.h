// nxr::selection (compute) — THE GRID METHODS: partitions of a MANIFOLD, defined before any data (selection.h: a grid's hierarchy is closed form), made in
// memory; io stores them. Compute takes arrays: a line is a count and a spacing, a surface its positions and faces.
//
//   time_grid         a time line   the dyadic tiling of tiles of 2^k samples from the first sample (the chunk grid is k = 15); level k' >= k holds member >> (k' - k)
//   time_rule         a time line   ONE rule family for tiles, bands, frames, Welch segments, windows and chunks: member k = [origin + k hop, origin + k hop + width) in samples
//                                   (`TimeRule`; the level of a dyadic kind is the one `grid.h` resolves from the stated quantity)
//   constant_q        a time line   a grid with LEVELS, each a tiling of tiles of 2^k samples (the level's identity is its position, finest first, its label k), the
//                                   levels a constant-Q band design needs, each with the band (Hz) it resolves; constant_q_design returns it with the facts the
//                                   design settled (cycles, strides, rates, samples a cycle) for the store
//   dyadic            a surface     a dyadic GRID with LEVELS: equal-area bisection, 2^l tiles a connected piece; the member number is the tile code h 2^D + t at the
//                                   finest level D and level l's member is that code >> (D - l) (closed form: nothing per level is held but its label l and its
//                                   eigenvalue band by Weyl's law, 1/unit^2)
//   (the band sets on the frequency and eigenvalue lines are made by selection/bands.h)
//   transfer          a surface     a dyadic grid carried from another surface through the registered spheres (nearest vertex, piece by piece): the same codes
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"

namespace nxr::select {

using namespace nxr::selection;   // the makers build the class side's Selection (selection/selection.h); it is declared there

using nxr::selection::Line;   // a line is described on the class side (selection/resolved.h): the form a selection is resolved over
/** a triangle mesh: positions [vertices x 3] in `unit` and faces [face_count x 3], row-major; each connected piece (a hemisphere) one contiguous block of
 *  vertices. The unit is the caller's (never defaulted): it gives the eigenvalues theirs, 1/unit^2. */
struct Surface {
  std::string unit;
  const double* positions = nullptr;
  std::int32_t vertices = 0;
  const std::int32_t* faces = nullptr;
  std::int32_t face_count = 0;
};

/** CONSTANT-Q (the Cortical Flow Studio's backend/src/db/constant-q.ts): bands of `voices` per octave down from f_max; a band's window is `cycles` periods of
 *  its low edge rounded UP to 2^k samples; bands sharing a window share level k. */
struct ConstantQ {
  std::string method = "hertz";            // hertz: cycles from the rate (about 6.8 at 1200 Hz), f_max 64 Hz (capped by Nyquist), f_min 0.5 Hz · cycles: 8 cycles, f_max Nyquist, f_min 0
  int voices = 6;                          // bands per octave
  std::optional<double> cycles;            // cycles of a band's low edge in its window; default: the method's
  std::optional<double> f_min, f_max;      // Hz; default: the method's (f_max at most Nyquist)
};
/** DYADIC (the Studio's engine/ladder.ts): each level halves every tile of the one above by equal area across its geodesic diameter, each tile one piece. */
struct Dyadic {
  std::optional<int> depth;                // the finest level; default: the finest at which every tile of every piece still holds a vertex
};
/** TRANSFER: `from` (a dyadic grid with levels, on another surface of `from_vertices` vertices) carried through the registered spheres; its depth, slots, levels'
 *  bands and lookup come with it, and its tile codes stay the member numbers (a tile no vertex reaches is an empty member) */
struct Transfer {
  const Selection* from = nullptr;
  const double* from_sphere = nullptr;     // [from_vertices x 3]
  const std::int32_t* from_faces = nullptr;
  std::int32_t from_vertices = 0, from_face_count = 0;
  const double* sphere = nullptr;          // this surface's registered sphere, [vertices x 3]
};

/** THE TIME RULE (frame density and frame origin are OPTIONS of the compute function, the pipeline's default is its YAML's, no method
 *  constant is hard-wired here). Every member is [origin + i hop, origin + i hop + width) on the line's samples; the kinds differ only in how width and hop are resolved from the stated
 *  quantity:
 *    tiles      duration_s            2^k samples, k = `level_of_duration` (Rheome's natural_level); hop = width
 *    band       band_hz               k = `time_level_of_band` (the Studio's fitLevelOf: 4 cycles of the band's centre); hop = width
 *    frames     band_hz + frames_per_tile (a power of two) OR frames_per_cycle   the band's tile of 2^k samples cut into `frames_per_tile` frames (width 2^(k - log2(frames_per_tile)) samples,
 *                                      nested in the band's level), OR frames of 1 / (centre x frames_per_cycle) seconds (not whole samples in general); hop = width
 *    segments   duration_s + overlap + segment_rule   width as Welch's presets resolve it (`tile`, `round`, `even`: welch::segmentFor); hop = width - floor(width x overlap); members overlap
 *               when hop < width (a family, not a partition); only whole segments
 *    windows    duration_s + step_s   width and hop each snapped up to 2^k samples; only whole windows
 *    chunks     none                  k = 15 (the chunk grid)
 *  ORIGIN: an option, a POINT of the line (a `point` selection: the sample, so the recording's start is sample 0, a clock time and an event are given as the sample they are at) or a SPAN
 *  (an event: its start, rounded outward); empty: the line's first sample. The grid runs from the origin to the end of the line (samples before it are in no member); the first
 *  frame starts AT the origin. A grid of the default origin, tiles, band or chunks kind, is one level of the line's dyadic time grid (member k' levels up is member >> (k' - k)). */
struct TimeRule {
  enum class Kind { tiles, band, frames, segments, windows, chunks };
  Kind kind = Kind::tiles;
  std::optional<double> duration_s;                    // tiles, segments, windows: the stated duration
  std::optional<std::pair<double, double>> band_hz;    // band, frames
  std::optional<std::int64_t> frames_per_tile;         // frames: a power of two (2^s); exactly one of this and frames_per_cycle
  std::optional<double> frames_per_cycle;              // frames: frames per cycle of the band's centre sqrt(lo hi)
  double overlap = 0;                                  // segments: the fraction of a segment shared with the next, [0, 1)
  std::string segment_rule = "round";                  // segments: tile | round | even (welch::segmentFor)
  std::optional<std::int64_t> segment_samples;         // segments: the width in samples instead of a duration (welch::segmentFor)
  std::optional<int> segment_level;                    // segments, rule tile: the level k instead of a duration
  std::optional<double> step_s;                        // windows: the step
  std::optional<Selection> origin;                     // a point or a span of the line; none: the first sample
  double cycles = 4;                                   // band, frames: the cycles of the band's centre a tile holds (the Studio's CYCLES)
};
/** what `time_rule` resolved: the grid, its width and hop in samples, the level k of a dyadic kind (else -1) and the SNAPPED width in seconds (what the level, or the frame, actually is) */
struct TimeGrid {
  Selection grid;
  std::int64_t width_samples = 0, hop_samples = 0;     // whole samples; 0 for frames per cycle (not whole samples)
  int level = -1;
  double width_s = 0, hop_s = 0;
  std::int64_t origin_sample = 0;
};
TimeGrid time_rule_resolved(const Line& time, const TimeRule& rule);
Selection time_rule(const Line& time, const TimeRule& rule);

/** what the design settled about one level (the facts the stored ladder carries), in the ladder's order */
struct ConstantQLevel {
  int power = 0;                           // k: tiles of 2^k samples (the level's label)
  std::int64_t window_samples = 0;         // 2^k
  int bands = 0;                           // how many of the design's bands share this window
  double lo = 0, hi = 0;                   // Hz: the union of those bands
  std::int64_t stride = 1;                 // the decimation at the low edge (constantq::strideFor)
  double rate_hz = 0;                      // sample rate / stride
  std::int64_t points = 0;                 // samples of a window after decimation
  double samples_per_cycle_lo = 0, samples_per_cycle_hi = 0;   // after decimation, at the low and high edge
  bool nyquist_bound = false;              // stride 1 and still fewer than constantq::kMultirateMin samples a cycle at the low edge
};
struct ConstantQDesign {
  Selection ladder;                        // what `constant_q` returns
  std::string method;                      // hertz | cycles
  int voices = 0;
  double sfreq = 0;
  double cycles = 0;                       // the cycles of a band's low edge in its window, as resolved (the option, else the method's)
  double f_min = 0, f_max = 0;             // Hz: the lowest band's low edge and the top of the design as resolved
  std::vector<ConstantQLevel> levels;      // one per ladder level, finest first
  std::vector<std::pair<double, double>> bands;   // the design's bands [lo, hi) in Hz, highest first (the rule `frequency` of selection/bands.h makes the band set from them)
};
/** the dyadic time grid of tiles of 2^k samples from the line's first sample (count = ceil(count / 2^k)): the chunk grid at k = 15, a frame tiling, a coupled
 *  level (grid.h). Its member k' levels up is member >> (k' - k). std::invalid_argument for k outside 0 .. 62. */
Selection time_grid(const Line& time, int k);
ConstantQDesign constant_q_design(const Line& time, const ConstantQ& method);
Selection constant_q(const Line& time, const ConstantQ& method);
Selection dyadic(const Surface& surface, const Dyadic& method);
Selection transfer(const Surface& surface, const Transfer& method);

}  // namespace nxr::select

// nxr::selection — A GRID ON A SAMPLED LINE: the partition of a line's samples into tiles of `width` samples, tile k = [origin + k·width, origin + (k+1)·width),
// clipped to the line's [0, samples). The time partitions are this one grid: the chunk grid (2^k samples), a constant-Q level (2^k samples), a Welch
// segmentation (width and hop), the frames of a tower level (when its period is whole samples). A DYADIC grid (width 2^level) nests: tile k of level L is
// inside tile k >> m of level L + m (`parent`), which is what every roll-up over time rests on.
//
// Codes are absolute on the grid (counted from `origin`, negative before it), so two lines on the same grid share their codes. Pure arithmetic of integers:
// no file, no recording, no clock (`seconds` and `from_seconds` convert to and from the stored closed form, seconds on the line's coordinate system).
//
//   const auto g = nxr::selection::dyadic(9, samples);       // 512-sample tiles from sample 0
//   g.count(); tile(g, k); tile_of(g, s);                     // its members, the member of a sample
//   touching(g, a, b); within(g, a, b);                       // the members a span of samples touches, or holds whole
//   coarser(g, 2); parent(k, 2); children(k, 2);              // the nesting
#pragma once
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace nxr::selection {

/** A regular grid of tiles on a line of `samples` samples. */
struct LineGrid {
  std::int64_t width = 1;      // samples a tile (> 0)
  std::int64_t origin = 0;     // the sample tile 0 starts at (the grid's phase; any integer)
  std::int64_t samples = 0;    // the line's length: the tiles are clipped to [0, samples)
  /** log2(width) when the width is a power of two, else -1 */
  int level() const;
  bool dyadic() const { return level() >= 0; }
  /** the first and one past the last code of the tiles that touch [0, samples) (empty when samples is 0) */
  std::int64_t first() const;
  std::int64_t end() const;
  std::int64_t count() const { return end() - first(); }
};

/** a dyadic grid: tiles of 2^level samples (0 <= level <= 62); std::invalid_argument otherwise */
LineGrid dyadic(int level, std::int64_t samples, std::int64_t origin = 0);
/** a regular grid of any width (> 0) */
LineGrid regular(std::int64_t width, std::int64_t samples, std::int64_t origin = 0);

/** THE LEVEL OF A DURATION on a line sampled at `sfreq` Hz: the nearest power of two of samples, round(log2(seconds · sfreq)), at least 0 (the rule the chunk
 *  grid and the constant-Q ladder snap to); `at_or_under` and `at_or_over` the floor and the ceiling instead */
int level_of(double seconds, double sfreq);
int level_at_or_under(double seconds, double sfreq);
int level_at_or_over(double seconds, double sfreq);

/** One member of a grid: its code and its samples [start, stop) on the line (clipped); `whole` when the clip took nothing off. */
struct Tile {
  std::int64_t code = 0, start = 0, stop = 0;
  bool whole = true;
};
/** member `code` (std::out_of_range when it does not touch the line) */
Tile tile(const LineGrid& g, std::int64_t code);
/** the member a sample lies in (the sample need not be on the line: the grid extends past it) */
std::int64_t tile_of(const LineGrid& g, std::int64_t sample);
/** the codes [first, end) of the members a span of samples [start, stop) touches, clipped to the line */
std::pair<std::int64_t, std::int64_t> touching(const LineGrid& g, std::int64_t start, std::int64_t stop);
/** the codes [first, end) of the members the span holds WHOLE (the tile inside the span and inside the line) */
std::pair<std::int64_t, std::int64_t> within(const LineGrid& g, std::int64_t start, std::int64_t stop);
/** a span of samples snapped onto the grid: outward (the members it touches) or inward (the members it holds whole); clipped to the line */
std::pair<std::int64_t, std::int64_t> snap_out(const LineGrid& g, std::int64_t start, std::int64_t stop);
std::pair<std::int64_t, std::int64_t> snap_in(const LineGrid& g, std::int64_t start, std::int64_t stop);

/** THE NESTING of a dyadic grid: the grid `m` levels coarser (same origin, width << m); a member's ancestor m levels up (floor(code / 2^m)); its descendants m
 *  levels down, codes [code << m, (code + 1) << m). std::invalid_argument for a grid that is not dyadic or a level past 62. */
LineGrid coarser(const LineGrid& g, int m);
std::int64_t parent(std::int64_t code, int m);
std::pair<std::int64_t, std::int64_t> children(std::int64_t code, int m);

/** THE PARTITION as an array: the code of every sample of [start, stop) (the line's part of it) */
std::vector<std::int64_t> codes(const LineGrid& g, std::int64_t start, std::int64_t stop);
/** samples per member, codes [first, end): all of them, or only the good ones (`good`: one flag a sample of the line, empty: every sample good) */
std::vector<std::int64_t> counts(const LineGrid& g, const std::vector<std::uint8_t>& good = {});

/** THE CLOSED FORM a grid is stored as (a spans selection's `params_json.tiling`, D64/D87): seconds on the line's coordinate system, `t0` the time of sample 0
 *  and `sfreq` its rate; `end` the line's end. */
struct Closed {
  double origin = 0, width = 0, hop = 0;
  std::int64_t count = 0;
  double end = 0;
};
Closed seconds(const LineGrid& g, double sfreq, double t0 = 0);
/** a stored closed form back as a grid: its width and origin must be whole samples (to within `tolerance` of a sample) and its hop its width (contiguous);
 *  std::invalid_argument otherwise */
LineGrid from_seconds(const Closed& c, double sfreq, double t0, std::int64_t samples, double tolerance = 1e-6);

}  // namespace nxr::selection

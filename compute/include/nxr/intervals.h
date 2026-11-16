#pragma once
// nxr::intervals — TEMPORAL EVENTS: the building blocks that find time windows of interest in a time series (spec docs/specs/2026-11-13-single-events.md,
// family A). PURE MATHS: a series in (any series: a sensor channel, a source or region signal, an amplitude; the blocks never know which), spans out.
// A span is [start, stop) in samples of the series given. Span-wide thresholds (a percentile, a median) are the caller's: measurements over a time
// partition, handed in as numbers.
//
//   crossings     spans where x ≥ `exit`, each holding at least one sample ≥ `entry` (hysteresis: enter high, extend while above the low level; entry =
//                 exit is a plain threshold) — nxr burst_stages / spindle_envelope (median + 1.5 SD to enter, + 1.0 SD to extend), burst_mask (one level)
//   mergeSpans    spans whose gap (next start − previous stop) is below `gap` samples joined — nxr: events closer than one cycle merged
//   keepSpans     spans whose length is in [minLength, maxLength] kept — nxr: bursts of at least 3 cycles; spindles 0.5–3 s
//   halfWaves     the negative half-waves: from a downward zero crossing to the next upward one, with the trough's sample and value, kept by length —
//                 nxr sleep_events.detect_slow_waves (0.25–1.0 s); the depth threshold is the caller's (a percentile of the troughs)
//   rollupEvents  a measurement of tiles rolled up over the tiles of each event, by the measure's reduction (mean: the count-weighted mean of the tile
//                 means; max: the max of the tile maxima): the exact value over the union of the tiles, as the lookup merges members
//   eventDurations, eventIntervalNext   the events' own geometry in seconds: stop - start, and the next event's start sharing a row minus this stop
//   median, mad   the median, and the robust SD 1.4826 · median |x − median| (nxr spindle_envelope's baseline)
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace nxr::intervals {

struct Span {
  std::int64_t start = 0, stop = 0;       // samples [start, stop)
  std::int64_t peak = -1;                 // the sample of the extreme value inside (the maximum for crossings, the trough for half-waves)
  double value = 0;                       // that value
  std::int64_t length() const { return stop - start; }
};

std::vector<Span> crossings(const Eigen::Ref<const Eigen::VectorXd>& x, double entry, double exit);
std::vector<Span> mergeSpans(const std::vector<Span>& spans, std::int64_t gap);
std::vector<Span> keepSpans(const std::vector<Span>& spans, std::int64_t minLength, std::int64_t maxLength);
std::vector<Span> halfWaves(const Eigen::Ref<const Eigen::VectorXd>& x, std::int64_t minLength, std::int64_t maxLength);
double median(const Eigen::Ref<const Eigen::VectorXd>& x);
double mad(const Eigen::Ref<const Eigen::VectorXd>& x);

// An event as a run of tiles [first, stop) of a tile level, found on `row` (-1: the mean over the rows, so on every row).
struct TileEvent { std::int64_t first = 0, stop = 0, row = -1; };
struct EventValues {
  Eigen::MatrixXd value;     // [rows x events]: the event's value on the rows it belongs to (its row, or all for row -1), NaN elsewhere
  Eigen::VectorXd count;     // [events]: the samples over the event's tiles (the sum of their counts)
};
/** `tiles` [rows x tiles] a measurement per tile (NaN: a tile not whole and good), `counts` [tiles] its samples, `reduction` "mean" or "max" (the registry's).
 *  An event holding a NaN tile on a row is NaN on that row. std::invalid_argument for another reduction or an event outside the tiles. */
EventValues rollupEvents(const Eigen::MatrixXd& tiles, const Eigen::VectorXd& counts, const std::vector<TileEvent>& events, const std::string& reduction);
/** the events' durations: (stop - first) * width, in the unit of `width` */
std::vector<double> eventDurations(const std::vector<TileEvent>& events, double width);
/** per event, the start of the next event that shares its row (any row for an event of row -1) minus its own stop, times `width`; NaN for the last */
std::vector<double> eventIntervalNext(const std::vector<TileEvent>& events, double width);

}  // namespace nxr::intervals

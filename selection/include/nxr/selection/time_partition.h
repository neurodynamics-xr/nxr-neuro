// nxr::selection — THE TIME PARTITION OF A RECORDING: its levels, each the partition of its samples into tiles of 2^k samples from the first sample (tile j =
// [j·2^k, (j+1)·2^k), the last clipped), and, by constant-Q, the band each level resolves. A level is named by k (tiles of 2^k samples); a tile is reached by
// arithmetic (j = sample >> k), never stored. Methods other than constant-Q come later, as options of the same function.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nxr::selection {

/** CONSTANT-Q (the Cortical Flow Studio's backend/src/db/constant-q.ts): bands of `voices` per octave down from f_max; a band's window is `cycles` periods of its
 *  low edge rounded UP to a power of two of samples; bands sharing a window share a level. */
struct ConstantQ {
  std::string method = "hertz";            // hertz: cycles from the rate (about 6.8 at 1200 Hz), f_max 64 Hz (capped by Nyquist), f_min 0.5 Hz · cycles: 8 cycles, f_max Nyquist, f_min 0
  int voices = 6;                          // bands per octave
  std::optional<double> cycles;            // cycles of a band's low edge in its window; default: the method's
  std::optional<double> f_min, f_max;      // Hz; default: the method's (f_max at most Nyquist)
};

/** one band of the spectrum: [lo_hz, hi_hz], and the level whose tiles resolve it */
struct TimeBand {
  double lo_hz = 0, hi_hz = 0;
  int k = 0;
};

/** one level of the partition: tiles of 2^k samples; `count` tiles cover the recording; [lo_hz, hi_hz] the bands it resolves */
struct TimeLevel {
  int k = 0;
  std::int64_t count = 0;
  double lo_hz = 0, hi_hz = 0;
};

struct TimePartition {
  std::int64_t samples = 0;
  double sfreq = 0;
  std::vector<TimeLevel> levels;           // finest first
  std::vector<TimeBand> bands;             // highest first
};

/** the time partition of a recording of `samples` samples at `sfreq` Hz, by constant-Q; std::invalid_argument for options it cannot meet */
TimePartition time_partition(std::int64_t samples, double sfreq, const ConstantQ& options = {});

}  // namespace nxr::selection

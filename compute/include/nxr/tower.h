#pragma once
// nxr::tower — THE TIME TOWER (D135; nxr `pipelines/atlas/tower.py`, `measure/rollup.py`; the Studio's `db/rollup.ts` `rollTime`): time is a tower of nested cycles.
// Level L is one cycle of 86 400 · 2^L seconds (the day is level 0; a level −18 cycle is 0.3296 s). A tile of level L has the code `floor(t / period(L))` (t from the anchor) and its parent at level
// L + m is `code >> m`: the dyadic nesting every reduction in time rests on (the maintainers' time-tiles rule). Frames are the cycles of one chosen level (`frame_level`); frame k is the code `frame_code0 + k`.
// Pure functions of numbers and arrays: no file, no chunk, no recording (the caller names the sampling rate, the sample count and the anchor).
//
//   Placement      a recording on the tower: anchor "recording" (tiles counted from its first sample), "day" (`start_s` seconds from local midnight) or "birth" (also `day`: days since birth). A
//                  frame is generally not a whole number of samples (395.5 at 1200 Hz), so each sample is assigned to its tile, floor((start_s · fs + i) / (period · fs)), computed IN SAMPLES so a boundary on a
//                  sample is exact (nxr `Placement.sample_codes`).
//   FrameTiling    the frames of a recording at one level: code0, count (the tiles touched by at least one sample: the first and the last may be partial), the frame's start in samples, the frame
//                  of a sample, the frames whose start falls in a span of samples (a partition over any partition of the samples), the good-sample count of every frame.
//   rollTime       the time roll-up of an array [rows × frames] to the tiles of 2^m frames ALIGNED ON THE TOWER (a neutral front pad of code0 mod 2^m frames, a neutral tail): only sum, max and min merge.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nxr::tower {

constexpr double kDayS = 86400.0;

/** The period of a level-L cycle (s): 86 400 · 2^L (exact in binary). */
double periodS(int level);
/** The finest level whose cycle is at least `seconds` long: ceil(log2(seconds / day)) (nxr `level_for_seconds`). */
int levelForSeconds(double seconds);
/** The coarsest level whose cycle is at most `seconds` long: floor(log2(seconds / day)). For a frame of at most 1/(centre × frames a cycle) seconds: the level whose frames are the fewest that are still fine enough. */
int levelAtOrUnder(double seconds);
/** The level whose frame is at or under a band's period / frames_per_cycle (the finest tower step that resolves that many frames to a cycle of `centre_hz`); throws std::invalid_argument for a centre or count at or below 0. */
int frameLevelFor(double centre_hz, int frames_per_cycle);
/** The tile `up` levels above a code: code >> up (codes are non-negative). */
std::int64_t parent(std::int64_t code, int up);

struct Placement {
    double sfreq = 0;
    std::int64_t n_samples = 0;
    std::optional<double> start_s;         // seconds from local midnight at the first sample (anchor day or birth)
    std::optional<std::int64_t> day;       // days since birth (anchor birth)
    std::string anchor() const { return !start_s ? "recording" : (!day ? "day" : "birth"); }
    /** The coarsest level a tile is known at: the recording's own length (anchor recording), 0 (the day), none (birth). */
    std::optional<int> coarsest() const;
    /** The tile (at `level`) of each sample in [start, stop) (stop < 0: the end), in samples; throws std::invalid_argument for a level above what is known. */
    std::vector<std::int64_t> sampleCodes(int level, std::int64_t start = 0, std::int64_t stop = -1) const;
};

/** The frames of a recording on the tower at `level` (≤ 0; anchor recording or day: a birth anchor is a Placement's only). */
struct FrameTiling {
    double sfreq = 0;
    std::int64_t samples = 0;
    int level = -18;
    double start_s = 0;                    // 0 for the anchor recording
    bool day_anchored = false;
    std::int64_t code0 = 0;                // the tower code of frame 0 (`frame_code0`)
    std::int64_t count = 0;                // tiles touched by a sample
    double frame_s() const;                // period of the level
    double fps() const { return 1.0 / frame_s(); }
    double tile_samples() const;           // the frame's length in samples (not a whole number in general)
    double offset_samples() const { return start_s * sfreq; }
    /** frame k's start in samples from the line's first sample (negative for a partial first frame) */
    double position(std::int64_t k) const;
    /** the origin of the stored tiling in seconds from the recording's own origin: code0 · frame_s − start_s (nxr `build.py`, `rec.origin + code0 * frame_s − start_s`) */
    double origin_s() const;
    /** the frame holding sample i: floor((offset + i) / (frame · fs)) − code0 */
    std::int64_t frameOfSample(std::int64_t i) const;
    /** the tower code of frame k */
    std::int64_t code(std::int64_t k) const { return code0 + k; }
    /** a frame's tile `up` levels above: (code0 + k) >> up */
    std::int64_t tileOfFrame(std::int64_t k, int up) const { return parent(code(k), up); }
};
/** Throws std::invalid_argument for a rate, sample count or level (above 0) that cannot be tiled, or a start outside [0, 86 400) s. */
FrameTiling makeFrameTiling(double sfreq, std::int64_t samples, int level, std::optional<double> start_s = std::nullopt);
/** The frames [k0, k1) whose START lies in the samples [first, first + n): a partition of the frames over any partition of the samples (a result built chunk by chunk does not depend on the chunk size).
 *  The frames before the first sample (a partial first frame) belong to the span that starts at sample 0. */
std::pair<std::int64_t, std::int64_t> frameSpan(const FrameTiling& t, std::int64_t first, std::int64_t n);
/** The number of samples of each frame (the measurement pyramid's per-frame `samples`); `good` (n entries, 0 = bad; empty: all good) counts the good ones only. */
std::vector<std::int64_t> sampleCounts(const FrameTiling& t, const std::vector<std::uint8_t>& good = {});

enum class Merge { Sum, Max, Min };
/** x [rows × frames] (row-major) rolled up to the tower's tiles of 2^m frames (m ≥ 0), frame 0 being the tower code `code0`: tile i of the result is the code (code0 >> m) + i. The first and last tiles keep
 *  what they have (a neutral pad: 0, −∞, +∞). Returns [rows × tiles]; `tiles` receives the count. (nxr `rollup.time`, the Studio's `rollTime`.) */
std::vector<double> rollTime(const std::vector<double>& x, std::int64_t rows, std::int64_t frames, int m, Merge op, std::int64_t code0, std::int64_t* tiles = nullptr);

}  // namespace nxr::tower

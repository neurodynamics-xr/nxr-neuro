#pragma once
// nxr::io — THE CONSTANT-Q TILING of a time Line, a Selection (D64): a ladder of time resolutions, each level a closed-form tessellation of the line. A line
// port of the Cortical Flow Studio's `backend/src/db/constant-q.ts` (the reference: `constantQTiling`, `levelsOf`, `anchorsFor`, `strideFor`, `tileOf`,
// `tileIndexAt`, `tilesWithin`, `levelInfos`, `tileInfoOf`, `constantQDictionaryRows`, `levelTileDictionaryRows`, `constantQRows`, `writeConstantQ`), the
// Studio's time-tile policy:
//   - bands at `voices` per octave from the top down; a band's window holds `cycles` cycles of its LOWEST frequency, rounded UP to a power of two of samples
//     (so a coarse tile is the union of finer ones and a tile never straddles a stored chunk); the ladder stores ONE LEVEL PER DISTINCT WINDOW (D84: a tiling
//     of TIME and nothing else; what a window resolves in Hz is derived and referenced by nothing);
//   - the `hertz` anchor (default): octave edges on powers of two in Hz, capped at 64 Hz, floored at 0.5 Hz; `cycles` (pre-2026-09-24) pins the cycle count;
//   - stored as a `ladder` Selection on the time Line (cell '1'), its members the levels, each a `spans` member whose membership is the closed form in
//     `params_json.tiling` {origin, width, hop, count} (seconds; D87: the ordinal is the code, nothing per sample); a dictionary of the levels (window in
//     samples and seconds, tile count, finest tiles, what it resolves) and an EMPTY tile dictionary per level (where a per-tile label or note lands, D83).
// It lives in io because it is the tiling's own closed form (as the chunk grid is): no signal is computed here.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"

namespace nxr::io {

class SubjectWriter;

namespace constantq {

enum class Method { Hertz, Cycles };
struct Params {
  std::optional<Method> method;      // default Hertz
  std::optional<int> voices;         // default 6
  std::optional<double> cycles;      // overrides the method's
  std::optional<double> fMax;        // Hz; overrides the method's (and the 64 Hz cap); ceiling Nyquist
  std::optional<double> fMin;        // Hz; default 0.5 under hertz, 0 under cycles
};
constexpr double kCapHz = 64, kFloorHz = 0.5;
constexpr int kMultirateMin = 24, kMultirateMax = 48;

struct Band {
  int ordinal = 0, octave = 0, voice = 0;
  double lo_hz = 0, hi_hz = 0, centre_hz = 0, q = 0;
  std::int64_t window_samples = 0;
  double window_s = 0;
  std::int64_t hop_samples = 0, count = 0;
};
struct Level {
  int level = 0;
  std::int64_t window_samples = 0;
  double window_s = 0;
  std::int64_t hop_samples = 0, count = 0, finest_tiles = 0;
  double resolves_lo = 0, resolves_hi = 0;
  int bands = 0;
  std::int64_t stride = 1;
  double rate_hz = 0;
  std::int64_t points = 0;
  double spc_lo = 0, spc_hi = 0;     // samples per cycle after decimation, at the band's low and high edges
  bool nyquist_bound = false;
};
struct Tiling {
  double sfreq = 0;
  std::int64_t samples = 0;
  double origin = 0;
  Method method = Method::Hertz;
  int voices = 6;
  double cycles = 8, f_max = 0, f_min = 0;
  std::vector<Band> bands;           // the design (never stored as rows)
  std::vector<Level> levels;         // what is stored: one partition of the time axis per distinct window
};
struct Closed { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };   // one member's windows (seconds)

/** the anchors a method implies at this rate: {cycles, fMax} */
std::pair<double, double> anchorsFor(Method method, double sfreq, double target = 8);
/** the decimation for a band, a power of two leaving at least `min` samples a cycle at `hz` (never below 1) */
std::int64_t strideFor(double sfreq, double hz, int min = kMultirateMin);
/** a band's window in samples, as the ladder sizes it: `cycles` cycles of its LOW edge rounded UP to a power of two (the Studio's `pow2ceil(cycles·sfreq/lo)`);
 *  `cycles` defaults to the hertz anchor's (`anchorsFor(Hertz, sfreq)`: 6.83 at 1200 Hz) */
std::int64_t windowFor(double sfreq, double lo, std::optional<double> cycles = std::nullopt);
/** the tile level of a duration: the NEAREST power of two of samples, round(log2(seconds·sfreq)) (the rounding `anchorsFor` uses;: seconds snap
 *  to the nearest division); at least 0 */
int levelOf(double seconds, double sfreq);
/** the level of a power-of-two width (`log2`), or -1 when the width is not a power of two */
int levelOfWidth(std::int64_t samples);
/** the distinct window widths of a design, finest first */
std::vector<Level> levelsOf(const std::vector<Band>& bands, std::int64_t samples, double sfreq);
/** the tiling of a line of `samples` at `sfreq` Hz from `origin` seconds */
Tiling constantQTiling(double sfreq, std::int64_t samples, double origin, const Params& p = {});
/** member m of a closed form as [start, stop) seconds, the last clipped to `end` */
std::pair<double, double> tileOf(const Closed& t, std::int64_t m, double end);
/** the member holding `value`, or nothing */
std::optional<std::int64_t> tileIndexAt(const Closed& t, double value, double end);
/** how many tiles of a finer window one tile of a coarser holds (a whole multiple, else std::invalid_argument) */
std::int64_t tilesWithin(std::int64_t coarser, std::int64_t finer);
/** a level as its dictionary entry's attributes (the Studio's `levelInfos`: the level, duration_s, finest_window_samples), in the Studio's key order */
Json levelInfo(const Tiling& t, const Level& l);
/** the set's params_json and a member's params_json, in the Studio's key order (`constantQRows`) */
Json setParams(const Tiling& t);
Json memberParams(const Tiling& t, const Level& l);

/** what `write` made, or found */
struct Written {
  std::string set_id, name;
  std::vector<std::string> member_ids;
  bool existed = false;
  Tiling tiling;
};
/** WRITE the tiling of a time Line `manifold_id` (a closed-form `line`) as rows: the level dictionary, a tile dictionary per level, the ladder and its
 *  members (`writeConstantQ`). Idempotent: a line that has one returns it. Inside the caller's composition. */
Written write(SubjectWriter& sub, const std::string& manifold_id, const Params& p = {}, const std::optional<std::string>& by = std::nullopt);

}  // namespace constantq
}  // namespace nxr::io

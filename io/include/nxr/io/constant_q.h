#pragma once
// nxr::io — THE TILE GRID OF A TIME LINE: the closed-form arithmetic of a constant-Q ladder's levels as stored (D64/D87), a port of the Cortical Flow Studio's
// `backend/src/db/constant-q.ts` (`tileOf`, `tileIndexAt`, `tilesWithin`, and the level of a width). These are the format's own rules (as the chunk grid is): which
// samples a tile holds, which tile holds a time, how many finer tiles a coarser holds. No signal is computed here.
//
// io does NOT DESIGN a constant-Q tiling (the bands, their windows, the levels): that is compute's (`nxr::selection::constant_q` and `constant_q_bands`,
// compute/include/nxr/selection/partition.h). io is handed the designed levels and stores them: selection_store.h (`save` of a set with levels, stored as a `ladder`
// Selection on the time Line). The pipelines call compute's design and hand the result to io (pipelines/include/nxr/pipelines/constant_q.h).
//
// The band-design arithmetic (`anchorsFor`, `strideFor`, `windowFor`) is compute's: compute/include/nxr/selection/constant_q.h.
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace nxr::io {
namespace constantq {

struct Closed { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };   // one member's windows (seconds)

/** the tile level of a duration: the NEAREST power of two of samples, round(log2(seconds·sfreq)) (the rounding compute's `anchorsFor` uses; seconds snap to the nearest
 *  division); at least 0 */
int levelOf(double seconds, double sfreq);
/** the level of a power-of-two width (`log2`), or -1 when the width is not a power of two */
int levelOfWidth(std::int64_t samples);
/** member m of a closed form as [start, stop) seconds, the last clipped to `end` */
std::pair<double, double> tileOf(const Closed& t, std::int64_t m, double end);
/** the member holding `value`, or nothing */
std::optional<std::int64_t> tileIndexAt(const Closed& t, double value, double end);
/** how many tiles of a finer window one tile of a coarser holds (a whole multiple, else an Error) */
std::int64_t tilesWithin(std::int64_t coarser, std::int64_t finer);

}  // namespace constantq
}  // namespace nxr::io

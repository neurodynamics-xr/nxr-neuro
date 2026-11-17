// nxr::selection (compute) — COUPLED LEVELS AND GRID ADDRESSES: the arithmetic that ties a band to a grid's level, and an irregular member to the grid cell holding
// it (spec docs/specs/2026-11-17-selection-module.md, section 9). Pure functions of numbers (and of a grid already made); std::invalid_argument for an argument
// outside its domain.
//
// COUPLED LEVELS (replicating the Studio; PI/coordinator 2026-11-17). A single band SELECTS (it is not a frequency axis); it sets the level of the other factors' grids:
//   time      the Studio's `fitLevelOf` (engine/toolbox/wavelets/jointTime.ts): the band's support is `cycles` (4) cycles at the band's CENTRE (sqrt(lo hi)), in
//             samples need = ceil(cycles fs / centre); the level is the first dyadic level whose tile (2^k samples) is at least that (Rheome's `natural_level`),
//             held inside the line (a tile no longer than the line). The Studio counts a level by the tiles of the line (log2n - k); here a level is k, the tile's
//             power of two, as everywhere in nxr-neuro.
//   space     an eigenvalue band's dyadic level by Weyl's law: level l (2^l tiles a piece of area A) pairs with the band [4 pi 2^(l-1) / A, 4 pi 2^l / A) it adds
//             (the Studio's engine/ladder.ts); a band's level is the one whose Weyl band holds its CENTRE (as the time rule does).
// Then: product({time_grid(line, time_level_of_band(..)), level(tiles, space_level_of_band(..))}) is the band's grid on time x cortex.
//
// ADDRESSES (on request only: an irregular family is not indexed by a grid; the grid it was detected over is its provenance). An irregular member's address on a
// grid is (the level its EXTENT sets, the cell holding its CENTRE).
//   time      the Studio's `supportOf`: the tile holding the centre at the level the extent sets (`support_of`, `address_on_line`).
//   cortex    Rheome's `jointcell` depth rule (`cortical_level_of_extent`): the COARSEST depth whose tile still holds the feature (tile diameter <= TileSigma x the
//             feature's sigma), or a refusal that says why (no depth is fine enough).
#pragma once
#include <cstdint>
#include <optional>
#include <string>

#include "nxr/selection/selection.h"

namespace nxr::selection {

/** Rheome's natural_level: the first dyadic level k whose tile (2^k samples) is at least `support_samples` (0 for a support of at most one sample) */
int natural_level(double support_samples);
/** the Studio's `CYCLES`: the cycles of the band's centre a time tile must hold (jointTime.ts) */
constexpr double kJointCycles = 4;
/** the Studio's `fitLevelOf` as a level k of tiles of 2^k samples: natural_level(ceil(cycles fs / centre)), centre = sqrt(lo_hz hi_hz) (the band's centre),
 *  held at most `max_level` (a line of 2^max_level samples: the whole line is one tile). std::invalid_argument for a band that is not 0 < lo <= hi, fs <= 0. */
int time_level_of_band(double lo_hz, double hi_hz, double sfreq, double cycles = kJointCycles, std::optional<int> max_level = std::nullopt);
/** the dyadic level whose Weyl band [4 pi 2^(l-1) / A, 4 pi 2^l / A) holds eigenvalue `lambda`, on a connected piece of area `piece_area` (lambda in 1/unit^2,
 *  the area in unit^2): max(0, ceil(log2(A lambda / 4 pi))) */
int weyl_level(double lambda, double piece_area);
/** the dyadic level an eigenvalue band sets: weyl_level of its centre (sqrt(lo hi); hi / 2 when lo is 0), at most `depth` (a grid's finest) */
int space_level_of_band(double lambda_lo, double lambda_hi, double piece_area, int depth);

/** the Studio's `supportOf` result: the tile of a time grid holding a centre, at the level a support sets */
struct Support {
  int level = 0;                    // k: tiles of 2^k samples
  std::int64_t tile = 0;            // the tile's member number at that level
  double t0 = 0, t1 = 0;            // its bounds, seconds
  std::int64_t samples = 0;         // its width, 2^k
};
/** the Studio's `supportOf` on a line of 2^log2n samples at `sfreq`: the tile holding `tc` (seconds) at `level` k (default: the band's fit; never finer than the fit
 *  when `clamp`; with clamp false (a steady atom, no carrier to fit) down to tiles of 2^kFinestTileLog2 samples), clamped inside the line */
constexpr int kFinestTileLog2 = 4;
Support support_of(double lo_hz, double hi_hz, double tc, double sfreq, int log2n, std::optional<int> level = std::nullopt, bool clamp = true);

/** a member's address on a grid: the level (its label) and the member holding the centre at that level */
struct Address {
  int level = 0;
  std::int64_t member = 0;
};
/** the Studio's supportOf on a dyadic time grid: the level the EXTENT sets (natural_level(extent_samples), held inside [finest, coarsest]) and the tile holding the
 *  CENTRE sample (the sample nearest `center_sample`) at that level, clamped to the line's `samples` */
Address address_on_line(double center_sample, double extent_samples, std::int64_t samples, int finest, int coarsest);
/** Rheome's `jointcell` depth rule for a cortical feature: the COARSEST level l >= 1 whose tile (a piece of area `piece_area` cut in 2^l, taken as a disc of its
 *  area: diameter 2 sqrt(A / (pi 2^l))) is at most `tile_sigma` x `sigma` (the feature's width, in the surface's unit; a distance member's radius), at most `depth`.
 *  A REFUSAL IS AN ANSWER: when no level is fine enough, `level` is empty and `why` says so, as jointcell's does. */
struct CorticalLevel {
  std::optional<int> level;
  std::string why;                  // the decision in a sentence, or the refusal's reason
};
CorticalLevel cortical_level_of_extent(double sigma, double piece_area, int depth, double tile_sigma = 1.0);
/** an irregular member on a surface (centre vertex + extent radius, as a `distance`) addressed on a dyadic grid: the level its extent sets (jointcell's rule) and the
 *  tile holding its centre (the closed form: the grid's code >> (depth - level)). std::invalid_argument for a grid that is not dyadic, a centre outside it or in no
 *  tile, or a feature no level is fine enough for (the message is jointcell's reason). */
Address dyadic_address(const Selection& grid, std::int64_t center_vertex, double radius, double piece_area, double tile_sigma = 1.0);

}  // namespace nxr::selection

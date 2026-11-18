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
//   cortex    Rheome's `jointcell` depth rule (`cortical_level_of_extent`): the COARSEST depth whose tile still holds the feature (MEASURED tile diameter, the ladder's, <= TileSigma x the
//             feature's sigma), or a refusal that says why (no depth is fine enough).
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/selection/selection.h"

namespace nxr::select {

using namespace nxr::selection;   // the makers build the class side's Selection (selection/selection.h); it is declared there

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

// ── LEVELS RESOLVED FROM QUANTITIES (spec docs/specs/2026-11-17-consolidation.md, section 3.3) ──────────────────────────────────────────────────────────────────────
// A parameter is stated as a quantity (a duration, a tile diameter, an area, a wavelength, parts per piece); its level is resolved in closed form and the run records the stated
// quantity, the level and the SNAPPED quantity (what the level actually is). The cores above (`natural_level`, `time_level_of_band`, `weyl_level`, `cortical_level_of_extent`) stay.
/** a resolved level and what that level is, in `unit` (std::invalid_argument, with the reason, for a quantity no level can hold) */
struct LevelOf {
  int level = 0;
  double snapped = 0;
  std::string unit;
};
/** a time line's dyadic level from a duration (s): the first k with 2^k samples at least the duration (`natural_level`), held at most `max_level`; snapped = 2^k / sfreq s */
LevelOf level_of_duration(double seconds, double sfreq, std::optional<int> max_level = std::nullopt);
/** a surface's dyadic level from a tile diameter: the COARSEST level whose measured mean tile diameter (`tile_diameters[l]`, levels 0 .. depth, in the surface's length unit: the
 *  level dictionary's `diameter`) is at most `diameter` (rule 16, Rheome's `tileMM`); snapped = that level's diameter, in `unit`. A diameter no level is fine enough for is refused
 *  with the finest diameter (as `cortical_level_of_extent`). */
LevelOf level_of_diameter(const std::vector<double>& tile_diameters, double diameter, const std::string& unit);
/** the same from a tile area: the coarsest level whose mean tile area (`tile_areas[l]`, levels 0 .. depth, in unit^2) is at most `area`; snapped = that area, in unit^2 */
LevelOf level_of_area(const std::vector<double>& tile_areas, double area, const std::string& unit);
/** the Weyl level holding a wavelength (the surface's length unit): lambda = (2 pi / w)^2, `weyl_level(lambda, piece_area)`, at most `depth`; snapped = the shortest wavelength the level
 *  resolves (the upper edge of its Weyl band), in `unit` */
LevelOf level_of_wavelength(double wavelength, double piece_area, int depth, const std::string& unit);
/** the same from an eigenvalue (1/unit^2); snapped = the level's shortest wavelength, in `unit` */
LevelOf level_of_eigenvalue(double lambda, double piece_area, int depth, const std::string& unit);
/** parts per connected piece: level l = log2(parts), exact powers of two only (else refused, naming the nearest two); at most `depth`; snapped = 2^l parts (unit "parts") */
LevelOf level_of_parts(double parts, int depth);

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
/** Rheome's `jointcell` depth rule for a cortical feature: the COARSEST level l >= 1 whose MEASURED tile diameter (`tile_diameters[l]`, the surface ladder's `diameter`,
 *  in the surface's unit: the level dictionary of a dyadic grid, `Selection::diameter`) is at most `tile_sigma` x `sigma` (the feature's width, in the same unit; a distance
 *  member's radius). `tile_diameters` holds levels 0 .. depth. A REFUSAL IS AN ANSWER: when no level is fine enough, `level` is empty and `why` says so, as jointcell's does. */
struct CorticalLevel {
  std::optional<int> level;
  std::string why;                  // the decision in a sentence, or the refusal's reason
};
CorticalLevel cortical_level_of_extent(double sigma, const std::vector<double>& tile_diameters, double tile_sigma = 1.0);
/** an irregular member on a surface (centre vertex + extent radius, as a `distance`) addressed on a dyadic grid: the level its extent sets (jointcell's rule on the grid's
 *  measured level diameters) and the tile holding its centre (the closed form: the grid's code >> (depth - level)). std::invalid_argument for a grid that is not dyadic or whose
 *  levels carry no measured diameter, a centre outside it or in no tile, or a feature no level is fine enough for (the message is jointcell's reason). */
Address dyadic_address(const Selection& grid, std::int64_t center_vertex, double radius, double tile_sigma = 1.0);

}  // namespace nxr::select

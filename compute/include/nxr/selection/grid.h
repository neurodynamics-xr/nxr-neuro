// nxr::selection (compute) — COUPLED LEVELS AND GRID ADDRESSES: the arithmetic that ties a band to a grid's level, and an irregular member to the grid cell holding
// it (spec docs/specs/2026-11-17-selection-module.md, section 9). Pure functions of numbers (and of a grid already made); std::invalid_argument for an argument
// outside its domain.
//
// COUPLED LEVELS. A single band SELECTS (it is not a frequency axis); it sets the level of the other factors' grids:
//   time      a frequency band's NATURAL LEVEL: the first dyadic level whose tile is at least the band's support (Rheome's `natural_level`, +ingest/bank.m:
//             max(0, ceil(log2(support / tile0))); the Studio's `fitLevelOf`, engine/toolbox/wavelets/jointTime.ts). The support is the constant-Q window rule
//             (`constantq::windowFor`: the cycles of the band's low edge), so the level is the one the constant-Q ladder gives that band.
//   space     an eigenvalue band's dyadic level by Weyl's law: level l (2^l tiles a piece of area A) pairs with the band [4 pi 2^(l-1) / A, 4 pi 2^l / A) it adds
//             (the Studio's engine/ladder.ts); a band's level is the one whose Weyl band holds its low edge.
// Then: product({time_grid(line, time_level_of_band(..)), level(tiles, space_level_of_band(..))}) is the band's grid on time x cortex.
//
// ADDRESSES (on request only: an irregular family is not indexed by a grid; the grid it was detected over is its provenance). An irregular member's address on a
// grid is (the level its EXTENT sets, the cell holding its CENTRE): the Studio's `supportOf` (the tile holding t_c at the fitted level).
#pragma once
#include <cstdint>
#include <optional>

#include "nxr/selection/selection.h"

namespace nxr::selection {

/** Rheome's natural_level: the first dyadic level k whose tile (2^k samples) is at least `support_samples` (0 for a support of at most one sample) */
int natural_level(double support_samples);
/** the time level a frequency band [lo_hz, ..) sets on a line of `sfreq` Hz: natural_level of its constant-Q window (cycles of its low edge; `cycles` defaults to
 *  the hertz anchor's, constantq::windowFor) */
int time_level_of_band(double lo_hz, double sfreq, std::optional<double> cycles = std::nullopt);
/** the dyadic level whose Weyl band [4 pi 2^(l-1) / A, 4 pi 2^l / A) holds eigenvalue `lambda`, on a connected piece of area `piece_area` (lambda in 1/unit^2,
 *  the area in unit^2): max(0, ceil(log2(A lambda / 4 pi))) */
int weyl_level(double lambda, double piece_area);
/** the dyadic level an eigenvalue band [lambda_lo, ..) sets: weyl_level of its low edge, at most `depth` (a grid's finest) */
int space_level_of_band(double lambda_lo, double piece_area, int depth);

/** a member's address on a grid: the level (its label) and the member holding the centre at that level */
struct Address {
  int level = 0;
  std::int64_t member = 0;
};
/** the Studio's supportOf on a dyadic time grid: the level the EXTENT sets (natural_level(extent_samples), held inside [finest, coarsest]) and the tile holding the
 *  CENTRE sample (the sample nearest `center_sample`) at that level, clamped to the line's `samples` */
Address address_on_line(double center_sample, double extent_samples, std::int64_t samples, int finest, int coarsest);
/** the finest dyadic level whose tiles (pieces of area `piece_area` cut in 2^l, each taken as a disc of its area) are at least a disc of radius `radius`, inside
 *  [0, depth] (a radius 0: depth) */
int dyadic_level_of_extent(double radius, double piece_area, int depth);
/** an irregular member on a surface (centre vertex + extent radius, as a `distance`) addressed on a dyadic grid: the level its extent sets and the tile holding its
 *  centre (the closed form: the grid's code >> (depth - level)). std::invalid_argument for a grid that is not dyadic, a centre outside it or in no tile. */
Address dyadic_address(const Selection& grid, std::int64_t center_vertex, double radius, double piece_area);

}  // namespace nxr::selection

// nxr::measurement (compute) — `roll_up`: through a grid's closed-form hierarchy, or across a group's subjects (spec docs/specs/2026-11-17-consolidation.md, section 2.3). The ONE
// implementation of a roll-up: partition-style merges, the tower's rollTime, io's rollup::space and rollup::time, pyramid reduce, an event's tiles merged into the event.
// Roll-ups exist only on GRIDS (a closed-form hierarchy: a dyadic shift); an irregular selection's members never roll up (the checker's `rolling_up()`, `rollup_refusal`).
//
// WHERE IT RUNS (PI, 2026-11-17, rule 35): roll-ups live in io's QUERY (in SQL) AND here (in memory). To keep ONE definition each measure's merge rule (sum, max, min, mean as the
// sum of weighted values over the sum of weights, none: never rolled up) is DATA in the measures registry (measures.h), applied by io in SQL and by `roll_values` below in memory,
// with a parity test between the two. A roll-up of INGREDIENTS (`merge`) adds, takes max and min: every measure is then read from the merged ingredients, ratios derived at read.
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/measurement/measures.h"
#include "nxr/measurement/reduce.h"
#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"

namespace nxr::measurement {

constexpr int kWhole = -1;       // the level of the whole: one member

// ── the groups of a coarser level, in the finer level's member numbers (closed form) ─────────────────────
/** a GRID's coarser level as groups of the finer level's members: coarse member j holds the fine members `m` with m >> shift == j. A dyadic surface grid (levels counted from the pieces:
 *  level l has slots << l members; coarser = fewer): from_level a, to_level b < a (b = kWhole: one group of all); a dyadic TIME grid (a tiling of tiles of 2^k samples, level k;
 *  coarser = larger k): from_level k_a, to_level k_b >= k_a (kWhole: one group), tiles counted from the line's first tile (no padding: the grid starts at the line's first sample).
 *  std::invalid_argument, with the reason, for an irregular selection (rollup_refusal) or a grid that does not roll up by shifts. */
selection::Elements coarser_groups(const selection::Selection& grid, int from_level, int to_level);
/** one FACTOR of a product (the first factor is 0): the product's members at the finer level as groups of the coarser product's member numbers (codes repacked with the rolled radices):
 *  group j holds the fine product codes that map to coarse code j (unpack, shift, repack: `coarser_code`). */
selection::Elements coarser_groups(const selection::Selection& product, std::size_t factor, int from_level, int to_level);

// ── rolling up ─────────────────────────────────────────────────────────────────────────────
/** (`merge` is declared with `reduce`, reduce.h: groups of members merged by the ingredients' rules.) `roll_up` is it at a level change. */
Ingredients roll_up(const Ingredients& fine, const selection::Elements& groups, const Request&);   // = merge
/** the same members in several subjects: ingredients summed code for code (a stack of subjects is one more factor, rolled to one member). std::invalid_argument for ingredients of
 *  different shapes or a median. */
Ingredients across(const std::vector<const Ingredients*>& subjects, const Request&);
// ── rolling up STORED VALUES by the registry's rule (the in-memory side of io's SQL roll-up; the parity test compares the two) ──────────
/** `values` [rows x fine members] of `measure` rolled up over `groups` by the measure's merge rule (measures.h, DATA): sum adds, max and min merge, mean is the sum of value x weight over
 *  the sum of weight with `weights` [rows x fine members, or 1 x fine members shared by every row] the values of the measure's `weight` measure; none (a median, a derived measure)
 *  is refused. A NaN in a group makes it NaN (a value not whole and good merges to NaN). An empty group: sum 0, max -inf, min +inf, mean NaN. */
Eigen::MatrixXd roll_values(const Registry& registry, const std::string& measure, const Eigen::MatrixXd& values, const Eigen::MatrixXd* weights, const selection::Elements& groups);

}  // namespace nxr::measurement

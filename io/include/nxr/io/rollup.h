// nxr-io — ROLL-UP IS READ-TIME ARITHMETIC (D132; nxr `measure/rollup.py`, the Studio's `db/rollup.ts` rollSpace / rollTime / mergeOf): the measurement pyramid never stores a level it can compute.
//   SPACE  the leaves are the codes h·2^D + t; a node of level ℓ ≤ D is `code >> (D − ℓ)`, so a node's leaves are ONE CONTIGUOUS BLOCK of 2^(D−ℓ) and a roll-up is a reshape and a reduction; ℓ = −1 the whole cortex (one node).
//   MEMBERS the same closed form for any grid whose coarser member is `code >> shift` (a line's time grid from its first sample; a product's factor in mixed radix; the same member of several subjects): `members`,
//          `product_factor`, `across`. THE ONE QUERY (measurement_query.h) does these in SQL for scalar rows and calls these for arrays.
//   TIME   (retired for new grids) frames are cycles of the day-anchored tower (frame k = tower code code0 + k); a level-m tile is `code >> m` = 2^m consecutive frames ALIGNED ON THE TOWER; kept to read what was made on it.
// Only MERGEABLE statistics roll up, by the measure's `merge` rule: DATA in schema/measures.json (measures.h `MergeRule`), shared with compute (roll-ups live in io's query as well as in compute; one definition,
// a parity test between the two): sum, max, min (neutral 0, −∞, +∞), a mean through its sums and weights. Ratios and variances are derived from rolled-up ingredients, never rolled up themselves (the callers' job).
// Pure functions of arrays, no file or row. They are the SAME arithmetic as compute's `nxr::measurement::roll_up` (io may not depend on compute and compute may not depend on io, so the two exist side by side, and a
// pipelines test compares them on the same arrays); nxr's own `rollup.space` / `rollup.time` is their golden (io/test/golden/gen_rollup_golden.py).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/io/measures.h"

namespace nxr::io::rollup {

/** The merge of a roll-up (measures.h `MergeRule`, the registry's data): Sum, Max, Min are folded here; Mean is folded as a SUM (the weighted numerator: the caller divides by the rolled-up weights, a mean through
 *  sums and a count, never a mean of means); None is never folded (a median or a member's own value: the folds refuse it, naming it). */
enum class Merge { Sum, Max, Min, Mean, None };
/** How a stored measure merges: the registry's `merge` rule (measures.h `measure_merge`: the ONE definition io's SQL roll-ups and compute's share). A name the registry does not know has no rule
 *  here (Error(unsupported)): a subject's declaration gives it, read with `measure_merge(db, name)` and passed to the second form. A derived measure is not merged (its ingredients are): Error(invalid_argument). */
Merge merge_of(const std::string& measure);
Merge merge_of(MergeRule rule, const std::string& measure);
const char* merge_name(Merge m);
/** x [rows x leaves] (row-major) with leaves = pieces · 2^depth → [rows x nodes] at `level` (−1: one node; ≤ depth), nodes = pieces · 2^level; `nodes` receives the count. Throws std::invalid_argument. */
std::vector<double> space(const std::vector<double>& x, std::int64_t rows, std::int64_t leaves, int depth, int level, Merge op, std::int64_t* nodes = nullptr);
/** x [rows x frames] → [rows x tiles] of 2^m frames aligned on the tower, frame 0 being the code `code0` (m ≥ 0): tile i is the code (code0 >> m) + i. A NaN merges to NaN. Throws std::invalid_argument.
 *  The day-anchored tower is RETIRED for new grids (the consolidation spec, section 1.2): this reads the frame tilings made on it (`retired_grid: tower`); a line's time grid is `members` below. */
std::vector<double> time(const std::vector<double>& x, std::int64_t rows, std::int64_t frames, int m, Merge op, std::int64_t code0, std::int64_t* tiles = nullptr);
/** THE CLOSED-FORM HIERARCHY of any grid whose member number at the coarser level is `code >> shift` (a dyadic surface grid; a line's time grid from its FIRST sample, no pad): x [rows x members] → [rows x
 *  ceil(members / 2^shift)] by `op`; shift 0 is the identity, a shift of 63 or more one member. A NaN merges to NaN under Max and Min. Throws std::invalid_argument (a negative shift; None). */
std::vector<double> members(const std::vector<double>& x, std::int64_t rows, std::int64_t members, int shift, Merge op, std::int64_t* groups = nullptr);
/** ONE FACTOR of a product rolled up: x [rows x product members] with the radices `radices` (mixed radix, the FIRST factor fastest), factor `factor`'s member number `>> shift`: → [rows x members of the rolled
 *  product] (that factor's radix ceil(radix / 2^shift)); the same arithmetic as io's `selection::coarser_product_code` and compute's `coarser_code`. `radices_out` receives the new radices. Throws std::invalid_argument. */
std::vector<double> product_factor(const std::vector<double>& x, std::int64_t rows, const std::vector<std::int64_t>& radices, std::size_t factor, int shift, Merge op, std::vector<std::int64_t>* radices_out = nullptr);
/** The same member of several subjects (a group's stack on the group tree's codes): `subjects` each [rows x members] of one shape, merged element by element by `op`; Mean is a sum (divide by the merged weights).
 *  A NaN in a subject is skipped under Sum and Mean (an absent member adds nothing) and propagates under Max and Min as `members`. Throws std::invalid_argument (shapes differ; None). */
std::vector<double> across(const std::vector<const std::vector<double>*>& subjects, Merge op);

}  // namespace nxr::io::rollup

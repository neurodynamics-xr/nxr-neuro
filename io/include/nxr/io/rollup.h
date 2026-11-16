// nxr-io — ROLL-UP IS READ-TIME ARITHMETIC (D132; nxr `measure/rollup.py`, the Studio's `db/rollup.ts` rollSpace / rollTime / mergeOf): the measurement pyramid never stores a level it can compute.
//   SPACE  the leaves are the codes h·2^D + t; a node of level ℓ ≤ D is `code >> (D − ℓ)`, so a node's leaves are ONE CONTIGUOUS BLOCK of 2^(D−ℓ) and a roll-up is a reshape and a reduction; ℓ = −1 the whole cortex (one node).
//   TIME   frames are cycles of the day-anchored tower (frame k = tower code code0 + k); a level-m tile is `code >> m` = 2^m consecutive frames ALIGNED ON THE TOWER (a neutral front pad of code0 mod 2^m frames, a neutral tail).
// Only MERGEABLE statistics roll up: sum, max, min (neutral 0, −∞, +∞). Means, variances and densities are derived from rolled-up sums, never rolled up themselves (the callers' job).
// Pure functions of arrays, no file or row. They are the SAME arithmetic as nxr::tower::rollTime (compute): io may not depend on compute and compute may not depend on io (the dependency table), so the two exist
// side by side and a pipelines test (test_pyramid_pipeline) compares them on the same arrays; nxr's own `rollup.space` / `rollup.time` is their golden (io/test/golden/gen_rollup_golden.py).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::io::rollup {

enum class Merge { Sum, Max, Min };
/** How a stored measure merges (nxr `mergeOf`): max if its name is `max` or ends in `_max`, min if it is `min`, else sum. */
Merge merge_of(const std::string& measure);
const char* merge_name(Merge m);
/** x [rows x leaves] (row-major) with leaves = pieces · 2^depth → [rows x nodes] at `level` (−1: one node; ≤ depth), nodes = pieces · 2^level; `nodes` receives the count. Throws std::invalid_argument. */
std::vector<double> space(const std::vector<double>& x, std::int64_t rows, std::int64_t leaves, int depth, int level, Merge op, std::int64_t* nodes = nullptr);
/** x [rows x frames] → [rows x tiles] of 2^m frames aligned on the tower, frame 0 being the code `code0` (m ≥ 0): tile i is the code (code0 >> m) + i. A NaN merges to NaN. Throws std::invalid_argument. */
std::vector<double> time(const std::vector<double>& x, std::int64_t rows, std::int64_t frames, int m, Merge op, std::int64_t code0, std::int64_t* tiles = nullptr);

}  // namespace nxr::io::rollup

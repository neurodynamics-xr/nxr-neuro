// nxr-pipelines — THE LAG-FREE NULL'S TESTS (`null compute`; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10): runs the draws of `vortex_null` (the vortex chain on a lag-free surrogate of the recording, one graph run per
// seed, `seed: i`) in BATCHES with EARLY STOPPING (Besag and Clifford 1991), reads each draw's stored tile crossings (small, a statistic: the surrogate itself is never stored) and tests the data's crossings against them. Orchestration only: the graph runs
// through the runner, the draws are read through io, the maths are nxr::crossings and nxr::surrogate (pure functions of arrays).
//
// THE TESTS (per tile of the level the crossings were measured over, the two senses pooled, the crossings of the stage asked for; a tile counts with at least `min_crossings` crossings in the data):
//   consistency   the unbiased rho2 = (|S|^2 - n) / (n (n - 1)) of the tile's pooled sum S of n crossings (never R: places with different counts are compared), one-sided: the draws at least as large as the data's
//   repeat        with two recordings: the dot product of the two runs' mean resultants (nxr `tile_dot`), draw i of one run against draw i of the other, and the count-weighted mean cosine over the tiles (nxr `repeat`)
// The sequential p of a tile is h / n when h exceedances were found after n draws, else (exceedances + 1) / (n + 1); the draws stop when every tile is decided (h exceedances) after at least `min_draws`, or at `max_draws`. Benjamini-Hochberg over the
// tiles. A null tile with fewer than `min_crossings` crossings (the repeat) or fewer than 2 (the consistency) has no statistic and is not an exceedance: the crossing ratio below says where that happens.
//
// STORED (statistics of the first recording's Field, over the level's Selection; one set a recording, the repeat's with the pair):
//   vortex_cores.null.<rec>.consistency_p   [tiles x (p, q, exceedances)]  measure null_p, method besag_clifford
//   vortex_cores.null.<rec>.crossing_ratio  [tiles]                         the data's crossings over the null's mean (the data-to-null crossing ratio), measure crossing_ratio
//   vortex_cores.null.<rec>.resultant       [tiles x (x, y, z, mean crossings)]  the null's mean resultant per tile (the group subtraction), measure null_resultant
//   vortex_cores.run_repeat.<a>.<b>.p       [tiles x (p, q, exceedances)]  and .whole_p [(p, exceedances, draws)] for the count-weighted cosine
// SEEDS: the white noise is a function of the seed, so the two runs' draws are made independent: run A's draw i is seed i, run B's seed i + 100000 (a draw is the node `vortex_null.draw<seed>.null_crossings.<recording>`).
// WHAT IT CANNOT SAY: the null has the data's zero-lag covariance, spectrum and envelope and nothing else: a tile beyond it is beyond what mixing and amplitude modulation produce, not a proof of a wave. The threshold and the burst frames are the data's;
// only the signal is replaced. Draws that exist are reused (a draw is its own cached node), so a rerun with a larger `max_draws` continues where the last stopped.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct NullOptions {
  std::string recording_a;                   // REQUIRED: a recording whose `tile_crossings` (`run vortex_cores`) are tested
  std::optional<std::string> recording_b;    // a second recording: the run-to-run repeat's test as well (draws in step)
  std::optional<std::string> over;           // the tile level's Selection (`<surface> tiles L<l>`) the crossings were measured over; default: the one the recordings have
  int stage = 0;                             // the stage by index (rows 2s, 2s+1 of the stored crossings) ...
  std::optional<std::string> stage_name;     // ... or by name (burst, onset, within, offset): each run's own rows name them
  nxr::io::Json set = nxr::io::Json::object();   // the `vortex_null` parameters of every draw (`{recording}` stands for each recording's name): space_level, cross_spectrum, band, transform, bursts, ...
  std::optional<std::string> chunks = std::string("all");   // what each draw reads
  int batch = 20, min_draws = 20, max_draws = 100;   // the draws are made in batches of `batch`; at least `min_draws`, at most `max_draws` (the project's 20, 20, 100)
  int h = 10;                                // Besag-Clifford: a tile is decided at h exceedances
  int min_crossings = 10;                    // a tile counts with at least this many crossings (nxr NMIN_TILE)
  bool replace = false;                      // replace the stored results, and any draw stored with other parameters (moved aside, never deleted); a draw that is the same is a hit and kept
};
struct NullResult { nxr::io::Json info; bool hit = false; };
NullResult null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const NullOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

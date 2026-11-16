#pragma once
// nxr::scale — SCALE SELECTION for events seen at several (spatial level, time level) pairs (spec docs/specs/2026-11-13-cortical-events.md, phase 2e). PURE MATHS on matrices: a pair's cells are a standardized
// amplitude [S × T] (z, NaN: no value) and a labelling [S × T] (an event's ordinal, −1 none); the functions never know where they came from (a stored measurement, a synthetic array), never read a file, never loop
// over samples or chunks, and never pick their inputs. The reference is Rheome's `detect.blobscale` (the scale of the largest normalised response, read AT the blob; a selection on the ladder's edge is unresolved).
//
//   nests / overlaps    the ladders' exact nesting: tile s of level l has the ancestor s >> (l − l') at the coarser level l' (tile t of a level has children 2t, 2t+1); dyadic time windows nest likewise, the test is the
//                       sample intervals' intersection (a window of W samples starting at the pair's origin)
//   eventFacts          per event of a pair: its cells, its largest and its mean z, its first and last sample
//   selectScale         events of DIFFERENT pairs are LINKED when ≥ minLink pairs of their cells nest and overlap; the connected components of the links are the cross-pair events; each one's response at every pair
//                       (the footprint), its best pair (the largest response; a tie goes to the first pair), the response there, how many pairs hold a member, its span in samples
// The reductions are the labelling's members' max and mean (the event is the partition); nothing is reduced by the chain.
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "nxr/extent.h"

namespace nxr::scale {

using Labels = nxr::extent::Labels;

/** one (spatial level, time level) pair's cells */
struct Pair {
  int level = 0;                    // the spatial level l of the cells' tiles (a tile's code is h·2^l + t)
  std::int64_t window = 1;          // samples a time tile holds (> 0)
  std::int64_t origin = 0;          // the sample the first time tile starts at
  Eigen::MatrixXd z;                // [S × T] standardized cells (NaN: no value)
  Labels label;                     // [S × T] the event's ordinal within the pair (0 … events − 1) or −1
  int events = 0;
};
enum class Response { Peak, Mean };
/** "peak" | "mean"; std::invalid_argument otherwise, naming both */
Response parseResponse(const std::string& name);
const char* responseName(Response r);

/** whether tile s of level a and tile t of level b are the same place or one contains the other (the finer one's tile shifted right by the level difference is the coarser one's). std::invalid_argument for a negative tile or level. */
bool nests(std::int64_t s, int levelA, std::int64_t t, int levelB);
/** whether [originA + f·windowA, +windowA) and [originB + g·windowB, +windowB) intersect */
bool overlaps(std::int64_t f, std::int64_t windowA, std::int64_t originA, std::int64_t g, std::int64_t windowB, std::int64_t originB);

struct EventFacts {
  int cells = 0;
  double peak = 0, mean = 0;        // the largest and the mean z over the event's cells (NaN cells do not count; NaN when none has a value)
  std::int64_t first = 0, stop = 0; // samples: the first time tile's start and the last time tile's end (stop exclusive)
};
/** the facts of each event of the pair. std::invalid_argument for z and label of different shapes, a label ≥ events, events < 0, or window < 1. */
std::vector<EventFacts> eventFacts(const Pair& p);

struct Selection {
  int events = 0;                                   // the cross-pair events
  std::vector<std::vector<int>> cross;              // [pair][event of the pair] → the cross-pair event
  Eigen::MatrixXd response;                         // [pairs × events]: the largest response of the cross-pair event's members at that pair, NaN where it has none
  Eigen::VectorXi best;                             // [events] the pair of the largest response (the first on a tie)
  Eigen::VectorXd peak;                             // [events] the response there
  Eigen::VectorXi pairs;                            // [events] the number of pairs with a member
  Eigen::VectorXd start, stop;                      // [events] the span in samples: the earliest first sample, the latest stop
  std::vector<std::vector<std::pair<int, int>>> members;   // [events] the (pair, event of the pair) members, ascending
};
/** `minLink` ≥ 1. std::invalid_argument for fewer than one pair, the facts' refusals, or minLink < 1. */
Selection selectScale(const std::vector<Pair>& pairs, Response response = Response::Peak, int minLink = 1);

}  // namespace nxr::scale

#pragma once
// nxr::select::scale_selection — SCALE SELECTION for events seen at several (spatial level, time level) pairs (spec docs/specs/2026-11-13-cortical-events.md, phase 2e). PURE MATHS on matrices: a pair's cells are a standardized
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

#include "nxr/select/extent.h"

namespace nxr::select::scale_selection {

using Labels = nxr::select::Labels;

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

}  // namespace nxr::select::scale_selection

#pragma once
// nxr::select::scale_selection — SCALE SELECTION at a point: the hat level whose scale-normalised response is the largest (Lindeberg 1998, "Feature detection with automatic scale selection"; spec
// docs/specs/2026-11-14-spatial-events.md, decision 4). PURE MATHS on matrices: the power of a field's hat response at several levels, the squared L2 norm of each level's atom, numbers in, the chosen level out. The
// function never knows where the matrices came from (a recording's band power through a kernel, a planted blob), which vertices or frames they hold, or what a level is in centimetres.
//
//   the response   at an element e (a vertex) and a level l, the hat operator's response is r_l(e) = <atom_l(e), f> with atom_l(e) the operator's column at e. Its SQUARE is the power P_l(e) = |r_l(e)|² (the caller sums
//                  a free current's components, as `power` does). Normalised to the UNIT-ENERGY atom, the correlation of f with it is r_l(e) / ‖atom_l(e)‖, so the normalised power is P_l(e) / n_l(e) with
//                  n_l(e) = ‖atom_l(e)‖² the squared L2 norm the caller gives. A blob of the atom's own size gives its largest normalised power at that level, and white noise the same at every level.
//   the choice     the level l* of the largest normalised power at each element and frame (the lowest index on a tie); its value is returned beside it.
// Layout: rows are element-major, level-minor (row = e · levels + l), as `apply_kernel`'s multi-level output and `power` of it are. Pointwise in the frames (columns) and the elements: nothing is reduced across time or space;
// the maximum is over the level axis, the scale space.

namespace nxr::select::scale_selection {

struct LevelChoice {
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> index;   // [elements × frames] the chosen level's position in the list (0 … levels − 1)
  Eigen::MatrixXd response;                                   // [elements × frames] the largest normalised power P_l / n_l
};

/** `power` [elements · levels × frames] (≥ 0, finite), `norm2` [elements · levels] (> 0, finite: the squared L2 norm of each element's atom at each level), `levels` ≥ 1.
 *  std::invalid_argument for sizes that disagree, a norm that is not finite and above 0, or a power that is negative or not finite. */
LevelChoice selectLevel(const Eigen::Ref<const Eigen::MatrixXd>& power, const Eigen::Ref<const Eigen::VectorXd>& norm2, int levels);

}  // namespace nxr::select::scale_selection

// nxr-graph — RUNNING A CHECKED GRAPH. The spec: docs/specs/2026-11-17-compute-and-run.md, section 3.4.
//
// The execution is the function (block_k, S_k) -> (output_k, S_k+1), S the states of the stateful and accumulate nodes; `Execution` holds S between calls so a
// caller need not. A block's values never depend on how the input was cut into blocks (tested).
//
// THE EXECUTOR DOES NOT LOOP OVER A RECORDING. It is handed blocks one at a time: in `run`, io's chunk service calls it once per chunk (spec step C5); in `compute`
// a caller hands whole arrays (`apply`) or streams blocks it read itself. Inside ONE block it cuts the pointwise tail into sub-blocks (`Blocking::samples`, and
// `frames` after a retiming node), as the runner does: a memory bound, never a result.
//
// Every node call goes through `dispatch` (dispatch.h). The executor knows nothing of files, subjects or storage.
#pragma once
#include <vector>

#include "nxr/graph/graph.h"

namespace nxr::graph {

/** the widths (columns) of the sub-blocks the pointwise tail runs on; 0: the whole block. The result does not depend on them (to the rounding of the folds' sums); only memory does. */
struct Blocking {
  std::int64_t samples = 1024;
  std::int64_t frames = 256;
};

class Execution {
 public:
  /** prepares every node and runs `before`. `constants` holds the graph's inputs that are not streamed (a kernel, a surface); `checked` must outlive the execution. */
  Execution(const Checked& checked, const Arrays& constants, const DispatchContext& ctx = {}, const Blocking& blocking = {});
  /** what each block handed must carry on each side */
  std::int64_t halo() const { return checked_.halo; }
  /** ONE BLOCK: the streamed inputs on [first - left, first + n + right), `at` the own columns' place. Runs the chunk nodes and steps the stateful and accumulate
   *  nodes. Blocks must come in time order when the graph has state (refused otherwise). Returns the streamed outputs' own columns (no halo). */
  Blocks block(const Blocks& in, const Place& at);
  /** finishes the accumulate nodes, runs `after`, and gives the array outputs. No block may follow. */
  Arrays finish();
  /** every node's resolved parameters and what its preparation resolved, for the provenance */
  Json record() const;

 private:
  const Checked& checked_;
  DispatchContext ctx_;
  Blocking blocking_;
  Shared shared_;
  Arrays data_;                       // the values outside the loop: the constants, and what the array nodes and finished accumulate nodes made
  std::vector<std::any> prepared_, states_;
  std::optional<std::int64_t> next_;  // the first column the next block must start at, when the graph has state
  bool finished_ = false;
  void run_arrays(std::size_t node_index);
};

/** whole arrays handed as ONE block (no halo: the nodes treat the ends as at a recording's ends): the streamed outputs and the array outputs, as arrays. The same as `block` then `finish`.
 *  A streamed array is [rows x samples x comps]; a complex one has comps 2 (re, im). */
Arrays apply(const Checked& checked, const Arrays& inputs, const DispatchContext& ctx = {}, const Blocking& blocking = {});

}  // namespace nxr::graph

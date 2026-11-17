// nxr-pipelines — THE GROUP SCOPE of the measurement pyramid.
//
// THE GROUP TREE PER SUBJECT (`group_tiles_compute`; `pyramid build --tree group` makes it when absent). Every subject's vertices take the code of the nearest vertex of the default subject's
// primary cortex on the REGISTERED SPHERE (`<surface>_sphere` of both), hemisphere (piece) by piece, in the default subject's group ladder (`<surface> tiles`, `tree: "group"`): so the leaves of the
// group tree at depth D are the SAME 2 x 2^D tiles in every subject, and what is measured on them adds cell for cell across subjects. Stored as nxr stores it:
//   the set `<surface>_group_tiles` (params_json: tree group, template, finest, the sphere match) with one member per level l = 0 .. finest, `<surface> group tiles L<l>` (a `set`, member_ordinal l, tile code
//   h 2^l + t of every vertex, the default subject's level dictionary), each with its node `<name>_atlas/{codes, w}` (w the leaf areas on THIS subject's surface) in its params_json {tree group, level, finest, node, areas}.
//   The partition the pyramid measures on gains nxr's `atlas` block (tree, level, finest, gauge, areas, node), as any leaf partition does.
//
// THE GROUP REDUCE (`pyramid_reduce`; `nxr-neuro pyramid reduce`). The subjects the CALLER NAMES (a cohort; never all by default) are summed, cell for cell, into the DEFAULT SUBJECT'S store (D131):
//   members read through their rows only (no path guessed): the measurements on their `group tiles L<D>` partition (spatial maps, by the field's KIND: a name carries a per-subject stamp, a kind does not)
//   and the joint Selections on it (source estimates, by task x kernel method x orientation, one kernel a member recording chosen by `kernels`; two of one method are refused, never pooled).
//   Every mergeable array adds across subjects (sums add, maxima take the maximum, minima the minimum: the measure's own rule); TIME IS COLLAPSED (subjects share no clock): each member's frames are summed
//   (maxima: maxed) first, as are its runs. The per-subject means and densities are kept (statistics need them), the members are the group fields' CONTRIBUTIONS (`field_contribution`).
//   Stored in the node of the default subject's tile partition `<surface> tiles L<D>` (nxr's `atlas_node`):
//     <node>/maps/<kind>/{w,s1,s2,n,min,max}, grad/{...}              sum over members                          rows: kind `group map`, pathless, function `sum`, the measures of the arrays
//     <node>/maps/<kind>/per_subject/mean [subjects, leaves]
//     <node>/time/<task>__<method>__<orientation>/{power|env|envmax|tensor/...} [bands, leaves]   sum / max over members, runs, frames     rows: kind `source estimate`, pathless, `function=sum`,
//     <node>/time/<key>/samples [1]                                                                the good samples summed                   within = the default subject's `atlas bands`, code = the band
//     <node>/time/<key>/per_subject/density [subjects, bands, leaves]   power (tensor total) per m^2 per sample, of each member
//   A re-run REPLACES the group fields and their arrays. REFUSED (as nxr, and further): a member with no group-tree partition at depth D, leaves that are not the default subject's (count, depth), another bank of
//   bands, a measure the reduce has no merge rule for is left out and named, two kernels of one method for a recording.
//
// MEMORY (stated): a member's array is read one BAND at a time ([leaves x frames] float64) and collapsed at once; held are the accumulators (the group arrays: [bands, leaves] per measure, [leaves] per map array)
// and the per-subject stacks (subjects x bands x leaves float64). Never a whole recording's frames of every band at once.
//
// CANNOT BE RESOLVED / NOT DONE: a member of ANOTHER dataset (nxr's `sources`) is not read; no group statistic (the group map G, nulls: the dataset-wide spec, section 3) is computed here, only the sums they start from.
#pragma once
#include <functional>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/pyramid.h"

namespace nxr::pipelines {

struct GroupTilesOptions {
  std::optional<std::string> surface;      // the subject's surface (default: the primary one)
  bool replace = false;                    // an existing group tree on the surface is removed and rewritten
};
struct GroupTilesResult {
  nxr::io::Json info;
  bool existed = false;
};
// Not for the default subject (its group tree IS its ladder: `dataset create`, `tiles compute`).
GroupTilesResult group_tiles_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const GroupTilesOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// The group leaves of a subject as `pyramid_leaves` returns them (the partition `<surface> group tiles L<D>`; the group tiles are made when absent). D defaults to `group_default_depth`.
PyramidLeaves group_leaves(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurfaceInput& surface, std::optional<int> depth, const std::string& who, const std::function<void(const nxr::io::Json&)>& say = {});

// The leaf depth the dataset's atlases share: the `depth` of the default subject's definitions (nxr's --depth, default 8).
int group_default_depth(nxr::io::DatasetWriter& ds);

struct PyramidReduceOptions {
  std::vector<std::string> subjects;       // REQUIRED: the members, by name (a cohort; never all by default)
  std::string kernels = "all";             // which members' source estimates are summed: comma tokens, kinds AND, tokens of one kind OR: all | constrained | free | latest | a kernel's method, stamp, name or glob
  std::optional<int> depth;                // the leaves' level D (default: the definitions' atlas depth)
};
struct PyramidReduceResult {
  nxr::io::Json info;                      // {members, spatial, time, depth, node, fields}
};
PyramidReduceResult pyramid_reduce(nxr::io::DatasetWriter& ds, const PyramidReduceOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// The same reduction on a CATALOGUE dataset (spec section 5): the members (`subjects`, by name) are read each from its own package, and the result is written into the GROUP
// PACKAGE `group_name` of the dataset (made if absent, else its group fields are replaced), whose rows reach the default subject's and the members' through anchor rows. One
// group package per analysis the caller names. The sums, merge rules, refusals and stored rows are `pyramid_reduce`'s; members registered to different template versions are
// refused. The default subject is the one the members' packages name, found through the catalogue and the template search path.
PyramidReduceResult pyramid_reduce_group(const std::filesystem::path& dataset, const PyramidReduceOptions& opt, const std::string& group_name, const nxr::io::TemplateSearch& templates,
                                         const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

// nxr-pipelines — THE MEASUREMENT PYRAMID: what is left of its hand-wired C++ (the rest is the definition `pyramid` of registered nodes, docs/specs/2026-11-17-pyramid-grid.md).
//
// The pyramid is a GRID on the product manifold cortex x time: dyadic spatial tiles (the leaves of a surface's tile ladder) x time tiles (the frames: cycles of one tower level), Studio D129's joint
// partition; each stored number is a measurement over a member of that grid, and the coarser levels are its closed-form hierarchy at read. `nxr-neuro run pyramid` makes it from
//   frame_tiling (the frames)  joint.cells (the grid)  pyramid.band_analytic  pyramid.band_sums  pyramid.frame_samples  pyramid.estimate   (pipelines/src/pyramid_steps.cpp)
// with `run pyramid_bands` (the band bank: pyramid.bands) and `run spatial_bands` (the spatial band bank: pyramid.spatial_bands).
//
// STILL HERE, shared with verbs not yet converted (each goes with its verb): `pyramid_leaves`, `pyramid_leaves_record`, `frame_cells`, `frame_sums_write` (`phasegradient`, `opticalflow`: the leaf x frame
// sums stored beside the build's), `pyramid_bands_selection` (`pyramid reduce`), `pyramid_spatial_bands` (the cycle pattern's spatial band). They hold no maths and no loop over samples.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/tower.h"
#include "nxr/pipelines/input.h"

namespace nxr::pipelines {

// THE LEAVES, shared by every verb that stores its ingredients on the pyramid's leaf partition (`pyramid build`, `phasegradient`): the tree partition `<surface> tiles L<D>` of the surface's ladder (made when the surface has none;
// D default the finest level every piece reaches), its leaf code of every vertex and its leaf areas, read as the ladder stored them. `pyramid_leaves_record` adds nxr's `atlas` block {tree, level, finest, gauge, areas, node}
// to the partition's params_json (what the roll-up reads).
struct PyramidLeaves {
  std::optional<nxr::io::Row> part;                       // the partition's Selection row
  nxr::io::Json record;                                   // its params_json
  std::string name, tree, node;                           // `<surface> tiles L<D>`, the tree kind, the node of its arrays
  int depth = 0, finest = 0;
  std::int64_t leaves = 0;
  std::vector<std::int32_t> codes;                        // [vertices] the leaf of each vertex (< 0: none)
  std::vector<double> leaf_area;                          // [leaves] m^2
};
// `tree` "subject" (the surface's own ladder; the default) or "group".
PyramidLeaves pyramid_leaves(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurfaceInput& surface, std::optional<int> depth, const std::string& who, const std::function<void(const nxr::io::Json&)>& say = {}, const std::string& tree = "subject");
// The stored `atlas bands` bank of a subject (nxr D158), or made from `bands` (spans on the dataset's Frequency manifold, code = the band's index); another bank than the one stored is refused.
std::string pyramid_bands_selection(nxr::io::SubjectWriter& sub, const std::vector<std::pair<double, double>>& bands);
void pyramid_leaves_record(nxr::io::SubjectWriter& sub, PyramidLeaves& leaves, const nxr::io::Json& gauge);

// LEAF x FRAME SUMS of a verb that is not `pyramid build` (step 5: `opticalflow`): per leaf of the leaf partition and per frame of the recording's tower tiling, ADDITIVE sums of a run's pointwise values, weighted by vertex area,
// stored as the build stores its own: arrays [1, leaf, frame] (float64, the whole frame axis of the recording; frames outside the run's span hold 0) in the leaves' node, `measurement` rows on a joint Selection
// `<recording> x <name> <leaves> x <frame_s> s` (cortex x time; params_json.joint, layout [band, tile, frame] with the one band, and the leaves' `atlas` block), of a byte-less Field of the verb, and the frame tiling's per-frame
// `count` (what a mean divides by) of that Field. The lookup (`measure query`) rolls them up in space and time and derives `<measure>_density` (the time mean per unit area) from them.
struct FrameCells {
  nxr::tower::FrameTiling ft;                             // the recording's tower tiling at the frame level
  std::string time_id, frames_id;                         // the recording's time line; the stored tiling `<recording> frames <s> s`
  std::int64_t n_frames = 0;
  int bits = 1;                                           // the joint code: (tile << bits) | frame
};
FrameCells frame_cells(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& recording, int frame_level, std::optional<double> start_s, const std::string& who);
struct FrameSumsRow { std::string measure, rel, unit; std::vector<double> v; };   // v [leaf x frames in the span]: leaf-major; `rel` the array's name in the series' group
struct FrameSumsSeries {
  std::string field_id;                                   // the byte-less Field the rows are of
  std::string group;                                      // the series' node group under the leaves' node
  std::vector<double> counts;                             // [frames in the span] the steps (flow frames) each tower frame holds
  std::vector<FrameSumsRow> rows;
};
// Writes the arrays, the `count` rows and the joint Selection with every series' rows; call it INSIDE `sub.compose`, after `pyramid_leaves_record`. Returns the joint's id.
std::string frame_sums_write(nxr::io::SubjectWriter& sub, const PyramidLeaves& leaves, const FrameCells& fc, const std::string& joint_name, const std::string& description, std::int64_t frame0, std::int64_t frames_span,
                             const std::vector<FrameSumsSeries>& series, const std::string& computed_by, const std::optional<std::string>& acquisition_id);

// THE SPATIAL BANDS (step 4; spec decision 2; nxr-io spatial_bands.h): a surface's `<surface> spatial bands` Selection, a `spans` Selection on the dataset's Eigenvalue line (λ, 1/m²), one span per level of the
// surface's tile ladder (the level IS its Weyl band; the ladder records each level's λ range), code = the level; made (with the ladder, when the surface has none) and stored as `atlas bands` is on the Frequency manifold.
// A measurement's spatial band is its `within` key (`measure query --spatial-band K`); a measurement that has a frequency band too takes the joint bands Selection (`band_key_selection`), code = band * n_spatial + level.
// On the dataset's DEFAULT SUBJECT (the template) the call also makes the Eigenvalue line itself; any other subject needs it already made there.
struct SpatialBandsOptions {
  std::optional<std::string> surface;                      // the subject's surface (default: the primary one)
};
struct SpatialBandsResult {
  nxr::io::Json info;                                      // {surface, selection, selection_id, levels, bands: [{level, lambda_min, lambda_max, wavelength_m}], line}
  std::string selection_id;                                // the bank's Selection
  std::vector<nxr::io::SpatialBandEdge> edges;             // [level] its spans, 1/m^2
  double piece_area = 0;                                   // the ladder's mean piece area (m^2): the bank is Weyl's law on it (lambda_l = 4 pi 2^l / A); nxr::filterbank::weyl(piece_area, n) member l IS level l
};
SpatialBandsResult pyramid_spatial_bands(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SpatialBandsOptions& opt);

}  // namespace nxr::pipelines

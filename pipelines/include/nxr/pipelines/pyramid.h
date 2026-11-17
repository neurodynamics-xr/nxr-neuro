// nxr-pipelines — THE MEASUREMENT PYRAMID BUILD.
// `pyramid build` measures the band power of a recording's cortical currents on the pyramid's cells, and stores it as nxr stores its atlas (what nxr-neuro writes, nxr and the Studio read):
//
//   the graph (a chain of pure nodes, the Studio's device chain; each a function of matrices in nxr::pyramid / nxr::partition, none of them knowing a file, a chunk or a subject):
//     sensors block (chunk + halo)  →  spectrum  →  band analytic signal (Morlet, or nxr's power-partition bank)  →  J = K A (the kernel, constrained orientation)  →
//     per band × leaf × frame: Σ a|J|² (power) · Σ a|J| (env) · max|J| (envmax)  [partition::analyticSums]  →  gather → io
//   the host loads (io's chunk service: the span, the halo, the bad-sample mask), prepares (the kernel's channels in its order, the leaf partition, the bands, the frame tiling, type checks)
//   and stores; it loops over no sample and no chunk of its own.
//
//   THE LEAVES   the tree partition: the `set` Selection `<surface> tiles L<D>` of the surface's ladder (`tiles compute`; made when the surface has none), leaf code h·2^D + t, `w` the leaf areas; D is
//                the finest level every piece reaches by default (leaves a few vertices each), `--depth` chooses. Its params_json gains nxr's `atlas` block {tree, level, finest, gauge, areas, node}.
//   THE BANDS    `atlas bands` (nxr's stored name, D158): a `spans` Selection on the dataset's Frequency manifold, code = the band's index; default octaves 1–2 … 32–64 Hz.
//   THE FRAMES   the tower frame tiling of the recording (`run frametiling`, step 1): `<recording> frames <s> s`, frame k = tower code frame_code0 + k; its per-frame good-sample `count`
//                (array `<recording>__<kernel>_samples`, a measurement row `count` on the tiling, of the source estimate).
//   THE JOINT    one `set` Selection per recording × kernel × tree on cortex × time, code (tile << B) | frame, `params_json.joint` {space, time, bits, tiles, frames, layout [band, tile, frame]};
//                its rows are the arrays `power_sum` · `envelope_sum` · `envelope_max` [band, leaf, frame] (float64, in the node `source-time/<recording>__<kernel>_atlas/<tree>/{power,env,envmax}`),
//                within = the `atlas bands` Selection, code = the band's index, of the SOURCE ESTIMATE: a Field with no bytes (kind `source estimate`, by_operator the kernel, derived_from the recording).
//   STORED       only mergeable ingredients (sums, maxima); means, densities and every ratio are read from them (the lookup, step 3); a bad sample carries no current and no count.
//
// MEMORY (stated): the block held is one chunk and its two halos of the kernel's channels (float64), its spectrum and one band's analytic signal (complex, the same size), the kernel (rows × channels float64),
// and the projection's sub-block (rows × `block_samples` × 4 float64); the accumulators are 3 float64 per (band, leaf, frame of the span), and the store writes the whole frame axis of the recording
// (frames outside the span hold 0, −∞ for the maxima, and a count of 0). Never the whole recording.
//
// WHAT IT CANNOT RESOLVE (recorded in the estimate's producer_json.warnings): the minimum-norm current is smooth (its spatial resolution is the kernel's point-spread, not the leaves': a leaf finer than it
// carries no independent information); a band's analytic amplitude means something only for a band narrow against its centre; the filter is circular over the block, so the first and last `halo` seconds of
// a recording carry its edge effect (nxr's whole-record filter has the same at both ends); a Morlet band overlaps its neighbours, so the band powers do not add to broadband (`power_partition` does).
//
// FREE ORIENTATION (a kernel of three rows a source, x y z;: nxr `measure.vectors.band_tensor`). The current J = K A is projected on the surface's CANONICAL FRAMES (the stored `<surface>_canonical_frames`, the
// gauge: e1 north, e2 west, n outward; `frames compute` first) and the Hermitian orientation tensor of the band's analytic current is summed per band x leaf x frame, nine numbers and the trace:
//   t11 t22 tnn = Σ a|J1|² |J2|² |Jn|²;  t12r t12i = Σ a Re / Im(J1 J2*);  t1nr t1ni (1 with n);  t2nr t2ni (2 with n);  total = Σ b (|J1|² + |J2|² + |Jn|²)
// in the node `.../<tree>/tensor/{t11,...,total}` (float64 [band, leaf, frame]); measures tensor_11 tensor_22 tensor_nn tensor_12_re tensor_12_im tensor_1n_re tensor_1n_im tensor_2n_re tensor_2n_im tensor_total. A vertex
// with no canonical frame (singular: the poles of the registered sphere) is out of the nine sums (weight 0) and in `total` with its tangential power under an arbitrary tangent basis (nxr's convention: its normal row is 0).
// The free estimate has no `power` / `env` / `envmax` (nxr stores the tensor instead; the trace is `tensor_total`). Its warnings add: the principal axis of a leaf is the dominant orientation of the current as the
// kernel's inverse made it, which for a minimum-norm kernel is partly its own prior (the kernel's orientation prior is not a measurement). `tensor_lc` (the Levi-Civita rotation per level) and the trajectory are not built.
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

struct PyramidBuildOptions {
  std::string recording;                                  // REQUIRED: the recording Field by name
  std::optional<std::string> kernel;                      // the inverse kernel Operator by name; default: the subject's only one
  std::optional<int> depth;                               // the leaves' level D of the surface's ladder (2^D tiles a piece); default: the finest level every piece reaches
  std::string tree = "subject";                           // the leaves' tree: subject (the surface's own ladder) | group (the dataset's group tree through the registered sphere: the same leaves in every subject; depth default the dataset's atlas depth)
  std::vector<std::pair<double, double>> bands;           // [lo, hi) Hz; default the octaves 1–2 … 32–64 Hz (nxr DEFAULT_BANDS)
  std::string shape = "morlet";                           // the band filter: morlet (the project's fixed-band rule: a Gaussian centred on (lo+hi)/2, FWHM hi−lo; the default) | power_partition (nxr's bank: band powers add to broadband)
  double halo_seconds = 8.0;                              // the context read on each side of a chunk for the filter (the chunk service's halo); the filter's edge effect is inside it
  int frame_level = -18;                                  // the tower level of a frame (nxr FRAME_LEVEL: 86 400·2^L s, −18 = 0.33 s)
  std::optional<double> start_s;                          // the recording's first sample as seconds from local midnight (anchor day); default: the anchor is the recording
  std::optional<std::string> chunks;                      // what is loaded: chunk "k", chunks "a-b", or "all"; REQUIRED: the caller states the span (or start and stop); the engine streams one chunk at a time
  std::optional<double> start, stop;                      // or an explicit span [start, stop) in seconds on the recording's clock (both given)
  bool exclude_bad_segments = true;                       // the recording's bad segments (the Selection `<recording>_bad_segments`) carry no current and no count
  std::int64_t block_samples = 0;                         // samples projected at a time (memory only: the result does not depend on it); 0 = the default of the kernel's orientation: 512 constrained (rows x block x 4 float64), 128 free (rows x block x 6)
  int threads = 0;                                        // workers of the kernel products (nxr::parallel): 0 = the machine's performance cores, 1 = serial; the result is BIT-IDENTICAL for every count (sums folded in block order), so it is recorded in the provenance but is NOT in the cache key; memory: (threads + 1) blocks of products held at once
  bool replace = false;                                   // an existing estimate and joint Selection of the same recording × kernel × tree are removed and rewritten
};
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

struct PyramidBuildResult {
  nxr::io::Json info;
};
PyramidBuildResult pyramid_build(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PyramidBuildOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});


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

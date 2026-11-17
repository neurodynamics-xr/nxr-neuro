// nxr-io — the inverse kernel and the head model, OPERATORS at the subject root, and what belongs to them: `nxr.convert.inverse` (D40/D106/D107) and, of
// `nxr.convert.condition`, `add_inverse` (`nxr convert inverse`) and the kernels step of a session.
//
//     <kernel>                Operator  float32 [nsrc, M]  channels -> cortex (vector3 when free); its METHOD recorded (`params_json.inverse`), PAIRED with the
//                                       leadfield it was computed from (`inverts_id`, regularized) when the ends swap
//     <kernel>_channels       Selection indices: the M channels it maps, in column order
//     <kernel>_noise_cov      Statistic measure covariance, method empirical (computed elsewhere), [M, M] on the kernel's channels; n_members = nAvg
//     <kernel>_whitener       Operator  W = C^(-1/2) [M, M], derived from the covariance statistic (`of_statistic_id`)
//     <kernel>_source_rr/_nn  Field     of the kernel (`of_operator`): kernel-row-ordered geometry on the cortex
//     <headmodel>             Operator  float32 [M, 3V]    cortex (vector3) -> channels (declared without bytes when no gain)
//     <headmodel>_GridLoc/_GridOrient   Fields of the head model, on the cortex
//
//   auto kernel = bst::read_all_variables("results_..._KERNEL_....mat");
//   bst::InverseInfo ii = bst::export_inverse(sub, kernel, {.acquisition = "rest", .name = "MN_MEG_KERNEL", .channels_id = ch, .surface_id = cortex});
#pragma once
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/bst/subject.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

// THE INVERSE METHOD, RECORDED (D107): what kind of inverse a kernel is, read once from its producer into ONE shape (`params_json.inverse`); nullopt when the
// producer names neither `inverse_method` nor `inverse_measure`.
std::optional<Json> inverse_method_of(const Json& producer);

struct InverseOptions {
  std::string acquisition, name, channels_id, surface_id;
  std::optional<std::string> forward_id;
  std::string kernel_file, bst_root;
  std::optional<std::vector<std::string>> channel_names;
  std::optional<std::int64_t> n_vertices;
  std::optional<std::vector<double>> grid_loc, grid_orient;    // [n x 3], row after row
  std::optional<std::string> source_sha1;
  std::optional<Json> producer_extra;                          // merged into the producer last (another reader's method, units and provenance)
};
struct InverseInfo {
  std::string name, id, channels_id, method, units;
  std::int64_t nsrc = 0, M = 0, n_components = 1;
};
// nxr's info object: name, id, channels_id, method, units, nsrc, M, n_components
Json inverse_json(const InverseInfo& info);

// `kernel` is Brainstorm's results struct (a `results_*KERNEL*.mat`'s variables). ValueError when it has no ImagingKernel.
InverseInfo export_inverse(nxr::io::SubjectWriter& sub, const std::map<std::string, Value>& kernel, const InverseOptions& opt);

struct ForwardOptions {
  std::string acquisition, name, channels_id, surface_id;
  std::int64_t n_channels = 0, n_vertices = 0;
  std::string head_model_file, bst_root;
  bool write_gain = true;
  std::optional<std::string> source_sha1;
};
struct ForwardInfo { std::string name, id; std::int64_t n_sources = 0; };
// `head_model` is Brainstorm's headmodel struct. DECLARED, BYTES ABSENT when `write_gain` is false: a head model without its gain is still the head model the kernel names.
ForwardInfo export_forward(nxr::io::SubjectWriter& sub, const std::map<std::string, Value>& head_model, const ForwardOptions& opt);

struct AddInverseOptions {
  std::string acquisition, bst_root;
  std::optional<std::string> channels_id, surface_id, forward_id, name, channel_file;
};
// ONE inverse kernel (+ its channel Indices, covariance, whitener and source geometry) into an existing subject (`nxr convert inverse`). References are rows of the
// subject: by id when given, else found (the session's channels, the cortex the kernel's own `SurfaceFile` names, the session's head model of the kernel's
// `HeadModelFile`). Its name is Brainstorm's stem, session-qualified if taken (D121); a source already in this session is REFUSED. Announces `{"stage": "inverse", ...}`.
InverseInfo add_inverse(nxr::io::SubjectWriter& sub, const std::filesystem::path& kernel_file, const AddInverseOptions& opt, const Emit& progress);

// SESSION HOOK. `convert_condition`'s step "kernels (+ head models, deduped per file)": every kernel of the condition with its head model, written on the session's
// channels and cortex. The session port calls it after the sensors and the recordings. Returns the names of the kernels written or shared; announces
// `skip-kernel`, `stale-headmodel-link`, `forward`, `forward-shared`, `inverse` and (no kernel at all) `no-kernel`, as nxr's does.
struct ConditionKernels {
  std::filesystem::path cond_dir, bst_root;
  std::string acquisition, channels_id, surface_id;
  std::int64_t n_channels = 0, n_vertices = 0;
  std::vector<std::string> channel_names;
  std::optional<std::vector<std::filesystem::path>> kernel_files;           // default: `cond_dir/results_*KERNEL*.mat`
  std::map<std::string, std::string> head_model_map;                        // a kernel's HeadModelFile link -> the file
  bool write_gain = true, require_source_grid = true;
};
std::vector<std::string> import_condition_kernels(nxr::io::SubjectWriter& sub, const ConditionKernels& opt, const Emit& emit);

}  // namespace nxr::bst

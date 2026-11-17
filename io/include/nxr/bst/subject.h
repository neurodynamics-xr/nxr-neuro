// nxr-io — a Brainstorm subject: `nxr.convert.surface.export_surface` (its anatomy) and `nxr.convert.subject` (the composition of the whole subject, the template and
// the group). `convert_subject` writes EVERY part of one subject as ONE composition (all or nothing), in nxr's order:
//   1. every condition with recordings or a raw link, in sorted order (a session each; the first brings the anatomy)    -> sessions (`convert_condition`)
//   2. every subject image (mri.h)                                                                                      -> volumes
//   3. every source map (`results_*` that is not a kernel) in every condition, map-only ones included                  -> source maps (`export_source_map`)
//   4. every fibres file, its ends on the cortex and its connectomes (fibers.h)
// The sessions are session.h's, the maps maps.h's; everything skipped is reported with its reason.
//
//   DatasetWriter ds = open_writer("store/demo");
//   Json r = bst::convert_subject(ds, "protocol", "S01", [](const Json& line) { ... }, "nxr-io 0.2.0");
//   Json t = bst::convert_template(ds, "protocol", std::nullopt, emit, "nxr-io 0.2.0");     // anat/@default_subject as the TEMPLATE subject (D55)
//   Json g = bst::convert_group(ds, "protocol", emit, "nxr-io 0.2.0");                       // data/Group_analysis/@intra as the GROUP subject (D55)
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/protocol.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

using Emit = std::function<void(const Json&)>;   // one progress line (`stage` keyed)

// One Brainstorm tess_*.mat as a `surface` manifold and what describes it: its hemispheres, its registration sphere (`Reg.Sphere`) and its parcellations.
// `primary`: the surface the kernel indexes (refused when its mesh is unhealthy only under NXR_STRICT_MESH=1; a display surface only warns); only a primary is
// marked `is_primary`. `bst_root`: the protocol root, which makes the manifold's `source` the file's path under `anat/`. `atlases`: only these parcellations (an empty list: none; unset: every one).
nxr::io::SurfaceInfo export_surface(nxr::io::SubjectWriter& sub, const std::filesystem::path& surface_file, const std::string& bst_root = "",
                                    bool primary = true, const std::optional<std::vector<std::string>>& atlases = std::nullopt);

// A surface's `{"stage": "surface", ...}` line as `nxr convert surface` prints it (name, folder, counts, flipped, mesh_health), without its id.
Json surface_json(const nxr::io::SurfaceInfo& info);

// The subject's anatomy folder, and whether it is INHERITED (`UseDefaultAnat`: it borrows `@default_subject`).
std::pair<std::filesystem::path, bool> anat_dir(const std::filesystem::path& root, const std::string& subject);

// The surface this condition's kernels are defined on (their own `SurfaceFile`); nullopt when there is no kernel or they disagree. Both condition homes.
std::optional<std::string> kernel_surface(const std::filesystem::path& cond_dir, const std::filesystem::path& raw_dir, const std::filesystem::path& anat);

// What `convert_condition` reads for one condition (`condition_plan`): THE PRIMARY SURFACE IS THE INVERSE'S (its kernels' own `SurfaceFile`; else the index's `iCortex`;
// else the pial literal), THE CONTINUOUS IS THE DEFAULT (the raw link; excerpts opt-in, or all there is without a link), channels and kernels in either home.
struct ConditionPlan {
  std::filesystem::path surface_file;
  std::vector<std::filesystem::path> extra_surfaces, data_files, kernel_files;
  std::filesystem::path channel_file;
  std::optional<std::filesystem::path> raw_data_file;
};
struct ConditionPlanOptions {
  std::optional<std::string> surface, recording, channel_file;
  std::vector<std::string> extra_surfaces;
  bool include_imported = false, no_raw = false;
};
// ValueError when there is nothing to import, no channel file, or not exactly one raw link.
ConditionPlan condition_plan(const std::filesystem::path& root, const std::string& subject, const std::string& condition, const ConditionPlanOptions& opt = {});

// The subject's cortex as the fibres were assigned on it: Brainstorm's `iCortex`, else `brainstormsubject.mat`'s `Cortex`, else the subject's primary surface (`tess_<name>.mat`).
std::string subject_cortex(const std::filesystem::path& root, const std::string& subject, nxr::io::SubjectWriter& sub);

// Every part of the Brainstorm subject into the SubjectWriter `sub` (rows pending; the caller completes). Returns {subject, sessions, volumes, maps, fibers, connectomes, skipped}.
Json compose_subject(nxr::io::SubjectWriter& sub, const std::filesystem::path& root, const std::string& subject, const Emit& emit, std::optional<double> resample = std::nullopt);

// The whole subject as ONE composition: created pending, completed once (one drain), or removed whole. Returns nxr's result object (the summary, id, rows).
Json convert_subject(nxr::io::DatasetWriter& ds, const std::filesystem::path& root, const std::string& subject, const Emit& emit, const std::string& created_by,
                     std::optional<double> resample = std::nullopt);

// The protocol's default anatomy (`anat/@default_subject`) as the dataset's TEMPLATE subject (D55/D127): every low-resolution cortex (pial first, the primary), its images and
// the STATIC MAPS on it; a map that cannot convert is reported and skipped. `surfaces`: only these `tess_*.mat`. FileNotFoundError when the protocol has no default anatomy.
Json convert_template(nxr::io::DatasetWriter& ds, const std::filesystem::path& root, const std::optional<std::vector<std::string>>& surfaces, const Emit& emit,
                      const std::string& created_by);
// The protocol's group results (`data/Group_analysis/@intra`) as the dataset's GROUP subject (D55): each average a Field whose contributions are the members' fields.
Json convert_group(nxr::io::DatasetWriter& ds, const std::filesystem::path& root, const Emit& emit, const std::string& created_by);

}  // namespace nxr::bst

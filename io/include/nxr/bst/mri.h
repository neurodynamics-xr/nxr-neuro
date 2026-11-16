// nxr-io — a subject's MRI: `nxr.convert.mri`. Brainstorm `subjectimage_*.mat` -> `mri/<name>` entities: the T1 (a volume Manifold in closed form with its
// voxel chart, a Field of intensities, the voxel -> SCS / NCS transitions as 3x4 affines), an atlas volume as CODES (a Selection on the T1's volume read with an
// app-wide dictionary, Brainstorm's label value kept per entry) and a float map (a Field on the T1's volume).
//
// THE GATE: the fiducials are in the file with the cube, so a left-right mirror is refused (`verify_scs`), never warned. A volume whose SCS frame or grid is not the
// T1's is not on its Volume and is refused.
//
//   auto img = bst::read_subjectimage("anat/S01/subjectimage_MRI_T1.mat");
//   sub.compose([&] { bst::import_subjectimages(sub, "anat/S01", std::nullopt, emit); });
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/entities.h"
#include "nxr/bst/mat.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

using nxr::io::Json;

struct MriLabels {
  std::vector<std::int32_t> ids, source_ids;         // the codes 0..N-1 and Brainstorm's label values
  std::vector<std::string> names;
  std::vector<Rgb> colors;
  std::vector<std::int64_t> unlabelled;              // values in the cube the table did not name (each got a row)
};

struct Mri {
  std::string name, kind;                            // kind: anatomical | atlas | scalar
  std::array<std::int64_t, 3> shape{};
  std::array<double, 3> voxsize{1, 1, 1};            // mm
  std::map<std::string, std::vector<double>> scs, ncs;   // R (9, row-major) · T · NAS · LPA · RPA · Origin / AC · PC · IH
  std::string comment, source;
  // the cube, C order [ni, nj, nk]: uint8 intensities (anatomical), the atlas's table rows (uint16), float32 values (scalar)
  std::vector<std::uint8_t> u8;
  std::vector<std::uint16_t> codes;
  std::vector<float> f32;
  std::optional<MriLabels> labels;
};

constexpr std::int64_t kMriChunk = 64;               // cubic chunks (any orthogonal slice touches a plane of chunks)
constexpr double kScsTolMm = 1e-2;

// `subjectimage_MRI_T1.mat` -> `T1`; `..._volatlas.mat` -> the atlas's name; a timestamp-only file is named by its comment.
std::string mri_name(const std::filesystem::path& file, const std::string& comment = "");
// voxel index -> SCS millimetres (`R @ (voxel * voxsize) + T`)
std::array<double, 3> to_scs(const std::array<double, 3>& voxel, const std::map<std::string, std::vector<double>>& scs, const std::array<double, 3>& voxsize);
// The four gates; ValueError, never a warning.
void verify_scs(const std::map<std::string, std::vector<double>>& scs, const std::array<double, 3>& voxsize);
// One file, its orientation verified. ValueError for what nxr raises ValueError for; NotImplementedError for a v7.3 file.
Mri read_subjectimage(const std::filesystem::path& file);
// One volume into `mri/`; ValueError when the T1 is not there, its frame differs, or the node exists. Returns the node's path.
std::string write_mri(nxr::io::SubjectWriter& sub, const Mri& img);
// `subjectimage_*.mat` of a folder, anatomical first.
std::vector<std::string> find_subjectimages(const std::filesystem::path& anat_dir);

using Emit = std::function<void(const Json&)>;
// Every subject image of `anat_dir`: the T1 first, atlases as codes, float maps as Fields. An unreadable one is reported (`stage: skipped`) and skipped; none
// imported is a FileNotFoundError. Returns the `stage: mri` records.
std::vector<Json> import_subjectimages(nxr::io::SubjectWriter& sub, const std::filesystem::path& anat_dir, const std::optional<std::vector<std::string>>& only,
                                       const Emit& progress);

}  // namespace nxr::bst

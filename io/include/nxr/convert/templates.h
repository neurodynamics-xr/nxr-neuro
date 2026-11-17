// nxr-io — TEMPLATE IMPORTERS (io/convert): the native readers that build a template package from the files a user downloaded, with no MATLAB, Brainstorm or other
// program at run time, each writing an ORIGIN record (source, licence, attribution, whether it may be redistributed) on every item it writes.
//
//   freesurfer   a FreeSurfer subject folder (fsaverage): the surfaces, the registered sphere, the parcellations NAMED by the caller and the cortex mask
//   icbm152      a Brainstorm default-anatomy folder (ICBM152): its cortex surfaces with their registered spheres, the parcellations NAMED by the caller
//   hcp1065      a TrackVis .trk (DSI Studio): the streamlines in the anatomy's coordinates, and their ends on a surface, gated by a distance
//
// The importer rules (maintainers, 2026-11-16): parcellations only when named explicitly; a template is the package its data were computed in. The source files are checked
// against `manifest/template_sources.json` (compiled in): a file whose SHA-256 differs, or that is not listed, is refused unless the caller accepts it, and the checksum
// actually read goes into the origin. `build_template` makes the package; `release_package` (templates.h) freezes it.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

using Json = nxr::io::Json;

// ── the manifest of source files ───────────────────────────────────────────────────────────────────────────────
// The compiled-in manifest (docs: `io/convert/manifest/template_sources.json`).
const Json& source_manifest();
// Checks the files a builder reads against the manifest, remembering what it found.
class SourceChecker {
 public:
  SourceChecker(std::string recipe, std::filesystem::path folder, bool accept_unlisted);
  // `relative` under the folder: its path, after checking its SHA-256 against the manifest (std::invalid_argument-like refusals are nxr::io::Error invalid_argument).
  std::filesystem::path file(const std::string& relative);
  // {recipe, folder-less list of {path, sha256, listed, matches}}: what was read.
  Json report() const;
  const Json& recipe() const;
  std::string sha256_of(const std::string& relative) const;   // of a file already read

 private:
  std::string recipe_name_;
  std::filesystem::path folder_;
  bool accept_;
  Json recipe_;
  Json read_ = Json::array();
};

// ── the TrackVis (.trk) reader ─────────────────────────────────────────────────────────────────────────────────
struct Tractogram {
  std::vector<float> points;                 // every streamline's points, [n x 3], in RAS+ mm as nibabel's `streamlines` reads them (world coordinates)
  std::vector<std::int64_t> offsets;         // n_streamlines + 1: streamline s is points offsets[s] .. offsets[s+1]
  std::array<int, 3> dimensions{};
  std::array<float, 3> voxel_size{};
  std::string voxel_order;
  std::int64_t n_streamlines() const { return static_cast<std::int64_t>(offsets.size()) - 1; }
};
// TrackVis version 2 with a valid voxel-to-RAS matrix (the header's streamline count may be 0: the records are walked). The points go from voxel-mm to RAS+ mm as
// nibabel does (scale to voxels, the half-voxel corner shift, the orientation of the header's voxel order against the matrix's, then the matrix). Refused: another
// version, an all-zero matrix, a scalar or property count it does not carry, a truncated file.
Tractogram read_trk(const std::filesystem::path& path);
// Every streamline resampled to `n_points` at equal arc length (linear along the polyline); a streamline of fewer than two distinct points is dropped. Returns [n x n_points x 3]
// float32 and the indices (into the tractogram) of the streamlines kept.
std::vector<float> resample_streamlines(const Tractogram& t, int n_points, std::vector<std::int64_t>* kept = nullptr);

// ── the scalp and the landmarks ─────────────────────────────────────────────────────────────────────────────────
// The landmarks of a template's head, in metres in the template's coordinate system: from Brainstorm's subjectimage_T1.mat (SCS.NAS, LPA, RPA through its SCS frame), or from MNE's
// fiducials file of fsaverage (`bem/fsaverage-fiducials.fif`, in FreeSurfer's surface RAS). The inion is not here: no template carries one (nxr::montage::inionFromLandmarks is the default).
struct HeadLandmarks {
  std::array<double, 3> nasion{}, lpa{}, rpa{};
};
HeadLandmarks landmarks_from_subjectimage(const std::filesystem::path& subjectimage_mat);
HeadLandmarks landmarks_from_fif(const std::filesystem::path& fiducials_fif);

// ── what to write, and the origin of what is written ───────────────────────────────────────────────────────────
struct FreeSurferSource {
  std::filesystem::path folder;              // the subject folder holding surf/ and label/
  std::string label = "fsaverage";           // its name in provenance
  std::vector<std::string> surfaces = {"pial", "white", "inflated"};   // the first is the primary; each is lh.<name>, rh.<name>; the registered sphere is sphere.reg
  std::vector<std::string> annotations;      // parcellations, NAMED: "aparc" (Desikan-Killiany), "aparc.a2009s" (Destrieux), any other `<hemi>.<name>.annot`; none by default
  bool cortex_mask = false;                  // the cortex label (the vertices that are not the medial wall), as a selection `cortex_pial_cortex`
};
struct BrainstormAnatomySource {
  std::filesystem::path folder;              // the default-anatomy folder: tess_cortex_*.mat, subjectimage_T1.mat
  std::vector<std::string> surfaces = {"tess_cortex_pial_high.mat"};   // the first is the primary
  std::vector<std::string> atlases;          // parcellations, NAMED (as the surface file lists them); none by default
};
// The atlas terms the importer knows (by name): its licence and attribution, and whether it may be redistributed. A name not in the table is written with the licence "not
// confirmed" and is NOT redistributable. Hammers and LPBA40 are forbidden everywhere: written to a full edition only when named, never into a redistributable one.
struct AtlasTerms { std::string licence, attribution; bool redistributable = false, forbidden = false; };
AtlasTerms atlas_terms(const std::string& name);

struct HcpSource {
  std::filesystem::path trk;                 // the .trk
  std::array<double, 12> mni_to_anatomy_m{}; // the 4x3 affine (row-major, last row the translation) from MNI mm (the .trk's, after the validated shift) to the anatomy's metres
  int points = 64;                           // points per streamline after resampling
  std::string end_surface;                   // the surface the ends are placed on (by name in the package): required
  double max_end_distance_mm = 5;            // an end farther than this from the end surface is left out; distances are stored either way
  std::array<double, 3> shift_mm = {-1, -1, 1};   // nibabel's TrackVis reading to MNI mm (the validated half-voxel shift)
  std::string name = "hcp1065";
};
// The affine from MNI mm to the anatomy's SCS metres, from the anatomy's own subjectimage (its NCS and SCS frames): scs_mm = SCS.R (NCS.R^-1 (mni - NCS.T)) + SCS.T.
std::array<double, 12> mni_to_scs_from_subjectimage(const std::filesystem::path& subjectimage_mat);

struct BuildRequest {
  std::string recipe;                        // fsaverage | icbm152
  std::filesystem::path sources;             // the folder of the downloaded files
  std::filesystem::path package;             // the working package to make; must not exist
  std::string name;                          // the subject name (default: the recipe's template name)
  std::string home_dataset = "templates";
  std::string created_by;
  bool accept_unlisted_sources = false;
  std::vector<std::string> annotations;      // fsaverage: annot names; icbm152: the atlases named in the surface files
  bool cortex_mask = false;                  // fsaverage
  std::vector<std::string> surfaces;         // fsaverage: the surface names; icbm152: the tess files (default as the sources')
  std::optional<std::filesystem::path> hcp_trk;   // icbm152: also import the streamlines (the manifest's `hcp1065` recipe checks the file)
  std::string hcp_end_surface;               // required with hcp_trk
  double hcp_max_end_distance_mm = 5;
  int hcp_points = 64;
  bool scalp = false;                        // also write the scalp (a closed head surface, `head`) and the landmarks NAS, LPA, RPA (`head_landmarks`), for the standard electrode montages
  std::optional<std::array<double, 12>> hcp_affine;   // replaces the transform read from subjectimage_T1.mat (a reference's fitted affine)
};
// Make the package: the importer, its origin on every item, the record's description. Returns {package, name, content_hash, items, sources, ...}. A failed build leaves nothing.
Json build_template(const BuildRequest& request);

}  // namespace nxr::convert

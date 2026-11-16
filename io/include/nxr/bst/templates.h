// nxr-io — `nxr.convert.templates`: the TEMPLATES-AND-REFERENCES datastore. fsaverage (ico7) as the template subject, reference maps as Fields on it, the HCP-1065
// streamlines as fibres (optional). Input is the prepared folder (`references-fsaverage-ico7`): `fsaverage/<fsaverage|fsaverage5|fsaverage6>/{surf,label}` and
// `maps/<id>/<name>_space-fsaverage_den-164k_hemi-{L,R}.func.gii` + `provenance.json`. Re-runnable: every step skips what the subject already holds.
//
//   Json r = bst::build_templates("references-fsaverage-ico7", "store/templates", {.trk = "hcp1065.trk"}, "nxr-io 0.2.0");
//
// The readers and writers nxr gets from nibabel and from `nxr.pipelines.atlas` (`default_subject`, `hcp`) are DECLARED below with nxr's names; they belong to the
// atlas port. They are NOT IMPLEMENTED: each throws `NotImplementedError`, naming the reader that is missing (defined at the end of `src/bst_templates.cpp`).
#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::bst {

using Json = nxr::io::Json;

struct TemplatesOptions {
  std::string subject = "fsaverage";
  std::optional<std::filesystem::path> trk;       // hcp1065.trk (DSI Studio): add the streamlines and their ends
  std::optional<std::string> surface;             // the surface the maps go on (default: the subject's surface with each map's vertex count)
};

// Build or extend the templates dataset at `dataset` (`<datastore>/<dataset>`), or add the maps to ANY subject that already has its surfaces. Returns {store, id, surfaces, fields,
// skipped, unplaced[, hcp1065]}.
Json build_templates(const std::filesystem::path& source, const std::filesystem::path& dataset, const TemplatesOptions& opt, const std::string& created_by);

struct HemiMesh { std::vector<double> pos; std::vector<std::int32_t> faces; };   // millimetres [n x 3]; 0-based [m x 3]
// A FreeSurfer surface of both hemispheres (`<fs_dir>/surf/lh.<surface>`, `rh.<surface>`): _read_hemis (nxr::io::read_freesurfer_surface).
std::pair<HemiMesh, HemiMesh> read_hemis(const std::filesystem::path& fs_dir, const std::string& surface);

// ── declared, not implemented (nxr.pipelines.atlas.default_subject / hcp; nibabel) ──
// A FreeSurfer label file's vertex indices (nibabel.freesurfer.read_label).
std::vector<std::int32_t> read_label(const std::filesystem::path& file);
// The first data array of a GIfTI file as float32 (nibabel).
std::vector<float> read_gifti(const std::filesystem::path& file);
// DefaultSubjectWriter.surface: a surface in metres with its optional sphere (registration), `n_lh` left vertices first. Returns the manifold's id.
std::string default_surface(nxr::io::SubjectWriter& sub, const std::string& name, const std::vector<double>& pos_m, const std::vector<std::int32_t>& faces,
                            const std::optional<std::vector<double>>& sphere, std::int64_t n_lh, const Json& producer, bool primary);
// DefaultSubjectWriter.parcellations: Structures, Desikan-Killiany, Destrieux as partitions of the surface.
void default_parcellations(nxr::io::SubjectWriter& sub, const std::filesystem::path& fs_dir, const std::string& surface_node, const std::string& surface_id, std::int64_t n_lh,
                           std::int64_t n);
// _fibres: the HCP-1065 streamlines as a curve manifold `hcp1065` and its ends operator `hcp1065_ends`. Returns {fibres, ends_on_cortex, of}.
Json hcp_fibres(nxr::io::SubjectWriter& sub, const std::filesystem::path& trk, const std::string& pial_id, const std::vector<double>& verts, const std::vector<double>& white);

}  // namespace nxr::bst

// nxr-io — a surface of a subject, from a mesh file: `nxr.io.mesh` (the reader) and `nxr.convert.surface.add_surface` (the writer every source
// shares) with `nxr.convert.winding` and `nxr.convert.mesh_health`.
//
//   Mesh m = read_mesh("bunny.obj");                      // vertices float64 [nV x 3], faces int32 [nF x 3] 0-based
//   SurfaceInfo info = add_surface(sub, {.name = "bunny"}, m.vertices, m.faces);
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::io {

struct Mesh {
  std::vector<double> vertices;        // [nV x 3] in the file's own units
  std::vector<std::int32_t> faces;     // [nF x 3], 0-based
  std::int64_t n_vertices() const { return static_cast<std::int64_t>(vertices.size() / 3); }
  std::int64_t n_faces() const { return static_cast<std::int64_t>(faces.size() / 3); }
};

// A file's default surface name: its file name without the mesh suffix (`bunny.obj` -> `bunny`).
std::string mesh_name(const std::filesystem::path& path);
// A triangle mesh from a file; the format follows the name: .obj · .ply (ascii, binary little- and big-endian) · .stl (ascii and binary).
// A polygon is triangulated as a fan from its first vertex; an STL's coincident vertices are merged. A FreeSurfer surface (`lh.*`, `rh.*`) is read
// by read_freesurfer_surface; GIfTI is nxr's Python reader's through nibabel: refused here, by name.
Mesh read_mesh(const std::filesystem::path& path);
// A FreeSurfer triangle surface (`lh.pial`, `rh.sphere.reg`, …; nibabel.freesurfer.read_geometry): vertices as stored (mm), faces 0-based.
// `read_mesh` reads a file named `lh.*` / `rh.*` with it.
Mesh read_freesurfer_surface(const std::filesystem::path& path);

// A FreeSurfer annotation (`lh.aparc.annot`, …) as nibabel.freesurfer.read_annot returns it: per vertex the ROW of the colour table its label packs to
// (r + g·2^8 + b·2^16), −1 for 0 (no label); the table [rows x 5] (r, g, b, t, the packed value); the names in the order the file lists them (the old
// format's order; the new format's entries as read, while their rows are their own indices, as nibabel does).
struct FsAnnot {
  std::vector<std::int32_t> labels;
  std::vector<std::array<std::int32_t, 5>> ctab;
  std::vector<std::string> names;
};
FsAnnot read_freesurfer_annot(const std::filesystem::path& path);
// A node name: a run of non-alphanumeric characters (hyphen, dot and underscore are safe) collapses to one underscore; ends are trimmed.
std::string sanitize_node_name(const std::string& name);

// Face winding canonicalized to CCW-outward per connected component (a negative signed volume flips the component).
std::vector<std::int32_t> canonicalize_winding(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, std::vector<std::int64_t>* flipped = nullptr);

struct MeshHealth {
  std::int64_t n_vertices = 0, n_faces = 0, non_manifold_edges = 0, duplicate_faces = 0, degenerate_faces = 0, isolated_vertices = 0, boundary_edges = 0,
               inconsistent_winding_edges = 0;
  bool fatal() const;
  Json as_attrs() const;
  std::string summary() const;
};
MeshHealth check_mesh(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces);

// A hemisphere: a component of a cortex (its label, its first vertex and its vertex count), the vertices lying in one run.
struct Hemisphere {
  std::string label;
  std::int64_t first = 0, count = 0;
};
// A parcellation of the vertices (a labelling): a code per vertex, the level names (code 0 first) and a colour per level.
struct AtlasMember {
  std::string family;
  std::vector<std::int32_t> codes;
  std::vector<std::string> levels;
  std::vector<std::array<int, 3>> colors;
};

struct SurfaceOptions {
  std::string name;
  bool primary = true;                    // the subject's primary surface if it has none and the mesh is a cortex
  std::optional<std::string> source;      // the file it came from
  std::optional<std::string> comment;
  std::string cs_key = "scs";
  std::optional<Json> producer;           // rides in the manifold's provenance
  // Brainstorm's anatomy (`nxr.convert.surface.add_surface`): the registration sphere (`sphere_cols` values a vertex; used when it holds one row a
  // vertex), the hemispheres (components of the cortex), and the parcellations (a set `<surface>_atlases`, each member a partition).
  std::vector<double> sphere;
  std::int64_t sphere_cols = 3;
  std::string sphere_source = "sphere.reg";
  std::vector<Hemisphere> hemispheres;
  std::vector<AtlasMember> atlases;
  std::optional<std::string> atlases_description;   // the set's description; default "the cortical parcellations — <families>"
};
struct SurfaceInfo {
  std::string name, id, folder;
  std::int64_t n_vertices = 0, n_faces = 0;
  std::int64_t atlases = 0, hemispheres = 0;   // a mesh file has none
  bool registration = false;
  std::vector<std::int64_t> flipped;
  Json mesh_health;
};
// Arrays in, a `surface` manifold out, in the folder the layout gives it (`source/<name>`, `mri/<name>` for a head surface).
SurfaceInfo add_surface(SubjectWriter& sub, const SurfaceOptions& opt, const std::vector<double>& vertices, const std::vector<std::int32_t>& faces);

}  // namespace nxr::io

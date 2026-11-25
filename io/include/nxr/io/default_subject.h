// nxr-io — the dataset's DEFAULT SUBJECT (D127): the dataset's template (the average anatomy subjects are registered onto), Brainstorm's `@default_subject`.
//
//   DefaultSubjectOptions o;
//   o.templates = ".../brain-maps/spaces/fsaverage";                       // a FreeSurfer subjects folder
//   Json r = create_default_subject(ds, o, "nxr-neuro 0.1.0");
//
// It is written as any subject is (SubjectWriter, add_surface, the same rows and folders), as ONE composition: complete, or nothing. Its row:
// name `@default_subject`, kind `template`, store `@default_subject.nxr.zarr`, source FreeSurfer. It holds the cohort (format 56): each subject's
// registration onto it and results carried onto it, and the averages and statistics over them; a subject never points at it. Its contents, the BASICS of nxr's `atlas default-subject` (nxr.pipelines.atlas.default_subject):
//
//   source/cortex_pial            the pial surface of `<templates>/<default>` (fsaverage5 by default), left then right, metres; the PRIMARY cortex
//   source/cortex_pial_1, _2      its hemispheres (left, right): components over the same arrays
//   source/cortex_pial_sphere     the registration sphere (sphere.reg), the pial's topology
//   source/cortex_pial_atlases    the parcellations (a set): Structures, Desikan-Killiany (`label/{lh,rh}.aparc.annot`), Destrieux (`aparc.a2009s`), each a
//                                 partition whose dictionary (app scope) names "<region> L|R" with FreeSurfer's colours (nxr's DefaultSubjectWriter.parcellations)
//   source/cortex_white, cortex_inflated   (each with its two hemispheres)
// and, written by the pipelines after it (`dataset create`): the pial's FRAMES (nxr::pipelines::frames_compute: the canonical frames from the registered sphere, the
// anatomical frame from FreeSurfer's RAS axes), then the GROUP TREE (nxr::pipelines::group_tree_compute): the pial's `ladder` with the atlas definitions and the
// Levi-Civita connection between its tile centres, and, with `dataset create --modes K`, the pial's Hodge–Dirac basis (nxr::pipelines::eigen_solve: it is compute's, so io
// never writes it).
//
// Not yet (nxr's default subject has them): the option resolutions,
// the HCP-1065 fibres, the Frequency manifold.
#pragma once
#include <filesystem>
#include <optional>
#include <string>

#include "nxr/io/writer.h"

namespace nxr::io {

constexpr const char* kDefaultSubject = "@default_subject";

struct DefaultSubjectOptions {
  std::filesystem::path templates;          // a FreeSurfer SUBJECTS folder holding `<default>/surf/{lh,rh}.{pial,white,inflated,sphere.reg}`
  std::string default_resolution = "fsaverage5";   // the subject in it: the dataset's resolution (nxr's --default)
};

// Write the dataset's default subject. Refused when the dataset already has one (`@default_subject`). Returns {subject, id, store, resolution,
// surfaces: [{name, n_vertices, n_faces, coincident_vertices_moved}]}.
Json create_default_subject(DatasetWriter& ds, const DefaultSubjectOptions& opt, const std::string& created_by);

// Move every vertex that sits exactly on an earlier one to the centroid of its other neighbours (nxr.geom.frames.separate_coincident): fsaverage (ico7)
// has one (vertex 40969 on vertex 0, in both hemispheres), two faces of zero area on which no cotan operator is defined. Returns the vertices moved.
std::vector<std::int64_t> separate_coincident(std::vector<double>& vertices, const std::vector<std::int32_t>& faces);

}  // namespace nxr::io

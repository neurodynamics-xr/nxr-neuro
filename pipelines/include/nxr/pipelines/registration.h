// nxr-pipelines — `create registration`: the registration between two templates (or a template and a subject) through their REGISTERED SPHERES, stored as a
// package of kind `registration` (docs/specs/2026-11-16-subject-centred-datasets.md section 4.4). This file only PREPARES the inputs (it finds the two released
// templates by name, edition and version, reads their spheres, type-checks them), runs the compute node (`nxr::operators::registration::resample_operator`, a pure function
// of matrices) and hands the operators to io; no maths, no loop over vertices.
//
//   graph        from sphere [Vf x 3], to sphere [Vt x 3]  --resample_operator(method, K)-->  W [Vt x Vf] (CSR), and in the other direction W' [Vf x Vt]
//   type checks  both surfaces exist and are spheres of their template; they have the same number of pieces (a cortex's hemispheres, as runs of vertices), matched
//                in order; a point at the centre is refused; the two templates differ
//   stored       one operator per direction (`<from>_to_<to>`, kind `resampling`, CSR, `from_manifold_id` on the source template and `to_manifold_id` on the
//                destination, both anchored in the package), with the method, its options and what it cannot resolve in the package record and the operator's producer.
//
// The default method is the reference's (modified Shepard weights over the 8 nearest vertices); `nearest` and `barycentric` are named options.
// What it cannot resolve: the correspondence is the spheres', not anatomy's; every method smooths; a field carried there and back is not the field.
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/io/templates.h"

namespace nxr::pipelines {

struct RegistrationCreateOptions {
  std::filesystem::path package;               // the registration package made (a working copy: `release template` freezes it); must not exist
  std::string name;                            // its subject name
  std::string from, to;                        // the two templates, `NAME@VERSION` or `NAME@VERSION:redistributable`; they must differ
  std::string from_sphere, to_sphere;          // the surfaces (by name) of the two templates that carry their registered spheres (e.g. `cortex_pial_sphere`)
  std::string method = "shepard";              // shepard (the reference's) | nearest | barycentric
  int neighbours = 8;                          // shepard: the K nearest source vertices (the K-th sets the radius and takes no weight); at most the piece's size
  bool both_directions = true;                 // also the operator from `to` to `from`; false: from -> to only
  std::vector<std::filesystem::path> templates;   // the folders to look for the templates in, before the environment and the user setting
  std::string home_dataset = "registrations";  // the package's home dataset (provenance, not membership)
  std::string created_by;                      // the program (the command's stamp)
  // The origin of every item: by default derived from the two spheres' own origin records (source: the two templates; licences and attributions joined;
  // redistributable only when both are). A caller's value replaces the derived one.
  std::optional<std::string> source, licence, attribution;
  std::optional<bool> redistributable;
};

struct RegistrationCreateResult {
  std::filesystem::path package;
  nxr::io::Json info;                          // {name, method, neighbours, from, to, operators: [{name, from, to, nnz, empty_columns, max_chord}], warnings, redistributable, licence}
};

RegistrationCreateResult registration_create(const RegistrationCreateOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

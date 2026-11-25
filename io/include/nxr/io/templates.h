// nxr-io — TEMPLATES AS PACKAGES (docs/specs/2026-11-16-subject-centred-datasets.md section 4): the anatomy every subject is measured against (surfaces, registration
// spheres, parcellations, eigenmodes, the group tree, fibres, reference maps) is a subject package of kind `template`, opened exactly as any subject is.
//
//   RELEASED AND FROZEN. `release_template` copies a template package into a templates folder as `<name>@<version>.nxr.zarr` and writes the release into its record; a
//   writer never opens a released package again (a change is a new version, a new package). It refuses a template with an item that has no ORIGIN record (source,
//   licence, attribution and whether its licence lets it be redistributed), so licences are checked before anything is shared. The content hash does not change by
//   being released: it is what every reference names.
//   TWO EDITIONS. `NAME@VERSION` is the FULL edition; `NAME@VERSION:redistributable` keeps only the items whose origin says they may be redistributed, byte for byte the
//   full edition's (their key digests are equal), and lists what it left out. As a folder name the second is `<name>@<version>+redistributable.nxr.zarr` (':' is not
//   portable in a file name).
//   FOUND BY NAME, VERSION AND HASH through the TEMPLATE SEARCH PATH, in order: the locations the command names, the environment variable NXR_TEMPLATE_PATH (a list
//   separated as PATH is), the user setting `template_path` of nxr's settings file, and the hint recorded with the reference. The path says only WHERE to look; WHICH
//   template is used is fixed by the reference, so no setting chooses data. A package with the right name and version but another hash is skipped and reported.
//   A SUBJECT names no template (format 56): the dataset's own template subject holds the cohort (each subject's registration onto it, the carried results, the
//   averages and statistics) and refers to its subjects. Released templates and registrations serve building and comparing templates only.
//   A REGISTRATION between two templates is a package of kind `registration`, released and versioned the same way.
//   The datastore and its `catalog.json` are RETIRED: nothing here reads or writes them (the old calls stay for reading a dataset before it is split).
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/subject_package.h"

namespace nxr::io {

constexpr const char* kTemplatePathEnv = "NXR_TEMPLATE_PATH";

struct TemplateRef {
  std::string name, version;
  std::string edition = "full";                       // "full" or "redistributable"
  std::string content_hash;                           // what the reference fixes: required to resolve
  std::optional<std::string> location_hint;           // where it was when the dataset was made (searched last)
  Json to_json() const;
  static TemplateRef from_json(const Json& j);
};
// "NAME@VERSION" or "NAME@VERSION:redistributable" (no hash: the caller adds it).
TemplateRef parse_template_ref(const std::string& text);
// `<name>@<version>.nxr.zarr`, or `<name>@<version>+redistributable.nxr.zarr`.
std::string template_package_name(const std::string& name, const std::string& version, const std::string& edition = "full");

struct TemplateSearch {
  std::vector<std::filesystem::path> command_paths;   // the locations the command names (--templates, repeatable)
  bool use_environment = true;                        // NXR_TEMPLATE_PATH
  std::optional<std::filesystem::path> settings_file; // the user settings file; default: user_settings_file()
};
// nxr's per-user settings file: $XDG_CONFIG_HOME/nxr/settings.json (~/.config/nxr/settings.json) on Linux, ~/Library/Application Support/nxr/settings.json on macOS,
// %APPDATA%\nxr\settings.json on Windows.
std::filesystem::path user_settings_file();
// The folders to look in, in order, without the hint.
std::vector<std::filesystem::path> template_search_path(const TemplateSearch& search);

struct ResolvedTemplate {
  std::filesystem::path folder;
  PackageRecord record;
};
// The released package named by `ref` whose content hash is the reference's. Refused, listing every location searched and what was found there (a name and version
// with another hash is named with both hashes), when there is none.
ResolvedTemplate resolve_template(const TemplateRef& ref, const TemplateSearch& search);

struct TemplateListing {
  std::filesystem::path folder;
  std::string name, version, edition, kind, content_hash;
};
// Every released template and registration found on the search path.
std::vector<TemplateListing> list_templates(const TemplateSearch& search);

struct ReleaseOptions {
  std::string name, version;                          // the releaser's: `name@version` must be new in `into` (or be the same hash again)
  std::string released_by;                            // the program and the releaser's stated name: recorded
  std::filesystem::path into;                         // the templates folder the released package is made in
};
struct ReleaseReport {
  std::filesystem::path folder;
  std::string content_hash;
  Json left_out = Json::array();                      // the redistributable edition: what it does not keep, "<table> <path>"
};
// Release a template (or a registration) package: a frozen copy in `into`. The working package is not changed.
ReleaseReport release_package(const std::filesystem::path& package, const ReleaseOptions& opt);
// The redistributable edition of a released full edition, in `into` (the same name and version).
ReleaseReport release_redistributable(const std::filesystem::path& full_edition, const std::filesystem::path& into, const std::string& released_by);

// The id of a manifold of a released template, by name.
std::string template_manifold_id(const ResolvedTemplate& tmpl, const std::string& manifold_name);

// ── a registration between two templates ───────────────────────────────────────────────────────────────────────
struct RegistrationIds {
  std::string from_manifold_id, to_manifold_id;       // the two templates' manifolds the operators map between (anchored in the package)
  std::string subject_name;
};
struct RegistrationSpec {
  std::string name;                                   // the package's subject name
  TemplateRef from, to;                               // the two templates, by name, edition, version and hash
  std::string from_manifold, to_manifold;             // the manifolds (by name, e.g. each template's registration sphere) the operators map between
  TemplateSearch search;
  std::string method;                                 // what the operators are (e.g. "nearest vertex on the registered spheres"): required
  std::vector<std::string> warnings;                  // what it cannot resolve, e.g. that the correspondence is the spheres', not anatomy's
  OriginRecord origin;                                // every item of the package carries it
  CreatePackageOptions package;                       // its home dataset (provenance); its subject kind is set here
};
// CREATE a registration package at `path`: the two templates' manifolds anchored, `write_operators` writes the operators (the resampling matrices, each
// `from_manifold_id` on one template and `to_manifold_id` on the other, one per direction it supports), then it is closed with its record (both templates, the method,
// the warnings, the origin of every item).
std::filesystem::path create_registration_package(const std::filesystem::path& path, const RegistrationSpec& spec,
                                                  const std::function<void(SubjectWriter&, const RegistrationIds&)>& write_operators);

}  // namespace nxr::io

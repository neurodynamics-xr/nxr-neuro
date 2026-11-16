// nxr-pipelines — a dataset made together with its default subject: the composition `dataset create --templates` asks for.
//
//   io        create_dataset, create_default_subject (the surfaces and parcellations, as any subject's)
//   pipelines frames_compute (the pial's frames), group_tree_compute (the group tree), and with `modes` eigen_solve (the pial's Hodge–Dirac basis, solved here, never read)
//
// Both or neither: a failure removes the dataset this call made. The command line only parses and prints.
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/default_subject.h"
#include "nxr/io/writer.h"

namespace nxr::pipelines {

struct DatasetWithTemplateOptions {
  nxr::io::CreateOptions dataset;                 // the dataset's own options
  nxr::io::DefaultSubjectOptions subject;        // the FreeSurfer subjects folder and the resolution
  int atlas_depth = 8;                            // the group tree's default leaf depth of the dataset's atlases
  std::vector<int> lc_levels = {4, 6};            // the levels of the Levi-Civita connection
  std::optional<int> modes;                       // solve the pial's Hodge–Dirac basis with about this many modes (`eigen solve --modes`); none: no basis
};

// Returns {dataset, id, database, default_subject: {…, frames, group_tree, modes}} (modes null when not asked for). Progress lines go to `progress`.
nxr::io::Json create_dataset_with_template(const std::filesystem::path& datastore, const std::string& name, const DatasetWithTemplateOptions& opt, const std::string& created_by,
                                           const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

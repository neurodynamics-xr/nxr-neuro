// nxr-pipelines — a dataset made together with its default subject: the composition `dataset create --templates` asks for.
//
//   io        create_dataset, create_default_subject (the surfaces and parcellations, as any subject's)
//   pipelines frames_compute (the pial's frames), `select_dyadic_surface` rule group (the group tree) and the Levi-Civita connection as a `measure` over a level x level product, and with `modes` eigen_solve (the pial's Hodge–Dirac basis, solved here, never read)
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
  std::string atlas_finest = "parts 256";         // the finest tile of the group tree (the dataset's atlas leaves) as a quantity: parts per hemisphere (256 = 2^8), a tile diameter (mm) or area (mm2)
  std::vector<std::string> lc_scales = {"parts 16", "parts 64"};   // the tile scales (quantities) at which the Levi-Civita connection between neighbouring tiles is measured
  std::string frame_rate = "36 frames/cycle";     // the default frame density recorded in the atlas definitions (frames per cycle of the band's centre; the origin is a parameter of `select_time_grid`)
  std::optional<int> modes;                       // solve the pial's Hodge–Dirac basis with about this many modes (`eigen solve --modes`); none: no basis
};

// Returns {dataset, id, database, default_subject: {…, frames, group_tree, levi_civita, modes}} (modes null when not asked for). Progress lines go to `progress`.
nxr::io::Json create_dataset_with_template(const std::filesystem::path& datastore, const std::string& name, const DatasetWithTemplateOptions& opt, const std::string& created_by,
                                           const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines

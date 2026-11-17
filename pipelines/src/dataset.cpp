// nxr-pipelines — a dataset made together with its default subject (see dataset.h).
#include "nxr/pipelines/dataset.h"
#include "nxr/pipelines/runner.h"

#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/tiles.h"

namespace nxr::pipelines {
using namespace nxr::io;

Json create_dataset_with_template(const std::filesystem::path& datastore, const std::string& name, const DatasetWithTemplateOptions& opt, const std::string& created_by,
                                  const std::function<void(const Json&)>& progress) {
  if (opt.modes && *opt.modes < 1) throw Error(code::invalid_argument, "modes is a whole number of at least 1");
  CreateOptions co = opt.dataset;
  co.created_by = created_by;
  DatasetWriter ds = create_dataset(datastore, name, co);
  Json out = Json{{"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"database", (ds.folder() / kDatasetFile).string()}};
  try {
    out["default_subject"] = create_default_subject(ds, opt.subject, created_by);
    OpenSubjectOptions oo;
    oo.created_by = created_by;
    SubjectWriter dsub = SubjectWriter::open(ds, kDefaultSubject, oo);
    // the canonical frames (north / west, from the registered sphere) and the anatomical frame from FreeSurfer's surface RAS axes (the template every subject's frame is carried from)
    {
      graph::RunOptions ro;
      const auto runs = graph::run(ds, dsub, graph::load("frames"), Json{{"surface", "cortex_pial"}, {"convention", "ras"}}, ro, false, progress);
      out["default_subject"]["frames"] = runs.empty() ? Json() : runs.front();
    }
    // the group tree: it names the frames, and carries the Levi-Civita connection between its tile centres
    GroupTreeOptions g;
    g.resolution = opt.subject.default_resolution;
    g.atlas_depth = opt.atlas_depth;
    g.lc_levels = opt.lc_levels;
    out["default_subject"]["group_tree"] = group_tree_compute(ds, dsub, g, progress).info;
    out["default_subject"]["modes"] = nullptr;
    if (opt.modes) {
      EigenSolveOptions eo;
      eo.modes = *opt.modes;
      eo.surface = "cortex_pial";
      out["default_subject"]["modes"] = eigen_solve(ds, dsub, eo, progress).info;
    }
  } catch (...) {
    ds.close();
    delete_dataset(datastore, name);
    throw;
  }
  write_catalog(datastore);                     // its subject count
  ds.close();
  return out;
}

}  // namespace nxr::pipelines

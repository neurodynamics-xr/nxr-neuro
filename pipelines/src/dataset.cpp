// nxr-pipelines — a dataset made together with its default subject (see dataset.h).
#include "nxr/pipelines/dataset.h"
#include "nxr/pipelines/runner.h"

#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/run_shipped.h"

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
      const auto runs = graph::run(ds, dsub, graph::load("surface_frames"), Json{{"surface", "cortex_pial"}, {"convention", "ras"}}, ro, false, progress);
      out["default_subject"]["frames"] = runs.empty() ? Json() : runs.front();
    }
    // the group tree: the default subject's grid (`select_dyadic_surface`, rule group) with the atlas definitions, its finest tile and its default frame density stated as quantities
    {
      const auto runs = run_shipped(ds, dsub, "select_dyadic_surface",
                                    Json{{"surface", "cortex_pial"}, {"rule", "group"}, {"resolution", opt.subject.default_resolution}, {"finest", opt.atlas_finest}, {"atlas_finest", opt.atlas_finest}, {"frame_rate", opt.frame_rate}},
                                    false, progress);
      out["default_subject"]["group_tree"] = runs.empty() ? Json() : runs.front();
    }
    // the Levi-Civita connection between neighbouring tiles: a measurement over the product of a level with itself (rule 26), one per stated scale, of the pial's canonical frames
    {
      Json lc = Json::array();
      for (const std::string& scale : opt.lc_scales) {
        const auto runs = run_shipped(ds, dsub, "measure",
                                      Json{{"field", "cortex_pial_canonical_frames"}, {"make", Json{{"node", "select.product"}, {"factors", Json::array({Json{{"grid", "surface"}, {"level", scale}}, Json{{"grid", "surface"}, {"level", scale}}})}}},
                                           {"measures", Json::array({"levi_civita"})}},
                                      false, progress);
        lc.push_back(runs.empty() ? Json() : runs.front());
      }
      out["default_subject"]["levi_civita"] = lc;
    }
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

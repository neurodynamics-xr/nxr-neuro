// nxr-pipelines — a SHIPPED DEFINITION run on a subject by the library's own callers (a pipeline that needs a surface's tile ladder first, `dataset create` making the group tree): the
// same plan, run and store as `nxr-neuro run NAME`, through the runner. For the definitions with no recording (scope subject), so no span is asked.
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"

namespace nxr::pipelines {

/** the definition `name` (one of nxr-neuro's) on `sub` with the parameters `set` (a JSON object, NAME -> value); `replace` moves/replaces what another job stored under the same names.
 *  Returns the run's reports ({pipeline, hit, stored, ...}, one per run). Errors are the runner's. */
inline std::vector<nxr::io::Json> run_shipped(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& name, const nxr::io::Json& set, bool replace,
                                              const std::function<void(const nxr::io::Json&)>& progress = {}, const std::string& preset = std::string()) {
  const graph::Definition d = graph::load(name);
  graph::RunOptions o;
  o.replace = replace;
  o.preset = preset;                           // a named parameter set of the definition (`measure --set cortical_band_pyramid`): provenance only; the key is the resolved parameters
  return graph::run(ds, sub, d, set, o, false, progress);
}

}  // namespace nxr::pipelines

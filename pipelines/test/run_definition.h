// Test helper: one of nxr-neuro's own definitions run on a subject (what `nxr-neuro run <name>` does), the parameters as JSON. For the tests that make their fixtures with a pipeline.
#pragma once
#include <vector>

#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"

namespace nxrtest {
inline std::vector<nxr::io::Json> run_definition(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& name, const nxr::io::Json& set = nxr::io::Json::object(), bool replace = false,
                                                 const char* chunks = "all") {
  nxr::pipelines::graph::RunOptions o;
  o.chunks = chunks;
  o.replace = replace;
  return nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load(name), set, o);
}
/** a definition run under a PRESET (`nxr-neuro run measure --set <preset>`), the parameters as JSON */
inline std::vector<nxr::io::Json> run_preset(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& name, const std::string& preset, const nxr::io::Json& set = nxr::io::Json::object(),
                                             bool replace = false, const char* chunks = "all") {
  nxr::pipelines::graph::RunOptions o;
  o.chunks = chunks;
  o.replace = replace;
  o.preset = preset;
  return nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load(name), set, o);
}
}  // namespace nxrtest

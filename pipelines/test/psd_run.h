// A test helper: the sensor PSD (and the rotary spectrum) of a recording, made by the RUNNER on the shipped definitions `psd` and `rotary_psd` (the hand-wired psd_compute is gone). A test that only
// needs a PSD in its fixture calls `run_psd`; the stored statistic is read back with `psd_values`.
#pragma once
#include <cstring>
#include <string>
#include <vector>

#include "nxr/io/array_reader.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"

namespace nxrtest {

struct PsdRun {
  bool hit = false;
  nxr::io::Json report;   // the run's report: {pipeline, stored: [{output, name, path, id, partition?}], hit, ...}
  std::string path;       // the stored statistic's store path
  std::string name;
};

/** the definition `pipeline` (psd | rotary_psd) on `recording`, the whole recording (every chunk); `set` overrides parameters (`{"segment_level": 8}`), `preset` is `--set NAME` */
inline PsdRun run_definition(const std::string& pipeline, nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& recording, nxr::io::Json set = nxr::io::Json::object(),
                             const std::string& preset = "", bool replace = false, std::optional<std::string> chunks = std::string("all")) {
  const auto d = nxr::pipelines::graph::load(pipeline);
  set["recording"] = recording;
  nxr::pipelines::graph::RunOptions opt;
  opt.chunks = chunks;
  opt.replace = replace;
  opt.preset = preset;
  const auto r = nxr::pipelines::graph::run(ds, sub, d, set, opt);
  PsdRun out;
  out.report = r.at(0);
  out.hit = out.report["hit"].get<bool>();
  for (const auto& s : out.report["stored"]) if (s["output"] == "psd") { out.path = s.value("path", std::string()); out.name = s.value("name", std::string()); }
  return out;
}
inline PsdRun run_psd(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::string& recording, nxr::io::Json set = nxr::io::Json::object(), const std::string& preset = "", bool replace = false) {
  return run_definition("psd", ds, sub, recording, std::move(set), preset, replace);
}
/** a stored statistic's values (float32) and shape */
inline std::vector<float> psd_values(nxr::io::SubjectWriter& sub, const PsdRun& r, std::vector<std::int64_t>* shape = nullptr) {
  nxr::io::ArrayReader a = nxr::io::ArrayReader::open(sub.store(), r.path);
  if (shape) *shape = a.shape();
  const auto raw = a.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}

}  // namespace nxrtest

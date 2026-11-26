// Every shipped definition LOADS (parses, resolves its steps and ports, passes the definition checker): each file is loaded on its own and every failure is printed, not only the first. Synthetic only: no dataset.
#include <cstdio>
#include <filesystem>
#include <string>

#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/registry.h"

namespace fs = std::filesystem;

int main() {
  int failures = 0, loaded = 0;
  for (const auto& e : fs::directory_iterator(NXR_TEST_DEFINITIONS_DIR)) {
    if (e.path().extension() != ".yaml") continue;
    const std::string name = e.path().stem().string();
    try {
      (void)nxr::pipelines::graph::load(name);
      ++loaded;
    } catch (const std::exception& ex) {
      std::printf("  %s: %s\n", name.c_str(), ex.what());
      ++failures;
    }
  }
  // source_flow given stored chunk spectra (--spectra): it reads them as connectome_flow does, streams no recording and stores no spectrum again (storage plan G8)
  try {
    namespace np = nxr::pipelines::graph;
    const np::Definition d0 = np::load("source_flow");
    nxr::graph::Json params = nxr::graph::Json::object();
    for (const auto& [k, v] : d0.parameters.items()) params[k] = v.is_object() && v.contains("default") ? v["default"] : v;
    params["spectra"] = "rest_helmet_chunk_spectra";
    params["columns"] = "sensor_coordinates";
    const np::Definition d = np::expand(d0, params);
    const bool reads = d.inputs.contains("spectra") && d.inputs["spectra"].contains("statistic");
    const bool streams = !d.do_.empty() || d.over.has_value();
    bool after_reads = false;
    for (const auto& u : d.after) if (u.step == "signal.cortical_flow" && u.ins.contains("spectra") && u.ins["spectra"] == "spectra") after_reads = true;
    if (!reads || streams || d.outputs.contains("spectra") || !after_reads) {
      std::printf("  source_flow --spectra: reads the stored spectra %d, streams the recording %d, stores spectra %d, the flow reads them %d\n", reads, streams, d.outputs.contains("spectra") ? 1 : 0, after_reads);
      ++failures;
    }
    params["spectra"] = nullptr;
    const np::Definition r = np::expand(d0, params);
    if (r.do_.empty() || !r.outputs.contains("spectra") || r.inputs.contains("spectra")) { std::printf("  source_flow without --spectra: the recording is no longer streamed\n"); ++failures; }
    // the mode operators are stored (storage plan G9): option B's, per hemisphere, from the channels to each hemisphere's eigen line, derived from the kernel
    params["columns"] = "channels";
    const np::Definition b = np::expand(d0, params);
    for (const char* o : {"left_operator", "right_operator"})
      if (!b.outputs.contains(o) || b.outputs[o].value("kind", std::string()) != "operator" || b.outputs[o].value("derived_from", std::string()) != "kernel") { std::printf("  source_flow: %s is not stored\n", o); ++failures; }
    // each hemisphere's measurements name the manifold their rows lie on, with no suffix on the measures (issue #264)
    if (b.outputs["left_whole_vertices"].value("rows", std::string()) != "left.to" || b.outputs["right_whole_modes"].value("rows", std::string()) != "right.from") { std::printf("  source_flow: the rows' manifold is not named\n"); ++failures; }
    for (const auto& u : b.after) if (u.with.contains("measure_suffix")) { std::printf("  source_flow: %s still suffixes its measures\n", u.step.c_str()); ++failures; }
    if (b.outputs.contains("glued_operator")) { std::printf("  source_flow per hemisphere stores a glued operator\n"); ++failures; }
  } catch (const std::exception& ex) {
    std::printf("  source_flow --spectra: %s\n", ex.what());
    ++failures;
  }
  std::printf("  %d definitions load, %d do not\n", loaded, failures);
  std::printf("%s test_definitions_load\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}

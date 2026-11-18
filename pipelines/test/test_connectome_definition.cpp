// The connectome's drafted definitions (pipelines/definitions/pending/connectome*.yaml) against the registered nodes: every step names a registered node, wires exactly its ports
// to earlier values, sets only its parameters, makes as many values as it has outputs; every node has a signature whose ports are the node's; the declared outputs are made by a
// step; the three required parameters are `required`; the template definition carries the bundle's origin; the defaults are the spec's (smoothing 3 at the vertex level, 0 for a
// partition; the gate 5 mm). Synthetic: no dataset. The run through `run connectome` waits for the runner's declared inputs (see the draft's "Needs").
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

#include "nxr/graph/signature.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/registry.h"

namespace ng = nxr::graph;
namespace np = nxr::pipelines::graph;
using ng::Json;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  CHECK failed: %s @ %d\n", #c, __LINE__); ++failures; } } while (0)

static Json read(const std::string& name) {
  std::string p = std::string(NXR_DEFINITIONS) + "/" + name + ".yaml";       // the shipped definitions first, then the drafts still pending
  if (!std::ifstream(p)) p = std::string(NXR_PENDING_DEFINITIONS) + "/" + name + ".yaml";
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return nxr::io::parse_yaml(ss.str(), p);
}

int main() {
  for (const char* name : {"connectome", "connectome_template", "connectome_vertices"}) {
    const Json d = read(name);
    CHECK(d["pipeline"] == name);
    std::set<std::string> known;
    for (const auto& [k, v] : d["inputs"].items()) { (void)v; known.insert(k); }
    for (const auto& [k, v] : d["parameters"].items()) {
      CHECK(v.is_object() && v.contains("meaning") && v.contains("unit"));
      if (k == "fibers" || k == "end_surface" || k == "over") CHECK(v["default"] == "required");   // a choice of data: refused when missing
    }
    for (const auto& s : d["steps"]["before"]) {
      const np::StepSpec& st = np::step(s["node"].get<std::string>());
      std::set<std::string> given;
      for (const auto& [port, thing] : s["in"].items()) {
        given.insert(port);
        CHECK(std::find(st.inputs.begin(), st.inputs.end(), port) != st.inputs.end());
        const std::string t = thing.get<std::string>();
        CHECK(known.count(t.substr(0, t.find('.'))) == 1);                  // an input or an earlier step's output
      }
      for (const auto& p : st.inputs) CHECK(given.count(p) == 1 || std::find(st.optional.begin(), st.optional.end(), p) != st.optional.end());
      if (s.contains("with"))
        for (const auto& [p, v] : s["with"].items()) { (void)v; CHECK(st.parameters.contains(p)); }
      const Json out = s["out"];
      const std::size_t n = out.is_array() ? out.size() : 1;
      CHECK(n == st.outputs.size());
      if (out.is_array()) for (const auto& o : out) known.insert(o.get<std::string>()); else known.insert(out.get<std::string>());
      const ng::Signature* sg = ng::builtin_signature(st.name);
      CHECK(sg != nullptr);
      if (sg) {
        std::vector<std::string> in_sig, out_sig;
        for (const auto& [port, pat] : sg->inputs) { in_sig.push_back(port); CHECK(pat.is_optional == (std::find(st.optional.begin(), st.optional.end(), port) != st.optional.end())); }
        for (const auto& [port, _] : sg->outputs) out_sig.push_back(port);
        CHECK(std::set<std::string>(in_sig.begin(), in_sig.end()) == std::set<std::string>(st.inputs.begin(), st.inputs.end()));
        CHECK(out_sig == st.outputs);
      }
    }
    for (const auto& [o, spec] : d["outputs"].items()) { CHECK(known.count(o) == 1); CHECK(spec.contains("kind")); }
  }
  const Json c = read("connectome"), t = read("connectome_template"), v = read("connectome_vertices");
  CHECK(c["parameters"]["smooth_hops"]["default"] == 0);                    // partitions: plain counts by default
  CHECK(t["parameters"]["smooth_hops"]["default"] == 0);
  CHECK(v["parameters"]["smooth_hops"]["default"] == 3);                    // the vertex level: the reference's
  CHECK(c["parameters"]["max_end_distance"]["default"] == 5);
  CHECK(c["outputs"]["counts"]["kind"] == "measurement" && c["outputs"]["counts"]["codes"] == 2);
  CHECK(t["outputs"]["counts"]["carry_origin"] == "fibers");                // the HCP-1065 licence and attribution travel into the result
  CHECK(!c["parameters"].contains("prune_below") && !t["parameters"].contains("prune_below") && v["parameters"].contains("prune_below"));   // materialised only
  if (failures) { std::printf("test_connectome_definition: %d failure(s)\n", failures); return 1; }
  std::printf("test_connectome_definition: ok\n");
  return 0;
}

// `compile` (nxr/pipelines/face.h): a definition expanded and type-checked without running it, on a small synthetic dataset made in a temp directory.
//   1. a shipped definition (psd) compiled against the subject's records: the compiled graph is the graph the runner checks (the same types, the same halo), the bound and
//      streamed inputs are named, and nothing is written (the subject's database is byte for byte unchanged);
//   2. a type error is refused before any data with the checker's message;
//   3. an external sink declared by its signature is checked with the rest: a complex field into a real-only sink is refused, the same field through the magnitude form is
//      accepted; a node neither registered nor declared is refused;
//   4. without a dataset the inputs' types are the caller's (the structured JSON form) and the graph is checked by the signatures alone;
//   5. the command `nxr-neuro compile` prints the same compiled graph;
//   6. stored inputs typed from their records (issues 157, 158): a per-vertex field stored under a kind outside the quantity vocabulary keeps its manifold, value type,
//      components and unit (only its quantity is unset), so an external sink wanting a real field on a surface S and the manifold S accepts it; the stored surface reaches the
//      sink's manifold port as `manifold<...>` with its id, kind and size; a field on another surface is refused; a field of a known kind keeps its quantity.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

#include "check.h"
#include "nxr/graph/type_json.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/errors.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace ng = nxr::graph;
using face::Json;

namespace {
std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const nxr::io::Error& e) { return std::string(e.code()) + ": " + e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
std::string bytes_of(const fs::path& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
void write(const fs::path& p, const std::string& text) { std::ofstream(p) << text; }

// a chain that draws a band's analytic signal, or its magnitude, through a sink another program implements
std::string trace_definition(bool magnitude) {
  return std::string(R"(pipeline: trace_check
version: 1
description: a band's analytic signal handed to an external trace sink
scope: subject
inputs:
  recording: {field: recording, each: true, where: {name: $recording}}
parameters:
  recording: {default: none, meaning: "the recording's name", unit: "-"}
steps:
  chunks:
    over: recording
    do:
      - {node: signal.analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}
)") + (magnitude ? "      - {node: signal.field_form, in: {x: z}, with: {form: magnitude}, out: a}\n      - {node: scope.trace, in: {field: a}, with: {colour: red}}\n"
                 : "      - {node: scope.trace, in: {field: z}, with: {colour: red}}\n") +
         R"(outputs:
  z: {kind: recording, name: "{recording}_trace"}
)";
}
const char* kSink = R"({"nodes": [{"name": "scope.trace", "kind": "chunk", "move": "pointwise",
  "inputs": [{"port": "field", "class": "field", "on": ["M", {"form": "variable", "key": "T", "unit": "s"}], "streamed": true, "value": "real", "components": 1}],
  "parameters": [{"name": "colour", "default": "black", "meaning": "the trace's colour", "unit": "-"}]}]})";

// a sink drawing a real field on the vertices of a surface S, given S itself (a renderer's signature: declared, never run here)
const char* kSurfaceSink = R"({"nodes": [{"name": "scope.surface", "kind": "array", "move": "pointwise",
  "inputs": [{"port": "field", "class": "field", "on": [{"form": "variable", "key": "S", "kind": "surface"}], "value": ["real"], "components": 1},
             {"port": "surface", "class": "manifold", "on": ["S"]}],
  "outputs": [{"port": "shown", "like": "field"}], "parameters": []}]})";
// the same sink with no output (a renderer draws; it makes nothing), and a definition that ends in it and stores nothing (issue 159)
const char* kSurfaceSinkOnly = R"({"nodes": [{"name": "scope.surface", "kind": "array", "move": "pointwise",
  "inputs": [{"port": "field", "class": "field", "on": [{"form": "variable", "key": "S", "kind": "surface"}], "value": ["real"], "components": 1},
             {"port": "surface", "class": "manifold", "on": ["S"]}],
  "parameters": []}]})";
const char* kSinkOnlyDefinition = R"(pipeline: surface_draw
version: 1
description: a stored field drawn on a stored surface by an external sink; nothing is stored
scope: subject
inputs:
  map: {field: any, where: {name: $map}}
  surface: {manifold: surface, where: {name: $surface}}
parameters:
  map: {default: none, meaning: "the stored field's name", unit: "-"}
  surface: {default: none, meaning: "the surface's name", unit: "-"}
steps:
  before:
    - {node: scope.surface, in: {field: map, surface: surface}}
)";
const char* kSurfaceDefinition = R"(pipeline: surface_check
version: 1
description: a stored field drawn on a stored surface by an external sink
scope: subject
inputs:
  map: {field: any, where: {name: $map}}
  surface: {manifold: surface, where: {name: $surface}}
parameters:
  map: {default: none, meaning: "the stored field's name", unit: "-"}
  surface: {default: none, meaning: "the surface's name", unit: "-"}
steps:
  before:
    - {node: scope.surface, in: {field: map, surface: surface}, out: shown}
outputs:
  shown: {kind: field, kind_name: map, name: shown, on: map.manifold.0}
)";
// an octahedron, scaled
void octahedron(double r, std::vector<double>& V, std::vector<std::int32_t>& F) {
  V = {r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r};
  F = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_compile_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  std::map<std::string, std::string> before;
  for (const auto& e : fs::recursive_directory_iterator(folder)) if (e.is_regular_file()) before[e.path().string()] = bytes_of(e.path());

  // 1. psd against the subject's records: the graph the runner checks
  face::CompileRequest rq;
  rq.definition = "psd";
  rq.dataset = folder.string();
  rq.subject = "S1";
  rq.parameters = {{"recording", "rest"}};
  const Json r = face::compile(rq).result;
  CHECK_EQ(r["types"].get<std::string>(), std::string("stored"));
  CHECK_EQ(r["plans"].size(), std::size_t(1));
  const Json& c = r["plans"][0]["graph"];
  {
    nxr::io::OpenOptions o;
    o.lock = false;
    nxr::io::DatasetWriter ds = nxr::io::open_writer(folder, o);
    nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(ds, "S1");
    const auto plans = graph::plan_runs(graph::load("psd"), sub, Json{{"recording", "rest"}});
    CHECK(plans.size() == 1 && plans[0].checked);
    if (!plans.empty() && plans[0].checked) {
      const ng::Checked& k = *plans[0].checked;
      CHECK(c["halo"] == k.halo && c["align"] == k.align);
      for (const auto& [name, t] : k.values) CHECK(c["values"][name] == ng::to_json(t));   // every value's type, as the runner checks it
    }
  }
  CHECK(c["nodes"].size() == 1 && c["nodes"][0]["node"] == "signal.psd" && c["nodes"][0]["section"] == "chunks");
  CHECK(c["nodes"][0]["parameters"]["segment_seconds"] == 4.0);                       // resolved: the definition's default
  CHECK(c["values"]["recording"]["streamed"] == true && c["values"]["recording"]["sfreq"] == 100.0);
  bool recording_to_x = false;
  for (const auto& e : c["edges"]) recording_to_x = recording_to_x || (e["from"].value("input", "") == "recording" && e["to"]["port"] == "x" && e["to"]["node"] == 0);
  CHECK(recording_to_x);
  CHECK(c["outputs"][0]["name"] == "psd");
  CHECK(r["plans"][0]["streams"].size() == 1 && r["plans"][0]["streams"][0]["input"] == "recording");
  bool bad_bound = false;
  for (const auto& b : r["plans"][0]["bind"]) bad_bound = bad_bound || (b["input"] == "bad" && b["name"] == "rest_bad_segments");
  CHECK(bad_bound);
  // nothing written
  for (const auto& e : fs::recursive_directory_iterator(folder))
    if (e.is_regular_file() && !(before.count(e.path().string()) && before.at(e.path().string()) == bytes_of(e.path()))) { std::printf("  changed: %s\n", e.path().c_str()); CHECK(false); }

  // 2. a type error, refused with the checker's message: the band power of a real field
  write(root / "power_of_real.yaml", std::string(R"(pipeline: power_of_real
kind: chain
version: 1
description: the band power of the recording itself, a type error since it is real
scope: subject
inputs:
  recording: {field: recording, each: true, where: {name: $recording}}
parameters:
  recording: {default: none, meaning: "the recording's name", unit: "-"}
steps:
  chunks:
    over: recording
    do:
      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}
outputs:
  p: {kind: field, kind_name: power, name: "{recording}_p", on: recording.manifold.0, derived_from: recording}
)"));
  face::CompileRequest bad = rq;
  bad.definition = (root / "power_of_real.yaml").string();
  bad.parameters = {{"recording", "rest"}};
  const std::string why = refusal([&] { face::compile(bad); });
  if (!(has(why, "invalid_argument") && has(why, "chunks[0] signal.field_form") && has(why, "is real; the band power"))) { std::printf("  refusal: %s\n", why.c_str()); CHECK(false); }

  // 3. an external sink: the complex analytic signal refused, its magnitude accepted; undeclared, the node is unknown
  write(root / "trace_z.yaml", trace_definition(false));
  write(root / "trace_a.yaml", trace_definition(true));
  face::CompileRequest sink = rq;
  sink.signatures = Json::parse(kSink);
  sink.definition = (root / "trace_z.yaml").string();
  const std::string complex_in = refusal([&] { face::compile(sink); });
  if (!(has(complex_in, "scope.trace") && has(complex_in, "port `field`") && has(complex_in, "is complex; this node reads real"))) { std::printf("  refusal: %s\n", complex_in.c_str()); CHECK(false); }
  sink.definition = (root / "trace_a.yaml").string();
  Json accepted;
  if (const std::string w = refusal([&] { accepted = face::compile(sink).result; }); !w.empty()) { std::printf("  refused: %s\n", w.c_str()); CHECK(false); }
  if (!accepted.is_null()) {
    const Json& g = accepted["plans"][0]["graph"];
    CHECK(g["nodes"].size() == 3 && g["nodes"][2]["node"] == "scope.trace" && g["nodes"][2]["parameters"]["colour"] == "red");
    CHECK(g["values"]["a"]["value"] == "real" && g["values"]["a"]["quantity"] == "amplitude" && g["values"]["z"]["value"] == "complex");
    CHECK(g["edges"][2]["from"]["node"] == 1 && g["edges"][2]["to"]["node"] == 2);
    CHECK(g["halo"].is_number());
  }
  face::CompileRequest undeclared = sink;
  undeclared.signatures = Json();
  CHECK(has(refusal([&] { face::compile(undeclared); }), "no pipeline step 'scope.trace'"));
  // a declared signature never replaces a registered node's
  face::CompileRequest shadow = sink;
  shadow.signatures = Json::parse(R"([{"name": "signal.analytic"}])");
  CHECK(has(refusal([&] { face::compile(shadow); }), "is a registered step"));

  // 4. without a dataset: the inputs' types given (structured), checked by the signatures alone
  face::CompileRequest given;
  given.definition = (root / "trace_a.yaml").string();
  given.parameters = {{"recording", "rest"}};
  given.signatures = Json::parse(kSink);
  given.inputs = {{"recording", Json{{"class", "field"}, {"value", "real"}, {"domain", "sensors(rest) × time(rest)"}, {"axis", "samples"}, {"sfreq", 100.0}, {"quantity", "recording"}}}};
  Json g2;
  if (const std::string w = refusal([&] { g2 = face::compile(given).result; }); !w.empty()) { std::printf("  refused: %s\n", w.c_str()); CHECK(false); }
  if (!g2.is_null()) {
    CHECK_EQ(g2["types"].get<std::string>(), std::string("given"));
    CHECK(g2["plans"][0]["graph"]["halo"].is_null() && g2["plans"][0]["graph"]["values"]["a"]["value"] == "real");
    CHECK(g2["plans"][0]["graph"]["values"]["z"]["resolution"]["band"] == Json::array({8.0, 12.0}));
  }
  face::CompileRequest none = given;
  none.inputs = Json::object();
  CHECK(has(refusal([&] { face::compile(none); }), "recording of trace_check have no type"));
  face::CompileRequest psd_given = given;
  psd_given.definition = "psd";
  psd_given.signatures = Json();
  CHECK(has(refusal([&] { face::compile(psd_given); }), "signal.psd: the node has no signature"));   // a node typed by its plan alone needs the records

  // 5. the command prints the same compiled graph
#ifdef NXR_TEST_CLI
  {
    write(root / "sink.json", kSink);
    const std::string out = (root / "compile.json").string();
    const std::string cmd = std::string(NXR_TEST_CLI) + " compile " + (root / "trace_a.yaml").string() + " --dataset " + folder.string() + " --subject S1 --set recording=rest --signatures " +
                            (root / "sink.json").string() + " --json > " + out;
    CHECK_EQ(std::system(cmd.c_str()), 0);
    std::ifstream f(out);
    std::string line, last;
    while (std::getline(f, line)) if (!line.empty()) last = line;
    const Json j = Json::parse(last, nullptr, false);
    CHECK(j.is_object() && j.value("stage", "") == "result");
    if (j.is_object() && !accepted.is_null()) CHECK(j["plans"][0]["graph"] == accepted["plans"][0]["graph"]);
    const std::string refused = std::string(NXR_TEST_CLI) + " compile " + (root / "trace_z.yaml").string() + " --dataset " + folder.string() + " --subject S1 --set recording=rest --signatures " +
                                (root / "sink.json").string() + " --json > /dev/null 2>&1";
    CHECK(std::system(refused.c_str()) != 0);
  }
#endif
  // 6. stored inputs typed from their records: a field of a kind outside the vocabulary, a stored surface (issues 157, 158)
  {
    std::string cortex_id, other_id;
    {
      nxr::io::DatasetWriter ds = nxr::io::open_writer(folder);
      nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(ds, "S1");
      std::vector<double> V, W;
      std::vector<std::int32_t> F, G;
      octahedron(0.08, V, F);
      octahedron(0.05, W, G);
      sub.compose([&] {
        cortex_id = nxr::io::add_surface(sub, {.name = "cortex"}, V, F).id;
        nxr::io::SurfaceOptions o;
        o.name = "inner"; o.primary = false;
        other_id = nxr::io::add_surface(sub, o, W, G).id;
        const auto subject = nxr::io::task::Subject::of(sub);
        auto field_on = [&](const std::string& name, const std::string& kind, const std::string& mid, const std::vector<float>& x, const char* unit) {
          nxr::io::FieldSpec f;
          f.name = name; f.path = subject.node_path(mid, name); f.kind = kind; f.manifold_id = mid;
          f.data = nxr::io::view(x, {static_cast<std::int64_t>(x.size())}); f.array.compress = false; f.cols.set("unit", unit);
          sub.field(std::move(f));
        };
        const std::vector<float> x6 = {1.0f, 1.2f, 0.9f, 1.4f, 1.1f, 1.3f};
        field_on("pet_suvr", "PET example_suvr", cortex_id, x6, "SUVR");      // a kind outside the vocabulary (an imported PET map)
        field_on("inner_suvr", "PET example_suvr", other_id, x6, "SUVR");     // the same, on another surface
        field_on("cortex_power", "power", cortex_id, x6, "W");                // a kind in the vocabulary
      });
    }
    face::CompileRequest sq = rq;
    write(root / "surface_check.yaml", kSurfaceDefinition);
    sq.definition = (root / "surface_check.yaml").string();
    sq.signatures = Json::parse(kSurfaceSink);
    sq.parameters = {{"map", "pet_suvr"}, {"surface", "cortex"}};
    Json ok;
    if (const std::string w = refusal([&] { ok = face::compile(sq).result; }); !w.empty()) { std::printf("  refused: %s\n", w.c_str()); CHECK(false); }
    if (!ok.is_null()) {
      const Json& v = ok["plans"][0]["graph"]["values"];
      const Json& m = v["map"];
      CHECK(m["class"] == "field" && m["value"] == "real" && m["components"] == 1 && m["unit"] == "SUVR" && m["typed"] == true);
      CHECK(!m.contains("quantity") || m["quantity"].is_null() || m["quantity"] == "");   // only the quantity is unset
      const Json& f = m["domain"]["factors"][0];
      CHECK(f["key"] == cortex_id && f["label"] == "cortex" && f["kind"] == "surface" && f["n"] == 6);
      const Json& s = v["surface"];
      CHECK(s["class"] == "manifold" && s["domain"]["factors"][0]["key"] == cortex_id && s["domain"]["factors"][0]["kind"] == "surface" && s["domain"]["factors"][0]["n"] == 6);
      CHECK(v["shown"]["domain"]["factors"][0]["key"] == cortex_id && v["shown"]["unit"] == "SUVR");
      if (m["unit"] != "SUVR" || s["class"] != "manifold") std::printf("  map: %s\n  surface: %s\n", m.dump().c_str(), s.dump().c_str());
    }
    // a field on another surface than the one given: the variable S meets two manifolds
    face::CompileRequest other = sq;
    other.parameters = {{"map", "inner_suvr"}, {"surface", "cortex"}};
    const std::string w2 = refusal([&] { face::compile(other); });
    if (!(has(w2, "scope.surface") && has(w2, "port `surface`"))) { std::printf("  refusal: %s\n", w2.c_str()); CHECK(false); }
    // a field of a known kind keeps its quantity
    face::CompileRequest known = sq;
    known.parameters = {{"map", "cortex_power"}, {"surface", "cortex"}};
    Json kr;
    if (const std::string w = refusal([&] { kr = face::compile(known).result; }); !w.empty()) { std::printf("  refused: %s\n", w.c_str()); CHECK(false); }
    if (!kr.is_null()) CHECK(kr["plans"][0]["graph"]["values"]["map"]["quantity"] == "power" && kr["plans"][0]["graph"]["values"]["map"]["unit"] == "W");
    // a definition ending in a sink declares no stored output: compile accepts it, run refuses it (nothing to store)
    face::CompileRequest draw = sq;
    write(root / "surface_draw.yaml", kSinkOnlyDefinition);
    draw.definition = (root / "surface_draw.yaml").string();
    draw.signatures = Json::parse(kSurfaceSinkOnly);
    Json dr;
    if (const std::string w = refusal([&] { dr = face::compile(draw).result; }); !w.empty()) { std::printf("  refused: %s\n", w.c_str()); CHECK(false); }
    if (!dr.is_null()) CHECK(dr["plans"][0]["graph"]["nodes"].size() == 1 && dr["plans"][0]["graph"]["nodes"][0]["node"] == "scope.surface");
    {
      const graph::ExternalNodes scope(ng::external_nodes(Json::parse(kSurfaceSinkOnly)));
      const graph::Definition d = graph::load((root / "surface_draw.yaml").string());
      CHECK(d.outputs.empty());
      nxr::io::DatasetWriter ds = nxr::io::open_writer(folder);
      nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(ds, "S1");
      const std::string w = refusal([&] { graph::run(ds, sub, d, Json{{"map", "pet_suvr"}, {"surface", "cortex"}}, graph::RunOptions{}, true); });
      if (!has(w, "stores nothing")) { std::printf("  run refusal: %s\n", w.c_str()); CHECK(false); }
    }
    // without a sink, a definition with no output is still refused
    write(root / "no_output.yaml", std::string(kSinkOnlyDefinition).substr(0, std::string(kSinkOnlyDefinition).find("steps:")) + "steps:\n  before: []\n");
    face::CompileRequest bare = draw;
    bare.definition = (root / "no_output.yaml").string();
    const std::string wb = refusal([&] { face::compile(bare); });
    if (!has(wb, "outputs: at least one")) { std::printf("  refusal: %s\n", wb.c_str()); CHECK(false); }
  }

  fs::remove_all(root);
  return nxrtest::finish("test_compile");
}

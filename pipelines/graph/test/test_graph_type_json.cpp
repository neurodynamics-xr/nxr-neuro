// The types as data (type_json.h): the structured JSON form of a type against the printed form (each printed type read back from its JSON prints the same, and a type
// survives JSON -> type -> JSON unchanged, kinds, sizes and coordinate systems included); external node signatures given as JSON, checked with the built-in nodes: a complex
// field into a real-only sink refused with the checker's message, the same field through the magnitude form accepted, a variable bound only to a surface, and a refused document.
// Synthetic types only: no dataset, no file.
#include <cstdio>
#include <functional>
#include <string>

#include "nxr/graph/graph.h"
#include "nxr/graph/type_json.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const std::invalid_argument& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

ng::Domain dom(std::vector<ng::ManifoldRef> f) { ng::Domain d; d.factors = std::move(f); return d; }

// a recording on a channel manifold in head/scs and its time line: streamed, real, 3 rows at 100 Hz
ng::PortType recording() {
  ng::PortType t;
  t.typed = true;
  t.cls = ng::PortClass::field;
  t.domain = dom({ng::manifold_id("sensors", "rest", ng::ManifoldKind::points, ng::CoordinateSystem{"head", "scs", "m"}, 3), ng::time_line("rest", 100.0)});
  t.quantity = "recording";
  t.axis = ng::Axis::samples;
  t.sfreq = 100.0;
  t.rows = 3;
  t.unit = "T";
  return t;
}

// the external nodes of the tests: a real, one-component trace sink inside the loop; a surface renderer (a field on a surface's vertices, and that surface)
const char* kSignatures = R"({"nodes": [
  {"name": "scope.trace", "kind": "chunk", "move": "pointwise",
   "inputs": [{"port": "field", "class": "field", "on": ["M", {"form": "variable", "key": "T", "unit": "s"}], "streamed": true, "value": "real", "components": 1}],
   "parameters": [{"name": "colour", "default": "black", "meaning": "the trace's colour", "unit": "-"}]},
  {"name": "render.surface", "kind": "array", "move": "pointwise",
   "inputs": [{"port": "field", "class": "field", "on": [{"form": "variable", "key": "S", "kind": "surface"}], "value": ["real"], "components": 1},
              {"port": "surface", "class": "manifold", "on": ["S"]}],
   "parameters": [{"name": "colormap", "default": "viridis", "meaning": "the colour map", "unit": "-", "choices": ["viridis", "magma"]}]}
]})";

}  // namespace

int main() {
  // 1. THE STRUCTURED FORM AND THE PRINTED FORM say the same: for the printed types of the spec, text -> type -> JSON -> type prints the text again
  for (const char* text : {"field<sensors(rest) × time(rest); complex; signal; band 8-12 Hz morlet>",
                           "measurement<members(cells_l5); amplitude; band 8-16 Hz>",
                           "selection<spans on time(rest); partition; detected over ladder@8-16>",
                           "statistic<summaries; axes sensors(rest); of amplitude band 8-16 Hz>",
                           "operator<sensors(rest)|channels(kernel) → cortex(kernel); real → vector3; inverse kernel>",
                           "field<cortex(kernel) × frames(tiling); complex vector3; current; non-linear>"}) {
    const ng::PortType t = ng::parse_type(text);
    const Json j = ng::to_json(t);
    const ng::PortType back = ng::port_type_from_json(j);
    if (ng::print(back) != ng::print(t)) std::printf("  %s\n  != %s\n", ng::print(back).c_str(), ng::print(t).c_str());
    CHECK(ng::print(back) == ng::print(t));
    CHECK(ng::to_json(back) == j);                                            // and the JSON again, member for member
    CHECK(ng::print(ng::port_type_from_json(Json(text))) == ng::print(t));    // a string is the printed form
    CHECK(j["text"] == ng::print(t));
  }
  // what the printed form leaves out (kinds, sizes, coordinate systems, the axis, the rate) survives the structured form
  {
    const ng::PortType t = recording();
    const Json j = ng::to_json(t);
    CHECK(j["domain"]["factors"][0]["coordinate_system"]["name"] == "scs");
    CHECK(j["domain"]["factors"][0]["n"] == 3);
    CHECK(j["domain"]["factors"][1]["topology"] == "path" && j["domain"]["factors"][1]["unit"] == "s" && j["domain"]["factors"][1]["kind"] == "line");
    CHECK(j["streamed"] == true && j["axis"] == "samples" && j["sfreq"] == 100.0 && j["rows"] == 3 && j["unit"] == "T");
    const ng::PortType back = ng::port_type_from_json(j);
    CHECK(ng::to_json(back) == j);
    CHECK(back.domain.factors[0]->cs && back.domain.factors[0]->cs->name == "scs" && back.domain.factors[1]->spacing && *back.domain.factors[1]->spacing == 0.01);
    CHECK(ng::same(back.domain, t.domain) && back.streamed());
  }
  // a malformed member is refused naming it
  CHECK(has(refusal([] { ng::port_type_from_json(Json{{"class", "field"}, {"value", "octonion"}}); }), "is not one of real, complex"));
  CHECK(has(refusal([] { ng::port_type_from_json(Json{{"class", "field"}, {"resolution", {{"band", Json::array({8.0})}}}}); }), "resolution.band: [low, high]"));

  // a line has no role (spec time-as-a-mesh): a manifold or a signature that still sends `role` is refused, and the refusal names the new form (no alias)
  {
    const std::string why = refusal([] { ng::manifold_from_json(Json{{"form", "id"}, {"key", "rest"}, {"label", "time"}, {"kind", "line"}, {"role", "time"}}); });
    CHECK(has(why, "has no role") && has(why, "`topology`") && has(why, "`unit`") && has(why, "`made_from`"));
    const std::string why2 = refusal([] { ng::manifold_from_json(Json{{"form", "variable"}, {"key", "T"}, {"role", "frames"}}); });
    CHECK(has(why2, "has no role"));
    // the new form: a variable that binds a line in seconds, a path; a made line carries what it was made from
    const ng::ManifoldRef v = ng::manifold_from_json(Json{{"form", "variable"}, {"key", "T"}, {"unit", "s"}, {"samples", true}, {"topology", "path"}});
    CHECK(v->unit == "s" && v->samples && v->topology == ng::Topology::path && ng::to_json(v)["topology"] == "path" && ng::to_json(v)["samples"] == true);
    const ng::ManifoldRef f = ng::manifold_from_json(Json{{"form", "id"}, {"key", "tiling"}, {"label", "frames"}, {"kind", "line"}, {"topology", "path"},
                                                         {"coordinate_system", {{"space", "time"}, {"name", "clock"}, {"unit", "s"}}}, {"made_from", {{"relation", "tiled"}, {"of", "time(rest)"}}}});
    CHECK(ng::is_tiled(*f) && ng::in_seconds(*f) && ng::to_json(f)["made_from"]["relation"] == "tiled" && ng::to_json(f)["made_from"]["of"]["key"] == "rest");
  }

  // 2. EXTERNAL SIGNATURES, checked with the built-in nodes
  const auto ext = ng::external_nodes(Json::parse(kSignatures));
  CHECK(ext.size() == 2 && ext[0].signature && ext[1].signature && ext[0].kind == ng::Kind::chunk && ext[0].traits.declared());
  auto lookup = [&](const std::string& n) -> const ng::Signature* {
    if (const ng::Signature* s = ng::builtin_signature(n)) return s;
    for (const auto& e : ext) if (e.name == n) return &*e.signature;
    return nullptr;
  };
  auto chain = [&](bool with_form) {
    Json uses = Json::array({Json{{"node", "signal.analytic"}, {"in", {{"x", "rec"}}}, {"with", {{"band", {8.0, 12.0}}}}, {"out", "z"}}});
    if (with_form) uses.push_back(Json{{"node", "signal.field_form"}, {"in", {{"x", "z"}}}, {"with", {{"form", "magnitude"}}}, {"out", "a"}});
    uses.push_back(Json{{"node", "scope.trace"}, {"in", {{"field", with_form ? "a" : "z"}}}, {"with", {{"colour", "red"}}}, {"out", Json::array()}});
    return ng::graph_from_json(Json{{"name", "trace"}, {"inputs", Json::array({"rec"})}, {"outputs", Json::array()}, {"chunks", uses}});
  };
  {
    const std::string why = refusal([&] { ng::type_check(chain(false), {{"rec", recording()}}, Json::object(), lookup); });
    CHECK(has(why, "chunks[1] scope.trace") && has(why, "port `field`") && has(why, "is complex; this node reads real"));
    // a node that reads a path (the analytic signal's zero-phase band) refuses a recording whose time is a cycle
    ng::PortType periodic = recording();
    periodic.domain.factors[1] = ng::time_line("rest", 100.0, ng::Topology::cycle);
    const std::string why_cycle = refusal([&] { ng::type_check(chain(true), {{"rec", periodic}}, Json::object(), lookup); });
    CHECK(has(why_cycle, "signal.analytic") && has(why_cycle, "a path was expected") && has(why_cycle, "a cycle"));
    const ng::TypedGraph ok = ng::type_check(chain(true), {{"rec", recording()}}, Json::object(), lookup);
    CHECK(ok.values.at("a").value == ng::ValueType::real && ok.values.at("a").quantity == "amplitude");
    const Json c = ng::compiled_json(chain(true), ok, Json::object(), lookup);
    CHECK(c["nodes"].size() == 3 && c["nodes"][2]["node"] == "scope.trace" && c["nodes"][2]["parameters"]["colour"] == "red");
    CHECK(c["edges"].size() == 3 && c["edges"][2]["from"]["node"] == 1 && c["edges"][2]["from"]["port"] == "y" && c["edges"][2]["to"]["port"] == "field");
    CHECK(c["values"]["a"]["value"] == "real" && c["halo"].is_null());
  }
  // a variable bound only to a surface: a field on the cortex and the cortex are accepted, a field on the sensors (points) is refused
  {
    auto surface_graph = [] {
      return ng::graph_from_json(Json{{"name", "map"}, {"inputs", Json::array({"map", "cortex"})}, {"outputs", Json::array()},
                                      {"after", Json::array({Json{{"node", "render.surface"}, {"in", {{"field", "map"}, {"surface", "cortex"}}}, {"out", Json::array()}}})}});
    };
    const ng::ManifoldRef cortex = ng::manifold_id("cortex", "white", ng::ManifoldKind::surface, ng::CoordinateSystem{"head", "scs", "m"}, 642);
    ng::PortType map = ng::parse_type("field<cortex(white); real; power>");
    map.domain = dom({cortex});
    ng::PortType surf;
    surf.typed = true; surf.cls = ng::PortClass::manifold; surf.domain = dom({cortex});
    CHECK(refusal([&] { ng::type_check(surface_graph(), {{"map", map}, {"cortex", surf}}, Json::object(), lookup); }).empty());
    ng::PortType on_sensors = map;
    on_sensors.domain = dom({ng::manifold_id("sensors", "rest", ng::ManifoldKind::points)});
    ng::PortType sensors = surf;
    sensors.domain = on_sensors.domain;
    CHECK(has(refusal([&] { ng::type_check(surface_graph(), {{"map", on_sensors}, {"cortex", sensors}}, Json::object(), lookup); }), "this node reads one on a surface"));
    // a field on a volume given with a surface: refused at the field's port, where the variable meets the volume, not at the surface's (issue 164)
    ng::PortType on_volume = map;
    on_volume.domain = dom({ng::manifold_id("mri", "t1", ng::ManifoldKind::volume)});
    const std::string wv = refusal([&] { ng::type_check(surface_graph(), {{"map", on_volume}, {"cortex", surf}}, Json::object(), lookup); });
    if (!(has(wv, "port `field`") && has(wv, "mri(t1) is a volume; this node reads one on a surface"))) { std::printf("  refusal: %s\n", wv.c_str()); CHECK(false); }
    ng::PortType vec = map;
    vec.value = ng::ValueType::vector3; vec.components = 3;
    CHECK(has(refusal([&] { ng::type_check(surface_graph(), {{"map", vec}, {"cortex", surf}}, Json::object(), lookup); }), "this node reads real"));
  }
  // the node refuses to run, and a malformed document is refused naming the place
  {
    const ng::NodeSpec& sink = ext[1];
    CHECK(has(refusal([&] { sink.run(std::any(), ng::Arrays{}); }), "never runs it"));
    CHECK(has(refusal([] { ng::external_nodes(Json::parse(R"({"nodes": [{"name": "x.y", "inputs": [{"port": "a", "class": "colour"}]}]})")); }), "external node x.y.inputs.a.class"));
    CHECK(has(refusal([] { ng::external_nodes(Json::parse(R"([{"name": "x.y"}, {"name": "x.y"}])")); }), "declared twice"));
    CHECK(has(refusal([] { ng::external_nodes(Json::parse(R"([{"name": "x.y", "outputs": [{"port": "y", "like": "nothing"}]}])")); }), "not an input port"));
  }

  if (failures) { std::printf("test_graph_type_json: %d failure(s)\n", failures); return 1; }
  std::printf("test_graph_type_json: ok\n");
  return 0;
}

// nxr-pipelines — THE GEOMETRY MAKERS AS REGISTERED STEPS: what a surface carries (its frames, ...), made through the run's bound services and handed back as described arrays
// the definition stores (pipelines/definitions/frames.yaml, ...).
//
//   surface_frames  setup  surface  ->  canonical (+ singular, gauge_angle) when computed here; angle, second, handed, reliable: the anatomical frame
#include <cmath>

#include "nxr/io/finder.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using nxr::io::text;

namespace {

[[noreturn]] void refuse(const std::string& m) { throw std::invalid_argument(m); }
Value of_cls(const char* cls) { Value v; v.cls = cls; return v; }
Array column(const std::vector<double>& x, const Json& describe) {
  Array a;
  a.rows = static_cast<std::int64_t>(x.size()); a.cols = 1;
  a.data = x;
  a.meta["describe"] = describe;
  return a;
}
Array picks(const std::vector<std::int32_t>& x, const std::string& description) {
  Array a;
  a.rows = static_cast<std::int64_t>(x.size()); a.cols = 1;
  for (const auto i : x) a.data.push_back(i);
  a.meta["describe"] = Json{{"description", description}};
  return a;
}

// ── frames ──────────────────────────────────────────────────────────────────────────────────────────────────────
// THE FRAMES OF A SURFACE (the hand-wired `frames compute` it replaces): the canonical frames (north e1, west e2, n) from the registered sphere `<surface>_sphere` when they are
// not stored yet, with their singular vertices and the angle of north in geometry-central's gauge (its gauge record with it); and the anatomical frame, from the surface's own axes
// in the stated convention or carried from a template subject's through the registered spheres: the angle from north to anterior, which way the second axis points (0 superior,
// 1 medial) and its sign, and the vertices where it is reliable.
struct FramesState { FramesComputeOptions o; };
StepSpec frames_step() {
  StepSpec s;
  s.name = "geometry.surface_frames";
  s.kind = "setup";
  s.inputs = {"surface"};
  s.outputs = {"canonical", "singular", "gauge_angle", "angle", "second", "handed", "reliable"};
  s.parameters = Json{{"template", nullptr}, {"template_surface", nullptr}, {"convention", "scs"}, {"side", "auto"}, {"min_projection", 0.34}, {"max_condition", 50.0}};
  s.matches = "Rheome sphereframe (canonical frames); nxr coords.py anatomical frame; compute nxr::geometry";
  s.description = "a surface's canonical frames (from its registered sphere) and its anatomical frame";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "positions in double, as stored");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    if (!ins.count("surface")) refuse("frames: surface is its input");
    const double mp = p.value("min_projection", 0.34);
    if (!(mp > 0 && mp < 1)) refuse("frames: min_projection is in (0, 1)");
    const std::string side = p.value("side", std::string("auto"));
    if (side != "auto" && side != "left" && side != "right") refuse("frames: side is auto, left or right");
    Values v;
    for (const char* o : {"canonical", "gauge_angle", "angle", "second", "handed"}) v[o] = of_cls("field");
    for (const char* o : {"singular", "reliable"}) v[o] = of_cls("selection");
    for (auto& [k, x] : v) x.manifolds = ins.at("surface").manifolds;
    return v;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    FramesState st;
    st.o.surface = text(*ins.at("surface").row, "name");
    if (p["template"].is_string()) st.o.templ = p["template"].get<std::string>();
    if (p["template_surface"].is_string()) st.o.template_surface = p["template_surface"].get<std::string>();
    st.o.convention = p.value("convention", std::string("scs"));
    st.o.side = p.value("side", std::string("auto"));
    st.o.min_projection = p.value("min_projection", 0.34);
    st.o.max_condition = p.value("max_condition", 50.0);
    return std::any(std::move(st));
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const FramesState& st = std::any_cast<const FramesState&>(state);
    const FramesMade m = ctx.bound.make_frames(st.o);
    Arrays out;
    if (!m.canonical.empty()) {
      Array c;
      c.rows = m.n; c.cols = 9;
      c.data = m.canonical;
      c.meta["describe"] = Json{{"value_type", "matrix3"}, {"shape", Json::array({m.n, 3, 3})}, {"precision", "float32"}, {"producer", m.canonical_producer},
                                {"description", "per vertex rows e1 (north), e2 = n x e1 (west), n (outward); NaN at singular vertices"}};
      out["canonical"] = std::move(c);
      out["singular"] = picks(m.singular, "the vertices with no canonical frame: the poles, and ill-conditioned push-forwards");
      Json gp = m.gauge.is_object() ? m.gauge : Json::object();
      gp["spin"] = 0; gp["gauge_dependent"] = true;
      out["gauge_angle"] = column(m.gauge_angle, Json{{"value_type", "scalar"}, {"unit", "rad"}, {"precision", "float32"}, {"producer", gp},
                                                      {"description", "the angle of north (e1) in each vertex's gauge (geometry-central's); a heading from north = heading in the gauge minus this"}});
    }
    else for (const char* o : {"canonical", "singular", "gauge_angle"}) { Array a; a.meta["absent"] = true; out[o] = std::move(a); }   // stored already: nothing made (an optional output)
    out["angle"] = column(m.angle, Json{{"value_type", "scalar"}, {"unit", "rad"}, {"precision", "float32"}, {"producer", m.producer},
                                        {"description", "the angle from north (towards west) to ANTERIOR at each vertex; anatomical components are a fixed rotation of the canonical ones"}});
    out["second"] = column(m.second, Json{{"value_type", "integer"}, {"precision", "float32"}, {"producer", m.producer}, {"description", "which way the second axis points: 0 superior, 1 medial"}});
    out["handed"] = column(m.handed, Json{{"value_type", "integer"}, {"precision", "float32"}, {"producer", m.producer}, {"description", "the sign of n x anterior the second axis takes (+1 / -1)"}});
    out["reliable"] = picks(m.reliable, "the vertices where the anatomical frame is reliable: the anterior axis projects on the tangent plane with at least min_projection");
    return out;
  };
  return s;
}

}  // namespace

void append_geometry_steps(std::vector<StepSpec>& r) {
  r.push_back(frames_step());
}

}  // namespace nxr::pipelines::graph

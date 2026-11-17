// nxr-pipelines — declared outputs (declared_outputs.h): the definition places, the node describes, io writes.
#include "nxr/pipelines/declared_outputs.h"

#include <cmath>

#include "nxr/io/errors.h"
#include "nxr/io/finder.h"
#include "nxr/io/layout.h"

namespace nxr::pipelines::graph {

namespace {
using nxr::io::text;
namespace ng = nxr::graph;

[[noreturn]] void bad(const std::string& output, const std::string& what) { refuse("outputs." + output + ": " + what); }

std::string kind_of(const Json& spec) { return spec.value("kind", std::string()); }

// `{input}` -> the found input's name (or a parameter's value), `$parameter` -> its value (the longest parameter name that fits); twice, so that a parameter whose value holds `{input}` is filled
std::string fill_once(const Plan& P, const std::string& t) {
  std::string o;
  for (std::size_t i = 0; i < t.size();) {
    if (t[i] == '$') {
      std::size_t j = i + 1;
      while (j < t.size() && (std::isalnum(static_cast<unsigned char>(t[j])) || t[j] == '_')) ++j;
      std::string best;
      for (std::size_t e = j; e > i + 1; --e)
        if (P.parameters.contains(t.substr(i + 1, e - i - 1))) { best = t.substr(i + 1, e - i - 1); break; }
      if (!best.empty()) {
        const Json& v = P.parameters[best];
        o += v.is_string() ? v.get<std::string>() : v.dump();
        i += 1 + best.size();
        continue;
      }
    } else if (t[i] == '{') {
      const auto e = t.find('}', i);
      if (e != std::string::npos) {
        const std::string k = t.substr(i + 1, e - i - 1);
        if (P.found.count(k) && P.found.at(k).row) { o += text(*P.found.at(k).row, "name"); i = e + 1; continue; }
        if (P.parameters.contains(k)) { const Json& v = P.parameters[k]; o += v.is_string() ? v.get<std::string>() : v.dump(); i = e + 1; continue; }
      }
    }
    o += t[i++];
  }
  return o;
}
std::string fill(const Plan& P, const std::string& t) { return fill_once(P, fill_once(P, t)); }

// the id a link names: an output made already in this unit, a found input (its row's id), or one of its manifolds (`X.manifold.N`, `X.time`, `X.from`, `X.to`)
std::string link(const Plan& P, const std::string& ref, const MadeIds* made, const std::string& where) {
  if (made && made->count(ref)) return made->at(ref);
  if (P.definition.outputs.contains(ref) && !P.found.count(ref)) {
    if (made) bad(where, "'" + ref + "' is an output not made yet (manifolds are made first; list the others after what they lie on)");
    return "";                                                                // at plan time: an output of this definition, made in the unit
  }
  if (P.found.count(ref)) {
    if (!P.found.at(ref).row) bad(where, "'" + ref + "' has no stored row (a link names a found input)");
    return text(*P.found.at(ref).row, "id");
  }
  try {
    return nxr::io::finder::relation(ref, P.found, "outputs." + where);
  } catch (const nxr::io::Error& e) {
    bad(where, std::string("the link '") + ref + "': " + e.what());
  }
}
std::string link_of(const Plan& P, const Json& spec, const char* key, const MadeIds* made, const std::string& o) {
  if (!spec.contains(key) || !spec[key].is_string()) bad(o, std::string(key) + ": a link (INPUT.manifold.N, INPUT.time, an output made in the unit)");
  return link(P, spec[key].get<std::string>(), made, o);
}
// the found input a link starts from (its time line gives the samples' rate and origin)
const Value* base_input(const Plan& P, const std::string& ref) {
  const std::string b = ref.substr(0, ref.find('.'));
  return P.found.count(b) ? &P.found.at(b) : nullptr;
}

task::Values values_of(const ng::Array& a, const std::vector<std::int64_t>& shape, bool single, const std::string& o) {
  std::int64_t n = 1;
  for (const auto d : shape) n *= d;
  if (n != static_cast<std::int64_t>(a.data.size())) bad(o, std::to_string(a.data.size()) + " values for the shape the node describes (" + std::to_string(n) + ")");
  task::Values v;
  v.shape = shape;
  if (single) v.f32.assign(a.data.begin(), a.data.end());
  else v.f64 = a.data;
  return v;
}
const Json& describe_of(const ng::Array& a, const std::string& o) {
  if (!a.meta.contains("describe") || !a.meta["describe"].is_object()) bad(o, "its node gives no description of what it made (meta.describe): a declared output is stored from it");
  return a.meta["describe"];
}
bool single_of(const Json& d) { return d.value("precision", std::string("float64")) == "float32"; }
std::vector<std::int64_t> ints(const Json& j) { std::vector<std::int64_t> v; for (const auto& x : j) v.push_back(x.get<std::int64_t>()); return v; }
std::vector<std::int64_t> shape_of(const ng::Array& a, const Json& d) {
  if (d.contains("shape")) return ints(d["shape"]);
  if (d.contains("axes")) { std::vector<std::int64_t> s; for (const auto& ax : d["axes"]) s.push_back(ax.value("size", std::int64_t(1))); return s; }
  return a.cols > 1 ? std::vector<std::int64_t>{a.rows, a.cols} : std::vector<std::int64_t>{a.rows * a.cols};
}

// a tiling the node described, in the unit of the line it lies on (samples of the found recording's line, or seconds)
task::Tiling tiling_of(const Plan& P, const Json& t, const std::string& on_ref, const std::string& o) {
  task::Tiling out;
  const std::string unit = t.value("unit", std::string("s"));
  double origin = t.value("origin", 0.0), width = t.value("width", 0.0), hop = t.value("hop", 0.0), end = t.value("end", 0.0);
  if (unit == "samples") {
    const Value* in = base_input(P, on_ref);
    if (!in || !(in->rate > 0)) bad(o, "a tiling in samples lies on the time line of a found recording (partition.on: INPUT.time)");
    origin = in->origin + origin / in->rate; width /= in->rate; hop /= in->rate; end = in->origin + end / in->rate;
  } else if (unit != "s") {
    bad(o, "the tiling's unit is samples or s");
  }
  out.origin = origin; out.width = width; out.hop = hop; out.end = end;
  out.count = t.value("count", std::int64_t(0));
  out.skipped = t.value("skipped", std::int64_t(0));
  out.unit = "s";
  return out;
}

std::string acquisition_of(const Plan& P, const std::string& input) {
  if (!P.found.count(input) || !P.found.at(input).row) return {};
  return nxr::io::opt_text(*P.found.at(input).row, "acquisition_id").value_or("");
}
}  // namespace

bool is_declared(const Json& spec) {
  const std::string k = kind_of(spec);
  return k == "operator" || k == "manifold" || k == "selection" || k == "field" || (k == "statistic" && (spec.contains("axes") || spec.contains("of")));
}

bool is_declared(const Plan& P, const std::string& o) {
  const Json& spec = P.definition.outputs[o];
  const std::string cls = P.values.count(o) ? P.values.at(o).cls : std::string();
  if (kind_of(spec) == "selection" && (cls == "spans" || cls == "labelling")) return false;
  return is_declared(spec);
}

std::string declared_name(const Plan& P, const std::string& o) {
  const Json& spec = P.definition.outputs[o];
  if (spec.contains("name") && spec["name"].is_string()) return nxr::io::safe_name(fill(P, spec["name"].get<std::string>()));
  const auto a = P.definition.anchor();
  return nxr::io::safe_name(P.definition.name + "." + o + (a && P.found.at(*a).row ? "." + text(*P.found.at(*a).row, "name") : std::string()));
}

void check_declared(const Plan& P, const std::string& o, const nxr::io::Database& db) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec);
  const Value& v = P.values.at(o);
  const std::string want = k == "operator" ? "operator" : k == "manifold" ? "manifold" : k == "selection" ? "selection" : k == "field" ? "field" : "statistic";
  if (v.cls != want) bad(o, "declared " + k + " but its steps make a " + v.cls);
  if (v.chunked) bad(o, "a value inside the chunk loop is not stored: only what leaves it");
  if (declared_name(P, o).empty()) bad(o, "its name is empty");
  if (k == "field") {
    if (spec.contains("on")) link_of(P, spec, "on", nullptr, o);
    else if (v.manifolds.empty()) bad(o, "a field output names the manifold it lies on (on: INPUT.manifold.N)");
    if (!spec.contains("kind_name") || !spec["kind_name"].is_string()) bad(o, "a field output names its kind (kind_name: …)");
  } else if (k == "statistic") {
    if (!spec.contains("of")) bad(o, "a statistic names what it is of (of: INPUT)");
    link_of(P, spec, "of", nullptr, o);
    for (const auto& ax : spec.value("axes", Json::array())) {
      if (!ax.is_object() || !ax.contains("name")) bad(o, "axes: each {name, on | line}");
      if (ax.contains("on")) link(P, ax["on"].get<std::string>(), nullptr, o);
    }
    if (spec.contains("partition")) {
      const Json& p = spec["partition"];
      if (!p.is_object() || !p.contains("tiling") || !p.contains("on")) bad(o, "partition: {tiling: NAME, on: LINK}");
      link_of(P, p, "on", nullptr, o);
    }
  } else if (k == "operator") {
    link_of(P, spec, "from", nullptr, o);
    link_of(P, spec, "to", nullptr, o);
  } else if (k == "selection") {
    const std::string t = spec.value("type", std::string());
    if (t != "partition" && t != "set" && t != "spans" && t != "tiling" && t != "ladder") bad(o, "type: partition, set, spans, tiling or ladder");
    link_of(P, spec, "on", nullptr, o);
  } else {
    const std::string t = spec.value("type", std::string());
    if (t != "line" && t != "surface" && t != "product") bad(o, "type: line, surface or product");
    if (t == "product") {
      if (!spec.contains("factors") || !spec["factors"].is_array() || spec["factors"].size() < 2) bad(o, "a product's factors: a list of two or more links or outputs");
      for (const auto& f : spec["factors"]) link(P, f.get<std::string>(), nullptr, o);
    }
    if (t == "surface" && spec.contains("derived_from")) link_of(P, spec, "derived_from", nullptr, o);
  }
  (void)db;
}

std::vector<std::pair<task::Cls, std::string>> declared_places(const Plan& P, const std::string& o, nxr::io::SubjectWriter& sub) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec), name = declared_name(P, o);
  const task::Subject ts = task::Subject::of(sub);                          // io's naming rule for a node's store key
  std::vector<std::pair<task::Cls, std::string>> out;
  if (k == "field") {
    const std::string mid = spec.contains("on") ? link_of(P, spec, "on", nullptr, o) : P.values.at(o).manifolds.at(0);
    out.emplace_back(task::Cls::field, ts.node_path(mid, name));
  } else if (k == "statistic") {
    const std::string of = spec["of"].get<std::string>();
    std::string path = "statistics/" + nxr::io::safe_name(name);
    if (P.found.count(of) && P.found.at(of).row && P.found.at(of).cls == "field") path = text(*P.found.at(of).row, "path") + "_statistics/" + nxr::io::safe_name(name);
    out.emplace_back(task::Cls::statistic, path);
    if (spec.contains("partition")) out.emplace_back(task::Cls::selection, ts.node_path(link_of(P, spec["partition"], "on", nullptr, o), nxr::io::safe_name(fill(P, spec["partition"]["tiling"].get<std::string>()))));
  } else if (k == "operator") {
    out.emplace_back(task::Cls::operator_, nxr::io::safe_name(name));
  } else if (k == "selection") {
    out.emplace_back(task::Cls::selection, ts.node_path(link_of(P, spec, "on", nullptr, o), name));
  }
  return out;
}

Json store_declared(task::Unit& unit, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const ng::Array& a, const Json& span, MadeIds& made, bool replace, bool& replaced) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec), name = declared_name(P, o);
  const Json& d = describe_of(a, o);
  const std::string what = P.definition.name + " output " + o;
  auto clear = [&](task::Cls c, const std::string& path) {                    // another job's node there: refused, or moved aside with --replace
    const bool there = sub.find(task::to_string(c), path).has_value();
    unit.replace(c, path, replace, what);
    if (there) replaced = true;
  };
  Json report = Json{{"output", o}, {"kind", k}, {"name", name}};

  if (k == "manifold") {
    const std::string t = spec["type"].get<std::string>();
    std::string id;
    if (t == "line") {
      task::LineOut l;
      l.name = spec.contains("name") ? nxr::io::safe_name(fill(P, spec["name"].get<std::string>())) : name;
      l.alt = spec.contains("alt") ? nxr::io::safe_name(fill(P, spec["alt"].get<std::string>())) : l.name + "_alt";
      l.n = d.value("count", std::int64_t(0));
      l.spacing = d.value("step", 1.0);
      l.origin = d.value("start", 0.0);
      l.unit = d.value("unit", std::string("Hz"));
      l.dimension = d.value("dimension", std::string("frequency"));
      l.folder = d.value("folder", l.dimension == "time" ? std::string("time") : std::string("spectral"));
      l.cs_key = spec.value("cs", d.value("cs", l.unit == "s" ? std::string("clock") : std::string("hz")));
      if (l.n < 1) bad(o, "the line the node describes has no point (count)");
      id = unit.line(l);
    } else if (t == "surface") {
      task::SurfaceOut s;
      s.name = name; s.output = o;
      s.cs_key = spec.value("cs", d.value("cs", std::string("scs")));
      if (spec.contains("derived_from")) s.derived_from_id = link_of(P, spec, "derived_from", &made, o);
      s.vertices = values_of(a, {a.rows, a.cols}, false, o);
      if (a.cols != 3) bad(o, "a surface's vertices are [n x 3]");
      for (const auto& f : d.value("faces", Json::array())) s.faces.push_back(f.get<std::int32_t>());
      id = unit.surface(s);
    } else {
      task::ProductOut pr;
      for (const auto& f : spec["factors"]) pr.factors.push_back(link(P, f.get<std::string>(), &made, o));
      id = unit.product(pr);
    }
    made[o] = id;
    report["id"] = id;
    return report;
  }

  if (k == "field") {
    task::FieldOut f;
    f.name = name; f.output = o; f.kind = spec["kind_name"].get<std::string>();
    f.manifold_id = spec.contains("on") ? link_of(P, spec, "on", &made, o) : P.values.at(o).manifolds.at(0);
    f.value_type = d.value("value_type", P.values.at(o).value_type);
    if (d.contains("unit") && d["unit"].is_string()) f.unit = d["unit"].get<std::string>();
    else if (spec.contains("unit")) f.unit = spec["unit"].get<std::string>();
    f.description = spec.contains("description") ? spec["description"].get<std::string>() : d.value("description", f.kind + " \xe2\x80\x94 pipeline " + P.definition.name);
    f.data = values_of(a, shape_of(a, d), single_of(d), o);
    const std::string path = unit.path_of(f);
    clear(task::Cls::field, path);
    report["id"] = unit.field(f); report["path"] = path;
    return report;
  }

  if (k == "operator") {
    task::OperatorOut op;
    op.name = name; op.output = o; op.kind = spec["operator_kind"].get<std::string>();
    op.from_manifold_id = link_of(P, spec, "from", &made, o);
    op.to_manifold_id = link_of(P, spec, "to", &made, o);
    op.from_value_type = d.value("from_value_type", std::string("scalar"));
    op.to_value_type = d.value("to_value_type", std::string("scalar"));
    if (d.contains("description")) op.description = d["description"].get<std::string>();
    if (d.contains("params")) op.params = d["params"];
    op.data = values_of(a, {a.rows, a.cols}, single_of(d), o);
    const std::string path = unit.path_of(op);
    clear(task::Cls::operator_, path);
    report["id"] = unit.operator_(op); report["path"] = path;
    return report;
  }

  if (k == "selection") {
    task::SelectionOut s;
    const std::string t = spec["type"].get<std::string>();
    s.name = name; s.output = o; s.type = t == "ladder" ? "tiling" : t;
    s.manifold_id = link_of(P, spec, "on", &made, o);
    if (d.contains("params")) s.params = d["params"];
    if (d.contains("description")) s.description = d["description"].get<std::string>();
    for (const auto& e : d.value("entries", Json::array())) s.entries.push_back(e.get<std::string>());
    if (t == "partition" || t == "set") for (const double c : a.data) s.codes.push_back(static_cast<std::int64_t>(c));
    else if (t == "spans") for (const auto& x : d.value("spans", Json::array())) { task::Span sp; sp.start = x["start"].get<double>(); sp.stop = x["stop"].get<double>(); if (x.contains("code")) sp.code = x["code"].get<std::int64_t>(); sp.ordinal = static_cast<std::int64_t>(s.spans.size()); s.spans.push_back(sp); }
    else if (!d.contains("tiling")) bad(o, "its node gives no tiling (describe.tiling)");
    else s.tiling = tiling_of(P, d["tiling"], spec["on"].get<std::string>(), o);
    const std::string path = unit.path_of(s);
    clear(task::Cls::selection, path);
    report["id"] = unit.selection(s); report["path"] = path;
    return report;
  }

  // a statistic
  task::StatisticOut st;
  st.name = name; st.output = o;
  st.measure = d.value("measure", std::string()); st.method = d.value("method", std::string());
  if (st.measure.empty() || st.method.empty()) bad(o, "its node describes no measure or method");
  const std::string of = spec["of"].get<std::string>();
  const std::string of_id = link_of(P, spec, "of", &made, o);
  const bool of_field = !P.found.count(of) || P.found.at(of).cls == "field";
  if (of_field) st.of_field_id = of_id; else st.of_statistic_id = of_id;
  const std::string acq = acquisition_of(P, of);
  const Json axes_decl = spec.value("axes", Json::array());
  const Json axes_desc = d.value("axes", Json::array());
  if (axes_decl.size() != axes_desc.size()) bad(o, "declares " + std::to_string(axes_decl.size()) + " axes, its node describes " + std::to_string(axes_desc.size()));
  std::vector<std::int64_t> shape;
  Json axes = Json::array();
  for (std::size_t i = 0; i < axes_decl.size(); ++i) {
    const Json& decl = axes_decl[i];
    Json ax = axes_desc[i];
    if (decl["name"] != ax.value("name", Json())) bad(o, "axis " + std::to_string(i) + " is declared '" + decl["name"].get<std::string>() + "', its node names it " + ax.value("name", Json()).dump());
    const std::int64_t size = ax.value("size", std::int64_t(0));
    Json out = Json{{"name", decl["name"]}, {"size", size}};
    if (decl.contains("on")) {
      out["manifold_id"] = link(P, decl["on"].get<std::string>(), &made, o);
    } else if (decl.contains("line")) {
      if (!ax.contains("line")) bad(o, "axis " + decl["name"].get<std::string>() + " declares a line; its node describes none");
      const Json& l = ax["line"];
      task::LineOut lo;
      lo.name = nxr::io::safe_name(fill(P, decl["line"].get<std::string>())); lo.alt = lo.name + "_alt";
      lo.n = size; lo.spacing = l.value("spacing", 1.0); lo.origin = l.value("origin", 0.0);
      lo.unit = l.value("unit", std::string("Hz")); lo.dimension = l.value("dimension", std::string("frequency"));
      lo.folder = l.value("folder", lo.dimension == "time" ? std::string("time") : std::string("spectral"));
      lo.cs_key = l.value("cs", lo.unit == "s" ? std::string("clock") : std::string("hz"));
      if (!acq.empty()) lo.acquisition_id = acq;
      out["manifold_id"] = unit.line(lo);
      out["unit"] = lo.unit;
    } else {
      for (const char* key : {"values", "edges", "scale", "unit", "selection_id"}) if (ax.contains(key)) out[key] = ax[key];
    }
    axes.push_back(out);
    shape.push_back(size);
  }
  st.axes = axes;
  if (spec.contains("partition")) {
    const Json& p = spec["partition"];
    if (!d.contains("tiling")) bad(o, "declares a partition; its node describes no tiling");
    task::SelectionOut seg;
    seg.name = nxr::io::safe_name(fill(P, p["tiling"].get<std::string>())); seg.output = o; seg.type = "tiling";
    seg.manifold_id = link_of(P, p, "on", &made, o);
    seg.tiling = tiling_of(P, d["tiling"], p["on"].get<std::string>(), o);
    if (of_field) seg.of_field_id = of_id;
    if (!acq.empty()) seg.acquisition = acq;
    const std::int64_t n = seg.tiling->count + seg.tiling->skipped;
    seg.description = "the " + std::to_string(n) + " segments of " + (P.found.count(of) && P.found.at(of).row ? text(*P.found.at(of).row, "name") : of) + " behind " + name + " (" + std::to_string(seg.tiling->skipped) + " left out)";
    const std::string spath = unit.path_of(seg);
    clear(task::Cls::selection, spath);
    st.partition_selection_id = unit.selection(seg);
    report["partition"] = Json{{"name", seg.name}, {"path", spath}, {"id", *st.partition_selection_id}};
    st.n_members = d.value("n_members", seg.tiling->count);
    st.members_skipped = d.value("members_skipped", seg.tiling->skipped);
  } else {
    if (d.contains("n_members")) st.n_members = d["n_members"].get<std::int64_t>();
    if (d.contains("members_skipped")) st.members_skipped = d["members_skipped"].get<std::int64_t>();
  }
  if (d.contains("components")) st.components = d["components"];
  if (d.contains("scaling")) st.scaling = d["scaling"];
  st.options = d.value("options", Json::object());
  if (d.contains("unit") && d["unit"].is_string()) st.unit = d["unit"].get<std::string>();
  if (d.contains("description")) st.description = d["description"].get<std::string>();
  st.label = d.value("label", st.method);
  if (d.contains("producer") && d["producer"].is_object()) st.producer = d["producer"];   // what the node records beside the provenance (a spectrum's resolution and rate: its readers ask for them)
  if (!acq.empty()) st.acquisition = acq;
  if (!span.is_null()) st.span = span;
  st.value_type = d.value("value_type", std::string("scalar"));
  st.data = values_of(a, shape, single_of(d), o);
  const std::string path = unit.path_of(st);
  clear(task::Cls::statistic, path);
  report["id"] = unit.statistic(st); report["path"] = path;
  made[o] = report["id"].get<std::string>();
  return report;
}

}  // namespace nxr::pipelines::graph

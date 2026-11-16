// nxr-pipelines — the runner (runner.h): nxr's nxr/pipelines/runner.py `plan` and the planning half of `run`. The execution (before · the chunk loop · after
// · store) is the next PR of the spec's order.
#include "nxr/pipelines/runner.h"

#include "nxr/bst/entities.h"
#include "nxr/io/measures.h"
#include "nxr/histogram.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <cstring>
#include <sstream>

#include "nxr/io/array_reader.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/dispatch.h"
#include "nxr/pipelines/reading.h"

#include "nxr/io/finder.h"

namespace nxr::pipelines::graph {

namespace {
// a `with:` value with its `$name`s replaced by the pipeline's parameters
Json with_value(const Json& v, const Json& params) {
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (!s.empty() && s[0] == '$') return params.at(s.substr(1));
    return v;
  }
  if (v.is_array()) { Json a = Json::array(); for (const auto& x : v) a.push_back(with_value(x, params)); return a; }
  if (v.is_object()) { Json o = Json::object(); for (const auto& [k, x] : v.items()) o[k] = with_value(x, params); return o; }
  return v;
}
std::string label(const std::string& sec) { return sec == "do" ? "chunks" : sec; }
std::string manifold_name(const nxr::io::Database& db, const std::string& id) {
  const auto m = db.one("SELECT name FROM manifold WHERE id = ?", {nxr::io::V(id)});
  return m ? nxr::io::text(*m, "name") : id;
}
std::string on(const nxr::io::Database& db, const Value& v) {
  std::string s;
  for (const auto& m : v.manifolds) s += (s.empty() ? "" : " × ") + manifold_name(db, m);
  return s;
}
std::string type_of(const Value& v) {
  return std::string(v.complex ? "ℂ " : "") + v.value_type + (v.unit ? ", " + *v.unit : "");
}
}  // namespace

std::vector<const Planned*> Plan::section(const std::string& sec) const {
  std::vector<const Planned*> out;
  for (const auto& p : steps) if (p.use.section == sec) out.push_back(&p);
  return out;
}

std::string Plan::text(const nxr::io::Database& db) const {
  const Definition& d = definition;
  std::ostringstream L;
  const auto anchor = d.anchor();
  L << "pipeline " << d.name << "@" << d.version;
  if (anchor) L << " — " << *anchor << " '" << nxr::io::text(*found.at(*anchor).row, "name") << "'";
  L << "\n";
  for (const auto& [k, _] : d.inputs.items()) {
    if (!found.count(k)) { L << "  input  " << k << "  (absent: optional)\n"; continue; }
    const Value& v = found.at(k);
    const auto& r = *v.row;
    L << "  input  " << k << "  " << v.cls << " '" << nxr::io::text(r, "name") << "' (" << (r.count("kind") ? nxr::io::text(r, "kind") : r.count("type") ? nxr::io::text(r, "type") : nxr::io::text(r, "method")) << ") "
      << (v.cls == "operator" ? "from " + manifold_name(db, v.manifolds[0]) + " to " + manifold_name(db, v.manifolds[1]) : v.cls == "statistic" ? std::string("(stored, read by name)") : "on " + on(db, v)) << "\n";
  }
  L << "  parameters " << parameters.dump() << "\n";
  for (const auto& p : steps) {
    Json ins = Json::object();
    for (const auto& [port, t] : p.ins) ins[port] = t;
    std::string outs;
    for (const auto& [k, n] : p.outs) {
      const Value& v = values.at(n);
      outs += (outs.empty() ? "" : ", ") + n + " [" + type_of(v) + " on " + on(db, v) + "]";
    }
    L << "  [" << label(p.use.section) << "] " << p.step->name << " (" << p.step->kind << ") in " << ins.dump() << " with " << p.params.dump() << " → " << outs << "\n";
  }
  for (const auto& [o, spec] : d.outputs.items()) {
    const Value& v = values.at(o);
    if (v.cls == "measurement") L << "  output " << o << "  measurement '" << v.facts.value("measure", std::string()) << "' over " << v.facts["over"].value("selection", std::string()) << " (rows: selection_measurement, array-backed)\n";
    else if (v.cls == "spans") L << "  output " << o << "  a spans Selection on " << on(db, v) << " (event types " << spec.value("types", Json::array({"event"})).dump() << ")\n";
    else if (v.cls == "labelling") L << "  output " << o << "  a labelling of the joint cells (a `set` Selection on " << on(db, v) << ", array-backed)\n";
    else if (v.cls == "statistic") L << "  output " << o << "  a statistic of the measurement, summaries " << v.facts["summaries"].dump() << " (" << v.facts.value("combine", std::string()) << ")\n";
    else L << "  output " << o << "  field on " << on(db, v) << " (" << type_of(v) << ") — kind '" << spec["kind"].get<std::string>() << "'\n";
  }
  return L.str();
}

Plan plan(const Definition& d, const Values& found, const Json& params, const nxr::io::Database& db) {
  Plan P{d, params, found, found, {}};
  if (d.over && P.values.count(*d.over)) P.values[*d.over].chunked = true;
  const PlanContext ctx{db};
  for (const auto& use : d.uses()) {
    const StepSpec& st = step(use.step);
    const std::string w = "steps: " + st.name + " (" + label(use.section) + ")";
    Values ins;
    for (const auto& [port, thing] : use.ins.items()) {
      const std::string t = thing.get<std::string>();
      if (P.values.count(t)) ins[port] = P.values.at(t);
      else if (std::find(st.optional.begin(), st.optional.end(), port) == st.optional.end())
        refuse(w + ": in." + port + " '" + t + "' is absent (an optional input the subject lacks), and the step needs it");
    }
    if (use.section == "before" || use.section == "after")
      for (const auto& [k, v] : ins) if (v.chunked) refuse(w + ": given recording data (" + k + "): only steps inside the chunk loop take it");
    Json p = st.parameters;
    for (const auto& [k, v] : use.with.items()) p[k] = with_value(v, params);
    Values made;
    try {
      made = st.plan(ins, p, ctx);
    } catch (const nxr::io::Error& e) {
      refuse(w + ": " + e.what());
    }
    Planned pl{use, &st, p, {}, {}};
    for (const auto& [port, thing] : use.ins.items()) pl.ins[port] = thing.get<std::string>();
    for (std::size_t i = 0; i < st.outputs.size(); ++i) {
      const std::string& name = use.outs[i];
      const Value& v = made.at(st.outputs[i]);
      if (use.section == "do" && st.kind == "accumulate" && v.chunked) refuse(w + ": an accumulate step's output '" + name + "' must leave the loop");
      P.values[name] = v;
      pl.outs[st.outputs[i]] = name;
    }
    P.steps.push_back(std::move(pl));
  }
  for (const auto& [o, spec] : d.outputs.items()) {
    const Value& v = P.values.at(o);
    // THREE KINDS OF OUTPUT: a Field (materialized), a measurement over a time partition (reduced: selection_measurement rows), a span set (a spans Selection)
    const std::string kind = spec.value("kind", std::string());
    if (v.cls == "measurement" || v.cls == "spans" || v.cls == "statistic" || v.cls == "labelling") {
      if (kind != v.cls) refuse("outputs." + o + ": its steps make a " + v.cls + "; declare it {kind: " + v.cls + "}");
      if (!d.over) refuse("outputs." + o + ": a " + v.cls + " is of the recording the chunk loop reads; this definition has no chunk loop");
      if (v.cls == "spans" && spec.contains("types") && !(spec["types"].is_array() && spec["types"].size() == 1 && spec["types"][0].is_string()))
        refuse("outputs." + o + ": types is one event type's name, e.g. [burst]");
      continue;
    }
    if (kind == "measurement" || kind == "spans" || kind == "statistic" || kind == "labelling") refuse("outputs." + o + ": declared " + kind + " but its steps make a " + v.cls);
    if (v.cls != "field" || v.chunked) refuse("outputs." + o + ": not a field outside the chunk loop: only those are stored");
    for (const auto& m : v.manifolds)
      if (!db.one("SELECT id FROM manifold WHERE id = ?", {nxr::io::V(m)})) refuse("outputs." + o + ": its manifold " + m + " is not among the subject's rows");
  }
  return P;
}

Json parameters_of(const Definition& d, const Json& set) {
  Json p = d.parameters;
  std::string unknown;
  for (const auto& [k, v] : set.items()) {
    if (!d.parameters.contains(k)) { unknown += (unknown.empty() ? "" : ", ") + k; continue; }
    p[k] = v;
  }
  if (!unknown.empty()) {
    std::string have;
    for (const auto& [k, _] : d.parameters.items()) have += (have.empty() ? "" : ", ") + k;
    refuse("--set " + unknown + ": not parameters of " + d.name + " (" + (have.empty() ? std::string("none") : have) + ")");
  }
  return p;
}

std::vector<Plan> plan_runs(const Definition& d, const nxr::io::Database& db, const std::string& subject_id, const Json& set) {
  const Json params = parameters_of(d, set);
  Definition e = expand(d, params);                                         // a definition that repeats over list parameters, written out for this run (flat: unchanged)
  if (!e.draw.empty()) {                                                    // `draw: seed`: every draw is a node of its own (the stored nodes' names, the cache key's pipeline), the parameter's value in the name
    if (!params.contains(e.draw) || !params[e.draw].is_number_integer() || params[e.draw].get<std::int64_t>() < 0) refuse(d.name + ": the parameter `" + e.draw + "` (draw) is a whole number >= 0");
    e.name += ".draw" + std::to_string(params[e.draw].get<std::int64_t>());
  }
  std::vector<Plan> out;
  for (const auto& found : nxr::io::finder::resolve(db, subject_id, e.inputs, params)) out.push_back(plan(e, found, params, db));   // every run planned before any is executed
  return out;
}

namespace {
using nxr::io::text;
using nxr::io::V;

// an exclude input's spans [start, stop) in seconds (a recording's bad segments)
std::vector<std::pair<double, double>> spans_of(const nxr::io::Database& db, const nxr::io::Row& selection) {
  std::vector<std::pair<double, double>> out;
  for (const auto& r : db.all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(selection, "id"))}))
    out.emplace_back(std::stod(text(r, "start")), std::stod(text(r, "stop")));
  return out;
}
// the components a value holds an element (its rows a vertex)
std::int64_t components(const Value& v) { return v.value_type == "scalar" ? 1 : v.value_type == "quaternion" ? 4 : 3; }
std::string node_name(const Definition& d, const std::string& output, const Values& found) {
  const auto a = d.anchor();
  return nxr::io::safe_name(d.name + "." + output + (a ? "." + text(*found.at(*a).row, "name") : std::string()));
}
// where a statistic output lives: beside the recording the chunk loop reads (`<recording>_statistics/<node>`)
std::string statistic_out_path(const Plan& P, const std::string& o) {
  return text(*P.found.at(*P.definition.over).row, "path") + "_statistics/" + node_name(P.definition, o, P.found);
}
// how a measurement output was made, as its rows say it (`computed_by`, the recipe policy of `measure compute`): the pipeline and its cache key's hash
std::string made_by(const nxr::pipelines::CacheKey& key) {
  return "pipeline " + key.pipeline + " \xc2\xb7 cache " + nxr::io::sha256_hex(key.json().dump()).substr(0, 16) + " \xc2\xb7 nxr-neuro " + NXR_NEURO_VERSION;
}
// the measures a measurement output stores: one row a component. A vector measure (the phase current) is stored as three, `<measure>_x`, `_y`, `_z` (the model's key holds one row a measure name and
// member, so the components differ by name, as the registry's phase_current_x / _y / _z); a scalar one under its own name
std::vector<std::string> measure_rows(const std::string& measure) {
  if (measure == "phase_current") return {"phase_current_x", "phase_current_y", "phase_current_z"};
  if (measure == "centre") return {"centre_x", "centre_y", "centre_z"};          // a point event's centre (point_events.h)
  if (measure == "tile_crossings") return {"crossing_sum_x", "crossing_sum_y", "crossing_sum_z", "crossings"};   // the tracks' crossings of the tiles: the vector sum, then the count (the step `tile_crossings`)
  return {measure};
}
// the computed_by of the rows a measurement output would replace (selection, recording, measure), or nothing
std::vector<std::string> measured_by(const nxr::io::Database& db, const std::string& selection_id, const Value& v, const std::string& field_id) {
  std::vector<std::string> out;
  if (selection_id.empty()) return out;
  for (const auto& r : db.all("SELECT DISTINCT computed_by FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = ?",
                              {V(selection_id), V(field_id), V(measure_rows(v.facts["measure"].get<std::string>())[0])}))
  {
    std::string c = nxr::io::opt_text(r, "computed_by").value_or("");
    const std::size_t at = c.find(" \xc2\xb7 stages ");              // the crossings' stage names are an annotation of the rows, not part of how they were made (the stages are in the cache key already)
    out.push_back(at == std::string::npos ? c : c.substr(0, at));
  }
  return out;
}
// a measurement over the EVENTS of the run (rollup_events, extent_measure): its Selection is the run's own events output, stored first: the span set (the default) or, `events_of: labelling`, the labelling
bool over_events(const Value& v) { return v.cls == "measurement" && v.facts["over"].value("events", false); }
std::string events_class(const Json& over) { return over.value("events_of", std::string("spans")); }
// a run that makes several sets of point events (the cores, the tracks made of them) names the set by its kind (point_events.h); a measurement is over the set of its kind
std::string events_kind_of(const Value& v) { return v.cls == "measurement" && v.facts["over"].contains("events_kind") ? v.facts["over"]["events_kind"].get<std::string>() : std::string(); }
std::string events_output(const Plan& P, const std::string& cls, const std::string& kind = std::string()) {
  std::string found;
  for (const auto& [o, _] : P.definition.outputs.items())
    // the run's EVENTS are the detected spans; the gated subset (`fit_gate`) and the stages (`burst_stages`) are spans outputs derived from them, never the events a measurement is over
    if (P.values.at(o).cls == cls && !P.values.at(o).facts.contains("gate") && !P.values.at(o).facts.contains("stage") && P.values.at(o).facts.value("events_kind", std::string()) == kind) { if (!found.empty()) refuse("outputs: a measurement over the run's events is ambiguous with two " + cls + " outputs (" + found + ", " + o + ")"); found = o; }
  if (found.empty()) refuse("outputs: a measurement over the run's events needs the run's " + cls + " output among its outputs");
  return found;
}
// the id of the Selection a measurement is over: the tile level's, or (for events) the run's events output as stored (empty: not stored)
std::string over_selection(nxr::io::SubjectWriter& sub, const Plan& P, const Value& v) {
  if (!over_events(v)) return v.facts["over"]["selection_id"].get<std::string>();
  const std::string out = events_output(P, events_class(v.facts["over"]), events_kind_of(v));
  const Value& sp = P.values.at(out);
  const auto row = sub.find("selection", nxr::io::node_path(sub.db(), sp.manifolds[0], node_name(P.definition, out, P.found)));
  return row ? text(*row, "id") : std::string();
}
// a measurement whose members or rows are the run's events, when the run found none: nothing is stored for it (no member), and a rerun finds that already done
bool no_events(nxr::io::SubjectWriter& sub, const Plan& P, const Value& v) {
  if (v.cls != "measurement" || !v.facts.contains("over")) return false;
  const Json& over = v.facts["over"];
  if (!over.value("events", false) && !over.contains("rows_events_of")) return false;
  const std::string cls = over.contains("rows_events_of") ? over["rows_events_of"].get<std::string>() : events_class(over);
  const std::string out = events_output(P, cls, events_kind_of(v));
  const auto row = sub.find("selection", nxr::io::node_path(sub.db(), P.values.at(out).manifolds[0], node_name(P.definition, out, P.found)));
  return row && nxr::io::integer(*row, "n_members") == 0;
}
}  // namespace

namespace {
// MEASUREMENT OUTPUTS, stored as the envelope and `measure over time` store theirs (D96): ONE array [rows x members x (components + 1)] beside the recording (the measures' components, the samples counted
// LAST), and one `selection_measurement` row a component (and the count) on the Selection: of_field_id the recording, computed_by how it was made.
// PACKED: the measurements of one run over the SAME Selection with the same rows, members and counts share ONE array, each its own components in the order of the
// definition's outputs, each measure's rows pointing at it with their `component` (the schema's array-backed rule). A tile not whole and good in the span is NaN.
// `group` are the outputs packed together (in the definition's order); each result says its `array`, its first `component` and its `components`.
std::vector<Json> store_measurements(nxr::io::SubjectWriter& sub, const Plan& P, const std::vector<std::string>& group, const std::map<std::string, Array>& data, const std::string& by, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string& o0 = group.front();
  const Value& v0 = P.values.at(o0);
  // over the tiles of a level, or (rollup_events) over the run's own events, whose Selection was stored just before: the member axis is then the events' order
  const std::string sel = over_selection(sub, P, v0);
  if (sel.empty()) refuse("outputs." + o0 + ": the Selection it is over was not stored");
  const Array& a0 = data.at(o0);
  std::vector<Json> results;
  if (a0.cols == 0) {                                                       // no event: nothing to store
    for (const auto& o : group) results.push_back(Json{{"output", o}, {"measurement", P.values.at(o).facts["measure"]}, {"over", P.values.at(o).facts["over"]["selection"]}, {"rows", a0.rows}, {"events", 0}});
    return results;
  }
  const std::int64_t R = a0.rows, T = a0.cols;
  std::vector<std::int64_t> count(static_cast<std::size_t>(T), 0);
  for (const auto& c : a0.meta["counts"]) count[static_cast<std::size_t>(c[0].get<std::int64_t>())] = c[1].get<std::int64_t>();
  struct Part { std::string o; std::vector<std::string> names; std::int64_t C, offset; };
  std::vector<Part> parts;
  std::int64_t CC = 0;
  for (const auto& o : group) {
    const Value& v = P.values.at(o);
    const Array& a = data.at(o);
    Part p{o, measure_rows(v.facts["measure"].get<std::string>()), a.comps, CC};
    if (static_cast<std::int64_t>(p.names.size()) != p.C) refuse("outputs." + o + ": measure " + v.facts["measure"].get<std::string>() + " stores " + std::to_string(p.names.size()) + " components, its array holds " + std::to_string(p.C));
    if (a.rows != R || a.cols != T) refuse("outputs." + o + ": packed with " + o0 + " but its array is not [rows x members] alike");
    CC += p.C;
    parts.push_back(std::move(p));
  }
  // the array is [rows x members x (CC + 1)]: each measure's components in order, then the count LAST
  std::vector<float> arr(static_cast<std::size_t>(R * T * (CC + 1)));
  for (const Part& p : parts) {
    const Array& a = data.at(p.o);
    for (std::int64_t r = 0; r < R; ++r)
      for (std::int64_t t = 0; t < T; ++t)
        for (std::int64_t c = 0; c < p.C; ++c) arr[static_cast<std::size_t>((r * T + t) * (CC + 1) + p.offset + c)] = static_cast<float>(a.data[static_cast<std::size_t>((r * T + t) * p.C + c)]);
  }
  for (std::int64_t r = 0; r < R; ++r)
    for (std::int64_t t = 0; t < T; ++t) arr[static_cast<std::size_t>((r * T + t) * (CC + 1) + CC)] = static_cast<float>(count[static_cast<std::size_t>(t)]);
  const std::string path = text(rec, "path") + "_" + nxr::io::safe_name(d.name + "." + o0 + (group.size() > 1 ? "+" + std::to_string(group.size() - 1) : std::string()));
  // THE COUNT ROW is the samples per member, a property of the members within the run's span: one per (Selection, recording), the roll-up weight shared by every
  // measure of them. Rows already there from the same span are kept (no second row); from another span (the counts differ) they are refused; with --replace the count row is
  // rewritten only when no other measure shares it (those are never deleted here: the refusal names them); a replaced measure leaves the count row in place while another measure (or the same counts) needs it.
  const std::string fid = text(rec, "id");
  std::vector<std::string> names;
  for (const Part& p : parts) for (const auto& n : p.names) names.push_back(n);
  bool keep_count = false, count_replaced = false;
  if (const auto cr = sub.db().one("SELECT array_path, component FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'count'", {V(sel), V(fid)})) {
    bool same = false;
    try {
      // the stored count row says which component of which array is the count (the last): read it there, never assume a layout
      nxr::io::ArrayReader rd = nxr::io::ArrayReader::open(sub.store(), text(*cr, "array_path"));
      const auto sh = rd.shape();
      const std::int64_t cc = nxr::io::integer(*cr, "component");
      if (sh.size() == 3 && sh[1] == T && cc >= 0 && cc < sh[2] && sh[0] >= 1 && rd.dtype() == nxr::io::DType::Float32) {
        const auto raw = rd.read();
        std::vector<float> old(raw.size() / sizeof(float));
        std::memcpy(old.data(), raw.data(), old.size() * sizeof(float));
        same = true;
        for (std::int64_t t = 0; t < T; ++t) if (static_cast<std::int64_t>(old[static_cast<std::size_t>(t * sh[2] + cc)]) != count[static_cast<std::size_t>(t)]) same = false;
      }
    } catch (const nxr::io::Error&) { same = false; }
    if (same) {
      keep_count = true;
      // a count row in the very array about to be rewritten is rewritten with it (its component may move)
      if (text(*cr, "array_path") == path) { nxr::io::delete_measurements(sub.db(), Json{{"selectionId", sel}, {"ofFieldId", fid}, {"measure", "count"}}); keep_count = false; }
    } else {
      if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o0 + ": the stored measurements of this Selection and recording were made over another span (their counts differ); pass --replace to rewrite their count row");
      // another span: the count row is rewritten only when no other measure shares it (their counts would no longer match); they are never deleted here
      // the measures THIS run writes over the same Selection are not others: each is replaced in its turn and then shares the new count row
      std::set<std::string> ours;
      for (const auto& [o2, _] : P.definition.outputs.items()) {
        const Value& v2 = P.values.at(o2);
        if (v2.cls == "measurement" && !over_events(v2) && over_selection(sub, P, v2) == sel) for (const auto& n2 : measure_rows(v2.facts["measure"].get<std::string>())) ours.insert(n2);
      }
      std::string others;
      for (const auto& r : sub.db().all("SELECT DISTINCT measure FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure <> 'count' ORDER BY measure", {V(sel), V(fid)}))
        if (!ours.count(text(r, "measure")) && std::find(names.begin(), names.end(), text(r, "measure")) == names.end()) others += (others.empty() ? "" : ", ") + text(r, "measure");
      if (!others.empty()) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o0 + ": the measurements " + others + " of this Selection and recording were made over another span (their counts differ); recompute them over this span (with --replace) or remove them first");
      nxr::io::delete_measurements(sub.db(), Json{{"selectionId", sel}, {"ofFieldId", fid}, {"measure", "count"}});
      count_replaced = true;
    }
  }
  for (const Part& p : parts) {
    if (measured_by(sub.db(), sel, P.values.at(p.o), fid).empty()) continue;
    if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + p.o + ": its measurements already exist; pass --replace");
    for (const auto& n : p.names) nxr::io::delete_measurements(sub.db(), Json{{"selectionId", sel}, {"ofFieldId", fid}, {"measure", n}});
    replaced = true;
  }
  sub.write_array(path, nxr::io::view(arr, {R, T, CC + 1}));
  Json rows = Json::array();
  for (const Part& p : parts) {
    const Value& v = P.values.at(p.o);
    const std::string measure = v.facts["measure"].get<std::string>();
    for (std::int64_t c = 0; c < p.C; ++c) {
      Json row0{{"selection_id", sel}, {"of_field_id", fid}, {"measure", p.names[static_cast<std::size_t>(c)]}, {"array_path", path}, {"component", p.offset + c}, {"computed_by", by}};
      if (v.unit) row0["unit"] = *v.unit;
      else if (const auto u = nxr::io::opt_text(rec, "unit")) if (measure == "amplitude" || measure == "envelope_max") row0["unit"] = *u;
      rows.push_back(row0);
    }
  }
  if (!keep_count) rows.push_back(Json{{"selection_id", sel}, {"of_field_id", fid}, {"measure", "count"}, {"array_path", path}, {"component", CC}, {"computed_by", by}});
  nxr::io::write_measurements(sub.db(), rows, by);
  for (const Part& p : parts) {
    const Value& v = P.values.at(p.o);
    const Array& a = data.at(p.o);
    results.push_back(Json{{"output", p.o}, {"count_kept", keep_count}, {"count_replaced", count_replaced}, {"measurement", v.facts["measure"]}, {"over", v.facts["over"]["selection"]}, {"array", path}, {"component", p.offset}, {"components", p.C},
                           {"array_components", CC + 1}, {"packed", static_cast<std::int64_t>(group.size())}, {"rows", R}, {"tiles", T}, {"whole_tiles", a.meta["whole_tiles"]}, {"tiles_measured", a.meta["tiles_measured"]}});
  }
  return results;
}
// THE CROSSINGS OF THE TILES (step `tile_crossings`): ONE array [rows x members x 4] beside the recording, rows = stage x sense, members = the level's tiles then the `bridge`, components the vector sum (x y z) then the
// crossings' count; four `selection_measurement` rows on the level (crossing_sum_x/y/z, crossings; of_field_id the recording). The count is the crossings, not a samples-per-member row: no `count` row is written.
// R = |S| / n and the unbiased rho2 are derived at read.
Json store_crossings(nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Value& v, const Array& a, const std::string& by, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string sel = over_selection(sub, P, v), fid = text(rec, "id");
  const std::vector<std::string> names = measure_rows("tile_crossings");
  if (a.comps != 4 || static_cast<std::int64_t>(a.data.size()) != a.rows * a.cols * 4) refuse("outputs." + o + ": the crossings' array is not [rows x members x 4]");
  if (!measured_by(sub.db(), sel, v, fid).empty()) {
    if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + ": its measurements already exist; pass --replace");
    for (const auto& n : names) nxr::io::delete_measurements(sub.db(), Json{{"selectionId", sel}, {"ofFieldId", fid}, {"measure", n}});
    replaced = true;
  }
  const std::string path = text(rec, "path") + "_" + nxr::io::safe_name(d.name + "." + o);
  std::vector<float> arr(a.data.begin(), a.data.end());
  sub.write_array(path, nxr::io::view(arr, {a.rows, a.cols, std::int64_t(4)}));
  // THE STAGES' NAMES ride in the rows' `computed_by` (no schema change): " \xc2\xb7 stages burst,onset,..." in the order of the rows' pairs (rows 2i, 2i+1 are stage i); `repeat compute --stage NAME` reads them
  std::string stages;
  for (const auto& sg : v.facts["stages"]) stages += (stages.empty() ? "" : ",") + sg.get<std::string>();
  const std::string by_rows = by + " \xc2\xb7 stages " + stages;
  Json rows = Json::array();
  for (std::int64_t c = 0; c < 4; ++c) rows.push_back(Json{{"selection_id", sel}, {"of_field_id", fid}, {"measure", names[static_cast<std::size_t>(c)]}, {"array_path", path}, {"component", c}, {"computed_by", by_rows}});
  nxr::io::write_measurements(sub.db(), rows, by_rows);
  return Json{{"output", o}, {"measurement", "tile_crossings"}, {"over", v.facts["over"]["selection"]}, {"array", path}, {"rows", a.rows}, {"members", a.cols}, {"bridge_member", a.meta.value("bridge_member", Json())}, {"stages", a.meta["stages_used"]}};
}
// A SPAN SET OUTPUT, stored as the importer stores events (nxr::bst::time_spans, D143): a spans Selection on the recording's time line, a row per event
// [start, stop) s, code = its type through a dictionary named after it, the channel that found it in params_json.channels; how it was made in params_json
// (the Studio's searchAtlas records its query there): the pipeline, its parameters, the detection's thresholds, the cache key.
Json store_spans(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Json& spec, const Value& v, const Array& a, const Json& producer,
                 const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string name = node_name(d, o, P.found), path = nxr::io::node_path(sub.db(), v.manifolds[0], name);
  nxr::pipelines::clear_stored(ds, sub, {{"selection", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& tiles = a.meta["tiles_of"];
  const double w = tiles["width_s"].get<double>(), t0 = tiles["origin"].get<double>();
  std::vector<nxr::bst::TimeSpan> spans;
  Json channels = Json::object();
  for (std::int64_t k = 0; k < a.rows; ++k) {
    const double* r = a.data.data() + k * 4;
    spans.push_back({t0 + r[0] * w, t0 + r[1] * w, 0});
    if (r[2] >= 0) channels[std::to_string(k)] = Json::array({static_cast<std::int64_t>(r[2])});
  }
  const std::vector<std::string> types = {spec.contains("types") ? spec["types"][0].get<std::string>() : std::string("event")};
  nxr::bst::TimeSpansOptions so;
  so.line_id = v.manifolds[0];
  so.of_field_id = text(rec, "id");
  if (const auto aid = nxr::io::opt_text(rec, "acquisition_id")) so.acquisition_id = *aid;
  so.description = spec.contains("description") ? spec["description"].get<std::string>()
                                                 : "the " + types[0] + " events of " + text(rec, "name") + " \xe2\x80\x94 pipeline " + key.pipeline + "; code = the event type";
  Json params = Json{{"pipeline", producer["pipeline"]}, {"parameters", producer["parameters"]}, {"inputs", producer["inputs"]}, {"run", producer["run"]},
                     {"detection", a.meta}, {"cache", key.json()}};
  if (!channels.empty()) params["channels"] = channels;
  so.params = params;
  const std::string id = nxr::bst::time_spans(sub, path, spans, types, so);
  return Json{{"output", o}, {"selection", name}, {"path", path}, {"id", id}, {"spans", static_cast<std::int64_t>(spans.size())}};
}
// A LABELLING OUTPUT (spec docs/specs/2026-11-13-cortical-events.md, phase 2b): a `set` Selection on a product of the run (here the joint partition of space x time), array-backed: an int32 array in the layout the
// step gives it ([spatial tile x time tile]: the cell layout), the event's ordinal for a cell in one, -1 for none (a negative code is in no member); `n_members` the events. How it was made in params_json:
// the pipeline, its parameters, the inputs, the run, the step's own record (thresholds, baseline, the bursts and their events), the bursts' Selection it was found against, and the cache key.
Json store_labelling(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Value& v, const Array& a, const Json& producer, const nxr::pipelines::CacheKey& key,
                     bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const std::string name = node_name(d, o, P.found), path = nxr::io::node_path(sub.db(), v.manifolds[0], name);
  nxr::pipelines::clear_stored(ds, sub, {{"selection", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  std::vector<std::int32_t> codes(a.data.size());
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::int32_t>(a.data[i]);
  // the bursts it was found against: the spans its OWN step took as `events` (a run over many pairs has one spans output a pair), when that is an output
  Json bursts = nullptr;
  for (const Planned& p : P.steps) {
    bool made = false;
    for (const auto& [port, name] : p.outs) made = made || name == o;
    const auto ev = p.ins.find("events");
    if (!made || ev == p.ins.end()) continue;
    if (!d.outputs.contains(ev->second) || P.values.at(ev->second).cls != "spans") break;
    const std::string bp = nxr::io::node_path(sub.db(), P.values.at(ev->second).manifolds[0], node_name(d, ev->second, P.found));
    if (const auto row = sub.find("selection", bp)) bursts = Json{{"output", ev->second}, {"selection", text(*row, "name")}, {"id", text(*row, "id")}};
    break;
  }
  const std::int64_t events = a.meta.value("events", std::int64_t(0));
  Json params = Json{{"pipeline", producer["pipeline"]}, {"parameters", producer["parameters"]}, {"inputs", producer["inputs"]}, {"run", producer["run"]}, {"extent", a.meta}, {"bursts", bursts},
                     {"cells_of", v.facts["cells"]}, {"cache", key.json()}};
  // a `set` with an array needs its dictionary (the model's rule): one entry per event, named after it; the dictionary is of this key (another choice of options makes another)
  nxr::io::DictionarySpec dsp;
  dsp.name = name + " events " + nxr::io::sha256_hex(key.json().dump()).substr(0, 8); dsp.scope = "subject";
  dsp.description = "the events of " + name + " (code = the event's ordinal: by its burst, then by its lowest cell)";
  dsp.entries = std::vector<nxr::io::DictionaryEntry>();
  for (std::int64_t k = 0; k < events; ++k) {
    nxr::io::DictionaryEntry e;
    e.code = k; e.ordinal = k; e.name = "event " + std::to_string(k);
    if (a.meta.contains("event_list") && static_cast<std::size_t>(k) < a.meta["event_list"].size()) e.cols.json("attributes_json", a.meta["event_list"][static_cast<std::size_t>(k)]);
    dsp.entries->push_back(e);
  }
  const nxr::io::DictionaryResult dr = sub.dictionary(dsp);
  nxr::io::SelectionSpec ss;
  ss.name = name; ss.path = path; ss.type = "set"; ss.manifold_id = v.manifolds[0]; ss.data = nxr::io::view(codes, {a.rows, a.cols});
  ss.cols.set("dictionary_id", dr.id).set("cell", "0,1").set("label", std::to_string(events) + " events")
      .set("description", "the space x time extent of each global burst: code = the event's ordinal, -1 for a cell in none (pipeline " + key.pipeline + "); the cells are [spatial tile x time tile]")
      .json("params_json", params);
  const std::string id = sub.selection(std::move(ss));
  return Json{{"output", o}, {"selection", name}, {"path", path}, {"id", id}, {"events", events}};
}
// A STATISTIC OUTPUT (summarize, spec docs/specs/2026-11-13-statistic-summaries.md): the summaries of a measurement of the run over its partition's members,
// beside the recording; rows' axis the recording's rows (channels) per row, else what the rows were combined into; partition the measured level
Json store_statistic(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Value& v, const Array& a, const Json& producer,
                     const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& of = a.meta["of"];
  const std::string combine = a.meta.value("combine", std::string("each"));
  Json rows_axis = combine == "each" && of.value("joint", false) ? Json{{"name", "spatial_tile"}, {"selection_id", Json::parse(text(*sub.db().read("selection", of["selection_id"].get<std::string>()), "params_json"))["joint"]["space"]}, {"size", a.rows}}   // the rows of a joint partition's measurement are the tiles of its spatial level
                 : combine == "each" ? Json{{"name", "row"}, {"manifold_id", nxr::io::finder::factors(sub.db(), text(rec, "manifold_id"))[0]}, {"size", a.rows}}
                                     : Json{{"name", "row"}, {"values", Json::array({Json{{"name", combine}}})}, {"size", a.rows}};
  std::int64_t least = of.value("tiles", a.cols);
  for (const auto& u : a.meta["members_used"]) least = std::min<std::int64_t>(least, u.get<std::int64_t>());
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = v.facts.value("measure", std::string("amplitude")); st.method = "summary";
  st.of_field_id = text(rec, "id");
  if (of.contains("selection_id")) st.partition_selection_id = of["selection_id"].get<std::string>();
  else if (of.value("events", false)) {                 // the members are the run's own events (stored before): their Selection is the partition
    const std::string eo = events_output(P, events_class(of));
    const Value& sp = P.values.at(eo);
    if (const auto row = sub.find("selection", nxr::io::node_path(sub.db(), sp.manifolds[0], node_name(d, eo, P.found)))) st.partition_selection_id = text(*row, "id");
  }
  st.n_members = a.meta.value("members", std::int64_t(0)); st.members_skipped = st.n_members.value_or(0) - least;
  st.axes = Json::array({rows_axis, Json{{"name", "summary"}, {"values", a.meta["summaries"]}, {"size", a.cols}}});
  st.data = nxr::io::view(a.data, {a.rows, a.cols});
  st.unit = nxr::io::opt_text(rec, "unit");
  st.description = "the " + st.measure + " of " + text(rec, "name") + " over the members of " + of.value("selection", std::string("its partition")) + ", summarised (" + combine + ") \xe2\x80\x94 pipeline " + key.pipeline;
  // the span and its GOOD SECONDS (the span's samples less the bad ones, over the rate): so a rate (count / good_s), an occupancy (Σ duration / good_s) and
  // a coefficient of variation are derived at read, never stored as reductions
  Json span = producer["run"].value("span", Json());
  double good_s = std::numeric_limits<double>::quiet_NaN();
  const double rate = P.found.at(*d.over).rate;
  if (span.is_object() && rate > 0) good_s = (span["stop_sample"].get<double>() - span["start_sample"].get<double>() - producer["run"].value("bad_samples", 0.0)) / rate;
  st.options = Json{{"summaries", a.meta["summaries"]}, {"combine", combine}, {"ddof", a.meta["ddof"]}, {"span", span}, {"good_s", std::isfinite(good_s) ? Json(good_s) : Json()}};
  Json pj = producer;
  pj["output"] = o; pj["members_used"] = a.meta["members_used"];
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}};
}
// A HISTOGRAM STATISTIC OUTPUT (`core_histogram`; nxr::histogram): counts in fixed log bins over the burst frames, pooled over the vertices, beside the recording: measure count, method histogram, the bins' range
// (lo, hi, bins) and what the distribution was made from (the chain) in its options, so a reader derives any percentile from the counts (the cores' threshold: the 98th) and the cores' step checks the chain
Json store_histogram(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Array& a, const Json& producer,
                     const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& h = a.meta["histogram"];
  const std::int64_t bins = h["bins"].get<std::int64_t>();
  nxr::histogram::LogBins lb;
  lb.lo = h["lo"].get<double>(); lb.hi = h["hi"].get<double>(); lb.bins = static_cast<int>(bins);
  Json edges = Json::array();
  for (int k = 0; k < lb.bins; ++k) edges.push_back(nxr::histogram::upperEdge(lb, k));
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = "count"; st.method = "histogram";
  st.of_field_id = text(rec, "id");
  st.n_members = h["frames"].get<std::int64_t>();
  st.axes = Json::array({Json{{"name", "pool"}, {"values", Json::array({Json{{"name", "every vertex of the surface, every good frame"}}})}, {"size", 1}},
                         Json{{"name", "bin"}, {"edges", edges}, {"scale", "log"}, {"size", bins}}});
  st.data = nxr::io::view(a.data, {1, bins});
  st.description = "the distribution of |x| over " + std::to_string(st.n_members.value_or(0)) + " good frames of " + text(rec, "name") + " in " + std::to_string(bins) + " log bins, pooled over the vertices \xe2\x80\x94 pipeline " + key.pipeline;
  Json opt = h;
  opt["edges_axis"] = "the bin axis's `edges` are the bins' UPPER edges";
  opt["percentile"] = "read: the upper edge of the first bin whose cumulative share reaches p (nxr::histogram::percentile)";
  opt["frames_in_range"] = std::min<std::int64_t>(h["range_frames"].get<std::int64_t>(), h["frames"].get<std::int64_t>());
  st.options = opt;
  Json pj = producer;
  pj["output"] = o;
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}, {"frames", h["frames"]}, {"lo", h["lo"]}, {"hi", h["hi"]}};
}
// A GENERIC STATISTIC OUTPUT (an array a step describes itself: `meta.generic_statistic` = {measure, method, shape, axes, components, n_members, description, options, unit}): the envelope's tile series (`power_series`), a null draw's
// tile crossings (`tile_crossings` as a statistic), beside the recording. The step knows what its values are; the runner stores them with the run's provenance and checks the shape against the axes.
Json store_generic(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Array& a, const Json& producer,
                   const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& g = a.meta["generic_statistic"];
  std::vector<std::int64_t> shape;
  for (const auto& e : g["shape"]) shape.push_back(e.get<std::int64_t>());
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = g["measure"].get<std::string>(); st.method = g["method"].get<std::string>();
  st.of_field_id = text(rec, "id");
  st.axes = g["axes"];
  if (g.contains("components")) st.components = g["components"];
  if (g.contains("n_members")) st.n_members = g["n_members"].get<std::int64_t>();
  if (g.contains("unit") && g["unit"].is_string()) st.unit = g["unit"].get<std::string>();
  st.data = nxr::io::view(a.data, shape);
  st.description = g.value("description", std::string("a statistic of ") + text(rec, "name")) + " \xe2\x80\x94 pipeline " + key.pipeline;
  st.options = g.value("options", Json::object());
  Json pj = producer;
  pj["output"] = o;
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}};
}
// A BASELINE STATISTIC OUTPUT (`baseline`; spatial-events spec, piece 1): the mean band power of each vertex (and hat level) over the run outside the named bursts and the bad segments, beside the recording: measure power,
// method mean, [vertex × level] with the vertex axis on the (glued) surface, so it is a stack of maps a reader slices by level; the chain it is of (band, spatial band, levels, form, glue, tiling) in its options, which
// `relative` checks before it divides by it
Json store_baseline(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Value& v, const Array& a, const Json& producer,
                    const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& h = a.meta["baseline"];
  const Json levels = v.facts.contains("levels") && v.facts["levels"].is_array() ? v.facts["levels"] : Json::array();
  const std::int64_t L = levels.empty() ? 1 : static_cast<std::int64_t>(levels.size()), V = a.rows / L;
  Json entries = Json::array();
  if (levels.empty()) entries.push_back(Json{{"name", "band"}});
  else for (const auto& l : levels) entries.push_back(Json{{"name", "level" + std::to_string(l.get<int>())}, {"level", l}});
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = "power"; st.method = "mean";
  st.of_field_id = text(rec, "id");
  st.n_members = h["columns"].get<std::int64_t>();
  st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", v.manifolds[0]}, {"size", V}}, Json{{"name", "level"}, {"values", entries}, {"size", L}}});
  st.data = nxr::io::view(a.data, {V, L});
  st.unit = nxr::io::opt_text(rec, "unit");
  if (st.unit) st.unit = "(" + *st.unit + ")^2";
  st.description = "the baseline band power of " + text(rec, "name") + ": the mean of |x|^2 over " + std::to_string(st.n_members.value_or(0)) + " good " + h.value("axis", std::string("sample")) + " (outside the named bursts and the bad segments), per vertex and level \xe2\x80\x94 pipeline " + key.pipeline;
  Json opt = h;
  opt["layout"] = "[vertex x level]: row = vertex * levels + level; the mean of the `power` step's |x|^2 over the good columns";
  opt["outside"] = v.facts.value("outside", Json());
  st.options = opt;
  Json pj = producer;
  pj["output"] = o;
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}, {"columns", h["columns"]}, {"vertices", V}, {"levels", L}};
}
// A HOME MAP STATISTIC OUTPUT (`patches`, output home; spatial-events spec, piece 6): per glued vertex, the good frames it lies inside a patch and the good frames, measure count, method occupancy, [vertex × (in_patch, frames)]:
// counts, additive over frames, so the fraction (frames in a patch / frames) is DERIVED at read, never stored
Json store_home(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Value& v, const Array& a, const Json& producer,
                const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& h = a.meta["home_map"];
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = "count"; st.method = "occupancy";
  st.of_field_id = text(rec, "id");
  st.n_members = h["frames"].get<std::int64_t>();
  st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", v.manifolds[0]}, {"size", a.rows}},
                         Json{{"name", "count"}, {"values", Json::array({Json{{"name", "frames_in_patch"}}, Json{{"name", "frames"}}})}, {"size", 2}}});
  st.data = nxr::io::view(a.data, {a.rows, 2});
  st.description = "the home map of " + text(rec, "name") + ": per vertex, the burst frames it lies in a spatial event and the burst frames (" + std::to_string(st.n_members.value_or(0)) + ") \xe2\x80\x94 pipeline " + key.pipeline;
  Json opt = h;
  opt["fraction"] = "read: frames_in_patch / frames";
  opt["chain"] = v.facts.value("chain", Json());
  st.options = opt;
  Json pj = producer;
  pj["output"] = o;
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}, {"frames", h["frames"]}};
}
}  // namespace

Json execute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const RunOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  const Definition& d = P.definition;
  const auto anchor = d.anchor();
  // THE SPAN (the chunk service's request: chunk 0 by default) and the bad spans the accumulate steps exclude
  std::optional<nxr::io::Span> span;
  std::optional<nxr::io::ChunkSource> source;
  Json span_json = nullptr;
  const auto loop = P.section("do");
  if (d.over) {
    const Value& over = P.found.at(*d.over);
    const auto line = sub.db().one("SELECT n_vertices FROM manifold WHERE id = ?", {V(*over.time)});
    span = nxr::pipelines::span_of(*over.row, line ? nxr::io::integer(*line, "n_vertices") : 0, over.rate, over.origin, opt.chunks, opt.start, opt.stop);
    span_json = nxr::io::span_record(*span, over.rate, over.origin);
    std::optional<std::string> exclude_id;
    std::optional<std::vector<std::pair<double, double>>> bad;
    for (const Planned* p : loop) {
      if (p->step->kind != "accumulate" || p->params.value("bad_segments", std::string()) != "exclude") continue;
      // the spans a step leaves out: its `exclude` (the bad segments) and its `outside` (named bursts: the run outside them), one union a run
      std::string ids;
      std::vector<std::pair<double, double>> spans;
      for (const char* port : {"exclude", "outside"}) {
        if (!p->ins.count(port) || !P.values.count(p->ins.at(port))) continue;
        const Value& ex = P.values.at(p->ins.at(port));
        ids += (ids.empty() ? "" : "+") + text(*ex.row, "id");
        for (const auto& sp : spans_of(sub.db(), *ex.row)) spans.push_back(sp);
      }
      if (ids.empty()) continue;
      if (exclude_id && *exclude_id != ids) refuse("steps: two accumulate steps exclude different spans; one exclusion a run is built");
      exclude_id = ids;
      std::sort(spans.begin(), spans.end());
      bad = spans;
    }
    source.emplace(nxr::pipelines::source_of(sub, *over.row, over.rate, over.origin, {}, bad));
  }
  // THE CACHE KEY: the definition, its parameters, the inputs, the span (a rerun with all four the same is a hit)
  nxr::pipelines::CacheKey key;
  key.pipeline = d.name + "@" + std::to_string(d.version);
  key.params = Json{{"parameters", P.parameters}, {"definition_sha256", nxr::io::sha256_hex(d.text)}};
  key.params["block_samples"] = opt.block_samples;   // the pointwise tail's sub-block width: the same values up to the rounding of the folds' sums, so in the key (method rule 3)
  for (const Planned* p : loop) if (p->step->frame_place) key.params["block_frames"] = opt.block_frames;   // likewise the frames' (only a run with a frame tiling has one)
  {
    // what each step's request RESOLVES to on what is stored (reading only): the choice made, not only the request, is in the key (method rule 3)
    Json resolved = Json::object();
    for (const Planned& p : P.steps) {
      if (!p.step->resolve) continue;
      Values ins;
      for (const auto& [port, t] : p.ins) if (P.values.count(t)) ins[port] = P.values.at(t);
      const Json r = p.step->resolve(ins, p.params, sub);
      if (!r.is_null()) resolved[p.step->name + ":" + p.use.outs.front()] = r;
    }
    if (!resolved.empty()) key.params["resolved"] = resolved;
  }
  Json ids = Json::object();
  for (const auto& [k, v] : P.found) ids[k] = text(*v.row, "id");
  key.input = Json{{"inputs", ids}, {"span", span_json}};
  key.compute = NXR_COMPUTE_VERSION;
  Json stored = Json::array();
  bool hit = true;
  const std::string measured_id = d.over ? text(*P.found.at(*d.over).row, "id") : std::string();
  for (const auto& [o, _] : d.outputs.items()) {
    const Value& v = P.values.at(o);
    if (v.cls == "measurement") {
      if (no_events(sub, P, v)) { stored.push_back(Json{{"output", o}, {"measurement", v.facts["measure"]}, {"events", 0}}); continue; }
      const auto by = measured_by(sub.db(), over_selection(sub, P, v), v, measured_id);
      if (by.size() != 1 || by[0] != made_by(key)) { hit = false; break; }
      stored.push_back(Json{{"output", o}, {"measurement", v.facts["measure"]}, {"over", v.facts["over"]["selection"]}});
      continue;
    }
    if (v.cls == "statistic") {
      const auto row = nxr::pipelines::find_cached(sub.db(), sub.id(), "statistic", key, statistic_out_path(P, o));
      if (!row) { hit = false; break; }
      stored.push_back(Json{{"output", o}, {"statistic", text(*row, "name")}, {"id", text(*row, "id")}});
      continue;
    }
    if (v.cls == "spans" || v.cls == "labelling") {
      const std::string path = nxr::io::node_path(sub.db(), v.manifolds[0], node_name(d, o, P.found));
      const auto row = nxr::pipelines::find_cached(sub.db(), sub.id(), "selection", key, path);
      if (!row) { hit = false; break; }
      stored.push_back(Json{{"output", o}, {"selection", text(*row, "name")}, {"id", text(*row, "id")}, {"spans", nxr::io::integer(*row, "n_members")}});
      continue;
    }
    const std::string mid = v.manifolds.size() == 1 ? v.manifolds[0] : std::string();
    const auto row = mid.empty() ? nxr::pipelines::find_cached(sub.db(), sub.id(), "field", key)
                                 : nxr::pipelines::find_cached(sub.db(), sub.id(), "field", key, nxr::io::node_path(sub.db(), mid, node_name(d, o, P.found)));
    if (!row || text(*row, "name") != node_name(d, o, P.found)) { hit = false; break; }
    stored.push_back(Json{{"output", o}, {"path", text(*row, "path")}, {"id", text(*row, "id")}});
  }
  Json report = Json{{"pipeline", key.pipeline}, {"subject", sub.name()}};
  if (anchor) report["anchor"] = text(*P.found.at(*anchor).row, "name");
  if (hit) {
    report["hit"] = true;
    report["stored"] = stored;
    return report;
  }
  // another job's outputs of the same names: refused before any data is read (unless replace)
  if (!opt.replace)
    for (const auto& [o, _] : d.outputs.items()) {
      const Value& v = P.values.at(o);
      if (v.cls == "measurement") {
        if (over_events(v)) continue;                           // over the run's own spans Selection: refused with it
        if (!measured_by(sub.db(), over_selection(sub, P, v), v, measured_id).empty())
          throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + ": measurements '" + v.facts["measure"].get<std::string>() + "' of " +
                               text(*P.found.at(*d.over).row, "name") + " over " + v.facts["over"]["selection"].get<std::string>() + " already exist, made by another job; pass --replace");
        continue;
      }
      if (v.cls == "statistic") {
        if (sub.find("statistic", statistic_out_path(P, o)))
          throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + " (" + statistic_out_path(P, o) + ") already exists, made with other parameters, inputs or span; pass --replace");
        continue;
      }
      if (v.cls == "spans" || v.cls == "labelling") {
        const std::string path = nxr::io::node_path(sub.db(), v.manifolds[0], node_name(d, o, P.found));
        if (sub.find("selection", path))
          throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + " (" + path + ") already exists, made with other parameters, inputs or span; pass --replace");
        continue;
      }
      if (v.manifolds.size() != 1) continue;
      const std::string path = nxr::io::node_path(sub.db(), v.manifolds[0], node_name(d, o, P.found));
      if (sub.find("field", path))
        throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + " (" + path + ") already exists, made with other parameters, inputs or span; pass --replace");
    }

  // BEFORE: each step's constants bound once (prepare), then the setup steps
  ExecContext ctx{sub, &ds, progress};
  std::vector<std::any> states(P.steps.size());
  for (std::size_t i = 0; i < P.steps.size(); ++i) {
    const Planned& p = P.steps[i];
    if (!p.step->prepare) continue;
    Values ins;
    for (const auto& [port, t] : p.ins) if (P.values.count(t)) ins[port] = P.values.at(t);
    NodeCall call = NodeCall::prepare(ins, p.params, ctx);
    dispatch(*p.step, call);
    states[i] = std::move(call.made_state);
  }
  auto index_of = [&](const Planned* p) { return static_cast<std::size_t>(p - P.steps.data()); };
  // what each step's preparation resolved (a degree chosen from a tolerance …): reported by the step (`record`), generic, into the provenance below
  Json prepared = Json::object();
  for (std::size_t i = 0; i < P.steps.size(); ++i)
    if (P.steps[i].step->record && states[i].has_value()) prepared[P.steps[i].step->name + ":" + P.steps[i].use.outs.front()] = P.steps[i].step->record(states[i]);
  // what a step's preparation did or fell back on (a spatial band's Chebyshev fallback, eigenmodes it computed) is said, and kept in the report and the provenance: never silent
  Json warnings = Json::array(), messages = Json::array();
  for (const auto& [k, rec] : prepared.items()) {
    if (!rec.is_object()) continue;
    const auto& sb = rec.contains("spatial_band") ? rec["spatial_band"] : rec;
    auto once = [](Json& list, const Json& x) { for (const auto& y : list) if (y == x) return; list.push_back(x); };   // a level read by several bands is said once
    if (sb.is_object() && sb.contains("warnings") && sb["warnings"].is_array()) for (const auto& w : sb["warnings"]) once(warnings, w);
    if (sb.is_object() && sb.contains("messages") && sb["messages"].is_array()) for (const auto& m : sb["messages"]) once(messages, m);
  }
  Arrays data;
  auto arrays_in = [&](const Planned& p) { Arrays a; for (const auto& [port, t] : p.ins) if (data.count(t)) a[port] = data.at(t); return a; };
  // a setup or after node: arrays in, arrays out, through the one dispatch point
  auto run_on_arrays = [&](const Planned* p) {
    const Arrays ins = arrays_in(*p);
    NodeCall call = NodeCall::run(states[index_of(p)], ins, ctx);
    dispatch(*p->step, call);
    for (auto& [k, v] : call.made_arrays) data[p->outs.at(k)] = std::move(v);
  };
  for (const Planned* p : P.section("before")) run_on_arrays(p);

  // THE CHUNK LOOP: io's chunk service hands each chunk with the composed halo; the chunk steps run on it, the accumulate steps fold it
  Json facts = Json::object();
  if (!loop.empty()) {
    std::vector<const Planned*> chunked;
    for (const Planned* p : loop) if (p->step->kind == "chunk") chunked.push_back(p);
    std::vector<std::int64_t> halos;
    for (const Planned* p : chunked) halos.push_back(p->step->halo ? p->step->halo(states[index_of(p)]) : 0);
    std::map<const Planned*, std::int64_t> keep;                             // the halo a step leaves for the chunk steps after it
    std::int64_t H = 0;
    for (std::size_t i = 0; i < chunked.size(); ++i) {
      std::int64_t after = 0;
      for (std::size_t j = i + 1; j < chunked.size(); ++j) after += halos[j];
      keep[chunked[i]] = after;
      H += halos[i];
    }
    // where the pointwise tail starts: after the last chunk step that needs a halo (the steps after it run on time sub-blocks, RunOptions::block_samples)
    std::size_t tail = 0;
    for (std::size_t i = 0; i < loop.size(); ++i)
      if (loop[i]->step->kind == "chunk" && loop[i]->step->halo && loop[i]->step->halo(states[index_of(loop[i])]) > 0) tail = i + 1;
    std::map<const Planned*, std::any> accs;
    for (const Planned* p : loop) if (p->step->kind == "accumulate") { NodeCall call = NodeCall::start(states[index_of(p)]); dispatch(*p->step, call); accs[p] = std::move(call.made_state); }
    // a block is released after the LAST step that reads it (a chunk step's outputs stay inside the loop): with a run over many pairs only one pair's block is held at a time
    std::map<std::string, std::size_t> last_use;
    for (std::size_t i = 0; i < loop.size(); ++i) for (const auto& [port, t] : loop[i]->ins) last_use[t] = i;
    say(Json{{"stage", "run"}, {"pipeline", key.pipeline}, {"span", span->what}, {"halo", H}, {"steps", static_cast<std::int64_t>(loop.size())}});
    std::int64_t blocks_n = 0;
    const nxr::io::ChunkRun cr = nxr::io::haloed_loop(*source, *span, H, [&](const nxr::io::HaloedChunk& h) {
      ++blocks_n;
      const std::int64_t w = h.left + h.n + h.right;
      Block x{source->rows(), w, h.left, h.right, std::vector<float>(h.data.size()), {}};
      for (std::size_t i = 0; i < h.data.size(); ++i) x.re[i] = static_cast<float>(h.data[i]);   // the stored float32, widened by the service, narrowed back exactly
      Blocks blocks{{*d.over, std::move(x)}};
      std::vector<std::uint8_t> good;
      if (!h.good.empty()) good.assign(h.good.begin() + h.left, h.good.begin() + h.left + h.n);
      // a step run on a set of blocks over `place` (n own columns): a chunk step's outputs join the set, an accumulate step folds; a block is released after its last reader
      auto run_step = [&](std::size_t li, Blocks& set, const Place& at) {
        const Planned* p = loop[li];
        Blocks ins;
        for (const auto& [port, t] : p->ins) if (set.count(t)) ins[port] = set.at(t);
        if (p->step->kind == "chunk") {
          NodeCall call = NodeCall::chunk(states[index_of(p)], ins, keep[p], at);
          dispatch(*p->step, call);
          for (auto& [k, v] : call.made_blocks) set[p->outs.at(k)] = std::move(v);
        } else {
          for (const auto& [port, b] : ins)
            if (b.own() != at.n) refuse(p->step->name + ": " + port + " has " + std::to_string(b.own()) + " own columns in a block of " + std::to_string(at.n) + ": a chunk step's halo bookkeeping is off");
          NodeCall call = NodeCall::fold(states[index_of(p)], accs[p], ins, at);
          dispatch(*p->step, call);
        }
        ins.clear();
        for (const auto& [port, t] : p->ins) { const auto u = last_use.find(t); if (u != last_use.end() && u->second == li) set.erase(t); }
      };
      const Place place{h.first, h.n, &good};
      // a RETIMING step (the frame tiling) moves the chunk's own columns from samples to the frames inside them: the steps after it run once on the chunk's frames (their Place the frames'), never on sample sub-blocks
      std::optional<std::size_t> retime;
      for (std::size_t li = 0; li < loop.size(); ++li) if (loop[li]->step->frame_place) retime = li;
      // a set of blocks cut to the columns [o, o + w) of their own columns (a block's own columns start at its `left`): the pointwise tail's sub-blocks, of samples or of frames
      auto cut = [](const Blocks& set, std::int64_t o, std::int64_t w, std::int64_t own) {
        Blocks sub;
        for (const auto& [t, b] : set) {
          if (b.own() != own) continue;                                  // not on this axis (a block left over from the other)
          Block c{b.rows, w, 0, 0, std::vector<float>(static_cast<std::size_t>(b.rows * w)), {}};
          if (b.complex()) c.im.resize(c.re.size());
          for (std::int64_t r = 0; r < b.rows; ++r) {
            const std::size_t from = static_cast<std::size_t>(r * b.width + b.left + o), to = static_cast<std::size_t>(r * w);
            std::copy_n(b.re.begin() + from, w, c.re.begin() + to);
            if (b.complex()) std::copy_n(b.im.begin() + from, w, c.im.begin() + to);
          }
          sub[t] = std::move(c);
        }
        return sub;
      };
      if (retime) {
        if (tail != *retime + 1) refuse(loop[*retime]->step->name + ": a retiming step must be the last step that needs a halo (a step after it needs one: " + loop[tail - 1]->step->name + "); the steps after it are pointwise on its frames");
        for (std::size_t li = 0; li <= *retime; ++li) run_step(li, blocks, place);
        // the steps after the frame tiling are pointwise in time: they run on BLOCKS OF FRAMES of the chunk's frames, in frame order (RunOptions::block_frames), never on the whole chunk's frames at once
        const FramePlace fp = loop[*retime]->step->frame_place(states[index_of(loop[*retime])], place);
        const std::int64_t B = opt.block_frames > 0 ? opt.block_frames : fp.n;
        for (std::int64_t o = 0; *retime + 1 < loop.size() && o < fp.n; o += B) {
          const std::int64_t w = std::min(B, fp.n - o);
          Blocks sub = cut(blocks, o, w, fp.n);
          std::vector<std::uint8_t> g;
          if (!fp.good.empty()) g.assign(fp.good.begin() + o, fp.good.begin() + o + w);
          const Place at{fp.first + o, w, &g};
          for (std::size_t li = *retime + 1; li < loop.size(); ++li) run_step(li, sub, at);
        }
        return;
      }
      for (std::size_t li = 0; li < tail; ++li) run_step(li, blocks, place);
      // THE POINTWISE TAIL on time sub-blocks of the chunk's own columns, in time order (no step after `tail` needs a halo): the same values, a block's worth of memory
      const std::int64_t B = opt.block_samples > 0 ? opt.block_samples : h.n;
      for (std::int64_t o = 0; tail < loop.size() && o < h.n; o += B) {
        const std::int64_t w = std::min(B, h.n - o);
        Blocks sub = cut(blocks, o, w, h.n);
        std::vector<std::uint8_t> g;
        if (!good.empty()) g.assign(good.begin() + o, good.begin() + o + w);
        const Place at{h.first + o, w, &g};
        for (std::size_t li = tail; li < loop.size(); ++li) run_step(li, sub, at);
      }
    });
    for (auto& [p, acc] : accs) {
      NodeCall call = NodeCall::finish(states[index_of(p)], acc);
      dispatch(*p->step, call);
      for (auto& [k, v] : call.made_arrays) data[p->outs.at(k)] = std::move(v);
    }
    facts = Json{{"span", span_json}, {"halo", H}, {"blocks", blocks_n}, {"block_samples", opt.block_samples}, {"block_frames", opt.block_frames}, {"chunks_read", cr.visited}, {"samples_read", cr.samples}, {"bad_samples", cr.bad_samples}};
  }
  // AFTER
  for (const Planned* p : P.section("after")) run_on_arrays(p);

  // STORE: each output a Field on the manifold its steps made, its provenance and the cache key with it; another job's node refused unless replace (moved aside)
  Json producer = Json{{"pipeline", Json{{"name", d.name}, {"version", d.version}, {"definition", d.source}, {"sha256", nxr::io::sha256_hex(d.text)}}},
                       {"parameters", P.parameters}, {"inputs", nxr::io::finder::describe(P.found)}, {"run", facts}, {"prepared", prepared}, {"warnings", warnings}, {"messages", messages}, {"nxr_neuro", NXR_NEURO_VERSION}};
  const Json stamped = nxr::pipelines::stamp(key);                         // held: iterating a temporary's items() would dangle
  for (const auto& [k, v] : stamped.items()) producer[k] = v;
  stored = Json::array();
  bool replaced = false;
  sub.compose([&] {
    // the measurement rows name how they were made (`computed_by`: the pipeline and its key's hash); the RECIPE it names is an entry of the dataset's `recipes` dictionary
    // (as `measure compute`'s) carrying the whole record: parameters, inputs, and what preparation resolved
    bool any_measurement = false;
    for (const auto& [o, _] : d.outputs.items()) any_measurement = any_measurement || P.values.at(o).cls == "measurement";
    if (any_measurement) {
      nxr::io::DictionaryEntry e;
      e.name = "pipeline " + key.pipeline + " \xc2\xb7 cache " + nxr::io::sha256_hex(key.json().dump()).substr(0, 16);
      e.cols.set("function", std::string("pipeline"));
      e.cols.set("params_json", nxr::io::jdump(producer));
      nxr::io::DictionarySpec spec;
      spec.name = "recipes"; spec.scope = "dataset"; spec.description = "the recipes of this dataset: one entry a version, the recipe in params_json"; spec.entries = std::vector<nxr::io::DictionaryEntry>{e};
      sub.dictionary(spec);
    }
    // FOUR PASSES: what is stored on its own (Fields, tile measurements, spans), then the labellings (named against the spans just stored), then what is over the events just stored (the events'
    // measurements), then the statistics (whose members may be those events)
    std::set<std::string> packed;                                          // the measurement outputs already stored with an earlier one's array
    auto pass = [](const Value& v) { return v.cls == "labelling" ? 1 : over_events(v) ? 2 : v.cls == "statistic" ? 3 : 0; };
    for (int ps = 0; ps < 4; ++ps)
    for (const auto& [o, spec] : d.outputs.items()) {
      const Value& v = P.values.at(o);
      if (pass(v) != ps) continue;
      if (v.cls == "measurement" && v.facts.value("measure", std::string()) == "tile_crossings") { stored.push_back(store_crossings(sub, P, o, v, data.at(o), made_by(key), opt.replace, replaced)); continue; }
      if (v.cls == "measurement") {
        if (packed.count(o)) continue;                                      // stored with the first of its group
        // the outputs packed with this one: the same pass, the same Selection, rows and members, and the same counts (store_measurements)
        std::vector<std::string> group{o};
        const std::string sel0 = over_selection(sub, P, v);
        const Array& a0 = data.at(o);
        for (const auto& [o2, _2] : d.outputs.items()) {
          const Value& v2 = P.values.at(o2);
          if (o2 == o || packed.count(o2) || v2.cls != "measurement" || pass(v2) != ps || v2.facts.value("measure", std::string()) == "tile_crossings" || sel0.empty() || a0.cols == 0) continue;
          const Array& a2 = data.at(o2);
          if (over_selection(sub, P, v2) == sel0 && a2.rows == a0.rows && a2.cols == a0.cols && a2.meta["counts"] == a0.meta["counts"]) group.push_back(o2);
        }
        for (const auto& g : group) packed.insert(g);
        for (auto& j : store_measurements(sub, P, group, data, made_by(key), opt.replace, replaced)) stored.push_back(std::move(j));
        continue;
      }
      if (v.cls == "labelling") { stored.push_back(store_labelling(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)); continue; }
      if (v.cls == "spans") { stored.push_back(store_spans(ds, sub, P, o, spec, v, data.at(o), producer, key, opt.replace, replaced)); continue; }
      if (v.cls == "statistic") { stored.push_back(data.at(o).meta.contains("generic_statistic") ? store_generic(ds, sub, P, o, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("histogram") ? store_histogram(ds, sub, P, o, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("home_map") ? store_home(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("baseline") ? store_baseline(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)
                                                                                                  : store_statistic(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)); continue; }
      const std::string mid = v.manifolds.size() == 1 ? v.manifolds[0] : sub.product(v.manifolds);
      const auto m = sub.db().one("SELECT name, n_vertices FROM manifold WHERE id = ?", {V(mid)});
      const std::string node = node_name(d, o, P.found), path = nxr::io::node_path(sub.db(), mid, node);
      nxr::pipelines::clear_stored(ds, sub, {{"field", path}}, opt.replace, d.name + " output " + o + " (" + path + ")", replaced);
      const Array& a = data.at(o);
      const std::int64_t comps = components(v), n = m ? nxr::io::integer(*m, "n_vertices") : 0;
      if (a.rows * a.cols != n * comps) refuse("outputs." + o + ": " + std::to_string(a.rows * a.cols) + " values for the " + std::to_string(n) + " vertices of " + (m ? text(*m, "name") : mid));
      std::vector<float> f(a.data.begin(), a.data.end());
      nxr::io::FieldSpec fs;
      fs.name = node; fs.path = path; fs.kind = spec["kind"].get<std::string>(); fs.manifold_id = mid; fs.value_type = v.value_type;
      fs.data = comps == 1 ? nxr::io::view(f, {n}) : nxr::io::view(f, {n, comps});
      fs.array.compress = false;
      if (spec.contains("unit")) fs.cols.set("unit", spec["unit"].get<std::string>());
      fs.cols.set("description", spec.contains("description") ? spec["description"].get<std::string>() : spec["kind"].get<std::string>() + " — pipeline " + key.pipeline)
          .set("source", "pipeline " + key.pipeline + " · nxr-neuro " + NXR_NEURO_VERSION);
      if (anchor) fs.cols.set("derived_from_id", text(*P.found.at(*anchor).row, "id"));
      Json pj = producer;
      pj["output"] = o;
      // a tangent vector field (a complex field's form, the phase current) is stored as a vector3 like a free-orientation current: the Studio's type says which (backend valueType.ts algebraOf)
      if (v.value_type == "vector3" && (v.facts.value("field_type", std::string()) == "complex" || v.facts.value("made_by", std::string()) == "phase_current")) pj["studio"] = Json{{"type", "complex"}};
      fs.cols.json("producer_json", pj);
      const std::string fid = sub.field(std::move(fs));
      stored.push_back(Json{{"output", o}, {"path", path}, {"id", fid}, {"on", m ? text(*m, "name") : mid}});
    }
  });
  report["hit"] = false;
  report["replaced"] = replaced;
  report["run"] = facts;
  report["prepared"] = prepared;
  report["warnings"] = warnings;
  report["messages"] = messages;
  report["stored"] = stored;
  return report;
}

std::vector<Json> run(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Definition& d, const Json& set, const RunOptions& opt, bool dry_run,
                      const std::function<void(const Json&)>& progress) {
  const auto plans = plan_runs(d, sub.db(), sub.id(), set);                 // every run planned before any is executed
  std::vector<Json> out;
  for (const auto& P : plans) {
    if (dry_run) {
      Json r = Json{{"pipeline", d.name + "@" + std::to_string(d.version)}, {"plan", P.text(sub.db())}, {"inputs", nxr::io::finder::describe(P.found)}, {"parameters", P.parameters}};
      if (const auto a = d.anchor()) r["anchor"] = text(*P.found.at(*a).row, "name");
      out.push_back(r);
    } else {
      out.push_back(execute(ds, sub, P, opt, progress));
    }
  }
  return out;
}

}  // namespace nxr::pipelines::graph

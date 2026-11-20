#include "nxr/io/selection_store.h"
#include "nxr/io/fibers.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
// nxr-pipelines — the runner (runner.h): nxr's nxr/pipelines/runner.py `plan` and the planning half of `run`. The execution (before · the chunk loop · after
// · store) is the next PR of the spec's order.
#include "nxr/pipelines/runner.h"

#include "nxr/bst/projectors.h"
#include "nxr/pipelines/input.h"
#include "nxr/bst/session.h"

#include "nxr/bst/entities.h"
#include "nxr/io/measures.h"
#include "nxr/stats/histogram.h"

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
#include "nxr/pipelines/declared_outputs.h"
#include "nxr/graph/execution.h"
#include "nxr/pipelines/step_nodes.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/reading.h"

#include "nxr/io/finder.h"
#include "nxr/io/task/jobs.h"

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
  const auto m = db.read("manifold", id);
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

Plan plan(const Definition& d, const Values& found, const Json& params, const nxr::io::Database& db, const std::string& subject_id, const ChannelFacts& channels) {
  Plan P{d, params, found, found, {}, {}, {}};
  if (d.over && P.values.count(*d.over)) {
    P.values[*d.over].chunked = true;
    if (d.inputs.at(*d.over).value("read", std::string("samples")) == "shape") P.values[*d.over].facts["read"] = "shape";   // the steps see it: only one that reads no sample may take it
  }
  // THE GRAPH the executor checks: the definition's steps as uses of nodes, its found inputs as typed ports (the streamed one is the recording the loop walks), its outputs as the graph's.
  std::vector<std::string> present;
  for (const auto& [k, v] : P.values) {
    present.push_back(k);
    P.types[k] = port_of(v);
    if (channels && v.cls == "field" && v.time && v.row) P.types[k].facts["channels"] = channels(nxr::io::text(*v.row, "name"));   // the recording's channels are facts its nodes read at plan
  }
  P.graph = graph_of(d, params, present);
  const nxr::graph::Graph& g = P.graph;
  // THE CHECK, on types, before any data: arrows, kinds by section, halo, retiming (the executor's). The steps read the subject's records through the plan's binding (nothing stored).
  const Bound bound{task::Subject::of(db, subject_id)};
  const auto binding = std::make_shared<Binding>();
  binding->bound = &bound;
  nxr::graph::Checked checked;
  try {
    checked = nxr::graph::check(g, P.types, Json::object(), node_registry(binding));
  } catch (const std::invalid_argument& e) {
    refuse(std::string("steps: ") + e.what());
  }
  for (const auto& [name, t] : checked.values) P.values[name] = value_of(t);
  P.align = checked.align;
  P.checked = std::make_shared<const nxr::graph::Checked>(checked);
  std::size_t at = 0;
  for (const auto& use : d.uses()) {
    const StepSpec& st = step(use.step);
    Planned pl{use, &st, checked.nodes[at++].params, {}, {}};
    for (const auto& [port, thing] : use.ins.items()) pl.ins[port] = thing.get<std::string>();
    {
      std::vector<std::string> used;
      const std::vector<std::string> only = st.outputs_for ? st.outputs_for(pl.params) : std::vector<std::string>();
      for (const auto& o : st.outputs) if (!st.outputs_for || std::find(only.begin(), only.end(), o) != only.end()) used.push_back(o);
      for (std::size_t i = 0; i < used.size() && i < use.outs.size(); ++i) pl.outs[used[i]] = use.outs[i];
    }
    P.steps.push_back(std::move(pl));
  }
  for (const auto& [o, spec] : d.outputs.items()) {
    const Value& v = P.values.at(o);
    // THREE KINDS OF OUTPUT: a Field (materialized), a measurement over a time partition (reduced: selection_measurement rows), a span set (a spans Selection)
    const std::string kind = spec.value("kind", std::string());
    if (is_declared(P, o)) { check_declared(P, o); continue; }            // a declared output (declared_outputs.h): its kind, links and node checked here, its values at the store
    if (kind == "recording") {                                             // a streamed value written as a new recording, block by block, as the loop runs (a filter's output)
      if (!v.chunked || !d.over) refuse("outputs." + o + ": a recording output is a value inside the chunk loop (what a chunk node makes of the recording)");
      if (!spec.contains("name")) refuse("outputs." + o + ": a recording output names the new recording (name: …)");
      continue;
    }
    const std::string kind_short = kind == "selection" ? v.cls : kind;      // {kind: selection, type: spans | set} are the short forms spans | labelling
    if (v.cls == "measurement" || v.cls == "spans" || v.cls == "statistic" || v.cls == "labelling") {
      if (kind_short != v.cls) refuse("outputs." + o + ": its steps make a " + v.cls + "; declare it {kind: " + v.cls + "}");
      if (!d.over) refuse("outputs." + o + ": a " + v.cls + " is of the recording the chunk loop reads; this definition has no chunk loop");
      if (v.cls == "spans" && spec.contains("types") && !(spec["types"].is_array() && spec["types"].size() == 1 && spec["types"][0].is_string()))
        refuse("outputs." + o + ": types is one event type's name, e.g. [burst]");
      continue;
    }
    if (kind == "measurement" || kind == "spans" || kind == "statistic" || kind == "labelling") refuse("outputs." + o + ": declared " + kind + " but its steps make a " + v.cls);
    if (v.cls != "field" || v.chunked) refuse("outputs." + o + ": not a field outside the chunk loop: only those are stored");
    for (const auto& m : v.manifolds)
      if (!db.read("manifold", m)) refuse("outputs." + o + ": its manifold " + m + " is not among the subject's rows");
  }
  return P;
}

// `with` is written out here ($names and nested ones replaced), so the executor sees values only; the parameters are the run's (defaults, then the caller's).
nxr::graph::Graph graph_of(const Definition& d, const Json& params, const std::vector<std::string>& present) {
  nxr::graph::Graph g;
  g.name = d.name; g.version = d.version; g.text = d.text;
  for (const auto& [k, v] : params.items()) g.parameters[k] = Json{{"default", v}, {"optional", v.is_null()}};   // the run's values; a null one (an optional bound) is "not given", read by the steps
  g.inputs = present;
  for (const auto& [o, _] : d.outputs.items()) g.outputs.push_back(o);
  std::set<std::string> defined(present.begin(), present.end());          // the inputs present, then what each step makes
  for (const auto& use : d.uses()) {
    const StepSpec& st = step(use.step);
    const std::string w = "steps: " + st.name + " (" + label(use.section) + ")";
    nxr::graph::Use u;
    u.node = st.name;
    for (const auto& [port, thing] : use.ins.items()) {
      const std::string t = thing.get<std::string>();
      if (defined.count(t)) u.in[port] = t;
      else if (std::find(st.optional.begin(), st.optional.end(), port) == st.optional.end())
        refuse(w + ": in." + port + " '" + t + "' is absent (an optional input the subject lacks), and the step needs it");
    }
    for (const auto& [k, v] : use.with.items()) u.with[k] = with_value(v, params);
    u.out = use.outs;
    for (const auto& o : use.outs) defined.insert(o);
    (use.section == "before" ? g.before : use.section == "after" ? g.after : g.chunks).push_back(std::move(u));
  }
  return g;
}

Json parameters_of(const Definition& d, const Json& set, const std::string& preset) {
  Json p = d.parameters;
  if (!preset.empty()) {                                                    // defaults < the preset < the caller's own
    if (!d.presets.contains(preset)) {
      std::string have;
      for (const auto& [k, _] : d.presets.items()) have += (have.empty() ? "" : ", ") + k;
      refuse("--set " + preset + ": not a preset of " + d.name + " (" + (have.empty() ? std::string("it names none") : have) + ")");
    }
    for (const auto& [k, v] : d.presets[preset].items()) p[k] = v;
  }
  std::string unknown;
  for (const auto& [k, v] : set.items()) {
    if (!d.parameters.contains(k)) { unknown += (unknown.empty() ? "" : ", ") + k; continue; }
    p[k] = v;
  }
  // A REQUIRED PARAMETER (default `required`): a choice of data the definition never defaults. Refused when the run does not give it, naming it, what it means and its valid values.
  for (const auto& [k, v] : p.items()) {
    if (!(v.is_string() && v.get<std::string>() == "required")) continue;
    const Json doc = d.parameter_docs.contains(k) ? d.parameter_docs[k] : Json::object();
    std::string valid = doc.contains("choices") ? " (valid: " + doc["choices"].dump() + ")" : std::string();
    refuse(d.name + ": the parameter `" + k + "` is required" + (doc.contains("meaning") ? " (" + doc["meaning"].get<std::string>() + ")" : std::string()) + valid + "; give it with --" + k);
  }
  if (!unknown.empty()) {
    std::string have;
    for (const auto& [k, _] : d.parameters.items()) have += (have.empty() ? "" : ", ") + k;
    refuse("--set " + unknown + ": not parameters of " + d.name + " (" + (have.empty() ? std::string("none") : have) + ")");
  }
  return p;
}

// ONE PLAN PER COMBINATION of the lists a definition's header `plans:` names (consolidation spec 4.2: a band set × spatial scales are several plans of one run, each cached on its own). Each plan has the
// element as the single parameter and the list parameter neutralised (null), so a plan's cache key holds its own pair and nothing of the lists it was drawn from. No list given: the one plan as it is.
std::vector<Json> plan_parameter_sets(const Definition& d, const Json& base) {
  std::vector<Json> sets{base};
  for (const auto& [var, src] : d.plans.items()) {
    const std::string list = src.get<std::string>().substr(1);
    const Json& values = base.contains(list) ? base[list] : Json();
    if (values.is_null() || (values.is_array() && values.empty())) continue;
    if (!values.is_array()) refuse(d.name + ": the parameter `" + list + "` is a list (one plan is made for each element)");
    if (!base[var].is_null()) refuse(d.name + ": give `" + var + "` (one) or `" + list + "` (a list), not both");
    std::vector<Json> next;
    for (const Json& s : sets)
      for (const Json& el : values) { Json p = s; p[var] = el; p[list] = nullptr; next.push_back(std::move(p)); }
    sets = std::move(next);
  }
  return sets;
}

// A POINT ORIGIN (rule 34: the origin of frames is an option of the compute function, with the pipeline's default written in its YAML; it may be the recording start, a clock time or an EVENT). A parameter
// declared `chart: origin` is given as `start`, a number (a clock time in seconds), {clock: s}, or {event: {selection: NAME, member: N (default 0), at: start|centre|stop (default start)}}; the run resolves
// it here, from the stored selection, to {kind, seconds, ...} so that the value in the cache key is the point itself and its source. A node reads `kind` and `seconds`; `start` carries no seconds
// (the line's first sample is the node's to know).
Json resolve_origins(const Definition& d, const Json& params, const nxr::io::Database& db, const std::string& subject_id) {
  Json out = params;
  for (const auto& [k, doc] : d.parameter_docs.items()) {
    if (doc.value("chart", std::string()) != "origin" || !params.contains(k) || params[k].is_null()) continue;
    const Json& v = params[k];
    const std::string w = d.name + ": parameter `" + k + "` (an origin)";
    if (v.is_string() && v == "start") { out[k] = Json{{"kind", "start"}}; continue; }
    if (v.is_number()) { out[k] = Json{{"kind", "clock"}, {"seconds", v}}; continue; }
    if (v.is_object() && v.contains("kind") && v.contains("seconds")) continue;          // already resolved (a rerun from a stored recipe)
    if (v.is_object() && v.contains("clock") && v["clock"].is_number()) { out[k] = Json{{"kind", "clock"}, {"seconds", v["clock"]}}; continue; }
    if (v.is_object() && v.contains("event") && v["event"].is_object() && v["event"].contains("selection")) {
      const Json& ev = v["event"];
      const task::Subject subj = task::Subject::of(db, subject_id);
      task::Want want{.cls = task::Cls::selection};
      const std::string event_name = ev["selection"].get<std::string>();
      want.name = event_name;
      const auto sel = subj.find(want);
      if (!sel) refuse(w + ": no selection named '" + event_name + "' (the events the origin is at)");
      const auto spans = subj.spans(*sel);
      const std::int64_t member = ev.value("member", std::int64_t(0));
      if (member < 0 || member >= static_cast<std::int64_t>(spans.size()))
        refuse(w + ": member " + std::to_string(member) + " of '" + event_name + "' does not exist (it has " + std::to_string(spans.size()) + " members; a selection of spans is needed)");
      const std::string at = ev.value("at", std::string("start"));
      if (at != "start" && at != "centre" && at != "stop") refuse(w + ": at is start, centre or stop");
      const auto& m = spans[static_cast<std::size_t>(member)];
      const double t = at == "start" ? m.start : at == "stop" ? m.stop : 0.5 * (m.start + m.stop);
      out[k] = Json{{"kind", "event"}, {"seconds", t}, {"selection", sel->id}, {"selection_name", event_name}, {"member", member}, {"at", at}};
      continue;
    }
    refuse(w + ": start, a number of seconds, {clock: seconds} or {event: {selection: NAME, member: N, at: start|centre|stop}}");
  }
  return out;
}

std::vector<Plan> plan_runs(const Definition& d, const nxr::io::Database& db, const std::string& subject_id, const Json& set, const std::string& preset, const ChannelFacts& channels) {
  const Json base = parameters_of(d, set, preset);
  std::vector<Plan> out;
  for (const Json& each : plan_parameter_sets(d, base)) {
    const Json params = resolve_origins(d, each, db, subject_id);
    Definition e = expand(d, params);                                       // a definition that repeats, decides (`when:`) or composes, written out for this run (flat: unchanged)
    if (!e.draw.empty()) {                                                  // `draw: seed`: every draw is a node of its own (the stored nodes' names, the cache key's pipeline), the parameter's value in the name
      if (!params.contains(e.draw) || !params[e.draw].is_number_integer() || params[e.draw].get<std::int64_t>() < 0) refuse(d.name + ": the parameter `" + e.draw + "` (draw) is a whole number >= 0");
      e.name += ".draw" + std::to_string(params[e.draw].get<std::int64_t>());
    }
    for (const auto& found : nxr::io::finder::resolve(db, subject_id, e.inputs, params)) out.push_back(plan(e, found, params, db, subject_id, channels));   // every run planned before any is executed
  }
  for (auto& P : out) P.preset = preset;
  return out;
}

std::vector<Plan> plan_runs(const Definition& d, nxr::io::SubjectWriter& sub, const Json& set, const std::string& preset) {
  return plan_runs(d, sub.db(), sub.id(), set, preset, [&sub](const std::string& recording) { return nxr::pipelines::channel_facts(sub, recording); });
}

namespace {
using nxr::io::text;
using nxr::io::V;
std::string ospec_kind(const Json& spec) { return spec.value("kind", std::string()); }

// an exclude input's spans [start, stop) in seconds (a recording's bad segments)
std::vector<std::pair<double, double>> spans_of(const nxr::io::Database& db, const nxr::io::Row& selection) {
  return nxr::io::selection::span_bounds(db, text(selection, "id"));
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
  for (const std::string& by : nxr::io::selection::computed_by_over_field(db, selection_id, field_id, measure_rows(v.facts["measure"].get<std::string>())[0]))
  {
    std::string c = by;
    const std::size_t at = c.find(" \xc2\xb7 stages ");              // the crossings' stage names are an annotation of the rows, not part of how they were made (the stages are in the cache key already)
    out.push_back(at == std::string::npos ? c : c.substr(0, at));
  }
  return out;
}
// a measurement over the EVENTS of the run (measure_events, extent_measure): its Selection is the run's own events output, stored first: the span set (the default) or, `events_of: labelling`, the labelling
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
std::vector<Json> store_measurements(nxr::io::SubjectWriter& sub, nxr::pipelines::task::Unit& unit, const Plan& P, const std::vector<std::string>& group, const std::map<std::string, Array>& data, const std::string& by, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string& o0 = group.front();
  const Value& v0 = P.values.at(o0);
  // over the tiles of a level, or (measure_events) over the run's own events, whose Selection was stored just before: the member axis is then the events' order
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
  if (const auto cr = nxr::io::selection::measurement_array(sub.db(), sel, fid, "count")) {
    bool same = false;
    try {
      // the stored count row says which component of which array is the count (the last): read it there, never assume a layout
      nxr::io::ArrayReader rd = nxr::io::ArrayReader::open(sub.store(), cr->array_path);
      const auto sh = rd.shape();
      const std::int64_t cc = cr->component;
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
      if (cr->array_path == path) { unit.remove_measurements(sel, fid, "count"); keep_count = false; }
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
      for (const std::string& r : nxr::io::selection::measures_over_field(sub.db(), sel, fid, std::string("count")))
        if (!ours.count(r) && std::find(names.begin(), names.end(), r) == names.end()) others += (others.empty() ? "" : ", ") + r;
      if (!others.empty()) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o0 + ": the measurements " + others + " of this Selection and recording were made over another span (their counts differ); recompute them over this span (with --replace) or remove them first");
      unit.remove_measurements(sel, fid, "count");
      count_replaced = true;
    }
  }
  for (const Part& p : parts) {
    if (measured_by(sub.db(), sel, P.values.at(p.o), fid).empty()) continue;
    if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + p.o + ": its measurements already exist; pass --replace");
    for (const auto& n : p.names) unit.remove_measurements(sel, fid, n);
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
  unit.write_measurements(rows);
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
Json store_crossings(nxr::io::SubjectWriter& sub, nxr::pipelines::task::Unit& unit, const Plan& P, const std::string& o, const Value& v, const Array& a, const std::string& by, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string sel = over_selection(sub, P, v), fid = text(rec, "id");
  const std::vector<std::string> names = measure_rows("tile_crossings");
  if (a.comps != 4 || static_cast<std::int64_t>(a.data.size()) != a.rows * a.cols * 4) refuse("outputs." + o + ": the crossings' array is not [rows x members x 4]");
  if (!measured_by(sub.db(), sel, v, fid).empty()) {
    if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + ": its measurements already exist; pass --replace");
    for (const auto& n : names) unit.remove_measurements(sel, fid, n);
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
  unit.write_measurements(rows);
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
  if (!a.meta.is_object() || !a.meta.contains("tiles_of") || !a.meta["tiles_of"].is_object() || !a.meta["tiles_of"].contains("width_s") || !a.meta["tiles_of"].contains("origin"))
    throw nxr::io::Error(nxr::io::code::invalid_argument, "outputs." + o + ": the events record no tile width or origin of the tiles they were found on (meta " + (a.meta.is_object() ? a.meta.value("tiles_of", Json()).dump().substr(0, 300) : std::string("null")) + ")");
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
  // THE BURSTS ARE AN IRREGULAR FAMILY on the product manifold of space tiles x time tiles (they overlap in nothing here, vary in extent, have no levels): a member per cell, the
  // event's ordinal its number (-1: a cell in none), the cells numbered row-major as the field's array is (time fastest); not a product of two grids
  nxr::io::selection::Description events_set;
  events_set.type = nxr::io::selection::Type::set;
  events_set.member_of.assign(codes.begin(), codes.end());
  events_set.member_codes.resize(static_cast<std::size_t>(events));
  for (std::int64_t k = 0; k < events; ++k) events_set.member_codes[static_cast<std::size_t>(k)] = k;
  nxr::io::selection::Placement at;
  at.name = name; at.path = path; at.manifold_id = v.manifolds[0]; at.dictionary_id = dr.id; at.cell = "0,1";
  at.label = std::to_string(events) + " events";
  at.description = "the space x time extent of each global burst: member = the event's ordinal, -1 for a cell in none (pipeline " + key.pipeline + "); the cells are [spatial tile x time tile], row-major, " + std::to_string(a.rows) + " x " + std::to_string(a.cols);
  at.params = params;
  const std::string id = nxr::io::selection::save(sub, events_set, at).id;
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
  if (a.meta.contains("members_used") && a.meta["members_used"].is_array()) for (const auto& u : a.meta["members_used"]) least = std::min<std::int64_t>(least, u.get<std::int64_t>());
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
  pj["output"] = o; if (a.meta.contains("members_used")) pj["members_used"] = a.meta["members_used"];
  st.producer = pj;
  const std::string id = sub.statistic(std::move(st));
  return Json{{"output", o}, {"statistic", name}, {"path", path}, {"id", id}};
}
// A HISTOGRAM STATISTIC OUTPUT (`core_histogram`; nxr::stats::histogram): counts in fixed log bins over the burst frames, pooled over the vertices, beside the recording: measure count, method histogram, the bins' range
// (lo, hi, bins) and what the distribution was made from (the chain) in its options, so a reader derives any percentile from the counts (the cores' threshold: the 98th) and the cores' step checks the chain
Json store_histogram(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const Array& a, const Json& producer,
                     const nxr::pipelines::CacheKey& key, bool replace, bool& replaced) {
  const Definition& d = P.definition;
  const nxr::io::Row& rec = *P.found.at(*d.over).row;
  const std::string path = statistic_out_path(P, o), name = node_name(d, o, P.found);
  nxr::pipelines::clear_stored(ds, sub, {{"statistic", path}}, replace, d.name + " output " + o + " (" + path + ")", replaced);
  const Json& h = a.meta["histogram"];
  const std::int64_t bins = h["bins"].get<std::int64_t>();
  nxr::stats::histogram::LogBins lb;
  lb.lo = h["lo"].get<double>(); lb.hi = h["hi"].get<double>(); lb.bins = static_cast<int>(bins);
  Json edges = Json::array();
  for (int k = 0; k < lb.bins; ++k) edges.push_back(nxr::stats::histogram::upperEdge(lb, k));
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
  opt["percentile"] = "read: the upper edge of the first bin whose cumulative share reaches p (nxr::stats::histogram::binnedPercentile)";
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
  if (g.contains("partition_selection_id") && g["partition_selection_id"].is_string()) st.partition_selection_id = g["partition_selection_id"].get<std::string>();   // the selection its members are (a surrogate draw's tiles: the comparison checks it is the data's)
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
  const Json levels = v.facts.is_object() && v.facts.contains("levels") && v.facts["levels"].is_array() ? v.facts["levels"] : Json::array();
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
  opt["outside"] = v.facts.is_object() ? v.facts.value("outside", Json()) : Json();
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

// the dense matrix of a stored operator [rows x cols] and, in `meta.channels`, the elements of its `from` manifold its columns lie over (its from_selection), for a node that applies it
nxr::graph::Array operator_constant(const Bound& bound, nxr::io::SubjectWriter& sub, const nxr::io::Row& op, const std::string& input) {
  const std::string name = text(op, "name");
  if (nxr::io::opt_text(op, "layout").value_or("dense") != "dense") refuse("steps: " + input + " (" + name + "): a node applies a dense operator (layout " + nxr::io::opt_text(op, "layout").value_or("") + ")");
  std::vector<std::int64_t> shape;
  const std::vector<float> m = bound.array_float(text(op, "path"), &shape);
  if (shape.size() != 2) refuse("steps: " + input + " (" + name + "): its array is not [rows x cols]");
  nxr::graph::Array a;
  a.rows = shape[0]; a.cols = shape[1];
  a.data.assign(m.begin(), m.end());
  Json channels = Json::array();
  const auto sel = nxr::io::is_null(op, "from_selection_id") ? std::nullopt : sub.db().read("selection", text(op, "from_selection_id"));
  if (sel && !nxr::io::is_null(*sel, "array_path")) for (const std::int32_t c : bound.array_int32(text(*sel, "array_path"))) channels.push_back(c);
  else for (std::int64_t c = 0; c < a.cols; ++c) channels.push_back(c);
  a.meta["channels"] = channels;
  a.meta["name"] = name;
  a.meta["id"] = text(op, "id");
  a.meta["to_value_type"] = nxr::io::opt_text(op, "to_value_type").value_or("vector3");
  a.meta["to_manifold_id"] = nxr::io::opt_text(op, "to_manifold_id").value_or("");
  a.meta["from_manifold_id"] = text(op, "from_manifold_id");
  a.meta["from_value_type"] = nxr::io::opt_text(op, "from_value_type").value_or("scalar");
  return a;
}

Json execute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const RunOptions& opt, const std::function<void(const Json&)>& progress, const GroupInputs* group) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  const Definition& d = P.definition;
  const auto anchor = d.anchor();
  // THE SPAN (the chunk service's request, always stated by the caller) and the bad spans the accumulate steps exclude
  std::optional<nxr::io::Span> span;
  std::optional<nxr::io::ChunkSource> source;
  Json span_json = nullptr;
  const auto loop = P.section("do");
  const bool static_field = d.over && !P.found.at(*d.over).time;           // a field with no time axis (a map): nothing streams; it is the loop's one block
  if (d.over && !static_field) {
    const Value& over = P.found.at(*d.over);
    const auto line = sub.db().read("manifold", *over.time);
    int level = 0;                                                          // the grid a node asks of the span (Planned::align): a segment of 2^k samples starts on the line's grid
    if (P.align > 1) {
      while ((std::int64_t(1) << (level + 1)) <= P.align) ++level;
      if ((std::int64_t(1) << level) != P.align) refuse("steps: a node asks the span to snap to " + std::to_string(P.align) + " samples, which is not a power of two");
    }
    span = nxr::pipelines::span_of(*over.row, line ? nxr::io::integer(*line, "n_vertices") : 0, over.rate, over.origin, opt.chunks, opt.start, opt.stop, level);
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
    // `read: shape` (a surrogate's draw): the chunk service hands the recording's shape and good samples, never a sample (spec 2026-11-17-surrogates.md, section 6)
    if (d.inputs.at(*d.over).value("read", std::string("samples")) == "shape") source.emplace(nxr::pipelines::shape_source_of(sub, *over.row, over.rate, over.origin, bad));
    else source.emplace(nxr::pipelines::source_of(sub, *over.row, over.rate, over.origin, {}, bad, opt.projectors));
  }
  // THE CACHE KEY: the definition, its parameters, the inputs, the span (a rerun with all four the same is a hit)
  nxr::pipelines::CacheKey key;
  key.pipeline = d.name + "@" + std::to_string(d.version);
  key.params = Json{{"parameters", P.parameters}, {"definition_sha256", nxr::io::sha256_hex(d.text)}};
  // A PRESET IS ONLY A NAMED PARAMETER SET (consolidation spec 4.2): its name is in the provenance (below), never in the key, so two presets that resolve to equal parameters are one cached result
  if (!d.chains.empty()) key.params["chains"] = d.chains;                    // the field chains composed in: their names, resolved parameters and sha256 (a changed chain is a new result)
  if (!d.made.empty()) key.params["made"] = d.made;                          // the makers composed in
  key.params["block_samples"] = opt.block_samples;   // the pointwise tail's sub-block width: the same values up to the rounding of the folds' sums, so in the key (method rule 3)
  for (const Planned* p : loop) if (p->step->frame_place) key.params["block_frames"] = opt.block_frames;   // likewise the frames' (only a run with a frame tiling has one)
  {
    // what each step's request RESOLVES to on what is stored (reading only): the choice made, not only the request, is in the key (method rule 3)
    Json resolved = Json::object();
    for (const Planned& p : P.steps) {
      if (!p.step->resolve) continue;
      Values ins;
      for (const auto& [port, t] : p.ins) if (P.values.count(t)) ins[port] = P.values.at(t);
      const Json r = p.step->resolve(ins, p.params, Bound(sub, &ds));
      if (!r.is_null()) resolved[p.step->name + ":" + p.use.outs.front()] = r;
    }
    if (!resolved.empty()) key.params["resolved"] = resolved;
  }
  Json ids = Json::object();
  for (const auto& [k, v] : P.found) {
    if (group && group->ids.count(k)) {                                      // a per-subject input: each member's id, member by member (a group result is of these members)
      Json per = Json::object();
      for (std::size_t i = 0; i < group->subjects.size(); ++i) per[group->subjects[i]] = group->ids.at(k)[i];
      ids[k] = per;
    } else {
      ids[k] = text(*v.row, "id");
    }
  }
  key.input = Json{{"inputs", ids}, {"span", span_json}};
  const Json projection_json = source ? source->projection_record() : Json();   // the projectors the load applied (null: the recording has no active projector)
  if (!projection_json.is_null()) key.input["projectors"] = projection_json;
  key.compute = NXR_COMPUTE_VERSION;
  Json stored = Json::array();
  bool hit = true;
  const std::string measured_id = d.over ? text(*P.found.at(*d.over).row, "id") : std::string();
  for (const auto& [o, ospec] : d.outputs.items()) {
    const Value& v = P.values.at(o);
    if (ospec.value("get_or_create", false)) { stored.push_back(Json{{"output", o}, {"get_or_create", true}}); continue; }   // a maker's selection: found by its maker key or made, never a miss or a refusal
    if (ospec.value("kind", std::string()) == "recording") {                // a recording output: the field of that name holding the same key
      const auto row = nxr::pipelines::find_cached(sub.db(), sub.id(), "field", key, std::nullopt);
      if (!row || text(*row, "name") != declared_name(P, o)) { hit = false; break; }
      stored.push_back(Json{{"output", o}, {"kind", "recording"}, {"name", text(*row, "name")}, {"id", text(*row, "id")}});
      continue;
    }
    // the field a declared measurement is of: a found one, or one of this run's own outputs stored just before (its id in `stored`)
    std::string of_id;
    if (is_declared(P, o) && ospec["kind"] == "measurement" && ospec.contains("of")) {
      const std::string of = ospec["of"].get<std::string>();
      if (P.found.count(of) && P.found.at(of).row) of_id = text(*P.found.at(of).row, "id");
      else for (const auto& e : stored) if (e.value("output", std::string()) == of && e.contains("id")) of_id = e["id"].get<std::string>();
    }
    if (is_declared(P, o) && ospec["kind"] == "measurement" && ospec.contains("of") && !of_id.empty() && ospec["over"].is_string()) {
      // a declared measurement of a FOUND field (or of the run's own stored field) over a found selection, or over a MAKER's selection of this run (get-or-create: stored by its maker key, so a rerun finds it
      // at its place) takes no node: it is a hit when every row of the pair was made by this very run (same key)
      const std::string over = ospec["over"].get<std::string>();
      std::string sel;
      if (P.found.count(over) && P.found.at(over).row) sel = text(*P.found.at(over).row, "id");
      else if (d.outputs.contains(over) && d.outputs[over].value("get_or_create", false)) {
        // the stored selection of that name made by this very maker (its maker key is in its parameters)
        task::Want w;
        w.cls = task::Cls::selection;
        w.name = declared_name(P, over);
        w.params.push_back(task::ParamIs{"maker_key", Json(nxr::io::sha256_hex(maker_key(P, over).dump()))});
        for (const auto& n : task::Subject::of(sub).find_all(w)) sel = n.id;
        if (sel.empty()) { hit = false; break; }                              // its maker's selection is not stored yet: a first run
      }
      if (!sel.empty()) {
        auto by = nxr::io::selection::computed_by_over_field(sub.db(), sel, of_id);
        if (by.empty())                                                   // a node may have measured over a member of the found selection (a level's tile set of a grid: describe.over_member)
          for (const nxr::io::Row& mr : nxr::io::selection::member_rows(sub.db(), sel, false)) { by = nxr::io::selection::computed_by_over_field(sub.db(), text(mr, "id"), of_id); if (!by.empty()) break; }
        if (by.empty() && ospec.value("optional", false)) { stored.push_back(Json{{"output", o}, {"kind", "measurement"}, {"over", sel}, {"stored", false}, {"why", "an optional output this run did not make"}}); continue; }
        if (by.size() != 1 || by[0] != made_by(key)) { hit = false; break; }
        stored.push_back(Json{{"output", o}, {"kind", "measurement"}, {"over", sel}});
        continue;
      }
    }
    if (is_declared(P, o)) {                                                // a declared output: each place it takes holds the same key (a manifold is found or made, never a miss)
      Json one = Json{{"output", o}, {"kind", ospec["kind"]}, {"name", declared_name(P, o)}};
      const auto places = declared_places(P, o, sub);
      // an output with no place to look up (a selection on a manifold this run makes, a measurement over it) cannot be shown stored: a miss, never a hit
      // (a measurement over another output of this run, the run's own events: it is stored with them and a hit with them)
      if (places.empty() && ospec["kind"] == "measurement" && ospec.contains("over") && ospec["over"].is_string() && d.outputs.contains(ospec["over"].get<std::string>()) && !P.found.count(ospec["over"].get<std::string>()) && (!is_declared(P, ospec["over"].get<std::string>()) || (ospec_kind(d.outputs[ospec["over"].get<std::string>()]) == "selection" && !d.outputs[ospec["over"].get<std::string>()].value("get_or_create", false)))) {   // (a declared selection the run stores itself is checked by its own key, listed before; a maker's is found or made, never a miss)
        stored.push_back(one);
        continue;
      }
      if (places.empty() && ospec["kind"] != "manifold" && !ospec.value("optional", false)) { hit = false; break; }
      for (const auto& [cls, path] : places) {
        const auto row = nxr::pipelines::find_cached(sub.db(), sub.id(), task::to_string(cls), key, path);
        if (!row && ospec.value("optional", false) && !sub.find(task::to_string(cls), path)) continue;   // an optional output this very run did not make (nothing stored there): not a miss
        if (!row) { hit = false; break; }
        if (cls != task::Cls::selection) { one["id"] = text(*row, "id"); one["path"] = path; }
      }
      if (!hit) break;
      stored.push_back(one);
      continue;
    }
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
    for (const auto& [o, ospec_o] : d.outputs.items()) {
      const Value& v = P.values.at(o);
      if (ospec_o.value("get_or_create", false)) continue;
      if (is_declared(P, o)) {
        for (const auto& [cls, path] : declared_places(P, o, sub))
          if (sub.find(task::to_string(cls), path))
            throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + d.name + " output " + o + " (" + path + ") already exists, made with other parameters, inputs or span; pass --replace");
        continue;
      }
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

  // THE EXECUTOR (nxr_graph, pipelines/graph/) does every step: `before` once, each chunk through the chunk and accumulate steps with their state carried, `after` once. The runner finds
  // (above), streams (io's chunk service, with the halo the checked graph declares) and stores (below); it holds no step logic.
  const Bound bound(sub, &ds, progress);
  const auto binding = std::make_shared<Binding>();
  binding->bound = &bound;
  binding->say = progress;
  nxr::graph::Types types = P.types;
  for (const auto& [k, v] : P.values)                                      // a streamed recording's channels (types, quality, units) are facts its nodes may read
    if (v.cls == "field" && v.time && P.found.count(k) && P.found.at(k).row && types.count(k))
      types[k].facts["channels"] = nxr::pipelines::channel_facts(sub, text(*P.found.at(k).row, "name"));
  nxr::graph::Checked checked;
  try {
    checked = nxr::graph::check(P.graph, types, Json::object(), node_registry(binding));
  } catch (const std::invalid_argument& e) {
    refuse(std::string("steps: ") + e.what());
  }
  nxr::graph::Arrays constants;                                            // a found input that is not streamed is a description the steps read through the binding: no array
  for (const auto& name : checked.inputs)
    if (!checked.values.at(name).streamed()) { nxr::graph::Array a; a.rows = 0; a.cols = 0; a.meta["placeholder"] = true; constants[name] = std::move(a); }
  if (group)                                                               // the per-subject inputs of a group run, stacked over the members: their values, not a description
    for (const auto& [name, stack] : group->stacked) constants[name] = stack;
  // what a native node (a grouped name: a pure function of arrays) reads is handed to it as arrays (the node reads no store): an OPERATOR as its matrix and the channels it
  // lies over; a stored FIELD that is not streamed as its values (rows x components); a MANIFOLD (a surface) as its vertices [n x 3], its faces in meta["faces"]
  for (const auto& n : checked.nodes) {
    if (const StepSpec& st = step(n.spec->name); !st.native && !st.reads_arrays) continue;   // a native node (a grouped name of nxr::graph::registry()), or a step that declares it reads its found inputs as arrays; a step with a grouped name of its own (surrogate.*) reads them through the binding
    for (const auto& [port, name] : n.in) {
      if (!constants.count(name) || !P.values.count(name) || !P.found.count(name) || !P.found.at(name).row) continue;
      if (group && group->stacked.count(name)) continue;                    // already the members' stacked values
      const Value& v = P.values.at(name);
      const nxr::io::Row& row = *P.found.at(name).row;
      if (v.cls == "selection" && nxr::io::is_null(row, "array_path")) {     // a closed-form selection (a tiling, a ladder's level): its record only
        nxr::graph::Array a;
        a.meta["name"] = text(row, "name"); a.meta["id"] = text(row, "id"); a.meta["type"] = text(row, "type");
        if (!nxr::io::is_null(row, "n_members")) a.meta["n_members"] = nxr::io::integer(row, "n_members");
        a.meta["member_of_id"] = nxr::io::opt_text(row, "member_of_id").value_or("");
        a.meta["params"] = Json::parse(nxr::io::opt_text(row, "params_json").value_or("{}"), nullptr, false);
        if (text(row, "type") == "ladder" || text(row, "type") == "grid") {   // a grid (dyadic tiles): the FINEST level's tile of every vertex, and the level dictionary (each level's measured diameter and band), as the nodes that read leaves expect
          const auto levels = bound.level_members(text(row, "id"));
          if (!levels.empty()) {
            Json lv = Json::array();
            for (const auto& m : levels) {
              const Json lj = Json::parse(nxr::io::opt_text(m, "params_json").value_or("{}"), nullptr, false);
              Json e = Json::object();
              if (lj.is_object() && lj.contains("diameter")) e["diameter"] = lj["diameter"];
              if (lj.is_object() && lj.contains("lambda_min")) { e["lambda_min"] = lj["lambda_min"]; e["lambda_max"] = lj["lambda_max"]; }
              lv.push_back(e);
            }
            const nxr::io::selection::ElementMembers em = nxr::io::selection::read_element_members(sub.db(), sub.store(), text(levels.back(), "id"));
            for (const std::int32_t c : em.member_of) a.data.push_back(c);
            a.rows = static_cast<std::int64_t>(a.data.size()); a.cols = 1;
            a.meta["levels"] = lv;
            a.meta["depth"] = static_cast<int>(levels.size()) - 1;
            const Json g = a.meta["params"].is_object() && a.meta["params"].contains("grid") ? a.meta["params"]["grid"] : Json::object();
            a.meta["slots"] = g.is_object() ? g.value("slots", std::int64_t(1)) : std::int64_t(1);
          }
        }
        constants[name] = std::move(a);
      } else if (v.cls == "selection" && !nxr::io::is_null(row, "array_path")) {   // a partition or a pick: its codes, its member count and parameters
        nxr::graph::Array a;
        if (text(row, "type") == "set") {                                // a set per element: its member NUMBERS, and each member's label beside them (never the raw array as labels)
          const nxr::io::selection::ElementMembers em = nxr::io::selection::read_element_members(sub.db(), sub.store(), text(row, "id"));
          for (const std::int32_t c : em.member_of) a.data.push_back(c);
          a.meta["member_codes"] = em.member_codes;
        } else {                                                           // a pick (indices): the element list as stored
          for (const std::int32_t c : bound.array_int32(text(row, "array_path"))) a.data.push_back(c);
        }
        a.rows = static_cast<std::int64_t>(a.data.size()); a.cols = 1;
        a.meta["name"] = text(row, "name"); a.meta["id"] = text(row, "id"); a.meta["type"] = text(row, "type");
        if (!nxr::io::is_null(row, "n_members")) a.meta["n_members"] = nxr::io::integer(row, "n_members");
        a.meta["member_of_id"] = nxr::io::opt_text(row, "member_of_id").value_or("");
        a.meta["params"] = Json::parse(nxr::io::opt_text(row, "params_json").value_or("{}"), nullptr, false);
        constants[name] = std::move(a);
      } else if (v.cls == "statistic") {                                 // its values, its axes; a first axis over a pick of channels: those channels and their types
        std::vector<std::int64_t> shape;
        const std::vector<double> f = bound.array_double(text(row, "path"), &shape);
        nxr::graph::Array a;
        a.rows = shape.empty() ? 0 : shape[0];
        a.cols = a.rows > 0 ? static_cast<std::int64_t>(f.size()) / a.rows : 0;
        a.data.assign(f.begin(), f.end());
        a.meta["name"] = text(row, "name"); a.meta["id"] = text(row, "id"); a.meta["shape"] = shape;
        a.meta["value_type"] = nxr::io::opt_text(row, "value_type").value_or("scalar");
        if (const auto u = nxr::io::opt_text(row, "unit"); u && !u->empty()) a.meta["unit"] = *u;
        if (!nxr::io::is_null(row, "path")) a.meta["stored_precision"] = nxr::io::ArrayReader::open(sub.store(), text(row, "path")).dtype() == nxr::io::DType::Float64 ? "float64" : "float32";
        const Json axes = Json::parse(nxr::io::opt_text(row, "axes_json").value_or("[]"), nullptr, false);
        a.meta["axes"] = axes;
        Json chax;                                                         // the first channel axis over a pick (after a band axis, when there is one)
        if (axes.is_array()) for (const auto& ax : axes) if (ax.contains("selection_id") && ax.value("name", std::string()) == "channel") { chax = ax; break; }
        a.meta["producer"] = Json::parse(nxr::io::opt_text(row, "producer_json").value_or("{}"), nullptr, false);
        a.meta["measure"] = nxr::io::opt_text(row, "measure").value_or("");
        a.meta["of_field_id"] = nxr::io::opt_text(row, "of_field_id").value_or("");
        if (!chax.is_null())
          if (const auto sel = sub.db().read("selection", chax["selection_id"].get<std::string>()); sel && !nxr::io::is_null(*sel, "array_path")) {
            const std::vector<std::int32_t> ch = bound.array_int32(text(*sel, "array_path"));
            a.meta["channels"] = ch;
            a.meta["channels_selection_id"] = text(*sel, "id");
            const std::string mid = nxr::io::opt_text(*sel, "manifold_id").value_or("");
            a.meta["channels_manifold_id"] = mid;
            const std::vector<std::string> all = nxr::pipelines::channel_types_of(sub, mid);
            Json types = Json::array();
            for (const auto c : ch) types.push_back(c >= 0 && static_cast<std::size_t>(c) < all.size() ? all[static_cast<std::size_t>(c)] : std::string());
            a.meta["channel_types"] = types;
            // the channels' projector Operators (SSP, ICA), for a node that applies them as its reference does (the inverse kernel)
            Json projs = Json::array();
            for (const auto& q : nxr::bst::stored_projectors(sub.db(), sub.store(), mid)) {
              Json j{{"name", q.name}, {"kind", q.kind}, {"method", q.method}, {"order", q.order}, {"status", q.status}, {"m", q.m}};
              if (!q.vectors.empty()) j["vectors"] = q.vectors;
              if (!q.matrix.empty()) j["matrix"] = q.matrix;
              projs.push_back(std::move(j));
            }
            if (!projs.empty()) a.meta["projectors"] = projs;
          }
        constants[name] = std::move(a);
      } else if (v.cls == "measurement") {                              // a stored measurement found by its key: its values [rows x members], the key in meta
        const std::optional<std::string> of_field = v.facts.contains("of_field_id") ? std::optional<std::string>(v.facts["of_field_id"].get<std::string>()) : std::nullopt;
        const std::optional<std::string> of_stat = v.facts.contains("of_statistic_id") ? std::optional<std::string>(v.facts["of_statistic_id"].get<std::string>()) : std::nullopt;
        const std::optional<std::int64_t> of_index = v.facts.contains("of_index") ? std::optional<std::int64_t>(v.facts["of_index"].get<std::int64_t>()) : std::nullopt;
        const nxr::io::MeasurementValues mv = nxr::io::measurement_values(sub, text(row, "id"), of_field, of_stat, of_index, v.facts.value("measure", std::string()));
        nxr::graph::Array a;
        a.rows = mv.rows; a.cols = mv.members;
        a.data = mv.values;
        a.meta["name"] = text(row, "name"); a.meta["id"] = text(row, "id"); a.meta["measure"] = v.facts.value("measure", std::string());
        if (mv.unit && !mv.unit->empty()) a.meta["unit"] = *mv.unit;
        constants[name] = std::move(a);
      } else if (v.cls == "field") {
        std::vector<std::int64_t> shape;
        const std::vector<double> f = bound.array_double(text(row, "path"), &shape);
        nxr::graph::Array a;
        a.rows = shape.empty() ? 0 : shape[0];
        a.cols = a.rows > 0 ? static_cast<std::int64_t>(f.size()) / a.rows : 0;
        a.data.assign(f.begin(), f.end());
        a.meta["name"] = text(row, "name"); a.meta["id"] = text(row, "id"); a.meta["shape"] = shape;
        constants[name] = std::move(a);
      } else if (v.cls == "manifold" && text(row, "type") == "curve") {     // a fiber bundle: its points [nF*nP x 3], fiber-major, held as double to carry them, `precision` in meta (its stored one: no node widens)
        const nxr::io::task::Subject subject = nxr::io::task::Subject::of(sub);
        const std::string bundle = text(row, "name"), stored = nxr::io::fibers::bundle_precision(subject, sub.store(), bundle);
        nxr::graph::Array a;
        std::int64_t nP = 0;
        if (stored == "float32") { const auto b = nxr::io::fibers::read_bundle<float>(subject, sub.store(), bundle); a.data.assign(b.points.begin(), b.points.end()); nP = b.points_per_fiber; a.meta["origin"] = b.origin; }
        else { const auto b = nxr::io::fibers::read_bundle<double>(subject, sub.store(), bundle); a.data.assign(b.points.begin(), b.points.end()); nP = b.points_per_fiber; a.meta["origin"] = b.origin; }
        a.rows = static_cast<std::int64_t>(a.data.size() / 3); a.cols = 3;
        a.meta["name"] = bundle; a.meta["id"] = text(row, "id"); a.meta["points_per_fiber"] = nP; a.meta["stored_precision"] = stored;
        a.meta["precision"] = stored;                                      // the nodes compute in the points' own precision and refuse a mix (no widening: the values are held as double only to carry them)
        constants[name] = std::move(a);
      } else if (v.cls == "manifold" && text(row, "type") == "points" && nxr::pipelines::channel_types_of(sub, text(row, "id")).empty()) {   // points with no channel types (a template's landmarks): positions and names
        const nxr::pipelines::NamedPoints g = nxr::pipelines::named_points(sub, text(row, "id"));
        nxr::graph::Array a;
        a.rows = static_cast<std::int64_t>(g.positions.size() / 3); a.cols = 3;
        a.data = g.positions;
        a.meta["name"] = g.name; a.meta["id"] = g.id; a.meta["names"] = g.names;
        if (!nxr::io::is_null(row, "positions_path"))
          a.meta["precision"] = nxr::io::ArrayReader::open(sub.store(), text(row, "positions_path")).dtype() == nxr::io::DType::Float64 ? "float64" : "float32";
        constants[name] = std::move(a);
      } else if (v.cls == "manifold" && text(row, "type") == "points") {    // a channels manifold: its sensor geometry (positions; types, orientations, coils in meta)
        const nxr::pipelines::SensorGeometry g = nxr::pipelines::sensor_geometry(sub, text(row, "id"));
        nxr::graph::Array a;
        a.rows = static_cast<std::int64_t>(g.positions.size() / 3); a.cols = 3;
        a.data = g.positions;
        a.meta["name"] = g.name; a.meta["id"] = g.id; a.meta["cs_id"] = g.cs_id; a.meta["types"] = g.types;
        if (g.orientation) a.meta["orientation"] = *g.orientation;
        if (g.coils) a.meta["coils"] = Json{{"positions", g.coil_positions}, {"orientation", g.coil_orientation}, {"indptr", g.indptr}, {"indices", g.indices}, {"weights", g.weights}};
        constants[name] = std::move(a);
      } else if (v.cls == "manifold") {                                     // a surface: its vertices [n x 3] as stored, its faces in meta
        const nxr::pipelines::SurfaceInput S = nxr::pipelines::read_surface(sub, text(row, "name"), true);
        nxr::graph::Array a;
        a.rows = S.n_vertices(); a.cols = 3;
        a.data = S.vertices;
        a.meta["faces"] = S.faces;
        a.meta["name"] = S.name; a.meta["id"] = S.id; a.meta["cs_id"] = nxr::pipelines::coordinate_system_of(sub, S.id);
        a.meta["unit"] = nxr::pipelines::coordinate_unit_of(sub, S.id);
        if (!nxr::io::is_null(S.manifold, "positions_path"))               // the positions' own precision: a node that meets fibers of another precision refuses the pair, naming both
          a.meta["precision"] = nxr::io::ArrayReader::open(sub.store(), text(S.manifold, "positions_path")).dtype() == nxr::io::DType::Float64 ? "float64" : "float32";
        constants[name] = std::move(a);
      }
    }
    for (const auto& [port, name] : n.in)                                   // A HODGE-DIRAC PACKAGE (its eigenvectors' operator): its basis, mass and meshes, read by pipelines
      if (constants.count(name) && P.values.count(name) && P.values.at(name).cls == "operator" && P.found.count(name) && P.found.at(name).row &&
          nxr::io::opt_text(*P.found.at(name).row, "kind") == std::optional<std::string>("eigenvectors")) {
        const nxr::pipelines::HodgePackageData h = nxr::pipelines::hodge_package_data(sub, text(*P.found.at(name).row, "name"), std::nullopt);
        nxr::graph::Array a;
        a.rows = h.basis.P.rows(); a.cols = h.basis.P.cols();
        a.data.resize(static_cast<std::size_t>(a.rows * a.cols));
        for (Eigen::Index k = 0; k < h.basis.P.rows(); ++k) for (Eigen::Index v = 0; v < h.basis.P.cols(); ++v) a.data[static_cast<std::size_t>(k * a.cols + v)] = h.basis.P(k, v);
        a.meta["hodge"] = true;
        a.meta["mu"] = std::vector<double>(h.basis.mu.data(), h.basis.mu.data() + h.basis.mu.size());
        a.meta["kind"] = h.basis.kind;
        a.meta["mass"] = std::vector<double>(h.basis.mass.data(), h.basis.mass.data() + h.basis.mass.size());
        a.meta["positions"] = std::vector<double>(h.basis.positions.data(), h.basis.positions.data() + h.basis.positions.size());
        a.meta["faces"] = std::vector<std::int32_t>(h.basis.faces.data(), h.basis.faces.data() + h.basis.faces.size());
        a.meta["keep"] = h.basis.keep;
        a.meta["input_positions"] = h.input_positions;
        a.meta["input_surface_id"] = h.input_surface_id;
        a.meta["name"] = h.stem; a.meta["id"] = h.id; a.meta["axis_id"] = h.axis_id;
        if (n.spec->reads_package_forms) {                                  // the full forms: the 2-form parts (as stored, float32 bytes) and the options of the package's complex
          const nxr::pipelines::HodgePackageForms f = nxr::pipelines::hodge_package_forms(sub, h);
          const auto* bytes = reinterpret_cast<const std::uint8_t*>(f.forms2.data());
          a.meta["forms2"] = Json::binary(std::vector<std::uint8_t>(bytes, bytes + f.forms2.size() * sizeof(float)));
          a.meta["forms2_shape"] = Json::array({f.nF, f.K});
          a.meta["complex"] = Json{{"mollify", f.mollify}, {"flip_eps", f.flip_eps}, {"star1_floor", f.star1_floor}};
        }
        constants[name] = std::move(a);
      }
    for (const auto& [port, name] : n.in)
      if (constants.count(name) && !constants.at(name).meta.value("hodge", false) && P.values.count(name) && P.values.at(name).cls == "operator" && P.found.count(name) && P.found.at(name).row) constants[name] = operator_constant(bound, sub, *P.found.at(name).row, name);
  }
  // EIGENMODES FIRST (the spec, question 6): when a step needs a surface's eigenmodes and none are stored, the runner runs the `eigenmodes` pipeline as its OWN run (its own provenance and
  // cache key), says what it made and why, and starts again: nothing is ever stored hidden inside this run. The Chebyshev series stays the announced fallback (`transform: chebyshev`).
  std::optional<nxr::graph::Execution> ex;
  Json made_first = Json::array();
  std::set<std::string> made_packages;
  for (;;) {
    try {
      ex.emplace(checked, constants, nxr::graph::DispatchContext{}, nxr::graph::Blocking{opt.block_samples, opt.block_frames});
      break;
    } catch (const MissingEigenmodes& m) {
      if (!made_packages.insert(m.plan.package).second) throw;             // made, and still missing: a fault, not a reason to start again
      const std::string why = "the pipeline " + key.pipeline + " needs the eigenmodes of " + m.surface + " and none are stored";
      say(Json{{"stage", "make"}, {"made", "eigenmodes"}, {"surface", m.surface}, {"package", m.plan.package}, {"modes", m.plan.compute_modes}, {"why", why}});
      Json r = nxr::pipelines::make_eigenmodes(ds, sub, m.surface, m.plan, progress);
      r["surface"] = m.surface;
      r["why"] = why;
      made_first.push_back(std::move(r));
    }
  }
  Json facts = Json::object();
  Json recording_report;                                                   // a recording output, written as the loop ran
  if (!checked.loop.empty()) {
    const std::int64_t H = ex->halo();
    say(Json{{"stage", "run"}, {"pipeline", key.pipeline}, {"span", span ? Json(span->what) : Json("the map")}, {"halo", H}, {"steps", static_cast<std::int64_t>(checked.loop.size())}});
    std::int64_t blocks_n = 0;
    // A RECORDING OUTPUT (a filter's): the streamed value written as a new recording as the loop runs, in io's blocks; the source's flags, events, bad segments and projectors
    // carried over by io (add_derived_recording). Nothing else of the run is held in memory.
    std::optional<std::string> rec_out;
    for (const auto& [o, ospec] : d.outputs.items()) if (ospec.value("kind", std::string()) == "recording") rec_out = o;
    nxr::io::ChunkRun cr;
    auto loop_with = [&](const std::function<void(const Blocks&)>& sink) {
      if (static_field) {                                                  // the map's values as ONE one-sample block: its elements are the rows (a vector's components consecutive)
        std::vector<std::int64_t> shape;
        const std::vector<double> f = bound.array_double(text(*P.found.at(*d.over).row, "path"), &shape);
        Block x{static_cast<std::int64_t>(f.size()), 1, 0, 0, std::vector<float>(f.size()), {}};
        for (std::size_t i = 0; i < f.size(); ++i) x.re[i] = static_cast<float>(f[i]);
        const std::vector<std::uint8_t> none;
        ++blocks_n;
        const Blocks outs = ex->block(Blocks{{*d.over, std::move(x)}}, Place{0, 1, &none});
        if (sink) sink(outs);
        return;
      }
      cr = nxr::io::haloed_loop(*source, *span, H, [&](const nxr::io::HaloedChunk& h) {
        ++blocks_n;
        const std::int64_t w = h.left + h.n + h.right;
        Block x{source->reads_samples() ? source->rows() : 0, w, h.left, h.right, std::vector<float>(h.data.size()), {}};   // a shape-only source hands no rows (no samples)
        for (std::size_t i = 0; i < h.data.size(); ++i) x.re[i] = static_cast<float>(h.data[i]);   // the stored float32, widened by the service, narrowed back exactly
        Blocks blocks{{*d.over, std::move(x)}};
        std::vector<std::uint8_t> good;
        if (!h.good.empty()) good.assign(h.good.begin() + h.left, h.good.begin() + h.left + h.n);
        std::vector<std::uint8_t> good_after;                                // the right halo's flags, for an accumulate node that reads past its block (a segment family)
        if (!h.good.empty()) good_after.assign(h.good.begin() + h.left + h.n, h.good.end());
        Place place{h.first, h.n, &good};
        place.origin = span->s0; place.end = span->s1; place.good_after = &good_after;   // the segment grid starts at the span's first sample; no segment reaches past its end
        const Blocks outs = ex->block(blocks, place);
        if (sink) sink(outs);
      });
    };
    if (!rec_out) loop_with({});
    else {
      const std::string rname = declared_name(P, *rec_out);
      const nxr::io::Row& src_row = *P.found.at(*d.over).row;
      Json rp = nxr::pipelines::stamp(key);
      rp["pipeline"] = Json{{"name", d.name}, {"version", d.version}, {"sha256", nxr::io::sha256_hex(d.text)}};
      rp["parameters"] = P.parameters;
      rp["derived_from"] = Json{{"field", text(src_row, "id")}, {"name", text(src_row, "name")}};
      rp["carried_over"] = Json::array({"channel flags", "events", "bad segments", "projectors", "ctf_compensation"});
      std::int64_t written = 0;
      const std::int64_t io_block = nxr::bst::recording_block_samples();
      const std::string what = d.outputs[*rec_out].value("description", d.name + " of " + text(src_row, "name"));
      sub.compose([&] {
        nxr::bst::add_derived_recording(sub, text(src_row, "name"), rname, what, d.name, rp, [&](const nxr::bst::BlockWriter& write) {
          std::vector<std::vector<float>> pending;                          // per channel, the samples not yet written
          std::int64_t rows = 0, held = 0;
          auto flush = [&](bool last) {
            while (held >= io_block || (last && held > 0)) {
              const std::int64_t n = std::min(io_block, held);
              std::vector<float> x(static_cast<std::size_t>(rows * n));
              for (std::int64_t r = 0; r < rows; ++r) {
                std::copy(pending[static_cast<std::size_t>(r)].begin(), pending[static_cast<std::size_t>(r)].begin() + n, x.begin() + r * n);
                pending[static_cast<std::size_t>(r)].erase(pending[static_cast<std::size_t>(r)].begin(), pending[static_cast<std::size_t>(r)].begin() + n);
              }
              write(written, x, n);
              written += n; held -= n;
              say(Json{{"stage", "written"}, {"samples", written}});
            }
          };
          loop_with([&](const Blocks& outs) {
            const auto it = outs.find(*rec_out);
            if (it == outs.end()) refuse("outputs." + *rec_out + ": the chunk loop made no block of it");
            const Block& b = it->second;
            if (pending.empty()) { rows = b.rows; pending.assign(static_cast<std::size_t>(rows), {}); }
            for (std::int64_t r = 0; r < rows; ++r)
              pending[static_cast<std::size_t>(r)].insert(pending[static_cast<std::size_t>(r)].end(), b.re.begin() + r * b.width + b.left, b.re.begin() + r * b.width + b.left + b.own());
            held += b.own();
            flush(false);
          });
          flush(true);
        }, [&](const Json& j) { say(j); });
      });
      recording_report = Json{{"output", *rec_out}, {"kind", "recording"}, {"name", rname}, {"samples", written}};
    }
    facts = Json{{"span", span_json}, {"projectors", projection_json}, {"halo", H}, {"blocks", blocks_n}, {"block_samples", opt.block_samples}, {"block_frames", opt.block_frames}, {"chunks_read", cr.visited}, {"samples_read", cr.samples}, {"bad_samples", cr.bad_samples}};
  }
  const nxr::graph::Arrays data = ex->finish();
  // what each step's preparation resolved (a degree chosen from a tolerance …): reported by the step (`record`), generic, into the provenance below
  Json prepared = Json::object();
  {
    const Json recorded = ex->record();
    for (std::size_t i = 0; i < P.steps.size(); ++i)
      if (P.steps[i].step->record && recorded[i]["resolved"].is_object() && !recorded[i]["resolved"].empty()) prepared[P.steps[i].step->name + ":" + P.steps[i].use.outs.front()] = recorded[i]["resolved"];
  }
  // what a step's preparation did or fell back on (a spatial band's Chebyshev fallback) is said, and kept in the report and the provenance: never silent
  Json warnings = Json::array(), messages = Json::array();
  for (const auto& [k, rec] : prepared.items()) {
    if (!rec.is_object()) continue;
    const auto& sb = rec.contains("spatial_band") ? rec["spatial_band"] : rec;
    auto once = [](Json& list, const Json& x) { for (const auto& y : list) if (y == x) return; list.push_back(x); };   // a level read by several bands is said once
    if (sb.is_object() && sb.contains("warnings") && sb["warnings"].is_array()) for (const auto& w : sb["warnings"]) once(warnings, w);
    if (sb.is_object() && sb.contains("messages") && sb["messages"].is_array()) for (const auto& m : sb["messages"]) once(messages, m);
  }
  for (const auto& m : made_first)
    messages.push_back("eigenmodes made first, as their own run: " + std::to_string(m.value("modes", std::int64_t(0))) + " modes of " + m.value("surface", std::string()) + " (" + m.value("why", std::string()) + ")");

  // STORE: each output a Field on the manifold its steps made, its provenance and the cache key with it; another job's node refused unless replace (moved aside)
  Json producer = Json{{"pipeline", Json{{"name", d.name}, {"version", d.version}, {"definition", d.source}, {"sha256", nxr::io::sha256_hex(d.text)}}},
                       {"parameters", P.parameters}, {"inputs", nxr::io::finder::describe(P.found)}, {"run", facts}, {"prepared", prepared}, {"warnings", warnings}, {"messages", messages}, {"made_first", made_first}, {"nxr_neuro", NXR_NEURO_VERSION}};
  if (group) {                                                              // the members, and what each one's input found: the result is of these
    Json members = Json::object();
    for (const auto& [k, list] : group->ids) { Json per = Json::object(); for (std::size_t i = 0; i < group->subjects.size(); ++i) per[group->subjects[i]] = list[i]; members[k] = per; }
    producer["group"] = Json{{"subjects", group->subjects}, {"inputs", members}};
  }
  if (!P.preset.empty()) producer["preset"] = P.preset;                      // provenance only: the key holds the resolved parameters
  if (!d.chains.empty()) producer["chains"] = d.chains;
  if (!d.made.empty()) producer["made"] = d.made;
  {                                                                         // T3: each output's type as the checker typed it, written with it and read back by the next run's finder
    Json types = Json::object();
    for (const auto& o : checked.outputs) if (checked.values.count(o)) types[o] = nxr::graph::print(checked.values.at(o));
    producer["types"] = types;
  }
  const Json stamped = nxr::pipelines::stamp(key);                         // held: iterating a temporary's items() would dangle
  for (const auto& [k, v] : stamped.items()) producer[k] = v;
  stored = Json::array();
  if (!recording_report.is_null()) stored.push_back(recording_report);
  bool replaced = false;
  // ONE UNIT for the whole result (io's task API): the nodes, their arrays, the measurement rows, the recipe and the cache key (in each producer record) are stored together, or nothing is
  nxr::pipelines::task::Subject subject = nxr::pipelines::task::Subject::of(sub);
  nxr::pipelines::task::Unit unit = subject.unit(nxr::pipelines::task::By{made_by(key), producer});
  {
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
    // DECLARED OUTPUTS (declared_outputs.h): the manifolds the definition makes FIRST, so that any other output can lie on them; then each of the others as the declaration and the node's description say
    MadeIds made;
    // an output the node made none of: not in its arrays, or an array it marks `absent` (a step must hand every port; an optional output it does not make is such a placeholder)
    auto made_none = [&](const std::string& o) { const auto it = data.find(o); return it == data.end() || it->second.meta.value("absent", false); };
    for (const auto& [o, ospec] : d.outputs.items())
      if (is_declared(P, o) && ospec["kind"] == "manifold") {
        if (made_none(o)) { stored.push_back(Json{{"output", o}, {"stored", false}, {"why", "its node made none (an optional output)"}}); continue; }
        stored.push_back(store_declared(unit, sub, P, o, data.at(o), span_json, made, opt.replace, replaced));
      }
    std::set<std::string> packed;                                          // the measurement outputs already stored with an earlier one's array
    auto pass = [](const Value& v) { return v.cls == "labelling" ? 1 : over_events(v) ? 2 : v.cls == "statistic" ? 3 : 0; };
    // two more passes: the declared operators (they name the picks and statistics stored before them), then the fields of an operator (`of_operator`)
    auto declared_pass = [&](const std::string& o, const Json& spec, const Value& v) {
      if (!is_declared(P, o)) return pass(v);
      if (ospec_kind(spec) == "operator") return 4;
      if (ospec_kind(spec) == "field" && spec.contains("of_operator")) return 5;
      if (ospec_kind(spec) == "selection" && spec.value("type", std::string()) == "product") return 6;   // a product: its factors (selections of this run) are stored before it
      if (ospec_kind(spec) == "measurement") return 7;                    // over a partition, of a field stored before them
      return pass(v);
    };
    for (int ps = 0; ps < 8; ++ps)
    for (const auto& [o, spec] : d.outputs.items()) try {
      const Value& v = P.values.at(o);
      if (declared_pass(o, spec, v) != ps) continue;
      if (spec.value("kind", std::string()) == "recording") continue;     // written as the loop ran
      if (is_declared(P, o) && made_none(o)) { stored.push_back(Json{{"output", o}, {"stored", false}, {"why", "its node made none (an optional output)"}}); continue; }
      if (is_declared(P, o)) { if (ospec_kind(spec) != "manifold") stored.push_back(store_declared(unit, sub, P, o, data.at(o), span_json, made, opt.replace, replaced)); continue; }
      if (v.cls == "measurement" && v.facts.value("measure", std::string()) == "tile_crossings") { stored.push_back(store_crossings(sub, unit, P, o, v, data.at(o), made_by(key), opt.replace, replaced)); continue; }
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
        for (auto& j : store_measurements(sub, unit, P, group, data, made_by(key), opt.replace, replaced)) stored.push_back(std::move(j));
        continue;
      }
      if (v.cls == "labelling") {
        Json lj = store_labelling(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced);
        if (lj.contains("id") && lj["id"].is_string()) made[o] = lj["id"].get<std::string>();   // a measurement declared over these events finds them by their output name
        stored.push_back(std::move(lj));
        continue;
      }
      if (v.cls == "spans") {
        Json js = store_spans(ds, sub, P, o, spec, v, data.at(o), producer, key, opt.replace, replaced);
        if (js.contains("id") && js["id"].is_string()) made[o] = js["id"].get<std::string>();   // a measurement declared over these events finds them by their output name
        stored.push_back(std::move(js));
        continue;
      }
      if (v.cls == "statistic") { stored.push_back(data.at(o).meta.contains("generic_statistic") ? store_generic(ds, sub, P, o, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("histogram") ? store_histogram(ds, sub, P, o, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("home_map") ? store_home(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)
                                                          : data.at(o).meta.contains("baseline") ? store_baseline(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)
                                                                                                  : store_statistic(ds, sub, P, o, v, data.at(o), producer, key, opt.replace, replaced)); continue; }
      std::string mid = v.manifolds.size() == 1 ? v.manifolds[0] : sub.product(v.manifolds);
      // THE GLUED CORTEX: a Field on the derived glued surface is RESTRICTED BACK to the original for storage (glued -> original, exact; the bridge dropped like the medial wall), its provenance saying so
      Array back;
      std::optional<GlueMap> glue;
      if (v.manifolds.size() == 1) glue = bound.glue_of(mid);
      if (glue && data.at(o).rows * data.at(o).cols == glue->n_glued() * components(v)) {
        const auto orig = sub.db().read("manifold", glue->original_id);
        if (!orig) refuse("outputs." + o + ": the original surface of the glued one it lies on is not among the subject's rows");
        back = data.at(o);
        back.data = restrict_to_original(data.at(o).data, components(v), *glue, nxr::io::integer(*orig, "n_vertices"));
        back.rows = nxr::io::integer(*orig, "n_vertices"); back.cols = components(v);
        mid = glue->original_id;
      } else {
        glue.reset();
      }
      const auto m = sub.db().read("manifold", mid);
      const std::string node = node_name(d, o, P.found), path = nxr::io::node_path(sub.db(), mid, node);
      nxr::pipelines::clear_stored(ds, sub, {{"field", path}}, opt.replace, d.name + " output " + o + " (" + path + ")", replaced);
      const Array& a = glue ? back : data.at(o);
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
      if (glue) pj["restricted_from"] = Json{{"glued", glue->glued_id}, {"original", glue->original_id}, {"dropped", "the bridge"}};
      // a tangent vector field (a complex field's form, the phase current) is stored as a vector3 like a free-orientation current: the Studio's type says which (backend valueType.ts algebraOf)
      if (v.value_type == "vector3" && (v.facts.value("field_type", std::string()) == "complex" || v.facts.value("made_by", std::string()) == "phase_current")) pj["studio"] = Json{{"type", "complex"}};
      fs.cols.json("producer_json", pj);
      const std::string fid = sub.field(std::move(fs));
      stored.push_back(Json{{"output", o}, {"path", path}, {"id", fid}, {"on", m ? text(*m, "name") : mid}});
    } catch (const nlohmann::json::exception& e) {
      throw nxr::io::Error(nxr::io::code::invalid_argument, "outputs." + o + ": a record it is stored from is incomplete (" + e.what() + ")");
    }
  }
  // THE OPERATORS A NODE BUILT (the fused operator of a chain): stored through io as operators with their provenance, under the key they were found by, in this unit; the next run finds them
  // (`Bound::stored_operator`) instead of rebuilding them. One already stored under the key is left as it is.
  for (const MadeOperator& op : bound.offered_operators()) {
    if (bound.stored_operator(op.key)) continue;
    nxr::pipelines::task::OperatorOut oo;
    oo.name = (op.name.empty() ? std::string("fused operator") : op.name) + " " + nxr::io::sha256_hex(op.key.dump()).substr(0, 12);   // one name per key: two operators of one kernel (another form, another hat) are two rows
    oo.output = "operator";
    oo.kind = op.kind;
    oo.from_manifold_id = op.from_manifold_id; oo.to_manifold_id = op.to_manifold_id;
    if (!op.description.empty()) oo.description = op.description;
    oo.params = Json{{"cache", op.key}};
    oo.producer = op.provenance.is_object() ? op.provenance : Json::object();
    oo.producer["pipeline"] = d.name;
    oo.cache_key = op.key;
    oo.data.shape = {op.matrix->rows(), op.matrix->cols()};
    oo.data.f64.resize(static_cast<std::size_t>(op.matrix->size()));
    for (Eigen::Index r = 0; r < op.matrix->rows(); ++r) for (Eigen::Index c = 0; c < op.matrix->cols(); ++c) oo.data.f64[static_cast<std::size_t>(r * op.matrix->cols() + c)] = (*op.matrix)(r, c);
    unit.operator_(oo);
    stored.push_back(Json{{"output", oo.name}, {"kind", "operator"}, {"stored", true}, {"cache_key", op.key}});
  }
  unit.commit();
  report["hit"] = false;
  report["replaced"] = replaced;
  report["run"] = facts;
  report["prepared"] = prepared;
  report["warnings"] = warnings;
  report["messages"] = messages;
  report["made_first"] = made_first;
  report["stored"] = stored;
  return report;
}


namespace {
// THE JOB ROW OF A RUN (docs/specs/2026-11-20-data-layer-queries.md, 3.8): written after the plan's execution, in its own short unit, as done (what it stored) or failed (its message). A run that was a cache hit did no
// work and leaves no row. The row is io's (nxr/io/task/jobs.h); when it cannot be written the run's own outcome stands and a note says so.
class JobCapture {
 public:
  JobCapture(nxr::io::SubjectWriter& sub, const Definition& d, const Plan& P, const RunOptions& opt, const std::function<void(const Json&)>& progress) : sub_(sub), progress_(progress), on_(!opt.requested_by.empty()) {
    if (!on_) return;
    job_.kind = "run:" + d.name;
    job_.requested_by = opt.requested_by;
    job_.started_utc = nxr::io::now_utc();
    job_.params = Json{{"pipeline", d.name + "@" + std::to_string(d.version)}, {"parameters", P.parameters}, {"chunks", opt.chunks ? Json(*opt.chunks) : Json()}, {"start", opt.start ? Json(*opt.start) : Json()},
                       {"stop", opt.stop ? Json(*opt.stop) : Json()}, {"replace", opt.replace}, {"preset", P.preset.empty() ? Json() : Json(P.preset)}};
    job_.inputs = nxr::io::finder::describe(P.found);
  }
  void done(const Json& report) {
    if (!on_ || report.value("hit", false)) return;
    job_.status = "done";
    job_.result = Json{{"stored", report.contains("stored") ? report["stored"] : Json::array()}};
    write();
  }
  void failed(const std::string& message) {
    if (!on_) return;
    job_.status = "failed";
    job_.error = message;
    write();
  }

 private:
  void write() {
    job_.finished_utc = nxr::io::now_utc();
    try { nxr::io::task::record_job(sub_, job_); }
    catch (const std::exception& e) { if (progress_) progress_(Json{{"stage", "note"}, {"note", std::string("the job row was not written: ") + e.what()}}); }
  }
  nxr::io::SubjectWriter& sub_;
  const std::function<void(const Json&)>& progress_;
  nxr::io::task::JobRecord job_;
  bool on_;
};
}  // namespace

std::vector<Json> run(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Definition& d, const Json& set, const RunOptions& opt, bool dry_run,
                      const std::function<void(const Json&)>& progress) {
  if (d.outputs.empty()) refuse("run " + d.name + ": the definition stores nothing (its steps end in sink nodes): compile it instead; run has nothing to store");
  const auto plans = plan_runs(d, sub.db(), sub.id(), set, opt.preset, [&sub](const std::string& recording) { return nxr::pipelines::channel_facts(sub, recording); });     // every run planned before any is executed
  std::vector<Json> out;
  for (const auto& P : plans) {
    if (dry_run) {
      Json r = Json{{"pipeline", d.name + "@" + std::to_string(d.version)}, {"plan", P.text(sub.db())}, {"inputs", nxr::io::finder::describe(P.found)}, {"parameters", P.parameters}};
      if (const auto a = d.anchor()) r["anchor"] = text(*P.found.at(*a).row, "name");
      out.push_back(r);
    } else {
      JobCapture job(sub, d, P, opt, progress);
      try {
        out.push_back(execute(ds, sub, P, opt, progress));
        job.done(out.back());
      }
      catch (const std::invalid_argument& e) { job.failed(e.what()); refuse(e.what()); }       // a node's refusal (compute throws std::invalid_argument) is the run's: an io error with its code
      catch (const std::exception& e) { job.failed(e.what()); throw; }
    }
  }
  return out;
}

namespace {
// the grid a member's selection records (`template`, else `grid_key`, in its parameters); null when it records none
Json grid_recorded(const nxr::io::Row& r) {
  const Json pj = Json::parse(nxr::io::opt_text(r, "params_json").value_or("{}"), nullptr, false);
  if (!pj.is_object()) return Json();
  if (pj.contains("template")) return pj["template"];
  if (pj.contains("grid_key")) return pj["grid_key"];
  return Json();
}
// an axis of a stored statistic as members compare it: its name, size and entries, without the ids of what it lies on (each package has its own)
Json axis_compared(const Json& axis) {
  Json a = Json::object();
  if (axis.is_object()) for (const auto& [key, v] : axis.items()) if (key.size() < 3 || key.compare(key.size() - 3, 3, "_id") != 0) a[key] = v;
  return a;
}
// A STORED STATISTIC of every member (a per-subject `statistic` input: each member's feature, found by name), its values a row: [subjects x values], C order over the statistic's axes. Every member's
// has one shape, the same axes (names, sizes, entries; not the ids of what they lie on) and one unit and value type; a mismatch is refused naming the member and the first one.
nxr::graph::Array stack_statistics(const std::vector<GroupMember>& members, const std::vector<Values>& found, const std::string& k, const std::vector<std::string>& subjects) {
  nxr::graph::Array a;
  a.rows = static_cast<std::int64_t>(members.size());
  std::vector<std::int64_t> shape0;
  Json axes0, unit0, type0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    const nxr::io::Row& row = *found[i].at(k).row;
    const Bound b(*members[i].sub, members[i].ds);
    std::vector<std::int64_t> shape;
    const std::vector<double> v = b.array_double(text(row, "path"), &shape);
    Json axes = Json::array();
    for (const auto& ax : Json::parse(nxr::io::opt_text(row, "axes_json").value_or("[]"), nullptr, false)) axes.push_back(axis_compared(ax));
    const auto u = nxr::io::opt_text(row, "unit");
    const Json unit = u && !u->empty() ? Json(*u) : Json();
    const Json type = nxr::io::opt_text(row, "value_type").value_or("scalar");
    if (i == 0) {
      shape0 = shape; axes0 = axes; unit0 = unit; type0 = type;
      a.meta["name"] = text(row, "name");
      a.meta["axes"] = Json::parse(nxr::io::opt_text(row, "axes_json").value_or("[]"), nullptr, false);
      a.meta["stored_precision"] = nxr::io::ArrayReader::open(members[i].sub->store(), text(row, "path")).dtype() == nxr::io::DType::Float64 ? "float64" : "float32";
    } else {
      const std::string who = members[i].name + ": its " + k + " '" + text(row, "name") + "'";
      if (shape != shape0) refuse(who + " has the shape " + Json(shape).dump() + ", " + members.front().name + "'s " + Json(shape0).dump() + " (the features of a cohort have one length)");
      if (axes != axes0) refuse(who + " lies on other axes (" + axes.dump() + ") than " + members.front().name + "'s (" + axes0.dump() + ")");
      if (unit != unit0) refuse(who + " is in " + unit.dump() + ", " + members.front().name + "'s in " + unit0.dump());
      if (type != type0) refuse(who + " holds " + type.dump() + " values, " + members.front().name + "'s " + type0.dump());
    }
    a.data.insert(a.data.end(), v.begin(), v.end());
  }
  a.cols = a.rows > 0 ? static_cast<std::int64_t>(a.data.size()) / a.rows : 0;
  a.meta["subjects"] = subjects; a.meta["member_shape"] = shape0; a.meta["axes_compared"] = axes0; a.meta["value_type"] = type0; a.meta["stack"] = "subjects";
  if (!unit0.is_null()) a.meta["unit"] = unit0;
  return a;
}

// THE MEMBERS' VALUES stacked: for each per-subject selection input, its members' measurements (the measures of the run's `measures` parameter, of the field the member's per-subject field input found, when there is
// one) as one array [subjects x measures x rows x members]. Every member must have the same rows and members (the group tree's codes); a mismatch is refused naming the subject.
// For each per-subject STATISTIC input, the members' stored statistics as rows (stack_statistics).
GroupInputs stack_members(const std::vector<GroupMember>& members, const std::vector<Values>& found, const std::vector<std::string>& per, const Json& params, const std::string& pipeline) {
  GroupInputs g;
  for (const auto& m : members) g.subjects.push_back(m.name);
  std::string field_input;
  for (const auto& k : per) if (found.front().at(k).cls == "field") { field_input = k; break; }
  bool any_keyed = false, any_selection = false;
  for (const auto& k : per) { any_keyed = any_keyed || found.front().at(k).cls == "measurement"; any_selection = any_selection || found.front().at(k).cls == "selection"; }
  if (any_selection && !any_keyed && (!params.contains("measures") || !params["measures"].is_array() || params["measures"].empty())) refuse(pipeline + ": a group run states the measures to roll up (the parameter `measures`, a list)");
  for (const auto& k : per) {
    for (std::size_t i = 0; i < members.size(); ++i) g.ids[k].push_back(text(*found[i].at(k).row, "id"));
    if (found.front().at(k).cls == "statistic") { g.stacked[k] = stack_statistics(members, found, k, g.subjects); continue; }
    const bool keyed = found.front().at(k).cls == "measurement";            // a stored measurement found by its key: its own measure and of, in each member
    if (found.front().at(k).cls != "selection" && !keyed) continue;
    nxr::graph::Array a;
    a.rows = static_cast<std::int64_t>(members.size());
    std::int64_t R = -1, M = -1;
    Json measures = Json::array();
    const Json measure_list = keyed ? Json::array({found.front().at(k).facts.value("measure", std::string())}) : params["measures"];
    for (std::size_t i = 0; i < members.size(); ++i) {
      const Bound b(*members[i].sub, members[i].ds);
      const Value& fk = found[i].at(k);
      const std::optional<std::string> of = keyed ? (fk.facts.contains("of_field_id") ? std::optional<std::string>(fk.facts["of_field_id"].get<std::string>()) : std::nullopt)
                                                  : (field_input.empty() ? std::nullopt : std::optional<std::string>(text(*found[i].at(field_input).row, "id")));
      for (const auto& mj : measure_list) {
        const std::string measure = mj.get<std::string>();
        const nxr::io::MeasurementValues mv = b.measurement_values(text(*found[i].at(k).row, "id"), of, std::nullopt, measure);
        if (R < 0) { R = mv.rows; M = mv.members; }
        if (mv.rows != R || mv.members != M)
          refuse(members[i].name + ": its '" + measure + "' has " + std::to_string(mv.rows) + " rows x " + std::to_string(mv.members) + " members, the first subject's " + std::to_string(R) + " x " + std::to_string(M) + " (a group lies on one tree's codes)");
        a.data.insert(a.data.end(), mv.values.begin(), mv.values.end());
        if (i == 0) measures.push_back(measure);
      }
    }
    a.cols = static_cast<std::int64_t>(measure_list.size()) * R * M;
    a.meta["subjects"] = g.subjects; a.meta["measures"] = measures; a.meta["rows"] = R; a.meta["members"] = M;
    a.meta["shape"] = Json::array({static_cast<std::int64_t>(members.size()), static_cast<std::int64_t>(measure_list.size()), R, M});
    a.meta["name"] = text(*found.front().at(k).row, "name"); a.meta["stack"] = "subjects";
    g.stacked[k] = std::move(a);
  }
  return g;
}
}  // namespace

std::vector<Json> run_group(nxr::io::DatasetWriter& group_ds, nxr::io::SubjectWriter& group_sub, const std::vector<GroupMember>& members, const Definition& d, const Json& set, const RunOptions& opt, bool dry_run,
                            const std::function<void(const Json&)>& progress) {
  if (d.outputs.empty()) refuse("run " + d.name + ": the definition stores nothing (its steps end in sink nodes): compile it instead; run has nothing to store");
  if (d.scope != "group") refuse(d.name + ": a group run takes a definition of scope group (this one is scope " + d.scope + ")");
  if (members.empty()) refuse(d.name + ": a group run states its subjects (--subjects A,B,C, or a stored group selection by name): there is no default");
  for (const auto& m : members) if (!m.sub || !m.ds) refuse(d.name + ": member '" + m.name + "' is not open");
  const Json base = parameters_of(d, set, opt.preset);
  std::vector<Json> out;
  for (const Json& each : plan_parameter_sets(d, base)) {
    const Json params = resolve_origins(d, each, group_sub.db(), group_sub.id());
    const Definition e = expand(d, params);
    Json shared = Json::object(), per = Json::object();
    for (const auto& [k, spec] : e.inputs.items()) {
      Json sp = spec;
      sp.erase("per_subject");                                              // the definition language's key: the finder reads the model's relations only
      (std::find(e.per_subject.begin(), e.per_subject.end(), k) != e.per_subject.end() ? per : shared)[k] = sp;
    }
    // each member's per-subject inputs: exactly one match, found in the member's own rows
    std::vector<Values> found;
    for (const auto& m : members) {
      std::vector<Values> matches;
      try { matches = nxr::io::finder::resolve(m.sub->db(), m.sub->id(), per, params); }
      catch (const std::exception& e) { refuse(m.name + ": " + d.name + "'s per-subject inputs: " + e.what()); }   // the member a missing input is missing from, named
      if (matches.size() != 1) refuse(m.name + ": " + d.name + " needs exactly one match of its per-subject inputs (" + std::to_string(matches.size()) + " found)");
      found.push_back(matches.front());
    }
    // the members' inputs lie on the group tree's codes: one grid recorded by all (checked before any data is read)
    for (const auto& k : e.per_subject) {
      if (found.front().at(k).cls != "selection") continue;
      const Json g0 = grid_recorded(*found.front().at(k).row);
      for (std::size_t i = 1; i < members.size(); ++i)
        if (grid_recorded(*found[i].at(k).row) != g0)
          refuse(members[i].name + ": its " + k + " lies on another grid (" + grid_recorded(*found[i].at(k).row).dump() + ") than " + members.front().name + "'s (" + g0.dump() + "); a group's inputs lie on the group tree's codes (transfer them to it first)");
    }
    std::vector<Values> shared_matches;
    if (shared.empty()) shared_matches.push_back(Values{});
    else shared_matches = nxr::io::finder::resolve(group_sub.db(), group_sub.id(), shared, params);
    for (const Values& sh : shared_matches) {
      Values all = sh;
      for (const auto& k : e.per_subject) {
        Value v = found.front().at(k);
        v.facts["stack"] = Json(std::vector<std::string>());
        for (const auto& m : members) v.facts["stack"].push_back(m.name);
        all[k] = v;
      }
      Plan P = plan(e, all, params, group_sub.db(), group_sub.id());
      P.preset = opt.preset;
      if (dry_run) {
        Json r = Json{{"pipeline", d.name + "@" + std::to_string(d.version)}, {"plan", P.text(group_sub.db())}, {"parameters", P.parameters}, {"subjects", Json(std::vector<std::string>())}};
        for (const auto& m : members) r["subjects"].push_back(m.name);
        out.push_back(std::move(r));
        continue;
      }
      const GroupInputs g = stack_members(members, found, e.per_subject, params, d.name);
      JobCapture job(group_sub, d, P, opt, progress);
      try {
        out.push_back(execute(group_ds, group_sub, P, opt, progress, &g));
        job.done(out.back());
      }
      catch (const std::exception& e) { job.failed(e.what()); throw; }
    }
  }
  return out;
}

}  // namespace nxr::pipelines::graph

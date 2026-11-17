// nxr-pipelines — a pipeline definition (definition.h): nxr's nxr/pipelines/definition.py `parse`, line for line.
#include "nxr/pipelines/definition.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include "nxr/io/errors.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {
// nxr-neuro's own definitions (pipelines/definitions/*.yaml), compiled in: {name, text}
struct Shipped { const char* name; const char* text; };
const Shipped kShipped[] = {
#include "gen/definitions.inc"
};

const std::vector<std::string> kHeader = {"pipeline", "version", "description", "scope", "draw"}, kSections = {"inputs", "parameters", "presets", "steps", "outputs"};
const std::vector<std::string> kInputClasses = {"field", "operator", "selection", "statistic", "manifold"};
const std::set<std::string> kInputKeys = {"each", "where", "from", "to", "manifold", "is", "required", "description", "hint"}, kStepKeys = {"step", "node", "in", "with", "out", "description"},
                            kOutputKeys = {"kind", "unit", "description", "types",   // types: a span set's event type (outputs of kind spans)
                                           // DECLARED OUTPUTS (docs/specs/2026-11-17-definition-outputs.md, section 3): where and of what; what the values are, the node says
                                           "name", "kind_name", "type", "on", "of", "axes", "partition", "operator_kind", "from", "to", "over", "alt", "cs", "derived_from", "factors",
                                           "from_selection", "to_selection", "of_statistic", "inverts", "inverse_kind", "acquisition_of", "of_operator", "within", "by_operator"};
const std::regex kPresetName("^[a-z][a-z0-9_-]*$");
const std::regex kName("^[a-z][a-z0-9_]*$");

std::string join(const std::vector<std::string>& v, const char* sep = ", ") { std::string s; for (const auto& x : v) s += (s.empty() ? "" : sep) + x; return s; }
std::vector<std::string> keys(const Json& o) { std::vector<std::string> k; for (const auto& [a, b] : o.items()) { (void)b; k.push_back(a); } return k; }
[[noreturn]] void err(const std::string& where, const std::string& msg) { refuse(where + ": " + msg); }

// A PARAMETER, short or long. Short: `name: default`. Long: `name: {default: X, meaning: "...", unit: Hz}` (and `choices: [...]`): the command line's help is made from these
// (docs/specs/2026-11-17-compute-and-run.md, 6.2). The long form is a table with a `default` key and no key but meaning, unit and choices; anything else is a default as written.
bool long_form(const Json& v) {
  if (!v.is_object() || !v.contains("default")) return false;
  for (const auto& [k, _] : v.items()) if (k != "default" && k != "meaning" && k != "unit" && k != "choices") return false;
  return true;
}
// a parameter named as a command-line option the verb reserves cannot be one (the option would be ambiguous): refused when the definition is read
const std::set<std::string> kReserved = {"dataset", "subject", "from", "to", "all", "replace", "dry_run", "json", "help", "quiet", "no_color", "human", "version", "chunks", "start", "stop"};
// the defaults (what the runner reads) and, apart, what each parameter declares about itself
Json split_parameters(const Json& raw, Json& docs, const std::string& where) {
  Json defaults = Json::object();
  for (const auto& [k, v] : raw.items()) {
    if (kReserved.count(k)) err(where, "parameters: '" + k + "' is a reserved option name of `run` (reserved: dataset, subject, from, to, all, replace, dry_run, json, help, quiet, no_color, human, version, chunks, start, stop)");
    Json doc = Json::object();
    if (long_form(v)) {
      defaults[k] = v["default"];
      for (const char* key : {"meaning", "unit", "choices"}) if (v.contains(key)) doc[key] = v[key];
      if (doc.contains("meaning") && !doc["meaning"].is_string()) err(where, "parameters: " + k + ": meaning is a sentence");
      if (doc.contains("unit") && !doc["unit"].is_string()) err(where, "parameters: " + k + ": unit is text ('-' for none)");
      if (doc.contains("choices") && !doc["choices"].is_array()) err(where, "parameters: " + k + ": choices is a list");
    } else {
      defaults[k] = v;
    }
    docs[k] = doc;
  }
  return defaults;
}
std::vector<std::string> names(const Json& x, const std::string& where) {
  std::vector<std::string> out;
  if (x.is_string()) out.push_back(x.get<std::string>());
  else if (x.is_array()) for (const auto& n : x) out.push_back(n.is_string() ? n.get<std::string>() : n.dump());
  else if (!x.is_null()) out.push_back(x.dump());
  for (const auto& n : out) if (!std::regex_match(n, kName)) err(where, "'" + n + "' is not a name (lower case, digits, '_')");
  return out;
}
// every `$name` in a value is a declared parameter
void params_used(const Json& v, const std::string& where, const Json& declared) {
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (!s.empty() && s[0] == '$' && s.rfind("${", 0) != 0 && !declared.contains(s.substr(1))) err(where, s + " is not a parameter (declared: " + (declared.empty() ? std::string("none") : join(keys(declared))) + ")");
    for (std::size_t a = 0; (a = s.find("${", a)) != std::string::npos; ++a) {        // `${name}` inside a name: a declared parameter too
      const std::size_t e = s.find('}', a);
      if (e == std::string::npos || !declared.contains(s.substr(a + 2, e - a - 2))) err(where, s + ": ${...} names no declared parameter");
    }
  } else if (v.is_object() || v.is_array()) {
    for (const auto& x : v) params_used(x, where, declared);
  }
}
std::string section_label(const std::string& sec) { return sec == "do" ? "chunks" : sec; }
bool allowed(const std::string& sec, const std::string& kind) {
  if (sec == "before") return kind == "setup" || kind == "array";
  if (sec == "do") return kind == "chunk" || kind == "accumulate";
  return kind == "after" || kind == "array";
}
std::string home_of(const std::string& kind) { return kind == "setup" ? "before" : kind == "after" ? "after" : kind == "array" ? "before or after" : "chunks"; }

// ── REPETITION (a definition repeated over the elements of list parameters; spec docs/specs/2026-11-13-cortical-events.md, phase 2c) ──
// An input or output entry with `for: {var: $list, …}`, and a step block `{for: {var: $list, …}, do: [steps]}` (nested blocks allowed), are written out once per combination of the lists'
// elements (the first variable outermost) BEFORE anything else reads the definition. In the written-out text a name's `{var}` and a string equal to `$var` are the element; `{var}` in a name is
// the element as digits (a list `[8, 16]` is `8_16`, a decimal point `p`). What is left is the flat definition, so plan, executor, cache key and storage see no repetition.
using VarMap = std::map<std::string, Json>;
bool has_for(const Json& v) {
  if (v.is_object()) { if (v.contains("for")) return true; for (const auto& [k, x] : v.items()) { (void)k; if (has_for(x)) return true; } }
  else if (v.is_array()) for (const auto& x : v) if (has_for(x)) return true;
  return false;
}
std::string as_digits(const Json& v) {
  if (v.is_array()) { std::string o; for (const auto& x : v) o += (o.empty() ? "" : "_") + as_digits(x); return o; }
  if (v.is_number_integer()) return std::to_string(v.get<std::int64_t>());
  if (v.is_number()) {
    const double x = v.get<double>();
    std::string t;
    if (x == std::floor(x) && std::abs(x) < 1e15) t = std::to_string(static_cast<std::int64_t>(x));
    else { char b[40]; std::snprintf(b, sizeof b, "%g", x); t = b; }
    for (char& c : t) if (c == '.') c = 'p'; else if (c == '-') c = 'm'; else if (c == '+') c = 'e';
    return t;
  }
  if (v.is_string()) return v.get<std::string>();
  return v.dump();
}
Json substitute(const Json& v, const VarMap& vars) {
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (s.size() > 1 && s[0] == '$' && vars.count(s.substr(1))) return vars.at(s.substr(1));
    std::string o = s;
    for (const auto& [k, x] : vars) {
      const std::string tag = "{" + k + "}";
      for (std::size_t p; (p = o.find(tag)) != std::string::npos;) o.replace(p, tag.size(), as_digits(x));
    }
    return o;
  }
  if (v.is_array()) { Json a = Json::array(); for (const auto& x : v) a.push_back(substitute(x, vars)); return a; }
  if (v.is_object()) { Json o = Json::object(); for (const auto& [k, x] : v.items()) o[substitute(Json(k), vars).get<std::string>()] = substitute(x, vars); return o; }
  return v;
}
// every combination of a `for:` mapping's lists, each added to `vars`; the first variable is the outermost
std::vector<VarMap> combinations(const Json& spec, const Json& params, const VarMap& vars, const std::string& where) {
  if (!spec.is_object() || spec.empty()) err(where, "for: a mapping of variable → list (or $parameter naming a list), e.g. {band: $bands}");
  std::vector<VarMap> out{vars};
  for (const auto& [var, src] : spec.items()) {
    if (!std::regex_match(var, kName)) err(where, "for: '" + var + "' is not a name (lower case, digits, '_')");
    Json list = src;
    if (src.is_string() && !src.get<std::string>().empty() && src.get<std::string>()[0] == '$') {
      const std::string pn = src.get<std::string>().substr(1);
      if (!params.contains(pn)) err(where, "for " + var + ": " + src.get<std::string>() + " is not a parameter");
      list = params[pn];
    }
    if (!list.is_array() || list.empty()) err(where, "for " + var + ": " + src.dump() + " must be a list with at least one element (it is " + list.dump() + ")");
    std::vector<VarMap> next;
    for (const VarMap& m : out)
      for (const auto& e : list) { VarMap n = m; if (n.count(var)) err(where, "for: '" + var + "' is already a loop variable here"); n[var] = e; next.push_back(std::move(n)); }
    out = std::move(next);
  }
  return out;
}
Json expand_steps(const Json& list, const Json& params, const VarMap& vars, const std::string& where) {
  if (!list.is_array()) return list;
  Json out = Json::array();
  for (const auto& el : list) {
    if (el.is_object() && el.contains("for")) {
      if (el.size() != 2 || !el.contains("do") || !el["do"].is_array()) err(where, "a repeated block is {for: {…}, do: [steps]}");
      for (const VarMap& m : combinations(el["for"], params, vars, where)) for (const auto& x : expand_steps(el["do"], params, m, where)) out.push_back(x);
    } else {
      out.push_back(substitute(el, vars));
    }
  }
  return out;
}
Json expand_entries(const Json& entries, const Json& params, const std::string& where) {
  if (!entries.is_object()) return entries;
  Json out = Json::object();
  for (const auto& [k, v] : entries.items()) {
    if (v.is_object() && v.contains("for")) {
      Json body = v;
      body.erase("for");
      for (const VarMap& m : combinations(v["for"], params, {}, where + "." + k)) {
        const std::string name = substitute(Json(k), m).get<std::string>();
        if (out.contains(name)) err(where, "'" + name + "' is made twice by `for` (name it with the loop variables: {band}, {level})");
        out[name] = substitute(body, m);
      }
    } else {
      out[k] = v;
    }
  }
  return out;
}
Json expand_doc(const Json& doc, const Json& params, const std::string& where) {
  Json d = doc;
  if (d.contains("inputs")) d["inputs"] = expand_entries(d["inputs"], params, where + ": inputs");
  if (d.contains("outputs")) d["outputs"] = expand_entries(d["outputs"], params, where + ": outputs");
  if (d.contains("steps") && d["steps"].is_object()) {
    for (const char* sec : {"before", "after"}) if (d["steps"].contains(sec)) d["steps"][sec] = expand_steps(d["steps"][sec], params, {}, where + ": steps." + sec);
    if (d["steps"].contains("chunks") && d["steps"]["chunks"].is_object() && d["steps"]["chunks"].contains("do"))
      d["steps"]["chunks"]["do"] = expand_steps(d["steps"]["chunks"]["do"], params, {}, where + ": steps.chunks.do");
  }
  return d;
}
}  // namespace

std::optional<std::string> Definition::anchor() const {
  for (const auto& [k, v] : inputs.items()) if (v.is_object() && v.value("each", false)) return k;
  return std::nullopt;
}
std::vector<StepUse> Definition::uses() const {
  std::vector<StepUse> u = before;
  u.insert(u.end(), do_.begin(), do_.end());
  u.insert(u.end(), after.begin(), after.end());
  return u;
}

Definition parse(const Json& written, const std::string& source, const std::string& text) {
  const std::string where = source.empty() ? "pipeline" : source;
  if (!written.is_object()) err(where, "a pipeline definition is a YAML mapping");
  // the parameters' long form read: from here on `parameters` holds the defaults, and what each declares goes to `parameter_docs`
  Json parameter_docs = Json::object();
  Json doc = written;
  if (written.contains("parameters") && written["parameters"].is_object()) doc["parameters"] = split_parameters(written["parameters"], parameter_docs, where);
  if (has_for(doc)) {                      // written out with the parameters' defaults to check it; `expand` writes it out again with a run's own
    const Json defaults = doc.contains("parameters") && doc["parameters"].is_object() ? doc["parameters"] : Json::object();
    Definition d = parse(expand_doc(doc, defaults, where), source, text);
    d.templ = doc;
    d.parameter_docs = parameter_docs;
    return d;
  }
  std::vector<std::string> unknown;
  for (const auto& k : keys(doc))
    if (std::find(kHeader.begin(), kHeader.end(), k) == kHeader.end() && std::find(kSections.begin(), kSections.end(), k) == kSections.end()) unknown.push_back(k);
  std::sort(unknown.begin(), unknown.end());
  if (!unknown.empty()) {
    std::vector<std::string> all = kHeader;
    all.insert(all.end(), kSections.begin(), kSections.end());
    err(where, "unknown section(s) " + join(unknown) + " (sections: " + join(all) + ")");
  }
  Definition d;
  d.source = source;
  d.text = text;
  const Json name = doc.value("pipeline", Json());
  if (!name.is_string() || !std::regex_match(name.get<std::string>(), kName)) err(where, "pipeline " + name.dump() + ": a name (lower case, digits, '_')");
  d.name = name.get<std::string>();
  const Json version = doc.value("version", Json(1));
  if (!version.is_number_integer() || version.get<std::int64_t>() < 1) err(where, "version " + version.dump() + ": a whole number from 1");
  d.version = version.get<int>();
  d.scope = doc.value("scope", Json("subject")).is_string() ? doc.value("scope", std::string("subject")) : std::string("?");
  if (d.scope != "subject") err(where, "scope '" + d.scope + "': only 'subject' runs today (group pipelines are not built yet)");
  if (doc.contains("draw")) {
    if (!doc["draw"].is_string()) err(where, "draw: the name of a parameter");
    d.draw = doc["draw"].get<std::string>();
  }
  d.description = doc.contains("description") && doc["description"].is_string() ? doc["description"].get<std::string>() : "";
  const Json params = doc.contains("parameters") && !doc["parameters"].is_null() ? doc["parameters"] : Json::object();
  if (!params.is_object()) err(where, "parameters: a mapping of name → default");
  names(Json(keys(params)), where + ": parameters");
  d.parameters = params;
  d.parameter_docs = parameter_docs;
  // PRESETS: named parameter sets, each a named alternative (`run NAME --set PRESET`); a preset names only declared parameters
  if (doc.contains("presets") && !doc["presets"].is_null()) {
    if (!doc["presets"].is_object()) err(where, "presets: a mapping of name → {parameter: value}");
    for (const auto& [pn, pv] : doc["presets"].items()) {
      const std::string w = where + ": presets." + pn;
      if (!std::regex_match(pn, kPresetName)) err(w, "a preset's name is lower case, digits, '_' and '-'");
      if (!pv.is_object()) err(w, "a mapping of parameter → value");
      for (const auto& [k, v] : pv.items()) {
        if (!params.contains(k)) err(w, "'" + k + "' is not a parameter (declared: " + (params.empty() ? std::string("none") : join(keys(params))) + ")");
        params_used(v, w, params);
      }
    }
    d.presets = doc["presets"];
  }

  const Json inputs = doc.contains("inputs") && !doc["inputs"].is_null() ? doc["inputs"] : Json::object();
  if (!inputs.is_object() || inputs.empty()) err(where, "inputs: at least one");
  std::vector<std::string> defined;
  for (const auto& [k, spec] : inputs.items()) {
    const std::string w = where + ": inputs." + k;
    names(Json(k), w);
    if (!spec.is_object()) err(w, "a mapping");
    int n = 0;
    for (const auto& c : kInputClasses) n += spec.contains(c) ? 1 : 0;
    if (n > 1 && spec.contains("manifold")) --n;                           // `manifold:` beside another class is that class's relation (a selection ON a manifold), not a class
    if (n != 1) err(w, "name exactly one class: " + join(kInputClasses, " · ") + " (with its kind)");
    std::vector<std::string> bad;
    for (const auto& kk : keys(spec))
      if (!kInputKeys.count(kk) && std::find(kInputClasses.begin(), kInputClasses.end(), kk) == kInputClasses.end()) bad.push_back(kk);
    std::sort(bad.begin(), bad.end());
    if (!bad.empty()) err(w, "unknown key(s) " + join(bad));
    const bool manifold_class = !spec.contains("field") && !spec.contains("operator") && !spec.contains("selection") && !spec.contains("statistic");
    for (const char* rel : {"from", "to", "manifold", "is"})
      if (spec.contains(rel) && !(manifold_class && std::string(rel) == "manifold")) {   // a manifold input's `manifold:` is its class (surface, points), not a relation
        const std::string r = spec[rel].is_string() ? spec[rel].get<std::string>() : spec[rel].dump();
        const std::string base = r.substr(0, r.find('.'));
        if (std::find(defined.begin(), defined.end(), base) == defined.end())
          err(w, std::string(rel) + ": '" + r + "': relations start from an input declared BEFORE this one (" + (defined.empty() ? std::string("none") : join(defined)) + ")");
      }
    if (spec.contains("where")) params_used(spec["where"], w, params);
    defined.push_back(k);
  }
  std::vector<std::string> anchors;
  for (const auto& [k, v] : inputs.items()) if (v.value("each", false)) anchors.push_back(k);
  if (anchors.size() > 1) err(where, "inputs: one input runs the pipeline per match (each: true), not " + join(anchors));
  d.inputs = inputs;

  const Json steps = doc.contains("steps") && !doc["steps"].is_null() ? doc["steps"] : Json::object();
  if (!steps.is_object()) err(where, "steps: a mapping of before · chunks · after");
  for (const auto& k : keys(steps)) if (k != "before" && k != "chunks" && k != "after") err(where, "steps: a mapping of before · chunks · after");
  const Json chunks = steps.contains("chunks") ? steps["chunks"] : Json();
  if (!chunks.is_null()) {
    if (!chunks.is_object()) err(where, "steps.chunks: {over: <input>, do: [steps]}");
    for (const auto& k : keys(chunks)) if (k != "over" && k != "do") err(where, "steps.chunks: {over: <input>, do: [steps]}");
    const Json over = chunks.value("over", Json());
    if (!over.is_string() || !inputs.contains(over.get<std::string>()) || !inputs[over.get<std::string>()].contains("field"))
      err(where, "steps.chunks.over " + over.dump() + ": an input field (the recording the loop walks)");
    d.over = over.get<std::string>();
  }

  std::set<std::string> known(defined.begin(), defined.end());
  bool accumulates = false;
  for (const std::string sec : {"before", "do", "after"}) {
    const Json raw = sec == "do" ? (chunks.is_object() ? chunks.value("do", Json()) : Json()) : steps.value(sec, Json());
    if (!raw.is_null() && !raw.is_array()) err(where, "steps." + section_label(sec) + ": a list of steps");
    std::vector<StepUse>& out = sec == "before" ? d.before : sec == "do" ? d.do_ : d.after;
    for (std::size_t i = 0; raw.is_array() && i < raw.size(); ++i) {
      const Json& s = raw[i];
      std::string w = where + ": steps." + (sec == "do" ? "chunks.do" : sec) + "[" + std::to_string(i) + "]";
      const char* use_key = s.is_object() && s.contains("node") ? "node" : "step";           // `node:` (the grouped names) and `step:` are one
      if (!s.is_object() || !s.contains(use_key) || !s[use_key].is_string() || (s.contains("node") && s.contains("step"))) err(w, "{node: <name>, in: {…}, with: {…}, out: …}");
      std::vector<std::string> bad;
      for (const auto& k : keys(s)) if (!kStepKeys.count(k)) bad.push_back(k);
      std::sort(bad.begin(), bad.end());
      if (!bad.empty()) err(w, "unknown key(s) " + join(bad));
      const StepSpec& st = step(s[use_key].get<std::string>());
      w += " (" + st.name + ")";
      if (!allowed(sec, st.kind)) err(w, "a " + st.kind + " step cannot be in '" + section_label(sec) + "' (it belongs in " + home_of(st.kind) + ")");
      if (sec == "do" && st.kind == "accumulate") accumulates = true;
      const Json ins = s.contains("in") && !s["in"].is_null() ? s["in"] : Json::object();
      if (!ins.is_object()) err(w, "in: a mapping of port → thing");
      for (const auto& [port, thing] : ins.items()) {
        if (std::find(st.inputs.begin(), st.inputs.end(), port) == st.inputs.end()) err(w, "in: '" + port + "' is not one of its ports (" + join(st.inputs) + ")");
        if (!thing.is_string() || !known.count(thing.get<std::string>())) err(w, "in." + port + ": " + thing.dump() + " is neither an input nor an earlier step's output");
      }
      std::vector<std::string> missing;
      for (const auto& p : st.inputs)
        if (!ins.contains(p) && std::find(st.optional.begin(), st.optional.end(), p) == st.optional.end()) missing.push_back(p);
      if (!missing.empty()) err(w, "in: its port(s) " + join(missing) + " are not given");
      const Json with = s.contains("with") && !s["with"].is_null() ? s["with"] : Json::object();
      if (!with.is_object()) err(w, "with: a mapping of parameter → value");
      for (const auto& p : keys(with))
        if (!st.parameters.contains(p)) err(w, "with: '" + p + "' is not one of its parameters (" + (st.parameters.empty() ? std::string("none") : join(keys(st.parameters))) + ")");
      params_used(with, w, params);
      const std::vector<std::string> outs = names(s.value("out", Json()), w + ": out");
      if (outs.size() != st.outputs.size())
        err(w, "out: it makes " + std::to_string(st.outputs.size()) + " thing(s) (" + join(st.outputs) + "), " + std::to_string(outs.size()) + " named");
      for (const auto& o : outs) {
        if (known.count(o)) err(w, "out: '" + o + "' is already defined");
        known.insert(o);
      }
      out.push_back(StepUse{st.name, ins, with, outs, sec});
    }
  }
  bool writes_recording = false;                                           // a recording output (a filter's) leaves the loop block by block, written as it runs
  if (doc.contains("outputs") && doc["outputs"].is_object())
    for (const auto& [o, spec] : doc["outputs"].items()) { (void)o; if (spec.is_object() && spec.value("kind", std::string()) == "recording") writes_recording = true; }
  if (d.over && !accumulates && !writes_recording) err(where, "steps.chunks: nothing leaves the loop without an accumulate step or a recording output");

  const Json outputs = doc.contains("outputs") && !doc["outputs"].is_null() ? doc["outputs"] : Json::object();
  if (!outputs.is_object() || outputs.empty()) err(where, "outputs: at least one (what the pipeline stores)");
  for (const auto& [k, spec] : outputs.items()) {
    const std::string w = where + ": outputs." + k;
    if (!known.count(k) || inputs.contains(k)) err(w, "not made by any step");
    bool ok = spec.is_object() && spec.contains("kind");
    if (ok) for (const auto& kk : keys(spec)) if (!kOutputKeys.count(kk)) ok = false;
    if (!ok) err(w, "{kind: …, unit: …, description: …} (and types: [name] for a span set; the declared kinds also name where and of what: on, of, axes, partition, …)");
    const std::string kind = spec["kind"].is_string() ? spec["kind"].get<std::string>() : std::string();
    if (kind.empty()) err(w, "kind: a name");
    const bool declared_kind = kind == "operator" || kind == "manifold" || (kind == "selection") || (kind == "statistic" && (spec.contains("axes") || spec.contains("of")));
    for (const auto& kk : keys(spec))                // the declaration's keys belong to the declared kinds only
      if (!declared_kind && kind != "field" && kind != "measurement" && kind != "recording" && std::set<std::string>{"name", "kind_name", "type", "on", "of", "axes", "partition", "operator_kind", "from", "to", "over", "alt", "cs", "derived_from", "factors", "from_selection", "to_selection", "of_statistic", "inverts", "inverse_kind", "acquisition_of", "of_operator", "within", "by_operator"}.count(kk))
        err(w, kk + ": belongs to the declared kinds (statistic with axes, operator, selection, manifold, field)");
    if (kind == "operator" && (!spec.contains("operator_kind") || !spec.contains("from") || !spec.contains("to"))) err(w, "an operator output names operator_kind, from and to (links to found manifolds)");
    if (kind == "selection" && !spec.contains("type")) err(w, "a selection output names its type: partition, spans, set, tiling, ladder or grid");
    if (kind == "manifold" && !spec.contains("type")) err(w, "a manifold output names its type: line, surface or product");
    if (spec.contains("axes") && !spec["axes"].is_array()) err(w, "axes: a list of {name, on | line}");
  }
  d.outputs = outputs;
  return d;
}

Definition expand(const Definition& d, const Json& params) {
  if (d.templ.is_null()) return d;
  Definition e = parse(expand_doc(d.templ, params, d.source.empty() ? "pipeline" : d.source), d.source, d.text);
  e.templ = d.templ;
  e.draw = d.draw;
  return e;
}

Definition load(const std::string& name_or_path) {
  const std::filesystem::path p(name_or_path);
  if (p.extension() == ".yaml" || p.extension() == ".yml") {
    std::ifstream f(p);
    if (!f) refuse(p.string() + ": no such pipeline definition");
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(nxr::io::parse_yaml(ss.str(), p.string()), p.string(), ss.str());
  }
  for (const auto& s : kShipped)
    if (name_or_path == s.name) return parse(nxr::io::parse_yaml(s.text, std::string(s.name) + ".yaml"), std::string(s.name) + ".yaml", s.text);
  refuse("no pipeline '" + name_or_path + "' (nxr-neuro's: " + (available().empty() ? std::string("none") : join(available())) + "; or give a .yaml path)");
}

std::vector<std::string> available() {
  std::vector<std::string> v;
  for (const auto& s : kShipped) v.push_back(s.name);
  std::sort(v.begin(), v.end());
  return v;
}

}  // namespace nxr::pipelines::graph

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
#include "nxr/io/hash.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {
// nxr-neuro's own definitions (pipelines/definitions/*.yaml), compiled in: {name, text}
struct Shipped { const char* name; const char* text; };
const Shipped kShipped[] = {
#include "gen/definitions.inc"
};

const std::vector<std::string> kHeader = {"pipeline", "version", "description", "scope", "draw", "kind", "plans", "one_of"}, kSections = {"inputs", "parameters", "presets", "steps", "outputs"};
const std::vector<std::string> kInputClasses = {"field", "operator", "selection", "measurement", "statistic", "manifold"};
const std::set<std::string> kInputKeys = {"each", "where", "from", "to", "manifold", "is", "required", "description", "hint", "read", "per_subject", "of", "over", "within"}, kStepKeys = {"step", "node", "in", "with", "out", "description"},
                            kOutputKeys = {"kind", "unit", "description", "types",   // types: a span set's event type (outputs of kind spans)
                                           // DECLARED OUTPUTS (docs/specs/2026-11-17-definition-outputs.md, section 3): where and of what; what the values are, the node says
                                           "name", "kind_name", "type", "on", "of", "axes", "partition", "operator_kind", "from", "to", "over", "alt", "cs", "derived_from", "factors",
                                           "from_selection", "to_selection", "of_statistic", "inverts", "inverse_kind", "acquisition_of", "of_operator", "within", "by_operator",
                                           "codes", "carry_origin", "optional", "get_or_create", "value_type", "components", "level", "detected_over"};   // detected_over: the grid input an irregular selection was detected over (provenance)
  //    // get_or_create: a maker's selection stored by its maker key (found when there)  // codes: 2 = two codes a row (a sparse matrix over a selection's members) · carry_origin: the input whose origin and licence the result carries  // optional: the node makes it only when the data asks for it; a link may name alternatives `a|b`
const std::regex kPresetName("^[a-z][a-z0-9_-]*$");
const std::regex kName("^[a-z][a-z0-9_]*$");

std::string join(const std::vector<std::string>& v, const char* sep = ", ") { std::string s; for (const auto& x : v) s += (s.empty() ? "" : sep) + x; return s; }
std::vector<std::string> keys(const Json& o) { std::vector<std::string> k; for (const auto& [a, b] : o.items()) { (void)b; k.push_back(a); } return k; }
[[noreturn]] void err(const std::string& where, const std::string& msg) { refuse(where + ": " + msg); }

// A PARAMETER, short or long. Short: `name: default`. Long: `name: {default: X, meaning: "...", unit: Hz}` (and `choices: [...]`): the command line's help is made from these
// (docs/specs/2026-11-17-compute-and-run.md, 6.2). The long form is a table with a `default` key and no key but meaning, unit and choices; anything else is a default as written.
bool long_form(const Json& v) {
  if (!v.is_object() || !v.contains("default")) return false;
  for (const auto& [k, _] : v.items()) if (k != "default" && k != "meaning" && k != "unit" && k != "chart" && k != "choices") return false;
  return true;
}
// A BARE LEVEL INDEX is never a parameter (consolidation spec 3.1 rule 2): a selection parameter is a QUANTITY (a number with a unit in a named chart). The names the spec's table 3.5 retires, and a
// parameter whose unit is "level" or "index", are refused unless the parameter says `chart: index` (the explicit diagnostics form), with what replaces them.
const std::map<std::string, std::string> kIndexNames = {
    {"depth", "finest: <tile diameter mm | tile area mm2 | parts per hemisphere>"}, {"atlas_depth", "finest: <tile diameter mm | tile area mm2 | parts per hemisphere>"},
    {"lc_levels", "the Levi-Civita levels as tile diameters (mm)"}, {"level", "tile: <duration s>, band: <Hz>, or spatial_scale: <tile diameter mm | wavelength mm | eigenvalue 1/m2 | named scale>"},
    {"space_level", "spatial_scale: <tile diameter mm | wavelength mm | eigenvalue 1/m2 | named scale>"}, {"space_levels", "spatial_scales: a list of quantities (tile diameter mm | wavelength mm | eigenvalue 1/m2 | named scale)"},
    {"levels", "spatial_scales: a list of quantities"}, {"reference_level", "reference_scale: a quantity"}, {"time_level", "band: <Hz> that couples it, or tile: <duration s>"},
    {"frame_level", "frames: band <Hz> with 2^s frames per tile, or frame: <duration s>"}, {"min_cells", "area: >= <mm2>"}};
void refuse_bare_index(const std::string& name, const Json& doc, const std::string& where) {
  const std::string chart = doc.contains("chart") && doc["chart"].is_string() ? doc["chart"].get<std::string>() : std::string();
  if (doc.contains("chart") && (!doc["chart"].is_string() || chart.empty())) err(where, "parameters: " + name + ": chart is the name of a coordinate map of the space the quantity lies in (linear, log10, clock, eigenvalue, wavelength, area, diameter, parts, origin, index)");
  if (chart == "index") return;
  const auto it = kIndexNames.find(name);
  const std::string unit = doc.contains("unit") && doc["unit"].is_string() ? doc["unit"].get<std::string>() : std::string();
  if (it != kIndexNames.end() && chart.empty())
    err(where, "parameters: '" + name + "' is a bare level index, which is never a parameter; state a quantity instead: " + it->second + " (`chart: index` names an index for diagnostics only)");
  if ((unit == "level" || unit == "levels" || unit == "index") && chart.empty())
    err(where, "parameters: " + name + ": a unit of 'level' or 'index' is a bare index; state the quantity (a duration, a band, a tile size, a wavelength), or `chart: index` for diagnostics");
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
      for (const char* key : {"meaning", "unit", "chart", "choices"}) if (v.contains(key)) doc[key] = v[key];
      if (doc.contains("meaning") && !doc["meaning"].is_string()) err(where, "parameters: " + k + ": meaning is a sentence");
      if (doc.contains("unit") && !doc["unit"].is_string()) err(where, "parameters: " + k + ": unit is text ('-' for none)");
      if (doc.contains("choices") && !doc["choices"].is_array()) err(where, "parameters: " + k + ": choices is a list");
    } else {
      defaults[k] = v;
    }
    refuse_bare_index(k, doc, where);
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

bool has_for(const Json& v);
// ── `when:` (a step or an output that is part of the run only when a parameter is given) ──
bool truthy(const Json& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number()) return v.get<double>() != 0;
  if (v.is_string()) return !v.get<std::string>().empty() && v.get<std::string>() != "false";
  return !v.empty();
}
bool has_key_deep(const Json& v, const char* key) {
  if (v.is_object()) { if (v.contains(key)) return true; for (const auto& [k, x] : v.items()) { (void)k; if (has_key_deep(x, key)) return true; } }
  else if (v.is_array()) for (const auto& x : v) if (has_key_deep(x, key)) return true;
  return false;
}
// a definition is a TEMPLATE (written out for each run by `expand`) when it repeats (`for:`), has conditions (`when:`) or composes (`chain:` and `make:` inputs)
bool needs_expansion(const Json& doc) {
  if (has_for(doc)) return true;
  if (doc.contains("steps") && has_key_deep(doc["steps"], "when")) return true;
  if (doc.contains("outputs") && has_key_deep(doc["outputs"], "when")) return true;
  if (doc.contains("inputs") && doc["inputs"].is_object())
    for (const auto& [k, v] : doc["inputs"].items()) { (void)k; if (v.is_object() && (v.contains("chain") || v.contains("make") || v.contains("when"))) return true; }
  return false;
}
// a value with its `$name`s replaced by the run's parameters (a name not declared is refused by the caller)
Json with_params(const Json& v, const Json& params) {
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (s.size() > 1 && s[0] == '$' && params.contains(s.substr(1))) return params[s.substr(1)];
    return v;
  }
  if (v.is_array()) { Json a = Json::array(); for (const auto& x : v) a.push_back(with_params(x, params)); return a; }
  if (v.is_object()) { Json o = Json::object(); for (const auto& [k, x] : v.items()) o[k] = with_params(x, params); return o; }
  return v;
}
bool holds(const Json& when, const Json& params, const std::string& where) {
  if (when.is_null()) return true;
  if (when.is_string()) {
    std::string s = when.get<std::string>();
    const bool neg = !s.empty() && s[0] == '!';
    if (neg) s = s.substr(1);
    if (s.empty() || s[0] != '$') return truthy(when) != neg;           // a loop variable already replaced by its value (a string)
    s = s.substr(1);
    if (!params.contains(s)) err(where, "when: $" + s + " is not a parameter (declared: " + (params.empty() ? std::string("none") : join(keys(params))) + ")");
    return truthy(params[s]) != neg;
  }
  if (when.is_object()) {
    for (const auto& [k, v] : when.items()) {
      if (!params.contains(k)) err(where, "when: " + k + " is not a parameter (declared: " + (params.empty() ? std::string("none") : join(keys(params))) + ")");
      if (params[k] != v) return false;
    }
    return true;
  }
  return truthy(when);                                                    // a loop variable's value
}

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
      Json one = substitute(el, vars);
      if (one.is_object() && one.contains("when")) {                      // a step left out when its condition fails; the key never reaches the flat definition
        const bool on = holds(one["when"], params, where);
        one.erase("when");
        if (!on) continue;
      }
      out.push_back(std::move(one));
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
    } else if (v.is_object() && v.contains("when")) {                      // an output left out when its condition fails
      Json body = v;
      const bool on = holds(v["when"], params, where + "." + k);
      body.erase("when");
      if (on) out[k] = body;
    } else {
      out[k] = v;
    }
  }
  return out;
}
Json expand_doc(const Json& doc, const Json& params, const std::string& where, Json* compositions = nullptr) {
  Json d = doc;
  if (d.contains("inputs")) d["inputs"] = expand_entries(d["inputs"], params, where + ": inputs");
  if (d.contains("inputs") && d["inputs"].is_object())                    // `chain:` and `make:` are composed by `expand` (on the parsed definition): taken out of the flat text, with their resolved values
    for (auto it = d["inputs"].begin(); it != d["inputs"].end(); ++it) {
      Json& v = it.value();
      if (!v.is_object()) continue;
      for (const char* key : {"chain", "make", "chain_parameters"})
        if (v.contains(key)) {
          if (compositions) { if (!compositions->contains(it.key())) (*compositions)[it.key()] = Json::object(); (*compositions)[it.key()][key] = with_params(with_params(v[key], params), params); }   // twice: a preset's `make` / `chain_parameters` carry `$name`s of their own
          v.erase(key);
        }
    }
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
  if (needs_expansion(doc)) {              // written out with the parameters' defaults to check it (`for:` repeated, `when:` decided; `chain:` and `make:` composed only at a run); `expand` writes it out again with a run's own
    Json defaults = doc.contains("parameters") && doc["parameters"].is_object() ? doc["parameters"] : Json::object();
    // `one_of: [a, b]`: exactly one of the parameters is given (checked at a run, `expand`); the defaults, which give none, are written out as if the first were given
    if (doc.contains("one_of") && doc["one_of"].is_array() && !doc["one_of"].empty() && doc["one_of"][0].is_string()) {
      bool any = false;
      for (const auto& n : doc["one_of"]) any = any || (n.is_string() && defaults.contains(n.get<std::string>()) && truthy(defaults[n.get<std::string>()]));
      if (!any) defaults[doc["one_of"][0].get<std::string>()] = "(one_of)";
    }
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
  if (d.scope != "subject" && d.scope != "group") err(where, "scope '" + d.scope + "': subject (one subject's rows) or group (inputs marked per_subject are found in each member subject; the outputs are the group's default subject's)");
  {
    const Json kind = doc.value("kind", Json("pipeline"));
    if (!kind.is_string() || (kind != "pipeline" && kind != "chain")) err(where, "kind " + kind.dump() + ": pipeline (the default) or chain (a field chain: one kind of pointwise field, composed into other definitions by a `chain:` input)");
    d.kind = kind.get<std::string>();
    if (d.kind == "chain" && d.scope != "subject") err(where, "a chain is of one subject (scope subject)");
  }
  if (doc.contains("draw")) {
    if (!doc["draw"].is_string()) err(where, "draw: the name of a parameter");
    d.draw = doc["draw"].get<std::string>();
  }
  if (doc.contains("plans")) {                                             // one plan per combination of the lists' elements: {parameter: $list}
    const Json pl = doc["plans"];
    const Json declared = doc.contains("parameters") && doc["parameters"].is_object() ? doc["parameters"] : Json::object();
    if (!pl.is_object() || pl.empty()) err(where, "plans: a mapping of parameter → $list parameter, e.g. {band: $bands, spatial_scale: $spatial_scales}");
    for (const auto& [var, src] : pl.items()) {
      if (!declared.contains(var)) err(where, "plans: '" + var + "' is not a parameter (declared: " + (declared.empty() ? std::string("none") : join(keys(declared))) + ")");
      if (!src.is_string() || src.get<std::string>().size() < 2 || src.get<std::string>()[0] != '$' || !declared.contains(src.get<std::string>().substr(1)))
        err(where, "plans." + var + ": " + src.dump() + " must name a declared list parameter as $name");
      if (src.get<std::string>().substr(1) == var) err(where, "plans." + var + ": the list parameter is another one than the element's (e.g. band and bands)");
    }
    d.plans = pl;
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
    const bool manifold_class = !spec.contains("field") && !spec.contains("operator") && !spec.contains("selection") && !spec.contains("statistic") && !spec.contains("measurement");
    for (const char* rel : {"from", "to", "manifold", "is"})
      if (spec.contains(rel) && !(manifold_class && std::string(rel) == "manifold")) {   // a manifold input's `manifold:` is its class (surface, points), not a relation
        const std::string r = spec[rel].is_string() ? spec[rel].get<std::string>() : spec[rel].dump();
        const std::string base = r.substr(0, r.find('.'));
        if (std::find(defined.begin(), defined.end(), base) == defined.end())
          err(w, std::string(rel) + ": '" + r + "': relations start from an input declared BEFORE this one (" + (defined.empty() ? std::string("none") : join(defined)) + ")");
      }
    for (const char* key : {"of", "over", "within"})                         // a measurement's key names inputs declared before it
      if (spec.contains(key)) {
        if (!spec.contains("measurement")) err(w, std::string(key) + ": belongs to a measurement input (its key: over, of, within)");
        if (!spec[key].is_string() || std::find(defined.begin(), defined.end(), spec[key].get<std::string>()) == defined.end())
          err(w, std::string(key) + ": " + spec[key].dump() + ": names an input declared BEFORE this one (" + (defined.empty() ? std::string("none") : join(defined)) + ")");
      }
    if (spec.contains("per_subject")) {
      if (!spec["per_subject"].is_boolean()) err(w, "per_subject: true or false");
      if (spec["per_subject"].get<bool>()) {
        if (d.scope != "group") err(w, "per_subject belongs to a definition of scope group (its inputs are found in each member subject)");
        d.per_subject.push_back(k);
      }
    }
    if (spec.contains("where")) params_used(spec["where"], w, params);
    // `read: shape` (spec docs/specs/2026-11-17-surrogates.md, section 6): a streamed recording whose SHAPE only the run reads (its channels, rate, span, chunk grid), never a sample
    if (spec.contains("read") && !(spec["read"] == "samples" || (spec["read"] == "shape" && spec.contains("field"))))
      err(w, "read: samples (the default) or shape (a recording's shape only: no sample is read; a field only)");
    defined.push_back(k);
  }
  std::vector<std::string> anchors;
  for (const auto& [k, v] : inputs.items()) if (v.value("each", false)) anchors.push_back(k);
  if (anchors.size() > 1) err(where, "inputs: one input runs the pipeline per match (each: true), not " + join(anchors));
  if (d.scope == "group" && d.per_subject.empty()) err(where, "scope group: at least one input is `per_subject: true` (found in each member subject)");
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
      std::vector<std::string> makes;                                      // what this use makes: every output, or the part its parameter chooses
      {
        const std::vector<std::string> only = st.outputs_for ? st.outputs_for(with) : std::vector<std::string>();
        for (const auto& o : st.outputs) if (!st.outputs_for || std::find(only.begin(), only.end(), o) != only.end()) makes.push_back(o);
      }
      if (outs.size() != makes.size())
        err(w, "out: it makes " + std::to_string(makes.size()) + " thing(s) (" + join(makes) + "), " + std::to_string(outs.size()) + " named");
      for (const auto& o : outs) {
        if (known.count(o)) err(w, "out: '" + o + "' is already defined");
        known.insert(o);
      }
      out.push_back(StepUse{st.name, ins, with, outs, sec, Json()});
    }
  }
  bool writes_recording = false;                                           // a recording output (a filter's) leaves the loop block by block, written as it runs
  if (doc.contains("outputs") && doc["outputs"].is_object())
    for (const auto& [o, spec] : doc["outputs"].items()) { (void)o; if (spec.is_object() && spec.value("kind", std::string()) == "recording") writes_recording = true; }
  if (d.over && !accumulates && !writes_recording && d.kind != "chain") err(where, "steps.chunks: nothing leaves the loop without an accumulate step or a recording output");

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
      if (!declared_kind && kind != "field" && kind != "measurement" && kind != "recording" && std::set<std::string>{"name", "kind_name", "type", "on", "of", "axes", "partition", "operator_kind", "from", "to", "over", "alt", "cs", "derived_from", "factors", "from_selection", "to_selection", "of_statistic", "inverts", "inverse_kind", "acquisition_of", "of_operator", "within", "by_operator", "codes", "carry_origin"}.count(kk))
        err(w, kk + ": belongs to the declared kinds (statistic with axes, operator, selection, manifold, field)");
    if (kind == "operator" && (!spec.contains("operator_kind") || !spec.contains("from") || !spec.contains("to"))) err(w, "an operator output names operator_kind, from and to (links to found manifolds)");
    if (kind == "selection" && !spec.contains("type")) err(w, "a selection output names its type: partition, spans, set, tiling, ladder or grid");
    if (kind == "manifold" && !spec.contains("type")) err(w, "a manifold output names its type: line, surface, product, points or complex");
    if (spec.contains("axes") && !spec["axes"].is_array()) err(w, "axes: a list of {name, on | line}");
  }
  if (d.kind == "chain")                                                   // a chain makes ONE kind of pointwise field: no reduction, no detection, no selection or statistic
    for (const auto& [k, spec] : outputs.items())
      if (std::set<std::string>{"measurement", "spans", "statistic", "labelling", "selection", "operator", "manifold", "recording"}.count(spec.value("kind", std::string())))
        err(where + ": outputs." + k, "a chain's output is a field (a field kind): it makes one kind of pointwise field; reductions (measurements), detections (spans, selections) and statistics belong to the definition that composes it");
  d.outputs = outputs;
  return d;
}

namespace {
// ── COMPOSITION (consolidation spec 4.1 and 4.2): a `chain:` input inlines a field chain's steps; a `make:` input is made by a maker's before-step ──
bool manifold_class_of(const Json& spec) { return !spec.contains("field") && !spec.contains("operator") && !spec.contains("selection") && !spec.contains("statistic") && !spec.contains("measurement"); }
std::string renamed(const std::string& thing, const std::map<std::string, std::string>& to) {
  const std::string base = thing.substr(0, thing.find('.'));
  const auto it = to.find(base);
  return it == to.end() ? thing : it->second + thing.substr(base.size());
}
Json rename_things(const Json& ins, const std::map<std::string, std::string>& to) {
  Json o = Json::object();
  for (const auto& [port, t] : ins.items()) o[port] = t.is_string() ? Json(renamed(t.get<std::string>(), to)) : t;
  return o;
}
void compose_chain(Definition& e, const std::string& input, const Json& chain_name, const Json& chain_params, const std::string& where) {
  if (!chain_name.is_string()) err(where, "chain: the name (or path) of a field chain");
  const std::string w = where + ": inputs." + input + " (chain " + chain_name.get<std::string>() + ")";
  const Definition C = load(chain_name.get<std::string>());
  if (C.kind != "chain") err(w, "'" + C.name + "' is not a chain (its header has no `kind: chain`)");
  Json cp = C.parameters;
  if (!chain_params.is_null()) {
    if (!chain_params.is_object()) err(w, "chain_parameters: a mapping of the chain's parameter → value");
    for (const auto& [k, v] : chain_params.items()) {
      if (!C.parameters.contains(k)) err(w, "chain_parameters: '" + k + "' is not a parameter of " + C.name + " (" + (C.parameters.empty() ? std::string("none") : join(keys(C.parameters))) + ")");
      cp[k] = v;
    }
  }
  for (const auto& [k, v] : cp.items())
    if (v.is_string() && v.get<std::string>() == "required") err(w, "the chain's parameter `" + k + "` is required: give it in chain_parameters");
  const Definition CE = expand(C, cp);
  if (CE.outputs.empty()) err(w, "the chain makes no output");
  const std::string out_name = CE.outputs.begin().key();
  const std::string prefix = input + "_";
  const bool each = e.inputs[input].value("each", false);
  std::map<std::string, std::string> to;                                   // the chain's names → the consumer's
  for (const auto& [k, v] : CE.inputs.items()) { (void)v; to[k] = prefix + k; }
  // the chain's own recording (its `each` input) IS the recording the consumer runs over, when the consumer has one (an `each` input of its own): one input, not two
  std::string consumer_each;
  for (const auto& [k, v] : e.inputs.items()) if (k != input && v.value("each", false)) consumer_each = k;
  std::string chain_each_input;
  if (!each && !consumer_each.empty()) for (const auto& [k, v] : CE.inputs.items()) if (v.value("each", false)) { chain_each_input = k; to[k] = consumer_each; }
  for (const StepUse& u : CE.uses()) for (const auto& o : u.outs) to[o] = o == out_name ? input : prefix + o;
  Json inputs = e.inputs;
  inputs.erase(input);
  bool chain_each = false;
  std::string chain_anchor;                                                  // the chain's own recording (its `each` input): what a measurement of the chain's field is OF
  for (const auto& [k, spec] : CE.inputs.items()) {
    Json sp = spec;
    if (k == chain_each_input) { chain_anchor = consumer_each; chain_each = true; continue; }
    if (sp.value("each", false)) chain_anchor = prefix + k;
    for (const char* rel : {"from", "to", "is"}) if (sp.contains(rel) && sp[rel].is_string()) sp[rel] = renamed(sp[rel].get<std::string>(), to);
    if (sp.contains("manifold") && !manifold_class_of(sp) && sp["manifold"].is_string()) sp["manifold"] = renamed(sp["manifold"].get<std::string>(), to);
    if (sp.contains("where")) {                                              // a `$name` filter whose parameter is null or `none` filters nothing (the finder's rule for a parameter), here before the name is substituted
      Json w = Json::object();
      for (const auto& [wk, wv] : sp["where"].items()) {
        if (wv.is_string() && wv.get<std::string>().size() > 1 && wv.get<std::string>()[0] == '$' && cp.contains(wv.get<std::string>().substr(1))) {
          const Json& pv = cp[wv.get<std::string>().substr(1)];
          if (pv.is_null() || (pv.is_string() && pv.get<std::string>() == "none")) { if (!sp.value("required", true)) w[wk] = nullptr; continue; }   // an optional input named by it is absent (the finder: a null filter of an optional input)
        }
        w[wk] = with_params(wv, cp);
      }
      sp["where"] = w;
    }
    if (sp.value("each", false)) { chain_each = true; if (!each) sp.erase("each"); }
    if (inputs.contains(prefix + k)) err(w, "the chain's input " + prefix + k + " collides with an input of this definition");
    inputs[prefix + k] = sp;
  }
  if (each && !chain_each) err(w, "this input is the one the definition runs once per match (each: true), and the chain has no such input");
  auto convert = [&](const std::vector<StepUse>& from, std::vector<StepUse>& into) {
    std::vector<StepUse> made;
    for (const StepUse& u : from) {
      StepUse n = u;
      n.ins = rename_things(u.ins, to);
      n.with = with_params(u.with, cp);
      n.outs.clear();
      for (const auto& o : u.outs) n.outs.push_back(to.at(o));
      made.push_back(std::move(n));
    }
    into.insert(into.begin(), made.begin(), made.end());
  };
  convert(CE.before, e.before);
  convert(CE.do_, e.do_);
  convert(CE.after, e.after);
  e.inputs = inputs;
  // what the consumer's other inputs say of the composed field (`field.time`, `field.manifold.0`: a time line, the space a maker reads) is of the recording the chain reads and the chain's own output space
  {
    const Json& oo = CE.outputs[out_name];
    const std::string on = oo.is_object() && oo.contains("on") && oo["on"].is_string() ? renamed(oo["on"].get<std::string>(), to) : std::string();
    for (auto& [nm, sp] : e.inputs.items()) {
      (void)nm;
      if (!sp.is_object()) continue;
      for (const char* rel : {"from", "to", "is", "manifold"}) {
        if (!sp.contains(rel) || !sp[rel].is_string()) continue;
        const std::string r = sp[rel].get<std::string>();
        if (r == input + ".time" && !chain_anchor.empty()) sp[rel] = chain_anchor + ".time";
        else if (r == input + ".manifold.0" && !on.empty()) sp[rel] = on;
      }
      if (sp.contains("where") && sp["where"].is_object() && !chain_anchor.empty())            // `{field}` in a name: the recording the chain reads
        for (auto& [wk, wv] : sp["where"].items()) {
          (void)wk;
          if (!wv.is_string()) continue;
          std::string t = wv.get<std::string>();
          const std::string tag = "{" + input + "}";
          for (std::size_t at = 0; (at = t.find(tag, at)) != std::string::npos; at += chain_anchor.size() + 2) t.replace(at, tag.size(), "{" + chain_anchor + "}");
          wv = t;
        }
    }
  }
  // a composed chain stores nothing: a measurement declared of its field is of the recording the chain reads (found, as any measured field)
  if (!chain_anchor.empty())
    for (auto& [name, o] : e.outputs.items()) if (o.is_object() && o.contains("of") && o["of"] == input) o["of"] = chain_anchor;
  if (e.over && *e.over == input) {
    if (!CE.over) err(w, "the definition's chunk loop walks this input, and the chain streams no recording");
    e.over = prefix + *CE.over;
  }
  e.chains.push_back(Json{{"input", input}, {"chain", C.name}, {"version", C.version}, {"parameters", cp}, {"sha256", nxr::io::sha256_hex(C.text)}});
}
// a thing a maker reads (a MANIFOLD input: the line is its own factor `.manifold.0`): an input of the definition, or an output of one of its steps (a chain's field)
bool is_thing(const Definition& e, const std::string& t) {
  if (e.inputs.contains(t)) return true;
  for (const auto* sec : {&e.before, &e.do_, &e.after}) for (const StepUse& u : *sec) if (std::find(u.outs.begin(), u.outs.end(), t) != u.outs.end()) return true;
  return false;
}
// the selection a FACTOR maker stores, and what it lies on (its input's manifold): the run's span on the time line, a set of elements on the space
struct FactorKind { const char* type; const char* port; const char* on; };
FactorKind factor_kind_of(const std::string& node, const std::string& w) {
  if (node == "select.run_span") return {"spans", "line", ".manifold.0"};
  if (node == "select.elements") return {"set", "space", ".manifold.0"};
  if (node == "select.time_grid") return {"tiling", "line", ".manifold.0"};
  err(w, "make.factors: '" + node + "' is not a factor maker (select.run_span, select.elements, select.time_grid), or give {input: NAME} for a selection the definition finds");
}
// A PRODUCT made in the run (a product's factors are always Selections, made by makers and named as factors, never virtual): `make: {node: select.product, factors: [...]}`. A factor is
// a maker, `{node: select.run_span | select.elements | select.time_grid, in: {line | space: THING}, with: {...}, name: "<template>"}`, stored get-or-create under `name`, or `{input: NAME}`, a selection the
// definition finds. The product's manifold and the product selection are outputs of the definition: factors, manifold, product.
void compose_product(Definition& e, const std::string& input, const Json& make, const std::string& w) {
  if (make["node"] != "select.product") err(w, "make.factors: only select.product takes factors");
  for (const auto& [k, v] : make.items()) { (void)v; if (k != "node" && k != "factors" && k != "with" && k != "layout" && k != "name") err(w, "make: unknown key '" + k + "' with factors (node, factors, with, layout, name)"); }
  const Json& fs = make["factors"];
  if (!fs.is_array() || fs.size() < 2 || fs.size() > 4) err(w, "make.factors: two to four factors");
  static const char* ports[] = {"first", "second", "third", "fourth"};
  Json prod_in = Json::object(), manifolds = Json::array(), selections = Json::array();
  e.inputs.erase(input);
  for (std::size_t k = 0; k < fs.size(); ++k) {
    const Json& f = fs[k];
    if (!f.is_object()) err(w, "make.factors: each factor is a mapping");
    if (f.contains("input")) {                                              // a selection the definition finds
      if (!f["input"].is_string() || !e.inputs.contains(f["input"].get<std::string>())) err(w, "make.factors: {input: NAME}: " + f["input"].dump() + " is not an input of this definition");
      const std::string nm = f["input"].get<std::string>();
      prod_in[ports[k]] = nm;
      manifolds.push_back(nm + ".manifold.0");
      selections.push_back(nm);
      continue;
    }
    if (!f.contains("node") || !f["node"].is_string()) err(w, "make.factors: a factor is {node, in, with, name} or {input}");
    const std::string node = f["node"].get<std::string>();
    const FactorKind fk = factor_kind_of(node, w);
    const Json ins = f.contains("in") ? f["in"] : Json::object();
    if (!ins.is_object() || !ins.contains(fk.port) || !ins[fk.port].is_string() || !is_thing(e, ins[fk.port].get<std::string>()))
      err(w, "make.factors: " + node + " reads `" + std::string(fk.port) + "`: an input of this definition (or a step's output)");
    const std::string thing = ins[fk.port].get<std::string>();
    const std::string out = input + "_f" + std::to_string(k);
    e.before.push_back(StepUse{node, ins, f.contains("with") ? f["with"] : Json::object(), {out}, "before", Json()});
    Json o = Json{{"kind", "selection"}, {"type", fk.type}, {"on", thing + fk.on}, {"get_or_create", true}};
    o["name"] = f.contains("name") ? f["name"] : Json(out);
    if (e.outputs.contains(out)) err(w, "an output named '" + out + "' exists already");
    e.outputs[out] = o;
    prod_in[ports[k]] = out;
    manifolds.push_back(thing + fk.on);
    selections.push_back(out);
  }
  const std::string space = input + "_space";
  Json with = make.contains("with") ? make["with"] : Json::object();
  if (make.contains("layout")) with["layout"] = make["layout"];
  e.before.push_back(StepUse{"select.product", prod_in, with, {space, input}, "before", Json()});
  if (e.outputs.contains(space) || e.outputs.contains(input)) err(w, "an output named '" + input + "' or '" + space + "' exists already");
  e.outputs[space] = Json{{"kind", "manifold"}, {"type", "product"}, {"factors", manifolds}};
  Json o = Json{{"kind", "selection"}, {"type", "product"}, {"on", space}, {"factors", selections}, {"get_or_create", true}};
  o["name"] = make.contains("name") ? make["name"] : Json(input);
  e.outputs[input] = o;
  e.made.push_back(Json{{"input", input}, {"node", "select.product"}, {"factors", fs}, {"stored", true}});
}
void compose_make(Definition& e, const std::string& input, const Json& make, const std::string& where) {
  const std::string w = where + ": inputs." + input + " (make)";
  if (!make.is_object() || !make.contains("node") || !make["node"].is_string()) err(w, "make: {node: select.*, in: {port: thing}, with: {parameter: value}, store: {type, on, name}}");
  if (make.contains("factors")) { compose_product(e, input, make, w); return; }
  for (const auto& [k, v] : make.items()) { (void)v; if (k != "node" && k != "in" && k != "with" && k != "store") err(w, "make: unknown key '" + k + "' (node, in, with, store, or factors for select.product)"); }
  const StepSpec& st = step(make["node"].get<std::string>());
  const Json ins = make.contains("in") ? make["in"] : Json::object(), with = make.contains("with") ? make["with"] : Json::object();
  if (!ins.is_object() || !with.is_object()) err(w, "make: in and with are mappings");
  for (const auto& [port, thing] : ins.items()) {
    if (std::find(st.inputs.begin(), st.inputs.end(), port) == st.inputs.end()) err(w, "make.in: '" + port + "' is not one of " + st.name + "'s ports (" + join(st.inputs) + ")");
    if (!thing.is_string() || !is_thing(e, thing.get<std::string>())) err(w, "make.in." + port + ": " + thing.dump() + " is not an input of this definition (a maker reads manifolds and stored selections the definition names)");
  }
  for (const auto& p : st.inputs)
    if (!ins.contains(p) && std::find(st.optional.begin(), st.optional.end(), p) == st.optional.end()) err(w, "make.in: its port " + p + " is not given");
  for (const auto& [p, v] : with.items()) { (void)v; if (!st.parameters.contains(p)) err(w, "make.with: '" + p + "' is not one of " + st.name + "'s parameters"); }
  if (st.outputs.size() != 1) err(w, "a maker makes one selection (" + st.name + " makes " + std::to_string(st.outputs.size()) + ")");
  e.inputs.erase(input);
  e.before.push_back(StepUse{st.name, ins, with, {input}, "before", Json()});
  Json stored = Json();
  if (make.contains("store")) {                                            // the maker's selection stored once: found by its maker key, never refused (get-or-create)
    stored = make["store"];
    if (!stored.is_object() || !stored.contains("type") || !stored.contains("on")) err(w, "make.store: {type: <selection type>, on: <link>, name: <template>}");
    Json o = Json{{"kind", "selection"}, {"type", stored["type"]}, {"on", stored["on"]}, {"get_or_create", true}};
    if (stored.contains("name")) o["name"] = stored["name"];
    if (e.outputs.contains(input)) err(w, "an output named '" + input + "' exists already; the stored selection takes its name");
    e.outputs[input] = o;
  }
  e.made.push_back(Json{{"input", input}, {"node", st.name}, {"in", ins}, {"with", with}, {"stored", !stored.is_null()}});
}
void compose(Definition& e, const Json& comps, const Json& params, const std::string& where) {
  for (const auto& [input, c] : comps.items()) {
    if (!e.inputs.contains(input)) err(where, "inputs." + input + ": composed, but not an input");
    const Json spec = e.inputs[input];
    const Json w = spec.contains("where") ? with_params(spec["where"], params) : Json::object();
    const bool named = w.is_object() && w.contains("name") && truthy(w["name"]);
    const char* what = c.contains("chain") ? "chain" : "make";
    const bool composed = truthy(c[what]);
    if (named && composed) err(where + ": inputs." + input, std::string("give exactly one of the stored name and the ") + what + ", not both");
    if (!named && !composed) err(where + ": inputs." + input, std::string("give exactly one of the stored name (where.name) and the ") + what + "; neither was given");
    if (!composed) continue;
    if (c.contains("chain")) compose_chain(e, input, c["chain"], c.contains("chain_parameters") ? c["chain_parameters"] : Json(), where);
    else compose_make(e, input, c["make"], where);
  }
}
}  // namespace

bool when_holds(const Json& when, const Json& params) { return holds(when, params, "when"); }

std::string quantity_for_index_parameter(const std::string& name) {
  const auto it = kIndexNames.find(name);
  return it == kIndexNames.end() ? std::string() : it->second;
}

Definition expand(const Definition& d, const Json& params) {
  if (d.templ.is_null()) return d;
  const std::string where = d.source.empty() ? "pipeline" : d.source;
  if (d.templ.contains("one_of") && d.templ["one_of"].is_array()) {
    std::vector<std::string> names, given;
    for (const auto& n : d.templ["one_of"]) if (n.is_string()) { names.push_back(n.get<std::string>()); if (params.contains(names.back()) && truthy(params[names.back()])) given.push_back(names.back()); }
    if (given.size() != 1) err(where, "give exactly one of " + join(names) + (given.empty() ? "; none was given" : "; " + join(given) + " were given"));
  }
  Json comps = Json::object();
  Definition e = parse(expand_doc(d.templ, params, where, &comps), d.source, d.text);
  e.templ = d.templ;
  e.draw = d.draw;
  e.parameter_docs = d.parameter_docs;
  compose(e, comps, params, where);
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

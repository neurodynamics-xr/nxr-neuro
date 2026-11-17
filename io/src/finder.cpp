// nxr-io — the input finder (finder.h): nxr's nxr/pipelines/inputs.py, line for line where the model allows.
#include "nxr/io/finder.h"

#include <regex>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/errors.h"
#include "nxr/io/layout.h"
#include "nxr/io/statistic.h"

namespace nxr::io::finder {

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
const std::regex kRel("^([a-z][a-z0-9_]*)\\.(manifold\\.([0-9]+)|time|from|to)$");

// a `$parameter`'s value; in a string, `{X}` is input X's name (`{recording}_bad_segments`)
Json param(const Json& v, const Json& params, const std::map<std::string, Value>* found) {
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    if (!s.empty() && s[0] == '$' && !(s.size() > 1 && s[1] == '{')) {   // `$name` is a parameter; `${name}…` is a name with a parameter inside
      const std::string k = s.substr(1);
      if (!params.contains(k)) refuse("where: " + s + " is not a parameter");
      if (!(found && params[k].is_string())) return params[k];
      s = params[k].get<std::string>();                    // a parameter that is a name may hold `{X}` too: the name of input X (`bursts: "global_bursts.bursts_kept.{recording}"`)
    }
    for (std::size_t p0; (p0 = s.find("${")) != std::string::npos;) {      // `${parameter}` inside a name: the parameter's value (`{surface} tiles L${depth}`)
      const std::size_t e = s.find('}', p0);
      if (e == std::string::npos) refuse("where: '" + s + "': an unclosed ${");
      const std::string k = s.substr(p0 + 2, e - p0 - 2);
      if (!params.contains(k)) refuse("where: ${" + k + "} is not a parameter");
      const Json& x = params[k];
      s.replace(p0, e - p0 + 1, x.is_string() ? x.get<std::string>() : x.is_number_integer() ? std::to_string(x.get<std::int64_t>()) : x.dump());
    }
    if (found && s.find('{') != std::string::npos) {
      for (const auto& [k, x] : *found) {
        const std::string tag = "{" + k + "}";
        for (std::size_t p; (p = s.find(tag)) != std::string::npos;) s.replace(p, tag.size(), x.row ? text(*x.row, "name") : k);
      }
      if (s.find('{') != std::string::npos) refuse("where: '" + s + "' names an input that was not found before it");
    }
    return s;
  }
  return v;
}
// a row's column, or a path into its params_json (`params.inverse.measure`)
Json column(const Row& r, const std::string& key) {
  if (key.rfind("params.", 0) == 0) {
    Json x = Json::parse(opt_text(r, "params_json").value_or("null"), nullptr, false);
    if (x.is_discarded()) return nullptr;
    std::size_t a = 7;
    while (true) {
      const std::size_t b = key.find('.', a);
      const std::string part = key.substr(a, b == std::string::npos ? std::string::npos : b - a);
      x = x.is_object() && x.contains(part) ? x[part] : Json(nullptr);
      if (b == std::string::npos) break;
      a = b + 1;
    }
    return x;
  }
  if (!r.count(key)) refuse("where: '" + key + "' is not a column of the row (or params.<path>)");
  return row_json(r)[key];
}
std::string class_of(const Json& spec) {
  for (const char* c : {"field", "operator", "selection", "statistic", "manifold"}) if (spec.contains(c)) return c;
  refuse("an input names one class: field · operator · selection · statistic · manifold");
}
std::vector<Row> matches(const Database& db, const std::string& subject_id, const std::string& name, const Json& spec, const std::map<std::string, Value>& found,
                         const Json& params) {
  const std::string w = "inputs." + name, cls = class_of(spec);
  const std::string table = cls == "field" ? "field" : cls == "operator" ? "operator" : cls == "statistic" ? "statistic" : cls == "manifold" ? "manifold" : "selection", col = (cls == "selection" || cls == "manifold") ? "type" : cls == "statistic" ? "method" : "kind";
  if (cls == "statistic") {   // a stored statistic read BY NAME (a global statistic is a stored node: the threshold's distribution); its `method` is the class's value (histogram, welch, …). It lies on no manifold.
    if (spec.contains("from") || spec.contains("to") || spec.contains("manifold")) refuse(w + ": a statistic input has no from / to / manifold (find it by `where: {name: …}`)");
    if (!has_table(db, "statistic")) return {};   // a dataset made before the statistic table has none
  }
  const Json want = param(spec[cls], params, nullptr);
  if (!want.is_string()) refuse(w + ": its " + col + " is a name");
  // `any`: no filter on the class's value (a statistic found by its measure and name whatever its method: a covariance computed or imported)
  std::vector<Row> out = want.get<std::string>() == "any"
                             ? db.all("SELECT * FROM \"" + table + "\" WHERE subject_id = ? AND status = 'complete' ORDER BY name", {V(subject_id)})
                             : db.all("SELECT * FROM \"" + table + "\" WHERE subject_id = ? AND status = 'complete' AND " + col + " = ? ORDER BY name", {V(subject_id), V(want.get<std::string>())});
  auto keep = [&](auto pred) { std::vector<Row> k; for (auto& r : out) if (pred(r)) k.push_back(std::move(r)); out = std::move(k); };
  if (spec.contains("from")) { const std::string m = relation(spec["from"].get<std::string>(), found, w); keep([&](const Row& r) { return text(r, "from_manifold_id") == m; }); }
  if (spec.contains("to")) { const std::string m = relation(spec["to"].get<std::string>(), found, w); keep([&](const Row& r) { return opt_text(r, "to_manifold_id") == m; }); }
  if (spec.contains("is")) { const std::string m = relation(spec["is"].get<std::string>(), found, w); keep([&](const Row& r) { return text(r, "id") == m; }); }   // a manifold: the one a relation names (kernel.to)
  if (cls != "manifold" && spec.contains("manifold")) {
    const std::string m = relation(spec["manifold"].get<std::string>(), found, w);
    keep([&](const Row& r) {
      if (is_null(r, "manifold_id")) return false;
      if (cls == "selection") return text(r, "manifold_id") == m;
      const auto f = factors(db, text(r, "manifold_id"));
      return std::find(f.begin(), f.end(), m) != f.end();
    });
  }
  if (spec.contains("where"))
    for (const auto& [k, v] : spec["where"].items()) {
      if (v.is_string() && v.get<std::string>().rfind('$', 0) == 0) {   // a `$parameter` that is `none` (or null) is no filter: the entry is left out
        const std::string pk = v.get<std::string>().substr(1);
        if (params.contains(pk) && (params[pk].is_null() || (params[pk].is_string() && params[pk].get<std::string>() == "none"))) continue;
      }
      const Json wv = param(v, params, &found);
      keep([&](const Row& r) { return column(r, k) == wv; });
    }
  return out;
}
}  // namespace

std::vector<std::string> factors(const Database& db, const std::string& manifold_id) {
  const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {V(manifold_id)});
  if (!m) refuse("manifold " + manifold_id + ": not among the subject's rows");
  if (text(*m, "type") != "product") return {manifold_id};
  std::vector<std::string> out;
  for (const Row& d : db.all("SELECT manifold_id FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {V(text(*m, "geometry_id"))})) out.push_back(text(d, "manifold_id"));
  return out;
}

Value value_of(const Database& db, const std::string& cls, const Row& row) {
  Value v;
  v.cls = cls;
  v.row = row;
  if (cls == "field") {
    v.manifolds = factors(db, text(row, "manifold_id"));
    v.value_type = text(row, "value_type");
    v.complex = v.value_type == "complex";
    if (v.complex) v.value_type = "scalar";
    v.unit = opt_text(row, "unit");
    if (v.unit && v.unit->empty()) v.unit.reset();
    for (const auto& f : v.manifolds) {
      const auto m = db.one("SELECT type FROM manifold WHERE id = ?", {V(f)});
      if (!m || text(*m, "type") != "line" || line_role(db, f) != "time") continue;
      const auto d = db.one("SELECT d.spacing AS spacing, d.origin AS origin FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? AND d.ordinal = 0", {V(f)});
      if (!d || is_null(*d, "spacing")) continue;
      v.time = f;
      v.rate = 1.0 / row_json(*d)["spacing"].get<double>();
      v.origin = is_null(*d, "origin") ? 0.0 : row_json(*d)["origin"].get<double>();
      break;
    }
  } else if (cls == "operator") {
    v.manifolds = {text(row, "from_manifold_id"), opt_text(row, "to_manifold_id").value_or("")};
    // a TRANSITION operator (T5): the two coordinate systems of one space it maps between, as its type carries them (the checker names `positions.transition`)
    if (opt_text(row, "kind") == std::optional<std::string>("transition"))
      for (const char* end : {"from", "to"}) {
        const auto cid = opt_text(row, std::string(end) + "_coordinate_system_id");
        if (!cid) continue;
        if (const auto c = db.one("SELECT cs.name AS name, cs.unit AS unit, sp.name AS space FROM coordinate_system cs JOIN space sp ON sp.id = cs.space_id WHERE cs.id = ?", {V(*cid)}))
          v.facts["transition"][end] = Json{{"space", opt_text(*c, "space").value_or("")}, {"name", text(*c, "name")}, {"unit", opt_text(*c, "unit").value_or("")}};
      }
    v.value_type = opt_text(row, "to_value_type").value_or("scalar");
    v.complex = v.value_type == "complex";
    if (v.complex) v.value_type = "scalar";
  } else if (cls == "statistic") {
    v.unit = opt_text(row, "unit");
  } else if (cls == "manifold") {
    v.manifolds = {text(row, "id")};
  } else {
    v.manifolds = {opt_text(row, "manifold_id").value_or("")};
  }
  // the types' facts of every manifold it lies on (T3): read here, so nothing outside io reads a manifold row
  for (const auto& id : v.manifolds) {
    if (id.empty() || v.manifold_facts.contains(id)) continue;
    const auto m = db.one("SELECT m.name AS name, m.type AS type, m.n_vertices AS n, sp.name AS space, cs.name AS cs_name, cs.unit AS cs_unit FROM manifold m "
                          "LEFT JOIN geometry g ON g.id = m.geometry_id LEFT JOIN coordinate_system cs ON cs.id = g.coordinate_system_id LEFT JOIN space sp ON sp.id = cs.space_id WHERE m.id = ?", {V(id)});
    if (!m) continue;
    Json f = Json{{"name", text(*m, "name")}, {"type", text(*m, "type")}, {"n", is_null(*m, "n") ? std::int64_t(0) : integer(*m, "n")}};
    if (text(*m, "type") == "line") f["role"] = line_role(db, id);
    if (!is_null(*m, "space") || !is_null(*m, "cs_name")) f["cs"] = Json{{"space", opt_text(*m, "space").value_or("")}, {"name", opt_text(*m, "cs_name").value_or("")}, {"unit", opt_text(*m, "cs_unit").value_or("")}};
    v.manifold_facts[id] = std::move(f);
  }
  return v;
}

std::string relation(const std::string& ref, const std::map<std::string, Value>& found, const std::string& where) {
  std::smatch m;
  if (!std::regex_match(ref, m, kRel)) refuse(where + ": '" + ref + "' is not a relation (X.manifold.i · X.time · X.from · X.to)");
  const std::string base = m[1], what = m[2];
  const auto it = found.find(base);
  if (it == found.end()) refuse(where + ": '" + ref + "': " + base + " was not found (an optional input that is absent?)");
  const Value& v = it->second;
  if (what == "time") {
    if (v.cls != "field" || !v.time) refuse(where + ": '" + ref + "': " + base + " has no time line");
    return *v.time;
  }
  if (what == "from" || what == "to") {
    if (v.cls != "operator") refuse(where + ": '" + ref + "': " + base + " is not an operator");
    return v.manifolds[what == "from" ? 0 : 1];
  }
  const std::size_t i = std::stoul(m[3].str());
  if ((v.cls != "field" && v.cls != "manifold") || i >= v.manifolds.size()) refuse(where + ": '" + ref + "': " + base + " has " + std::to_string(v.manifolds.size()) + " factor(s)");
  return v.manifolds[i];
}

std::vector<std::map<std::string, Value>> resolve(const Database& db, const std::string& subject_id, const Json& inputs, const Json& params) {
  std::optional<std::string> anchor;
  std::vector<std::string> names;
  for (const auto& [k, spec] : inputs.items()) {
    if (spec.value("each", false)) anchor = k;
    else names.push_back(k);
  }
  std::vector<std::map<std::string, Value>> starts;
  if (anchor) {
    const auto hits = matches(db, subject_id, *anchor, inputs[*anchor], {}, params);
    if (hits.empty()) refuse("inputs." + *anchor + ": nothing in the subject matches " + inputs[*anchor].dump());
    for (const Row& r : hits) starts.push_back({{*anchor, value_of(db, class_of(inputs[*anchor]), r)}});
  } else {
    starts.emplace_back();
  }
  // inputs are found in the order their relations need: one that names another (`from: kernel.to`, `{surface}` in a name) after it
  {
    std::set<std::string> all(names.begin(), names.end());
    auto deps = [&](const std::string& n) {
      std::set<std::string> d;
      const Json& spec = inputs[n];
      for (const char* k : {"from", "to", "manifold", "is"})
        if (spec.contains(k) && spec[k].is_string()) { const std::string r = spec[k].get<std::string>(); const auto dot = r.find('.'); if (dot != std::string::npos && all.count(r.substr(0, dot))) d.insert(r.substr(0, dot)); }
      if (spec.contains("where"))
        for (const auto& [k, v] : spec["where"].items()) {
          (void)k;
          if (!v.is_string()) continue;
          const std::string t = v.get<std::string>();
          for (std::size_t a = 0; (a = t.find('{', a)) != std::string::npos; ++a) {
            if (a > 0 && t[a - 1] == '$') continue;
            const auto e = t.find('}', a);
            if (e != std::string::npos && all.count(t.substr(a + 1, e - a - 1))) d.insert(t.substr(a + 1, e - a - 1));
          }
        }
      d.erase(n);
      return d;
    };
    std::vector<std::string> ordered;
    std::set<std::string> placed;
    while (ordered.size() < names.size()) {
      bool moved = false;
      for (const auto& n : names) {
        if (placed.count(n)) continue;
        bool ready = true;
        for (const auto& x : deps(n)) if (!placed.count(x)) ready = false;
        if (ready) { ordered.push_back(n); placed.insert(n); moved = true; }
      }
      if (!moved) refuse("inputs: their relations name each other in a cycle");
    }
    names = std::move(ordered);
  }
  std::vector<std::map<std::string, Value>> out;
  for (auto found : starts) {
    std::string missing;
    for (const auto& name : names) {
      const Json& spec = inputs[name];
      bool named_by_caller = false;                                        // an optional input the caller NAMED (a parameter given): it must exist, never silently absent
      if (!spec.value("required", true) && spec.contains("where")) {      // an optional input named by a parameter left null is not looked for (absent)
        bool unset = false;
        for (const auto& [k, v] : spec["where"].items())
          if (v.is_string() && v.get<std::string>().rfind('$', 0) == 0) {
            const std::string pk = v.get<std::string>().substr(1);
            if (params.contains(pk) && params[pk].is_null()) unset = true;
            else if (params.contains(pk) && params[pk].is_string() && params[pk].get<std::string>() != "none") named_by_caller = true;
          }
        if (unset) continue;
      }
      const auto hits = matches(db, subject_id, name, spec, found, params);
      const std::string label = anchor ? " (for " + *anchor + " '" + text(*found.at(*anchor).row, "name") + "')" : "";
      if (hits.empty()) {
        if (spec.value("required", true) || named_by_caller) {
          // EVERY missing input is named (a run over many pairs names all its missing joint partitions at once), each with its `hint` (what makes it), `{X}` being input X's name
          Json shown = spec;
          shown.erase("hint");
          std::string line = "inputs." + name + label + ": nothing in the subject matches " + shown.dump();
          if (spec.contains("hint") && spec["hint"].is_string()) line += " \xe2\x80\x94 " + param(spec["hint"], params, &found).get<std::string>();
          missing += (missing.empty() ? "" : "\n") + line;
        }
        continue;
      }
      if (hits.size() > 1) {
        std::string list;
        for (const Row& r : hits) list += (list.empty() ? "" : ", ") + text(r, "name");
        refuse("inputs." + name + label + ": " + std::to_string(hits.size()) + " match (" + list + "); add a `where` that names one");
      }
      found[name] = value_of(db, class_of(spec), hits[0]);
    }
    if (!missing.empty()) refuse(missing);
    out.push_back(std::move(found));
  }
  return out;
}

Json describe(const std::map<std::string, Value>& found) {
  Json o = Json::object();
  for (const auto& [k, v] : found) {
    if (!v.row) continue;
    const Row& r = *v.row;
    o[k] = Json{{"class", v.cls}, {"name", text(r, "name")}, {"id", text(r, "id")}, {"kind", r.count("kind") ? text(r, "kind") : r.count("type") ? text(r, "type") : text(r, "method")}};
  }
  return o;
}

}  // namespace nxr::io::finder

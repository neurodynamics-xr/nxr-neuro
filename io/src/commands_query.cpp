// The queries a client asks of a subject (docs/specs/2026-11-20-data-layer-queries.md, sections 2 and 3): `read subject` (every node's DESCRIPTION, no arrays), `list <class>` (the nodes of one
// class, filtered, paged and with only the columns asked for) and the count of each class a dataset holds (`show dataset`). The descriptions are nxr-neuro's typed model, named, nested and
// versioned (`nxr-neuro-subject@1`): a node is named, never numbered; what it refers to is named by its name (a node of another subject as SUBJECT:NAME); no key is a column of the schema
// (`*_id`, `*_json`). The SQL stays inside this file (io): the filters are named options, never SQL from the caller.
#include "nxr/io/task/commands.h"
#include "nxr/io/task/handle.h"
#include "commands_open.h"

#include <algorithm>
#include <map>
#include <set>

#include "nxr/io/catalogue.h"
#include "nxr/io/errors.h"
#include "nxr/io/finder.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/dataset.h"

namespace nxr::io::task {

using namespace commands_detail;

namespace {

const char* const kSubjectSchema = "nxr-neuro-subject@1";
const char* const kListSchema = "nxr-neuro-list@1";

// the classes `read subject` describes (in this order), and `list` also takes `jobs`
const std::vector<std::string> kSubjectClasses = {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions"};
const std::vector<std::string> kListClasses = {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions", "jobs"};

std::string joined(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
  return s;
}

// ---- values as JSON ----
Json J(const Value& v) {
  if (std::holds_alternative<std::int64_t>(v)) return std::get<std::int64_t>(v);
  if (std::holds_alternative<double>(v)) return std::get<double>(v);
  if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
  return Json();
}
Json Jt(const Row& r, const char* c) {
  const auto it = r.find(c);
  return it == r.end() ? Json() : J(it->second);
}
bool flag(const Row& r, const char* c) { return integer(r, c) != 0; }
// a `*_json` column parsed (null when absent or unreadable)
Json Jj(const Row& r, const char* c) {
  const auto it = r.find(c);
  if (it == r.end() || !std::holds_alternative<std::string>(it->second)) return Json();
  Json j = Json::parse(std::get<std::string>(it->second), nullptr, false);
  return j.is_discarded() ? Json() : j;
}

// ---- one subject of one database: its names ----
struct Ctx {
  const Database& db;
  std::string subject_id, subject_name;
  mutable std::map<std::string, std::string> cache;
  const NodeTyper* typer = nullptr;
  Ctx(const Database& d, std::string id, std::string name, const NodeTyper* t = nullptr) : db(d), subject_id(std::move(id)), subject_name(std::move(name)), typer(t) {}
  // the node's TYPE in the type system's JSON, from io's description of the row. Null: no typer (a build without the pipelines). A node the type system refuses to type
  // (e.g. a tangent field with no gauge record) is not dropped and does not fail the read: its type is {"refused": the type system's message}, never a silent null.
  Json typed(const char* cls, const Row& r) const {
    if (!typer || !*typer) return Json();
    try { return (*typer)(cls, finder::value_of(db, cls, r)); } catch (const std::exception& e) { return Json{{"refused", e.what()}}; }
  }

  // the name a row of `table` goes by, as a reference: NAME, or SUBJECT:NAME when it belongs to another subject
  std::string name_of(const char* table, const std::string& id) const {
    const std::string key = std::string(table) + "/" + id;
    const auto hit = cache.find(key);
    if (hit != cache.end()) return hit->second;
    std::string out;
    const auto r = db.one(std::string("SELECT x.name AS name, x.subject_id AS sid, s.name AS sname FROM ") + table + " x LEFT JOIN subject s ON s.id = x.subject_id WHERE x.id = ?", {V(id)});
    if (r) out = (is_null(*r, "sid") || text(*r, "sid") == subject_id) ? text(*r, "name") : text(*r, "sname") + ":" + text(*r, "name");
    cache[key] = out;
    return out;
  }
  Json ref(const char* table, const Row& r, const char* column) const {
    if (is_null(r, column)) return Json();
    const std::string n = name_of(table, text(r, column));
    return n.empty() ? Json() : Json(n);
  }
};

// the node's producer record (how it was made: the command or pipeline, the program, its inputs, and what the maker said with `--producer`), as `read <class>` gives it: a selection
// keeps it in its parameters (it has no producer column), every other class in its own record. Null: none.
Json producer_record(const Row& r) {
  if (r.find("producer_json") != r.end()) return Jj(r, "producer_json");
  const Json p = Jj(r, "params_json");
  return p.is_object() && p.contains("producer") ? p["producer"] : Json();
}
Json provenance(const Row& r) {
  return Json{{"source", Jt(r, "source")}, {"comment", Jt(r, "comment")}, {"created_utc", Jt(r, "created_utc")}, {"created_by", Jt(r, "created_by")},
              {"modified_utc", Jt(r, "modified_utc")}, {"modified_by", Jt(r, "modified_by")}, {"producer", producer_record(r)}};
}
Json acquisition_of(const Ctx& c, const Row& r) {
  if (is_null(r, "acquisition_id")) return Json();
  const auto a = c.db.one("SELECT label FROM acquisition WHERE id = ?", {V(text(r, "acquisition_id"))});
  return a ? Json(text(*a, "label")) : Json();
}

// ---- manifolds ----
Json axes_of(const Ctx& c, const Row& m) {
  Json axes = Json::array();
  if (is_null(m, "geometry_id")) return axes;
  for (const Row& d : c.db.all("SELECT d.ordinal AS ordinal, d.name AS name, d.n_vertices AS n, d.spacing AS spacing, d.origin AS origin, d.unit AS unit, d.manifold_id AS factor "
                               "FROM geometry_dimension d WHERE d.geometry_id = ? ORDER BY d.ordinal", {V(text(m, "geometry_id"))}))
    axes.push_back(Json{{"name", Jt(d, "name")}, {"size", Jt(d, "n")}, {"spacing", Jt(d, "spacing")}, {"origin", Jt(d, "origin")}, {"unit", Jt(d, "unit")}, {"factor", c.ref("manifold", d, "factor")}});
  return axes;
}
Json describe_manifold(const Ctx& c, const Row& m) {
  Json factors = Json::array();
  for (const auto& a : axes_of(c, m)) if (!a["factor"].is_null()) factors.push_back(a["factor"]);
  // the coordinate system its points are in (its name and unit), and whether each point of a line carries a code (`kinds`)
  Json coordinates;
  if (!is_null(m, "geometry_id")) {
    const auto g = c.db.one("SELECT g.form AS form, cs.name AS cs, cs.unit AS unit FROM geometry g LEFT JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE g.id = ?", {V(text(m, "geometry_id"))});
    if (g) {
      const Json producer = Jj(m, "producer_json");
      // a closed-form geometry's coordinate along each of its axes (a line's: one): the first point's coordinate, the step between points and its unit
      Json closed = Json::array();
      for (const auto& a : axes_of(c, m))
        if (a["factor"].is_null() && !a["spacing"].is_null()) closed.push_back(Json{{"name", a["name"]}, {"size", a["size"]}, {"origin", a["origin"]}, {"spacing", a["spacing"]}, {"unit", a["unit"]}});
      coordinates = Json{{"system", Jt(*g, "cs")}, {"unit", Jt(*g, "unit")}, {"kinds", producer.is_object() && producer.contains("kinds") && producer["kinds"].is_string()},
                         {"axes", closed}};
    }
  }
  return Json{{"id", Jt(m, "id")}, {"name", Jt(m, "name")}, {"kind", Jt(m, "type")}, {"type", c.typed("manifold", m)}, {"label", Jt(m, "label")}, {"description", Jt(m, "comment")}, {"primary", flag(m, "is_primary")}, {"acquisition", acquisition_of(c, m)},
              {"coordinates", coordinates},
              {"size", Json{{"vertices", Jt(m, "n_vertices")}, {"edges", Jt(m, "n_edges")}, {"faces", Jt(m, "n_faces")}, {"volumes", Jt(m, "n_volumes")}}},
              {"factors", factors},
              {"component_of", c.ref("manifold", m, "component_of_id")}, {"component_ordinal", Jt(m, "component_ordinal")}, {"from_index", Jt(m, "from_index")},
              {"derived_from", c.ref("manifold", m, "derived_from_id")}, {"derived_by", Jt(m, "operator")}, {"stored", !is_null(m, "path")}, {"provenance", provenance(m)}};
}

// ---- fields ----
Json describe_field(const Ctx& c, const Row& f) {
  Json contributions = Json::array();
  for (const Row& k : c.db.all("SELECT fc.source_field_id AS source, fc.weight AS weight FROM field_contribution fc WHERE fc.field_id = ? ORDER BY fc.ordinal", {V(text(f, "id"))}))
    contributions.push_back(Json{{"field", c.ref("field", k, "source")}, {"weight", Jt(k, "weight")}});
  return Json{{"id", Jt(f, "id")}, {"name", Jt(f, "name")}, {"kind", Jt(f, "kind")}, {"type", c.typed("field", f)}, {"label", Jt(f, "label")}, {"acquisition", acquisition_of(c, f)},
              {"description", Jt(f, "description")}, {"stored", !is_null(f, "path")},
              {"lineage", Json{{"derived_from", c.ref("field", f, "derived_from_id")}, {"of_field", c.ref("field", f, "of_field_id")}, {"of_operator", c.ref("operator", f, "of_operator_id")},
                               {"by_operator", c.ref("operator", f, "by_operator_id")}, {"function", Jt(f, "function")}, {"contributions", contributions}}},
              {"provenance", provenance(f)}};
}

// ---- selections ----
// an entry's colour as #rrggbb (null: none); the label and the description it carries beyond the schema's columns
Json colour_of(const Row& e) {
  const Json c = Jj(e, "color_json");
  if (!c.is_array() || c.size() != 3) return Json();
  static const char* hex = "0123456789abcdef";
  std::string t = "#";
  for (const auto& x : c) { const int v = x.is_number() ? x.get<int>() : 0; t += hex[(v >> 4) & 15]; t += hex[v & 15]; }
  return t;
}
Json attribute_of(const Row& e, const char* key) {
  const Json a = Jj(e, "attributes_json");
  return a.is_object() && a.contains(key) ? a[key] : Json();
}
Json entries_of(const Ctx& c, const Row& s, bool all_columns) {
  Json out = Json::array();
  if (is_null(s, "dictionary_id")) return out;
  for (const Row& e : c.db.all("SELECT code, name, color_json, measurement, measurement_unit, function, params_json, attributes_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(text(s, "dictionary_id"))})) {
    Json x = Json{{"code", Jt(e, "code")}, {"name", Jt(e, "name")}, {"colour", colour_of(e)}, {"label", attribute_of(e, "label")}, {"description", attribute_of(e, "description")}};
    if (all_columns) {
      x["measurement"] = Jt(e, "measurement");
      x["unit"] = Jt(e, "measurement_unit");
      x["function"] = Jt(e, "function");
      x["params"] = Jj(e, "params_json");
    }
    out.push_back(std::move(x));
  }
  return out;
}
Json describe_selection(const Ctx& c, const Row& s) {
  Json dictionary;
  if (!is_null(s, "dictionary_id")) {
    const std::string n = c.name_of("dictionary", text(s, "dictionary_id"));
    if (!n.empty()) dictionary = n;
  }
  return Json{{"id", Jt(s, "id")}, {"name", Jt(s, "name")}, {"kind", Jt(s, "type")}, {"type", c.typed("selection", s)}, {"label", Jt(s, "label")}, {"acquisition", acquisition_of(c, s)},
              {"description", Jt(s, "description")}, {"elements", Jt(s, "n_elements")}, {"members", Jt(s, "n_members")},
              // a member of a set is a selection of its own (an atlas of the set of a surface's parcellations, a level of a grid): the set it belongs to, its place there and its code
              {"member_of", c.ref("selection", s, "member_of_id")}, {"member_ordinal", Jt(s, "member_ordinal")}, {"code", Jt(s, "code")},
              {"of_field", c.ref("field", s, "of_field_id")},
              {"weights", [&] { const Json p = Jj(s, "params_json"); return p.is_object() && p.contains("weights") && p["weights"].is_string(); }()},
              {"stored", !is_null(s, "array_path")}, {"dictionary", dictionary}, {"entries", entries_of(c, s, false)}, {"provenance", provenance(s)}};
}

// ---- operators ----
Json describe_operator(const Ctx& c, const Row& o) {
  auto end = [&](const char* manifold, const char* selection) { return Json{{"manifold", c.ref("manifold", o, manifold)}, {"selection", c.ref("selection", o, selection)}}; };
  return Json{{"id", Jt(o, "id")}, {"name", Jt(o, "name")}, {"kind", Jt(o, "kind")}, {"type", c.typed("operator", o)}, {"label", Jt(o, "label")}, {"acquisition", acquisition_of(c, o)},
              {"description", Jt(o, "description")}, {"from", end("from_manifold_id", "from_selection_id")}, {"to", end("to_manifold_id", "to_selection_id")},
              {"method", Jt(o, "method")}, {"parameters", Jj(o, "params_json")}, {"shape", Json{{"rows", Jt(o, "n_rows")}, {"columns", Jt(o, "n_cols")}, {"nonzeros", Jt(o, "nnz")}}},
              {"stored", !is_null(o, "path")},
              {"lineage", Json{{"derived_from", c.ref("operator", o, "derived_from_id")}, {"of_statistic", c.ref("statistic", o, "of_statistic_id")}, {"inverts", c.ref("operator", o, "inverts_id")},
                               {"inverse_kind", Jt(o, "inverse_kind")}}},
              {"provenance", provenance(o)}};
}

// ---- statistics ----
Json describe_statistic(const Ctx& c, const Row& s) {
  Json axes = Json::array();
  const Json raw = Jj(s, "axes_json");
  if (raw.is_array())
    for (const auto& a : raw) {
      Json x = Json{{"name", a.value("name", Json())}, {"manifold", Json()}, {"selection", Json()}, {"size", a.value("size", Json())}, {"unit", a.value("unit", Json())}};
      if (a.contains("manifold_id") && a["manifold_id"].is_string()) x["manifold"] = c.name_of("manifold", a["manifold_id"].get<std::string>());
      if (a.contains("selection_id") && a["selection_id"].is_string()) x["selection"] = c.name_of("selection", a["selection_id"].get<std::string>());
      axes.push_back(std::move(x));
    }
  return Json{{"id", Jt(s, "id")}, {"name", Jt(s, "name")}, {"measure", Jt(s, "measure")}, {"method", Jt(s, "method")}, {"type", c.typed("statistic", s)}, {"label", Jt(s, "label")}, {"acquisition", acquisition_of(c, s)},
              {"description", Jt(s, "description")},
              {"of_field", c.ref("field", s, "of_field_id")}, {"of_statistic", c.ref("statistic", s, "of_statistic_id")}, {"partition", c.ref("selection", s, "partition_selection_id")},
              {"members", Jt(s, "n_members")}, {"members_skipped", Jt(s, "members_skipped")}, {"dof", Jt(s, "dof")}, {"axes", axes}, {"stored", !is_null(s, "path")},
              {"provenance", Json{{"created_utc", Jt(s, "created_utc")}, {"created_by", Jt(s, "created_by")}, {"modified_utc", Jt(s, "modified_utc")}, {"modified_by", Jt(s, "modified_by")},
                                  {"producer", producer_record(s)}}}};
}

// ---- dictionaries, spaces, acquisitions, jobs ----
Json describe_dictionary(const Ctx& c, const Row& d) {
  Json entries = Json::array();
  for (const Row& e : c.db.all("SELECT code, name, color_json, measurement, measurement_unit, function, params_json, attributes_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(text(d, "id"))}))
    entries.push_back(Json{{"code", Jt(e, "code")}, {"name", Jt(e, "name")}, {"colour", colour_of(e)}, {"label", attribute_of(e, "label")}, {"description", attribute_of(e, "description")},
                           {"measurement", Jt(e, "measurement")}, {"unit", Jt(e, "measurement_unit")}, {"function", Jt(e, "function")}, {"params", Jj(e, "params_json")}});
  return Json{{"id", Jt(d, "id")}, {"name", Jt(d, "name")}, {"scope", Jt(d, "scope")}, {"description", Jt(d, "description")}, {"entries", entries},
              {"provenance", Json{{"created_utc", Jt(d, "created_utc")}, {"created_by", Jt(d, "created_by")}, {"modified_utc", Jt(d, "modified_utc")}, {"modified_by", Jt(d, "modified_by")}}}};
}
Json describe_space(const Ctx& c, const Row& s) {
  Json systems = Json::array();
  for (const Row& cs : c.db.all("SELECT name, unit, is_reference, transition, description FROM coordinate_system WHERE space_id = ? ORDER BY is_reference DESC, name", {V(text(s, "id"))}))
    systems.push_back(Json{{"name", Jt(cs, "name")}, {"unit", Jt(cs, "unit")}, {"reference", flag(cs, "is_reference")}, {"transition", Jt(cs, "transition")}, {"description", Jt(cs, "description")}});
  return Json{{"id", Jt(s, "id")}, {"name", Jt(s, "name")}, {"geometry", Jt(s, "geometry")}, {"rank", Jt(s, "rank")}, {"parameters", Jj(s, "parameters_json")}, {"description", Jt(s, "description")},
              {"coordinate_systems", systems}};
}
Json describe_acquisition(const Ctx&, const Row& a) {
  return Json{{"id", Jt(a, "id")}, {"name", Jt(a, "label")}, {"session", Jt(a, "ses")}, {"task", Jt(a, "task")}, {"run", Jt(a, "run")}, {"acq", Jt(a, "acq")}, {"processing", Jt(a, "proc")}, {"split", Jt(a, "split")}};
}
Json describe_job(const Ctx&, const Row& j) {
  return Json{{"id", Jt(j, "id")}, {"kind", Jt(j, "kind")}, {"host", Jt(j, "host")}, {"status", Jt(j, "status")}, {"subject", Jt(j, "subject_name")}, {"inputs", Jj(j, "inputs_json")}, {"parameters", Jj(j, "params_json")},
              {"result", Jj(j, "result_json")}, {"progress", Jt(j, "progress")}, {"error", Jt(j, "error")}, {"requested_by", Jt(j, "requested_by")}, {"created_utc", Jt(j, "created_utc")},
              {"started_utc", Jt(j, "started_utc")}, {"finished_utc", Jt(j, "finished_utc")}};
}

// ---- how a class is selected ----
struct Filter {
  std::optional<std::string> name, kind, on;   // on: the id of the manifold
  std::optional<std::vector<std::string>> after;   // the cursor's [sort key, id] within the subject
  std::int64_t limit = 0;                       // 0: all
};

struct ClassDef {
  const char* noun;
  const char* table;
  const char* name_column;                      // the column the rows are ordered and `--name` filtered by
  const char* select;                           // SELECT ... FROM ... (the rows' columns)
  const char* where;                            // the class's own condition, `?` bound to the subject id (count given by `subject_binds`)
  int subject_binds;
  const char* kind_column;                      // null: the class has no kind
  Json (*describe)(const Ctx&, const Row&);
};

const char* const kManifoldSelect =
    "SELECT x.*, t.kind AS topology_kind, t.rank AS topology_rank, t.n_components AS topology_components, t.closed AS topology_closed, t.orientable AS topology_orientable, "
    "cs.name AS cs_name, cs.unit AS cs_unit, sp.name AS space_name FROM manifold x JOIN topology t ON t.id = x.topology_id LEFT JOIN geometry g ON g.id = x.geometry_id "
    "LEFT JOIN coordinate_system cs ON cs.id = g.coordinate_system_id LEFT JOIN space sp ON sp.id = cs.space_id";

const ClassDef kDefs[] = {
    {"manifolds", "manifold", "x.name", kManifoldSelect, "x.subject_id = ? AND x.status = 'complete'", 1, "x.type", describe_manifold},
    {"fields", "field", "x.name", "SELECT x.* FROM field x", "x.subject_id = ? AND x.status = 'complete'", 1, "x.kind", describe_field},
    {"selections", "selection", "x.name", "SELECT x.* FROM selection x", "x.subject_id = ? AND x.status = 'complete'", 1, "x.type", describe_selection},
    {"operators", "operator", "x.name", "SELECT x.* FROM operator x", "x.subject_id = ? AND x.status = 'complete'", 1, "x.kind", describe_operator},
    {"statistics", "statistic", "x.name", "SELECT x.* FROM statistic x", "x.subject_id = ? AND x.status = 'complete'", 1, "x.measure", describe_statistic},
    {"dictionaries", "dictionary", "x.name", "SELECT x.* FROM dictionary x", "(x.subject_id = ? OR x.id IN (SELECT dictionary_id FROM selection WHERE subject_id = ? AND dictionary_id IS NOT NULL) OR (x.scope = 'dataset' AND x.dataset_id IN (SELECT dataset_id FROM subject WHERE id = ?)))", 3, "x.scope", describe_dictionary},
    {"spaces", "space", "x.name", "SELECT x.* FROM space x", "x.subject_id = ?", 1, "x.geometry", describe_space},
    {"acquisitions", "acquisition", "x.label", "SELECT x.* FROM acquisition x", "x.subject_id = ?", 1, nullptr, describe_acquisition},
    {"jobs", "job", "x.created_utc", "SELECT x.*, s.name AS subject_name FROM job x LEFT JOIN subject s ON s.id = x.subject_id", "x.subject_id = ?", 1, "x.kind", describe_job},
};

const ClassDef& def_of(const std::string& noun) {
  for (const auto& d : kDefs) if (noun == d.noun) return d;
  throw Refused("'" + noun + "' is not a class of nodes (" + joined(kListClasses) + ")");
}

std::string like_escape(const std::string& s) {
  std::string out;
  for (const char ch : s) {
    if (ch == '%' || ch == '_' || ch == '\\') out += '\\';
    out += ch;
  }
  return out;
}

// the rows of one class of one subject, in order, after the cursor, at most `limit` + 1 (the extra one says there is more)
std::vector<Row> select_rows(const Ctx& c, const ClassDef& d, const Filter& f, std::int64_t want, bool subject_scoped) {
  const std::string table = d.table;
  if ((table == "statistic" || table == "job") && !has_table(c.db, table)) return {};
  std::string sql = std::string(d.select) + " WHERE ";
  std::vector<Value> args;
  if (subject_scoped) {
    sql += d.where;
    for (int i = 0; i < d.subject_binds; ++i) args.push_back(V(c.subject_id));
  } else {
    sql += "1 = 1";   // a one-database dataset's jobs, whoever they belong to
  }
  if (f.name) {
    sql += std::string(" AND ") + d.name_column + " LIKE ? ESCAPE '\\'";
    args.push_back(V("%" + like_escape(*f.name) + "%"));
  }
  if (f.kind) {
    sql += std::string(" AND ") + d.kind_column + " = ?";
    args.push_back(V(*f.kind));
  }
  if (f.on) {
    const Value m = V(*f.on);
    if (table == "manifold") {
      sql += " AND (x.component_of_id = ? OR x.derived_from_id = ? OR EXISTS (SELECT 1 FROM geometry_dimension dd WHERE dd.geometry_id = x.geometry_id AND dd.manifold_id = ?))";
      args.insert(args.end(), {m, m, m});
    } else if (table == "field") {
      sql += " AND (x.manifold_id = ? OR EXISTS (SELECT 1 FROM manifold pm JOIN geometry_dimension dd ON dd.geometry_id = pm.geometry_id WHERE pm.id = x.manifold_id AND dd.manifold_id = ?))";
      args.insert(args.end(), {m, m});
    } else if (table == "selection") {
      sql += " AND x.manifold_id = ?";
      args.push_back(m);
    } else if (table == "operator") {
      sql += " AND (x.from_manifold_id = ? OR x.to_manifold_id = ?)";
      args.insert(args.end(), {m, m});
    } else {   // statistic: any of its axes
      sql += " AND EXISTS (SELECT 1 FROM json_each(x.axes_json) a WHERE json_extract(a.value, '$.manifold_id') = ?)";
      args.push_back(m);
    }
  }
  if (f.after) {
    sql += std::string(" AND (") + d.name_column + " > ? OR (" + d.name_column + " = ? AND x.id > ?))";
    args.insert(args.end(), {V((*f.after)[0]), V((*f.after)[0]), V((*f.after)[1])});
  }
  sql += std::string(" ORDER BY ") + d.name_column + ", x.id";
  if (want > 0) {
    sql += " LIMIT ?";
    args.push_back(V(want));
  }
  return c.db.all(sql, args);
}

// ---- the subjects of a database ----
struct SubjectRef { std::string id, name, kind; std::optional<std::string> template_id; };
std::optional<SubjectRef> find_subject_ref(const Database& db, const std::string& key) {
  const auto r = db.one("SELECT id, name, kind, template_id FROM subject WHERE (name = ? OR id = ?) AND status = 'complete' ORDER BY (name = ?) DESC LIMIT 1", {V(key), V(key), V(key)});
  if (!r) return std::nullopt;
  return SubjectRef{text(*r, "id"), text(*r, "name"), text(*r, "kind"), opt_text(*r, "template_id")};
}
std::vector<SubjectRef> all_subject_refs(const Database& db) {
  std::vector<SubjectRef> out;
  for (const Row& r : db.all("SELECT id, name, kind, template_id FROM subject WHERE status = 'complete' ORDER BY name, id"))
    out.push_back(SubjectRef{text(r, "id"), text(r, "name"), text(r, "kind"), opt_text(r, "template_id")});
  return out;
}

std::int64_t count_of(const Ctx& c, const ClassDef& d, bool subject_scoped) {
  return static_cast<std::int64_t>(select_rows(c, d, Filter{}, 0, subject_scoped).size());
}

// ---- the cursor: opaque to the caller (the hex of its content) ----
std::string hex_of(const std::string& s) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (const unsigned char ch : s) {
    out += digits[ch >> 4];
    out += digits[ch & 15];
  }
  return out;
}
std::optional<std::string> unhex(const std::string& s) {
  if (s.size() % 2) return std::nullopt;
  std::string out;
  for (std::size_t i = 0; i < s.size(); i += 2) {
    auto nib = [](char ch) -> int { return ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : -1; };
    const int a = nib(s[i]), b = nib(s[i + 1]);
    if (a < 0 || b < 0) return std::nullopt;
    out += static_cast<char>(a * 16 + b);
  }
  return out;
}

std::string name_key(const ClassDef& d, const Row& r) { return std::string(d.name_column) == "x.label" ? text(r, "label") : std::string(d.name_column) == "x.created_utc" ? text(r, "created_utc") : text(r, "name"); }

}  // namespace

// ============================================================================ read subject
CommandResult read_subject(const ReadSubjectRequest& rq, const Progress& progress) {
  std::vector<std::string> classes = kSubjectClasses;
  if (rq.classes) {
    classes.clear();
    for (const auto& k : *rq.classes) {
      if (std::find(kSubjectClasses.begin(), kSubjectClasses.end(), k) == kSubjectClasses.end()) throw Refused("--classes: '" + k + "' is not a class (" + joined(kSubjectClasses) + ")");
      if (std::find(classes.begin(), classes.end(), k) == classes.end()) classes.push_back(k);
    }
    std::vector<std::string> ordered;   // in the model's order, whatever order they were asked in
    for (const auto& k : kSubjectClasses) if (std::find(classes.begin(), classes.end(), k) != classes.end()) ordered.push_back(k);
    classes = ordered;
  }
  Lease lease = open_reader(rq.dataset, rq.subject, rq.templates, progress);
  const Database& db = (*lease).db();
  const auto subject = find_subject_ref(db, rq.subject);
  if (!subject) throw Error(code::no_such_row, "dataset " + rq.dataset + " holds no subject named '" + rq.subject + "'");

  auto describe = [&](const SubjectRef& s) {
    const Ctx c(db, s.id, s.name, &rq.typer);
    Json out = Json{{"schema", kSubjectSchema}, {"subject", Json{{"id", s.id}, {"name", s.name}, {"kind", s.kind}, {"template", Json()}}}};
    if (s.template_id) {
      if (const auto t = db.one("SELECT name FROM subject WHERE id = ?", {V(*s.template_id)})) out["subject"]["template"] = text(*t, "name");
    }
    for (const std::string& k : classes) {
      const ClassDef& d = def_of(k);
      Json rows = Json::array();
      for (const Row& r : select_rows(c, d, Filter{}, 0, true)) rows.push_back(d.describe(c, r));
      out[k] = std::move(rows);
    }
    return out;
  };

  CommandResult r;
  r.result = describe(*subject);
  if (rq.related) {
    // the subjects this one refers to: the template its anatomy is taken from, and the owners of the manifolds its operators map from and to and its fields lie on (a registration's source and target)
    std::vector<std::pair<std::string, std::string>> found;   // (subject id, relation)
    auto add = [&](const std::string& id, const std::string& relation) {
      if (id == subject->id) return;
      for (const auto& f : found) if (f.first == id && f.second == relation) return;
      found.push_back({id, relation});
    };
    if (subject->template_id) add(*subject->template_id, "template");
    for (const Row& x : db.all("SELECT DISTINCT m.subject_id AS sid FROM operator o JOIN manifold m ON m.id = o.from_manifold_id WHERE o.subject_id = ? AND m.subject_id <> ?", {V(subject->id), V(subject->id)})) add(text(x, "sid"), "source");
    for (const Row& x : db.all("SELECT DISTINCT m.subject_id AS sid FROM operator o JOIN manifold m ON m.id = o.to_manifold_id WHERE o.subject_id = ? AND m.subject_id <> ?", {V(subject->id), V(subject->id)})) add(text(x, "sid"), "target");
    for (const Row& x : db.all("SELECT DISTINCT m.subject_id AS sid FROM field f JOIN manifold m ON m.id = f.manifold_id WHERE f.subject_id = ? AND m.subject_id <> ?", {V(subject->id), V(subject->id)})) add(text(x, "sid"), "domain");
    Json related = Json::array();
    for (const auto& [id, relation] : found) {
      const auto row = db.one("SELECT id, name, kind, template_id FROM subject WHERE id = ?", {V(id)});
      if (!row) continue;
      const SubjectRef s{text(*row, "id"), text(*row, "name"), text(*row, "kind"), opt_text(*row, "template_id")};
      // what the database holds of that subject: the whole description in a one-database dataset; a package holds only the anchor rows of its template
      related.push_back(Json{{"subject", s.name}, {"kind", s.kind}, {"relation", relation}, {"description", describe(s)}});
    }
    r.result["related"] = std::move(related);
  }
  return r;
}

// ============================================================================ list <class>
CommandResult list_class(const ListClassRequest& rq, const Progress& progress) {
  const ClassDef& d = def_of(rq.cls);
  if (rq.limit && *rq.limit < 1) throw Refused("--limit: a number of rows, at least 1 (got " + std::to_string(*rq.limit) + ")");
  const std::string noun = rq.cls;
  if (rq.name && std::string(d.noun) == "jobs") throw Refused("list jobs: a job has no name; filter by --kind");
  if (rq.kind && !d.kind_column) throw Refused("list " + noun + ": " + noun + " have no kind; --kind is not an option here");
  if (rq.on && !(noun == "manifolds" || noun == "fields" || noun == "selections" || noun == "operators" || noun == "statistics"))
    throw Refused("list " + noun + ": --on names the manifold the nodes lie on (manifolds, fields, selections, operators, statistics)");
  const bool catalogue = is_catalogue_dataset(rq.dataset);
  if (catalogue && !rq.subject) throw Refused("list " + noun + ": a catalogue of subject packages is listed one subject at a time: name it with --subject", true);
  Lease lease = open_reader(rq.dataset, rq.subject, rq.templates, progress);
  const Database& db = (*lease).db();
  std::vector<SubjectRef> subjects;
  bool dataset_wide_jobs = false;
  if (rq.subject) {
    const auto s = find_subject_ref(db, *rq.subject);
    if (!s) throw Error(code::no_such_row, "dataset " + rq.dataset + " holds no subject named '" + *rq.subject + "'");
    subjects.push_back(*s);
  } else if (noun == "jobs") {
    dataset_wide_jobs = true;
  } else {
    subjects = all_subject_refs(db);
  }

  // the cursor: [subject, sort key, id]
  std::optional<std::vector<std::string>> cursor;
  if (rq.after) {
    const auto text_of = unhex(*rq.after);
    Json j = text_of ? Json::parse(*text_of, nullptr, false) : Json();
    if (!j.is_array() || j.size() != 3 || !j[0].is_string() || !j[1].is_string() || !j[2].is_string()) throw Refused("--after: not a cursor this command gave (take the `next` of a previous page)");
    cursor = std::vector<std::string>{j[0].get<std::string>(), j[1].get<std::string>(), j[2].get<std::string>()};
  }

  Filter base;
  base.name = rq.name;
  base.kind = rq.kind;
  struct Out { Json row; std::string subject, key, id; };
  std::vector<Out> out;
  const std::int64_t want = rq.limit.value_or(0);
  bool more = false;
  auto take = [&](const SubjectRef* s, bool scoped) {
    const Ctx c(db, s ? s->id : "", s ? s->name : "", &rq.typer);
    Filter f = base;
    if (rq.on) {
      const auto m = db.one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(c.subject_id), V(*rq.on)});
      if (!m && !dataset_wide_jobs) throw Error(code::no_such_row, "--on: subject " + c.subject_name + " holds no manifold named '" + *rq.on + "'");
      if (m) f.on = text(*m, "id");
    }
    if (cursor && s && (*cursor)[0] == s->name) f.after = std::vector<std::string>{(*cursor)[1], (*cursor)[2]};
    else if (cursor && !s) f.after = std::vector<std::string>{(*cursor)[1], (*cursor)[2]};
    const std::int64_t room = want ? want - static_cast<std::int64_t>(out.size()) : 0;
    for (const Row& r : select_rows(c, d, f, want ? room + 1 : 0, scoped)) {
      if (want && static_cast<std::int64_t>(out.size()) >= want) { more = true; return; }
      Json row = d.describe(c, r);
      out.push_back(Out{std::move(row), c.subject_name, name_key(d, r), text(r, "id")});
    }
  };
  if (dataset_wide_jobs) {
    take(nullptr, false);
  } else {
    for (const SubjectRef& s : subjects) {
      if (cursor && s.name < (*cursor)[0] && !rq.subject) continue;
      take(&s, true);
      if (more) break;
    }
  }

  // the rows, each with its subject first, and only the columns asked for
  std::vector<std::string> known;
  Json rows = Json::array();
  for (auto& o : out) {
    Json row = Json::object();
    row["subject"] = o.subject.empty() ? (o.row.contains("subject") ? o.row["subject"] : Json()) : Json(o.subject);
    for (auto it = o.row.begin(); it != o.row.end(); ++it) if (it.key() != "subject") row[it.key()] = it.value();
    if (known.empty()) for (auto it = row.begin(); it != row.end(); ++it) known.push_back(it.key());
    rows.push_back(std::move(row));
  }
  if (rq.columns) {
    // the columns of the class: from the first row, or, with no row, from an empty description of the class (a name no node can have)
    if (known.empty()) {
      static const std::map<std::string, std::vector<std::string>> by_class = {
          {"manifolds", {"subject", "id", "name", "kind", "type", "label", "description", "primary", "acquisition", "coordinates", "size", "factors", "component_of", "component_ordinal", "derived_from", "derived_by", "stored", "provenance"}},
          {"fields", {"subject", "id", "name", "kind", "type", "label", "acquisition", "description", "stored", "lineage", "provenance"}},
          {"selections", {"subject", "id", "name", "kind", "type", "label", "acquisition", "description", "elements", "members", "of_field", "weights", "stored", "dictionary", "entries", "provenance"}},
          {"operators", {"subject", "id", "name", "kind", "type", "label", "acquisition", "description", "from", "to", "method", "parameters", "shape", "stored", "lineage", "provenance"}},
          {"statistics", {"subject", "id", "name", "measure", "method", "type", "label", "acquisition", "description", "of_field", "of_statistic", "partition", "members", "members_skipped", "dof", "axes", "stored", "provenance"}},
          {"dictionaries", {"subject", "id", "name", "scope", "description", "entries", "provenance"}},
          {"spaces", {"subject", "id", "name", "geometry", "rank", "parameters", "description", "coordinate_systems"}},
          {"acquisitions", {"subject", "id", "name", "session", "task", "run", "acq", "processing", "split"}},
          {"jobs", {"subject", "id", "kind", "host", "status", "inputs", "parameters", "result", "progress", "error", "requested_by", "created_utc", "started_utc", "finished_utc"}}};
      known = by_class.at(noun);
    }
    for (const auto& k : *rq.columns)
      if (std::find(known.begin(), known.end(), k) == known.end()) throw Refused("--columns: '" + k + "' is not a column of " + noun + " (" + joined(known) + ")");
    Json kept = Json::array();
    for (const auto& row : rows) {
      Json one = Json::object();
      for (const auto& k : *rq.columns) one[k] = row[k];
      kept.push_back(std::move(one));
    }
    rows = std::move(kept);
  }
  CommandResult r;
  r.result = Json{{"schema", kListSchema}, {"class", noun}, {"subject", rq.subject ? Json(*rq.subject) : Json()}, {"rows", rows}, {"next", Json()}};
  if (more && !out.empty()) r.result["next"] = hex_of(Json::array({out.back().subject, out.back().key, out.back().id}).dump());
  return r;
}

// ============================================================================ show dataset: the count of each class
Json class_counts(const std::string& dataset, const Progress& progress, Json* unavailable) {
  std::map<std::string, std::int64_t> total;
  for (const auto& k : kListClasses) total[k] = 0;
  auto add = [&](const Database& db, const SubjectRef& s) {
    const Ctx c(db, s.id, s.name);
    for (const auto& k : kListClasses) if (k != "jobs") total[k] += count_of(c, def_of(k), true);
    if (has_table(db, "job")) total["jobs"] += count_of(c, def_of("jobs"), true);
  };
  if (is_catalogue_dataset(dataset)) {
    const Catalogue cat = Catalogue::open(dataset);
    for (const auto& e : cat.entries()) {
      if (e.status == "replaced") continue;
      try {
        DatasetWriter ds = open_subject_package(dataset, e.name, reader());
        if (const auto s = find_subject_ref(ds.db(), e.name)) add(ds.db(), *s);
      } catch (const std::exception&) {
        if (unavailable) unavailable->push_back(e.name);
      }
    }
  } else {
    Lease lease = open_reader(dataset, std::nullopt, {}, progress);
    const Database& db = (*lease).db();
    for (const SubjectRef& s : all_subject_refs(db)) add(db, s);
    if (has_table(db, "job")) total["jobs"] += static_cast<std::int64_t>(db.all("SELECT 1 FROM job WHERE subject_id IS NULL").size());
  }
  Json out = Json::object();
  for (const auto& k : kListClasses) out[k] = total[k];
  return out;
}

}  // namespace nxr::io::task

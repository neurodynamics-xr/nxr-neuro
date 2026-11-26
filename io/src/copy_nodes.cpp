// nxr-io — copy a subject's stored nodes from another package (copy_nodes.h; register D171). Rows and bytes only: nothing is computed, nothing solved.
#include "nxr/io/copy_nodes.h"

#include <algorithm>
#include <deque>
#include <map>
#include <set>

#include "nxr/io/hash.h"
#include "nxr/io/statistic.h"
#include "nxr/io/subject_package.h"

namespace nxr::io {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, "copy nodes: " + m); }

const std::vector<std::string> kNodes = {"manifold", "field", "operator", "selection", "statistic"};
// the rows that belong to one row of another table (their owner's column): copied with it
const std::vector<std::pair<std::string, std::string>> kChildren = {
    {"topology_cell", "topology_id"},    {"geometry_dimension", "geometry_id"}, {"dictionary_entry", "dictionary_id"}, {"selection_element", "selection_id"},
    {"selection_extent", "selection_id"}, {"selection_span", "selection_id"},     {"selection_factor", "selection_id"}, {"selection_measurement", "selection_id"},
    {"field_contribution", "field_id"},  {"operator_factor", "operator_id"}};
// the order the rows go in (parents before what names them; the foreign keys are deferred to the commit all the same)
const std::vector<std::string> kOrder = {"space", "coordinate_system", "dictionary", "dictionary_entry", "topology", "topology_cell", "geometry", "manifold", "geometry_dimension",
                                         "field", "operator", "selection", "statistic", "selection_element", "selection_extent", "selection_span", "selection_factor",
                                         "selection_measurement", "field_contribution", "operator_factor"};
// the columns that name an array (a store path): what the rows copied name is copied with them, or found the same in the target
const std::vector<std::string> kArrayColumns = {"path", "array_path", "positions_path"};

bool is_node(const std::string& t) { return std::find(kNodes.begin(), kNodes.end(), t) != kNodes.end(); }
std::string producer_col(const std::string& t) { return t == "selection" ? "params_json" : "producer_json"; }
Json parsed(const std::optional<std::string>& s) {
  if (!s || s->empty()) return Json::object();
  Json j = Json::parse(*s, nullptr, false);
  return j.is_discarded() ? Json::object() : j;
}
std::string what(const std::string& table, const Row& r) {
  return table + " " + text(r, "name") + (opt_text(r, "path") ? " (" + text(r, "path") + ")" : std::string());
}
bool hexdig(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
bool uuid_at(const std::string& s, std::size_t i) {
  if (i + 36 > s.size()) return false;
  for (std::size_t k = 0; k < 36; ++k) {
    const char c = s[i + k];
    if (k == 8 || k == 13 || k == 18 || k == 23) { if (c != '-') return false; } else if (!hexdig(c)) return false;
  }
  return true;
}
// every id of `map` in a text (a JSON column, a zarr.json document) replaced by its counterpart
std::string remap_text(const std::string& s, const std::map<std::string, std::string>& map) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    if (uuid_at(s, i)) {
      const std::string id = s.substr(i, 36);
      const auto it = map.find(id);
      out += it == map.end() ? id : it->second;
      i += 36;
    } else {
      out += s[i++];
    }
  }
  return out;
}
struct Digest { std::int64_t size = 0; std::string sha; };
bool operator==(const Digest& a, const Digest& b) { return a.size == b.size && a.sha == b.sha; }
bool operator!=(const Digest& a, const Digest& b) { return !(a == b); }
// a package's recorded digests of the keys under `prefix/` (relative key -> digest), zarr.json documents left out unless asked
std::map<std::string, Digest> digests_under(const Database& db, const std::string& prefix, bool documents = false) {
  std::map<std::string, Digest> out;
  for (const Row& r : db.all("SELECT key, size_bytes, sha256 FROM key_digest WHERE key >= ? AND key < ? ORDER BY key", {prefix + "/", prefix + "0"})) {
    const std::string k = text(r, "key");
    if (!documents && (k == prefix + "/zarr.json" || (k.size() > 10 && k.compare(k.size() - 10, 10, "/zarr.json") == 0))) continue;
    out[k.substr(prefix.size() + 1)] = Digest{integer(r, "size_bytes"), text(r, "sha256")};
  }
  return out;
}
bool has_keys(const Database& db, const std::string& prefix) {
  return db.one("SELECT 1 AS x FROM key_digest WHERE key >= ? AND key < ? LIMIT 1", {prefix + "/", prefix + "0"}).has_value();
}
bool inside(const std::string& key, const std::string& folder) { return key == folder || (key.size() > folder.size() && key.compare(0, folder.size(), folder) == 0 && key[folder.size()] == '/'); }
void strings_of(const Json& j, std::vector<std::string>& out) {
  if (j.is_string()) out.push_back(j.get<std::string>());
  else if (j.is_array() || j.is_object()) for (const auto& v : j) strings_of(v, out);
}

struct Plan {
  const Database* s = nullptr;   // the source's rows
  Database* t = nullptr;         // the target's
  std::string s_subject, t_subject, s_dataset, t_dataset;
  std::set<std::pair<std::string, std::string>> closure;          // (table, source id): the nodes that go
  std::map<std::string, std::string> ids;                          // source id -> target id (copied, kept or found)
  std::map<std::string, std::pair<std::string, Row>> rows;         // source id -> (table, source row): the id-keyed rows to insert
  std::vector<std::pair<std::string, Row>> children;               // (table, source row): the rows of an owner that is copied
  std::set<std::string> kept;                                      // source ids of nodes the target holds already, the same
  Json copied = Json::array(), kept_list = Json::array(), found = Json::array();
  std::set<std::string> anchors;                                   // ids the source lists in anchor_row (another package's own rows)
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> fks;   // table -> (column, referenced table)
};

const std::vector<std::pair<std::string, std::string>>& fks_of(Plan& p, const std::string& table) {
  auto it = p.fks.find(table);
  if (it != p.fks.end()) return it->second;
  std::vector<std::pair<std::string, std::string>> v;
  for (const Row& r : p.s->all("SELECT \"from\" AS col, \"table\" AS tab FROM pragma_foreign_key_list(?)", {table})) v.emplace_back(text(r, "col"), text(r, "tab"));
  return p.fks[table] = v;
}

// THE SAME NODE in the target: its counts and the digests of every chunk under its folder (and, for a manifold, of the arrays its geometry and cells name) equal
std::optional<std::string> same_in_target(Plan& p, const std::string& table, const Row& src) {
  const auto path = opt_text(src, "path");
  std::optional<Row> tgt;
  if (path) tgt = p.t->one("SELECT * FROM " + table + " WHERE subject_id = ? AND path = ? AND status = 'complete'", {p.t_subject, *path});
  else tgt = p.t->one("SELECT * FROM " + table + " WHERE subject_id = ? AND name = ? AND path IS NULL AND status = 'complete'", {p.t_subject, text(src, "name")});
  if (!tgt) return std::nullopt;
  std::vector<std::string> cols;
  if (table == "manifold") cols = {"type", "n_vertices", "n_edges", "n_faces", "n_volumes"};
  else if (table == "field") cols = {"kind", "value_type", "cell", "data_type", "layout", "nnz"};
  else if (table == "operator") cols = {"kind", "layout", "method", "params_json", "n_rows", "n_cols", "nnz", "from_cell", "to_cell"};
  else if (table == "selection") cols = {"type", "n_elements", "n_members", "code"};
  else cols = {"measure", "method", "value_type"};
  for (const auto& c : cols) {
    const bool sn = !src.count(c) || is_null(src, c), tn = !tgt->count(c) || is_null(*tgt, c);
    if (sn != tn || (!sn && src.at(c) != tgt->at(c)))
      refuse(what(table, src) + " is in the target and differs from the source's (" + c + "): delete it there first (`delete subject --node " + table + ":" + text(src, "name") + "`)");
  }
  std::vector<std::string> arrays;
  if (path) arrays.push_back(*path);
  if (table == "manifold") {
    if (const auto g = p.s->one("SELECT positions_path FROM geometry WHERE id = ?", {V(opt_text(src, "geometry_id"))}); g && opt_text(*g, "positions_path")) arrays.push_back(text(*g, "positions_path"));
    for (const Row& c : p.s->all("SELECT path FROM topology_cell WHERE topology_id = ?", {text(src, "topology_id")})) arrays.push_back(text(c, "path"));
  }
  for (const auto& a : arrays)
    if (digests_under(*p.s, a) != digests_under(*p.t, a))
      refuse(what(table, src) + " is in the target and its arrays at " + a + " differ from the source's: delete it there first (`delete subject --node " + table + ":" +
             text(src, "name") + "`)");
  return text(*tgt, "id");
}

// THE CLOSURE: the seeds, and every node that lies on or is made from one that goes (copy_nodes.h)
void close_over(Plan& p, const std::vector<std::string>& seeds) {
  // the outputs of one run: nodes by their provenance's cache key
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> by_run;
  std::map<std::string, std::string> run_of;
  for (const auto& t : kNodes) {
    if (!has_table(*p.s, t)) continue;
    const std::string col = producer_col(t);
    for (const Row& r : p.s->all("SELECT id, " + col + " AS pj FROM " + t + " WHERE subject_id = ? AND status = 'complete'", {p.s_subject})) {
      const Json j = parsed(opt_text(r, "pj"));
      if (!j.is_object() || !j.contains("cache") || j["cache"].is_null()) continue;
      const std::string key = j["cache"].dump();
      by_run[key].emplace_back(t, text(r, "id"));
      run_of[text(r, "id")] = key;
    }
  }
  std::deque<std::pair<std::string, std::string>> todo;
  auto add = [&](const std::string& t, const std::string& id) {
    if (p.closure.insert({t, id}).second) todo.emplace_back(t, id);
  };
  for (const auto& s : seeds) add("manifold", s);
  auto each = [&](const std::string& t, const std::string& where, const std::vector<Value>& args) {
    if (!has_table(*p.s, t)) return;
    for (const Row& r : p.s->all("SELECT id FROM " + t + " WHERE subject_id = ? AND status = 'complete' AND (" + where + ")", [&] {
           std::vector<Value> a = {p.s_subject};
           a.insert(a.end(), args.begin(), args.end());
           return a;
         }()))
      add(t, text(r, "id"));
  };
  while (!todo.empty()) {
    const auto [t, id] = todo.front();
    todo.pop_front();
    if (t == "manifold") {
      each("manifold", "(derived_from_id = ? OR component_of_id = ?) AND type <> 'product'", {id, id});
      each("field", "manifold_id = ?", {id});
      each("selection", "manifold_id = ?", {id});
      each("operator", "from_manifold_id = ? OR to_manifold_id = ?", {id, id});
    } else if (t == "selection") {
      each("selection", "member_of_id = ?", {id});
    } else if (t == "field") {
      each("statistic", "of_field_id = ?", {id});
    }
    if (const auto it = run_of.find(id); it != run_of.end())
      for (const auto& [rt, rid] : by_run[it->second]) add(rt, rid);
  }
}

// a row the copied rows name, found in the target (or, for a space, a coordinate system or a dictionary the target lacks, copied); its target id
std::string resolve(Plan& p, const std::string& table, const std::string& id, const std::string& named_by);

void plan_row(Plan& p, const std::string& table, const Row& r) {
  const std::string id = text(r, "id");
  if (p.ids.count(id)) return;
  p.ids[id] = uuid7();
  p.rows[id] = {table, r};
  for (const auto& [child, owner] : kChildren) {
    if (!has_table(*p.s, child)) continue;
    bool owned = false;
    for (const auto& [col, tab] : fks_of(p, child)) if (col == owner && tab == table) owned = true;
    if (owned) for (const Row& c : p.s->all("SELECT * FROM " + child + " WHERE " + owner + " = ?", {id})) p.children.emplace_back(child, c);
  }
}

std::string resolve(Plan& p, const std::string& table, const std::string& id, const std::string& named_by) {
  if (const auto it = p.ids.find(id); it != p.ids.end()) return it->second;
  if (table == "subject") {
    if (id == p.s_subject) return p.ids[id] = p.t_subject;
    if (p.t->one("SELECT id FROM subject WHERE id = ?", {id})) return p.ids[id] = id;
    refuse(named_by + " names another subject (" + id + ") the target does not hold");
  }
  if (table == "dataset") {
    if (p.t->one("SELECT id FROM dataset WHERE id = ?", {id})) return p.ids[id] = id;
    return p.ids[id] = p.t_dataset;   // the home dataset a package carries: the target's own
  }
  if (p.anchors.count(id)) {   // another package's own row (a template's): the same id in the target
    if (p.t->one("SELECT id FROM " + table + " WHERE id = ?", {id})) return p.ids[id] = id;
    refuse(named_by + " names a " + table + " of another package (" + id + ") the target does not carry: register the target to it first");
  }
  const auto row = p.s->read(table, id);
  if (!row) refuse(named_by + " names a " + table + " (" + id + ") the source does not hold");
  if (table == "acquisition") {
    if (const auto t = p.t->one("SELECT id FROM acquisition WHERE subject_id = ? AND label = ?", {p.t_subject, text(*row, "label")})) return p.ids[id] = text(*t, "id");
    refuse(named_by + " was made for the acquisition " + text(*row, "label") + ", which the target does not hold: import it first");
  }
  if (table == "space") {
    if (const auto t = p.t->one("SELECT id FROM space WHERE subject_id = ? AND name = ?", {p.t_subject, text(*row, "name")})) return p.ids[id] = text(*t, "id");
    plan_row(p, table, *row);
    return p.ids[id];
  }
  if (table == "coordinate_system") {
    const std::string space = resolve(p, "space", text(*row, "space_id"), named_by);
    if (const auto t = p.t->one("SELECT id FROM coordinate_system WHERE space_id = ? AND name = ?", {space, text(*row, "name")})) return p.ids[id] = text(*t, "id");
    plan_row(p, table, *row);
    return p.ids[id];
  }
  if (table == "dictionary") {
    const std::string scope = text(*row, "scope");
    std::optional<Row> t;
    if (scope == "subject") t = p.t->one("SELECT id FROM dictionary WHERE name = ? AND scope = 'subject' AND subject_id = ?", {text(*row, "name"), p.t_subject});
    else t = p.t->one("SELECT id FROM dictionary WHERE name = ? AND scope = ?", {text(*row, "name"), scope});
    if (t) {
      auto entries = [](const Database& db, const std::string& d) {
        std::vector<std::pair<std::int64_t, std::string>> e;
        for (const Row& r : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {d})) e.emplace_back(integer(r, "code"), text(r, "name"));
        return e;
      };
      if (entries(*p.s, id) != entries(*p.t, text(*t, "id"))) refuse("the dictionary " + text(*row, "name") + " (" + scope + ") is in the target with other entries");
      return p.ids[id] = text(*t, "id");
    }
    plan_row(p, table, *row);
    return p.ids[id];
  }
  if (table == "topology" || table == "geometry") {   // a manifold's own construction: copied with it
    plan_row(p, table, *row);
    return p.ids[id];
  }
  if (is_node(table)) {
    if (const auto t = same_in_target(p, table, *row)) {
      p.found.push_back(Json{{"class", table}, {"name", text(*row, "name")}, {"path", opt_text(*row, "path").value_or("")}});
      return p.ids[id] = *t;
    }
    refuse(named_by + " names " + what(table, *row) + ", which the target does not hold: name its manifold with --manifold, or make it there first");
  }
  refuse(named_by + " names a row of " + table + " that cannot be copied");
}

// every foreign key of the rows to insert resolved (resolving may add rows: a space, a coordinate system, a dictionary, a topology, a geometry)
void resolve_all(Plan& p) {
  std::set<std::string> done;
  std::size_t children_done = 0;
  for (bool more = true; more;) {
    more = false;
    std::vector<std::string> pending;
    for (const auto& [id, tr] : p.rows) if (!done.count(id)) pending.push_back(id);
    for (const auto& id : pending) {
      done.insert(id);
      more = true;
      const auto tr = p.rows.at(id);
      for (const auto& [col, tab] : fks_of(p, tr.first))
        if (tr.second.count(col) && !is_null(tr.second, col)) resolve(p, tab, text(tr.second, col), what(tr.first, tr.second));
    }
    for (; children_done < p.children.size(); ++children_done) {
      more = true;
      const auto c = p.children[children_done];
      for (const auto& [col, tab] : fks_of(p, c.first))
        if (c.second.count(col) && !is_null(c.second, col)) resolve(p, tab, text(c.second, col), c.first + " of a copied row");
    }
  }
}

Row target_row(Plan& p, const std::string& table, const Row& src, const std::vector<std::string>& columns, const Json& copied) {
  Row out;
  std::set<std::string> fk_cols;
  for (const auto& [col, tab] : fks_of(p, table)) fk_cols.insert(col);
  for (const auto& c : columns) {
    const auto it = src.find(c);
    if (it == src.end()) continue;
    Value v = it->second;
    if (!is_null(v) && std::holds_alternative<std::string>(v)) {
      const std::string& s = std::get<std::string>(v);
      if (c == "id" || fk_cols.count(c)) {
        const auto m = p.ids.find(s);
        if (m == p.ids.end()) refuse(table + "." + c + " " + s + " is not resolved");
        v = m->second;
      } else if (c.size() > 5 && c.compare(c.size() - 5, 5, "_json") == 0) {
        v = remap_text(s, p.ids);
      }
    }
    out[c] = v;
  }
  if (out.count("status")) out["status"] = std::string("complete");
  if (!copied.is_null() && is_node(table) && table != "selection" && std::find(columns.begin(), columns.end(), "producer_json") != columns.end()) {
    Json pj = parsed(out.count("producer_json") && !is_null(out["producer_json"]) ? std::optional<std::string>(std::get<std::string>(out["producer_json"])) : std::nullopt);
    if (!pj.is_object()) pj = Json::object();
    Json c = copied;
    c["from_id"] = text(src, "id");
    pj["copied"] = c;
    out["producer_json"] = jdump(pj);
  }
  return out;
}

}  // namespace

Json copy_nodes(const fs::path& source, SubjectWriter& target, const CopyNodesOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.manifolds.empty()) refuse("name at least one manifold of the source to copy (--manifold)");
  if (!target.dataset().is_package()) refuse("the target is not a subject package (a one-database dataset keeps no digests to check the copy against): `split dataset` first");
  std::error_code ec;
  if (fs::equivalent(source, target.dataset().root(), ec)) refuse("the source and the target are the same package");
  const SubjectPackage src = SubjectPackage::open(source);
  const PackageRecord rec = src.record();
  // CHECKED: the source as it was closed (its rows and the digests of its keys hash to the recorded content hash), at this schema version, of a subject of the target's name
  if (rec.content_hash.empty()) refuse(source.filename().string() + " has no recorded content hash (it was never closed by a writer)");
  if (src.content_hash() != rec.content_hash)
    refuse(source.filename().string() + ": its rows or digests changed since its content hash was recorded (`verify package --package " + source.filename().string() + "`)");
  if (rec.schema_version != schema_version())
    refuse(source.filename().string() + " is at schema " + std::to_string(rec.schema_version) + ", this program at " + std::to_string(schema_version()) + ": migrate it first");
  if (rec.name != target.name()) refuse(source.filename().string() + " is the package of " + rec.name + ", the target's subject is " + target.name() + ": a copy stays within one subject");

  Plan p;
  p.s = &src.db();
  p.t = &target.db();
  p.s_subject = text(src.subject_row(), "id");
  p.t_subject = target.id();
  if (const auto d = p.t->one("SELECT id FROM dataset LIMIT 1")) p.t_dataset = text(*d, "id");
  if (has_table(*p.s, "anchor_row")) for (const Row& r : p.s->all("SELECT row_id FROM anchor_row")) p.anchors.insert(text(r, "row_id"));

  std::vector<std::string> seeds;
  for (const auto& m : opt.manifolds) {
    const auto rows = p.s->all("SELECT id, name, path FROM manifold WHERE subject_id = ? AND status = 'complete' AND (name = ? OR path = ?)", {p.s_subject, m, m});
    if (rows.empty()) refuse(source.filename().string() + " holds no manifold " + m);
    if (rows.size() > 1) refuse(source.filename().string() + " holds " + std::to_string(rows.size()) + " manifolds named " + m + ": name one by its path");
    seeds.push_back(text(rows[0], "id"));
  }
  close_over(p, seeds);

  // KEPT: the nodes that go and that the target holds already, the same; the others are copied
  std::vector<std::pair<std::string, Row>> nodes;
  for (const auto& t : kNodes)
    for (const auto& [tt, id] : p.closure)
      if (tt == t) nodes.emplace_back(t, *p.s->read(t, id));
  for (const auto& [t, r] : nodes) {
    const std::string id = text(r, "id");
    if (const auto k = same_in_target(p, t, r)) {
      p.ids[id] = *k;
      p.kept.insert(id);
      p.kept_list.push_back(Json{{"class", t}, {"name", text(r, "name")}, {"path", opt_text(r, "path").value_or("")}, {"id", *k}});
      continue;
    }
    if (p.t->one("SELECT id FROM " + t + " WHERE subject_id = ? AND name = ? AND status = 'complete'", {p.t_subject, text(r, "name")}))
      refuse("the target holds another " + t + " named " + text(r, "name") + " (at another path): delete it there first");
  }
  for (const auto& [t, r] : nodes) if (!p.kept.count(text(r, "id"))) plan_row(p, t, r);
  resolve_all(p);

  // THE ARRAYS: every key under a copied node's folder, and each array a copied row names outside one: copied when no node of the target holds it, else found the same
  std::vector<std::string> folders;
  for (const auto& [id, tr] : p.rows)
    if (is_node(tr.first) && opt_text(tr.second, "path")) folders.push_back(text(tr.second, "path"));
  std::set<std::string> named;
  auto note_paths = [&](const Row& r) {
    for (const auto& c : kArrayColumns) if (r.count(c) && !is_null(r, c) && std::holds_alternative<std::string>(r.at(c))) named.insert(std::get<std::string>(r.at(c)));
    for (const auto& [c, v] : r)
      if (c.size() > 5 && c.compare(c.size() - 5, 5, "_json") == 0 && std::holds_alternative<std::string>(v)) {
        std::vector<std::string> strs;
        strings_of(parsed(std::get<std::string>(v)), strs);
        for (const auto& s : strs) if (!s.empty() && s.size() < 512 && p.s->one("SELECT 1 AS x FROM key_digest WHERE key = ?", {s + "/zarr.json"})) named.insert(s);
      }
  };
  for (const auto& [id, tr] : p.rows) note_paths(tr.second);
  for (const auto& c : p.children) note_paths(c.second);
  std::vector<std::string> free_arrays;
  for (const auto& a : named) {
    if (std::any_of(folders.begin(), folders.end(), [&](const std::string& f) { return inside(a, f); })) continue;
    if (!has_keys(*p.s, a)) continue;
    if (has_keys(*p.t, a)) {
      if (digests_under(*p.s, a) != digests_under(*p.t, a)) refuse("the array " + a + " a copied row names is in the target with other bytes");
      continue;
    }
    if (named_by(*p.t, p.t_subject, a)) refuse("the array " + a + " a copied row names belongs to a node of the target, which does not hold it");
    free_arrays.push_back(a);
  }
  for (const auto& f : folders) {
    if (has_keys(*p.t, f) || p.t->one("SELECT 1 AS x FROM key_digest WHERE key = ?", {f})) refuse("the target already holds keys under " + f + " (no row names them): remove them first");
  }
  std::vector<std::pair<std::string, Digest>> keys;
  for (const auto& f : folders) for (const auto& [k, d] : digests_under(*p.s, f, true)) keys.emplace_back(f + "/" + k, d);
  for (const auto& a : free_arrays) for (const auto& [k, d] : digests_under(*p.s, a, true)) keys.emplace_back(a + "/" + k, d);
  std::int64_t bytes = 0;
  for (const auto& [k, d] : keys) bytes += d.size;

  for (const auto& [id, tr] : p.rows)
    if (is_node(tr.first))
      p.copied.push_back(Json{{"class", tr.first}, {"name", text(tr.second, "name")}, {"path", opt_text(tr.second, "path").value_or("")}, {"id", p.ids.at(id)}, {"from_id", id}});
  Json out = Json{{"from", Json{{"package", source.filename().string()}, {"subject", rec.name}, {"content_hash", rec.content_hash}}},
                  {"copied", p.copied}, {"kept", p.kept_list}, {"found", p.found}, {"keys", static_cast<std::int64_t>(keys.size())}, {"bytes", bytes}, {"dry_run", opt.dry_run}};
  if (opt.dry_run) return out;

  // BYTES FIRST: each key copied (a document with the target's ids), each chunk's digest checked against the source's; a failure removes what was written
  Store& to = target.store();
  const Store& from = src.store();
  std::vector<std::string> written;
  auto undo = [&] {
    for (const auto& k : written) try { to.erase(k); } catch (...) {}
  };
  try {
    constexpr std::uint64_t kSlab = 64ULL << 20;
    for (const auto& [k, d] : keys) {
      const bool document = k.size() >= 9 && k.compare(k.size() - 9, 9, "zarr.json") == 0;
      if (document) {
        const auto b = from.get(k);
        if (!b) refuse(k + " is in the source's digests but not in its store");
        const std::string txt = remap_text(std::string(b->begin(), b->end()), p.ids);
        to.set(k, reinterpret_cast<const std::uint8_t*>(txt.data()), txt.size());
      } else {
        const auto size = from.size(k);
        if (!size) refuse(k + " is in the source's digests but not in its store");
        auto w = to.open_writer(k);
        for (std::uint64_t at = 0; at < *size; at += kSlab) {
          const auto part = from.get_range(k, at, std::min<std::uint64_t>(kSlab, *size - at));
          if (!part) refuse(k + " vanished from the source while it was copied");
          w->append(part->data(), part->size());
        }
        w->close();
      }
      written.push_back(k);
      if (!document) {
        const auto got = p.t->one("SELECT size_bytes, sha256 FROM key_digest WHERE key = ?", {k});
        if (!got || integer(*got, "size_bytes") != d.size || text(*got, "sha256") != d.sha)
          refuse(k + ": the bytes copied do not have the size and SHA-256 the source recorded (" + d.sha + ")");
      }
    }
    if (progress) progress(Json{{"stage", "copied-keys"}, {"keys", static_cast<std::int64_t>(keys.size())}, {"bytes", bytes}});

    // THE ROWS LAST, in one transaction (the foreign keys checked at its commit)
    const Json copied = Json{{"from_package", source.filename().string()}, {"from_subject", rec.name}, {"from_content_hash", rec.content_hash}, {"utc", now_utc(false)}, {"by", opt.by}};
    Database& db = *p.t;
    db.tx([&] {
      db.execute("PRAGMA defer_foreign_keys = ON");
      for (const auto& table : kOrder) {
        if (!has_table(db, table)) continue;
        const auto& cols = db.columns(table);
        for (const auto& [id, tr] : p.rows) {
          if (tr.first != table) continue;
          Row r = target_row(p, table, tr.second, cols, copied);
          // one primary manifold of a type in a subject: a copy of the source's primary is not the target's when the target has one
          if (table == "manifold" && r.count("is_primary") && !is_null(r, "is_primary") && integer(r, "is_primary") == 1 &&
              db.one("SELECT id FROM manifold WHERE subject_id = ? AND type = ? AND is_primary = 1", {p.t_subject, text(r, "type")}))
            r["is_primary"] = static_cast<std::int64_t>(0);
          db.insert(table, r);
        }
        for (const auto& c : p.children) if (c.first == table) db.insert(table, target_row(p, table, c.second, cols, Json()));
      }
      for (const auto& [id, tr] : p.rows) {
        if (!is_node(tr.first)) continue;
        if (const auto o = origin_of(*p.s, tr.first, id)) set_origin(db, tr.first, p.ids.at(id), *o);
      }
    });
  } catch (...) {
    undo();
    throw;
  }
  drain(*p.t, target.datastore());   // each copied node's zarr.json attributes from its target row (a failure here is finished by the next drain: the rows and bytes are in)
  if (progress) progress(Json{{"stage", "copied-rows"}, {"nodes", p.copied.size()}});
  return out;
}

}  // namespace nxr::io

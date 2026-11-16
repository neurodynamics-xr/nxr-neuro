#include "named_paths.h"

#include <algorithm>
#include <functional>

#include "nxr/io/errors.h"
#include "nxr/io/writer.h"
#include <set>

namespace nxr::io {

namespace {
std::string fold(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(c < 128 ? std::tolower(c) : c); });
  return t;
}
bool starts_with(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
}  // namespace

std::vector<std::pair<std::string, std::string>> named_paths(const Database& db, const std::string& subject_id) {
  std::vector<std::pair<std::string, std::string>> out;
  std::set<std::string> have;
  auto add = [&](const Row& r, const std::string& col, const std::string& table) {
    if (is_null(r, col)) return;
    std::string q = text(r, col);
    while (!q.empty() && q.front() == '/') q.erase(q.begin());
    while (!q.empty() && q.back() == '/') q.pop_back();
    if (!q.empty() && have.insert(q).second) out.push_back({q, table + " " + text(r, "id")});
  };
  struct Col { const char* table; const char* sql; };
  const Col columns[] = {
      {"field", "SELECT id, path AS p FROM field WHERE subject_id = ?"},
      {"selection", "SELECT id, path AS p FROM selection WHERE subject_id = ?"},
      {"selection", "SELECT id, array_path AS p FROM selection WHERE subject_id = ?"},
      {"operator", "SELECT id, path AS p FROM operator WHERE subject_id = ?"},
      {"manifold", "SELECT id, path AS p FROM manifold WHERE subject_id = ?"},
      // a geometry's positions and a topology's cells name arrays only while a manifold stands on them (the rows a manifold's delete leaves behind name nothing)
      {"geometry", "SELECT g.id AS id, g.positions_path AS p FROM geometry g JOIN manifold m ON m.geometry_id = g.id WHERE g.subject_id = ?"},
      {"topology", "SELECT DISTINCT c.topology_id AS id, c.path AS p FROM topology_cell c JOIN topology t ON t.id = c.topology_id "
                   "JOIN manifold m ON m.topology_id = t.id WHERE t.subject_id = ?"},
      {"selection measurement of", "SELECT m.selection_id AS id, m.array_path AS p FROM selection_measurement m JOIN selection s ON s.id = m.selection_id WHERE s.subject_id = ?"},
  };
  for (const auto& c : columns)
    for (const auto& r : db.all(c.sql, {subject_id})) add(r, "p", c.table);

  auto from_json = [&](const std::string& sql, const std::string& table, const std::function<std::vector<Json>(const Json&)>& paths) {
    for (const auto& r : db.all(sql, {subject_id})) {
      Json j;
      try { j = Json::parse(text(r, "j")); } catch (...) { continue; }  // not JSON: names nothing
      if (!j.is_object()) continue;
      for (const auto& p : paths(j)) {
        if (!p.is_string()) continue;  // as the Studio's `add`: only a string names a path
        Row one{{"id", text(r, "id")}, {"p", p.get<std::string>()}};
        add(one, "p", table);
      }
    }
  };
  from_json("SELECT id, params_json AS j FROM selection WHERE subject_id = ? AND (params_json LIKE '%\"weights\"%' OR params_json LIKE '%\"atlas\"%')",
            "selection", [](const Json& j) {
              std::vector<Json> v;
              v.push_back(j.contains("weights") ? j["weights"] : Json());
              if (j.contains("atlas") && j["atlas"].is_object()) {
                v.push_back(j["atlas"].contains("node") ? j["atlas"]["node"] : Json());
                v.push_back(j["atlas"].contains("areas") ? j["atlas"]["areas"] : Json());
              }
              return v;
            });
  from_json("SELECT id, producer_json AS j FROM manifold WHERE subject_id = ? AND producer_json LIKE '%\"kinds\"%'", "manifold", [](const Json& j) {
    std::vector<Json> v;
    if (j.contains("spectrum") && j["spectrum"].is_object() && j["spectrum"].contains("kinds")) v.push_back(j["spectrum"]["kinds"]);
    return v;
  });
  return out;
}

bool overlaps(const std::string& a, const std::string& b) {
  const std::string x = fold(a), y = fold(b);
  return x == y || starts_with(x, y + "/") || starts_with(y, x + "/");
}

std::optional<std::pair<std::string, std::string>> named_overlapping(const std::vector<std::pair<std::string, std::string>>& named,
                                                                     const std::string& path) {
  for (const auto& q : named) if (overlaps(q.first, path)) return q;
  return std::nullopt;
}

bool named_by_remaining(const Database& db, const std::optional<std::string>& dataset, const std::string& location) {
  if (!dataset) return false;
  for (const auto& s : db.all("SELECT id, path FROM subject")) {
    const std::string prefix = *dataset + "/" + text(s, "path") + "/";
    if (starts_with(location, prefix)) return named_overlapping(named_paths(db, text(s, "id")), location.substr(prefix.size())).has_value();
  }
  return false;
}

bool remove_store_node(Store& store, const std::string& rel) {
  const std::string location = rel;
  std::vector<std::string> segs;
  std::size_t at = 0;
  while (true) {
    const std::size_t k = rel.find('/', at);
    segs.push_back(rel.substr(at, k == std::string::npos ? std::string::npos : k - at));
    if (k == std::string::npos) break;
    at = k + 1;
  }
  for (const auto& x : segs)
    if (x.empty() || x == "." || x == ".." || x.find('\\') != std::string::npos || x.find('\0') != std::string::npos) throw Error(code::invalid_argument, location + ": not a path inside the store — refused");
  if (!store.exists_prefix("")) return false;
  if (store.is_link("")) throw Error(code::invalid_argument, "the store is a link — refused");
  std::string dir;
  for (const auto& seg : segs) {
    const Listing in = store.list(dir);   // the names found EXACTLY, byte-equal
    const bool group = std::find(in.prefixes.begin(), in.prefixes.end(), seg) != in.prefixes.end();
    const bool value = std::find(in.keys.begin(), in.keys.end(), seg) != in.keys.end();
    if (!group && !value) return false;
    dir = join_key(dir, seg);
    if (store.is_link(dir)) throw Error(code::invalid_argument, location + ": a link — refused");
    if (!group) throw Error(code::invalid_argument, location + ": not a group — refused");
  }
  store.erase_prefix(dir);
  return true;
}

std::vector<std::string> deleting_rows(const Database& db, Store& store, const std::string& subject_id, const std::function<void()>& del) {
  const auto before = named_paths(db, subject_id);
  del();
  const auto after = named_paths(db, subject_id);
  std::vector<std::string> removed;
  for (const auto& [p, by] : before) {
    if (named_overlapping(after, p)) continue;
    try { if (remove_store_node(store, p)) removed.push_back(p); } catch (const Error&) {}
  }
  return removed;
}

std::optional<std::string> named_by(const Database& db, const std::string& subject_id, const std::string& rel) {
  if (const auto q = named_overlapping(named_paths(db, subject_id), rel)) return q->second;
  return std::nullopt;
}

}  // namespace nxr::io

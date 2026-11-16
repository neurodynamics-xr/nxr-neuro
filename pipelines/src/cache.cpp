#include "nxr/pipelines/cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace nxr::pipelines {

Json CacheKey::json() const {
  Json j = Json::object();
  j["pipeline"] = pipeline;
  j["params"] = params;
  j["input"] = input;
  j["compute"] = compute;
  return j;
}

Json stamp(const CacheKey& key) { return Json{{"cache", key.json()}}; }

namespace {
// nlohmann's ordered_json compares objects key by key irrespective of order; normalise anyway by comparing sorted dumps of the parts.
bool same(const Json& a, const Json& b) { return a == b; }
}  // namespace

std::string mesh_key(const std::vector<std::int32_t>& faces) {
  std::uint32_t h = 0x811C9DC5u;
  for (std::int32_t f : faces) {
    const auto u = static_cast<std::uint32_t>(f);
    for (int b = 0; b < 4; ++b) h = (h ^ ((u >> (8 * b)) & 0xFFu)) * 0x01000193u;   // little-endian bytes
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, "%zu:%08x", faces.size() / 3, h);
  return buf;
}

std::string positions_key(const std::vector<double>& vertices) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (double d : vertices) {
    const float f = static_cast<float>(d);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    for (int b = 0; b < 4; ++b) h = (h ^ ((u >> (8 * b)) & 0xFFu)) * 0x100000001B3ull;
  }
  char buf[48];
  std::snprintf(buf, sizeof buf, "%zu:%016llx", vertices.size() / 3, static_cast<unsigned long long>(h));
  return buf;
}

std::optional<Row> find_cached(const nxr::io::Database& db, const std::string& subject_id, const std::string& table, const CacheKey& key,
                               const std::optional<std::string>& path) {
  if (std::find(std::begin(nxr::io::kStatusTables), std::end(nxr::io::kStatusTables), table) == std::end(nxr::io::kStatusTables) || table == "subject")
    throw nxr::io::Error(nxr::io::code::invalid_argument, "find_cached: " + table + " is not a node table (manifold, selection, field, operator)");
  if (table == "statistic" && !nxr::io::has_table(db, "statistic")) return std::nullopt;   // a dataset made before the statistic table has none
  const Json want = key.json();
  // a Selection has no producer_json: its producer records how it was made in params_json (the Studio's searchAtlas: params_json.search), the key under "cache"
  const std::string col = table == "selection" ? "params_json" : "producer_json";
  std::string sql = "SELECT * FROM " + table + " WHERE subject_id = ? AND status = 'complete' AND " + col + " IS NOT NULL";
  std::vector<nxr::io::Value> args{nxr::io::V(subject_id)};
  if (path) {
    sql += " AND path = ?";
    args.push_back(nxr::io::V(*path));
  }
  for (const Row& r : db.all(sql, args)) {
    Json p = Json::parse(nxr::io::text(r, col), nullptr, false);
    if (p.is_object() && p.contains("cache") && same(p["cache"], want)) return r;
  }
  return std::nullopt;
}

// Arrays and rows of a name already stored: refused, or (replace) moved aside - never deleted - and the rows removed.
std::string clear_stored(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::vector<std::pair<const char*, std::string>>& nodes, bool replace, const std::string& what, bool& replaced) {
  using namespace nxr::io;
  std::string moved;
  std::vector<std::pair<std::string, Row>> have;
  for (const auto& [table, path] : nodes) if (const auto r = sub.find(table, path)) have.emplace_back(table, *r);
  if (have.empty()) return moved;
  if (!replace) throw Error(code::already_exists, sub.name() + ": " + what + " already exists (made by another job); pass --replace");
  for (const auto& [table, r] : have) {
    replaced = true;
    const std::string loc = table == "selection" ? opt_text(r, "array_path").value_or("") : opt_text(r, "path").value_or("");
    if (!loc.empty() && sub.store().exists_prefix(loc)) {
      // a folder of its own for each replacement of the same node the same day (`_replaced-<date>`, then `-2`, `-3`, …): never over what an earlier one moved
      std::string aside = "_replaced-" + now_utc(false).substr(0, 10) + "/" + text(sub.row(), "path");
      for (int n = 2; ds.store().exists_prefix(nxr::io::join_key(aside, loc)); ++n)
        aside = "_replaced-" + now_utc(false).substr(0, 10) + "-" + std::to_string(n) + "/" + text(sub.row(), "path");
      sub.datastore().rename_prefix(nxr::io::join_key(sub.prefix(), loc), nxr::io::join_key(aside, loc));
      moved = (ds.root() / aside).string();
    }
    sub.remove(table, text(r, "id"));
  }
  return moved;
}

}  // namespace nxr::pipelines

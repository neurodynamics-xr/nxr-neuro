// nxr-io — cells (see cells.h): the Studio's db/cells.ts.
#include "nxr/io/cells.h"

#include "nxr/io/errors.h"

namespace nxr::io {

namespace {
std::optional<std::string> s(const Row& r, const char* c) {
  const auto it = r.find(c);
  if (it == r.end()) return std::nullopt;
  if (const auto* x = std::get_if<std::string>(&it->second)) return *x;
  return std::nullopt;
}
std::optional<std::int64_t> i(const Row& r, const char* c) {
  const auto it = r.find(c);
  if (it == r.end()) return std::nullopt;
  if (const auto* x = std::get_if<std::int64_t>(&it->second)) return *x;
  if (const auto* x = std::get_if<double>(&it->second)) return static_cast<std::int64_t>(*x);
  return std::nullopt;
}
// a product's factors: its geometry's dimension rows (D37), in order
std::vector<std::optional<Row>> factors_of(const Database& db, const Row& m) {
  std::vector<std::optional<Row>> out;
  const auto g = s(m, "geometry_id");
  if (s(m, "type").value_or("") != "product" || !g) return out;
  for (const Row& d : db.all("SELECT * FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {Value(*g)})) {
    const auto mid = s(d, "manifold_id");
    out.push_back(mid ? db.one("SELECT * FROM manifold WHERE id = ?", {Value(*mid)}) : std::nullopt);
  }
  return out;
}
std::optional<std::int64_t> topology_rank(const Database& db, const Row& m) {
  const auto t = s(m, "topology_id");
  if (!t) return std::nullopt;
  const auto r = db.one("SELECT rank FROM topology WHERE id = ?", {Value(*t)});
  return r ? i(*r, "rank") : std::nullopt;
}
}  // namespace

std::vector<CellPart> parse_cell(const std::string& key) {
  if (key.empty()) throw Error(code::invalid_argument, "cell key '': empty");
  std::vector<CellPart> out;
  std::size_t at = 0;
  while (true) {
    const std::size_t comma = key.find(',', at);
    const std::string p = key.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
    const bool ok = (p.size() == 1 || (p.size() == 2 && p[1] == '*')) && p[0] >= '0' && p[0] <= '3';
    if (!ok) throw Error(code::invalid_argument, "cell key '" + key + "': '" + p + "' is not a rank 0–3 with an optional '*'");
    out.push_back({p[0] - '0', p.size() == 2});
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return out;
}

std::vector<CellPart> cell_parts(const std::string& key, std::size_t factors) {
  auto parts = parse_cell(key);
  if (key == "0" && factors > 1) return std::vector<CellPart>(factors, CellPart{0, false});
  return parts;
}

std::optional<std::int64_t> cell_count_of(const Database& db, const std::string& manifold_id, const std::string& key) {
  const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {Value(manifold_id)});
  if (!m) return std::nullopt;
  std::vector<std::optional<Row>> ends;
  if (s(*m, "type").value_or("") == "product" && s(*m, "geometry_id")) ends = factors_of(db, *m);
  else ends.push_back(*m);
  const auto parts = cell_parts(key, ends.size());
  if (ends.size() != parts.size()) return std::nullopt;
  for (const auto& e : ends) if (!e) return std::nullopt;
  std::int64_t n = 1;
  for (std::size_t k = 0; k < parts.size(); ++k) {
    const Row& e = *ends[k];
    const std::int64_t top = topology_rank(db, e).value_or(0);
    const std::int64_t rank = parts[k].dual ? top - parts[k].rank : parts[k].rank;
    const char* col = rank == 0 ? "n_vertices" : rank == 1 ? "n_edges" : rank == 2 ? "n_faces" : rank == 3 ? "n_volumes" : nullptr;
    const auto c = col ? i(e, col) : std::nullopt;
    if (!c) return std::nullopt;
    n *= *c;
  }
  return n;
}

void check_cell(const Database& db, const std::string& manifold_id, const std::string& key, const std::string& what) {
  const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {Value(manifold_id)});
  if (!m) return;   // the foreign key reports a missing manifold
  const auto top = topology_rank(db, *m);
  if (!top) return;
  const std::string name = s(*m, "name").value_or("");
  const bool product = s(*m, "type").value_or("") == "product";
  struct End { std::string name; std::int64_t rank; };
  std::vector<End> ends;
  if (product) {
    for (const auto& f : factors_of(db, *m)) {
      if (!f) continue;
      if (const auto r = topology_rank(db, *f)) ends.push_back({s(*f, "name").value_or(""), *r});
    }
  } else {
    ends.push_back({name, *top});
  }
  auto fail = [&](const std::string& m2) { throw Error(code::invalid_argument, what + ": " + m2); };
  std::vector<CellPart> parts;
  try { parts = cell_parts(key, ends.size()); } catch (const Error& e) { fail(e.what()); }
  if (product && ends.empty()) fail("cell key '" + key + "' on product '" + name + "': its factors are not known");
  if (parts.size() != ends.size()) fail("cell key '" + key + "' on '" + name + "': " + std::to_string(parts.size()) + " part(s) for " + std::to_string(ends.size()) + " factor(s)");
  for (std::size_t k = 0; k < parts.size(); ++k)
    if (parts[k].rank > ends[k].rank) fail("cell key '" + key + "' on '" + name + "': rank " + std::to_string(parts[k].rank) + " above " + ends[k].name + "'s top rank " + std::to_string(ends[k].rank));
}

}  // namespace nxr::io

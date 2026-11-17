#include "nxr/pipelines/input.h"

#include <cstring>
#include <numeric>
#include <set>

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
// An array's values as T, whatever its stored type (the rows say which; here the two kinds a surface uses).
template <class T>
std::vector<T> read_as(const nxr::io::Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  const std::size_t n = raw.size() / r.itemsize();
  std::vector<T> out(n);
  auto conv = [&](auto tag) {
    using S = decltype(tag);
    for (std::size_t i = 0; i < n; ++i) {
      S v;
      std::memcpy(&v, raw.data() + i * sizeof(S), sizeof(S));
      out[i] = static_cast<T>(v);
    }
  };
  switch (r.dtype()) {
    case DType::Float32: conv(float{}); break;
    case DType::Float64: conv(double{}); break;
    case DType::Int32: conv(std::int32_t{}); break;
    case DType::Int64: conv(std::int64_t{}); break;
    case DType::UInt32: conv(std::uint32_t{}); break;
    default: throw Error(code::unsupported, key + ": a surface array of type " + dtype_to_string(r.dtype()) + " is not read here");
  }
  return out;
}
}  // namespace

SurfaceInput read_surface(SubjectWriter& sub, const std::optional<std::string>& name, bool as_stored) {
  const std::string q = "SELECT m.*, g.positions_path AS positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.status = 'complete' AND " +
                        std::string(name ? "m.type = 'surface' AND m.name = ?" : "m.is_primary = 1");
  std::vector<Value> args{V(sub.id())};
  if (name) args.push_back(V(*name));
  const auto row = sub.db().one(q, args);
  if (!row) throw Error(code::no_such_row, sub.name() + ": no " + (name ? "surface named '" + *name + "'" : std::string("primary surface")));
  const auto faces = sub.db().one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", {V(text(*row, "topology_id"))});
  if (!faces) throw Error(code::no_such_row, text(*row, "name") + ": no faces (a surface's topology has a rank-2 cell array)");
  SurfaceInput s;
  s.manifold = *row;
  s.id = text(*row, "id");
  s.name = text(*row, "name");
  s.primary = integer(*row, "is_primary") != 0;
  if (as_stored) {
    s.vertices = read_as<double>(sub.store(), text(*row, "positions_path"));
  } else {
    // positions as the Studio's pipelines receive them: float32, then float64
    const std::vector<float> p32 = read_as<float>(sub.store(), text(*row, "positions_path"));
    s.vertices.assign(p32.begin(), p32.end());
  }
  s.faces = read_as<std::int32_t>(sub.store(), text(*faces, "path"));
  return s;
}

SurfaceInput read_surface(const SubjectPackage& pk, const std::string& name) {
  const std::string sid = text(pk.subject_row(), "id");
  const auto row = pk.db().one("SELECT m.*, g.positions_path AS positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.status = 'complete' AND m.type = 'surface' AND m.name = ?",
                               {V(sid), V(name)});
  if (!row) throw Error(code::no_such_row, pk.record().name + ": no surface named '" + name + "'");
  const auto faces = pk.db().one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", {V(text(*row, "topology_id"))});
  if (!faces) throw Error(code::no_such_row, name + ": no faces (a surface's topology has a rank-2 cell array)");
  SurfaceInput s;
  s.manifold = *row;
  s.id = text(*row, "id");
  s.name = name;
  s.primary = integer(*row, "is_primary") != 0;
  s.vertices = read_as<double>(pk.store(), text(*row, "positions_path"));
  s.faces = read_as<std::int32_t>(pk.store(), text(*faces, "path"));
  return s;
}

int connected_pieces(const std::vector<std::int32_t>& faces, std::int64_t n) {
  std::vector<std::int64_t> parent(static_cast<std::size_t>(n));
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::int64_t x) {
    while (parent[static_cast<std::size_t>(x)] != x) x = parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    return x;
  };
  std::set<std::int64_t> used;
  for (std::size_t f = 0; f + 2 < faces.size(); f += 3) {
    for (int k = 0; k < 3; ++k) used.insert(faces[f + static_cast<std::size_t>(k)]);
    for (int k = 1; k < 3; ++k) parent[static_cast<std::size_t>(find(faces[f]))] = find(faces[f + static_cast<std::size_t>(k)]);
  }
  std::set<std::int64_t> roots;
  for (std::int64_t v : used) roots.insert(find(v));
  return static_cast<int>(roots.size());
}

std::vector<std::uint8_t> medial_wall(SubjectWriter& sub, const SurfaceInput& s) {
  const auto dk = sub.db().one("SELECT d.* FROM selection d JOIN selection p ON p.id = d.member_of_id WHERE p.manifold_id = ? AND p.name = ? AND d.name = 'Desikan-Killiany' AND d.status = 'complete'",
                               {V(s.id), V(s.name + "_atlases")});
  const auto un = dk && !is_null(*dk, "dictionary_id")
                      ? sub.db().one("SELECT code FROM dictionary_entry WHERE dictionary_id = ? AND name IN ('unassigned', 'unknown') ORDER BY name", {V(text(*dk, "dictionary_id"))})
                      : std::nullopt;
  if (!dk || !un)
    throw Error(code::no_such_row, sub.name() + ": " + s.name + " carries no Desikan-Killiany atlas with its medial wall (" + s.name + "_atlases/Desikan-Killiany, its 'unassigned' code); the hemispheres are glued by anatomy only");
  const std::vector<std::int64_t> codes = read_as<std::int64_t>(sub.store(), text(*dk, "array_path"));
  if (static_cast<std::int64_t>(codes.size()) != s.n_vertices())
    throw Error(code::invalid_argument, s.name + ": its Desikan-Killiany atlas has " + std::to_string(codes.size()) + " values for " + std::to_string(s.n_vertices()) + " vertices");
  const std::int64_t wall = integer(*un, "code");
  std::vector<std::uint8_t> out(codes.size());
  for (std::size_t i = 0; i < codes.size(); ++i) out[i] = codes[i] == wall ? 1 : 0;
  return out;
}

}  // namespace nxr::pipelines

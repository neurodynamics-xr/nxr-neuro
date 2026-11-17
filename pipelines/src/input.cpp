#include "nxr/pipelines/input.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/pipelines/task.h"

#include <cmath>
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
  std::optional<Row> row;
  if (name) {
    row = surface_with_geometry(sub.db(), sub.id(), name);
    if (!row) throw Error(code::no_such_row, sub.name() + ": no surface named '" + *name + "'");
  } else {                                                                 // no name: the subject's one surface; several are the caller's choice to name (rule 19), none is a default
    const std::vector<Row> all = sub.db().all("SELECT m.name AS name FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.status = 'complete' AND m.type = 'surface' ORDER BY m.name", {V(sub.id())});
    if (all.empty()) throw Error(code::no_such_row, sub.name() + ": no surface");
    if (all.size() > 1) {
      std::string names;
      for (const auto& r : all) names += (names.empty() ? "" : ", ") + text(r, "name");
      throw Error(code::invalid_argument, sub.name() + ": several surfaces (" + names + "); name the one to use");
    }
    row = surface_with_geometry(sub.db(), sub.id(), text(all.front(), "name"));
    if (!row) throw Error(code::no_such_row, sub.name() + ": no surface");
  }
  const auto faces = topology_faces_path(sub.db(), text(*row, "topology_id"));
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
  s.faces = read_as<std::int32_t>(sub.store(), *faces);
  return s;
}

SurfaceInput read_surface(const SubjectPackage& pk, const std::string& name) {
  const std::string sid = text(pk.subject_row(), "id");
  const auto row = surface_with_geometry(pk.db(), sid, name);
  if (!row) throw Error(code::no_such_row, pk.record().name + ": no surface named '" + name + "'");
  const auto faces = topology_faces_path(pk.db(), text(*row, "topology_id"));
  if (!faces) throw Error(code::no_such_row, name + ": no faces (a surface's topology has a rank-2 cell array)");
  SurfaceInput s;
  s.manifold = *row;
  s.id = text(*row, "id");
  s.name = name;
  s.primary = integer(*row, "is_primary") != 0;
  s.vertices = read_as<double>(pk.store(), text(*row, "positions_path"));
  s.faces = read_as<std::int32_t>(pk.store(), *faces);
  return s;
}

std::optional<GlueMap> glue_map_of(SubjectWriter& sub, const std::string& manifold_id) {
  const auto node = task::Subject::of(sub).by_id(task::Cls::manifold, manifold_id);
  if (!node || !node->producer.is_object() || !node->producer.contains("glue")) return std::nullopt;
  const Json& g = node->producer["glue"];
  if (!g.is_object() || !g.contains("original") || !g.contains("map_path") || !g["original"].is_string() || !g["map_path"].is_string())
    throw Error(code::invalid_argument, node->name + ": its glue record names the original manifold and the map's path ({glue: {original, map_path}})");
  GlueMap m;
  m.glued_id = manifold_id;
  m.original_id = g["original"].get<std::string>();
  m.original_of = read_as<std::int32_t>(sub.store(), g["map_path"].get<std::string>());
  return m;
}

std::vector<double> restrict_to_original(const std::vector<double>& glued, std::int64_t comps, const GlueMap& map, std::int64_t n_original) {
  if (comps < 1 || static_cast<std::int64_t>(glued.size()) != map.n_glued() * comps)
    throw Error(code::invalid_argument, "restricting a result to the original surface: " + std::to_string(glued.size()) + " values for " + std::to_string(map.n_glued()) + " glued vertices of " + std::to_string(comps) + " components");
  std::vector<double> out(static_cast<std::size_t>(n_original * comps), std::nan(""));
  for (std::int64_t g = 0; g < map.n_glued(); ++g) {
    const std::int32_t o = map.original_of[static_cast<std::size_t>(g)];
    if (o < 0) continue;                                                   // the bridge: not cortex, dropped
    if (o >= n_original) throw Error(code::invalid_argument, "the glue map names original vertex " + std::to_string(o) + " of a surface of " + std::to_string(n_original) + " vertices");
    for (std::int64_t c = 0; c < comps; ++c) out[static_cast<std::size_t>(o * comps + c)] = glued[static_cast<std::size_t>(g * comps + c)];
  }
  return out;
}

std::vector<double> carry_to_glued(const std::vector<double>& original, std::int64_t comps, const GlueMap& map, const std::vector<std::vector<std::int32_t>>& sources) {
  const std::int64_t n_original = comps > 0 ? static_cast<std::int64_t>(original.size()) / comps : 0;
  std::vector<double> out(static_cast<std::size_t>(map.n_glued() * comps), std::nan(""));
  for (std::int64_t g = 0; g < map.n_glued(); ++g) {
    const std::int32_t o = map.original_of[static_cast<std::size_t>(g)];
    if (o >= 0 && o < n_original) { for (std::int64_t c = 0; c < comps; ++c) out[static_cast<std::size_t>(g * comps + c)] = original[static_cast<std::size_t>(o * comps + c)]; continue; }
    if (o >= 0 || static_cast<std::size_t>(g) >= sources.size() || sources[static_cast<std::size_t>(g)].empty()) continue;   // a bridge vertex interpolates from its sources, by their mean
    for (std::int64_t c = 0; c < comps; ++c) {
      double sum = 0;
      for (const std::int32_t src : sources[static_cast<std::size_t>(g)]) sum += original[static_cast<std::size_t>(src * comps + c)];
      out[static_cast<std::size_t>(g * comps + c)] = sum / static_cast<double>(sources[static_cast<std::size_t>(g)].size());
    }
  }
  return out;
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
  const auto atlases = nxr::io::selection::named_row(sub.db(), sub.id(), s.name + "_atlases", std::nullopt, s.id);
  const auto dk = atlases ? nxr::io::selection::member_row(sub.db(), text(*atlases, "id"), "Desikan-Killiany") : std::nullopt;
  const auto un = dk && !is_null(*dk, "dictionary_id") ? nxr::io::selection::lookup_code_of_any(sub.db(), text(*dk, "dictionary_id"), {"unassigned", "unknown"}) : std::nullopt;
  if (!dk || !un)
    throw Error(code::no_such_row, sub.name() + ": " + s.name + " carries no Desikan-Killiany atlas with its medial wall (" + s.name + "_atlases/Desikan-Killiany, its 'unassigned' code); the hemispheres are glued by anatomy only");
  const std::vector<std::int64_t> codes = nxr::io::selection::read_element_members(sub.db(), sub.store(), text(*dk, "id")).element_labels();   // the arrays hold member numbers; the wall is a label
  if (static_cast<std::int64_t>(codes.size()) != s.n_vertices())
    throw Error(code::invalid_argument, s.name + ": its Desikan-Killiany atlas has " + std::to_string(codes.size()) + " values for " + std::to_string(s.n_vertices()) + " vertices");
  const std::int64_t wall = *un;
  std::vector<std::uint8_t> out(codes.size());
  for (std::size_t i = 0; i < codes.size(); ++i) out[i] = codes[i] == wall ? 1 : 0;
  return out;
}

}  // namespace nxr::pipelines

#include "nxr/pipelines/input.h"
#include "nxr/io/subject_api.h"
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
// A hemisphere is a COMPONENT of the cortex: its own manifold over the union's arrays (io's surface writer), the vertices [from_index, from_index + n_vertices)
// and the faces lying wholly inside them. Read as a surface it is that view, re-indexed from 0; the union's arrays are never handed out under its name.
void apply_component_view(SurfaceInput& s, const Row& row) {
  if (is_null(row, "component_of_id") || is_null(row, "from_index")) return;
  const std::int64_t lo = integer(row, "from_index"), n = integer(row, "n_vertices");
  const std::int64_t total = static_cast<std::int64_t>(s.vertices.size() / 3);
  if (lo < 0 || n <= 0 || lo + n > total)
    throw Error(code::invalid_argument, s.name + ": its vertices [" + std::to_string(lo) + ", " + std::to_string(lo + n) + ") lie outside the " + std::to_string(total) + " of the surface it is a component of");
  std::vector<double> v(s.vertices.begin() + 3 * lo, s.vertices.begin() + 3 * (lo + n));
  std::vector<std::int32_t> f;
  for (std::size_t i = 0; i + 2 < s.faces.size(); i += 3) {
    bool in = true;
    for (int c = 0; c < 3; ++c) in = in && s.faces[i + c] >= lo && s.faces[i + c] < lo + n;
    if (in)
      for (int c = 0; c < 3; ++c) f.push_back(static_cast<std::int32_t>(s.faces[i + c] - lo));
  }
  s.vertices = std::move(v);
  s.faces = std::move(f);
}

}  // namespace

std::string hodge_package_stem(const SurfaceInput& S) {
  if (connected_pieces(S.faces, S.n_vertices()) == 2) return "hodgeDirac__" + S.name + "_glued";
  return S.primary ? std::string("hodgeDirac") : "hodgeDirac__" + S.name;
}

SurfaceInput read_surface(SubjectWriter& sub, const std::optional<std::string>& name, bool as_stored) {
  std::optional<Row> row;
  if (name) {
    row = surface_with_geometry(sub.db(), sub.id(), name);
    if (!row) throw Error(code::no_such_row, sub.name() + ": no surface named '" + *name + "'");
  } else {                                                                 // no name: the subject's one surface; several are the caller's choice to name, none is a default
    const std::vector<std::string> all = surface_names(sub);
    if (all.empty()) throw Error(code::no_such_row, sub.name() + ": no surface");
    if (all.size() > 1) {
      std::string names;
      for (const auto& r : all) names += (names.empty() ? "" : ", ") + r;
      throw Error(code::invalid_argument, sub.name() + ": several surfaces (" + names + "); name the one to use");
    }
    row = surface_with_geometry(sub.db(), sub.id(), all.front());
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
  apply_component_view(s, *row);
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
  apply_component_view(s, *row);
  return s;
}

StoredLabelling stored_labelling(SubjectWriter& sub, const std::string& selection_id) {
  const task::Subject ts = task::Subject::of(sub);
  const auto node = ts.by_id(task::Cls::selection, selection_id);
  if (!node) throw Error(code::no_such_row, "the labelling " + selection_id + " is not stored");
  StoredLabelling l;
  l.id = selection_id; l.name = node->name;
  l.labels = ts.element_members(selection_id).element_labels();
  if (node->dictionary_id) l.entries = ts.dictionary_entries(*node->dictionary_id);
  return l;
}

namespace {
// the connected pieces of the vertices labelled `code`, joined by the faces all of whose vertices carry it (a member vertex in no such face is a piece of its own)
int member_pieces(const std::vector<std::int32_t>& faces, const std::vector<std::int64_t>& labels, std::int64_t code) {
  const std::size_t n = labels.size();
  std::vector<std::size_t> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::size_t x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
  for (std::size_t f = 0; f + 2 < faces.size(); f += 3) {
    const auto a = static_cast<std::size_t>(faces[f]), b = static_cast<std::size_t>(faces[f + 1]), c = static_cast<std::size_t>(faces[f + 2]);
    if (a >= n || b >= n || c >= n || labels[a] != code || labels[b] != code || labels[c] != code) continue;
    parent[find(a)] = find(b);
    parent[find(b)] = find(c);
  }
  std::set<std::size_t> roots;
  for (std::size_t v = 0; v < n; ++v) if (labels[v] == code) roots.insert(find(v));
  return static_cast<int>(roots.size());
}
}  // namespace

std::string hemispheres_disagree(const std::vector<std::int32_t>& faces, const StoredLabelling& H, const std::string& left, const std::string& right) {
  const auto L = H.code_of(left), R = H.code_of(right), U = H.code_of("unassigned");
  if (!L || !R) return H.name + " has no entry '" + (L ? right : left) + "' (the hemispheres are named by --left and --right)";
  for (const auto& [code, nm] : {std::pair<std::int64_t, std::string>{*L, left}, {*R, right}}) {
    const int pc = member_pieces(faces, H.labels, code);
    if (pc != 1) return H.name + ": its '" + nm + "' is " + std::to_string(pc) + " connected pieces, not one hemisphere";
  }
  std::int64_t outside = 0;
  for (const auto l : H.labels) if (l != *L && l != *R && !(U && l == *U)) ++outside;
  if (outside) return H.name + ": " + std::to_string(outside) + " vertices are in neither '" + left + "' nor '" + right + "' nor its own 'unassigned'";
  for (std::size_t f = 0; f + 2 < faces.size(); f += 3) {
    bool l = false, r = false;
    for (int k = 0; k < 3; ++k) { const auto x = H.labels[static_cast<std::size_t>(faces[f + static_cast<std::size_t>(k)])]; l = l || x == *L; r = r || x == *R; }
    if (l && r) return H.name + ": face " + std::to_string(f / 3) + " has vertices in both hemispheres: the labelling does not follow the surface's pieces";
  }
  return {};
}

std::optional<GlueMap> glue_map_of(SubjectWriter& sub, const std::string& manifold_id) {
  const task::Subject ts = task::Subject::of(sub);
  const auto node = ts.by_id(task::Cls::manifold, manifold_id);
  if (!node || !node->producer.is_object() || !node->producer.contains("glue")) return std::nullopt;
  const Json& g = node->producer["glue"];
  if (!g.is_object() || !g.contains("original") || !g["original"].is_string())
    throw Error(code::invalid_argument, node->name + ": its glue record names the original manifold ({glue: {original}})");
  GlueMap m;
  m.glued_id = manifold_id;
  m.original_id = g["original"].get<std::string>();
  // the labelling of the glued vertices (its non-bridge vertices first, then the bridge) and the original's medial wall: the glue removes exactly the wall, so the i-th non-bridge glued vertex
  // is the i-th vertex of the original off the wall. A `_kept` selection still stored marks a glue of the earlier form, whose cut may have removed more: `migrate glue` checks and converts it.
  task::Want wk{.cls = task::Cls::selection}; wk.name = node->name + "_kept"; wk.manifold_id = m.original_id;
  task::Want wh{.cls = task::Cls::selection}; wh.name = node->name + "_hemispheres"; wh.manifold_id = manifold_id;
  const auto hn = ts.find(wh);
  if (!hn || ts.find(wk))
    throw Error(code::invalid_argument, node->name + ": a glued cortex in an earlier form (" + (hn ? "its kept vertices " + wk.name.value_or("") + " stored beside it" : "no hemispheres labelling " + wh.name.value_or("")) +
                                        "): it is converted in place by `nxr-neuro migrate glue`, which refuses a glue that removed more than the medial wall");
  const auto original = ts.by_id(task::Cls::manifold, m.original_id);
  if (!original || !original->n_vertices) throw Error(code::invalid_argument, node->name + ": its original manifold " + m.original_id + " is not stored");
  const std::vector<std::uint8_t> wall = medial_wall_of(sub, m.original_id, original->name, *original->n_vertices);
  const StoredLabelling h = stored_labelling(sub, hn->id);
  const auto bridge = h.code_of("bridge");
  if (!bridge) throw Error(code::invalid_argument, node->name + ": its hemispheres labelling has no `bridge` entry");
  std::vector<std::int32_t> off_wall;
  for (std::size_t v = 0; v < wall.size(); ++v) if (!wall[v]) off_wall.push_back(static_cast<std::int32_t>(v));
  const std::int64_t n = static_cast<std::int64_t>(h.labels.size()), k = static_cast<std::int64_t>(off_wall.size());
  std::int64_t cortex = 0;
  for (const auto l : h.labels) cortex += l != *bridge;
  if (cortex != k)
    throw Error(code::invalid_argument, node->name + ": its " + std::to_string(cortex) + " cortex vertices are not the " + std::to_string(k) + " vertices of " + original->name +
                                        " off the medial wall: the stored glue removed more than the medial wall; re-run glue");
  for (std::int64_t i = 0; i < n; ++i)
    if ((h.labels[static_cast<std::size_t>(i)] == *bridge) != (i >= k))
      throw Error(code::invalid_argument, node->name + ": glued vertex " + std::to_string(i) + " breaks the glue's order (the " + std::to_string(k) + " vertices off the medial wall first, then the bridge)");
  m.original_of.assign(static_cast<std::size_t>(n), -1);
  for (std::int64_t i = 0; i < k; ++i) m.original_of[static_cast<std::size_t>(i)] = off_wall[static_cast<std::size_t>(i)];
  return m;
}

std::vector<std::uint8_t> carried_onto_glued(const std::vector<std::uint8_t>& rows, std::int64_t n_rows, const GlueMap& map, std::int64_t n_original) {
  if (n_original < 1 || n_rows < 1 || n_rows % n_original != 0 || rows.size() % static_cast<std::size_t>(n_rows) != 0)
    throw Error(code::invalid_argument, "carrying onto the glued cortex: " + std::to_string(n_rows) + " rows are not a whole number of rows for each of the " + std::to_string(n_original) + " original vertices");
  const std::int64_t k = n_rows / n_original, n_glued = map.n_glued();
  const std::size_t row = rows.size() / static_cast<std::size_t>(n_rows), block = row * static_cast<std::size_t>(k);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(n_glued) * block, 0);
  for (std::int64_t g = 0; g < n_glued; ++g) {
    const std::int32_t o = map.original_of[static_cast<std::size_t>(g)];
    if (o < 0) continue;                                                   // the bridge: no vertex of the original, its rows zero
    if (o >= n_original) throw Error(code::invalid_argument, "the glue map names original vertex " + std::to_string(o) + " of a surface of " + std::to_string(n_original) + " vertices");
    std::memcpy(out.data() + static_cast<std::size_t>(g) * block, rows.data() + static_cast<std::size_t>(o) * block, block);
  }
  return out;
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

std::vector<std::uint8_t> medial_wall(SubjectWriter& sub, const SurfaceInput& s) { return medial_wall_of(sub, s.id, s.name, s.n_vertices()); }

std::vector<std::uint8_t> medial_wall_of(SubjectWriter& sub, const std::string& surface_id, const std::string& surface_name, std::int64_t n_vertices) {
  const auto atlases = nxr::io::selection::named_row(sub.db(), sub.id(), surface_name + "_atlases", std::nullopt, surface_id);
  const auto dk = atlases ? nxr::io::selection::member_row(sub.db(), text(*atlases, "id"), "Desikan-Killiany") : std::nullopt;
  const auto un = dk && !is_null(*dk, "dictionary_id") ? nxr::io::selection::lookup_code_of_any(sub.db(), text(*dk, "dictionary_id"), {"unassigned", "unknown"}) : std::nullopt;
  if (!dk || !un)
    throw Error(code::no_such_row, sub.name() + ": " + surface_name + " carries no Desikan-Killiany atlas with its medial wall (" + surface_name + "_atlases/Desikan-Killiany, its 'unassigned' code); the hemispheres are glued by anatomy only");
  const std::vector<std::int64_t> codes = nxr::io::selection::read_element_members(sub.db(), sub.store(), text(*dk, "id")).element_labels();   // the arrays hold member numbers; the wall is a label
  if (static_cast<std::int64_t>(codes.size()) != n_vertices)
    throw Error(code::invalid_argument, surface_name + ": its Desikan-Killiany atlas has " + std::to_string(codes.size()) + " values for " + std::to_string(n_vertices) + " vertices");
  const std::int64_t wall = *un;
  std::vector<std::uint8_t> out(codes.size());
  for (std::size_t i = 0; i < codes.size(); ++i) out[i] = codes[i] == wall ? 1 : 0;
  return out;
}

Json medial_wall_record(SubjectWriter& sub, const std::string& surface_id, const std::string& surface_name) {
  const auto atlases = nxr::io::selection::named_row(sub.db(), sub.id(), surface_name + "_atlases", std::nullopt, surface_id);
  const auto dk = atlases ? nxr::io::selection::member_row(sub.db(), text(*atlases, "id"), "Desikan-Killiany") : std::nullopt;
  if (!dk || is_null(*dk, "dictionary_id"))
    throw Error(code::no_such_row, sub.name() + ": " + surface_name + " carries no Desikan-Killiany atlas with its medial wall (" + surface_name + "_atlases/Desikan-Killiany)");
  for (const char* entry : {"unassigned", "unknown"})                     // the order medial_wall_of reads them in
    if (const auto c = nxr::io::selection::lookup_code_of_any(sub.db(), text(*dk, "dictionary_id"), {entry}))
      return Json{{"labelling", surface_name + "_atlases"}, {"member", "Desikan-Killiany"}, {"id", text(*dk, "id")}, {"entry", entry}, {"code", *c}};
  throw Error(code::no_such_row, sub.name() + ": the Desikan-Killiany atlas of " + surface_name + " has no 'unassigned' entry (its medial wall)");
}

Json glue_relation(const std::string& original_id, const std::string& original_name, const Json& hemispheres, const Json& wall, std::int64_t n_cortex, std::int64_t n_glued) {
  const std::string w = wall.value("labelling", std::string()) + "/" + wall.value("member", std::string());
  const std::string rule = "glued vertex i < " + std::to_string(n_cortex) + " (labelled other than `bridge` in the hemispheres labelling) is the i-th vertex, in ascending index, of " + original_name +
                           " outside the medial wall (the vertices " + w + " labels other than '" + wall.value("entry", std::string()) + "'); glued vertices " + std::to_string(n_cortex) + " to " +
                           std::to_string(n_glued - 1) + " (labelled `bridge`) are no vertex of " + original_name + ". A node whose rows are " + original_name + "'s vertices is carried onto the glued cortex "
                           "by selecting its rows in that order, a bridge row zero (`read operator|field --onto`)";
  return Json{{"original", original_id}, {"original_name", original_name}, {"form", "labelling"}, {"hemispheres", hemispheres}, {"bridge_entry", "bridge"}, {"medial_wall", wall},
              {"cortex_vertices", n_cortex}, {"bridge_vertices", n_glued - n_cortex}, {"rule", rule}};
}

}  // namespace nxr::pipelines

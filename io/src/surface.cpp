// nxr.convert.winding, nxr.convert.mesh_health and nxr.convert.surface.add_surface (a mesh file's way in).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <set>

#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::io {

namespace {
using V3 = std::array<double, 3>;
V3 vert(const std::vector<double>& v, std::int64_t i) { return {v[3 * i], v[3 * i + 1], v[3 * i + 2]}; }
V3 sub3(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

struct Dsu {
  std::vector<std::int64_t> p;
  explicit Dsu(std::int64_t n) : p(static_cast<std::size_t>(n)) { std::iota(p.begin(), p.end(), 0); }
  std::int64_t find(std::int64_t x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  void join(std::int64_t a, std::int64_t b) { a = find(a); b = find(b); if (a != b) p[a] = b; }
};
}  // namespace

std::vector<std::int32_t> canonicalize_winding(const std::vector<double>& v, const std::vector<std::int32_t>& faces, std::vector<std::int64_t>* flipped) {
  std::vector<std::int32_t> f = faces;
  const std::int64_t nv = static_cast<std::int64_t>(v.size() / 3), nf = static_cast<std::int64_t>(f.size() / 3);
  Dsu d(nv);
  for (std::int64_t i = 0; i < nf; ++i) { d.join(f[3 * i], f[3 * i + 1]); d.join(f[3 * i + 1], f[3 * i + 2]); }
  // component labels in order of first appearance by vertex index, as scipy's connected_components numbers them
  std::vector<std::int64_t> label(static_cast<std::size_t>(nv), -1);
  std::int64_t next = 0;
  std::vector<std::int64_t> comp(static_cast<std::size_t>(nv));
  for (std::int64_t i = 0; i < nv; ++i) {
    const std::int64_t r = d.find(i);
    if (label[r] < 0) label[r] = next++;
    comp[i] = label[r];
  }
  std::vector<double> vol(static_cast<std::size_t>(next), 0.0);
  std::vector<std::vector<std::int64_t>> of(static_cast<std::size_t>(next));
  for (std::int64_t i = 0; i < nf; ++i) {
    const std::int64_t c = comp[f[3 * i]];
    of[c].push_back(i);
    vol[c] += dot(vert(v, f[3 * i]), cross(vert(v, f[3 * i + 1]), vert(v, f[3 * i + 2])));
  }
  for (std::int64_t c = 0; c < next; ++c) {
    if (of[c].empty()) continue;
    const double volume = vol[c] / 6.0;
    if (std::fabs(volume) < 1.1920928955078125e-07) continue;  // open / non-watertight: no orientation fact to act on
    if (volume < 0) {
      for (auto i : of[c]) std::swap(f[3 * i + 1], f[3 * i + 2]);
      if (flipped) flipped->push_back(c);
    }
  }
  return f;
}

bool MeshHealth::fatal() const {
  return non_manifold_edges || duplicate_faces || degenerate_faces || isolated_vertices || inconsistent_winding_edges;
}

Json MeshHealth::as_attrs() const {
  return Json{{"n_vertices", n_vertices}, {"n_faces", n_faces}, {"non_manifold_edges", non_manifold_edges}, {"duplicate_faces", duplicate_faces},
              {"degenerate_faces", degenerate_faces}, {"isolated_vertices", isolated_vertices}, {"boundary_edges", boundary_edges},
              {"inconsistent_winding_edges", inconsistent_winding_edges}};
}

std::string MeshHealth::summary() const {
  std::vector<std::string> bits;
  if (non_manifold_edges) bits.push_back(std::to_string(non_manifold_edges) + " non-manifold edge(s) (>2 incident faces)");
  if (inconsistent_winding_edges) bits.push_back(std::to_string(inconsistent_winding_edges) + " edge(s) with inconsistent winding (two faces traverse the same directed edge)");
  if (duplicate_faces) bits.push_back(std::to_string(duplicate_faces) + " duplicate face(s)");
  if (degenerate_faces) bits.push_back(std::to_string(degenerate_faces) + " degenerate face(s) (zero area)");
  if (isolated_vertices) bits.push_back(std::to_string(isolated_vertices) + " isolated vertex/vertices");
  if (boundary_edges) bits.push_back(std::to_string(boundary_edges) + " boundary edge(s)");
  std::string out;
  for (const auto& b : bits) out += (out.empty() ? "" : "; ") + b;
  return out.empty() ? "clean" : out;
}

MeshHealth check_mesh(const std::vector<double>& v, const std::vector<std::int32_t>& f) {
  MeshHealth h;
  const std::int64_t nv = static_cast<std::int64_t>(v.size() / 3), nf = static_cast<std::int64_t>(f.size() / 3);
  h.n_vertices = nv;
  h.n_faces = nf;
  using T3 = std::array<std::int32_t, 3>;
  std::vector<T3> sorted;
  for (std::int64_t i = 0; i < nf; ++i) { T3 t{f[3 * i], f[3 * i + 1], f[3 * i + 2]}; std::sort(t.begin(), t.end()); sorted.push_back(t); }
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 1; i < sorted.size(); ++i) if (sorted[i] == sorted[i - 1]) ++h.duplicate_faces;  // sum(counts - 1)
  for (std::int64_t i = 0; i < nf; ++i) {
    const V3 a = sub3(vert(v, f[3 * i + 1]), vert(v, f[3 * i])), b = sub3(vert(v, f[3 * i + 2]), vert(v, f[3 * i]));
    const V3 c = cross(a, b);
    if (0.5 * std::sqrt(dot(c, c)) <= 1e-14) ++h.degenerate_faces;
  }
  using E = std::array<std::int32_t, 2>;
  std::vector<E> und, dir;
  for (std::int64_t i = 0; i < nf; ++i)
    for (int k = 0; k < 3; ++k) {
      const std::int32_t a = f[3 * i + k], b = f[3 * i + (k + 1) % 3];
      dir.push_back({a, b});
      und.push_back({std::min(a, b), std::max(a, b)});
    }
  std::sort(und.begin(), und.end());
  for (std::size_t i = 0; i < und.size();) {
    std::size_t j = i;
    while (j < und.size() && und[j] == und[i]) ++j;
    if (j - i > 2) ++h.non_manifold_edges;
    if (j - i == 1) ++h.boundary_edges;
    i = j;
  }
  std::sort(dir.begin(), dir.end());
  for (std::size_t i = 0; i < dir.size();) {
    std::size_t j = i;
    while (j < dir.size() && dir[j] == dir[i]) ++j;
    if (j - i > 1) ++h.inconsistent_winding_edges;
    i = j;
  }
  std::vector<bool> used(static_cast<std::size_t>(nv), false);
  for (auto i : f) used[i] = true;
  for (bool u : used) if (!u) ++h.isolated_vertices;
  return h;
}

namespace {
bool has_primary(SubjectWriter& sub) {
  return integer(*sub.db().one("SELECT count(*) AS n FROM manifold WHERE subject_id = ? AND type = 'surface' AND is_primary = 1", {sub.id()}), "n") > 0;
}

// WARN about a fatal surface and let the import proceed (register D56); `NXR_STRICT_MESH=1` restores the refusal for a primary surface
void require_healthy(const MeshHealth& h, const std::string& surface, bool primary) {
  if (!h.fatal()) return;
  const std::string detail = h.summary();
  const char* strict = std::getenv("NXR_STRICT_MESH");
  if (!primary || !(strict && std::string(strict) == "1")) {
    std::cerr << "UserWarning: " << (primary ? "primary" : "display") << " surface '" << surface << "' is not a clean manifold (" << detail
              << ") — exported anyway; operators built from this mesh may be wrong until it is repaired upstream.\n";
    return;
  }
  throw Error(code::invalid_argument, "primary surface '" + surface + "' is not a clean manifold: " + detail +
                                          ". geometry-central refuses a non-manifold mesh when it builds the halfedge structure and ABORTS THE PROCESS (SIGABRT) rather than raising, "
                                          "so this store would abort any consumer that builds its halfedge mesh. Repair the surface in Brainstorm (remove the offending faces, then any "
                                          "vertex left isolated, renumbering faces and scouts) and recompute the head model and inverse on the repaired mesh; the converter mirrors "
                                          "anatomy and does not transform it.");
}
}  // namespace

SurfaceInfo add_surface(SubjectWriter& sub, const SurfaceOptions& opt, const std::vector<double>& vertices, const std::vector<std::int32_t>& faces_in) {
  const std::string& name = opt.name;
  const std::string folder = folder_for("surface", name);
  const std::string path = folder + "/" + name;
  for (const char* table : {"manifold", "field", "operator", "selection"})
    if (sub.find(table, path)) throw Error(code::invalid_argument, name + ": a " + table + " already occupies " + path + " — remove it first");
  std::vector<std::int64_t> flipped;
  const std::vector<std::int32_t> faces = canonicalize_winding(vertices, faces_in, &flipped);
  const MeshHealth health = check_mesh(vertices, faces);  // BEFORE anything is written
  require_healthy(health, name, opt.primary);
  const std::int64_t nv = static_cast<std::int64_t>(vertices.size() / 3), nf = static_cast<std::int64_t>(faces.size() / 3);
  const std::string stamp = now_utc(false);

  ManifoldSpec m;
  m.name = name;
  m.path = path;
  m.type = "surface";
  m.n_vertices = nv;
  m.topology = {"simplicial", 2, opt.hemispheres.empty() ? 1 : static_cast<int>(opt.hemispheres.size()), std::string("ccw-outward"), {{2, path + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = opt.cs_key;
  g.positions_path = path + "/vertices";
  m.geometry = g;
  Json producer = Json::object();
  producer["mesh_health"] = health.as_attrs();
  if (!flipped.empty()) producer["flipped"] = flipped;
  if (opt.producer) for (auto it = opt.producer->begin(); it != opt.producer->end(); ++it) producer[it.key()] = it.value();
  const bool primary = opt.primary && folder == folder_for("surface", "cortex") && !has_primary(sub);
  m.cols.set("n_faces", nf).set("is_primary", primary ? 1 : 0);
  if (opt.comment) m.cols.set("comment", *opt.comment); else m.cols.null("comment");
  if (opt.source) m.cols.set("source", *opt.source); else m.cols.null("source");
  m.cols.json("producer_json", producer);
  m.created_utc = stamp;
  m.populate = [&](Store&, const std::string&) {
    sub.write_array(path + "/vertices", view(vertices, {nv, 3}));
    sub.write_array(path + "/faces", view(faces, {nf, 3}));
  };
  SurfaceInfo info;
  info.id = sub.manifold(std::move(m));
  const std::string topology_id = text(*sub.db().read("manifold", info.id), "topology_id");

  // THE HEMISPHERES: components of the cortex, each its own simplicial manifold over the union's arrays
  for (std::size_t k = 0; k < opt.hemispheres.size(); ++k) {
    const Hemisphere& h = opt.hemispheres[k];
    std::int64_t inside = 0;
    for (std::int64_t i = 0; i < nf; ++i) {
      bool in = true;
      for (int c = 0; c < 3; ++c) in = in && faces[3 * i + c] >= h.first && faces[3 * i + c] < h.first + h.count;
      inside += in ? 1 : 0;
    }
    ManifoldSpec hm;
    hm.name = name + "_" + std::to_string(k + 1);
    hm.path = path + "_" + std::to_string(k + 1);
    hm.type = "surface";
    hm.n_vertices = h.count;
    hm.topology = {"simplicial", 2, 1, std::nullopt, {{2, path + "/faces"}}};
    GeometrySpec hg;
    hg.form = "stored";
    hg.cs_key = opt.cs_key;
    hg.positions_path = path + "/vertices";
    hm.geometry = hg;
    hm.cols.set("n_faces", inside).set("label", h.label).set("component_of_id", info.id).set("component_ordinal", static_cast<std::int64_t>(k)).set("from_index", h.first);
    sub.manifold(std::move(hm));
  }

  // THE SAME MESH ON A SPHERE: a second manifold (different positions are a different metric) of the SAME topology
  const bool has_reg = opt.sphere_cols > 0 && !opt.sphere.empty() && static_cast<std::int64_t>(opt.sphere.size()) == nv * opt.sphere_cols;
  if (has_reg) {
    const std::string sp = path + "_sphere";
    ManifoldSpec sm;
    sm.name = name + "_sphere";
    sm.path = sp;
    sm.type = "surface";
    sm.n_vertices = nv;
    sm.topology_id = topology_id;
    GeometrySpec sg;
    sg.form = "stored";
    sg.cs_key = "freesurfer-sphere";
    sg.positions_path = sp + "/vertices";
    sm.geometry = sg;
    sm.cols.set("n_faces", nf).set("source", opt.sphere_source);
    sm.populate = [&, sp](Store&, const std::string&) {
      sub.write_array(sp + "/vertices", view(opt.sphere, {nv, opt.sphere_cols}));
      sub.write_array(sp + "/faces", view(faces, {nf, 3}));
    };
    sub.manifold(std::move(sm));
  }

  // THE PARCELLATIONS ARE ONE SET (O9): each atlas a member partition, its codes the app-wide dictionary's
  if (!opt.atlases.empty()) {
    const std::string set_path = path + "_atlases";
    std::string families;
    for (std::size_t k = 0; k < opt.atlases.size(); ++k) families += (k ? ", " : "") + opt.atlases[k].family;
    SelectionSpec set;
    set.name = set_path.substr(set_path.rfind('/') + 1);
    set.path = set_path;
    set.type = "set";
    set.manifold_id = info.id;
    set.cols.set("n_members", static_cast<std::int64_t>(opt.atlases.size()))
        .set("description", opt.atlases_description ? *opt.atlases_description : "the cortical parcellations \xe2\x80\x94 " + families);
    const std::string set_id = sub.selection(std::move(set));
    for (std::size_t k = 0; k < opt.atlases.size(); ++k) {
      const AtlasMember& a = opt.atlases[k];
      DictionarySpec d;
      d.name = a.family;
      d.scope = "app";
      d.names = a.levels;
      for (const auto& c : a.colors) d.colors.push_back(c);
      const DictionaryResult dr = sub.dictionary(d);
      std::vector<std::int32_t> codes(a.codes.size());
      for (std::size_t i = 0; i < codes.size(); ++i) {
        const auto lvl = static_cast<std::size_t>(a.codes[i]);
        codes[i] = dr.codes.empty() ? a.codes[i] : static_cast<std::int32_t>(dr.codes.at(lvl));
      }
      SelectionSpec ms;
      ms.name = a.family;
      ms.path = set_path + "/" + a.family;
      ms.type = "set";
      ms.manifold_id = info.id;
      ms.data = view(codes, {static_cast<std::int64_t>(codes.size())});
      ms.cols.set("dictionary_id", dr.id).set("n_elements", static_cast<std::int64_t>(codes.size()))
          .set("description", a.family + " \xe2\x80\x94 one level per element").set("member_of_id", set_id).set("member_ordinal", static_cast<std::int64_t>(k));
      sub.selection(std::move(ms));
    }
  }

  info.name = name;
  info.folder = folder;
  info.n_vertices = nv;
  info.n_faces = nf;
  info.registration = has_reg;
  info.hemispheres = static_cast<std::int64_t>(opt.hemispheres.size());
  info.atlases = static_cast<std::int64_t>(opt.atlases.size());
  info.flipped = flipped;
  info.mesh_health = health.as_attrs();
  return info;
}

}  // namespace nxr::io

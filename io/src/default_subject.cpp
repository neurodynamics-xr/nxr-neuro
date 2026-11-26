// nxr-io — the dataset's default subject (see default_subject.h).
#include "nxr/io/default_subject.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <map>
#include <set>

#include "nxr/io/surface.h"

namespace nxr::io {

namespace fs = std::filesystem;

namespace {

const char* const kSurfaces[] = {"pial", "white", "inflated"};   // nxr's SURFACES; the pial first: the primary
// nxr's ANNOTS: the atlas family and its FreeSurfer annotation (`label/{lh,rh}.<annot>.annot`)
const std::pair<const char*, const char*> kAnnots[] = {{"Desikan-Killiany", "aparc"}, {"Destrieux", "aparc.a2009s"}};

struct Both { std::vector<double> pos; std::vector<std::int32_t> faces; std::int64_t n_lh = 0; };

// lh then rh as one surface, in millimetres as stored (nxr's _read_hemis + _combined)
Both read_both(const fs::path& fs_dir, const std::string& surface) {
  Both b;
  const Mesh lh = read_freesurfer_surface(fs_dir / "surf" / ("lh." + surface));
  const Mesh rh = read_freesurfer_surface(fs_dir / "surf" / ("rh." + surface));
  b.n_lh = lh.n_vertices();
  b.pos = lh.vertices;
  b.pos.insert(b.pos.end(), rh.vertices.begin(), rh.vertices.end());
  b.faces = lh.faces;
  for (auto f : rh.faces) b.faces.push_back(f + static_cast<std::int32_t>(b.n_lh));
  return b;
}

// THE PARCELLATIONS of the pial (nxr's DefaultSubjectWriter.parcellations): Structures (1 left, 2 right), then each annotation pair as a partition whose
// levels are `unassigned` (0, black) and "<region> L|R" with FreeSurfer's colour; `unknown`, `Medial_wall`, `corpuscallosum` and regions with no vertex
// are left unassigned. A family whose annotation is missing for either hemisphere is left out.
std::vector<AtlasMember> parcellations(const fs::path& fs_dir, std::int64_t n_lh, std::int64_t n) {
  std::vector<AtlasMember> out;
  AtlasMember st;
  st.family = "Structures";
  st.codes.assign(static_cast<std::size_t>(n), 2);
  std::fill(st.codes.begin(), st.codes.begin() + n_lh, 1);
  st.levels = {"unassigned", "Cortex L", "Cortex R"};
  st.colors = {{0, 0, 0}, {200, 120, 120}, {120, 120, 200}};
  out.push_back(std::move(st));
  for (const auto& [family, annot] : kAnnots) {
    AtlasMember m;
    m.family = family;
    m.codes.assign(static_cast<std::size_t>(n), 0);
    m.levels = {"unassigned"};
    m.colors = {{0, 0, 0}};
    bool complete = true;
    for (const auto& [h, side, off] : {std::tuple<const char*, const char*, std::int64_t>{"lh", "L", 0}, {"rh", "R", n_lh}}) {
      const fs::path f = fs_dir / "label" / (std::string(h) + "." + annot + ".annot");
      if (!fs::is_regular_file(f)) { complete = false; break; }
      const FsAnnot a = read_freesurfer_annot(f);
      const std::int64_t nh = std::string(h) == "lh" ? n_lh : n - n_lh;
      if (static_cast<std::int64_t>(a.labels.size()) != nh)
        throw Error(code::invalid_argument, f.filename().string() + ": " + std::to_string(a.labels.size()) + " labels for " + std::to_string(nh) + " vertices");
      for (std::size_t k = 0; k < a.names.size(); ++k) {
        const std::string& name = a.names[k];
        if (name == "unknown" || name == "Medial_wall" || name == "corpuscallosum") continue;
        if (std::find(a.labels.begin(), a.labels.end(), static_cast<std::int32_t>(k)) == a.labels.end()) continue;
        m.levels.push_back(name + " " + side);
        const auto& c = a.ctab.at(k);
        m.colors.push_back({c[0], c[1], c[2]});
        const auto code = static_cast<std::int32_t>(m.levels.size() - 1);
        for (std::size_t v = 0; v < a.labels.size(); ++v)
          if (a.labels[v] == static_cast<std::int32_t>(k)) m.codes[static_cast<std::size_t>(off) + v] = code;
      }
    }
    if (complete) out.push_back(std::move(m));
  }
  return out;
}

}  // namespace

std::vector<std::int64_t> separate_coincident(std::vector<double>& p, const std::vector<std::int32_t>& f) {
  const std::size_t n = p.size() / 3;
  std::map<std::array<double, 3>, std::size_t> first;
  std::vector<std::int64_t> dup;
  for (std::size_t v = 0; v < n; ++v) {
    const std::array<double, 3> key = {p[3 * v], p[3 * v + 1], p[3 * v + 2]};
    if (!first.emplace(key, v).second) dup.push_back(static_cast<std::int64_t>(v));
  }
  const std::vector<double> orig = p;
  for (const std::int64_t v : dup) {
    std::set<std::int32_t> nb;
    for (std::size_t i = 0; i < f.size() / 3; ++i)
      if (f[3 * i] == v || f[3 * i + 1] == v || f[3 * i + 2] == v)
        for (int k = 0; k < 3; ++k) nb.insert(f[3 * i + static_cast<std::size_t>(k)]);
    std::array<double, 3> c = {0, 0, 0};
    std::size_t m = 0;
    for (const std::int32_t u : nb) {
      const auto U = static_cast<std::size_t>(u), Vv = static_cast<std::size_t>(v);
      if (u == v || (orig[3 * U] == orig[3 * Vv] && orig[3 * U + 1] == orig[3 * Vv + 1] && orig[3 * U + 2] == orig[3 * Vv + 2])) continue;
      for (int k = 0; k < 3; ++k) c[static_cast<std::size_t>(k)] += orig[3 * U + static_cast<std::size_t>(k)];
      ++m;
    }
    if (m) for (int k = 0; k < 3; ++k) p[3 * static_cast<std::size_t>(v) + static_cast<std::size_t>(k)] = c[static_cast<std::size_t>(k)] / static_cast<double>(m);
  }
  return dup;
}

Json create_default_subject(DatasetWriter& ds, const DefaultSubjectOptions& opt, const std::string& created_by) {
  const fs::path fs0 = opt.templates / opt.default_resolution;
  if (!fs::is_directory(fs0 / "surf"))
    throw Error(code::not_found, (fs0 / "surf").string() + ": no " + opt.default_resolution + " folder (surf/) — --templates is a FreeSurfer subjects folder");
  if (ds.subject(kDefaultSubject)) throw Error(code::already_exists, text(ds.row(), "name") + ": already has a " + std::string(kDefaultSubject));
  SubjectOptions so;
  so.kind = "template";
  so.folder = std::string(kDefaultSubject) + ".nxr.zarr";
  so.source_format = "freesurfer";
  so.source_path = fs0.string();
  so.created_by = created_by;
  SubjectWriter sub = SubjectWriter::create(ds, kDefaultSubject, so);

  Json surfaces = Json::array();
  sub.compose([&] {
    const Both sphere = read_both(fs0, "sphere.reg");
    for (const char* surf : kSurfaces) {
      Both b = read_both(fs0, surf);
      const std::vector<std::int64_t> moved = separate_coincident(b.pos, b.faces);
      for (double& x : b.pos) x /= 1000.0;                 // mm -> m
      const std::int64_t n = static_cast<std::int64_t>(b.pos.size() / 3);
      SurfaceOptions o;
      o.name = std::string("cortex_") + surf;
      o.primary = std::string(surf) == "pial";
      o.source = (fs0 / "surf" / (std::string("{lh,rh}.") + surf)).string();
      o.cs_key = "fsaverage";
      o.producer = Json{{"source", opt.default_resolution + " " + surf}, {"coincident_vertices_moved", moved}};
      o.hemispheres = {{"left", 0, b.n_lh}, {"right", b.n_lh, n - b.n_lh}};
      if (o.primary) {
        if (static_cast<std::int64_t>(sphere.pos.size() / 3) != n) throw Error(code::invalid_argument, opt.default_resolution + ": sphere.reg and pial differ in vertex count");
        o.atlases = parcellations(fs0, b.n_lh, n);
        std::string fams;
        for (const auto& a : o.atlases) fams += (fams.empty() ? "" : ", ") + a.family;
        o.atlases_description = "fsaverage's cortical parcellations \xe2\x80\x94 " + fams;
        o.sphere = sphere.pos;                             // the registration sphere AS STORED (FreeSurfer's radius-100 sphere, no unit): nxr's default subject writes it so
      }
      const SurfaceInfo info = add_surface(sub, o, b.pos, b.faces);
      surfaces.push_back(Json{{"name", info.name}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"coincident_vertices_moved", moved}});
    }
  });
  return Json{{"subject", kDefaultSubject}, {"id", sub.id()}, {"store", sub.folder().string()}, {"resolution", opt.default_resolution}, {"surfaces", surfaces}};
}

}  // namespace nxr::io

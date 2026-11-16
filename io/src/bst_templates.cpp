// nxr::bst — nxr.convert.templates: the templates-and-references datastore (fsaverage as the template subject, reference maps as Fields).
#include "nxr/bst/templates.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "nxr/bst/protocol.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {

const std::vector<std::pair<std::string, std::int64_t>> kDensities = {{"fsaverage5", 10242}, {"fsaverage6", 40962}};

bool is_dir(const fs::path& p) { std::error_code ec; return fs::is_directory(p, ec); }
bool is_file(const fs::path& p) { std::error_code ec; return fs::is_regular_file(p, ec); }

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

// str(prov.get(key)) as Python prints it: "None" when absent or null
std::string get_str(const Json& prov, const char* key) {
  if (!prov.is_object() || !prov.contains(key) || prov[key].is_null()) return "None";
  return prov[key].is_string() ? prov[key].get<std::string>() : prov[key].dump();
}
// prov.get(key) or "" (the `or` of Python: None, "" and absent alike)
std::string get_or_empty(const Json& prov, const char* key) {
  if (!prov.is_object() || !prov.contains(key) || prov[key].is_null()) return "";
  return prov[key].is_string() ? prov[key].get<std::string>() : prov[key].dump();
}

// both hemispheres of a surface, in metres, left then right; `sphere`: and the registration sphere
struct Surf { std::vector<double> pos, sph; std::vector<std::int32_t> faces; std::int64_t n_lh = 0; };
Surf load_surface(const fs::path& src, const std::string& res, const std::string& surf, bool sphere) {
  auto [lh, rh] = read_hemis(src / res, surf);
  Surf s;
  s.n_lh = static_cast<std::int64_t>(lh.pos.size() / 3);
  for (const auto* h : {&lh, &rh}) for (double x : h->pos) s.pos.push_back(x / 1000.0);   // mm -> m
  s.faces = lh.faces;
  for (auto f : rh.faces) s.faces.push_back(f + static_cast<std::int32_t>(s.n_lh));
  if (sphere) {
    auto [sl, sr] = read_hemis(src / res, "sphere.reg");
    for (const auto* h : {&sl, &sr}) for (double x : h->pos) s.sph.push_back(x / 1000.0);
  }
  return s;
}

struct MapFile { std::string name, id; Json prov; fs::path L, R; };
// (name, id, provenance, L file, R file) for every hemisphere pair under maps/<id>/
std::vector<MapFile> map_files(const fs::path& maps) {
  std::vector<MapFile> out;
  if (!is_dir(maps)) return out;
  std::vector<fs::path> dirs;
  for (const auto& d : fs::directory_iterator(maps)) if (d.is_directory()) dirs.push_back(d.path());
  std::sort(dirs.begin(), dirs.end());
  for (const auto& d : dirs) {
    Json prov = Json::object();
    if (is_file(d / "provenance.json")) { std::ifstream f(d / "provenance.json"); prov = Json::parse(f); }
    for (const auto& n : glob_names(d, "", "_hemi-L.func.gii")) {
      std::string rn = n;
      rn.replace(rn.find("_hemi-L"), 7, "_hemi-R");
      if (!is_file(d / rn)) continue;
      out.push_back({n.substr(0, n.find("_space-")), d.filename().string(), prov, d / n, d / rn});
    }
  }
  return out;
}

// the surface a map of `n` values goes on: `surface` when named, else the subject's surface with `n` vertices — the primary first, then a pial one; a sphere never; more than one
// left is refused (name it with --surface)
nxr::io::Row target(SubjectWriter& sub, std::int64_t n, const std::optional<std::string>& surface) {
  if (surface) {
    const auto m = sub.find("manifold", *surface);
    if (!m) throw nxr::io::Error(code::no_such_row, sub.name() + ": no surface '" + *surface + "'");
    return *m;
  }
  std::vector<nxr::io::Row> c;
  for (auto& m : sub.db().all("SELECT * FROM manifold WHERE subject_id = ? AND type = 'surface' AND n_vertices = ? ORDER BY is_primary DESC, path", {nxr::io::V(sub.id()), nxr::io::V(n)})) {
    const std::string p = nxr::io::text(m, "path");
    if (p.size() < 7 || p.compare(p.size() - 7, 7, "_sphere") != 0) c.push_back(std::move(m));
  }
  auto keep = [&](auto pred) { std::vector<nxr::io::Row> r; for (const auto& m : c) if (pred(m)) r.push_back(m); return r; };
  if (auto p = keep([](const nxr::io::Row& m) { return nxr::io::integer(m, "is_primary") != 0; }); !p.empty()) c = std::move(p);
  else if (auto q = keep([](const nxr::io::Row& m) { return nxr::io::text(m, "path").find("pial") != std::string::npos; }); !q.empty()) c = std::move(q);
  if (c.size() != 1) {
    std::string names;
    for (std::size_t i = 0; i < c.size(); ++i) names += (i ? ", " : "") + nxr::io::text(c[i], "path");
    value_error(sub.name() + ": " + std::to_string(c.size()) + " surfaces with " + std::to_string(n) + " vertices (" + (names.empty() ? "none" : names) +
                ") \xe2\x80\x94 name one with --surface");
  }
  return c.front();
}

void write_readme(const fs::path& source, const fs::path& datastore, const std::string& dataset, const std::string& subject, const std::vector<MapFile>& fields) {
  std::string rows;
  for (const auto& f : fields) {
    const Json& p = f.prov;
    rows += (rows.empty() ? "" : "\n") + std::string("| `") + f.name + "` | " + get_str(p, "map") + " | " + get_str(p, "original_space") + " " + get_str(p, "original_density") + " | " +
            get_str(p, "method_class") + " | " + get_str(p, "source_paper") + " | " + get_str(p, "licence") + " |";
  }
  const std::string src = source.filename().string();
  std::ostringstream o;
  o << "# Templates and references (nxr datastore)\n\n"
    << "Built by `nxr-convert templates` (schema 50) from `" << src << "`; opened by Cortical Flow like any datastore\n"
    << "(dataset `" << dataset << "`, template subject `" << subject << "`). Re-run the command to add maps; Hodge-Dirac modes go on later with\n"
    << "`nxr-neuro eigen solve --dataset " << (datastore / dataset).string() << " --subject " << subject << "`.\n\n"
    << "**Template:** FreeSurfer 7.4.1 fsaverage, ico7, 163 842 vertices per hemisphere, left then right (327 684), metres.\n"
    << "**Manifolds** (`source/`): `pial` (primary), `white`, `inflated`, `pial_sphere` (sphere.reg); `pial_fsaverage5|6`,\n"
    << "`pial_fsaverage5|6_sphere`, `inflated_fsaverage5|6`. fsaverage5/6 spheres are the first 10 242 / 40 962 vertices of\n"
    << "ico7's *per hemisphere*; their pial/inflated coordinates are not ico7 subsets (use index correspondence on spheres only).\n"
    << "**Labels:** `pial_atlases/{Structures, Desikan-Killiany, Destrieux}`; `pial_cortex` (cortex mask, indices).\n"
    << "**Fibres:** `hcp1065` (curves, 64 points, fsaverage metres) and `hcp1065_ends` (ends -> nearest white vertex within 5 mm),\n"
    << "when built with `--trk`: HCP-1065 template tracking by DSI Studio (Yeh et al.), one random-threshold realisation.\n"
    << "**Fields** (`source/<id>`, on `pial`, NaN = not given; provenance JSON in each field's producer record):\n\n"
    << "| id | map | original | method | source | licence |\n|---|---|---|---|---|---|\n"
    << rows << "\n\n"
    << "MNI152 volume maps are APPROXIMATE (registration fusion); their volumes are not stored here (see `references/neuromaps-receptors`).\n"
    << "Resampling to ico7 adds no resolution (README of `" << src << "`).\n";
  std::ofstream f(datastore / "README.md", std::ios::binary);
  if (!f) throw nxr::io::Error(code::io, (datastore / "README.md").string() + ": cannot be written");
  f << o.str();
}

}  // namespace

Json build_templates(const fs::path& source, const fs::path& dataset, const TemplatesOptions& opt, const std::string& created_by) {
  const fs::path& src = source;
  const std::string& subject = opt.subject;
  nxr::io::OpenSubjectOptions oo;
  oo.created_by = created_by;
  std::optional<nxr::io::DatasetWriter> ds;
  if (is_file(dataset / nxr::io::kDatasetFile)) ds.emplace(nxr::io::open_writer(dataset));
  else {
    nxr::io::CreateOptions co;
    co.source_tool = "freesurfer+neuromaps";
    co.source_path = src.string();
    co.created_by = created_by;
    ds.emplace(nxr::io::create_dataset(dataset.parent_path(), dataset.filename().string(), co));
  }
  const bool exists = ds->db().one("SELECT * FROM subject WHERE dataset_id = ? AND name = ?", {nxr::io::V(ds->id()), nxr::io::V(subject)}).has_value();
  std::optional<SubjectWriter> sub_opt;
  if (exists) sub_opt.emplace(SubjectWriter::open(*ds, subject, oo));
  else {
    nxr::io::SubjectOptions so;
    so.kind = "template";
    so.folder = subject + ".nxr.zarr";
    so.source_format = "freesurfer";
    so.source_path = (src / "fsaverage").string();
    so.created_by = created_by;
    sub_opt.emplace(SubjectWriter::create(*ds, subject, so));
  }
  SubjectWriter& sub = *sub_opt;
  Json surfaces = Json::array(), fields = Json::array(), skipped = Json::array(), unplaced = Json::array();
  Json hcp;
  bool own = false;
  sub.compose([&] {
    const fs::path fsdir = src / "fsaverage" / "fsaverage";
    const Json prod = Json{{"source", "FreeSurfer 7.4.1 fsaverage"}};
    std::optional<nxr::io::Row> pial = sub.find("manifold", "source/pial");
    own = pial.has_value() || !sub.db().one("SELECT 1 FROM manifold WHERE subject_id = ? AND is_primary = 1", {nxr::io::V(sub.id())});
    if (!pial && own) {   // a subject with no surfaces yet: fsaverage is written
      std::string pial_id;
      std::int64_t n = 0, nl = 0;
      for (const auto& [name, surf, primary] : {std::tuple<const char*, const char*, bool>{"pial", "pial", true}, {"white", "white", false}, {"inflated", "inflated", false}}) {
        Surf s = load_surface(src / "fsaverage", "fsaverage", surf, primary);
        Json producer = prod;
        producer["surface"] = surf;
        producer["units"] = "m";
        const std::string sid = default_surface(sub, name, s.pos, s.faces, primary ? std::optional<std::vector<double>>(s.sph) : std::nullopt, s.n_lh, producer, primary);
        surfaces.push_back(name);
        if (primary) { pial_id = sid; n = static_cast<std::int64_t>(s.pos.size() / 3); nl = s.n_lh; }
      }
      pial = sub.find("manifold", "source/pial");
      default_parcellations(sub, fsdir, "source/pial", pial_id, nl, n);
      std::vector<std::int32_t> idx;
      for (const auto& [h, off] : {std::pair<const char*, std::int64_t>{"lh", 0}, {"rh", nl}})
        for (auto v : read_label(fsdir / "label" / (std::string(h) + ".cortex.label"))) idx.push_back(static_cast<std::int32_t>(v + off));
      std::sort(idx.begin(), idx.end());
      {   // DefaultSubjectWriter.indices
        nxr::io::SelectionSpec sel;
        sel.name = "pial_cortex";
        sel.path = "source/pial_cortex";
        sel.type = "indices";
        sel.manifold_id = pial_id;
        sel.data = view(idx, {static_cast<std::int64_t>(idx.size())});
        sel.picked = true;
        sel.cols.set("description", "the cortex mask (FreeSurfer cortex.label): vertices that are not the medial wall");
        sub.selection(std::move(sel));
      }
      for (const auto& [res, nv] : kDensities)   // nested by the sphere: vertex i of fsaverage5/6 is vertex i of ico7
        for (const auto& [name, surf, primary] : {std::tuple<std::string, const char*, bool>{"pial_" + res, "pial", true}, {"inflated_" + res, "inflated", false}}) {
          Surf s = load_surface(src / "fsaverage", res, surf, primary);
          Json producer = prod;
          producer["surface"] = surf;
          producer["units"] = "m";
          producer["density"] = res;
          producer["n_per_hemisphere"] = nv;
          producer["note"] = "sphere vertices are the first N of ico7's; pial/inflated coordinates are NOT ico7 subsets";
          default_surface(sub, name, s.pos, s.faces, primary ? std::optional<std::vector<double>>(s.sph) : std::nullopt, s.n_lh, producer, false);
          surfaces.push_back(name);
        }
    }
    for (const auto& mf : map_files(src / "maps")) {
      const std::string node_name = nxr::io::sanitize_node_name(mf.name);
      const std::string path = "source/" + node_name;
      if (sub.find("field", path)) { skipped.push_back(mf.name); continue; }
      std::vector<float> v = read_gifti(mf.L);
      const std::vector<float> r = read_gifti(mf.R);
      v.insert(v.end(), r.begin(), r.end());
      const std::int64_t size = static_cast<std::int64_t>(v.size());
      nxr::io::Row on;
      try {
        on = (own && !opt.surface) ? *pial : target(sub, size, opt.surface);
      } catch (const nxr::io::Error& e) {   // no single surface fits: reported, the others go on
        if (std::string(e.code()) != code::invalid_argument) throw;
        unplaced.push_back(Json{{"map", mf.name}, {"reason", e.what()}});
        continue;
      }
      if (size != nxr::io::integer(on, "n_vertices"))
        value_error(mf.name + ": " + std::to_string(size) + " values for a " + std::to_string(nxr::io::integer(on, "n_vertices")) + "-vertex " + nxr::io::text(on, "path"));
      const Json& prov = mf.prov;
      const std::string flag = upper(get_or_empty(prov, "method_class")).find("APPROX") != std::string::npos ? " APPROXIMATE (volume projected by registration fusion)" : "";
      const std::string map_name = get_str(prov, "map");
      nxr::io::FieldSpec f;
      f.name = node_name;
      f.path = path;
      f.kind = get_or_empty(prov, "map").empty() ? mf.name : get_or_empty(prov, "map");
      f.manifold_id = nxr::io::text(on, "id");
      f.data = view(v, {size});
      f.array.compress = true;
      f.cols.set("source", get_or_empty(prov, "source_paper"));
      if (prov.is_object() && prov.contains("map") && !prov["map"].is_null()) f.cols.set("comment", get_str(prov, "map")); else f.cols.null("comment");
      f.cols.set("description", map_name + " - from " + get_str(prov, "original_space") + " " + get_str(prov, "original_density") + "; " + get_str(prov, "method_class") + flag +
                                    ". Licence: " + get_str(prov, "licence") + ". NaN = not given (e.g. medial wall).");
      f.cols.json("producer_json", prov);
      sub.field(std::move(f));
      fields.push_back(mf.name);
    }
    if (opt.trk && own && !sub.find("manifold", nxr::io::folder_for("curve") + "/hcp1065")) {
      std::vector<double> white;
      auto [lh, rh] = read_hemis(src / "fsaverage" / "fsaverage", "white");
      for (const auto* h : {&lh, &rh}) for (double x : h->pos) white.push_back(x / 1000.0);
      hcp = hcp_fibres(sub, *opt.trk, nxr::io::text(*pial, "id"), white, white);
    }
  });
  if (own) write_readme(src, dataset.parent_path(), dataset.filename().string(), subject, map_files(src / "maps"));   // the templates datastore's README — never another datastore's
  Json out = Json{{"store", sub.folder().string()}, {"id", sub.id()}, {"surfaces", surfaces}, {"fields", fields}, {"skipped", skipped}, {"unplaced", unplaced}};
  if (!hcp.is_null()) out["hcp1065"] = hcp;
  return out;
}

// A FreeSurfer subject's surface, both hemispheres (`<fs_dir>/surf/lh.<surface>`, `rh.<surface>`), as stored (mm): nxr's _read_hemis.
std::pair<HemiMesh, HemiMesh> read_hemis(const fs::path& fs_dir, const std::string& surface) {
  auto one = [&](const char* h) {
    nxr::io::Mesh m = nxr::io::read_freesurfer_surface(fs_dir / "surf" / (std::string(h) + "." + surface));
    return HemiMesh{std::move(m.vertices), std::move(m.faces)};
  };
  return {one("lh"), one("rh")};
}

// ── the atlas/nibabel layer: NOT IMPLEMENTED ──
// `convert templates` BUILDS the templates folder (it is not a Brainstorm import); it needs the readers nxr gets from nibabel (FreeSurfer surfaces and labels, GIfTI, the
// HCP-1065 .trk) and the default-subject writer of `nxr.pipelines.atlas`. None of them is ported: each says so, naming what is missing.
namespace {
[[noreturn]] void not_implemented(const std::string& what) { throw nxr::io::Error(code::unsupported, what + " is not implemented in nxr-io (nibabel / nxr.pipelines.atlas)"); }
}  // namespace

std::vector<std::int32_t> read_label(const fs::path&) { not_implemented("the FreeSurfer label reader (read_label)"); }
std::vector<float> read_gifti(const fs::path&) { not_implemented("the GIfTI reader (read_gifti)"); }
std::string default_surface(SubjectWriter&, const std::string&, const std::vector<double>&, const std::vector<std::int32_t>&, const std::optional<std::vector<double>>&,
                            std::int64_t, const Json&, bool) {
  not_implemented("the default-subject surface writer (default_surface)");
}
void default_parcellations(SubjectWriter&, const fs::path&, const std::string&, const std::string&, std::int64_t, std::int64_t) {
  not_implemented("the default-subject parcellations writer (default_parcellations)");
}
Json hcp_fibres(SubjectWriter&, const fs::path&, const std::string&, const std::vector<double>&, const std::vector<double>&) {
  not_implemented("the HCP-1065 streamline reader (hcp_fibres)");
}

}  // namespace nxr::bst

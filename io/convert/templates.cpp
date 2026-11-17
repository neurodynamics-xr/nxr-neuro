// The template importers of io/convert: the source manifest, FreeSurfer (fsaverage), Brainstorm's default anatomy (ICBM152), DSI Studio streamlines (HCP-1065), and
// `build_template`, which makes a template package from them with an origin record on every item. See nxr/convert/templates.h.
#include "nxr/convert/templates.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

#include "nxr/bst/entities.h"
#include "nxr/bst/mat.h"
#include "nxr/bst/subject.h"
#include "nxr/io/default_subject.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/formats/fif.h"
#include "nxr/io/surface.h"

namespace nxr::convert {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {

[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

const char* const kManifestText =
#include "gen/template_sources_json.inc"
    ;

std::string file_sha256(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw nxr::io::Error(code::not_found, p.string() + ": cannot be read");
  nxr::io::Sha256 h;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(f.gcount()));
  }
  return h.hex();
}

}  // namespace

const Json& source_manifest() {
  static const Json m = Json::parse(kManifestText);
  return m;
}

SourceChecker::SourceChecker(std::string recipe, fs::path folder, bool accept_unlisted) : recipe_name_(std::move(recipe)), folder_(std::move(folder)), accept_(accept_unlisted) {
  const Json& r = source_manifest()["recipes"];
  if (!r.contains(recipe_name_)) {
    std::string names;
    for (auto it = r.begin(); it != r.end(); ++it) names += (names.empty() ? "" : ", ") + it.key();
    refuse("no recipe '" + recipe_name_ + "' in the source manifest (choose from " + names + ")");
  }
  recipe_ = r[recipe_name_];
}
const Json& SourceChecker::recipe() const { return recipe_; }

fs::path SourceChecker::file(const std::string& relative) {
  const fs::path p = folder_ / relative;
  if (!fs::is_regular_file(p)) throw nxr::io::Error(code::not_found, relative + ": not in " + folder_.string() + " (the recipe '" + recipe_name_ + "' reads it)");
  const std::string got = file_sha256(p);
  const Json* listed = nullptr;
  for (const auto& e : recipe_["files"]) if (e["path"] == relative) listed = &e;
  const bool matches = listed && (*listed)["sha256"] == got;
  if (!listed && !accept_) refuse(relative + ": not a listed source file of the recipe '" + recipe_name_ + "' (sha256 " + got + "); --accept-unlisted-sources reads it and records its checksum");
  if (listed && !matches && !accept_)
    refuse(relative + ": its sha256 is " + got + ", the manifest lists " + (*listed)["sha256"].get<std::string>() + " (another release of the file?); --accept-unlisted-sources reads it and records its checksum");
  read_.push_back(Json{{"path", relative}, {"sha256", got}, {"listed", listed != nullptr}, {"matches_manifest", matches}});
  return p;
}
Json SourceChecker::report() const { return Json{{"recipe", recipe_name_}, {"files", read_}}; }
std::string SourceChecker::sha256_of(const std::string& relative) const {
  for (const auto& e : read_) if (e["path"] == relative) return e["sha256"].get<std::string>();
  return "";
}

// ── the atlas terms ────────────────────────────────────────────────────────────────────────────────────────────

AtlasTerms atlas_terms(const std::string& name) {
  static const std::string fs_licence = "FreeSurfer Software License Agreement Part B v1.0 (https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)";
  if (name == "Desikan-Killiany" || name == "aparc") return {fs_licence, "Desikan RS et al. (2006) NeuroImage 31:968-980; FreeSurfer (Fischl et al. 1999)", true, false};
  if (name == "Destrieux" || name == "aparc.a2009s") return {fs_licence, "Destrieux C et al. (2010) NeuroImage 53:1-15; FreeSurfer (Fischl et al. 1999)", true, false};
  if (name == "DKT") return {fs_licence, "Klein A, Tourville J (2012) Front Neurosci 6:171; FreeSurfer (Fischl et al. 1999)", true, false};
  if (name.rfind("Schaefer_", 0) == 0) return {"MIT (CBIG repository, https://github.com/ThomasYeoLab/CBIG)", "Schaefer A et al. (2018) Cereb Cortex 28:3095-3114", true, false};
  if (name == "Hammers") return {"Imperial College / Hammers atlas licence: research and teaching only; never redistributed", "Hammers A et al. (2003) Hum Brain Mapp 19:224-247", false, true};
  if (name == "LPBA40") return {"LONI research licence: no distribution outside the organisation; never redistributed", "Shattuck DW et al. (2008) NeuroImage 39:1064-1080", false, true};
  // everything else: its terms are not confirmed, and what is not confirmed is not redistributed
  return {"not confirmed", "the atlas's own citation (" + name + ")", false, false};
}

namespace {

nxr::io::OriginRecord freesurfer_origin(const std::string& source) {
  return {source,
          "FreeSurfer Software License Agreement Part B v1.0 (https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense): with the licence text and notice, attributions kept, modified versions marked",
          "FreeSurfer fsaverage. Fischl B, Sereno MI, Dale AM (1999) NeuroImage 9:195-207; Fischl B et al. (1999) Hum Brain Mapp 8:272-284. Converted to the nxr package format by nxr-neuro "
          "(MODIFIED: vertex positions in metres, coincident vertices separated, left and right hemispheres in one surface)",
          true};
}

// ── FreeSurfer ────────────────────────────────────────────────────────────────────────────────────────────────

struct Both { std::vector<double> pos; std::vector<std::int32_t> faces; std::int64_t n_lh = 0; };
Both read_both(SourceChecker& sc, const std::string& surface) {
  Both b;
  const nxr::io::Mesh lh = nxr::io::read_freesurfer_surface(sc.file("surf/lh." + surface));
  const nxr::io::Mesh rh = nxr::io::read_freesurfer_surface(sc.file("surf/rh." + surface));
  b.n_lh = lh.n_vertices();
  b.pos = lh.vertices;
  b.pos.insert(b.pos.end(), rh.vertices.begin(), rh.vertices.end());
  b.faces = lh.faces;
  for (auto f : rh.faces) b.faces.push_back(f + static_cast<std::int32_t>(b.n_lh));
  return b;
}

std::string annot_family(const std::string& annot) {
  if (annot == "aparc") return "Desikan-Killiany";
  if (annot == "aparc.a2009s") return "Destrieux";
  return annot;
}

nxr::io::AtlasMember parcellation(SourceChecker& sc, const std::string& annot, std::int64_t n_lh, std::int64_t n) {
  nxr::io::AtlasMember m;
  m.family = annot_family(annot);
  m.codes.assign(static_cast<std::size_t>(n), 0);
  m.levels = {"unassigned"};
  m.colors = {{0, 0, 0}};
  for (const auto& [h, side, off] : {std::tuple<const char*, const char*, std::int64_t>{"lh", "L", 0}, {"rh", "R", n_lh}}) {
    const nxr::io::FsAnnot a = nxr::io::read_freesurfer_annot(sc.file(std::string("label/") + h + "." + annot + ".annot"));
    const std::int64_t nh = std::string(h) == "lh" ? n_lh : n - n_lh;
    if (static_cast<std::int64_t>(a.labels.size()) != nh) refuse(std::string(h) + "." + annot + ".annot: " + std::to_string(a.labels.size()) + " labels for " + std::to_string(nh) + " vertices");
    for (std::size_t k = 0; k < a.names.size(); ++k) {
      const std::string& name = a.names[k];
      if (name == "unknown" || name == "Medial_wall" || name == "corpuscallosum") continue;
      if (std::find(a.labels.begin(), a.labels.end(), static_cast<std::int32_t>(k)) == a.labels.end()) continue;
      m.levels.push_back(name + " " + side);
      const auto& c = a.ctab.at(k);
      m.colors.push_back({c[0], c[1], c[2]});
      const auto cd = static_cast<std::int32_t>(m.levels.size() - 1);
      for (std::size_t v = 0; v < a.labels.size(); ++v) if (a.labels[v] == static_cast<std::int32_t>(k)) m.codes[static_cast<std::size_t>(off) + v] = cd;
    }
  }
  return m;
}

std::vector<std::int32_t> read_label_vertices(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw nxr::io::Error(code::not_found, p.string() + ": cannot be read");
  std::string line;
  std::getline(f, line);                                    // "#!ascii label , from subject ..."
  std::getline(f, line);
  std::int64_t n = 0;
  try { n = std::stoll(line); } catch (const std::exception&) { refuse(p.filename().string() + ": the second line is not a vertex count"); }
  std::vector<std::int32_t> out;
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t v;
    double x, y, z, s;
    if (!(f >> v >> x >> y >> z >> s)) refuse(p.filename().string() + ": fewer vertices than its count says");
    out.push_back(static_cast<std::int32_t>(v));
  }
  return out;
}

Json import_freesurfer(SubjectWriter& sub, const FreeSurferSource& src, SourceChecker& sc) {
  if (src.surfaces.empty()) refuse("fsaverage: name at least one surface");
  Json surfaces = Json::array();
  const Both sphere = read_both(sc, "sphere.reg");
  std::string pial_id;
  std::int64_t n_primary = 0, nl = 0;
  for (std::size_t k = 0; k < src.surfaces.size(); ++k) {
    const std::string& surf = src.surfaces[k];
    Both b = read_both(sc, surf);
    const std::vector<std::int64_t> moved = nxr::io::separate_coincident(b.pos, b.faces);
    for (double& x : b.pos) x /= 1000.0;   // mm -> m
    const std::int64_t n = static_cast<std::int64_t>(b.pos.size() / 3);
    nxr::io::SurfaceOptions o;
    o.name = "cortex_" + surf;
    o.primary = k == 0;
    o.source = src.label + " " + surf;
    o.cs_key = "freesurfer-tkr";   // FreeSurfer's surface RAS in metres (G13): a system with a unit, as the scalp and the montage are in
    o.producer = Json{{"source", src.label + " " + surf}, {"units", "m"}, {"coincident_vertices_moved", moved}};
    o.hemispheres = {{"left", 0, b.n_lh}, {"right", b.n_lh, n - b.n_lh}};
    if (o.primary) {
      if (static_cast<std::int64_t>(sphere.pos.size() / 3) != n) refuse(src.label + ": sphere.reg and " + surf + " differ in vertex count");
      for (const auto& a : src.annotations) o.atlases.push_back(parcellation(sc, a, b.n_lh, n));
      if (!o.atlases.empty()) {
        std::string fams;
        for (const auto& a : o.atlases) fams += (fams.empty() ? "" : ", ") + a.family;
        o.atlases_description = src.label + "'s cortical parcellations, named by the caller: " + fams;
      }
      o.sphere = sphere.pos;   // FreeSurfer's radius-100 sphere as stored
    }
    const nxr::io::SurfaceInfo info = nxr::io::add_surface(sub, o, b.pos, b.faces);
    surfaces.push_back(Json{{"name", info.name}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"coincident_vertices_moved", moved.size()}});
    if (o.primary) { pial_id = info.id; n_primary = n; nl = b.n_lh; }
  }
  if (src.cortex_mask) {
    std::vector<std::int32_t> idx;
    for (const auto& [h, off] : {std::pair<const char*, std::int64_t>{"lh", 0}, {"rh", nl}})
      for (auto v : read_label_vertices(sc.file(std::string("label/") + h + ".cortex.label"))) idx.push_back(static_cast<std::int32_t>(v + off));
    std::sort(idx.begin(), idx.end());
    if (!idx.empty() && idx.back() >= n_primary) refuse("the cortex label names a vertex beyond the surface");
    nxr::io::SelectionSpec sel;
    sel.name = "cortex_" + src.surfaces[0] + "_cortex";
    sel.path = "source/" + sel.name;
    sel.type = "indices";
    sel.manifold_id = pial_id;
    sel.data = view(idx, {static_cast<std::int64_t>(idx.size())});
    sel.picked = true;
    sel.cols.set("description", "the cortex mask (FreeSurfer cortex.label): the vertices that are not the medial wall");
    sub.selection(std::move(sel));
  }
  return Json{{"surfaces", surfaces}, {"annotations", src.annotations}, {"cortex_mask", src.cortex_mask}};
}

// ── Brainstorm's default anatomy ─────────────────────────────────────────────────────────────────────────────

Json import_bst_anatomy(SubjectWriter& sub, const BrainstormAnatomySource& src, SourceChecker& sc) {
  if (src.surfaces.empty()) refuse("icbm152: name at least one surface file");
  Json surfaces = Json::array();
  for (std::size_t k = 0; k < src.surfaces.size(); ++k) {
    const fs::path p = sc.file(src.surfaces[k]);
    // `atlases` is the explicit list: an empty one writes no parcellation
    const auto info = nxr::bst::export_surface(sub, p, "", k == 0, std::vector<std::string>(src.atlases));
    surfaces.push_back(Json{{"name", info.name}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"atlases", info.atlases}, {"registration", info.registration}});
  }
  return Json{{"surfaces", surfaces}, {"atlases", src.atlases}};
}

}  // namespace

// ── MNI to the anatomy ───────────────────────────────────────────────────────────────────────────────────────

std::array<double, 12> mni_to_scs_from_subjectimage(const fs::path& mat) {
  auto frame = [&](const char* name, std::array<double, 9>& R, std::array<double, 3>& T) {
    const auto v = nxr::bst::read_variable(mat, name);
    if (!v || !v->is_struct() || !v->has_field("R") || !v->has_field("T")) refuse(mat.filename().string() + ": no " + name + " frame with R and T (the MNI and SCS transforms of the anatomy)");
    const std::vector<double> r = v->field("R")->row_major(), t = v->field("T")->doubles();
    if (r.size() != 9 || t.size() != 3) refuse(mat.filename().string() + ": " + name + ".R is not 3x3 or " + name + ".T not 3 values");
    std::copy(r.begin(), r.end(), R.begin());
    std::copy(t.begin(), t.end(), T.begin());
  };
  std::array<double, 9> Rs, Rn;
  std::array<double, 3> Ts, Tn;
  frame("SCS", Rs, Ts);
  frame("NCS", Rn, Tn);
  // inverse of the 3x3 NCS.R
  const double* a = Rn.data();
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (std::abs(det) < 1e-12) refuse(mat.filename().string() + ": NCS.R is singular");
  const double inv[9] = {(a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
                         (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
                         (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det};
  // scs_mm = Rs (inv (mni - Tn)) + Ts = M mni + c,  M = Rs inv, c = Ts - M Tn; metres; stored as the 4x3 for row vectors: [M^T / 1000; c / 1000]
  double M[9], c[3];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { M[3 * i + j] = 0; for (int k = 0; k < 3; ++k) M[3 * i + j] += Rs[static_cast<std::size_t>(3 * i + k)] * inv[3 * k + j]; }
  for (int i = 0; i < 3; ++i) { c[i] = Ts[static_cast<std::size_t>(i)]; for (int j = 0; j < 3; ++j) c[i] -= M[3 * i + j] * Tn[static_cast<std::size_t>(j)]; }
  std::array<double, 12> out{};
  for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) out[static_cast<std::size_t>(3 * j + i)] = M[3 * i + j] / 1000.0;   // row j of the 4x3: the coefficient of mni_j
  for (int i = 0; i < 3; ++i) out[static_cast<std::size_t>(9 + i)] = c[i] / 1000.0;
  return out;
}

HeadLandmarks landmarks_from_subjectimage(const fs::path& mat) {
  const auto scs = nxr::bst::read_variable(mat, "SCS");
  const auto vox = nxr::bst::read_variable(mat, "Voxsize");
  if (!scs || !scs->is_struct() || !vox) refuse(mat.filename().string() + ": no SCS frame or no Voxsize");
  for (const char* k : {"R", "T", "NAS", "LPA", "RPA"}) if (!scs->has_field(k)) refuse(mat.filename().string() + ": SCS has no " + k);
  const std::vector<double> R = scs->field("R")->row_major(), T = scs->field("T")->doubles(), vs = vox->doubles();
  if (R.size() != 9 || T.size() != 3 || vs.size() != 3) refuse(mat.filename().string() + ": SCS.R, SCS.T or Voxsize has the wrong size");
  // scs_mm = R (voxel * voxsize) + T, as the anatomy import does; metres
  auto point = [&](const char* k) {
    const std::vector<double> v = scs->field(k)->doubles();
    if (v.size() != 3) refuse(mat.filename().string() + ": SCS." + k + " is not 3 numbers");
    std::array<double, 3> o{};
    for (int r = 0; r < 3; ++r) o[static_cast<std::size_t>(r)] = (R[static_cast<std::size_t>(3 * r)] * v[0] * vs[0] + R[static_cast<std::size_t>(3 * r + 1)] * v[1] * vs[1] + R[static_cast<std::size_t>(3 * r + 2)] * v[2] * vs[2] + T[static_cast<std::size_t>(r)]) / 1000.0;
    return o;
  };
  return {point("NAS"), point("LPA"), point("RPA")};
}

HeadLandmarks landmarks_from_fif(const fs::path& path) {
  const nxr::formats::fif::File f(std::make_shared<nxr::formats::FileSource>(path.string()));
  const nxr::formats::fif::Fiducials fid = nxr::formats::fif::read_fiducials(f);
  if (fid.coord_frame != 5) refuse(path.filename().string() + ": the fiducials are in FIF coordinate frame " + std::to_string(fid.coord_frame) + ", not the MRI's surface RAS (5)");
  HeadLandmarks out;
  int found = 0;
  for (const auto& d : fid.points) {
    if (d.kind != 1) continue;
    std::array<double, 3>* to = d.ident == 1 ? &out.lpa : d.ident == 2 ? &out.nasion : d.ident == 3 ? &out.rpa : nullptr;
    if (!to) continue;
    *to = d.r;
    ++found;
  }
  if (found != 3) refuse(path.filename().string() + ": it holds " + std::to_string(found) + " of the nasion, LPA and RPA, not 3");
  return out;
}

namespace {

// the landmarks as a points manifold `head_landmarks` (NAS, LPA, RPA; positions in `cs_key`, metres) with their names as a labelling
void write_head_landmarks(SubjectWriter& sub, const std::string& cs_key, const HeadLandmarks& lm, const Json& producer) {
  const std::string name = "head_landmarks", path = nxr::io::folder_for("points") + "/" + name, mid = nxr::io::uuid7();
  std::vector<float> pos;
  for (const auto* p : {&lm.nasion, &lm.lpa, &lm.rpa}) for (int c = 0; c < 3; ++c) pos.push_back(static_cast<float>((*p)[static_cast<std::size_t>(c)]));
  nxr::io::ManifoldSpec m;
  m.id = mid;
  m.name = name;
  m.path = path;
  m.type = "points";
  m.n_vertices = 3;
  m.topology = {"none", 0, 1, std::nullopt, {}};
  nxr::io::GeometrySpec g;
  g.form = "stored";
  g.cs_key = cs_key;
  g.positions_path = path;
  m.geometry = g;
  m.populate = [&sub, path, pos, mid](nxr::io::Store&, const std::string&) { sub.write_array(path, view(pos, {3, 3}), {}, std::nullopt, "manifold", mid); };
  m.cols.json("producer_json", producer);
  sub.manifold(std::move(m));
  nxr::bst::LabellingOptions lo;
  lo.manifold_id = mid;
  lo.dictionary = name + "_names";
  lo.scope = "subject";
  lo.description = "the landmarks' names: the nasion, the left and the right preauricular points";
  nxr::bst::labelling(sub, path + "_names", {0, 1, 2}, {"NAS", "LPA", "RPA"}, lo);
}

// ── HCP-1065 ──────────────────────────────────────────────────────────────────────────────────────────────────

class KdTree {
 public:
  explicit KdTree(const std::vector<double>& pts) : p_(pts), order_(pts.size() / 3) {
    for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<std::int64_t>(i);
    build(0, static_cast<std::int64_t>(order_.size()), 0);
  }
  // the nearest point's index (the lowest on a tie) and its squared distance
  std::pair<std::int64_t, double> nearest(const double* q) const {
    std::pair<std::int64_t, double> best{-1, std::numeric_limits<double>::infinity()};
    search(0, static_cast<std::int64_t>(order_.size()), 0, q, best);
    return best;
  }

 private:
  const std::vector<double>& p_;
  std::vector<std::int64_t> order_;
  double at(std::int64_t i, int axis) const { return p_[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(axis)]; }
  void build(std::int64_t lo, std::int64_t hi, int axis) {
    if (hi - lo < 2) return;
    const std::int64_t mid = (lo + hi) / 2;
    std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi, [&](std::int64_t a, std::int64_t b) { return at(a, axis) < at(b, axis) || (at(a, axis) == at(b, axis) && a < b); });
    build(lo, mid, (axis + 1) % 3);
    build(mid + 1, hi, (axis + 1) % 3);
  }
  void search(std::int64_t lo, std::int64_t hi, int axis, const double* q, std::pair<std::int64_t, double>& best) const {
    if (hi <= lo) return;
    const std::int64_t mid = (lo + hi) / 2, i = order_[static_cast<std::size_t>(mid)];
    double d = 0;
    for (int a = 0; a < 3; ++a) { const double t = at(i, a) - q[a]; d += t * t; }
    if (d < best.second || (d == best.second && i < best.first)) best = {i, d};
    const double diff = q[axis] - at(i, axis);
    const int next = (axis + 1) % 3;
    if (diff < 0) { search(lo, mid, next, q, best); if (diff * diff <= best.second) search(mid + 1, hi, next, q, best); }
    else { search(mid + 1, hi, next, q, best); if (diff * diff <= best.second) search(lo, mid, next, q, best); }
  }
};

std::vector<double> read_positions(SubjectWriter& sub, const std::string& path) {
  const auto rd = nxr::io::ArrayReader::open(sub.store(), path + "/vertices");
  const auto bytes = rd.read();
  const std::size_t n = bytes.size() / rd.itemsize();
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (rd.dtype() == nxr::io::DType::Float32) { float x; std::memcpy(&x, bytes.data() + i * 4, 4); v[i] = x; }
    else { double x; std::memcpy(&x, bytes.data() + i * 8, 8); v[i] = x; }
  }
  return v;
}

Json hcp_origin_doc(const HcpSource& h, const Json& sources, std::int64_t n_stream, std::int64_t n_points, std::int64_t n_gated, double median_mm) {
  return Json{{"schema", "nxr-origin/1"},
              {"source", Json{{"name", "HCP-1065 population-averaged diffusion template (DSI Studio; Yeh) and whole-brain tractography tracked on it"},
                              {"url", "https://brain.labsolver.org/hcp_template.html"},
                              {"files", sources}}},
              {"citation", Json::array({"Yeh F-C (2022) Population-based tract-to-region connectome of the human brain and its hierarchical topology. Nat Commun 13:4933, doi:10.1038/s41467-022-32595-4",
                                        "Yeh F-C et al. (2018) Population-averaged atlas of the macroscale human structural connectome and its network topology. NeuroImage 178:57-68, doi:10.1016/j.neuroimage.2018.05.027",
                                        "Van Essen DC et al. (2013) The WU-Minn Human Connectome Project: an overview. NeuroImage 80:62-79"})},
              {"licence", Json{{"id", "CC-BY-SA-4.0"}, {"url", "https://creativecommons.org/licenses/by-sa/4.0/"}, {"share_alike", true},
                               {"also", "WU-Minn HCP Open Access Data Use Terms"}}},
              {"terms", "WU-Minn HCP Open Access Data Use Terms: no attempt to identify or contact any participant; redistribution of HCP data and of data derived from them only under these same terms (term 4); "
                        "acknowledge the use of HCP data when presenting results (term 5). ShareAlike (CC BY-SA 4.0): adaptations are shared under the same licence. Group-template fibres: no participant's data."},
              {"processing", Json::array({"streamlines read from the TrackVis .trk (voxel-mm to RAS mm as nibabel reads them), then shifted by (" + std::to_string(static_cast<int>(h.shift_mm[0])) + ", " +
                                              std::to_string(static_cast<int>(h.shift_mm[1])) + ", " + std::to_string(static_cast<int>(h.shift_mm[2])) + ") mm to MNI mm (half a voxel; validated against the ends' distance to the cortex)",
                                          "MNI mm to the anatomy's SCS metres by the anatomy's own NCS and SCS frames",
                                          "each streamline resampled to " + std::to_string(h.points) + " points at equal arc length (its two ends kept exactly)",
                                          "each end placed on the nearest vertex of '" + h.end_surface + "' when within " + std::to_string(h.max_end_distance_mm) + " mm; the distance of every end is stored"})},
              {"counts", Json{{"streamlines", n_stream}, {"points_per_streamline", h.points}, {"input_points", n_points}, {"ends_within_gate", n_gated}, {"median_end_distance_mm", median_mm}}}};
}

Json import_hcp(SubjectWriter& sub, const HcpSource& h, SourceChecker& sc) {
  if (h.end_surface.empty()) refuse("hcp1065: name the surface the ends are placed on (--end-surface): a choice of data, with no default");
  if (h.points < 2) refuse("hcp1065: points per streamline is at least 2");
  const auto surf = sub.find("manifold", "source/" + h.end_surface);
  if (!surf) refuse("hcp1065: the package has no surface '" + h.end_surface + "' to place the ends on");
  const Tractogram t = read_trk(sc.file(h.trk.filename().string()));
  std::vector<std::int64_t> kept;
  const std::vector<float> res = resample_streamlines(t, h.points, &kept);
  const std::int64_t nf = static_cast<std::int64_t>(kept.size()), np = h.points, nfp = nf * np;
  if (nf == 0) refuse("hcp1065: the .trk holds no streamline");
  // MNI mm -> the anatomy's metres, in place
  std::vector<float> pts(res.size());
  for (std::size_t i = 0; i < res.size() / 3; ++i) {
    double m[3];
    for (int k = 0; k < 3; ++k) m[k] = static_cast<double>(res[3 * i + static_cast<std::size_t>(k)]) + h.shift_mm[static_cast<std::size_t>(k)];
    for (int j = 0; j < 3; ++j) pts[3 * i + static_cast<std::size_t>(j)] = static_cast<float>(m[0] * h.mni_to_anatomy_m[static_cast<std::size_t>(j)] + m[1] * h.mni_to_anatomy_m[static_cast<std::size_t>(3 + j)] +
                                                                                           m[2] * h.mni_to_anatomy_m[static_cast<std::size_t>(6 + j)] + h.mni_to_anatomy_m[static_cast<std::size_t>(9 + j)]);
  }
  // the ends on the surface
  const std::vector<double> verts = read_positions(sub, "source/" + h.end_surface);
  const std::int64_t nv = static_cast<std::int64_t>(verts.size() / 3);
  const KdTree tree(verts);
  std::vector<std::int64_t> vertex(static_cast<std::size_t>(2 * nf));
  std::vector<float> dist_mm(static_cast<std::size_t>(2 * nf));
  std::int64_t within = 0;
  for (std::int64_t f = 0; f < nf; ++f)
    for (int e = 0; e < 2; ++e) {
      const std::int64_t row = f * np + (e ? np - 1 : 0);
      const double q[3] = {pts[static_cast<std::size_t>(3 * row)], pts[static_cast<std::size_t>(3 * row + 1)], pts[static_cast<std::size_t>(3 * row + 2)]};
      const auto [v, d2] = tree.nearest(q);
      vertex[static_cast<std::size_t>(e * nf + f)] = v;
      dist_mm[static_cast<std::size_t>(e * nf + f)] = static_cast<float>(std::sqrt(d2) * 1000.0);
    }
  std::vector<float> sorted(dist_mm);
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];
  // a fibre goes to the connectome only when BOTH ends are within the gate; an end is in the operator when it is
  std::vector<std::int64_t> ends_rows, ends_vertex;
  for (std::int64_t f = 0; f < nf; ++f)
    for (int e = 0; e < 2; ++e)
      if (dist_mm[static_cast<std::size_t>(e * nf + f)] <= h.max_end_distance_mm) {
        ends_rows.push_back(f * np + (e ? np - 1 : 0));
        ends_vertex.push_back(vertex[static_cast<std::size_t>(e * nf + f)]);
        ++within;
      }
  std::vector<std::int64_t> order(ends_rows.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int64_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
    const auto va = ends_vertex[static_cast<std::size_t>(a)], vb = ends_vertex[static_cast<std::size_t>(b)];
    return va != vb ? va < vb : ends_rows[static_cast<std::size_t>(a)] < ends_rows[static_cast<std::size_t>(b)];
  });
  std::vector<std::int64_t> indptr(static_cast<std::size_t>(nv + 1), 0), indices(order.size());
  for (auto v : ends_vertex) ++indptr[static_cast<std::size_t>(v) + 1];
  for (std::int64_t i = 0; i < nv; ++i) indptr[static_cast<std::size_t>(i) + 1] += indptr[static_cast<std::size_t>(i)];
  for (std::size_t i = 0; i < order.size(); ++i) indices[i] = ends_rows[static_cast<std::size_t>(order[i])];
  const std::vector<float> ones(order.size(), 1.0f);

  const Json files = sc.report()["files"];
  const Json origin_doc = hcp_origin_doc(h, files, static_cast<std::int64_t>(t.n_streamlines()), static_cast<std::int64_t>(t.points.size() / 3), within, median);
  const std::string node = nxr::io::folder_for("curve") + "/" + h.name;
  nxr::io::ManifoldSpec ms;
  ms.name = h.name;
  ms.path = node;
  ms.type = "curve";
  ms.n_vertices = nfp;
  ms.topology = {"path", 1, static_cast<int>(nf), std::nullopt, {}};
  nxr::io::GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = node + "/points";
  ms.geometry = g;
  ms.cols.set("n_edges", nf * (np - 1));
  ms.cols.set("source", h.trk.filename().string());
  ms.cols.json("producer_json", Json{{"points_per_component", np}, {"origin", origin_doc},
                                      {"description", std::to_string(nf) + " streamlines of " + std::to_string(np) + " points each (streamline-major), the anatomy's SCS in metres"}});
  ms.populate = [&](nxr::io::Store&, const std::string&) {
    nxr::io::ArrayOptions ao;
    ao.compress = true;
    sub.write_array(node + "/points", view(pts, {nfp, 3}), ao);
    sub.write_array(node + "/end_distance_mm", view(dist_mm, {2 * nf}), ao);   // each end's distance to the end surface, before any gate: ends of streamline f are rows f and nf + f
  };
  const std::string fid = sub.manifold(std::move(ms));
  nxr::io::OperatorSpec os;
  os.name = h.name + "_ends";
  os.path = os.name;
  os.kind = "fibre ends";
  os.from_manifold_id = fid;
  os.layout = "csr";
  os.n_rows = nv;
  os.n_cols = nfp;
  const std::int64_t nnz = static_cast<std::int64_t>(ends_rows.size());
  os.cols.set("to_manifold_id", nxr::io::text(*surf, "id")).set("nnz", nnz)
      .set("description", "each streamline end within the gate -> its nearest vertex of " + h.end_surface + " (1 per end)").set("source", h.trk.filename().string());
  os.cols.json("producer_json", Json{{"method", "nearest vertex, Euclidean, SCS metres"}, {"end_surface", h.end_surface}, {"max_end_distance_mm", h.max_end_distance_mm},
                                      {"ends_total", 2 * nf}, {"ends_within_gate", within}, {"origin", origin_doc}});
  os.sparse = nxr::io::SparseArrays{{{"indptr", view(indptr, {nv + 1})}, {"indices", view(indices, {nnz})}, {"data", view(ones, {nnz})}}};
  sub.operator_(std::move(os));
  return Json{{"name", h.name}, {"streamlines", nf}, {"dropped", t.n_streamlines() - nf}, {"points", np}, {"ends", 2 * nf}, {"ends_within_gate", within}, {"median_end_distance_mm", median},
              {"end_surface", h.end_surface}, {"max_end_distance_mm", h.max_end_distance_mm}};
}

}  // namespace

// ── building a template package ─────────────────────────────────────────────────────────────────────────────────

Json build_template(const BuildRequest& rq) {
  const bool icbm = rq.recipe.rfind("icbm152", 0) == 0;   // icbm152 (Brainstorm's current default anatomy) or icbm152_2023b
  if (rq.recipe != "fsaverage" && !icbm) refuse("recipe '" + rq.recipe + "': choose fsaverage, icbm152 or icbm152_2023b");
  if (rq.hcp_trk && !icbm) refuse("the streamlines are imported into the ICBM152 template (their coordinates are MNI's): use the icbm152 recipe");
  if (rq.hcp_trk && rq.hcp_end_surface.empty()) refuse("--hcp-trk needs --hcp-end-surface: the surface the ends are placed on is a choice of data");
  if (fs::exists(rq.package)) refuse(rq.package.string() + ": exists (a package is made new)");
  SourceChecker sc(rq.recipe, rq.sources, rq.accept_unlisted_sources);
  const std::string name = rq.name.empty() ? sc.recipe()["template"].get<std::string>() : rq.name;
  nxr::io::CreatePackageOptions po;
  po.home_dataset = rq.home_dataset;
  po.subject.kind = "template";
  po.subject.created_by = rq.created_by;
  po.subject.source_format = rq.recipe == "fsaverage" ? "freesurfer" : "brainstorm";
  po.built_by = rq.created_by;
  nxr::io::DatasetWriter w = nxr::io::create_package(rq.package, name, po);
  Json info;
  std::optional<SourceChecker> hcp_checker;
  try {
    SubjectWriter sub = SubjectWriter::open(w, name);
    sub.compose([&] {
      if (rq.recipe == "fsaverage") {
        FreeSurferSource s;
        s.folder = rq.sources;
        if (!rq.surfaces.empty()) s.surfaces = rq.surfaces;
        s.annotations = rq.annotations;
        s.cortex_mask = rq.cortex_mask;
        info = import_freesurfer(sub, s, sc);
        if (rq.scalp) {
          // MNE-Python's fsaverage scalp (`bem/outer_skin.surf`, FreeSurfer surface format, mm) and its fiducials: for placement only, not a boundary-element model
          nxr::io::Mesh sk = nxr::io::read_freesurfer_surface(sc.file("bem/outer_skin.surf"));
          for (double& x : sk.vertices) x /= 1000.0;
          nxr::io::SurfaceOptions o;
          o.name = "head";
          o.primary = false;
          o.source = "fsaverage bem/outer_skin.surf";
          o.cs_key = "freesurfer-tkr";
          o.producer = Json{{"source", "fsaverage bem/outer_skin.surf"}, {"units", "m"}, {"use", "scalp for placing the standard electrode montages; not a boundary-element model"}};
          const auto si = nxr::io::add_surface(sub, o, sk.vertices, sk.faces);
          const HeadLandmarks lm = landmarks_from_fif(sc.file("bem/fsaverage-fiducials.fif"));
          write_head_landmarks(sub, "freesurfer-tkr", lm, Json{{"source", "fsaverage bem/fsaverage-fiducials.fif"}, {"frame", "FreeSurfer surface RAS (FIF frame 5), metres"}, {"points", "NAS, LPA, RPA"}});
          info["scalp"] = Json{{"name", si.name}, {"n_vertices", si.n_vertices}, {"n_faces", si.n_faces}, {"landmarks", "head_landmarks"}};
        }
      } else {
        BrainstormAnatomySource s;
        s.folder = rq.sources;
        if (!rq.surfaces.empty()) s.surfaces = rq.surfaces;
        s.atlases = rq.annotations;
        info = import_bst_anatomy(sub, s, sc);
        if (rq.scalp) {
          const auto si = nxr::bst::export_surface(sub, sc.file("tess_head.mat"), "", false, std::vector<std::string>());
          const HeadLandmarks lm = landmarks_from_subjectimage(sc.file("subjectimage_T1.mat"));
          write_head_landmarks(sub, "scs", lm, Json{{"source", "subjectimage_T1.mat SCS.NAS, SCS.LPA, SCS.RPA"}, {"frame", "scs (metres)"}, {"points", "NAS, LPA, RPA"}});
          info["scalp"] = Json{{"name", si.name}, {"n_vertices", si.n_vertices}, {"n_faces", si.n_faces}, {"landmarks", "head_landmarks"}};
        }
        if (rq.hcp_trk) {
          HcpSource h;
          h.trk = *rq.hcp_trk;
          h.end_surface = rq.hcp_end_surface;
          h.max_end_distance_mm = rq.hcp_max_end_distance_mm;
          h.points = rq.hcp_points;
          h.mni_to_anatomy_m = rq.hcp_affine ? *rq.hcp_affine : mni_to_scs_from_subjectimage(sc.file("subjectimage_T1.mat"));
          hcp_checker.emplace("hcp1065", rq.hcp_trk->parent_path(), rq.accept_unlisted_sources);
          info["hcp1065"] = import_hcp(sub, h, *hcp_checker);
        }
      }
    });
    // the origin of every item
    const Json sources = sc.report();
    std::string files_text;
    for (const auto& e : sources["files"]) files_text += (files_text.empty() ? "" : ", ") + e["path"].get<std::string>() + " (sha256 " + e["sha256"].get<std::string>().substr(0, 12) + "...)";
    nxr::io::OriginRecord base;
    if (rq.recipe == "fsaverage") base = freesurfer_origin("FreeSurfer fsaverage, read natively by nxr-neuro from: " + files_text);
    else {
      base = {"Brainstorm default anatomy ICBM152 (ICBM 2009c nonlinear asymmetric), read natively by nxr-neuro from: " + files_text,
              "MNI/ICBM152 permissive notice (use, copy, modify and distribute for any purpose, notice kept); cortex surfaces: FreeSurfer Software License Agreement Part B v1.0",
              "Copyright (C) 1993-2004 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. Fonov VS et al. (2011) NeuroImage 54:313-327; "
              "Fonov VS et al. (2009) NeuroImage 47:S102; Brainstorm: Tadel F et al. (2011) Comput Intell Neurosci 879716. Converted to the nxr package format by nxr-neuro (MODIFIED)",
              true};
    }
    const std::set<std::string> atlas_names(rq.annotations.begin(), rq.annotations.end());
    Json terms = Json::array();
    for (const auto& it : nxr::io::package_items(w.db())) {
      nxr::io::OriginRecord o = base;
      if (it.table == "selection" && icbm && atlas_names.count(it.name)) {
        const AtlasTerms t = atlas_terms(it.name);
        o.source = "the '" + it.name + "' parcellation of the Brainstorm default anatomy ICBM152, read natively by nxr-neuro from: " + files_text;
        o.licence = t.licence;
        o.attribution = t.attribution;
        o.redistributable = t.redistributable && !t.forbidden;
        terms.push_back(Json{{"atlas", it.name}, {"licence", t.licence}, {"redistributable", o.redistributable}});
      } else if (it.table == "manifold" && it.name.rfind("hcp1065", 0) == 0) {
        o.source = "HCP-1065 population-averaged tractography (DSI Studio; Yeh), read natively by nxr-neuro from: " + (hcp_checker ? hcp_checker->report()["files"][0]["path"].get<std::string>() + " (sha256 " + hcp_checker->report()["files"][0]["sha256"].get<std::string>().substr(0, 12) + "...)" : "");
        o.licence = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/) together with the WU-Minn HCP Open Access Data Use Terms: ShareAlike; redistribution only under the same terms";
        o.attribution = "Yeh F-C (2022) Nat Commun 13:4933, doi:10.1038/s41467-022-32595-4. Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; "
                        "1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.";
        o.redistributable = true;
      }
      if (it.name == "head" || it.name == "head_landmarks") {
        o.source = std::string(it.name == "head" ? "the scalp surface" : "the landmarks") + " of " + (rq.recipe == "fsaverage" ? "MNE-Python's fsaverage (bem/outer_skin.surf, bem/fsaverage-fiducials.fif; derived from FreeSurfer's fsaverage MRI)" : "the Brainstorm default anatomy ICBM152 (tess_head.mat, subjectimage_T1.mat)") +
                   ", read natively by nxr-neuro from: " + files_text;
      }
      nxr::io::set_origin(w.db(), it.table, it.id, o);
    }
    // the operator of the streamlines' ends is HCP's too
    if (hcp_checker)
      for (const auto& it : nxr::io::package_items(w.db()))
        if (it.table == "operator" && it.name.rfind("hcp1065", 0) == 0) {
          nxr::io::OriginRecord o = *nxr::io::origin_of(w.db(), "manifold", nxr::io::text(*w.db().one("SELECT id FROM manifold WHERE name = 'hcp1065'"), "id"));
          nxr::io::set_origin(w.db(), it.table, it.id, o);
        }
    info["atlas_terms"] = terms;
    info["sources"] = sources;
    if (hcp_checker) info["hcp_sources"] = hcp_checker->report();
  } catch (...) {
    const fs::path folder = w.root();
    try { w.close(); } catch (...) {}
    std::error_code ec;
    fs::remove_all(folder, ec);
    throw;
  }
  w.close();
  info["package"] = rq.package.string();
  info["name"] = name;
  info["content_hash"] = nxr::io::SubjectPackage::open(rq.package).record().content_hash;
  info["items"] = nxr::io::package_items(nxr::io::SubjectPackage::open(rq.package).db()).size();
  return info;
}

}  // namespace nxr::convert

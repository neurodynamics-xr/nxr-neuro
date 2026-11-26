// nxr::bst — a Brainstorm subject: nxr.convert.surface (export_surface, hemispheres, the parcellations) and nxr.convert.subject (condition_plan, compose_subject,
// convert_subject, convert_template, convert_group).
#include "nxr/bst/subject.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <set>

#include "nxr/bst/mat.h"
#include "nxr/bst/fibers.h"
#include "nxr/bst/mri.h"
#include "nxr/bst/maps.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/session.h"
#include "nxr/bst/timefreq.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::AtlasMember;
using nxr::io::Hemisphere;
using nxr::io::SubjectWriter;

namespace {

bool is_dir(const fs::path& p) { std::error_code ec; return fs::is_directory(p, ec); }
bool is_file(const fs::path& p) { std::error_code ec; return fs::is_regular_file(p, ec); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// str(x) of a char; "" for anything else (Python's `str(a.get("Name", ""))`)
std::string raw_string(const Value* v) {
  if (!v || !v->is_char()) return "";
  return v->string();
}

[[noreturn]] void value_error(const std::string& m) { throw Error(code::invalid_argument, m); }

// A struct array's elements as (value, element) pairs; anything but a struct has none (struct_rows)
struct Rec {
  const Value* st;
  std::size_t e;
  const Value* get(const char* k) const { return st->field(k, e); }
};
std::vector<Rec> struct_rows(const Value* v) {
  std::vector<Rec> out;
  if (!v || !v->is_struct()) return out;
  for (std::size_t e = 0; e < v->numel(); ++e) out.push_back({v, e});
  return out;
}

std::vector<std::int64_t> vertex_indices(const Value* v) {   // np.asarray(.., int64).ravel()
  if (!v || !v->is_numeric()) value_error("Vertices: not a numeric array");
  std::vector<std::int64_t> out;
  for (double x : v->doubles()) out.push_back(static_cast<std::int64_t>(x));
  return out;
}

// side_of: 'Cortex L' -> left, 'Cortex R' -> right
std::optional<std::string> side_of(const std::string& name) {
  static const std::regex left(R"(\bL\b|left)", std::regex::icase), right(R"(\bR\b|right)", std::regex::icase);
  if (std::regex_search(name, left)) return "left";
  if (std::regex_search(name, right)) return "right";
  return std::nullopt;
}

// hemispheres: (label, first vertex, count) in vertex order when exactly two left/right scouts of Structures tile the vertices in two runs
std::optional<std::vector<Hemisphere>> hemispheres(const std::vector<Rec>& entries, std::int64_t n_v) {
  const Rec* st = nullptr;
  for (const auto& a : entries) if (lower(raw_string(a.get("Name"))) == "structures") { st = &a; break; }
  if (!st) return std::nullopt;
  std::vector<Hemisphere> runs;
  for (const Rec& s : struct_rows(st->get("Scouts"))) {
    const auto side = side_of(raw_string(s.get("Label")));
    std::vector<std::int64_t> v = vertex_indices(s.get("Vertices"));
    for (auto& x : v) x -= 1;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    if (!side || v.empty()) continue;
    if (v.back() - v.front() + 1 != static_cast<std::int64_t>(v.size())) return std::nullopt;
    runs.push_back({*side, v.front(), static_cast<std::int64_t>(v.size())});
  }
  std::stable_sort(runs.begin(), runs.end(), [](const Hemisphere& a, const Hemisphere& b) { return a.first < b.first; });
  if (runs.size() != 2 || runs[0].label == runs[1].label || runs[0].first != 0 || runs[0].count != runs[1].first || runs[1].first + runs[1].count != n_v) return std::nullopt;
  return runs;
}

// _export_atlas_set: the parcellations as members (family, codes, level names, colours)
std::vector<AtlasMember> atlas_members(const std::vector<Rec>& entries, std::int64_t n_v, const std::optional<std::vector<std::string>>& wanted) {
  std::vector<AtlasMember> out;
  if (entries.empty()) return out;
  std::vector<std::string> names_all;
  for (const auto& a : entries) names_all.push_back(raw_string(a.get("Name")));
  std::vector<std::size_t> idx;
  if (wanted) {   // an explicit list, an empty one included (none but Structures); nullopt: every atlas with two or more scouts
    std::vector<std::string> missing;
    for (const auto& w : *wanted) if (std::find(names_all.begin(), names_all.end(), w) == names_all.end()) missing.push_back(w);
    if (!missing.empty()) {
      std::string m = "unknown atlas name(s): [";
      for (std::size_t k = 0; k < missing.size(); ++k) m += (k ? ", '" : "'") + missing[k] + "'";
      value_error(m + "]");
    }
    for (const auto& w : *wanted) idx.push_back(static_cast<std::size_t>(std::find(names_all.begin(), names_all.end(), w) - names_all.begin()));
  } else {
    for (std::size_t i = 0; i < entries.size(); ++i)
      if (struct_rows(entries[i].get("Scouts")).size() >= 2) idx.push_back(i);
  }
  for (std::size_t i = 0; i < names_all.size(); ++i)   // Structures is the L/R partition: always included
    if (lower(names_all[i]) == "structures" && std::find(idx.begin(), idx.end(), i) == idx.end()) idx.push_back(i);
  if (idx.empty()) return out;
  for (std::size_t i : idx) {
    AtlasMember m;
    m.family = nxr::io::sanitize_node_name(names_all[i]);
    m.codes.assign(static_cast<std::size_t>(n_v), 0);
    m.levels = {"unassigned"};
    m.colors = {{0, 0, 0}};   // `unassigned` is a real label (0), drawn transparent
    std::int32_t k = 0;
    for (const Rec& s : struct_rows(entries[i].get("Scouts"))) {
      ++k;
      for (std::int64_t v : vertex_indices(s.get("Vertices"))) {
        std::int64_t x = v - 1;
        if (x < 0) x += n_v;   // numpy's negative index
        if (x < 0 || x >= n_v) value_error("index " + std::to_string(v - 1) + " is out of bounds for axis 0 with size " + std::to_string(n_v));
        m.codes[static_cast<std::size_t>(x)] = k;
      }
      m.levels.push_back(raw_string(s.get("Label")));
      std::vector<double> c = {0.5, 0.5, 0.5};
      if (const Value* col = s.get("Color")) {
        if (!col->is_numeric()) value_error("Color: not a numeric array");
        c = col->doubles();
        if (c.size() > 3) c.resize(3);
        if (c.size() < 3) value_error(names_all[i] + ": a scout's Color holds " + std::to_string(c.size()) + " value(s), not 3");
      }
      std::array<int, 3> rgb{};
      for (int j = 0; j < 3; ++j) rgb[static_cast<std::size_t>(j)] = static_cast<int>(std::nearbyint(c[static_cast<std::size_t>(j)] * 255.0));
      m.colors.push_back(rgb);
    }
    out.push_back(std::move(m));
  }
  return out;
}

// the path of `file` under `<bst_root>/anat`, or the file as given
std::string source_of(const fs::path& surface_file, const std::string& bst_root) {
  if (bst_root.empty()) return "";
  std::error_code ec;
  const fs::path f = fs::weakly_canonical(surface_file, ec), a = fs::weakly_canonical(fs::path(bst_root) / "anat", ec);
  auto fi = f.begin(), ai = a.begin();
  for (; ai != a.end() && fi != f.end() && *fi == *ai; ++ai, ++fi) {}
  if (ai == a.end() && fi != f.end()) {
    fs::path rel;
    for (; fi != f.end(); ++fi) rel /= *fi;
    return rel.generic_string();
  }
  return py_path(surface_file.string());
}

}  // namespace

nxr::io::SurfaceInfo export_surface(SubjectWriter& sub, const fs::path& surface_file, const std::string& bst_root, bool primary,
                                    const std::optional<std::vector<std::string>>& atlases) {
  const std::string fname = name_of(surface_file.string());
  if (fname.rfind("tess_fibers", 0) == 0)
    value_error(fname + ": fibres are curves, not a surface \xe2\x80\x94 `nxr-io convert subject` converts them");
  const auto vars = read_variables(surface_file, {"Vertices", "Faces", "Comment", "Atlas", "Reg"});
  auto var = [&](const char* k) -> const Value* { auto it = vars.find(k); return it == vars.end() ? nullptr : &it->second; };
  const std::string name = surface_node_name(surface_file);
  const std::string folder = nxr::io::folder_for("surface", name);

  const Value* V = var("Vertices");
  if (!V) throw Error(code::no_such_row, "Vertices");
  const Value* F = var("Faces");
  if (!F) throw Error(code::no_such_row, "Faces");
  if (!V->is_numeric() || V->dims.size() != 2 || V->cols() != 3) value_error(fname + ": Vertices is not an nV x 3 numeric array");
  if (!F->is_numeric() || F->dims.size() != 2 || F->cols() != 3) value_error(fname + ": Faces is not an nF x 3 numeric array");
  const std::int64_t n_v = static_cast<std::int64_t>(V->rows());
  const std::vector<double> vertices = V->row_major();
  std::vector<std::int32_t> faces;
  for (double x : F->row_major()) faces.push_back(static_cast<std::int32_t>(static_cast<std::int64_t>(x) - 1));

  nxr::io::SurfaceOptions opt;
  opt.name = name;
  opt.primary = primary;
  const std::string source = source_of(surface_file, bst_root);
  if (!bst_root.empty()) opt.source = source;
  if (const Value* c = var("Comment")) if (c->is_char()) { const std::string s = c->string(); if (!s.empty()) opt.comment = s; }

  if (const Value* reg = var("Reg")) {
    if (reg->is_struct()) {
      const Value* sph = reg->field("Sphere");
      const Value* sv = sph && sph->is_struct() ? sph->field("Vertices") : nullptr;
      if (sv && sv->is_numeric() && sv->numel() > 0 && sv->dims.size() == 2 && static_cast<std::int64_t>(sv->rows()) == n_v) {
        opt.sphere = sv->row_major();
        opt.sphere_cols = static_cast<std::int64_t>(sv->cols());
        opt.sphere_source = "Surf.Reg.Sphere";
      }
    }
  }
  const std::vector<Rec> entries = struct_rows(var("Atlas"));
  if (folder == nxr::io::folder_for("surface", "cortex"))
    if (auto h = hemispheres(entries, n_v)) opt.hemispheres = std::move(*h);
  opt.atlases = atlas_members(entries, n_v, atlases);
  return nxr::io::add_surface(sub, opt, vertices, faces);
}

Json surface_json(const nxr::io::SurfaceInfo& info) {
  Json flipped = Json::array();
  for (auto c : info.flipped) flipped.push_back(c);
  return Json{{"name", info.name}, {"folder", info.folder}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"atlases", info.atlases},
              {"registration", info.registration}, {"hemispheres", info.hemispheres}, {"flipped", flipped}, {"mesh_health", info.mesh_health}};
}

std::pair<fs::path, bool> anat_dir(const fs::path& root, const std::string& subject) {
  const fs::path own = root / "anat" / subject;
  const fs::path bs = own / "brainstormsubject.mat";
  if (is_file(bs)) {
    const auto v = read_variable(bs, "UseDefaultAnat");
    if (v && v->is_numeric() && !v->re.empty() && static_cast<std::int64_t>(v->scalar()) == 1) return {root / "anat" / "@default_subject", true};
  }
  return {own, false};
}

std::optional<std::string> kernel_surface(const fs::path& cond_dir, const fs::path& raw_dir, const fs::path& anat) {
  std::set<std::string> found;
  for (const fs::path& home : {cond_dir, raw_dir}) {
    if (!is_dir(home)) continue;
    for (const auto& kf : glob_names(home, "results_", ".mat", "KERNEL")) {
      if (kf.rfind("._", 0) == 0) continue;
      const auto sf = read_variable(home / kf, "SurfaceFile");
      if (sf && sf->is_char() && !sf->string().empty() && is_file(anat / name_of(sf->string()))) found.insert(name_of(sf->string()));
    }
  }
  if (found.size() == 1) return *found.begin();
  return std::nullopt;
}

namespace {

// str(list of {what, reason}) as Python prints it
std::string py_repr_skipped(const Json& skipped) {
  std::string o = "[";
  for (std::size_t i = 0; i < skipped.size(); ++i)
    o += std::string(i ? ", " : "") + "{'what': " + str_repr(skipped[i]["what"].get<std::string>()) + ", 'reason': " + str_repr(skipped[i]["reason"].get<std::string>()) + "}";
  return o + "]";
}

bool is_one_of(const Error& e, std::initializer_list<const char*> codes) {
  for (const char* c : codes) if (std::string(e.code()) == c) return true;
  return false;
}

// (convertible conditions, every condition folder name) for the subject
std::pair<std::vector<std::string>, std::vector<std::string>> conditions(const fs::path& root, const std::string& subject) {
  std::vector<std::string> convertible, folders;
  for (const Json& s : list_subjects(root))
    if (s["subject"].get<std::string>() == subject) {
      convertible.clear();
      for (const Json& c : s["conditions"])
        if (!c["recordings"].empty() || !c["raw_links"].empty()) convertible.push_back(c["name"].get<std::string>());
    }
  std::sort(convertible.begin(), convertible.end());
  const fs::path data = root / "data" / subject;
  if (is_dir(data))
    for (const auto& d : fs::directory_iterator(data)) {
      const std::string n = d.path().filename().string();
      if (!d.is_directory() || (n.rfind("@", 0) == 0 && n.rfind("@raw", 0) != 0)) continue;
      folders.push_back(n.rfind("@raw", 0) == 0 ? n.substr(4) : n);
    }
  std::sort(folders.begin(), folders.end());
  folders.erase(std::unique(folders.begin(), folders.end()), folders.end());
  return {convertible, folders};
}

std::vector<fs::path> files_of(const fs::path& dir, const std::string& prefix, const std::string& suffix, const std::string& middle = "") {
  std::vector<fs::path> out;
  for (const auto& n : glob_names(dir, prefix, suffix, middle)) if (n.rfind("._", 0) != 0) out.push_back(dir / n);
  return out;
}

Json without_id(Json j) {
  if (j.is_object()) j.erase("id");
  return j;
}

Json with_stage(const char* stage, const Json& body) {
  Json line = Json{{"stage", stage}};
  for (auto it = body.begin(); it != body.end(); ++it) line[it.key()] = it.value();
  return line;
}

}  // namespace

ConditionPlan condition_plan(const fs::path& root, const std::string& subject, const std::string& condition, const ConditionPlanOptions& opt) {
  const fs::path cond = root / "data" / subject / condition;
  const fs::path raw_dir = root / "data" / subject / ("@raw" + condition);
  const fs::path anat = anat_dir(root, subject).first;
  std::string surf_name;
  if (opt.surface && !opt.surface->empty()) surf_name = *opt.surface;
  else if (const auto k = kernel_surface(cond, raw_dir, anat); k && !k->empty()) surf_name = *k;
  else if (const auto c = index_cortex(root, subject); c && !c->empty()) surf_name = *c;
  else surf_name = "tess_cortex_pial_low.mat";
  ConditionPlan plan;
  plan.surface_file = anat / surf_name;
  for (const auto& e : opt.extra_surfaces) if (e != surf_name) plan.extra_surfaces.push_back(anat / e);
  const std::vector<fs::path> raws = files_of(raw_dir, "data_0raw_", ".mat");
  const bool has_raw = is_dir(raw_dir) && !raws.empty();
  const bool include_raw = has_raw && !opt.no_raw;
  if (opt.recording) plan.data_files = {cond / *opt.recording};
  else if (opt.include_imported || !has_raw) plan.data_files = files_of(cond, "data_", ".mat");
  if (plan.data_files.empty() && !include_raw) value_error("nothing to import in " + condition + ": no raw link and no data_*.mat");
  std::vector<fs::path> chans = files_of(cond, "channel", ".mat");
  if (chans.empty()) chans = files_of(raw_dir, "channel", ".mat");
  if (opt.channel_file) {
    std::error_code ec;
    plan.channel_file = (fs::exists(cond / *opt.channel_file, ec) ? cond : raw_dir) / *opt.channel_file;
  } else if (!chans.empty()) {
    plan.channel_file = chans.front();
  } else {
    value_error("no channel*.mat in " + condition + " (either home)");
  }
  std::set<fs::path> kernels;
  for (const fs::path& home : {cond, raw_dir}) for (const auto& k : files_of(home, "results_", ".mat", "KERNEL")) kernels.insert(k);
  plan.kernel_files.assign(kernels.begin(), kernels.end());
  if (include_raw) {
    if (raws.size() != 1) value_error(std::to_string(raws.size()) + " raw link(s) in @raw" + condition);
    plan.raw_data_file = raws.front();
  }
  return plan;
}

std::string subject_cortex(const fs::path& root, const std::string& subject, SubjectWriter& sub) {
  if (const auto found = index_cortex(root, subject); found && !found->empty()) return name_of(*found);
  const fs::path bs = root / "anat" / subject / "brainstormsubject.mat";
  if (is_file(bs))
    if (const auto c = read_variable(bs, "Cortex"); c && c->is_char() && !c->string().empty()) return name_of(c->string());
  if (const auto prim = sub.db().one("SELECT name FROM manifold WHERE subject_id = ? AND type = 'surface' AND is_primary = 1", {nxr::io::V(sub.id())}))
    return "tess_" + nxr::io::text(*prim, "name") + ".mat";
  value_error(subject + ": cannot tell which cortex the fibres were assigned on");
}

bool or_skip_unreadable_v73(SubjectWriter& sub, const Emit& emit, const std::function<void()>& body) {
  const std::size_t mark = sub.mark();
  try {
    body();
    return true;
  } catch (const UnreadableV73& e) {
    sub.rollback_to(mark);
    const std::string file = py_path(e.file.string());
    note_skipped(file, kUnreadableV73);
    if (emit) emit(Json{{"stage", "skipped"}, {"what", file}, {"reason", kUnreadableV73}});
    return false;
  }
}

Json compose_subject(SubjectWriter& sub, const fs::path& root, const std::string& subject, const Emit& emit, std::optional<double> resample) {
  Json sessions = Json::array(), volumes = Json::array(), maps = Json::array(), fibers = Json::array(), connectomes = Json::array(), skipped = Json::array(), truncated = Json::array();
  auto skip = [&](const std::string& what, const std::string& reason) {
    skipped.push_back(Json{{"what", what}, {"reason", reason}});
    emit(Json{{"stage", "skipped"}, {"what", what}, {"reason", reason}});
  };

  const auto [convertible, folders] = conditions(root, subject);
  if (convertible.empty()) value_error(subject + ": no condition with recordings or a raw link \xe2\x80\x94 nothing to anchor a subject");
  // 1. sessions: a condition that cannot convert (an unsupported raw link, no channels ...) is rolled back and REPORTED
  for (const auto& cond : convertible) {
    const std::size_t mark = sub.mark();
    try {
      const ConditionPlan plan = condition_plan(root, subject, cond);
      const Json cr = convert_condition(sub, root, cond, plan, ConvertOptions{true, false, resample}, emit);
      sessions.push_back(cond);
      if (cr.contains("truncated")) { Json w = cr["truncated"]; w["condition"] = cond; truncated.push_back(w); }
    } catch (const Error& e) {
      if (!is_one_of(e, {code::invalid_argument, code::not_found, code::unsupported})) throw;
      sub.rollback_to(mark);
      skip("condition " + cond, e.python_name() + ": " + e.what());
    }
  }
  if (sessions.empty()) value_error(subject + ": no condition converted \xe2\x80\x94 " + py_repr_skipped(skipped));
  // 2. subject images
  const fs::path anat = anat_dir(root, subject).first;
  try {
    for (const Json& v : import_subjectimages(sub, anat, std::nullopt, emit)) volumes.push_back(v["name"]);
  } catch (const Error& e) {
    if (std::string(e.code()) != code::not_found) throw;
    skip("subject images", e.what());
  }
  // 3. source maps, in every condition folder (map-only ones included)
  for (const auto& cond : folders)
    for (const fs::path& home : {root / "data" / subject / cond, root / "data" / subject / ("@raw" + cond)}) {
      std::vector<std::pair<fs::path, bool>> todo;   // (file, a matrix series)
      if (is_dir(home)) {
        for (const auto& q : find_source_maps(home)) todo.push_back({q, false});
        for (const auto& q : files_of(home, "matrix_", ".mat")) if (is_matrix_series(q)) todo.push_back({q, true});
      }
      for (const auto& [mf, series] : todo) {
        const std::size_t mark = sub.mark();
        try {   // the maps first: a confounds matrix takes its BOLD map's time Line
          or_skip_unreadable_v73(sub, emit, [&] {
            const Json info = series ? export_matrix_series(sub, mf, cond, &recording_map_hooks()) : export_source_map(sub, mf, root, cond, false, std::nullopt, &recording_map_hooks());
            maps.push_back(info["name"]);
            emit(with_stage("map", without_id(info)));
          });
        } catch (const Error& e) {
          if (!is_one_of(e, {code::invalid_argument, code::not_found})) throw;
          sub.rollback_to(mark);
          skip(mf.filename().string(), e.what());
        }
      }
    }
  // 4. fibres: ends on the subject's cortex (iCortex), connectomes per assigned atlas
  std::vector<fs::path> fibre_files;
  for (const auto& q : files_of(anat, "tess_", ".mat")) if (is_fibers_file(q)) fibre_files.push_back(q);
  if (!fibre_files.empty()) {
    const std::string cortex_file = subject_cortex(root, subject, sub);
    const std::string cname = surface_node_name(cortex_file);
    const std::string cpath = nxr::io::folder_for("surface", cname) + "/" + cname;
    const bool cortex = sub.find("manifold", cpath) || or_skip_unreadable_v73(sub, emit, [&] { export_surface(sub, anat / cortex_file, root.string(), false); });
    for (const auto& ff : fibre_files) {
      if (!cortex) {   // the fibres' ends lie on the cortex this build could not read
        note_skipped(py_path(ff.string()), std::string("its cortex ") + cortex_file + " was skipped (" + kUnreadableV73 + ")");
        emit(Json{{"stage", "skipped"}, {"what", py_path(ff.string())}, {"reason", "its cortex was skipped"}});
        continue;
      }
      or_skip_unreadable_v73(sub, emit, [&] {   // a fibres file is often v7.3: skipped alone where it cannot be read
        const Json info = export_fibers(sub, ff, subject, cpath, anat / cortex_file, emit);
        fibers.push_back(info["name"]);
        connectomes.push_back(info["connectome"]);
      });
    }
  }
  Json res{{"subject", subject}, {"sessions", sessions}, {"volumes", volumes}, {"maps", maps}, {"fibers", fibers}, {"connectomes", connectomes}, {"skipped", skipped}};
  if (!truncated.empty()) res["truncated"] = truncated;   // raw recordings whose binary was cut short, each imported to its last whole sample
  return res;
}

nxr::io::SubjectOptions subject_options(const fs::path& root, const std::string& subject, const std::string& created_by) {
  nxr::io::SubjectOptions so;
  so.source_format = "brainstorm";
  so.source_path = py_path(anat_dir(root, subject).first.string());
  so.created_by = created_by;
  return so;
}
nxr::io::SubjectOptions template_options(const fs::path& root, const std::string& created_by) {
  nxr::io::SubjectOptions so;
  so.kind = "template";
  so.source_format = "brainstorm";
  so.source_path = py_path((root / "anat" / "@default_subject").string());
  so.created_by = created_by;
  return so;
}

namespace {
// the subject an import fills: created in a one-database dataset; in a package (a catalogue's member made for this import, with the same row) the package's own subject
SubjectWriter subject_to_fill(nxr::io::DatasetWriter& ds, const std::string& name, const nxr::io::SubjectOptions& so) {
  if (!ds.is_package()) return SubjectWriter::create(ds, name, so);
  nxr::io::OpenSubjectOptions oo;
  oo.created_by = so.created_by;
  return SubjectWriter::open(ds, name, oo);
}
}  // namespace

Json convert_subject(nxr::io::DatasetWriter& ds, const fs::path& root, const std::string& subject, const Emit& emit, const std::string& created_by,
                     std::optional<double> resample) {
  const auto anat_pair = anat_dir(root, subject);
  SubjectWriter sub = subject_to_fill(ds, subject, subject_options(root, subject, created_by));
  Json summary;
  sub.compose([&] { summary = compose_subject(sub, root, subject, emit, resample); });

  Json rows = Json::object();
  for (const char* t : {"manifold", "field", "operator", "selection"})
    rows[t] = nxr::io::integer(*ds.db().one(std::string("SELECT count(*) AS n FROM ") + t + " WHERE subject_id = ?", {sub.id()}), "n");
  Json done = Json{{"stage", "subject-done"}, {"subject", subject}, {"id", sub.id()}, {"store", sub.folder().string()}, {"rows", rows},
                   {"anatomy", anat_pair.second ? "@default_subject" : "own"}};
  for (auto it = summary.begin(); it != summary.end(); ++it) if (it.key() != "subject") done[it.key()] = it.value();
  emit(done);
  Json result = summary;
  result["id"] = sub.id();
  result["rows"] = rows;
  return result;
}

Json convert_template(nxr::io::DatasetWriter& ds, const fs::path& root, const std::optional<std::vector<std::string>>& surfaces, const Emit& emit, const std::string& created_by) {
  const fs::path src = root / "anat" / "@default_subject";
  if (!is_dir(src)) throw Error(code::not_found, py_path(src.string()) + ": not in this protocol");
  std::vector<std::string> files;
  if (surfaces && !surfaces->empty()) files = *surfaces;
  else {
    for (const auto& n : glob_names(src, "tess_cortex_", "_low.mat")) if (n.rfind("._", 0) != 0) files.push_back(n);
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
      const int ka = a == "tess_cortex_pial_low.mat" ? 0 : 1, kb = b == "tess_cortex_pial_low.mat" ? 0 : 1;
      return ka != kb ? ka < kb : a < b;
    });
  }
  if (files.empty()) throw Error(code::not_found, py_path(src.string()) + ": no tess_cortex_*_low.mat to export as the template's cortex");
  SubjectWriter sub = subject_to_fill(ds, kTemplateSubject, template_options(root, created_by));
  Json names = Json::array(), vols = Json::array(), maps = Json::array(), skipped = Json::array();
  sub.compose([&] {
    for (std::size_t k = 0; k < files.size(); ++k) {
      const auto one = [&] {
        const auto info = export_surface(sub, src / files[k], root.string(), names.empty());   // the first surface read is the primary one
        names.push_back(info.name);
        emit(with_stage("surface", surface_json(info)));
      };
      if (k + 1 < files.size() || !names.empty()) or_skip_unreadable_v73(sub, emit, one);
      else one();   // the last chance of a cortex: refused as it stands
    }
    for (const Json& v : import_subjectimages(sub, src, std::nullopt, emit)) vols.push_back(v["name"]);
    for (const auto& mf : template_map_files(root)) {
      const std::size_t mark = sub.mark();
      try {
        or_skip_unreadable_v73(sub, emit, [&] {
          const Json info = export_map(sub, mf, root, std::nullopt, false, std::nullopt, &recording_map_hooks());
          maps.push_back(info["name"]);
          emit(with_stage("map", without_id(info)));
        });
      } catch (const Error& e) {
        if (!is_one_of(e, {code::invalid_argument, code::not_found})) throw;
        sub.rollback_to(mark);
        skipped.push_back(Json{{"what", mf.filename().string()}, {"reason", e.what()}});
        emit(Json{{"stage", "skipped"}, {"what", mf.filename().string()}, {"reason", e.what()}});
      }
    }
  });
  return Json{{"id", sub.id()}, {"surfaces", names}, {"volumes", vols}, {"maps", maps}, {"skipped", skipped}};
}

Json convert_group(nxr::io::DatasetWriter& ds, const fs::path& root, const Emit& emit, const std::string& created_by) {
  const fs::path src = root / "data" / "Group_analysis" / "@intra";
  if (!is_dir(src)) throw Error(code::not_found, py_path(src.string()) + ": not in this protocol");
  // format 56: the group results go into the dataset's template (the template holds the cohort); a one-database dataset's template row, or the template's own package
  const auto tpl = ds.template_();
  if (!tpl) throw Error(code::not_found, "import group: the dataset has no template subject (import the template first): a cohort's results go into it");
  nxr::io::OpenSubjectOptions oo;
  oo.created_by = created_by;
  SubjectWriter sub = SubjectWriter::open(ds, nxr::io::text(*tpl, "name"), oo);
  Json wrote = Json::array();
  sub.compose([&] {
    for (const auto& f : files_of(src, "timefreq_", ".mat"))
      or_skip_unreadable_v73(sub, emit, [&] {
        if (const auto info = export_timefreq(sub, f, std::nullopt, true)) {
          wrote.push_back((*info)["name"]);
          emit(with_stage("field", without_id(*info)));
        }
      });
  });
  return Json{{"id", sub.id()}, {"fields", wrote}};
}

}  // namespace nxr::bst

// The importer of a GIFTI file: see nxr/convert/gifti_import.h.
#include "nxr/convert/gifti_import.h"

#include <cmath>
#include <filesystem>
#include <map>

#include "nxr/convert/import_origin.h"
#include "nxr/convert/world_frames.h"
#include "nxr/formats/gifti.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::convert {

namespace gii = nxr::formats::gifti;
namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string base_name(const std::string& path) {
  std::string n = fs::path(path).filename().string();
  for (const char* suf : {".surf.gii", ".func.gii", ".shape.gii", ".label.gii", ".gii"}) {
    const std::string s = suf;
    if (n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0) { n.resize(n.size() - s.size()); break; }
  }
  return nxr::io::sanitize_node_name(n.empty() ? "gifti" : n);
}
bool is_identity(const std::array<double, 16>& m) {
  for (int i = 0; i < 16; ++i) if (std::abs(m[i] - ((i % 5) == 0 ? 1.0 : 0.0)) > 1e-12) return false;
  return true;
}
const std::string* array_meta(const gii::DataArray& a, const char* k) { return a.meta(k); }
std::optional<double> number(const std::string* s) { if (!s) return std::nullopt; try { return std::stod(*s); } catch (const std::exception&) { return std::nullopt; } }
}  // namespace

Json import_gifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const GiftiImport& opt) {
  if (!fs::exists(path)) throw nxr::io::Error(code::not_found, "no such file: " + path);
  const gii::Image img = gii::read_file(path);
  const std::string name = opt.name.empty() ? base_name(path) : nxr::io::sanitize_node_name(opt.name);
  const gii::DataArray *pts = nullptr, *tri = nullptr;
  std::vector<const gii::DataArray*> values, labels;
  for (const auto& a : img.arrays) {
    if (a.intent == "NIFTI_INTENT_POINTSET") pts = &a;
    else if (a.intent == "NIFTI_INTENT_TRIANGLE") tri = &a;
    else if (a.intent == "NIFTI_INTENT_LABEL") labels.push_back(&a);
    else values.push_back(&a);
  }
  std::string kind = opt.kind;
  if (kind.empty()) kind = pts ? "surface" : !labels.empty() ? "labels" : "overlay";
  if (kind != "surface" && kind != "overlay" && kind != "labels") refuse("--kind " + kind + ": choose from surface, overlay, labels");
  Json out = Json{{"kind", kind}, {"name", name}};
  const ImportOrigin org{"gifti", opt.source, opt.source_file, opt.licence, opt.attribution, opt.redistributable};
  auto stamp = [&] {
    const Json st = stamp_origin(w, org, path);
    out["origin"] = st["origin"];
    out["origin_items"] = st["origin_items"];
    return out;
  };

  if (kind == "surface") {
    if (!pts || !tri) refuse(path + ": a surface needs a POINTSET and a TRIANGLE array");
    if (pts->dims.size() != 2 || pts->dims[1] != 3 || tri->dims.size() != 2 || tri->dims[1] != 3) refuse(path + ": the point set or the triangles are not [n x 3]");
    if (sub.by_name("manifold", name)) refuse(name + " is already in the package; give another --name");
    const std::int64_t n = pts->dims[0];
    const gii::Transform xf = pts->transforms.empty() ? gii::Transform{} : pts->transforms[0];
    std::optional<World> data_world = world_of_gifti_space(xf.dataspace);
    const std::optional<World> to_world = world_of_gifti_space(xf.xformspace);
    const auto cr = number(array_meta(*pts, "VolGeomC_R")), ca = number(array_meta(*pts, "VolGeomC_A")), cs_ = number(array_meta(*pts, "VolGeomC_S"));
    const bool tkr = !data_world && cr && ca && cs_;
    std::string key, system_unit;
    if (data_world) { key = data_world->key; system_unit = data_world->unit; }
    else if (tkr) { key = "freesurfer-tkr"; system_unit = "m"; }
    else { key = "gifti:" + name; system_unit = "m"; }
    const double to_unit = system_unit == "m" ? 1e-3 : 1.0;   // the file's millimetres into the system's unit
    std::vector<double> pos(static_cast<std::size_t>(3 * n));
    for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = pts->at(i) * to_unit;
    std::vector<std::int32_t> faces(tri->i32.begin(), tri->i32.end());
    if (faces.empty()) for (std::size_t i = 0; i < tri->size(); ++i) faces.push_back(static_cast<std::int32_t>(tri->at(i)));
    Json prod = Json{{"format", "GIFTI " + img.version}, {"data_space", xf.dataspace}, {"transformed_space", xf.xformspace}, {"matrix_row_major", xf.matrix},
                     {"matrix_applied", false}, {"file_unit", "mm (the format's convention)"}, {"system", key}};
    if (const auto* s = array_meta(*pts, "AnatomicalStructurePrimary")) prod["structure"] = *s;
    if (!opt.anonymize) for (const auto& [k, v] : img.metadata) prod["metadata"][k] = v;
    nxr::io::SurfaceOptions so;
    so.name = name;
    so.primary = false;
    so.source = opt.source.empty() ? fs::path(path).filename().string() : opt.source;
    so.cs_key = key;
    so.producer = prod;
    if (const auto* s = array_meta(*pts, "AnatomicalStructurePrimary")) {
      if (*s == "CortexLeft") so.hemispheres = {{"left", 0, n}};
      else if (*s == "CortexRight") so.hemispheres = {{"right", 0, n}};
    }
    sub.compose([&] {
      if (data_world) ensure_world(sub, *data_world);
      else if (!tkr) sub.frame({"head", key, std::string("m"), "the file's own coordinate system: its data space is not stated (NIFTI_XFORM_UNKNOWN); the format's millimetres converted to metres. Not comparable to any other system", ""});
      const nxr::io::SurfaceInfo info = nxr::io::add_surface(sub, so, pos, faces);
      out["manifold"] = info.name;
      out["n_vertices"] = info.n_vertices;
      out["n_faces"] = info.n_faces;
      out["system"] = key;
      auto transition = [&](const std::string& from_key, const std::string& to_key, const std::array<double, 12>& m, const std::string& unit, const std::string& why) {
        nxr::io::OperatorSpec op;
        op.name = name + "_" + from_key + "_" + to_key;
        op.path = op.name;
        op.kind = "transition";
        op.from_manifold_id = info.id;
        op.cols.set("from_coordinate_system_id", sub.cs(from_key));
        op.cols.set("to_coordinate_system_id", sub.cs(to_key));
        op.n_rows = 3;
        op.n_cols = 4;
        std::array<double, 12> mm = m;
        op.data = view(mm.data(), {3, 4});
        op.cols.set("description", from_key + " \xe2\x86\x92 " + to_key + " (3x4 affine, output in " + unit + "): " + why);
        op.cols.json("producer_json", Json{{"output_unit", unit}, {"convention", "to = M @ [x, y, z, 1]"}});
        const std::string nm = op.name;
        sub.operator_(std::move(op));
        return nm;
      };
      if (tkr) {
        ensure_world(sub, scanner_ras());
        out["transition"] = transition("freesurfer-tkr", "scanner-ras", {1, 0, 0, *cr / 1000.0, 0, 1, 0, *ca / 1000.0, 0, 0, 1, *cs_ / 1000.0}, "m", "tkr + c_ras, from the file's VolGeomC_R/A/S");
      } else if (to_world && data_world && to_world->key != data_world->key && !is_identity(xf.matrix)) {
        // the matrix maps the data space (mm) onto the transformed space (mm): into the units of the two systems
        const double si = data_world->mm_per_unit(), so_ = 1.0 / to_world->mm_per_unit();
        std::array<double, 12> m{};
        for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) m[4 * r + c] = xf.matrix[4 * r + c] * si * so_; m[4 * r + 3] = xf.matrix[4 * r + 3] * so_; }
        ensure_world(sub, *to_world);
        out["transition"] = transition(data_world->key, to_world->key, m, to_world->unit, "the file's CoordinateSystemTransformMatrix");
      } else {
        out["transition"] = nullptr;
      }
    });
    if (!data_world && !tkr) out["note"] = "the file states no data space: the surface stands in a system of its own, not comparable to any other";
    return stamp();
  }

  // maps and labels belong on a surface that is already in the package
  if (opt.surface.empty()) refuse("--surface NAME: the surface of the package these values are on");
  const auto srf = sub.by_name("manifold", opt.surface);
  if (!srf) throw nxr::io::Error(code::not_found, "no surface '" + opt.surface + "' in the package");
  const std::string mid = nxr::io::text(*srf, "id");
  const std::int64_t nv = nxr::io::integer(*srf, "n_vertices");
  const auto& arrays = kind == "overlay" ? values : labels;
  if (arrays.empty()) refuse(path + ": no " + std::string(kind == "overlay" ? "map" : "label") + " array");
  if (kind == "labels" && opt.parcellation.empty()) refuse("a label file is imported only when its parcellation is named: give --parcellation NAME");
  for (const auto* a : arrays) {
    const bool flat = a->dims.size() == 1 || (a->dims.size() == 2 && a->dims[1] == 1);
    if (!flat) refuse(path + ": an array of shape " + std::to_string(a->dims.size()) + " dimensions is not a value per vertex");
    if (a->dims[0] != nv) refuse(path + ": " + std::to_string(a->dims[0]) + " values for the " + std::to_string(nv) + " vertices of " + opt.surface);
  }
  sub.compose([&] {
    Json made = Json::array();
    for (std::size_t k = 0; k < arrays.size(); ++k) {
      const gii::DataArray& a = *arrays[k];
      const std::string nm = arrays.size() == 1 ? name : name + "_" + std::to_string(k);
      if (sub.by_name(kind == "overlay" ? "field" : "selection", nm)) refuse(nm + " is already in the package; give another --name");
      if (kind == "overlay") {
        std::vector<float> v(static_cast<std::size_t>(nv));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(a.at(i));
        nxr::io::FieldSpec f;
        f.name = nm;
        f.path = "source/" + nm;
        f.kind = opt.field_kind;
        f.manifold_id = mid;
        f.data = view(v, {nv});
        f.array.compress = true;
        f.cols.set("description", "a value per vertex of " + opt.surface + " (GIFTI " + fs::path(path).filename().string() + ")").set("source", opt.source.empty() ? fs::path(path).filename().string() : opt.source);
        if (!opt.unit.empty()) f.cols.set("unit", opt.unit); else f.cols.null("unit");
        sub.field(std::move(f));
      } else {
        nxr::io::DictionarySpec d;
        d.name = opt.parcellation;
        d.scope = "app";
        std::map<std::int32_t, std::size_t> at;
        std::vector<nxr::io::DictionaryEntry> entries;
        std::vector<std::int32_t> raw(static_cast<std::size_t>(nv));
        for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<std::int32_t>(a.at(i));
        for (auto key : raw) {
          if (at.count(key)) continue;
          nxr::io::DictionaryEntry e;
          e.code = key;
          e.ordinal = key;
          e.name = "label " + std::to_string(key);
          for (const auto& l : img.labels) if (l.key == key) { e.name = l.name; if (l.has_colour) e.color = std::array<int, 3>{static_cast<int>(std::lround(l.rgba[0] * 255)), static_cast<int>(std::lround(l.rgba[1] * 255)), static_cast<int>(std::lround(l.rgba[2] * 255))}; }
          e.cols.json("attributes_json", Json{{"source_key", key}});
          at[key] = entries.size();
          entries.push_back(e);
        }
        d.entries = entries;
        const auto dr = sub.dictionary(d);
        std::vector<std::int32_t> codes(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) codes[i] = static_cast<std::int32_t>(dr.codes.at(at.at(raw[i])));
        nxr::io::SelectionSpec s;
        s.name = nm;
        s.path = "source/" + nm;
        s.type = "set";
        s.manifold_id = mid;
        s.no_elements = true;
        s.dense = true;
        s.data = view(codes, {nv});
        s.cols.set("dictionary_id", dr.id).set("n_elements", nv).set("n_members", static_cast<std::int64_t>(entries.size()))
            .set("description", "one label of the parcellation '" + opt.parcellation + "' per vertex of " + opt.surface + ": the CODE is the dictionary entry; its source_key is the file's key");
        sub.selection(std::move(s));
      }
      made.push_back(nm);
    }
    out[kind == "overlay" ? "fields" : "selections"] = made;
    out["surface"] = opt.surface;
  });
  return stamp();
}

}  // namespace nxr::convert

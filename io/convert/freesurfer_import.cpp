// The importer of a FreeSurfer subject: see nxr/convert/freesurfer_import.h.
#include "nxr/convert/freesurfer_import.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "nxr/convert/import_origin.h"
#include "nxr/convert/world_frames.h"
#include "nxr/formats/deflate.h"
#include "nxr/io/default_subject.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::convert {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::vector<unsigned char> slurp(const std::string& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) throw nxr::io::Error(code::not_found, "no such file: " + file);
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
std::uint32_t be32(const std::vector<unsigned char>& b, std::size_t at) {
  if (at + 4 > b.size()) refuse("truncated FreeSurfer file");
  return (std::uint32_t(b[at]) << 24) | (std::uint32_t(b[at + 1]) << 16) | (std::uint32_t(b[at + 2]) << 8) | b[at + 3];
}
float bef(const std::vector<unsigned char>& b, std::size_t at) { const std::uint32_t u = be32(b, at); float x; std::memcpy(&x, &u, 4); return x; }

// "cras = x y z" in the volume-geometry text a surface file ends with
std::optional<std::array<double, 3>> cras_from_text(const std::string& tail) {
  const auto p = tail.find("cras");
  if (p == std::string::npos) return std::nullopt;
  const auto eq = tail.find('=', p);
  if (eq == std::string::npos) return std::nullopt;
  std::istringstream is(tail.substr(eq + 1, tail.find('\n', eq) - eq - 1));
  std::array<double, 3> c{};
  if (!(is >> c[0] >> c[1] >> c[2])) return std::nullopt;
  return c;
}

struct Both { std::vector<double> pos; std::vector<std::int32_t> faces; std::int64_t n_lh = 0; };
Both read_both(const fs::path& dir, const std::string& surface) {
  Both b;
  const nxr::io::Mesh lh = nxr::io::read_freesurfer_surface(dir / ("lh." + surface));
  const nxr::io::Mesh rh = nxr::io::read_freesurfer_surface(dir / ("rh." + surface));
  b.n_lh = lh.n_vertices();
  b.pos = lh.vertices;
  b.pos.insert(b.pos.end(), rh.vertices.begin(), rh.vertices.end());
  b.faces = lh.faces;
  for (auto f : rh.faces) b.faces.push_back(f + static_cast<std::int32_t>(b.n_lh));
  return b;
}

// the "created by" line a surface file opens with (after its magic)
std::string created_by(const std::string& file) {
  const auto b = slurp(file);
  std::size_t i = 3;
  std::string s;
  while (i < b.size() && b[i] != '\n') s += static_cast<char>(b[i++]);
  return s;
}
}  // namespace

std::optional<std::array<double, 3>> freesurfer_c_ras(const std::string& file) {
  std::vector<unsigned char> b = slurp(file);
  const std::string name = fs::path(file).filename().string();
  const bool volume = name.size() >= 4 && (name.compare(name.size() - 4, 4, ".mgh") == 0 || name.size() >= 4 && name.compare(name.size() - 4, 4, ".mgz") == 0);
  if (volume) {
    if (b.size() > 2 && b[0] == 0x1f && b[1] == 0x8b) b = nxr::formats::inflate_bytes(b.data(), b.size(), nxr::formats::DeflateWrapper::gzip);
    if (b.size() < 90 || be32(b, 0) != 1) refuse(name + ": not an MGH volume");
    if (b[28] == 0 && b[29] == 0) return std::nullopt;   // goodRASFlag 0: the file holds no geometry
    return std::array<double, 3>{bef(b, 78), bef(b, 82), bef(b, 86)};
  }
  // a surface: the text after the faces (nibabel's volume_info)
  if (b.size() < 20 || b[0] != 0xff || b[1] != 0xff || b[2] != 0xfe) refuse(name + ": not a FreeSurfer triangle surface or an MGH volume");
  std::size_t i = 3;
  for (int line = 0; line < 2; ++line) { while (i < b.size() && b[i] != '\n') ++i; ++i; }
  const std::size_t nv = be32(b, i), nf = be32(b, i + 4);
  const std::size_t end = i + 8 + 12 * nv + 12 * nf;
  if (end >= b.size()) return std::nullopt;
  return cras_from_text(std::string(b.begin() + static_cast<std::ptrdiff_t>(end), b.end()));
}

std::vector<float> read_freesurfer_curv(const std::string& file) {
  const auto b = slurp(file);
  const std::string name = fs::path(file).filename().string();
  if (b.size() < 15) refuse(name + ": not a FreeSurfer curvature file (too short)");
  std::vector<float> out;
  if (b[0] == 0xff && b[1] == 0xff && b[2] == 0xff) {
    const std::size_t nv = be32(b, 3), per = be32(b, 11);
    if (per != 1) refuse(name + ": " + std::to_string(per) + " values per vertex (one is read)");
    if (15 + 4 * nv > b.size()) refuse(name + ": truncated");
    out.resize(nv);
    for (std::size_t k = 0; k < nv; ++k) out[k] = bef(b, 15 + 4 * k);
    return out;
  }
  const std::size_t nv = (std::size_t(b[0]) << 16) | (std::size_t(b[1]) << 8) | b[2];   // the old format: int16 values / 100
  if (6 + 2 * nv > b.size()) refuse(name + ": truncated");
  out.resize(nv);
  for (std::size_t k = 0; k < nv; ++k) out[k] = static_cast<float>(static_cast<std::int16_t>((b[6 + 2 * k] << 8) | b[7 + 2 * k])) / 100.0f;
  return out;
}

Json import_freesurfer_subject(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& folder, const FreeSurferImport& opt) {
  const fs::path root = folder, surf = root / "surf", label = root / "label";
  if (!fs::is_directory(surf)) throw nxr::io::Error(code::not_found, folder + ": not a FreeSurfer subject folder (no surf/)");
  if (opt.surfaces.empty()) refuse("name at least one surface (--surfaces white,pial)");
  const std::string prefix = nxr::io::sanitize_node_name(opt.name);
  for (const auto& s : opt.surfaces)
    if (sub.by_name("manifold", prefix + "_" + s)) refuse(prefix + "_" + s + " is already in the package; give another --name");

  // c_ras: the volume, else the surface's own tail; the two must agree
  std::optional<std::array<double, 3>> cras;
  std::string cras_from;
  const fs::path orig = !opt.orig.empty() ? fs::path(opt.orig) : root / "mri" / "orig.mgz";
  if (!opt.orig.empty() || fs::exists(orig)) { cras = freesurfer_c_ras(orig.string()); cras_from = orig.filename().string(); }
  const auto tail = freesurfer_c_ras((surf / ("lh." + opt.surfaces[0])).string());
  if (!cras && tail) { cras = tail; cras_from = "lh." + opt.surfaces[0] + " (its volume geometry)"; }
  if (cras && tail) for (int d = 0; d < 3; ++d) if (std::abs((*cras)[d] - (*tail)[d]) > 1e-3) refuse("c_ras of " + cras_from + " and of the surface's volume geometry disagree");

  Json out = Json{{"name", prefix}, {"surfaces", Json::array()}, {"annotations", Json::array()}, {"overlays", Json::array()}};
  const ImportOrigin org{"freesurfer", opt.source, opt.source_file, opt.licence, opt.attribution, opt.redistributable};
  sub.compose([&] {
    std::string first_id;
    std::int64_t n_first = 0, nl = 0;
    for (std::size_t k = 0; k < opt.surfaces.size(); ++k) {
      const std::string& s = opt.surfaces[k];
      Both b = read_both(surf, s);
      const std::vector<std::int64_t> moved = nxr::io::separate_coincident(b.pos, b.faces);
      for (double& x : b.pos) x /= 1000.0;   // mm -> m
      const std::int64_t n = static_cast<std::int64_t>(b.pos.size() / 3);
      nxr::io::SurfaceOptions o;
      o.name = prefix + "_" + s;
      o.primary = k == 0;
      o.source = "FreeSurfer surf/{lh,rh}." + s;
      o.cs_key = "freesurfer-tkr";
      Json prod = Json{{"format", "freesurfer"}, {"surface", s}, {"units", "m (the file's millimetres / 1000)"}, {"frame", "FreeSurfer surface RAS (tkr)"}, {"coincident_vertices_moved", moved}};
      if (!opt.anonymize) prod["created_by"] = created_by((surf / ("lh." + s)).string());
      o.producer = prod;
      o.hemispheres = {{"left", 0, b.n_lh}, {"right", b.n_lh, n - b.n_lh}};
      if (k == 0 && fs::exists(surf / "lh.sphere.reg") && fs::exists(surf / "rh.sphere.reg")) {
        const Both sp = read_both(surf, "sphere.reg");
        if (static_cast<std::int64_t>(sp.pos.size() / 3) != n) refuse("sphere.reg and " + s + " differ in vertex count");
        o.sphere = sp.pos;
      }
      const nxr::io::SurfaceInfo info = nxr::io::add_surface(sub, o, b.pos, b.faces);
      out["surfaces"].push_back(Json{{"name", info.name}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"coincident_vertices_moved", moved.size()}});
      if (k == 0) { first_id = info.id; n_first = n; nl = b.n_lh; }
    }
    // tkr -> scanner RAS: a translation by c_ras (metres)
    if (cras) {
      ensure_world(sub, scanner_ras());
      std::array<double, 12> m = {1, 0, 0, (*cras)[0] / 1000.0, 0, 1, 0, (*cras)[1] / 1000.0, 0, 0, 1, (*cras)[2] / 1000.0};
      nxr::io::OperatorSpec op;
      op.name = prefix + "_freesurfer-tkr_scanner-ras";
      op.path = op.name;
      op.kind = "transition";
      op.from_manifold_id = first_id;
      op.cols.set("from_coordinate_system_id", sub.cs("freesurfer-tkr"));
      op.cols.set("to_coordinate_system_id", sub.cs("scanner-ras"));
      op.n_rows = 3;
      op.n_cols = 4;
      op.data = view(m.data(), {3, 4});
      op.cols.set("description", "FreeSurfer surface RAS (tkr) \xe2\x86\x92 scanner RAS, metres: tkr + c_ras (from " + cras_from + ")");
      op.cols.json("producer_json", Json{{"convention", "scanner = tkr + c_ras"}, {"c_ras_mm", Json::array({(*cras)[0], (*cras)[1], (*cras)[2]})}, {"c_ras_from", cras_from}, {"output_unit", "m"}});
      out["transition"] = op.name;
      sub.operator_(std::move(op));
    } else {
      out["transition"] = nullptr;
      out["note"] = "no c_ras (no orig.mgz, and the surface file holds no volume geometry): the surfaces are not placed in scanner RAS";
    }
    // annotations: one codes Selection per named annotation, over both hemispheres
    for (const auto& a : opt.annotations) {
      std::vector<std::int32_t> raw(static_cast<std::size_t>(n_first), -1);
      nxr::io::DictionarySpec d;
      d.name = a;
      d.scope = "app";
      std::vector<nxr::io::DictionaryEntry> entries;
      nxr::io::DictionaryEntry none;
      none.code = 0; none.name = "unlabelled"; none.ordinal = 0;
      entries.push_back(none);
      std::vector<std::int32_t> slot(static_cast<std::size_t>(n_first), 0);   // the entry index of each vertex
      for (const auto& [h, side, off] : {std::tuple<const char*, const char*, std::int64_t>{"lh", "L", 0}, {"rh", "R", nl}}) {
        const nxr::io::FsAnnot an = nxr::io::read_freesurfer_annot(label / (std::string(h) + "." + a + ".annot"));
        const std::int64_t nh = std::string(h) == "lh" ? nl : n_first - nl;
        if (static_cast<std::int64_t>(an.labels.size()) != nh) refuse(std::string(h) + "." + a + ".annot: " + std::to_string(an.labels.size()) + " labels for " + std::to_string(nh) + " vertices");
        std::map<std::int32_t, std::size_t> at;
        for (std::size_t v = 0; v < an.labels.size(); ++v) {
          const std::int32_t row = an.labels[v];
          if (row < 0) continue;
          auto it = at.find(row);
          if (it == at.end()) {
            nxr::io::DictionaryEntry e;
            e.code = static_cast<std::int64_t>(entries.size());
            e.name = (static_cast<std::size_t>(row) < an.names.size() ? an.names[static_cast<std::size_t>(row)] : "label " + std::to_string(row)) + " " + side;
            e.ordinal = e.code;
            if (static_cast<std::size_t>(row) < an.ctab.size()) e.color = std::array<int, 3>{an.ctab[static_cast<std::size_t>(row)][0], an.ctab[static_cast<std::size_t>(row)][1], an.ctab[static_cast<std::size_t>(row)][2]};
            e.cols.json("attributes_json", Json{{"hemisphere", side}, {"source_row", row}});
            it = at.emplace(row, entries.size()).first;
            entries.push_back(e);
          }
          slot[static_cast<std::size_t>(off) + v] = static_cast<std::int32_t>(it->second);
        }
      }
      d.entries = entries;
      const auto dr = sub.dictionary(d);
      std::vector<std::int32_t> codes(static_cast<std::size_t>(n_first));
      for (std::size_t v = 0; v < codes.size(); ++v) codes[v] = static_cast<std::int32_t>(dr.codes.at(static_cast<std::size_t>(slot[v])));
      std::int64_t used = 0;
      { std::vector<char> seen(entries.size(), 0); for (auto s : slot) seen[static_cast<std::size_t>(s)] = 1; for (char c : seen) used += c; }
      nxr::io::SelectionSpec sel;
      sel.name = prefix + "_" + a;
      sel.path = "source/" + sel.name;
      sel.type = "set";
      sel.manifold_id = first_id;
      sel.no_elements = true;
      sel.dense = true;
      sel.data = view(codes, {n_first});
      sel.cols.set("dictionary_id", dr.id).set("n_elements", n_first).set("n_members", used)
          .set("description", "one label of the annotation '" + a + "' per vertex (FreeSurfer label/{lh,rh}." + a + ".annot): the CODE is the dictionary entry, named from the file's colour table with its hemisphere");
      sub.selection(std::move(sel));
      out["annotations"].push_back(Json{{"name", a}, {"n_labels", used}});
    }
    // overlays: one Field per named file pair
    for (const auto& ov : opt.overlays) {
      std::vector<float> v = read_freesurfer_curv((surf / ("lh." + ov)).string());
      const std::vector<float> r = read_freesurfer_curv((surf / ("rh." + ov)).string());
      v.insert(v.end(), r.begin(), r.end());
      if (static_cast<std::int64_t>(v.size()) != n_first) refuse("lh/rh." + ov + ": " + std::to_string(v.size()) + " values for " + std::to_string(n_first) + " vertices");
      nxr::io::FieldSpec f;
      f.name = prefix + "_" + ov;
      f.path = "source/" + f.name;
      f.kind = opt.field_kind;
      f.manifold_id = first_id;
      f.data = view(v, {n_first});
      f.array.compress = true;
      f.cols.set("description", "a value per vertex (FreeSurfer surf/{lh,rh}." + ov + ")").set("source", "FreeSurfer surf/{lh,rh}." + ov).null("unit");
      sub.field(std::move(f));
      out["overlays"].push_back(ov);
    }
  });
  const Json st = stamp_origin(w, org, folder);
  out["origin"] = st["origin"];
  out["origin_items"] = st["origin_items"];
  return out;
}

}  // namespace nxr::convert

// The importer of a NIfTI volume: see nxr/convert/nifti_import.h.
#include "nxr/convert/nifti_import.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "nxr/convert/import_origin.h"
#include "nxr/convert/world_frames.h"
#include "nxr/formats/nifti.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::convert {

namespace nifti = nxr::formats::nifti;
namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string base_name(const std::string& path) {
  std::string n = std::filesystem::path(path).filename().string();
  for (const char* suf : {".nii.gz", ".nii", ".hdr", ".img"}) {
    const std::string s = suf;
    if (n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0) { n.resize(n.size() - s.size()); break; }
  }
  return nxr::io::sanitize_node_name(n.empty() ? "volume" : n);
}

std::map<std::int64_t, std::string> read_label_names(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw nxr::io::Error(code::not_found, "--label-names: no such file: " + file);
  std::map<std::int64_t, std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string c;
    if (!(ls >> c) || c[0] == '#') continue;
    std::int64_t v;
    try { std::size_t used = 0; v = std::stoll(c, &used); if (used != c.size()) throw std::invalid_argument(c); }
    catch (const std::exception&) { refuse("--label-names " + file + ": '" + c + "' is not a whole number (lines are `code name`)"); }
    std::string rest;
    std::getline(ls, rest);
    const auto b = rest.find_first_not_of(" \t");
    out[v] = b == std::string::npos ? std::to_string(v) : rest.substr(b, rest.find_last_not_of(" \t\r") - b + 1);
  }
  return out;
}
}  // namespace

Json import_nifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const NiftiImport& opt) {
  if (!std::filesystem::exists(path)) throw nxr::io::Error(code::not_found, "no such file: " + path);
  const nifti::Volume vol = nifti::read_file(path);
  const nifti::Header& h = vol.header;
  std::vector<std::int64_t> shape = h.shape;
  while (shape.size() > 3 && shape.back() == 1) shape.pop_back();
  if (shape.size() != 3) refuse(path + ": a volume of " + std::to_string(shape.size()) + " dimensions is not imported (a series of volumes has no importer yet)");
  const std::int64_t ni = shape[0], nj = shape[1], nk = shape[2];
  const std::string name = opt.name.empty() ? base_name(path) : nxr::io::sanitize_node_name(opt.name);
  const bool labels = opt.kind.empty() ? !opt.parcellation.empty() : opt.kind == "labels";
  if (!opt.kind.empty() && opt.kind != "volume" && opt.kind != "labels") refuse("--kind " + opt.kind + ": choose from volume, labels");
  if (labels && opt.parcellation.empty()) refuse("a label volume is imported only when its parcellation is named: give --parcellation NAME");
  if (!labels && !opt.parcellation.empty()) refuse("--parcellation names a label volume: use --kind labels, or leave it out for an intensity volume");

  const double mm_per_unit = 1000.0 * h.unit_to_metres();
  const std::string node = "mri/" + name;
  for (const char* table : {"manifold", "field", "operator", "selection"})
    if (sub.find(table, node) || sub.find(table, node + "_volume")) refuse(name + ": " + node + " is already in the package; give another --name");

  const auto aff = h.affine();   // voxel index -> world, in the file's spatial unit
  const int world_code = h.affine_source() == nifti::Header::AffineSource::sform ? h.sform_code : h.affine_source() == nifti::Header::AffineSource::qform ? h.qform_code : 0;
  const auto world = world_of_nifti_code(world_code);
  const auto form = h.affine_source() == nifti::Header::AffineSource::sform ? "sform" : h.affine_source() == nifti::Header::AffineSource::qform ? "qform" : "pixdim";

  // the intensities, in C order (the lattice's), from the file's x-fastest order
  const bool f64 = vol.precision == nxr::formats::Precision::float64;
  const std::size_t n = static_cast<std::size_t>(ni * nj * nk);
  std::vector<float> a32;
  std::vector<double> a64;
  if (f64) a64.resize(n); else a32.resize(n);
  for (std::int64_t k = 0; k < nk; ++k)
    for (std::int64_t j = 0; j < nj; ++j)
      for (std::int64_t i = 0; i < ni; ++i) {
        const std::size_t src = static_cast<std::size_t>(i + ni * (j + nj * k)), dst = static_cast<std::size_t>((i * nj + j) * nk + k);
        if (f64) a64[dst] = vol.f64[src]; else a32[dst] = vol.f32[src];
      }

  Json out = Json{{"kind", labels ? "labels" : "volume"}, {"name", name}};
  Json prod = Json{{"format", h.version == 2 ? "NIfTI-2" : "NIfTI-1"}, {"shape", shape}, {"datatype", h.datatype}, {"transform_source", form}, {"qform_code", h.qform_code}, {"sform_code", h.sform_code},
                   {"convention", "world_mm = M @ [i, j, k, 1]"}, {"spatial_unit", h.spatial_unit().empty() ? "mm (the format's convention; the file states none)" : h.spatial_unit()}};
  if (!opt.anonymize && !h.descrip.empty()) prod["descrip"] = h.descrip;
  if (!opt.anonymize && !h.aux_file.empty()) prod["aux_file"] = h.aux_file;
  const ImportOrigin org{"nifti", opt.source, opt.source_file, opt.licence, opt.attribution, opt.redistributable};

  sub.compose([&] {
    const std::string vname = name + "_volume";
    nxr::io::ManifoldSpec ms;
    ms.name = vname;
    ms.path = node + "_volume";
    ms.type = "volume";
    ms.n_vertices = static_cast<std::int64_t>(n);
    ms.topology = {"lattice", 3, 1, std::nullopt, {}};
    const auto lc = nxr::io::lattice_counts({ni, nj, nk});
    if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
    if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
    if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
    nxr::io::GeometrySpec g;
    g.form = "closed_form";
    g.cs_key = "voxel:" + vname;
    const std::int64_t dims[3] = {ni, nj, nk};
    for (int d = 0; d < 3; ++d) {
      nxr::io::DimensionSpec ds;
      ds.name = std::string(1, "ijk"[d]);
      ds.n_vertices = dims[d];
      // the voxel size: the length of the axis' column of the affine (what the pixdim is, for a rigid map)
      ds.spacing = mm_per_unit * std::sqrt(aff[d] * aff[d] + aff[4 + d] * aff[4 + d] + aff[8 + d] * aff[8 + d]);
      ds.unit = "mm";
      g.dimensions.push_back(ds);
    }
    ms.geometry = g;
    ms.cols.set("source", opt.source.empty() ? std::filesystem::path(path).filename().string() : opt.source);
    ms.cols.json("producer_json", prod);
    const std::string vid = sub.manifold(std::move(ms));
    out["manifold"] = vname;

    if (world) {
      ensure_world(sub, *world);
      std::array<double, 12> mm{};
      for (int i = 0; i < 12; ++i) mm[i] = mm_per_unit * aff[i];
      nxr::io::OperatorSpec op;
      op.name = vname + "_" + world->key;
      op.path = op.name;
      op.kind = "transition";
      op.from_manifold_id = vid;
      op.cols.set("from_coordinate_system_id", sub.cs("voxel:" + vname));
      op.cols.set("to_coordinate_system_id", sub.cs(world->key));
      op.n_rows = 3;
      op.n_cols = 4;
      op.data = view(mm.data(), {3, 4});
      op.cols.set("description", std::string("voxel index \xe2\x86\x92 ") + world->key + " millimetres (3x4 affine, voxel size folded in), from the file's " + form + " (code " + std::to_string(world_code) + ")");
      op.cols.json("producer_json", Json{{"convention", "world_mm = M @ [i, j, k, 1]"}, {"output_unit", "mm"}, {"transform_source", form}, {"code", world_code}});
      out["transition"] = op.name;
      sub.operator_(std::move(op));
    } else {
      out["transition"] = nullptr;
      out["note"] = "the file states no world frame (transform codes 0): the voxel chart stands alone";
    }

    const nxr::io::Shape chunks{std::min<std::int64_t>(64, ni), std::min<std::int64_t>(64, nj), std::min<std::int64_t>(64, nk)};
    const nxr::io::Shape sh{ni, nj, nk};
    if (!labels) {
      nxr::io::FieldSpec f;
      f.name = name;
      f.path = node;
      f.kind = opt.field_kind;
      f.manifold_id = vid;
      f.data = f64 ? view(a64, sh) : view(a32, sh);
      f.array.chunks = chunks;
      f.array.compress = true;
      f.array.dense = true;
      f.cols.set("description", "a value per voxel of the NIfTI volume").set("source", opt.source.empty() ? std::filesystem::path(path).filename().string() : opt.source);
      if (!opt.unit.empty()) f.cols.set("unit", opt.unit); else f.cols.null("unit");
      sub.field(std::move(f));
      out["field"] = name;
      return;
    }
    // labels: whole numbers only; the entries named by the caller's file or by their code
    std::set<std::int64_t> distinct;
    std::vector<std::int64_t> vals(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double v = f64 ? a64[i] : static_cast<double>(a32[i]);
      if (v != std::round(v) || v < 0) refuse(path + ": not a label volume: a voxel holds " + std::to_string(v) + " (labels are non-negative whole numbers)");
      vals[i] = static_cast<std::int64_t>(v);
      distinct.insert(vals[i]);
    }
    std::map<std::int64_t, std::string> names;
    if (!opt.label_names.empty()) names = read_label_names(opt.label_names);
    nxr::io::DictionarySpec d;
    d.name = opt.parcellation;
    d.scope = "app";
    std::vector<nxr::io::DictionaryEntry> entries;
    std::map<std::int64_t, std::size_t> at;
    for (auto v : distinct) {
      nxr::io::DictionaryEntry e;
      e.code = v;
      const auto it = names.find(v);
      e.name = it != names.end() ? it->second : (v == 0 ? "unlabelled" : "label " + std::to_string(v));
      e.ordinal = v;
      e.cols.json("attributes_json", Json{{"source_id", v}});
      at[v] = entries.size();
      entries.push_back(e);
    }
    d.entries = entries;
    const auto dr = sub.dictionary(d);
    std::vector<std::int32_t> mapped(n);
    for (std::size_t i = 0; i < n; ++i) mapped[i] = static_cast<std::int32_t>(dr.codes.at(at.at(vals[i])));
    nxr::io::SelectionSpec s;
    s.name = name;
    s.path = node;
    s.type = "set";
    s.manifold_id = vid;
    s.no_elements = true;
    s.dense = true;
    s.data = view(mapped, sh);
    s.cols.set("dictionary_id", dr.id).set("n_elements", static_cast<std::int64_t>(n)).set("n_members", static_cast<std::int64_t>(distinct.size()))
        .set("description", "one label of the parcellation '" + opt.parcellation + "' per voxel: the CODE, i.e. the dictionary entry; its source_id is the file's value");
    sub.selection(std::move(s));
    out["selection"] = name;
    out["parcellation"] = opt.parcellation;
    out["n_labels"] = distinct.size();
  });
  const Json st = stamp_origin(w, org, path);
  out["origin"] = st["origin"];
  out["origin_items"] = st["origin_items"];
  return out;
}

}  // namespace nxr::convert

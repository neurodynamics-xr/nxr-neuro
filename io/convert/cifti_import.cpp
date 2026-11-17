// The importer of a CIFTI-2 dense map: see nxr/convert/cifti_import.h.
#include "nxr/convert/cifti_import.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <map>

#include "nxr/convert/import_origin.h"
#include "nxr/formats/cifti.h"
#include "nxr/io/layout.h"

namespace nxr::convert {

namespace ci = nxr::formats::cifti;
namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
}  // namespace

Json import_cifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const CiftiImport& opt) {
  if (!fs::exists(path)) throw nxr::io::Error(code::not_found, "no such file: " + path);
  const auto r = ci::open_file(path);
  const ci::Mapping& cols = r->map(1);
  const ci::Mapping& rows = r->map(0);
  if (cols.type != "CIFTI_INDEX_TYPE_BRAIN_MODELS") refuse(path + ": a parcellated file is not imported (only dense maps over brain models)");
  if (rows.type == "CIFTI_INDEX_TYPE_SERIES") refuse(path + ": a dtseries (a series over the grayordinates) is not imported yet");
  const bool is_label = rows.type == "CIFTI_INDEX_TYPE_LABELS";
  if (!is_label && rows.type != "CIFTI_INDEX_TYPE_SCALARS") refuse(path + ": maps of type " + rows.type + " are not imported");
  if (is_label && opt.parcellation.empty()) refuse("a dlabel is imported only when its parcellation is named: give --parcellation NAME");
  std::string name = fs::path(path).filename().string();
  for (const char* suf : {".dscalar.nii", ".dlabel.nii", ".nii"}) { const std::string s = suf; if (name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) { name.resize(name.size() - s.size()); break; } }
  if (!opt.name.empty()) name = opt.name;
  name = nxr::io::sanitize_node_name(name);
  const std::uint64_t n_maps = rows.named_maps.size(), n_cols = r->cols();
  std::vector<float> all(static_cast<std::size_t>(r->rows() * n_cols));
  r->read(0, r->rows(), 0, n_cols, all.data());

  Json out = Json{{"name", name}, {"kind", is_label ? "labels" : "maps"}, {"structures", Json::array()}, {"skipped", Json::array()}};
  const ImportOrigin org{"cifti", opt.source, opt.source_file, opt.licence, opt.attribution, opt.redistributable};
  sub.compose([&] {
    for (const auto& bm : cols.brain_models) {
      const bool left = bm.structure == "CIFTI_STRUCTURE_CORTEX_LEFT", right = bm.structure == "CIFTI_STRUCTURE_CORTEX_RIGHT";
      if (bm.model_type != "CIFTI_MODEL_TYPE_SURFACE" || !(left || right)) { out["skipped"].push_back(bm.structure); continue; }
      const std::string& sname = left ? opt.left : opt.right;
      if (sname.empty()) refuse(std::string("name the package's surface for ") + (left ? "the left" : "the right") + " cortex (--" + (left ? "left" : "right") + " NAME)");
      const auto srf = sub.by_name("manifold", sname);
      if (!srf) throw nxr::io::Error(code::not_found, "no surface '" + sname + "' in the package");
      const std::int64_t nv = nxr::io::integer(*srf, "n_vertices");
      if (bm.surface_vertex_count && static_cast<std::int64_t>(*bm.surface_vertex_count) != nv) refuse(bm.structure + ": the file's surface has " + std::to_string(*bm.surface_vertex_count) + " vertices, " + sname + " has " + std::to_string(nv));
      const std::string mid = nxr::io::text(*srf, "id");
      const std::string tag = left ? "left" : "right";
      std::vector<std::int32_t> idx(bm.vertex_indices.begin(), bm.vertex_indices.end());
      nxr::io::SelectionSpec sel;
      sel.name = name + "_" + tag + "_vertices";
      sel.path = "source/" + sel.name;
      sel.type = "indices";
      sel.manifold_id = mid;
      sel.data = view(idx, {static_cast<std::int64_t>(idx.size())});
      sel.picked = true;
      sel.cols.set("description", "the vertices the CIFTI file covers on " + sname + " (" + bm.structure + "); the others (the medial wall) are not in the file");
      sub.selection(std::move(sel));
      for (std::uint64_t k = 0; k < n_maps; ++k) {
        const std::string nm = n_maps == 1 ? name + "_" + tag : name + "_" + tag + "_" + std::to_string(k);
        const std::string label = rows.named_maps[k].name;
        std::vector<float> full(static_cast<std::size_t>(nv), std::numeric_limits<float>::quiet_NaN());
        for (std::size_t j = 0; j < idx.size(); ++j) full[static_cast<std::size_t>(idx[j])] = all[static_cast<std::size_t>(k * n_cols + bm.index_offset + j)];
        if (!is_label) {
          nxr::io::FieldSpec f;
          f.name = nm;
          f.path = "source/" + nm;
          f.kind = opt.field_kind;
          f.manifold_id = mid;
          f.data = view(full, {nv});
          f.array.compress = true;
          f.cols.set("description", "map '" + label + "' of " + fs::path(path).filename().string() + " on " + sname + ", NaN where the file has no vertex (outside " + name + "_" + tag + "_vertices)");
          if (!opt.unit.empty()) f.cols.set("unit", opt.unit); else f.cols.null("unit");
          sub.field(std::move(f));
        } else {
          nxr::io::DictionarySpec d;
          d.name = opt.parcellation;
          d.scope = "app";
          std::map<std::int32_t, std::size_t> at;
          std::vector<nxr::io::DictionaryEntry> entries;
          auto entry_of = [&](std::int32_t key) {
            auto it = at.find(key);
            if (it != at.end()) return it->second;
            nxr::io::DictionaryEntry e;
            e.code = key;
            e.ordinal = key;
            e.name = "label " + std::to_string(key);
            for (const auto& l : rows.named_maps[k].labels) if (l.key == key) { e.name = l.name; e.color = std::array<int, 3>{static_cast<int>(std::lround(l.red * 255)), static_cast<int>(std::lround(l.green * 255)), static_cast<int>(std::lround(l.blue * 255))}; }
            e.cols.json("attributes_json", Json{{"source_key", key}});
            at[key] = entries.size();
            entries.push_back(e);
            return entries.size() - 1;
          };
          entry_of(0);
          std::vector<std::size_t> slot(static_cast<std::size_t>(nv), 0);
          for (std::size_t j = 0; j < idx.size(); ++j) slot[static_cast<std::size_t>(idx[j])] = entry_of(static_cast<std::int32_t>(std::lround(all[static_cast<std::size_t>(k * n_cols + bm.index_offset + j)])));
          d.entries = entries;
          const auto dr = sub.dictionary(d);
          std::vector<std::int32_t> codes(static_cast<std::size_t>(nv));
          for (std::size_t v = 0; v < codes.size(); ++v) codes[v] = static_cast<std::int32_t>(dr.codes.at(slot[v]));
          nxr::io::SelectionSpec s;
          s.name = nm;
          s.path = "source/" + nm;
          s.type = "set";
          s.manifold_id = mid;
          s.no_elements = true;
          s.dense = true;
          s.data = view(codes, {nv});
          s.cols.set("dictionary_id", dr.id).set("n_elements", nv).set("n_members", static_cast<std::int64_t>(entries.size()))
              .set("description", "map '" + label + "' of the parcellation '" + opt.parcellation + "' on " + sname + ": the code of the dictionary entry per vertex (0: not in the file)");
          sub.selection(std::move(s));
        }
      }
      out["structures"].push_back(bm.structure);
    }
  });
  const Json st = stamp_origin(w, org, path);
  out["origin"] = st["origin"];
  out["origin_items"] = st["origin_items"];
  return out;
}

}  // namespace nxr::convert

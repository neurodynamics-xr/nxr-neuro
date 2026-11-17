// nxr-io — the spatial-scale (Weyl) axis of the measurement pyramid (spatial_bands.h).
#include "nxr/io/spatial_bands.h"

#include <cmath>
#include <sstream>

#include "nxr/io/layout.h"
#include "nxr/io/selection_store.h"

namespace nxr::io {

namespace {
std::string g(double v) { std::ostringstream s; s << v; return s.str(); }
}  // namespace

std::string spatial_bands_name(const std::string& surface) { return surface + " spatial bands"; }

std::optional<Row> find_eigenvalue(Database& db) {
  return db.one("SELECT m.* FROM manifold m JOIN subject s ON s.id = m.subject_id WHERE s.kind = 'template' AND m.name = ? AND m.type = 'line' ORDER BY m.created_utc LIMIT 1", {std::string(kEigenvalueName)});
}

std::string ensure_eigenvalue(SubjectWriter& sub) {
  const std::string kind = text(sub.row(), "kind");
  if (kind != "template") throw Error(code::invalid_argument, sub.name() + ": the Eigenvalue manifold lives in the dataset's default subject (a template), not in a " + kind);
  if (const auto have = sub.db().one("SELECT id FROM manifold WHERE subject_id = ? AND name = ?", {sub.id(), std::string(kEigenvalueName)})) return text(*have, "id");
  ManifoldSpec ms;
  ms.name = kEigenvalueName;
  ms.path = folder_for("line", "", std::string("eigen")) + "/" + kEigenvalueName;
  ms.type = "line";
  ms.n_vertices = kEigenvalueSamples;
  ms.topology = {"lattice", 1, 1, std::nullopt, {}};
  const auto lc = lattice_counts({kEigenvalueSamples});
  if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
  if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
  if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
  GeometrySpec geo;
  geo.form = "closed_form";
  geo.cs_key = "eigenvalue:Laplace-Beltrami";
  DimensionSpec d;
  d.name = "eigenvalue";
  d.n_vertices = kEigenvalueSamples;
  d.spacing = kEigenvalueSpacing;
  d.origin = 0.0;
  d.unit = "1/m\xc2\xb2";
  geo.dimensions.push_back(d);
  ms.geometry = geo;
  ms.cols.set("comment", "the line the spatial bands are read on: the Laplace\xe2\x80\x93" "Beltrami eigenvalue (1/m\xc2\xb2), 0 to 4.1e6 in steps of 1000");
  return sub.manifold(std::move(ms));
}

std::string spatial_bands_selection(SubjectWriter& sub, const std::string& surface, const std::vector<SpatialBandEdge>& levels) {
  if (levels.empty()) throw Error(code::invalid_argument, "spatial bands: a surface with no ladder levels has none");
  const std::string name = spatial_bands_name(surface);
  if (const auto have = selection::find_by_name(sub.db(), sub.id(), name, std::string("spans"))) {
    const auto rows = selection::spans_of(sub.db(), have->id);
    bool same = rows.size() == levels.size();
    for (std::size_t k = 0; same && k < rows.size(); ++k)
      same = std::abs(rows[k].start - levels[k].lo) <= 1e-9 * (1 + levels[k].lo) && std::abs(rows[k].stop - levels[k].hi) <= 1e-9 * (1 + levels[k].hi) && rows[k].code.value_or(-1) == levels[k].level;
    if (!same) throw Error(code::invalid_argument, sub.name() + ": the stored '" + name + "' are the bands of another ladder (a band's code is its level, so a surface has one bank; remove them to rebuild)");
    return have->id;
  }
  auto m = find_eigenvalue(sub.db());
  std::string mid;
  if (m) mid = text(*m, "id");
  else if (text(sub.row(), "kind") == "template") mid = ensure_eigenvalue(sub);
  else throw Error(code::no_such_row, sub.name() + ": the spatial bands are spans on the Eigenvalue manifold of the dataset's default subject, which has none (`nxr-neuro pyramid spatial-bands` on the default subject makes it)");
  std::string id;
  sub.compose([&] {
    DictionarySpec ds;
    ds.name = name; ds.scope = "subject";
    ds.description = "the measurement pyramid's spatial bands of " + surface + ": code = the ladder's level, its Weyl band";
    ds.entries = std::vector<DictionaryEntry>();
    for (const SpatialBandEdge& e : levels) {
      DictionaryEntry d;
      d.code = e.level; d.ordinal = e.level;
      d.name = "level " + std::to_string(e.level);
      d.cols.set("measurement", std::sqrt(e.lo * e.hi)).set("measurement_unit", "1/m\xc2\xb2").json("attributes_json", Json{{"lambda_min", e.lo}, {"lambda_max", e.hi}, {"level", e.level}});
      ds.entries->push_back(d);
    }
    const DictionaryResult dr = sub.dictionary(ds);
    SelectionSpec ss;
    ss.name = name; ss.type = "spans"; ss.manifold_id = mid;
    std::vector<SpanSpec> spans;
    for (const SpatialBandEdge& e : levels) spans.push_back({e.lo, e.hi, e.level});
    ss.spans = spans;
    ss.cols.set("cell", "1").set("dictionary_id", dr.id).set("unit", "1/m\xc2\xb2").set("description", "the spatial bands of " + surface + ": a span [lambda_min, lambda_max) of the Eigenvalue line per level of its tile ladder (the level's Weyl band), code = the level")
        .json("params_json", Json{{"spatial_bands", Json{{"surface", surface}, {"levels", static_cast<std::int64_t>(levels.size())}, {"unit", "1/m^2"}}}});
    id = sub.selection(std::move(ss));
  });
  return id;
}

std::string band_key_selection(SubjectWriter& sub, const std::string& bands_id, const std::string& spatial_bands_id, const std::string& name) {
  const auto b = sub.db().read("selection", bands_id), s = sub.db().read("selection", spatial_bands_id);
  if (!b || !s) throw Error(code::no_such_row, "band key: the frequency bands or the spatial bands Selection is missing");
  const std::int64_t nb = integer(*b, "n_members"), ns = integer(*s, "n_members");
  const Json key = Json{{"bands", bands_id}, {"spatial_bands", spatial_bands_id}, {"n_bands", nb}, {"n_spatial", ns}, {"layout", "code = band * n_spatial + spatial_band"}};
  if (const auto have = selection::find_by_name(sub.db(), sub.id(), name)) {
    const Json& p = have->params_json;
    if (!p.is_object() || !p.contains("band_key") || p["band_key"].value("bands", "") != bands_id || p["band_key"].value("spatial_bands", "") != spatial_bands_id)
      throw Error(code::invalid_argument, sub.name() + ": '" + name + "' is a band key of other banks");
    return have->id;
  }
  std::string id;
  sub.compose([&] {
    const std::string prod = sub.product({text(*b, "manifold_id"), text(*s, "manifold_id")}, std::nullopt, false);
    SelectionSpec js;
    js.name = name; js.type = "set"; js.manifold_id = prod; js.no_elements = true;
    js.cols.set("cell", "0,1").set("n_members", nb * ns).set("description", "the joint bands: a frequency band and a spatial band as one within key, code = band * " + std::to_string(ns) + " + spatial band")
        .json("params_json", Json{{"band_key", key}});
    id = sub.selection(std::move(js));
  });
  return id;
}

WithinKind within_kind(const Database& db, const std::optional<std::string>& within_selection_id) {
  WithinKind k;
  if (!within_selection_id) return k;
  const auto r = db.read("selection", *within_selection_id);
  if (!r) return k;
  const Json p = Json::parse(opt_text(*r, "params_json").value_or("{}"), nullptr, false);
  if (p.is_object() && p.contains("band_key")) { k.kind = WithinKind::BandKey; k.n_spatial = std::max<std::int64_t>(1, p["band_key"].value("n_spatial", std::int64_t(1))); }
  else if (p.is_object() && p.contains("spatial_bands")) k.kind = WithinKind::SpatialBands;
  else k.kind = WithinKind::Bands;
  return k;
}

std::int64_t band_key_code(std::int64_t band, std::int64_t spatial_band, std::int64_t n_spatial) { return band * n_spatial + spatial_band; }
void band_key_split(std::int64_t code, std::int64_t n_spatial, std::int64_t& band, std::int64_t& spatial_band) { band = code / n_spatial; spatial_band = code % n_spatial; }

}  // namespace nxr::io

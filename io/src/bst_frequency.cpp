// nxr::bst — nxr.convert.frequency: the Frequency manifold and the dataset's bands.
#include "nxr/bst/frequency.h"

#include "nxr/io/layout.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace code = nxr::io::code;
using nxr::io::Cols;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::SubjectWriter;

namespace {

constexpr const char* kBandDictionary = "frequency bands";

std::string band_dictionary_id(nxr::io::Database& db) { return nxr::io::ensure_band_dictionary(db); }

}  // namespace

std::string frequency_folder() { return nxr::io::folder_for("line", "", std::string("spectral")); }

std::vector<BandEntry> band_entries(nxr::io::Database& db) {
  const std::string did = band_dictionary_id(db);
  std::vector<BandEntry> out;
  for (const nxr::io::records::DictionaryEntryRow& e : nxr::io::selection::lookup_entries(db, did)) {
    const Json& a = e.attributes_json;
    if (a.is_object() && a.contains("lo_hz") && a.contains("hi_hz")) out.push_back({e.code, e.name, a["lo_hz"].get<double>(), a["hi_hz"].get<double>()});
  }
  return out;
}

std::optional<FrequencyRows> find_frequency(nxr::io::Database& db) {
  const auto m = db.one("SELECT m.* FROM manifold m JOIN subject s ON s.id = m.subject_id WHERE s.kind = 'template' AND m.name = ? AND m.type = 'line' "
                        "ORDER BY m.created_utc LIMIT 1", {std::string(kFrequencyName)});
  if (!m) return std::nullopt;
  const auto b = nxr::io::selection::find_on_manifold(db, nxr::io::text(*m, "id"), std::string(kBandsName), std::string("spans"));
  if (!b) return std::nullopt;
  return FrequencyRows{*m, *db.read("selection", b->id)};
}

FrequencyIds ensure_frequency(SubjectWriter& sub) {
  const std::string kind = nxr::io::text(sub.row(), "kind");
  if (kind != "template")
    throw nxr::io::Error(code::invalid_argument, sub.name() + ": the Frequency manifold lives in the dataset's default subject (a template), not in a " + kind);
  const auto have = sub.db().one("SELECT * FROM manifold WHERE subject_id = ? AND name = ?", {sub.id(), std::string(kFrequencyName)});
  std::string mid;
  if (!have) {
    nxr::io::ManifoldSpec ms;
    ms.name = kFrequencyName;
    ms.path = frequency_folder() + "/" + kFrequencyName;
    ms.type = "line";
    ms.n_vertices = kFrequencySamples;
    ms.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto lc = nxr::io::lattice_counts({kFrequencySamples});
    if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
    if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
    if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
    nxr::io::GeometrySpec g;
    g.form = "closed_form";
    g.cs_key = "frequency:line";   // charts `linear` (Hz, the reference), `log10` (schema 53) and `log2` (octaves); a Frequency line made before has the one chart it was made with
    nxr::io::DimensionSpec d;
    d.name = "frequency";
    d.n_vertices = kFrequencySamples;
    d.spacing = kFrequencySpacingHz;
    d.origin = 0.0;
    d.unit = "Hz";
    g.dimensions.push_back(d);
    ms.geometry = g;
    ms.cols.set("comment", "the frequency line every spectrum and band is read on: 0\xe2\x80\x93" "500 Hz in 0.5 Hz steps");
    mid = sub.manifold(std::move(ms));
  } else {
    mid = nxr::io::text(*have, "id");
  }
  if (const auto sel = nxr::io::selection::find_on_manifold(sub.db(), mid, std::string(kBandsName)))
    return {mid, sel->id};
  const std::string did = band_dictionary_id(sub.db());
  nxr::io::SelectionSpec s;
  s.name = kBandsName;
  s.path = frequency_folder() + "/" + kBandsName;
  s.type = "spans";
  s.manifold_id = mid;
  std::vector<nxr::io::SpanSpec> spans;
  for (const BandEntry& e : band_entries(sub.db())) spans.push_back({e.lo_hz, e.hi_hz, e.code});
  s.spans = spans;
  s.cols.set("cell", "1").set("dictionary_id", did).set("unit", "Hz")
      .set("description", std::string("the dataset's frequency bands: a span [lo, hi) per entry of the '") + kBandDictionary + "' dictionary, code = the entry's code");
  return {mid, sub.selection(std::move(s))};
}

Json band_ref(nxr::io::Database& db, const std::string& name, double lo_hz, double hi_hz) {
  Json ref = Json{{"name", name}, {"lo_hz", lo_hz}, {"hi_hz", hi_hz}, {"selection_id", nullptr}, {"code", nullptr}};
  const auto f = find_frequency(db);
  if (!f) return ref;
  auto low = [](std::string s) { for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; };
  for (const BandEntry& e : band_entries(db)) {
    if (low(e.name) == low(name)) {
      ref["selection_id"] = nxr::io::text(f->bands, "id");
      ref["code"] = e.code;
      return ref;
    }
  }
  return ref;
}

std::string band_field_name(const std::string& stem, const std::string& band) { return nxr::io::sanitize_node_name(stem + "_" + band); }

}  // namespace nxr::bst

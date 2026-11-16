// nxr::bst — nxr.convert.timefreq: source power by band as Fields.
#include "nxr/bst/timefreq.h"

#include <cmath>
#include <regex>
#include <set>

#include "bst_pyvalue.h"
#include "nxr/bst/frequency.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace code = nxr::io::code;
namespace fs = std::filesystem;
using nxr::io::Row;
using nxr::io::SubjectWriter;
using namespace detail;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

struct Band { std::string name; double low_hz, high_hz; std::string function; };

// Freqs: a cell of rows [name, 'lo, hi', function]
std::vector<Band> bands_of(const Value* freqs) {
  std::vector<Band> out;
  if (!freqs || !freqs->is_cell()) return out;
  const std::size_t n = freqs->rows(), c = freqs->cols();
  for (std::size_t r = 0; r < n; ++r) {
    const std::string name = py_str(&freqs->cell(r));
    const std::string rng = strip(c > 1 ? py_str(&freqs->cell(r + n)) : "");
    const std::string fn = c > 2 ? py_str(&freqs->cell(r + 2 * n)) : "mean";
    std::vector<double> edges;
    std::size_t i = 0;
    while (i < rng.size()) {
      while (i < rng.size() && (rng[i] == ',' || std::isspace(static_cast<unsigned char>(rng[i])))) ++i;
      std::size_t j = i;
      while (j < rng.size() && rng[j] != ',' && !std::isspace(static_cast<unsigned char>(rng[j]))) ++j;
      if (j > i) {
        char* end = nullptr;
        const std::string tok = rng.substr(i, j - i);
        const double x = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str() || *end) value_error("could not convert string to float: '" + tok + "'");
        edges.push_back(x);
      }
      i = j;
    }
    if (edges.size() != 2) value_error(edges.size() < 2 ? "not enough values to unpack (expected 2, got " + std::to_string(edges.size()) + ")"
                                                         : "too many values to unpack (expected 2)");
    out.push_back({name, edges[0], edges[1], fn});
  }
  return out;
}

std::optional<std::string> stage_of(const std::string& comment, const std::string& stem) {
  std::smatch m;
  if (std::regex_search(comment, m, std::regex(R"(power:\s*(\w+))"))) return m[1].str();
  if (std::regex_search(stem, m, std::regex(R"(srcpow_(\w+?)__)"))) return m[1].str();
  return std::nullopt;
}

// The averaged files, as Brainstorm lists them: `<subject>/<condition>/<file>`.
std::vector<std::string> contributions_of(const Value* history) {
  std::vector<std::string> out;
  if (!history || !history->is_cell()) return out;
  bool listing = false;
  const std::size_t n = history->rows(), c = history->cols();
  for (std::size_t r = 0; r < n; ++r) {
    const std::string text = c > 2 ? py_str(&history->cell(r + 2 * n)) : "";
    const std::string t = strip(text);
    if (text.find("List of averaged files") != std::string::npos) { listing = true; continue; }
    if (listing && starts_with(t, "- ")) out.push_back(strip(t.substr(2)));
    else if (listing && !t.empty()) listing = false;
  }
  return out;
}

}  // namespace

std::optional<Json> export_timefreq(SubjectWriter& sub, const fs::path& mat_path, const std::optional<std::string>& acquisition, bool group) {
  const fs::path& p = mat_path;
  const std::string fname = p.filename().string();
  {
    const auto probe = read_variables(p, {"DataType"});      // probed: a sensor-level TF is never loaded
    const auto it = probe.find("DataType");
    if (py_str(it == probe.end() ? nullptr : &it->second) != "results") return std::nullopt;
  }
  const std::map<std::string, Value> m = read_all_variables(p);
  const Value* TF = get(m, "TF");
  if (!TF || !TF->is_numeric()) return std::nullopt;
  std::vector<std::size_t> sq;
  for (std::size_t n : TF->dims) if (n != 1) sq.push_back(n);   // pymatreader squeezes; a [nSources, 1, nBands] array is [nSources, nBands]
  if (sq.size() != 2) return std::nullopt;
  const std::size_t n_src = sq[0], n_bands = sq[1];
  const std::vector<double> tf = TF->doubles();                 // tf[i, k] = tf[i + k * n_src]
  const std::string comment = py_str(get(m, "Comment"));
  const std::string stem = p.stem().string();
  const Value* sf = get(m, "SurfaceFile");
  const std::string surface = py_truthy(sf) ? surface_node_name(fs::path(py_str(sf)).filename()) : "";
  const std::vector<Band> bands = bands_of(get(m, "Freqs"));
  const std::optional<std::string> statistic = upper(comment).find("MEDIAN") != std::string::npos ? std::optional<std::string>("median")
                                                : group ? std::optional<std::string>("mean") : std::nullopt;
  const char* by_name = "SELECT * FROM manifold WHERE subject_id = ? AND type = 'surface' AND name = ? AND component_of_id IS NULL";
  std::optional<Row> surf;
  if (!surface.empty()) surf = sub.db().one(by_name, {sub.id(), surface});
  if (!surf && !surface.empty()) {
    const auto t = sub.db().one("SELECT * FROM subject WHERE dataset_id = (SELECT dataset_id FROM subject WHERE id = ?) AND kind = 'template' AND status = 'complete' "
                                "ORDER BY created_utc LIMIT 1", {sub.id()});
    if (t) surf = sub.db().one(by_name, {nxr::io::text(*t, "id"), surface});
  }
  if (!surf)
    value_error(fname + ": its surface '" + surface + "' is neither in this subject nor the dataset's template \xe2\x80\x94 convert the template first (D55)");
  std::vector<std::pair<std::string, std::string>> pending;       // (subject id, file)
  std::optional<std::int64_t> n_avg;
  if (group) {
    const std::vector<std::string> names = contributions_of(get(m, "History"));
    const Value* na = get(m, "nAvg");
    n_avg = na ? static_cast<std::int64_t>(na->numel() ? na->scalar() : 0) : static_cast<std::int64_t>(names.size());
    for (const std::string& c : names) {
      std::vector<std::string> parts;
      std::size_t i = 0;
      for (;;) { const auto j = c.find('/', i); parts.push_back(c.substr(i, j == std::string::npos ? j : j - i)); if (j == std::string::npos) break; i = j + 1; }
      parts.resize(std::max<std::size_t>(parts.size(), 3));
      const auto s = sub.db().one("SELECT * FROM subject WHERE dataset_id = (SELECT dataset_id FROM subject WHERE id = ?) AND name = ?", {sub.id(), parts[0]});
      if (!s) continue;           // a contributor not (yet) in the dataset is not a row; n_avg says how many
      pending.push_back({nxr::io::text(*s, "id"), parts[2]});
    }
  }
  if (n_bands != bands.size()) value_error(fname + ": TF has " + std::to_string(n_bands) + " columns, Freqs names " + std::to_string(bands.size()) + " bands");
  const auto stage = stage_of(comment, stem);
  Json producer = Json{{"stage", stage ? Json(*stage) : Json(nullptr)}, {"relative", ends_with(stem, "_relative")}, {"method", py_str(get(m, "Method"))},
                       {"measure", py_str(get(m, "Measure"))}, {"n_avg", n_avg ? Json(*n_avg) : Json(nullptr)}};
  std::vector<std::string> ids, fields;
  const std::string surf_id = nxr::io::text(*surf, "id");
  for (std::size_t k = 0; k < bands.size(); ++k) {            // ONE FIELD PER BAND on the cortex: a band is a Selection on the Frequency manifold
    const Band& b = bands[k];
    const std::string name = band_field_name(stem, b.name);
    std::vector<nxr::io::ContributionSpec> contribs;
    for (const auto& [subject_id, file] : pending) {            // the member's field of the same band
      std::optional<Row> src;
      if (!file.empty())
        src = sub.db().one("SELECT id FROM field WHERE subject_id = ? AND source = ? AND json_extract(producer_json, '$.band.name') = ?", {subject_id, file, b.name});
      nxr::io::ContributionSpec c;
      c.subject_id = subject_id;
      if (src) c.source_field_id = nxr::io::text(*src, "id");
      contribs.push_back(c);
    }
    std::vector<float> col(n_src);
    for (std::size_t i = 0; i < n_src; ++i) col[i] = static_cast<float>(tf[i + k * n_src]);
    char range[80];
    std::snprintf(range, sizeof range, "(%g-%g Hz)", b.low_hz, b.high_hz);
    nxr::io::FieldSpec f;
    f.name = name;
    f.path = nxr::io::node_path(sub.db(), surf_id, name);
    f.kind = "source power";
    f.manifold_id = surf_id;
    f.data = nxr::io::view(col, {static_cast<std::int64_t>(n_src)});
    f.acquisition = acquisition;
    f.contributions = contribs;
    Json prod = producer;
    prod["band"] = band_ref(sub.db(), b.name, b.low_hz, b.high_hz);
    prod["band_function"] = b.function;
    f.cols.set("description", comment + " \xe2\x80\x94 " + b.name + " " + range).set("source", fname).set("function", statistic).json("producer_json", prod);
    ids.push_back(sub.field(std::move(f)));
    fields.push_back(name);
  }
  return Json{{"name", stem}, {"id", ids.empty() ? Json(nullptr) : Json(ids[0])}, {"fields", fields}, {"n_sources", static_cast<std::int64_t>(n_src)},
              {"n_bands", static_cast<std::int64_t>(bands.size())}, {"sleep_stage", stage ? Json(*stage) : Json(nullptr)}, {"group", group}};
}

int export_condition_timefreqs(SubjectWriter& sub, const fs::path& cond_dir, const std::string& condition, const std::optional<std::string>& acquisition, const Emit& emit) {
  std::set<fs::path> files;
  for (const fs::path& dir : {cond_dir, cond_dir.parent_path() / ("@raw" + condition)}) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    for (const std::string& n : glob_names(dir, "timefreq_", ".mat")) files.insert(dir / n);
  }
  int n = 0;
  for (const fs::path& tf : files) {
    if (starts_with(tf.filename().string(), "._")) continue;
    if (auto info = export_timefreq(sub, tf, acquisition)) {
      Json line = Json{{"stage", "field"}};
      for (auto it = info->begin(); it != info->end(); ++it) line[it.key()] = it.value();
      emit(line);
      ++n;
    }
  }
  return n;
}

}  // namespace nxr::bst

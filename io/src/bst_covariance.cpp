// nxr::bst — a Brainstorm study's noise or data covariance as a stored statistic (nxr/bst/covariance.h).
#include "nxr/bst/covariance.h"

#include <cmath>

#include "bst_pyvalue.h"
#include "nxr/bst/protocol.h"

namespace fs = std::filesystem;
using nxr::io::Error;
using nxr::io::SubjectWriter;
using nxr::io::view;
using namespace nxr::bst::detail;
namespace code = nxr::io::code;

namespace nxr::bst {
namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

// Brainstorm's History rows as "date | type | text"
Json history_rows(const Value* h) {
  Json out = Json::array();
  if (!h || h->numel() == 0) return out;
  if (h->is_cell()) {
    const std::size_t n = h->rows(), c = h->cols();
    for (std::size_t r = 0; r < n; ++r) {
      std::string row;
      for (std::size_t j = 0; j < c; ++j) row += (j ? " | " : "") + py_str(&h->cell(r + j * n));
      out.push_back(row);
    }
  } else {
    out.push_back(py_str(h));
  }
  return out;
}

// the recording the covariance was computed from: the text of the history row of type `src`
std::optional<std::string> source_recording(const Value* h) {
  if (!h || !h->is_cell() || h->numel() == 0) return std::nullopt;
  const std::size_t n = h->rows(), c = h->cols();
  if (c < 3) return std::nullopt;
  for (std::size_t r = 0; r < n; ++r)
    if (lower(strip(py_str(&h->cell(r + 1 * n)))) == "src") return strip(py_str(&h->cell(r + 2 * n)));
  return std::nullopt;
}

// the study's record of this kind (`NoiseCov(1)` noise, `NoiseCov(2)` data) from brainstormstudy.mat, as a path in the study's folder; else the first file beside it
std::optional<fs::path> covariance_file(const fs::path& dir, bool data) {
  std::error_code ec;
  if (fs::is_regular_file(dir / "brainstormstudy.mat", ec)) {
    const auto rec = read_variables(dir / "brainstormstudy.mat", {"NoiseCov"});
    if (const Value* nc = get(rec, "NoiseCov"); nc && nc->is_struct() && nc->numel() > (data ? 1u : 0u)) {
      const std::size_t e = data ? 1 : 0;
      const std::string fn = py_str(nc->field("FileName", e));
      if (!fn.empty()) {
        const fs::path p = dir / fs::path(fn).filename();
        if (fs::is_regular_file(p, ec)) return p;
      }
    }
  }
  const auto names = glob_names(dir, data ? "ndatacov_" : "noisecov_", ".mat");
  for (const std::string& n : names) if (!starts_with(n, "._")) return dir / n;
  return std::nullopt;
}

}  // namespace

Json import_covariance(SubjectWriter& sub, const fs::path& dir_in, const std::string& study, const CovarianceOptions& opt, const Emit& progress) {
  // the ACQUISITION a study was imported as: its folder's name without Brainstorm's `@raw` prefix (the condition import's rule, bst_protocol)
  const std::string condition = starts_with(study, "@raw") ? study.substr(4) : study;
  if (opt.kind != "noise" && opt.kind != "data") refuse("--kind: '" + opt.kind + "' is not one of noise, data");
  const bool data = opt.kind == "data";
  const fs::path& dir = dir_in;
  const auto file = covariance_file(dir, data);
  if (!file) refuse(dir.string() + ": the study has no " + (data ? "data covariance (ndatacov_*.mat)" : "noise covariance (noisecov_*.mat)"));
  const auto m = read_all_variables(*file);
  const Value* cv = get(m, "NoiseCov");
  if (!cv || !cv->is_numeric() || cv->numel() == 0) refuse(file->filename().string() + ": not a covariance file (no NoiseCov)");
  const Mat c = as_mat(cv);
  if (c.rows != c.cols) refuse(file->filename().string() + ": NoiseCov is " + std::to_string(c.rows) + " x " + std::to_string(c.cols) + ", not square");

  // the channels it is on: the study's channel file and the channel manifold imported from it
  std::vector<fs::path> chans;
  for (const std::string& n : glob_names(dir, "channel", ".mat")) if (!starts_with(n, "._")) chans.push_back(dir / n);
  if (chans.size() != 1) throw Error(code::not_found, dir.string() + ": " + std::to_string(chans.size()) + " channel*.mat in the study, one is needed");
  const Value* chv = nullptr;
  const auto cm = read_variables(chans[0], {"Channel"});
  chv = get(cm, "Channel");
  const std::size_t n_file = chv ? chv->numel() : 0;
  if (n_file != c.rows)
    refuse(file->filename().string() + ": NoiseCov has " + std::to_string(c.rows) + " rows, the study's channel file " + std::to_string(n_file) + " channels");
  const auto ch = sub.db().one("SELECT id FROM manifold m WHERE m.subject_id = ? AND m.acquisition_id IN (SELECT id FROM acquisition WHERE subject_id = m.subject_id AND label = ?) "
                               "AND m.type = 'points' AND m.operator IS NULL ORDER BY path LIMIT 1", {sub.id(), condition});
  if (!ch) refuse(sub.name() + ": no channels in acquisition " + condition + " (import the condition first)");
  const std::string channels_id = nxr::io::text(*ch, "id");

  // the covered channels: the rows with a non-zero value, in the channel file's order
  std::vector<std::int32_t> picked;
  for (std::size_t r = 0; r < c.rows; ++r) {
    bool any = false;
    for (std::size_t k = 0; k < c.cols && !any; ++k) any = c.at(r, k) != 0.0;
    if (any) picked.push_back(static_cast<std::int32_t>(r));
  }
  if (picked.empty()) refuse(file->filename().string() + ": every row of NoiseCov is zero");
  const std::size_t n = picked.size();
  std::vector<double> sub_cov(n * n);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) sub_cov[i * n + j] = c.at(static_cast<std::size_t>(picked[i]), static_cast<std::size_t>(picked[j]));

  const std::string name = opt.name ? *opt.name : condition + (data ? "_data_covariance" : "_noise_covariance");
  const std::string sel_name = name + "_channels";
  const auto have = sub.db().one("SELECT id FROM statistic WHERE subject_id = ? AND name = ?", {sub.id(), name});
  const auto have_sel = sub.find("selection", sel_name);
  if (have || have_sel) {
    if (!opt.replace) refuse(sub.name() + ": " + name + " is already stored (--replace)");
    if (have) sub.remove("statistic", nxr::io::text(*have, "id"));
    if (have_sel) sub.remove("selection", nxr::io::text(*have_sel, "id"));
  }

  nxr::io::SelectionSpec sel;
  sel.name = sel_name;
  sel.path = sel_name;
  sel.type = "indices";
  sel.manifold_id = channels_id;
  sel.data = view(picked, {static_cast<std::int64_t>(n)});
  sel.picked = true;
  sel.acquisition = condition;
  sel.cols.set("description", name + " \xe2\x80\x94 the " + std::to_string(n) + " channels the covariance covers, in the channel file's order");
  const std::string sel_id = sub.selection(std::move(sel));

  Json options = Json{{"source", "Brainstorm NoiseCov"}, {"kind", opt.kind}, {"source_file", fs::relative(fs::absolute(*file), fs::absolute(opt.bst_root)).generic_string()}};
  if (const std::string cmt = text_field(get(m, "Comment")); !cmt.empty()) options["comment"] = cmt;
  const Json hist = history_rows(get(m, "History"));
  if (!hist.empty()) options["history"] = hist;
  if (const auto src = source_recording(get(m, "History"))) options["source_recording"] = *src;
  nxr::io::StatisticSpec st;
  st.name = name;
  st.measure = "covariance";
  st.method = "empirical";
  st.detached = true;
  st.axes = Json::array({Json{{"name", "channel"}, {"selection_id", sel_id}, {"size", static_cast<std::int64_t>(n)}},
                         Json{{"name", "channel"}, {"selection_id", sel_id}, {"size", static_cast<std::int64_t>(n)}}});
  st.data = view(sub_cov, {static_cast<std::int64_t>(n), static_cast<std::int64_t>(n)});
  st.array.compress = false;
  st.acquisition = condition;
  // the samples it was made from: the diagonal of nSamples at the first covered channel (Brainstorm's count per pair of channels)
  if (const Value* ns = get(m, "nSamples"); ns && ns->is_numeric() && ns->numel() == c.rows * c.cols) {
    const Mat nm = as_mat(ns);
    st.n_members = static_cast<std::int64_t>(nm.at(static_cast<std::size_t>(picked[0]), static_cast<std::size_t>(picked[0])));
    options["n_samples"] = *st.n_members;
  }
  st.options = options;
  st.description = std::string("the study's own ") + (data ? "data" : "noise") + " covariance, on its " + std::to_string(n) + " covered channels";
  const std::string id = sub.statistic(std::move(st));

  Json body = Json{{"name", name}, {"id", id}, {"kind", opt.kind}, {"channels", sel_name}, {"n_channels", static_cast<std::int64_t>(n)}, {"source_file", options["source_file"]}};
  if (options.contains("source_recording")) body["source_recording"] = options["source_recording"];
  Json line = Json{{"stage", "covariance"}};
  for (auto it = body.begin(); it != body.end(); ++it) line[it.key()] = it.value();
  if (progress) progress(line);
  return body;
}

}  // namespace nxr::bst

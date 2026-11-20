// nxr::bst — nxr.convert.inverse (the inverse kernel, the head model) and nxr.convert.condition's add_inverse and kernels step.
#include "nxr/bst/inverse.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>

#include "bst_pyvalue.h"
#include "nxr/bst/naming.h"
#include "nxr/bst/subject.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace code = nxr::io::code;
namespace fs = std::filesystem;
using nxr::io::Cols;
using nxr::io::Row;
using nxr::io::SubjectWriter;
using nxr::io::view;
using namespace detail;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
[[noreturn]] void not_found(const std::string& m) { throw nxr::io::Error(code::not_found, m); }

// ── the readers of the producer's record ──────────────────────────────────────────────────────────────────────────

const Value* opt_struct(const std::map<std::string, Value>& m, const char* name) {
  const Value* v = get(m, name);
  return v && v->is_struct() ? v : nullptr;
}

// `_set_if`: the value as JSON unless empty
void set_if(Json& attrs, const std::string& key, const Value* v) {
  if (py_empty(v)) return;
  attrs[key] = to_json(*v);
}

// `_source_orient`: the strings of a (nested) list, a lone string as a one-element list, anything else dropped
Json source_orient(const Json& j) {
  Json out = Json::array();
  if (j.is_string()) {
    if (!j.get<std::string>().empty()) out.push_back(j);
  } else if (j.is_array()) {
    for (const auto& e : j) for (const auto& x : source_orient(e)) out.push_back(x);
  }
  return out;
}

// `_history`: Brainstorm's History rows as "date | type | text"
std::optional<Json> history_of(const Value* h) {
  if (!h || h->numel() == 0) return std::nullopt;
  Json out = Json::array();
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

// `_rel`: a path relative to `root` when it starts with it; the path itself otherwise (null when empty)
Json rel(const std::string& p, const std::string& root) {
  if (!root.empty() && starts_with(p, root)) {
    std::string r = p.substr(root.size());
    r.erase(0, r.find_first_not_of('/') == std::string::npos ? r.size() : r.find_first_not_of('/'));
    return r;
  }
  return p.empty() ? Json(nullptr) : Json(p);
}

std::string path_str(const fs::path& p) { return py_path(p.string()); }

// `s` and `num` of inverse_method_of
std::optional<std::string> s_of(const Json& p, const char* key) {
  if (!p.contains(key)) return std::nullopt;
  const Json& v = p[key];
  if (v.is_string()) return v.get<std::string>();
  if (v.is_array() && !v.empty() && v[0].is_string()) return v[0].get<std::string>();
  return std::nullopt;
}
std::string s_or(const Json& p, const char* key, const std::string& dflt) {
  const auto s = s_of(p, key);
  return s && !s->empty() ? *s : dflt;
}
std::optional<Json> num_of(const Json& p, const char* key) {
  if (!p.contains(key)) return std::nullopt;
  const Json& v = p[key];
  if (!v.is_number()) return std::nullopt;
  if (v.is_number_float() && !std::isfinite(v.get<double>())) return std::nullopt;
  return v;
}
Json num_or_null(const Json& p, const char* key) { const auto n = num_of(p, key); return n ? *n : Json(nullptr); }

// the 0-based kernel columns: GoodChannel is 1-BASED (MATLAB)
std::vector<std::int32_t> good_channels(const Value* v) {
  std::vector<std::int32_t> out;
  if (v && v->is_numeric()) for (std::int64_t g : v->int64s()) out.push_back(static_cast<std::int32_t>(g - 1));
  return out;
}

}  // namespace

std::optional<Json> inverse_method_of(const Json& p) {
  if (!p.contains("inverse_method") && !p.contains("inverse_measure")) return std::nullopt;
  const std::string m = lower(s_or(p, "inverse_measure", ""));
  std::string measure;
  if (m == "amplitude" || m == "current") measure = "MNE";
  else if (starts_with(m, "dspm")) measure = "dSPM";
  else if (m == "sloreta") measure = "sLORETA";
  else if (m == "eloreta") measure = "eLORETA";
  else measure = s_or(p, "method", m.empty() ? "unknown" : m);
  const std::string orientation = s_or(p, "source_orient", s_or(p, "source_ori", "unknown"));
  std::optional<Json> snr;
  if (s_of(p, "snr_method") == std::optional<std::string>("fixed")) snr = num_of(p, "snr_fixed");
  Json depth = nullptr;
  if (!(p.contains("use_depth") && p["use_depth"].is_boolean() && !p["use_depth"].get<bool>()) && num_of(p, "weight_exp"))
    depth = Json{{"exponent", *num_of(p, "weight_exp")}, {"limit", num_or_null(p, "weight_limit")}};
  Json lambda2 = nullptr;
  if (snr && snr->get<double>() != 0) lambda2 = 1 / (snr->get<double>() * snr->get<double>());
  return Json{{"measure", measure}, {"estimate", measure == "MNE" ? "current" : "statistic"}, {"solver", s_or(p, "inverse_method", "unknown")},
              {"orientation", orientation}, {"loose", orientation == "loose" ? num_or_null(p, "loose") : Json(nullptr)}, {"depth", depth},
              {"noise", Json{{"method", s_or(p, "noise_method", "unknown")}, {"reg", num_or_null(p, "noise_reg")}}},
              {"snr", snr ? *snr : Json(nullptr)}, {"lambda2", lambda2},
              {"from", p.contains("bst_results_file") && p["bst_results_file"].is_string() ? "brainstorm" : "producer"}};
}

Json inverse_json(const InverseInfo& i) {
  return Json{{"name", i.name}, {"id", i.id}, {"channels_id", i.channels_id}, {"method", i.method}, {"units", i.units},
              {"nsrc", i.nsrc}, {"M", i.M}, {"n_components", i.n_components}};
}

InverseInfo export_inverse(SubjectWriter& sub, const std::map<std::string, Value>& kernel, const InverseOptions& opt) {
  const std::string& name = opt.name;
  const Value* K = get(kernel, "ImagingKernel");
  if (!K || !K->is_numeric() || K->numel() == 0) value_error("not an inverse KERNEL results file (need .ImagingKernel)");
  const std::int64_t nsrc = static_cast<std::int64_t>(K->rows()), m = static_cast<std::int64_t>(K->cols());
  // WHICH CHANNELS: `GoodChannel` is 1-BASED (MATLAB); the Indices are 0-based, in the kernel's column order
  const std::vector<std::int32_t> picked = good_channels(get(kernel, "GoodChannel"));
  if (static_cast<std::int64_t>(picked.size()) != m)
    value_error(name + ": GoodChannel names " + std::to_string(picked.size()) + " channels for a kernel with " + std::to_string(m) + " columns");
  Json ch_names = Json::array();
  if (opt.channel_names) for (std::int32_t i : picked) ch_names.push_back(opt.channel_names->at(static_cast<std::size_t>(i)));
  const Value* nc = get(kernel, "nComponents");
  const std::int64_t n_comp = py_truthy(nc) ? static_cast<std::int64_t>(nc->scalar()) : 1;
  const bool is_free = n_comp == 3;
  if (opt.n_vertices && nsrc != *opt.n_vertices * std::max<std::int64_t>(1, n_comp))
    value_error("ImagingKernel rows (" + std::to_string(nsrc) + ") != nComponents*nV (" + std::to_string(*opt.n_vertices * std::max<std::int64_t>(1, n_comp)) + ")");

  const Value* opts = opt_struct(kernel, "Options");
  auto of = [&](const char* k) { return sfield(opts, k); };
  const std::string im = lower(py_str(of("InverseMethod"))), ims = lower(py_str(of("InverseMeasure")));
  std::string method = "unknown", comment_tag;
  if (im == "minnorm" && ims == "amplitude") { method = "MNE"; comment_tag = "MN"; }
  else if (im == "minnorm" && ims == "dspm2018") { method = "dSPM"; comment_tag = "dSPM-unscaled"; }
  else if (im == "minnorm" && ims == "sloreta") { method = "sLORETA"; comment_tag = "sLORETA"; }
  const Value* du = get(kernel, "DisplayUnits");
  const std::string units = ims == "amplitude" ? "A.m" : ims == "dspm2018" ? "dSPM" : ims == "sloreta" ? "sLORETA" : (py_empty(du) ? "" : py_str(du));
  Json producer = Json{{"method", method}, {"method_comment", comment_tag}, {"units", units}, {"ch_names", ch_names}, {"n_channels", m}, {"n_sources", nsrc},
                       {"n_components", n_comp}, {"source_ori", is_free ? "free" : "fixed"}};
  set_if(producer, "inverse_method", of("InverseMethod"));
  set_if(producer, "inverse_measure", of("InverseMeasure"));
  const Value* fn = py_truthy(get(kernel, "Function")) ? get(kernel, "Function") : of("FunctionName");
  set_if(producer, "function", fn);
  if (const Value* so = of("SourceOrient"); so) {
    // `_source_orient`: always a list of strings (a char or a 1x1 cell -> [it], a cell of chars -> the list, empty -> [])
    producer["source_orient"] = source_orient(to_json(*so));
  }
  for (const auto& [key, src] : {std::pair<const char*, const char*>{"loose", "Loose"}, {"weight_exp", "WeightExp"}, {"weight_limit", "WeightLimit"},
                                 {"noise_method", "NoiseMethod"}, {"noise_reg", "NoiseReg"}, {"snr_method", "SnrMethod"}, {"snr_rms", "SnrRms"},
                                 {"snr_fixed", "SnrFixed"}})
    set_if(producer, key, of(src));
  if (const Value* ud = of("UseDepth"); !py_empty(ud)) producer["use_depth"] = ud->is_numeric() ? ud->scalar() != 0 : true;
  for (const auto& [key, src] : {std::pair<const char*, const char*>{"n_avg", "nAvg"}, {"leff", "Leff"}, {"display_units", "DisplayUnits"},
                                 {"colormap_type", "ColormapType"}, {"head_model_type", "HeadModelType"}, {"bst_data_file", "DataFile"},
                                 {"bst_head_model_file", "HeadModelFile"}, {"bst_surface_file", "SurfaceFile"}})
    set_if(producer, key, get(kernel, src));
  if (!opt.kernel_file.empty()) producer["bst_results_file"] = rel(opt.kernel_file, opt.bst_root);
  if (const auto h = history_of(get(kernel, "History"))) producer["bst_history"] = *h;
  if (opt.source_sha1 && !opt.source_sha1->empty()) producer["source_sha1"] = *opt.source_sha1;
  if (opt.producer_extra) for (auto it = opt.producer_extra->begin(); it != opt.producer_extra->end(); ++it) producer[it.key()] = it.value();
  producer = jsonable(producer);
  method = producer["method"].get<std::string>();
  const std::string units_out = producer["units"].get<std::string>();
  const std::string stamp = nxr::io::now_utc(false);

  nxr::io::SelectionSpec sel;
  sel.name = name + "_channels";
  sel.path = sel.name;
  sel.type = "indices";
  sel.manifold_id = opt.channels_id;
  sel.data = view(picked, {m});
  sel.picked = true;
  sel.acquisition = opt.acquisition;
  sel.cols.set("description", name + " \xe2\x80\x94 the " + std::to_string(m) + " channels it maps, in column order");
  const std::string sel_id = sub.selection(std::move(sel));

  // THE PAIR (D106): W of G - a kernel derived from a leadfield whose ends are its swapped is its regularized inverse
  Cols pair;
  if (opt.forward_id) {
    const auto g = sub.db().read("operator", *opt.forward_id);
    if (g && nxr::io::text(*g, "kind") == "forward" && nxr::io::text(*g, "to_manifold_id") == opt.channels_id && nxr::io::text(*g, "from_manifold_id") == opt.surface_id) {
      const std::string want = is_free ? "vector3" : "scalar";
      const auto fvt = nxr::io::opt_text(*g, "from_value_type"), tvt = nxr::io::opt_text(*g, "to_value_type");
      if ((!fvt || fvt->empty() || *fvt == want) && (!tvt || *tvt == "scalar")) pair.set("inverts_id", *opt.forward_id).set("inverse_kind", "regularized");
    }
  }
  const auto inv = inverse_method_of(producer);
  const std::vector<double> kd = K->row_major();
  std::vector<float> kf32(kd.begin(), kd.end());
  nxr::io::OperatorSpec op;
  op.name = name;
  op.path = name;
  op.kind = "inverse kernel";
  op.from_manifold_id = opt.channels_id;
  op.layout = "dense";
  op.n_rows = nsrc;
  op.n_cols = m;
  op.data = view(kf32, {nsrc, m});
  op.array.chunks = nxr::io::default_chunks({nsrc, m}, 4);
  op.array.compress = false;
  op.acquisition = opt.acquisition;
  op.cols = pair;
  op.cols.set("to_manifold_id", opt.surface_id).set("from_selection_id", sel_id).set("from_value_type", "scalar").set("to_value_type", is_free ? "vector3" : "scalar")
      .set("derived_from_id", opt.forward_id).set("description", name + " \xe2\x80\x94 sensor field to source field on the cortex (" + method + ")");
  op.created_utc = stamp;                // the spec's own field: the row's modified_utc is its created_utc (nxr's _prov)
  if (const Value* c = get(kernel, "Comment"); !py_empty(c)) op.cols.set("comment", py_str(c));
  if (inv) op.cols.json("params_json", Json{{"inverse", *inv}});
  op.cols.json("producer_json", producer);
  const std::string kid = sub.operator_(std::move(op));

  // THE NOISE COVARIANCE AND THE WHITENER: operators on the kernel's channels (channels -> channels)
  auto on_channels = [&](nxr::io::OperatorSpec& o, const std::string& oname, const std::string& kind, const Mat& a, const std::string& description,
                         const std::optional<std::string>& derived) {
    o.name = oname;
    o.path = oname;
    o.kind = kind;
    o.from_manifold_id = opt.channels_id;
    o.layout = "dense";
    o.n_rows = static_cast<std::int64_t>(a.rows);
    o.n_cols = static_cast<std::int64_t>(a.cols);
    o.array.compress = false;
    o.acquisition = opt.acquisition;
    o.cols.set("to_manifold_id", opt.channels_id).set("from_selection_id", sel_id).set("to_selection_id", sel_id).set("from_value_type", "scalar")
        .set("to_value_type", "scalar").set("derived_from_id", derived).set("description", description);
    o.cols.json("producer_json", Json{{"kernel", name}});
  };
  const Value* ncm = of("NoiseCovMat");
  const Value* ncov = sfield(ncm, "NoiseCov");
  std::optional<std::string> cov_id;
  const std::string ms = std::to_string(m);
  if (ncov && ncov->is_numeric() && ncov->numel()) {
    // a COVARIANCE IS A STATISTIC (the zero-lag cross-spectrum), imported as computed elsewhere: of no field in this package, its provenance in the options
    const Mat c = as_mat(ncov);
    nxr::io::StatisticSpec st;
    st.name = name + "_noise_cov";
    st.measure = "covariance";
    st.method = "empirical";
    st.detached = true;
    st.axes = Json::array({Json{{"name", "channel"}, {"selection_id", sel_id}, {"size", static_cast<std::int64_t>(c.rows)}},
                           Json{{"name", "channel"}, {"selection_id", sel_id}, {"size", static_cast<std::int64_t>(c.cols)}}});
    st.data = view(c.d, {static_cast<std::int64_t>(c.rows), static_cast<std::int64_t>(c.cols)});
    st.array.compress = false;
    st.acquisition = opt.acquisition;
    Json sopt = Json{{"source", "Brainstorm NoiseCovMat.NoiseCov"}, {"kernel", name}};
    if (const Value* na = sfield(ncm, "nAvg"); na && na->is_numeric() && na->numel() == 1) {
      st.n_members = static_cast<std::int64_t>(na->scalar());
      sopt["nAvg"] = *st.n_members;
    }
    st.options = sopt;
    st.description = "the noise covariance " + name + " was computed with, on its " + ms + " channels";
    cov_id = sub.statistic(std::move(st));
  }
  if (const Value* whit = get(kernel, "Whitener"); whit && whit->is_numeric() && whit->numel()) {
    const Mat w = as_mat(whit);
    nxr::io::OperatorSpec o;
    on_channels(o, name + "_whitener", "whitener", w, "the whitener of " + name + "'s noise covariance, on its " + ms + " channels", std::nullopt);
    if (cov_id) o.cols.set("of_statistic_id", *cov_id);     // derived from the covariance STATISTIC
    o.data = view(w.d, {static_cast<std::int64_t>(w.rows), static_cast<std::int64_t>(w.cols)});
    sub.operator_(std::move(o));
  }

  // THE SOURCE GEOMETRY: kernel-row-ordered, fields OF the kernel on the cortex
  if (opt.grid_loc && !opt.grid_loc->empty()) {
    const std::vector<double>& gl = *opt.grid_loc;
    const std::size_t nv = gl.size() / 3;
    std::vector<double> rr, nn;
    if (is_free) {
      for (std::size_t i = 0; i < nv; ++i) for (int k = 0; k < 3; ++k) rr.insert(rr.end(), gl.begin() + static_cast<std::ptrdiff_t>(3 * i), gl.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
      for (std::size_t i = 0; i < nv; ++i) for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) nn.push_back(r == c ? 1.0 : 0.0);
    } else {
      rr = gl;
      if (opt.grid_orient) nn = *opt.grid_orient;
    }
    const std::int64_t rows = static_cast<std::int64_t>(rr.size() / 3);
    if (rows != nsrc) value_error("source geometry (" + std::to_string(rows) + ") != kernel rows (" + std::to_string(nsrc) + ")");
    const std::string vt = is_free ? "matrix3" : "vector3";    // rows 3V x 3 (one row a component) | V x 3
    auto geom = [&](const std::string& suffix, const std::string& kind, const std::vector<double>& a, bool unit) {
      std::vector<float> f(a.begin(), a.end());
      nxr::io::FieldSpec fs;
      fs.name = name + suffix;
      fs.path = fs.name;
      fs.kind = kind;
      fs.manifold_id = opt.surface_id;
      fs.value_type = vt;
      fs.data = view(f, {static_cast<std::int64_t>(f.size() / 3), 3});
      fs.array.compress = false;
      fs.acquisition = opt.acquisition;
      fs.cols.set("of_operator_id", kid);
      if (unit) fs.cols.set("unit", "m");
      sub.field(std::move(fs));
    };
    geom("_source_rr", "source_rr", rr, true);
    if (!nn.empty()) geom("_source_nn", "source_nn", nn, false);
  }
  return {name, kid, sel_id, method, units_out, nsrc, m, n_comp};
}

ForwardInfo export_forward(SubjectWriter& sub, const std::map<std::string, Value>& head_model, const ForwardOptions& opt) {
  const Value* GL = get(head_model, "GridLoc");
  const std::int64_t n_src = GL && GL->is_numeric() && GL->numel() ? static_cast<std::int64_t>(as_mat(GL).rows) : 0;
  const bool has_grid = n_src > 0;
  Json producer = Json{{"units", "m"}, {"n_sources", n_src}, {"has_source_grid", has_grid}};
  for (const auto& [key, src] : {std::pair<const char*, const char*>{"head_model_type", "HeadModelType"}, {"meg_method", "MEGMethod"},
                                 {"eeg_method", "EEGMethod"}, {"bst_surface_file", "SurfaceFile"}})
    set_if(producer, key, get(head_model, src));
  const Value* fnv = get(head_model, "FileName");
  const std::string hmf = py_truthy(fnv) ? py_str(fnv) : opt.head_model_file;
  if (!hmf.empty()) producer["bst_head_model_file"] = rel(hmf, opt.bst_root.empty() ? "" : path_str(fs::path(opt.bst_root) / "data"));
  if (opt.source_sha1 && !opt.source_sha1->empty()) producer["source_sha1"] = *opt.source_sha1;
  const Value* gain = get(head_model, "Gain");
  std::vector<float> g;
  std::int64_t gr = opt.n_channels, gc = 3 * opt.n_vertices;
  const bool with_gain = opt.write_gain && gain && gain->is_numeric() && gain->numel();
  if (with_gain) {
    const std::vector<double> gd = gain->row_major();
    g.assign(gd.begin(), gd.end());
    gr = static_cast<std::int64_t>(gain->rows());
    gc = static_cast<std::int64_t>(gain->cols());
  }
  // DECLARED, BYTES ABSENT: a head model without its gain is still the head model the kernel names
  nxr::io::OperatorSpec op;
  op.name = opt.name;
  op.path = opt.name;
  op.kind = "forward";
  op.from_manifold_id = opt.surface_id;
  op.layout = "dense";
  op.n_rows = gr;
  op.n_cols = gc;
  if (with_gain) { op.data = view(g, {gr, gc}); op.array.compress = true; }
  op.acquisition = opt.acquisition;
  op.cols.set("to_manifold_id", opt.channels_id).set("from_value_type", "vector3").set("to_value_type", "scalar")
      .set("description", opt.name + " \xe2\x80\x94 the leadfield, source field to sensor field");
  op.created_utc = nxr::io::now_utc(false);
  if (const Value* c = get(head_model, "Comment"); !py_empty(c)) op.cols.set("comment", py_str(c));
  op.cols.json("producer_json", jsonable(producer));
  const std::string hid = sub.operator_(std::move(op));
  auto vec3 = [&](const std::string& suffix, const std::vector<double>& a, bool unit) {
    std::vector<float> f(a.begin(), a.end());
    nxr::io::FieldSpec fs;
    fs.name = opt.name + suffix;
    fs.path = fs.name;
    fs.kind = suffix.substr(1);
    fs.manifold_id = opt.surface_id;
    fs.value_type = "vector3";
    fs.data = view(f, {static_cast<std::int64_t>(f.size() / 3), 3});
    fs.acquisition = opt.acquisition;
    fs.cols.set("of_operator_id", hid);
    if (unit) fs.cols.set("unit", "m");
    sub.field(std::move(fs));
  };
  if (has_grid) vec3("_GridLoc", as_mat(GL).d, true);
  if (const Value* go = get(head_model, "GridOrient"); go && go->is_numeric() && go->numel()) vec3("_GridOrient", as_mat(go).d, false);
  return {opt.name, hid, n_src};
}

namespace {

// _spatial_head_models: the headmodel_*.mat beside a kernel with a source grid
std::vector<fs::path> spatial_head_models(const fs::path& cond_dir) {
  std::vector<fs::path> out;
  std::error_code ec;
  if (!fs::is_directory(cond_dir, ec)) return out;
  for (const std::string& n : glob_names(cond_dir, "headmodel_", ".mat")) {
    const auto g = read_variables(cond_dir / n, {"GridLoc"});
    const auto it = g.find("GridLoc");
    if (it != g.end() && it->second.numel()) out.push_back(cond_dir / n);
  }
  return out;
}

std::vector<double> rows_of(const Value* v) { return v && v->is_numeric() && v->numel() ? as_mat(v).d : std::vector<double>(); }

Json with_stage(const char* stage, const Json& body) {
  Json line = Json{{"stage", stage}};
  for (auto it = body.begin(); it != body.end(); ++it) line[it.key()] = it.value();
  return line;
}

// _kernel_head_model: the head model a kernel names: the recorded link, else (a stale link is normal) the one spatial head model beside it
std::optional<fs::path> kernel_head_model(const fs::path& kernel_file, const std::map<std::string, Value>& kern, const fs::path& bst_root, const Emit& progress) {
  const Value* hv = get(kern, "HeadModelFile");
  const std::string link = py_truthy(hv) ? py_str(hv) : "";
  if (link.empty()) return std::nullopt;
  const fs::path p = bst_root / "data" / link;
  std::error_code ec;
  if (fs::is_regular_file(p, ec)) return p;
  const auto spatial = spatial_head_models(kernel_file.parent_path());
  if (spatial.size() == 1) {
    progress(Json{{"stage", "stale-headmodel-link"}, {"file", path_str(kernel_file)}, {"link", link}, {"resolved", spatial[0].filename().string()}});
    return spatial[0];
  }
  not_found(kernel_file.filename().string() + ": head model " + link + " is missing and " + std::to_string(spatial.size()) + " candidates are beside it");
}

}  // namespace

InverseInfo add_inverse(SubjectWriter& sub, const fs::path& kernel_path, const AddInverseOptions& a, const Emit& progress) {
  const fs::path kernel_file = path_str(kernel_path), bst_root = path_str(a.bst_root);
  const std::map<std::string, Value> kern = read_all_variables(kernel_file);
  const Value* K = get(kern, "ImagingKernel");
  if (!K || !K->numel()) value_error(kernel_file.filename().string() + ": not an inverse KERNEL results file (no ImagingKernel)");
  std::string channels_id;
  if (a.channels_id) channels_id = *a.channels_id;
  else {
    const auto ch = sub.db().one("SELECT id FROM manifold m WHERE m.subject_id = ? AND m.acquisition_id IN (SELECT id FROM acquisition WHERE subject_id = m.subject_id AND label = ?) "
                                 "AND m.type = 'points' AND m.operator IS NULL ORDER BY path LIMIT 1", {sub.id(), a.acquisition});
    if (!ch) value_error(sub.name() + ": no channels in acquisition " + a.acquisition);
    channels_id = nxr::io::text(*ch, "id");
  }
  const Value* sfv = get(kern, "SurfaceFile");
  const std::string sf = py_truthy(sfv) ? py_str(sfv) : "";
  std::string surface_id;
  if (a.surface_id) surface_id = *a.surface_id;
  else {
    std::optional<Row> row;
    if (!sf.empty()) {
      const std::string sname = surface_node_name(fs::path(sf).filename());
      row = sub.find("manifold", nxr::io::folder_for("surface", sname) + "/" + sname);
    }
    if (!row) value_error(kernel_file.filename().string() + ": its surface " + (sf.empty() ? "(none)" : sf) + " is not a manifold of " + sub.name());
    surface_id = nxr::io::text(*row, "id");
  }
  const Row surface = *sub.db().read("manifold", surface_id);
  const auto hm = kernel_head_model(kernel_file, kern, bst_root, progress);
  std::optional<std::vector<double>> gl, go;
  std::optional<std::string> forward_id = a.forward_id;
  if (hm) {
    const auto g = read_variables(*hm, {"GridLoc", "GridOrient"});
    if (const Value* v = get(g, "GridLoc"); v && v->numel()) gl = rows_of(v);
    if (const Value* v = get(g, "GridOrient"); v && v->numel()) go = rows_of(v);
    if (!forward_id) {
      const auto row = sub.db().one("SELECT id FROM operator o WHERE o.subject_id = ? AND o.kind = 'forward' AND o.acquisition_id IN (SELECT id FROM acquisition WHERE subject_id = o.subject_id AND label = ?) "
                                    "AND name LIKE ? ORDER BY name LIMIT 1", {sub.id(), a.acquisition, headmodel_node_name(*hm) + "%"});
      if (row) forward_id = nxr::io::text(*row, "id");
    }
  }
  const std::string fp = source_fingerprint({kernel_file}, channels_id + "|" + surface_id + "|" + (forward_id ? *forward_id : "None"));
  std::string name;
  if (a.name) name = *a.name;
  else {
    const std::string stem = kernel_node_name(kernel_file);
    if (const auto had = sub.find("operator", stem)) {
      const Json pj = Json::parse(nxr::io::opt_text(*had, "producer_json").value_or("{}"), nullptr, false);
      // `a.acquisition` is a LABEL, a row's acquisition_id an id: compare like with like
      bool in_acquisition = false;
      if (const auto aid = nxr::io::opt_text(*had, "acquisition_id")) {
        const auto ar = sub.db().one("SELECT label FROM acquisition WHERE id = ?", {*aid});
        in_acquisition = ar && nxr::io::opt_text(*ar, "label") == std::optional<std::string>(a.acquisition);
      }
      if (in_acquisition || (pj.is_object() && pj.contains("source_sha1") && pj["source_sha1"] == fp))
        value_error(sub.name() + ": " + stem + " is already in acquisition " + a.acquisition + " \xe2\x80\x94 refusing to add it twice");
    }
    name = claim_node_name(sub, "", stem, a.acquisition, fp).first;
  }
  std::vector<std::string> taken;
  for (const char* suffix : {"", "_channels", "_whitener", "_noise_cov", "_source_rr", "_source_nn"}) {
    const std::string part = name + suffix;
    for (const char* t : {"operator", "selection", "field", "manifold"}) if (sub.find(t, part)) { taken.push_back(part); break; }
  }
  if (!taken.empty()) {
    std::string list;
    for (std::size_t i = 0; i < taken.size(); ++i) list += (i ? ", " : "") + taken[i];
    value_error(sub.name() + ": refusing to overwrite " + list);
  }
  fs::path channel_file;
  if (a.channel_file) channel_file = *a.channel_file;
  else {
    std::vector<fs::path> chans;
    std::error_code ec;
    if (fs::is_directory(kernel_file.parent_path(), ec))
      for (const std::string& n : glob_names(kernel_file.parent_path(), "channel", ".mat")) if (!starts_with(n, "._")) chans.push_back(kernel_file.parent_path() / n);
    if (chans.size() != 1) not_found(path_str(kernel_file.parent_path()) + ": " + std::to_string(chans.size()) + " channel*.mat \xe2\x80\x94 pass --channel-file");
    channel_file = chans[0];
  }
  std::vector<std::string> channel_names;
  {
    const auto cm = read_variables(channel_file, {"Channel"});
    const Value* ch = get(cm, "Channel");
    if (!ch) throw nxr::io::Error(code::no_such_row, "'Channel'");
    if (ch->is_struct() && ch->has_field("Name")) for (std::size_t e = 0; e < ch->numel(); ++e) channel_names.push_back(py_str(ch->field("Name", e)));
  }
  InverseOptions io;
  io.acquisition = a.acquisition;
  io.name = name;
  io.channels_id = channels_id;
  io.surface_id = surface_id;
  io.forward_id = forward_id;
  io.kernel_file = kernel_file.string();
  io.bst_root = bst_root.string();
  io.channel_names = channel_names;
  io.n_vertices = nxr::io::integer(surface, "n_vertices");
  io.grid_loc = gl;
  io.grid_orient = go;
  io.source_sha1 = fp;
  const InverseInfo info = export_inverse(sub, kern, io);
  progress(with_stage("inverse", inverse_json(info)));
  return info;
}

std::vector<std::string> import_condition_kernels(SubjectWriter& sub, const ConditionKernels& c, const Emit& emit) {
  struct Done { std::string name, id; std::vector<double> grid_loc, grid_orient; bool has_loc = false, has_orient = false; };
  std::vector<fs::path> kernel_files;
  if (c.kernel_files) kernel_files = *c.kernel_files;
  else {
    std::error_code ec;
    if (fs::is_directory(c.cond_dir, ec)) for (const std::string& n : glob_names(c.cond_dir, "results_", ".mat", "KERNEL")) kernel_files.push_back(c.cond_dir / n);
  }
  std::map<std::string, Done> fwd_done;
  std::vector<std::string> inv_names;
  for (const fs::path& kf : kernel_files) {
    or_skip_unreadable_v73(sub, emit, [&] {   // a kernel or its head model in MATLAB v7.3 this build cannot read: that kernel is skipped, named
      if (starts_with(kf.filename().string(), "._")) return;
      const std::map<std::string, Value> kern = read_all_variables(kf);
      if (const Value* K = get(kern, "ImagingKernel"); !K || !K->numel()) {
        emit(Json{{"stage", "skip-kernel"}, {"file", path_str(kf)}, {"reason", "no ImagingKernel"}});
        return;
      }
      const Value* hv = get(kern, "HeadModelFile");
      const std::string hm_link = py_truthy(hv) ? py_str(hv) : "";
      std::string hm_path;
      if (const auto it = c.head_model_map.find(hm_link); it != c.head_model_map.end() && !it->second.empty()) hm_path = it->second;
      else if (!hm_link.empty()) hm_path = path_str(c.bst_root / "data" / hm_link);
      std::error_code ec;
      if (!hm_path.empty() && !fs::is_regular_file(hm_path, ec)) {
        // Brainstorm never rewrites a kernel's HeadModelFile when the head model is renamed: stale links are NORMAL. One spatial head model beside it is the answer.
        const auto spatial = spatial_head_models(c.cond_dir);
        if (spatial.size() == 1) {
          emit(Json{{"stage", "stale-headmodel-link"}, {"file", path_str(kf)}, {"link", hm_link}, {"resolved", spatial[0].filename().string()}});
          hm_path = spatial[0].string();
        } else {
          Json cands = Json::array();
          for (const auto& s : spatial) cands.push_back(s.filename().string());
          emit(Json{{"stage", "skip-kernel"}, {"file", path_str(kf)}, {"reason", "head model missing: " + hm_link}, {"candidates", cands}});
          return;
        }
      }
      std::optional<std::vector<double>> gl, go;
      std::optional<std::string> fwd_id;
      if (!hm_path.empty()) {
        // PROBED, and once per head model: the Gain (~84 MB) is decoded only when the forward is written
        auto have = fwd_done.find(hm_path);
        std::map<std::string, Value> grid;
        if (have == fwd_done.end()) grid = read_variables(hm_path, {"GridLoc", "GridOrient"});
        const Value* gv = get(grid, "GridLoc");
        if (have == fwd_done.end() && c.require_source_grid && (!gv || gv->numel() == 0)) {
          emit(Json{{"stage", "skip-kernel"}, {"file", path_str(kf)}, {"reason", "modal head model (empty GridLoc)"}});
          return;
        }
        if (have == fwd_done.end()) {
          const std::string hm_fp = source_fingerprint({hm_path}, c.channels_id + "|" + c.surface_id + "|" + (c.write_gain ? "True" : "False"));
          const auto [hm_name, hm_shared] = claim_node_name(sub, "", headmodel_node_name(hm_path), c.acquisition, hm_fp);
          Done d;
          if (hm_shared) {
            d.name = hm_name;
            d.id = nxr::io::text(*sub.find("operator", hm_name), "id");
            emit(Json{{"stage", "forward-shared"}, {"name", hm_name}});
          } else {
            ForwardOptions fo;
            fo.acquisition = c.acquisition;
            fo.name = hm_name;
            fo.channels_id = c.channels_id;
            fo.surface_id = c.surface_id;
            fo.n_channels = c.n_channels;
            fo.n_vertices = c.n_vertices;
            fo.head_model_file = hm_path;
            fo.bst_root = c.bst_root.string();
            fo.write_gain = c.write_gain;
            fo.source_sha1 = hm_fp;
            const ForwardInfo fi = export_forward(sub, read_all_variables(hm_path), fo);
            d.name = fi.name;
            d.id = fi.id;
            emit(Json{{"stage", "forward"}, {"name", fi.name}});
          }
          d.grid_loc = rows_of(gv);
          d.has_loc = gv && gv->numel();
          d.grid_orient = rows_of(get(grid, "GridOrient"));
          d.has_orient = !d.grid_orient.empty();
          have = fwd_done.emplace(hm_path, std::move(d)).first;
        }
        const Done& f = have->second;
        fwd_id = f.id;
        if (f.has_loc) gl = f.grid_loc;
        if (f.has_orient) go = f.grid_orient;
      }
      const std::string k_fp = source_fingerprint({kf}, c.channels_id + "|" + c.surface_id + "|" + (fwd_id ? *fwd_id : "None"));
      const auto [k_name, k_shared] = claim_node_name(sub, "", kernel_node_name(kf), c.acquisition, k_fp);
      std::string ii_name, ii_method;
      if (k_shared) {
        const Json pj = Json::parse(nxr::io::opt_text(*sub.find("operator", k_name), "producer_json").value_or("{}"), nullptr, false);
        ii_name = k_name;
        ii_method = pj.is_object() && pj.contains("method") && pj["method"].is_string() ? pj["method"].get<std::string>() : "unknown";
      } else {
        InverseOptions io;
        io.acquisition = c.acquisition;
        io.name = k_name;
        io.channels_id = c.channels_id;
        io.surface_id = c.surface_id;
        io.forward_id = fwd_id;
        io.kernel_file = path_str(kf);
        io.bst_root = path_str(c.bst_root);
        io.channel_names = c.channel_names;
        io.n_vertices = c.n_vertices;
        io.grid_loc = gl;
        io.grid_orient = go;
        io.source_sha1 = k_fp;
        const InverseInfo ii = export_inverse(sub, kern, io);
        ii_name = ii.name;
        ii_method = ii.method;
      }
      inv_names.push_back(ii_name);
      emit(Json{{"stage", "inverse"}, {"name", ii_name}, {"method", ii_method}});
    });
  }
  // NO KERNEL IS NOT AN ERROR: a legitimate anatomy + timeseries session
  if (inv_names.empty()) emit(Json{{"stage", "no-kernel"}, {"message", "no inverse kernel in this condition \xe2\x80\x94 the session is anatomy + timeseries"}});
  return inv_names;
}

}  // namespace nxr::bst

// The importers of MNE's FIF object files: see nxr/convert/fif_objects.h.
#include "nxr/convert/fif_objects.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

#include "nxr/bst/entities.h"
#include "nxr/convert/sensor_frames.h"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/fif_objects.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"

namespace nxr::convert {

namespace fif = nxr::formats::fif;
namespace code = nxr::io::code;
using nxr::io::Json;
using nxr::io::view;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string stem_of(const std::string& path) {
  std::string n = std::filesystem::path(path).filename().string();
  for (const char* suf : {".fif.gz", ".fif"}) {
    const std::string s = suf;
    if (n.size() > s.size() && lower(n).compare(n.size() - s.size(), s.size(), s) == 0) { n.resize(n.size() - s.size()); break; }
  }
  for (const char* suf : {"-src", "-cov", "-fwd", "-inv", "-eve", "-annot", "-trans", "_src", "_cov", "_fwd", "_inv", "_eve", "_annot", "_trans"}) {
    const std::string s = suf;
    if (n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0) { n.resize(n.size() - s.size()); break; }
  }
  return nxr::io::sanitize_node_name(n.empty() ? "fif" : n);
}

fif::File open_fif(const std::string& path) {
  if (!std::filesystem::exists(path)) throw nxr::io::Error(code::not_found, "no such file: " + path);
  return fif::File(std::make_shared<nxr::formats::FileSource>(path));
}

// ---- the package's sensor manifold ----------------------------------------------------------------------------------------------------

std::vector<std::string> member_names(nxr::io::SubjectWriter& sub, const std::string& channels_id) {
  auto& db = sub.db();
  const auto sel = db.one("SELECT * FROM selection WHERE manifold_id = ? AND name LIKE '%\\_names' ESCAPE '\\' AND member_of_id IS NULL", {nxr::io::V(channels_id)});
  if (!sel) return {};
  const std::string rel = nxr::io::opt_text(*sel, "array_path") ? *nxr::io::opt_text(*sel, "array_path") : nxr::io::text(*sel, "path");
  const auto bytes = nxr::io::ArrayReader::open(sub.store(), rel).read();
  std::map<std::int64_t, std::string> by;
  for (const auto& e : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {nxr::io::V(nxr::io::text(*sel, "dictionary_id"))}))
    by[nxr::io::integer(e, "code")] = nxr::io::text(e, "name");
  std::vector<std::string> out;
  for (std::size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    std::int32_t c;
    std::memcpy(&c, bytes.data() + i, 4);
    out.push_back(by.count(c) ? by[c] : "");
  }
  return out;
}

struct Channels {
  std::string id, name;
  std::vector<std::string> names;
};

Channels find_channels(nxr::io::SubjectWriter& sub, const std::string& wanted) {
  Channels c;
  if (!wanted.empty()) {
    const auto m = sub.by_name("manifold", wanted);
    if (!m) refuse("--channels " + wanted + ": the package has no such sensor manifold");
    c.id = nxr::io::text(*m, "id");
    c.name = wanted;
  } else {
    std::vector<nxr::io::Row> found;
    for (const auto& m : sub.db().all("SELECT * FROM manifold WHERE subject_id = ? AND type = 'points'", {nxr::io::V(sub.id())}))
      if (!member_names(sub, nxr::io::text(m, "id")).empty()) found.push_back(m);
    if (found.empty()) refuse("the package has no sensor manifold (import the recording first, or name one with --channels)");
    if (found.size() > 1) {
      std::string l;
      for (const auto& m : found) l += (l.empty() ? "" : ", ") + nxr::io::text(m, "name");
      refuse("the package has several sensor manifolds (" + l + "): name one with --channels");
    }
    c.id = nxr::io::text(found[0], "id");
    c.name = nxr::io::text(found[0], "name");
  }
  c.names = member_names(sub, c.id);
  if (c.names.empty()) refuse("--channels " + c.name + ": it is not a sensor manifold with named members");
  return c;
}

// the file's channels as positions of the manifold's; refused by name when one is missing
std::vector<std::int32_t> match_channels(const Channels& ch, const std::vector<std::string>& file_names) {
  std::map<std::string, std::int32_t> at;
  for (std::size_t i = 0; i < ch.names.size(); ++i) at[ch.names[i]] = static_cast<std::int32_t>(i);
  std::vector<std::int32_t> idx;
  std::string missing;
  for (const auto& n : file_names) {
    const auto it = at.find(n);
    if (it == at.end()) missing += (missing.empty() ? "" : ", ") + n;
    else idx.push_back(it->second);
  }
  if (!missing.empty()) refuse("the file's channels are not all in " + ch.name + ": missing " + missing);
  return idx;
}

std::string channel_selection(nxr::io::SubjectWriter& sub, const Channels& ch, const std::vector<std::int32_t>& idx, const std::string& name, const std::string& acq) {
  nxr::io::SelectionSpec sel;
  sel.name = name;
  sel.path = name;
  sel.type = "indices";
  sel.manifold_id = ch.id;
  sel.data = view(idx, {static_cast<std::int64_t>(idx.size())});
  sel.picked = true;
  sel.acquisition = acq;
  sel.cols.set("description", name + " \xe2\x80\x94 the " + std::to_string(idx.size()) + " channels of the file, in the file's order");
  return sub.selection(std::move(sel));
}

// ---- frames ---------------------------------------------------------------------------------------------------------------------------

std::string frame_key(std::int32_t f) {
  if (f == 5) return "freesurfer-tkr";
  if (f == 4) return "head";
  refuse("the file's coordinate frame (" + std::to_string(f) + ") is neither the MRI (5) nor the head (4) frame");
}

void ensure_frame(nxr::io::SubjectWriter& sub, const std::string& key) {
  if (key == "head") sub.frame({"head", "head", std::string("m"), "the Neuromag head frame: x toward the right ear, y toward the nasion, z up", ""});
}

Json producer_of(const FifObjectImport& opt, const std::string& kind) {
  return Json{{"format", "fif"}, {"object", kind}, {"reader", "nxr-neuro"}, {"anonymized", opt.anonymize}};
}

// ---- source spaces --------------------------------------------------------------------------------------------------------------------

struct SourceSpaceIds {
  std::string manifold_id, selection_id, frame, type;
  std::int64_t n_vertices = 0, n_used = 0;
};

SourceSpaceIds write_source_space(nxr::io::SubjectWriter& sub, const fif::SourceSpaces& src, const std::string& name, const FifObjectImport& opt) {
  if (src.spaces.empty()) refuse("the file has no source space");
  const std::string type = src.spaces[0].type;
  for (const auto& s : src.spaces) {
    if (s.type != type) refuse("a source space mixing " + type + " and " + s.type + " spaces is not imported");
    if (s.coord_frame != src.spaces[0].coord_frame) refuse("the source spaces are in different coordinate frames");
  }
  SourceSpaceIds out;
  out.type = type;
  out.frame = frame_key(src.spaces[0].coord_frame);
  ensure_frame(sub, out.frame);
  std::vector<std::int32_t> used;
  Json spaces = Json::array();
  for (const auto& s : src.spaces) spaces.push_back(Json{{"id", s.id}, {"np", s.np}, {"nuse", s.nuse}, {"ntri", s.ntri}, {"nuse_tri", s.nuse_tri}});
  Json prod = producer_of(opt, "source space");
  prod["spaces"] = spaces;
  prod["frame"] = out.frame == "head" ? "head" : "mri";
  if (!opt.anonymize) {
    if (src.working_dir) prod["working_dir"] = *src.working_dir;
    if (src.command_line) prod["command_line"] = *src.command_line;
  }
  const std::string sel_name = name + "_source_space";

  if (type == "surf") {
    std::int64_t np = 0;
    for (const auto& s : src.spaces) np += s.np;
    std::vector<double> rr;
    std::vector<std::int32_t> tris;
    std::vector<nxr::io::Hemisphere> hemis;
    std::int64_t off = 0;
    for (const auto& s : src.spaces) {
      if (s.np <= 0 || static_cast<std::int64_t>(s.rr.size()) != 3 * s.np) refuse("a surface source space with no vertices");
      rr.insert(rr.end(), s.rr.begin(), s.rr.end());
      for (auto t : s.tris) tris.push_back(static_cast<std::int32_t>(t + off));
      for (auto v : s.vertno) used.push_back(static_cast<std::int32_t>(v + off));
      if (src.spaces.size() > 1) hemis.push_back({s.id == 102 ? "right" : "left", off, s.np});
      off += s.np;
    }
    if (!opt.surface.empty()) {
      const auto m = sub.by_name("manifold", opt.surface);
      if (!m) refuse("--surface " + opt.surface + ": the package has no such surface");
      if (nxr::io::integer(*m, "n_vertices") != np)
        refuse("--surface " + opt.surface + " has " + std::to_string(nxr::io::integer(*m, "n_vertices")) + " vertices; the source space's surface has " + std::to_string(np));
      out.manifold_id = nxr::io::text(*m, "id");
    } else {
      if (tris.empty()) refuse("the surface source space holds no triangles of its surface (needed to make its surface)");
      nxr::io::SurfaceOptions so;
      so.name = name + "_surface";
      so.primary = false;
      so.source = "FIF source space";
      so.cs_key = out.frame;
      so.hemispheres = hemis;
      so.producer = prod;
      out.manifold_id = nxr::io::add_surface(sub, so, rr, tris).id;
    }
    out.n_vertices = np;
  } else if (type == "vol") {
    if (src.spaces.size() != 1) refuse("a volume source space of several volumes is not imported");
    const fif::SourceSpace& s = src.spaces[0];
    if (!s.vox_mri_t) refuse("the volume source space has no voxel-to-MRI transform");
    if (s.shape.size() != 3) refuse("the volume source space has no voxel grid shape");
    const std::int64_t nx = s.shape[0], ny = s.shape[1], nz = s.shape[2];
    if (nx * ny * nz != s.np) refuse("the volume source space's grid does not match its vertex count");
    const auto& T = s.vox_mri_t->trans;   // voxel index -> MRI, metres
    std::array<double, 3> sp{};
    for (int c = 0; c < 3; ++c) sp[c] = 1000.0 * std::sqrt(T[c] * T[c] + T[4 + c] * T[4 + c] + T[8 + c] * T[8 + c]);
    const std::string vname = name + "_volume";
    nxr::io::ManifoldSpec ms;
    ms.name = vname;
    ms.path = "mri/" + vname;
    ms.type = "volume";
    ms.n_vertices = s.np;
    ms.topology = {"lattice", 3, 1, std::nullopt, {}};
    const auto lc = nxr::io::lattice_counts({nx, ny, nz});
    if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
    if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
    if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
    nxr::io::GeometrySpec g;
    g.form = "closed_form";
    g.cs_key = "voxel:" + vname;
    const std::int64_t dims[3] = {nx, ny, nz};
    for (int d = 0; d < 3; ++d) {
      nxr::io::DimensionSpec ds;
      ds.name = std::string(1, "ijk"[d]);
      ds.n_vertices = dims[d];
      ds.spacing = sp[d];
      ds.unit = "mm";
      g.dimensions.push_back(ds);
    }
    ms.geometry = g;
    ms.cols.json("producer_json", prod);
    out.manifold_id = sub.manifold(std::move(ms));
    std::array<double, 12> mm{};
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) mm[4 * r + c] = 1000.0 * T[4 * r + c];
    nxr::io::OperatorSpec op;
    op.name = vname + "_" + (out.frame == "head" ? "head" : "tkr");
    op.path = op.name;
    op.kind = "transition";
    op.from_manifold_id = out.manifold_id;
    op.cols.set("from_coordinate_system_id", sub.cs("voxel:" + vname));
    op.cols.set("to_coordinate_system_id", sub.cs(out.frame));
    op.n_rows = 3;
    op.n_cols = 4;
    op.data = view(mm.data(), {3, 4});
    op.cols.set("description", "voxel index \xe2\x86\x92 " + out.frame + " millimetres (3x4 affine, voxel size folded in), from the source space's voxel-to-MRI transform");
    op.cols.json("producer_json", Json{{"convention", "frame_mm = M @ [i, j, k, 1]"}});
    sub.operator_(std::move(op));
    // MNE numbers a grid's vertices with the first axis fastest; the lattice manifold is in C order
    for (auto v : s.vertno) {
      const std::int64_t i = v % nx, j = (v / nx) % ny, k = v / (nx * ny);
      used.push_back(static_cast<std::int32_t>((i * ny + j) * nz + k));
    }
    std::sort(used.begin(), used.end());
    out.n_vertices = s.np;
  } else if (type == "discrete") {
    std::vector<double> rr;
    std::int64_t np = 0;
    for (const auto& s : src.spaces) {
      rr.insert(rr.end(), s.rr.begin(), s.rr.end());
      for (auto v : s.vertno) used.push_back(static_cast<std::int32_t>(v + np));
      np += s.np;
    }
    const std::string pname = name + "_points";
    const std::string path = nxr::io::folder_for("points") + "/" + pname;
    const std::string mid = nxr::io::uuid7();
    nxr::io::ManifoldSpec m;
    m.id = mid;
    m.name = pname;
    m.path = path;
    m.type = "points";
    m.n_vertices = np;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    nxr::io::GeometrySpec g;
    g.form = "stored";
    g.cs_key = out.frame;
    g.positions_path = path;
    m.geometry = g;
    std::vector<float> pos(rr.begin(), rr.end());
    m.populate = [&sub, path, &pos, np, mid](nxr::io::Store&, const std::string&) { sub.write_array(path, view(pos, {np, 3}), {}, std::nullopt, "manifold", mid); };
    m.cols.json("producer_json", prod);
    out.manifold_id = sub.manifold(std::move(m));
    out.n_vertices = np;
  } else {
    refuse("a source space of type '" + type + "' is not imported");
  }
  std::sort(used.begin(), used.end());
  nxr::io::SelectionSpec sel;
  sel.name = sel_name;
  sel.path = (type == "vol" ? "mri/" : "source/") + sel_name;
  sel.type = "indices";
  sel.manifold_id = out.manifold_id;
  sel.data = view(used, {static_cast<std::int64_t>(used.size())});
  sel.picked = true;
  sel.acquisition = opt.acquisition;
  sel.cols.set("description", "the source space: the " + std::to_string(used.size()) + " of " + std::to_string(out.n_vertices) + " vertices the source space uses (MNE's vertno)");
  sel.cols.json("params_json", Json{{"source_space", type}});
  out.selection_id = sub.selection(std::move(sel));
  out.n_used = static_cast<std::int64_t>(used.size());
  return out;
}

// ---- covariances ----------------------------------------------------------------------------------------------------------------------

std::string cov_label(std::int32_t kind) {
  switch (kind) {
    case fif::mne::kNoiseCov: return "noise_cov";
    case fif::mne::kSignalCov: return "data_cov";
    case fif::mne::kSourceCov: return "source_cov";
    case fif::mne::kFmriPriorCov: return "fmri_prior_cov";
    case fif::mne::kDepthPriorCov: return "depth_prior_cov";
    case fif::mne::kOrientPriorCov: return "orient_prior_cov";
  }
  return "cov" + std::to_string(kind);
}

Json cov_options(const fif::Covariance& c, const FifObjectImport& opt) {
  Json o = Json{{"source", "FIF covariance (kind " + std::to_string(c.kind) + ")"}, {"diagonal", c.diag}, {"eigen_decomposition_in_file", !c.eig.empty()}};
  if (c.nfree >= 0) o["nfree"] = c.nfree;
  if (c.method) o["estimator"] = *c.method;
  if (c.score) o["score"] = *c.score;
  Json bads = Json::array();
  for (const auto& b : c.bads) bads.push_back(b);
  o["bads"] = bads;
  Json projs = Json::array();
  for (const auto& p : c.projs) projs.push_back(Json{{"description", p.desc}, {"active", p.active}, {"n_vectors", p.nvec}});
  o["projectors"] = projs;
  o["anonymized"] = opt.anonymize;
  return o;
}

std::string write_covariance(nxr::io::SubjectWriter& sub, const fif::Covariance& c, const std::string& name, const std::string& channel_sel_id, std::optional<std::string> derived_from_ignored,
                             const FifObjectImport& opt) {
  (void)derived_from_ignored;
  const std::int64_t m = c.dim;
  std::vector<double> full(static_cast<std::size_t>(m * m), 0.0);
  if (c.diag) {
    if (static_cast<std::int64_t>(c.data.size()) != m) refuse(name + ": the diagonal has " + std::to_string(c.data.size()) + " values for " + std::to_string(m) + " channels");
    for (std::int64_t i = 0; i < m; ++i) full[static_cast<std::size_t>(i * m + i)] = c.data[static_cast<std::size_t>(i)];
  } else {
    if (static_cast<std::int64_t>(c.data.size()) != m * m) refuse(name + ": the matrix is not " + std::to_string(m) + " x " + std::to_string(m));
    full = c.data;
  }
  nxr::io::StatisticSpec st;
  st.name = name;
  st.measure = "covariance";
  st.method = "empirical";
  st.detached = true;
  st.axes = Json::array({Json{{"name", "channel"}, {"selection_id", channel_sel_id}, {"size", m}}, Json{{"name", "channel"}, {"selection_id", channel_sel_id}, {"size", m}}});
  st.data = view(full, {m, m});
  st.array.compress = false;
  st.acquisition = opt.acquisition;
  if (c.nfree >= 0) { st.n_members = c.nfree; st.dof = static_cast<double>(c.nfree); }
  st.options = cov_options(c, opt);
  st.description = "the " + cov_label(c.kind) + " the file holds, on its " + std::to_string(m) + " channels (computed elsewhere: no recording of this package)";
  return sub.statistic(std::move(st));
}

// ---- operators from named matrices ----------------------------------------------------------------------------------------------------

std::string write_matrix_operator(nxr::io::SubjectWriter& sub, const std::string& name, const std::string& kind, const std::string& from_manifold, const std::string& to_manifold,
                                  std::int64_t rows, std::int64_t cols, const std::vector<float>& data, const std::string& from_vt, const std::string& to_vt,
                                  const std::optional<std::string>& from_sel, const std::optional<std::string>& to_sel, const std::string& description, const Json& producer,
                                  const FifObjectImport& opt) {
  nxr::io::OperatorSpec op;
  op.name = name;
  op.path = name;
  op.kind = kind;
  op.from_manifold_id = from_manifold;
  op.layout = "dense";
  op.n_rows = rows;
  op.n_cols = cols;
  op.data = view(data, {rows, cols});
  op.array.chunks = nxr::io::default_chunks({rows, cols}, 4);
  op.array.compress = true;
  op.acquisition = opt.acquisition;
  op.cols.set("to_manifold_id", to_manifold).set("from_value_type", from_vt).set("to_value_type", to_vt).set("description", description);
  if (from_sel) op.cols.set("from_selection_id", *from_sel);
  if (to_sel) op.cols.set("to_selection_id", *to_sel);
  op.cols.json("producer_json", producer);
  return sub.operator_(std::move(op));
}

std::vector<float> to_f32(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

// the transpose of a row-major [r x c] matrix
std::vector<double> transpose(const std::vector<double>& a, std::int64_t r, std::int64_t c) {
  std::vector<double> t(a.size());
  for (std::int64_t i = 0; i < r; ++i) for (std::int64_t j = 0; j < c; ++j) t[static_cast<std::size_t>(j * r + i)] = a[static_cast<std::size_t>(i * c + j)];
  return t;
}

Json parent_record(const fif::ParentInfo& p, const FifObjectImport& opt) {
  Json r = Json::object();
  r["n_channels"] = p.channels.size();
  Json bads = Json::array();
  for (const auto& b : p.bads) bads.push_back(b);
  r["bads"] = bads;
  r["custom_ref_applied"] = p.custom_ref_applied;
  if (!opt.anonymize) {
    if (p.mri_file) r["mri_file"] = *p.mri_file;
    if (p.meas_file) r["meas_file"] = *p.meas_file;
  }
  return r;
}

std::string head_mri_transition(nxr::io::SubjectWriter& sub, const fif::ParentInfo& p, const std::string& name, const std::string& from_manifold) {
  if (!p.mri_head_t) return "";
  return import_head_mri_transition(sub, *p.mri_head_t, name + "_head_to_mri", from_manifold);
}

// ---- events and annotations -----------------------------------------------------------------------------------------------------------

struct RecordingRef {
  std::string id, name, line_id, base;
  double sfreq = 0, t0 = 0;
  std::optional<std::string> acquisition_id;
  std::optional<double> meas_date;   // seconds since the epoch
};

RecordingRef find_recording(nxr::io::SubjectWriter& sub, const std::string& name) {
  if (name.empty()) refuse("an event or annotation file belongs to a recording: name it with --recording");
  const auto f = sub.by_name("field", name);
  if (!f || nxr::io::text(*f, "kind") != "recording") refuse("--recording " + name + ": the package has no such recording");
  RecordingRef r;
  r.id = nxr::io::text(*f, "id");
  r.name = name;
  std::vector<std::string> factors;
  for (const auto& d : sub.db().all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE p.id = ? ORDER BY d.ordinal", {nxr::io::V(nxr::io::text(*f, "manifold_id"))}))
    factors.push_back(nxr::io::text(d, "manifold_id"));
  if (factors.size() != 2) refuse(name + ": not a sensor x time recording");
  r.line_id = factors[1];
  const auto d = sub.db().one("SELECT spacing, origin FROM geometry_dimension WHERE geometry_id = (SELECT geometry_id FROM manifold WHERE id = ?)", {nxr::io::V(r.line_id)});
  if (!d || nxr::io::is_null(*d, "spacing")) refuse(name + ": its time line has no spacing");
  r.sfreq = 1.0 / std::get<double>(d->at("spacing"));
  r.base = nxr::io::node_path(sub.db(), r.line_id, name);
  if (!nxr::io::is_null(*f, "acquisition_id")) r.acquisition_id = nxr::io::text(*f, "acquisition_id");
  if (const auto pj = nxr::io::opt_text(*f, "producer_json")) {
    const Json p = Json::parse(*pj, nullptr, false);
    if (p.is_object() && p.contains("meas_date") && p["meas_date"].is_array() && p["meas_date"].size() == 2)
      r.meas_date = p["meas_date"][0].get<double>() + 1e-6 * p["meas_date"][1].get<double>();
  }
  return r;
}

void refuse_if_events_exist(nxr::io::SubjectWriter& sub, const RecordingRef& rec) {
  for (const char* suffix : {"_events"})
    if (sub.find("selection", rec.base + suffix)) refuse(rec.name + " already has events (" + rec.base + suffix + "): this import never overwrites them");
}

// ---- the dispatch ---------------------------------------------------------------------------------------------------------------------

std::string detect(const fif::File& f) {
  using namespace fif;
  if (!find_blocks(f.tree(), block::kMneInverseSolution).empty()) return "inv";
  if (!find_blocks(f.tree(), block::kMneForwardSolution).empty()) return "fwd";
  if (!find_blocks(f.tree(), block::kMneCov).empty()) return "cov";
  if (!find_blocks(f.tree(), block::kMneSourceSpace).empty()) return "src";
  if (!find_blocks(f.tree(), block::kMneEvents).empty()) return "eve";
  if (!find_blocks(f.tree(), block::kMneAnnotations).empty()) return "annot";
  for (const auto& t : f.directory()) if (t.kind == tag::kCoordTrans) return "trans";
  refuse("the file holds none of a source space, covariance, forward solution, inverse operator, events, annotations or coordinate transform");
}
}  // namespace

std::string fif_object_kind(const std::string& path) {
  const fif::File f = open_fif(path);
  return detect(f);
}

Json import_fif_object(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const FifObjectImport& opt) {
  const fif::File f = open_fif(path);
  const std::string kind = opt.kind.empty() ? detect(f) : opt.kind;
  const std::string name = opt.name.empty() ? stem_of(path) : opt.name;
  Json out = Json{{"kind", kind}, {"name", name}};
  Json origin = Json{{"schema", "nxr-origin/1"}, {"source", "fif:" + (opt.source.empty() ? std::filesystem::path(path).filename().string() : opt.source)}, {"licence", opt.licence},
                     {"attribution", opt.attribution}, {"redistributable", opt.redistributable}};
  if (!opt.source_file.empty()) {
    std::ifstream in(opt.source_file, std::ios::binary);
    nxr::io::Sha256 h;
    std::vector<char> buf(1 << 20);
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(in.gcount()));
    }
    origin["source"] = origin["source"].get<std::string>() + "#sha256=" + h.hex();
  }

  sub.compose([&] {
    if (kind == "src") {
      const fif::SourceSpaces src = fif::read_source_spaces(f);
      const SourceSpaceIds ids = write_source_space(sub, src, name, opt);
      out["frame"] = ids.frame;
      out["source_space"] = ids.type;
      out["n_vertices"] = ids.n_vertices;
      out["n_used"] = ids.n_used;
    } else if (kind == "cov") {
      const std::vector<fif::Covariance> covs = fif::read_covariances(f);
      if (covs.empty()) refuse("the file holds no covariance");
      const Channels ch = find_channels(sub, opt.channels);
      Json items = Json::array();
      for (const auto& c : covs) {
        const auto idx = match_channels(ch, c.names);
        const std::string label = cov_label(c.kind);
        const std::string sel = channel_selection(sub, ch, idx, name + "_" + label + "_channels", opt.acquisition);
        write_covariance(sub, c, name + "_" + label, sel, std::nullopt, opt);
        items.push_back(Json{{"statistic", name + "_" + label}, {"channels", c.dim}});
      }
      out["covariances"] = items;
    } else if (kind == "fwd") {
      const fif::ForwardFile fw = fif::read_forward(f);
      if (fw.solutions.empty()) refuse("the file holds no forward solution");
      std::int64_t nchan = 0;
      for (const auto& s : fw.solutions) nchan += s.nchan;
      if (static_cast<std::int64_t>(fw.parent.channels.size()) != nchan) refuse("the file's channel list does not match its forward solution's rows");
      const std::int32_t ori = fw.solutions[0].source_ori, frame = fw.solutions[0].coord_frame;
      for (const auto& s : fw.solutions) if (s.source_ori != ori || s.coord_frame != frame || s.nsource != fw.solutions[0].nsource) refuse("the forward solutions differ in orientation, frame or sources");
      const bool free_ori = ori == fif::mne::kFreeOri;
      const std::int64_t V = fw.solutions[0].nsource, comps = free_ori ? 3 * V : V;
      const SourceSpaceIds ids = write_source_space(sub, fw.src, name, opt);
      if (ids.n_used != V) refuse("the forward solution has " + std::to_string(V) + " sources; its source space uses " + std::to_string(ids.n_used) + " vertices");
      const Channels ch = find_channels(sub, opt.channels);
      std::vector<std::string> names;
      for (const auto& c : fw.parent.channels) names.push_back(c.name);
      const std::string csel = channel_selection(sub, ch, match_channels(ch, names), name + "_channels", opt.acquisition);
      std::vector<double> gain;   // [nchan x comps]: the stored blocks are [comps x channels of the block]; the blocks' channels follow one another
      gain.reserve(static_cast<std::size_t>(nchan * comps));
      for (const auto& s : fw.solutions) {
        if (s.sol.nrow != comps || s.sol.ncol != s.nchan) refuse("a forward solution's matrix is not " + std::to_string(comps) + " x " + std::to_string(s.nchan));
        const std::vector<double> t = transpose(s.sol.data, s.sol.nrow, s.sol.ncol);   // [nchan_block x comps]
        gain.insert(gain.end(), t.begin(), t.end());
      }
      Json prod = producer_of(opt, "forward solution");
      prod["component_frame"] = frame == 4 ? "head" : "mri";
      prod["orientation"] = free_ori ? "free" : "fixed";
      prod["methods"] = fw.solutions.size() == 1 ? (fw.solutions[0].methods == fif::mne::kEeg ? "EEG" : "MEG") : "MEG+EEG";
      prod["units"] = "channel unit per A m";
      prod["parent"] = parent_record(fw.parent, opt);
      const std::string fid = write_matrix_operator(sub, name, "forward", ids.manifold_id, ch.id, nchan, comps, to_f32(gain), free_ori ? "vector3" : "scalar", "scalar", ids.selection_id,
                                                    csel, name + " \xe2\x80\x94 the forward solution of the file, source field to sensor field", prod, opt);
      head_mri_transition(sub, fw.parent, name, ch.id);
      out["operator"] = name;
      out["n_channels"] = nchan;
      out["n_sources"] = V;
      out["orientation"] = prod["orientation"];
      out["operator_id"] = fid;
    } else if (kind == "inv") {
      const fif::InverseOperator iv = fif::read_inverse(f);
      const bool free_ori = iv.source_ori == fif::mne::kFreeOri;
      const std::int64_t V = iv.nsource, comps = free_ori ? 3 * V : V, rank = static_cast<std::int64_t>(iv.sing.size());
      const SourceSpaceIds ids = write_source_space(sub, iv.src, name, opt);
      if (ids.n_used != V) refuse("the inverse operator has " + std::to_string(V) + " sources; its source space uses " + std::to_string(ids.n_used) + " vertices");
      const Channels ch = find_channels(sub, opt.channels);
      std::vector<std::string> names;
      for (const auto& c : iv.parent.channels) names.push_back(c.name);
      const std::string csel = channel_selection(sub, ch, match_channels(ch, names), name + "_channels", opt.acquisition);
      const std::int64_t M = static_cast<std::int64_t>(names.size());
      // the rank axis: the eigenmodes of the inverse, a points manifold with no place
      nxr::io::ManifoldSpec rm;
      rm.name = name + "_eigenmodes";
      rm.type = "points";
      rm.n_vertices = rank;
      rm.acquisition = opt.acquisition;
      rm.topology = {"none", 0, 1, std::nullopt, {}};
      rm.cols.json("producer_json", Json{{"description", "the eigenmodes of an inverse operator: the rank axis its singular values are on"}});
      const std::string rank_id = sub.manifold(std::move(rm));
      Json prod = producer_of(opt, "inverse operator");
      prod["component_frame"] = iv.coord_frame == 4 ? "head" : "mri";
      prod["orientation"] = free_ori ? "free" : "fixed";
      prod["eigen_leads_weighted"] = iv.eigen_leads_weighted;
      if (iv.units) prod["units_code"] = *iv.units;
      prod["parent"] = parent_record(iv.parent, opt);
      // eigen fields: [rank x channels]; the file stores them that way
      if (iv.eigen_fields.nrow != rank || iv.eigen_fields.ncol != M) refuse("the inverse operator's eigen fields are not " + std::to_string(rank) + " x " + std::to_string(M));
      write_matrix_operator(sub, name + "_eigen_fields", "eigen fields", ch.id, rank_id, rank, M, to_f32(iv.eigen_fields.data), "scalar", "scalar", csel, std::nullopt,
                            name + "_eigen_fields \xe2\x80\x94 sensor field to the eigenmodes (the inverse operator's eigen fields)", prod, opt);
      // eigen leads: stored [rank x components]; the operator is [components x rank]
      if (iv.eigen_leads.nrow != rank || iv.eigen_leads.ncol != comps) refuse("the inverse operator's eigen leads are not " + std::to_string(rank) + " x " + std::to_string(comps) + " as stored");
      write_matrix_operator(sub, name + "_eigen_leads", "eigen leads", rank_id, ids.manifold_id, comps, rank, to_f32(transpose(iv.eigen_leads.data, rank, comps)), "scalar",
                            free_ori ? "vector3" : "scalar", std::nullopt, ids.selection_id, name + "_eigen_leads \xe2\x80\x94 the eigenmodes to the source field (the inverse operator's eigen leads)", prod, opt);
      nxr::io::FieldSpec sv;
      sv.name = name + "_singular_values";
      sv.path = sv.name;
      sv.kind = "singular_values";
      sv.manifold_id = rank_id;
      sv.value_type = "scalar";
      std::vector<float> sing = to_f32(iv.sing);
      sv.data = view(sing, {rank});
      sv.array.compress = false;
      sv.acquisition = opt.acquisition;
      sv.cols.set("description", "the singular values of the inverse operator, on its eigenmodes");
      sub.field(std::move(sv));
      std::optional<std::string> noise_id;
      if (iv.noise_cov) {
        const std::string nsel = channel_selection(sub, ch, match_channels(ch, iv.noise_cov->names), name + "_noise_cov_channels", opt.acquisition);
        noise_id = write_covariance(sub, *iv.noise_cov, name + "_noise_cov", nsel, std::nullopt, opt);
      }
      // priors on the sources: diagonal ones, as Fields on the source surface
      Json skipped = Json::array();
      auto prior = [&](const std::optional<fif::Covariance>& c, const std::string& suffix) {
        if (!c) return;
        if (!c->diag || (static_cast<std::int64_t>(c->data.size()) != V && static_cast<std::int64_t>(c->data.size()) != 3 * V)) { skipped.push_back(suffix); return; }
        const bool three = static_cast<std::int64_t>(c->data.size()) == 3 * V;
        std::vector<float> d = to_f32(c->data);
        nxr::io::FieldSpec fs;
        fs.name = name + "_" + suffix;
        fs.path = fs.name;
        fs.kind = suffix;
        fs.manifold_id = ids.manifold_id;
        fs.value_type = three ? "vector3" : "scalar";
        fs.data = three ? view(d, {V, 3}) : view(d, {V});
        fs.array.compress = false;
        fs.acquisition = opt.acquisition;
        fs.cols.set("description", suffix + " of the inverse operator (a prior, not a statistic of data), diagonal, on the used sources");
        sub.field(std::move(fs));
      };
      prior(iv.source_cov, "source_cov");
      prior(iv.depth_prior, "depth_prior");
      prior(iv.orient_prior, "orient_prior");
      head_mri_transition(sub, iv.parent, name, ch.id);
      out["rank"] = rank;
      out["n_sources"] = V;
      out["orientation"] = prod["orientation"];
      out["noise_covariance"] = noise_id.has_value();
      out["priors_not_imported"] = skipped;
    } else if (kind == "eve") {
      const fif::EventFile ev = fif::read_events(f);
      const RecordingRef rec = find_recording(sub, opt.recording);
      refuse_if_events_exist(sub, rec);
      nxr::bst::EventTable t;
      std::map<std::int32_t, std::string> named;
      for (const auto& [n, v] : ev.event_id) named[v] = n;
      for (const auto& e : ev.events) {
        const std::string label = named.count(e[2]) ? named[e[2]] : std::to_string(e[2]);
        auto it = std::find(t.type_names.begin(), t.type_names.end(), label);
        if (it == t.type_names.end()) { t.type_names.push_back(label); t.type_colors.push_back({0.5, 0.5, 0.5}); it = t.type_names.end() - 1; }
        t.type.push_back(static_cast<std::int32_t>(it - t.type_names.begin()));
        t.onset.push_back(static_cast<double>(e[0]) / rec.sfreq);
        t.duration.push_back(0.0);
      }
      t.channel_offsets.assign(t.onset.size() + 1, 0);
      const Json r = nxr::bst::export_events(sub, rec.base, t, rec.line_id, rec.id, std::nullopt, rec.acquisition_id);
      out["recording"] = rec.name;
      out["n_events"] = r["n_events"];
      out["n_types"] = r["n_types"];
    } else if (kind == "annot") {
      const fif::AnnotationSet as = fif::read_annotations_file(f);
      const RecordingRef rec = find_recording(sub, opt.recording);
      refuse_if_events_exist(sub, rec);
      double shift = 0;
      if (as.orig_time) {
        if (!rec.meas_date) refuse("the annotations are relative to a date (orig_time) and the recording records none (it was imported with --anonymize, or has no date)");
        shift = *as.orig_time - *rec.meas_date;
      } else {
        const auto d = sub.db().one("SELECT origin FROM geometry_dimension WHERE geometry_id = (SELECT geometry_id FROM manifold WHERE id = ?)", {nxr::io::V(rec.line_id)});
        shift = d && !nxr::io::is_null(*d, "origin") ? std::get<double>(d->at("origin")) : 0.0;
      }
      const Channels ch = [&] {
        bool any = false;
        for (const auto& a : as.items) any = any || !a.channels.empty();
        return any ? find_channels(sub, opt.channels) : Channels{};
      }();
      std::map<std::string, std::int32_t> at;
      for (std::size_t i = 0; i < ch.names.size(); ++i) at[ch.names[i]] = static_cast<std::int32_t>(i);
      nxr::bst::EventTable t;
      t.channel_offsets.push_back(0);
      for (const auto& a : as.items) {
        const std::string label = a.description.empty() ? "event" : a.description;
        auto it = std::find(t.type_names.begin(), t.type_names.end(), label);
        if (it == t.type_names.end()) { t.type_names.push_back(label); t.type_colors.push_back({0.5, 0.5, 0.5}); it = t.type_names.end() - 1; }
        t.type.push_back(static_cast<std::int32_t>(it - t.type_names.begin()));
        t.onset.push_back(shift + a.onset);
        t.duration.push_back(a.duration);
        for (const auto& c : a.channels) {
          const auto ci = at.find(c);
          if (ci == at.end()) refuse("an annotation names the channel " + c + ", which " + ch.name + " does not have");
          t.channel_index.push_back(ci->second);
        }
        t.channel_offsets.push_back(static_cast<std::int32_t>(t.channel_index.size()));
      }
      const Json r = nxr::bst::export_events(sub, rec.base, t, rec.line_id, rec.id, std::nullopt, rec.acquisition_id);
      out["recording"] = rec.name;
      out["n_events"] = r["n_events"];
      out["n_types"] = r["n_types"];
      out["bad_segments"] = r["bad_segments"];
    } else if (kind == "trans") {
      const fif::CoordTrans t = fif::read_transform(f);
      std::string from_manifold;
      if (!opt.channels.empty() || true) {
        try { from_manifold = find_channels(sub, opt.channels).id; } catch (const nxr::io::Error&) {}
      }
      if (from_manifold.empty()) {
        const auto m = sub.db().one("SELECT id FROM manifold WHERE subject_id = ? ORDER BY created_utc LIMIT 1", {nxr::io::V(sub.id())});
        if (!m) refuse("the package has no manifold to hold the transform's operator (import a recording or a surface first)");
        from_manifold = nxr::io::text(*m, "id");
      }
      out["operator_id"] = import_head_mri_transition(sub, t, name, from_manifold);
    } else {
      refuse("'" + kind + "' is not a FIF object kind (choose from src, cov, fwd, inv, eve, annot, trans)");
    }
  });
  int n_items = 0;
  for (const auto& it : nxr::io::items_without_origin(w.db())) {
    nxr::io::set_origin(w.db(), it.table, it.id, {origin["source"].get<std::string>(), opt.licence, opt.attribution, opt.redistributable});
    ++n_items;
  }
  out["origin"] = origin;
  out["origin_items"] = n_items;
  return out;
}

}  // namespace nxr::convert

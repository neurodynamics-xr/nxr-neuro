// FIF objects against goldens made with MNE-Python (test/golden/gen_fif_objects.py): source spaces (surface, volume, discrete), noise covariances,
// forward solutions, an inverse operator, events, annotation files and a transform; the tag writer; written files equal MNE's byte for byte.
#include <cmath>
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/fif_objects.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static fif::File open_obj(const std::string& name) { return fif::File(std::make_shared<FileSource>(kData + name)); }
static bool near(double a, double b, double rel = 1e-9, double abs_ = 1e-12) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }

static void check_vec(const std::vector<double>& got, const json& want, const char* what, double rel = 1e-9) {
  if (want.is_null()) { CHECK(got.empty()); return; }
  std::vector<double> flat;
  std::function<void(const json&)> walk = [&](const json& j) { if (j.is_array()) for (const auto& x : j) walk(x); else flat.push_back(j.is_null() ? std::nan("") : j.get<double>()); };   // null: NaN
  walk(want);
  if (got.size() != flat.size()) { std::printf("  %s: %zu values, want %zu\n", what, got.size(), flat.size()); CHECK(false); return; }
  bool ok = true;
  for (std::size_t i = 0; i < flat.size(); ++i) ok = ok && ((std::isnan(got[i]) && std::isnan(flat[i])) || near(got[i], flat[i], rel));
  if (!ok) std::printf("  %s differs\n", what);
  CHECK(ok);
}
static void check_ints(const std::vector<std::int32_t>& got, const json& want, const char* what) {
  std::vector<std::int64_t> flat;
  if (!want.is_null()) {
    std::function<void(const json&)> walk = [&](const json& j) { if (j.is_array()) for (const auto& x : j) walk(x); else flat.push_back(j.get<std::int64_t>()); };
    walk(want);
  }
  bool ok = got.size() == flat.size();
  for (std::size_t i = 0; ok && i < flat.size(); ++i) ok = got[i] == flat[i];
  if (!ok) std::printf("  %s differs (%zu vs %zu values)\n", what, got.size(), flat.size());
  CHECK(ok);
}
static void check_trans(const std::optional<fif::CoordTrans>& got, const json& want, const char* what) {
  if (want.is_null()) { CHECK(!got); return; }
  CHECK(got.has_value());
  if (!got) return;
  CHECK_EQ(got->from, want["from"].get<std::int32_t>());
  CHECK_EQ(got->to, want["to"].get<std::int32_t>());
  bool ok = true;
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) ok = ok && near(got->trans[r * 4 + c], want["trans"][r][c].get<double>(), 1e-6, 1e-9);
  if (!ok) std::printf("  transform %s differs\n", what);
  CHECK(ok);
}
static void check_sparse(const std::optional<fif::SparseMatrix>& got, const json& want, const char* what) {
  if (want.is_null()) { CHECK(!got); return; }
  CHECK(got.has_value());
  if (!got) return;
  CHECK_EQ(got->nrow, want["shape"][0].get<std::int32_t>());
  CHECK_EQ(got->ncol, want["shape"][1].get<std::int32_t>());
  check_vec(got->data, want["data"], what);
  check_ints(got->indices, want["indices"], what);
  check_ints(got->indptr, want["indptr"], what);
}

// what a writer produced is what MNE wrote: the bytes after the file id's own bytes (a file id is a clock reading)
static void check_same_bytes(const std::string& name, const std::string& rewritten) {
  const std::string orig = slurp(kData + name);
  const bool same = orig == rewritten;
  if (!same) {
    std::size_t i = 0;
    while (i < orig.size() && i < rewritten.size() && orig[i] == rewritten[i]) ++i;
    { std::ofstream o(std::string(NXR_FORMATS_WORK_DIR) + "/" + name + ".rewritten", std::ios::binary); o.write(rewritten.data(), static_cast<std::streamsize>(rewritten.size())); }
    std::printf("  %s: the rewritten file differs from MNE's (%zu vs %zu bytes, first difference at %zu)\n", name.c_str(), rewritten.size(), orig.size(), i);
  }
  CHECK(same);
}

static const json& golden() {
  static json g = [] { std::ifstream f(kGold + "fif_objects_golden.json"); return json::parse(f); }();
  return g;
}

static void check_spaces(const fif::SourceSpaces& s, const json& g, const char* name) {
  std::printf("  case %s\n", name);
  CHECK_EQ(s.spaces.size(), g["spaces"].size());
  for (std::size_t i = 0; i < s.spaces.size() && i < g["spaces"].size(); ++i) {
    const fif::SourceSpace& a = s.spaces[i];
    const json& w = g["spaces"][i];
    CHECK_EQ(a.type, w["type"].get<std::string>());
    CHECK_EQ(a.coord_frame, w["coord_frame"].get<std::int32_t>());
    CHECK_EQ(a.np, w["np"].get<std::int32_t>());
    CHECK_EQ(a.ntri, w["ntri"].get<std::int32_t>());
    CHECK_EQ(a.nuse, w["nuse"].get<std::int32_t>());
    CHECK_EQ(a.nuse_tri, w["nuse_tri"].get<std::int32_t>());
    if (a.type == "surf") CHECK_EQ(a.id, w["id"].get<std::int32_t>());
    check_vec(a.rr, w["rr"], "rr", 1e-7);
    check_vec(a.nn, w["nn"], "nn", 1e-7);
    check_ints(a.tris, w["tris"], "tris");
    check_ints(a.inuse, w["inuse"], "inuse");
    check_ints(a.vertno, w["vertno"], "vertno");
    check_ints(a.use_tris, w["use_tris"], "use_tris");
    check_ints(a.nearest, w["nearest"], "nearest");
    check_vec(a.nearest_dist, w["nearest_dist"], "nearest_dist", 1e-7);
    check_sparse(a.dist, w["dist"], "dist");
    CHECK_EQ(a.dist_limit.has_value(), !w["dist_limit"].is_null());
    if (a.dist_limit && w["dist_limit"].is_string()) CHECK(std::isinf(*a.dist_limit));
    else if (a.dist_limit && !w["dist_limit"].is_null()) CHECK(near(*a.dist_limit, w["dist_limit"].get<double>(), 1e-6));
    CHECK_EQ(a.subject_his_id.has_value(), !w["subject_his_id"].is_null());
    if (a.subject_his_id && !w["subject_his_id"].is_null()) CHECK_EQ(*a.subject_his_id, w["subject_his_id"].get<std::string>());
    check_ints(a.shape, w["shape"], "shape");
    check_trans(a.src_mri_t, w["src_mri_t"], "src_mri_t");
    check_trans(a.vox_mri_t, w["vox_mri_t"], "vox_mri_t");
    check_trans(a.mri_ras_t, w["mri_ras_t"], "mri_ras_t");
    check_sparse(a.interpolator, w["interpolator"], "interpolator");
    CHECK_EQ(a.mri_file.has_value(), !w["mri_file"].is_null());
    CHECK_EQ(a.mri_width.has_value(), !w["mri_width"].is_null());
    if (a.mri_width && !w["mri_width"].is_null()) CHECK_EQ(*a.mri_width, w["mri_width"].get<std::int32_t>());
    CHECK_EQ(a.seg_name.has_value(), !w["seg_name"].is_null());
    CHECK_EQ(a.neighbor_vert.size(), w["neighbor_vert"].is_null() ? std::size_t(0) : w["neighbor_vert"].size());
    for (std::size_t k = 0; k < a.neighbor_vert.size(); ++k) check_ints(a.neighbor_vert[k], w["neighbor_vert"][k], "neighbors");
  }
}

static void check_cov(const fif::Covariance& c, const json& g, const char* name) {
  std::printf("  case %s\n", name);
  CHECK_EQ(c.kind, g["kind"].get<std::int32_t>());
  CHECK_EQ(c.diag, g["diag"].get<bool>());
  CHECK_EQ(c.dim, g["dim"].get<std::int32_t>());
  if (g.contains("nfree")) CHECK_EQ(c.nfree, g["nfree"].get<std::int32_t>());
  check_vec(c.data, g["data"], "cov data", 1e-12);
  if (!g.contains("names")) return;
  CHECK_EQ(c.names.size(), g["names"].size());
  for (std::size_t i = 0; i < c.names.size() && i < g["names"].size(); ++i) CHECK_EQ(c.names[i], g["names"][i].get<std::string>());
  CHECK_EQ(c.bads.size(), g["bads"].size());
  CHECK_EQ(c.method.has_value(), !g["method"].is_null());
  CHECK_EQ(c.score.has_value(), !g["score"].is_null());
  if (c.score && !g["score"].is_null()) CHECK(near(*c.score, g["score"].get<double>()));
  check_vec(c.eig, g["eig"], "eig");
  check_vec(c.eigvec, g["eigvec"], "eigvec", 1e-6);
  CHECK_EQ(c.projs.size(), g["projs"].size());
  for (std::size_t i = 0; i < c.projs.size() && i < g["projs"].size(); ++i) {
    const auto& p = c.projs[i];
    const json& w = g["projs"][i];
    CHECK_EQ(p.desc, w["desc"].get<std::string>());
    CHECK_EQ(p.kind, w["kind"].get<std::int32_t>());
    CHECK_EQ(p.active, w["active"].get<bool>());
    CHECK_EQ(p.nvec, w["nvec"].get<std::int32_t>());
    check_vec(p.vectors, w["vectors"], "projector vectors", 1e-6);
    CHECK_EQ(p.channels.size(), w["channels"].size());
  }
}

static std::string serialise(const std::function<void(MemorySink&)>& f) { MemorySink s; f(s); return s.str(); }

int main() {
  const json& G = golden();
  // ---- tag writing: every writer, read back through the tag layer ----
  {
    MemorySink sink;
    fif::TagWriter w(sink);
    w.start_file(fif::make_id(1700000000, 5));
    w.start_block(fif::block::kMne);
    w.write_int(3000, std::vector<std::int32_t>{1, -2, 3});
    w.write_float(3001, 0.5);
    w.write_double(3002, std::vector<double>{1.25, -2.5});
    w.write_string(3003, "caf\xC3\xA9 \xE2\x82\xAC");
    w.write_name_list(3004, {"a:b", "c"});
    w.write_float_matrix(3005, {1, 2, 3, 4, 5, 6}, 2, 3);
    w.write_double_matrix(3006, {1, 2, 3, 4}, 2, 2);
    w.write_int_matrix(3007, {1, 2, 3, 4, 5, 6}, 3, 2);
    w.write_complex_matrix(3008, {{1, 2}, {3, 4}}, 1, 2, true);
    fif::SparseMatrix sp;
    sp.nrow = 2; sp.ncol = 3; sp.data = {1.5, 2.5}; sp.indices = {0, 2}; sp.indptr = {0, 1, 2};
    w.write_sparse(3009, sp);
    fif::CoordTrans t;
    t.from = 5; t.to = 4;
    t.trans = {0, -1, 0, 0.01, 1, 0, 0, 0.02, 0, 0, 1, 0.03, 0, 0, 0, 1};
    w.write_coord_trans(t);
    w.end_block(fif::block::kMne);
    w.end_file();
    fif::File f(std::make_shared<MemorySource>(sink.data()));
    CHECK_EQ(f.file_id().secs, 1700000000);
    const fif::Block& m = *fif::find_blocks(f.tree(), fif::block::kMne)[0];
    CHECK_EQ(f.value(*fif::find_tag(m, 3000)).ints.size(), std::size_t(3));
    CHECK_EQ(f.value(*fif::find_tag(m, 3000)).ints[1], std::int64_t(-2));
    CHECK(near(f.value(*fif::find_tag(m, 3001)).reals[0], 0.5));
    CHECK(near(f.value(*fif::find_tag(m, 3002)).reals[1], -2.5));
    CHECK_EQ(f.value(*fif::find_tag(m, 3003)).text, std::string("caf\xC3\xA9 \xE2\x82\xAC"));   // beyond Latin-1 survives as a character reference
    CHECK_EQ(f.value(*fif::find_tag(m, 3004)).text, std::string("a{COLON}b:c"));
    const auto m5 = f.value(*fif::find_tag(m, 3005));
    CHECK_EQ(m5.dims[0], 2);
    CHECK_EQ(m5.dims[1], 3);
    CHECK(near(m5.reals[5], 6.0));
    CHECK_EQ(f.value(*fif::find_tag(m, 3007)).ints[5], std::int64_t(6));
    CHECK(near(f.value(*fif::find_tag(m, 3008)).complexes[1].imag(), 4.0));
    const fif::SparseMatrix back = fif::read_sparse(f, *fif::find_tag(m, 3009));
    CHECK(back.row_compressed);
    CHECK_EQ(back.indices[1], 2);
    CHECK(near(back.data[1], 2.5));
    const fif::CoordTrans tb = fif::read_transform(f);
    CHECK(near(tb.trans[3], 0.01, 1e-6));
    // a generic tag survives a round trip
    MemorySink s2;
    fif::TagWriter w2(s2);
    w2.start_file(fif::make_id(1, 2));
    for (const auto& th : m.tags) if (th.kind >= 3000 && th.kind <= 3008) w2.write_generic(fif::GenericTag{th.kind, th.type, f.value(th)});
    w2.end_file();
    fif::File f2(std::make_shared<MemorySource>(s2.data()));
    CHECK_EQ(f2.directory().size(), std::size_t(3 + 9 + 1));
  }

  // ---- transform ----
  {
    auto f = open_obj("fif_obj-trans.fif");
    check_trans(fif::read_transform(f), G["trans"]["trans"], "trans");
    check_same_bytes("fif_obj-trans.fif", serialise([&](MemorySink& s) { fif::write_transform_file(s, fif::read_transform(f), f.file_id()); }));
  }
  // ---- source spaces ----
  for (const char* key : {"src_surf", "src_vol", "src_discrete"}) {
    const std::string file = G[key]["file"].get<std::string>();
    auto f = open_obj(file);
    const fif::SourceSpaces s = fif::read_source_spaces(f);
    check_spaces(s, G[key], key);
    check_same_bytes(file, serialise([&](MemorySink& o) { fif::write_source_spaces_file(o, s, f.file_id()); }));
  }
  // ---- covariances ----
  for (const char* key : {"cov_full", "cov_diag"}) {
    const std::string file = G[key]["file"].get<std::string>();
    auto f = open_obj(file);
    const fif::Covariance c = fif::read_covariance(f);
    check_cov(c, G[key], key);
    check_same_bytes(file, serialise([&](MemorySink& o) { fif::write_covariance_file(o, c, f.file_id()); }));
  }
  // ---- forward solutions: the matrix is as stored (sources x channels); MNE's is its transpose, MEG rows then EEG rows ----
  for (const char* key : {"fwd_free", "fwd_fixed"}) {
    const json& g = G[key];
    auto f = open_obj(g["file"].get<std::string>());
    const fif::ForwardFile fw = fif::read_forward(f);
    std::printf("  case %s\n", key);
    CHECK_EQ(fw.solutions[0].source_ori, g["source_ori"].get<std::int32_t>());
    CHECK_EQ(fw.solutions[0].coord_frame, g["coord_frame"].get<std::int32_t>());
    CHECK_EQ(fw.solutions[0].nsource, g["nsource"].get<std::int32_t>());
    std::int32_t nchan = 0;
    for (const auto& s : fw.solutions) nchan += s.nchan;
    CHECK_EQ(nchan, g["nchan"].get<std::int32_t>());
    const std::size_t ncomp = static_cast<std::size_t>(fw.solutions[0].sol.nrow);
    CHECK_EQ(ncomp, g["sol"][0].size());
    std::vector<double> mne_order;   // MNE's (channels x components), assembled from the stored blocks
    for (const auto& s : fw.solutions)
      for (std::int32_t c = 0; c < s.sol.ncol; ++c)
        for (std::int32_t r = 0; r < s.sol.nrow; ++r) mne_order.push_back(s.sol.data[static_cast<std::size_t>(r) * s.sol.ncol + c]);
    check_vec(mne_order, g["sol"], "sol", 1e-6);
    CHECK_EQ(fw.parent.channels.size(), g["ch_names"].size());
    for (std::size_t i = 0; i < fw.parent.channels.size() && i < g["ch_names"].size(); ++i) {
      CHECK_EQ(fw.parent.channels[i].name, g["ch_names"][i].get<std::string>());
      CHECK_EQ(fw.parent.channels[i].kind, g["ch_kinds"][i].get<std::int32_t>());
      check_vec(std::vector<double>(fw.parent.channels[i].loc.begin(), fw.parent.channels[i].loc.end()), g["ch_loc"][i], "channel loc", 1e-6);
    }
    CHECK_EQ(fw.parent.bads.size(), g["bads"].size());
    check_trans(fw.parent.mri_head_t, g["mri_head_t"], "mri_head_t");
    check_trans(fw.parent.dev_head_t, g["dev_head_t"], "dev_head_t");
    CHECK_EQ(fw.parent.mri_file.has_value(), !g["mri_file"].is_null());
    std::int32_t nuse = 0;
    for (const auto& sp : fw.src.spaces) nuse += sp.nuse;
    CHECK_EQ(nuse, g["nuse"].get<std::int32_t>());
  }
  // ---- the inverse operator ----
  {
    const json& g = G["inv"];
    auto f = open_obj(g["file"].get<std::string>());
    const fif::InverseOperator iv = fif::read_inverse(f);
    std::printf("  case inv\n");
    CHECK_EQ(iv.methods, g["methods"].get<std::int32_t>());
    CHECK_EQ(iv.source_ori, g["source_ori"].get<std::int32_t>());
    CHECK_EQ(iv.nsource, g["nsource"].get<std::int32_t>());
    CHECK_EQ(iv.coord_frame, g["coord_frame"].get<std::int32_t>());
    CHECK_EQ(static_cast<std::size_t>(iv.sing.size()), g["sing"].size());
    check_vec(iv.source_nn, g["source_nn"], "source_nn", 1e-6);
    check_vec(iv.sing, g["sing"], "sing", 1e-6);
    CHECK_EQ(iv.eigen_leads_weighted, g["eigen_leads_weighted"].get<bool>());
    std::vector<double> leads_t;   // MNE transposes the stored eigenleads
    for (std::int32_t c = 0; c < iv.eigen_leads.ncol; ++c)
      for (std::int32_t r = 0; r < iv.eigen_leads.nrow; ++r) leads_t.push_back(iv.eigen_leads.data[static_cast<std::size_t>(r) * iv.eigen_leads.ncol + c]);
    check_vec(leads_t, g["eigen_leads"], "eigen_leads", 1e-6);
    check_vec(iv.eigen_fields.data, g["eigen_fields"], "eigen_fields", 1e-6);
    CHECK(iv.noise_cov.has_value());
    if (iv.noise_cov) check_cov(*iv.noise_cov, g["noise_cov"], "inv noise_cov");
    CHECK(iv.source_cov.has_value());
    if (iv.source_cov) check_cov(*iv.source_cov, g["source_cov"], "inv source_cov");
    CHECK_EQ(iv.depth_prior.has_value(), !g["depth_prior"].is_null());
    if (iv.depth_prior && !g["depth_prior"].is_null()) check_cov(*iv.depth_prior, g["depth_prior"], "inv depth_prior");
    CHECK_EQ(iv.orient_prior.has_value(), !g["orient_prior"].is_null());
    if (iv.orient_prior && !g["orient_prior"].is_null()) check_cov(*iv.orient_prior, g["orient_prior"], "inv orient_prior");
    CHECK_EQ(iv.projs.size(), g["n_projs"].get<std::size_t>());
    check_trans(iv.parent.mri_head_t, g["mri_head_t"], "inv mri_head_t");
    CHECK_EQ(iv.parent.channels.size(), g["ch_names"].size());
  }
  // ---- events ----
  for (const char* key : {"events", "events_named"}) {
    const json& g = G[key];
    auto f = open_obj(g["file"].get<std::string>());
    const fif::EventFile e = fif::read_events(f);
    std::printf("  case %s\n", key);
    CHECK_EQ(e.events.size(), g["events"].size());
    for (std::size_t i = 0; i < e.events.size() && i < g["events"].size(); ++i)
      for (int k = 0; k < 3; ++k) CHECK_EQ(e.events[i][k], g["events"][i][k].get<std::int32_t>());
    if (g["event_id"].is_null()) {
      CHECK(e.event_id.empty());
    } else {
      CHECK_EQ(e.event_id.size(), g["event_id"].size());
      for (const auto& [name, v] : e.event_id) CHECK_EQ(v, g["event_id"][name].get<std::int32_t>());
    }
    check_same_bytes(g["file"].get<std::string>(), serialise([&](MemorySink& o) { fif::write_events_file(o, e, f.file_id()); }));
  }
  // ---- annotation files ----
  for (const char* key : {"annot", "annot_rel"}) {
    const json& g = G[key];
    auto f = open_obj(g["file"].get<std::string>());
    const fif::AnnotationSet a = fif::read_annotations_file(f);
    std::printf("  case %s\n", key);
    CHECK_EQ(a.items.size(), g["onset"].size());
    CHECK_EQ(a.orig_time.has_value(), !g["orig_time"].is_null());
    if (a.orig_time && !g["orig_time"].is_null()) CHECK(near(*a.orig_time, g["orig_time"].get<double>(), 1e-9, 1e-6));
    for (std::size_t i = 0; i < a.items.size() && i < g["onset"].size(); ++i) {
      CHECK(near(a.items[i].onset, g["onset"][i].get<double>(), 1e-6));
      CHECK(near(a.items[i].duration, g["duration"][i].get<double>(), 1e-6, 1e-6));
      CHECK_EQ(a.items[i].description, g["description"][i].get<std::string>());
      CHECK_EQ(a.items[i].channels.size(), g["ch_names"][i].size());
      CHECK_EQ(a.items[i].extras_json.empty() || a.items[i].extras_json == "{}", g["extras"][i].is_null());   // MNE stores no extras as {}
      if (!g["extras"][i].is_null()) CHECK(json::parse(a.items[i].extras_json) == g["extras"][i]);
    }
    check_same_bytes(g["file"].get<std::string>(), serialise([&](MemorySink& o) { fif::write_annotations_file(o, a, f.file_id()); }));
  }
  // ---- refusals ----
  auto refusal = [&](const std::function<void()>& fn, const char* code_, const char* needle) {
    try {
      fn();
      std::printf("  expected a refusal containing '%s'\n", needle);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  message '%s' lacks '%s'\n", e.what(), needle);
      CHECK(found);
    }
  };
  refusal([&] { auto f = open_obj("fif_obj_full-cov.fif"); fif::read_source_spaces(f); }, code::refused, "no source spaces");
  refusal([&] { auto f = open_obj("fif_obj_surf-src.fif"); fif::read_covariance(f); }, code::refused, "no covariance");
  refusal([&] { auto f = open_obj("fif_obj_full-cov.fif"); fif::read_covariance(f, fif::mne::kSourceCov); }, code::refused, "kind 2");
  refusal([&] { auto f = open_obj("fif_obj_surf-src.fif"); fif::read_forward(f); }, code::refused, "no forward");
  refusal([&] { auto f = open_obj("fif_obj_free-fwd.fif"); fif::read_inverse(f); }, code::refused, "no inverse");
  refusal([&] { auto f = open_obj("fif_obj_surf-src.fif"); fif::read_events(f); }, code::refused, "event data");
  refusal([&] { auto f = open_obj("fif_obj_surf-src.fif"); fif::read_annotations_file(f); }, code::refused, "no annotations");
  refusal([&] { MemorySink s; fif::TagWriter w(s); w.write_float_matrix(1, {1, 2, 3}, 2, 2); }, code::format, "given 3 values");
  refusal([&] { const std::string b = slurp(kData + "fif_obj_surf-src.fif"); fif::File f(std::make_shared<MemorySource>(b.substr(0, b.size() / 2))); fif::read_source_spaces(f); }, code::format, "");
  return nxrtest::finish("formats_fif_objects");
}

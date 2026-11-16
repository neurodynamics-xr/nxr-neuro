// nxr-pipelines — the measurement pyramid build (pyramid.h).
#include "nxr/pipelines/pyramid.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

#include "nxr/bst/frequency.h"
#include "nxr/io/finder.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/ladder.h"
#include "nxr/parallel.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/group.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/reading.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/pyramid.h"
#include "nxr/tower.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace lad = nxr::ladder;
namespace py = nxr::pyramid;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
std::string g(double v) { std::ostringstream s; s << v; return s.str(); }

template <class T> std::vector<T> read_vec(const nxr::io::Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

// the stored bank of `atlas bands` (nxr D158), or made: a spans Selection on the dataset's Frequency manifold, code = the band's index
std::string bands_selection_impl(SubjectWriter& sub, const py::Bands& bands, bool check_only = false, DatasetWriter* dset = nullptr) {   // check_only: refuse what would be refused, write nothing (the run checks BEFORE it computes)
  if (const auto have = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = 'atlas bands' AND type = 'spans' AND status = 'complete'", {V(sub.id())})) {
    const auto rows = sub.db().all("SELECT start, stop, code FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*have, "id"))});
    bool same = rows.size() == bands.size();
    for (std::size_t k = 0; same && k < rows.size(); ++k)
      same = std::abs(std::stod(text(rows[k], "start")) - bands[k].first) < 1e-9 && std::abs(std::stod(text(rows[k], "stop")) - bands[k].second) < 1e-9 && integer(rows[k], "code") == static_cast<std::int64_t>(k);
    if (!same) invalid(sub.name() + ": the stored `atlas bands` are another bank than the one asked for; a band's code is its index in the bank, so the bank of a subject is one (name the stored bands, or rebuild with --replace after removing them)");
    return text(*have, "id");
  }
  auto f = nxr::bst::find_frequency(sub.db());
  const bool is_template = text(sub.row(), "kind") == "template";
  // the Frequency manifold (with its standard bands) lives in the dataset's default subject; made there when absent, as `atlas default-subject` (nxr) makes it: additive, nothing existing changes
  const std::string no_default = sub.name() + ": the bands are spans on the Frequency manifold of the dataset's default subject, which has none and the dataset has no default subject to make it in (create the default subject first)";
  if (check_only && (f || is_template)) return "";
  if (check_only) { if (!dset || !dset->template_()) throw Error(code::no_such_row, no_default); return ""; }
  if (!f && is_template) { nxr::bst::ensure_frequency(sub); f = nxr::bst::find_frequency(sub.db()); }
  if (!f && dset) {
    if (const auto def = dset->template_()) {
      SubjectWriter dsub = SubjectWriter::open(*dset, text(*def, "name"));
      dsub.compose([&] { nxr::bst::ensure_frequency(dsub); });
      f = nxr::bst::find_frequency(sub.db());
    }
  }
  if (!f) throw Error(code::no_such_row, no_default);
  std::string id;
  sub.compose([&] {
    DictionarySpec ds;
    ds.name = "atlas bands"; ds.scope = "dataset";
    ds.description = "the measurement pyramid's band bank: code = the band's index";
    ds.entries = std::vector<DictionaryEntry>();
    for (std::size_t k = 0; k < bands.size(); ++k) {
      DictionaryEntry e;
      e.code = static_cast<std::int64_t>(k); e.ordinal = static_cast<std::int64_t>(k);
      e.name = g(bands[k].first) + "\xe2\x80\x93" + g(bands[k].second) + " Hz";
      e.cols.set("measurement", std::sqrt(bands[k].first * bands[k].second)).set("measurement_unit", "Hz").json("attributes_json", Json{{"low_hz", bands[k].first}, {"high_hz", bands[k].second}});
      ds.entries->push_back(e);
    }
    const DictionaryResult dr = sub.dictionary(ds);
    SelectionSpec ss;
    ss.name = "atlas bands"; ss.type = "spans"; ss.manifold_id = text(f->manifold, "id");
    std::vector<SpanSpec> spans;
    for (std::size_t k = 0; k < bands.size(); ++k) spans.push_back({bands[k].first, bands[k].second, static_cast<std::int64_t>(k)});
    ss.spans = spans;
    ss.cols.set("cell", "1").set("dictionary_id", dr.id).set("unit", "Hz").set("description", "the pyramid's frequency bands: a span [lo, hi) of the Frequency manifold each, code = the band's index");
    id = sub.selection(std::move(ss));
  });
  return id;
}
}  // namespace

std::string pyramid_bands_selection(SubjectWriter& sub, const std::vector<std::pair<double, double>>& bands) { return bands_selection_impl(sub, bands); }

PyramidLeaves pyramid_leaves(DatasetWriter& ds, SubjectWriter& sub, const SurfaceInput& S, std::optional<int> depth, const std::string& who, const std::function<void(const Json&)>& say, const std::string& tree) {
  const std::string surface_name = S.name;
  if (tree != "subject" && tree != "group") invalid(who + ": --tree is subject or group, not " + tree);
  if (tree == "group" && text(sub.row(), "kind") != "template") return group_leaves(ds, sub, S, depth, who, say);
  const std::int64_t N = S.n_vertices();
  auto ladder_of = [&]() { return sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND name = ? AND type = 'ladder' AND status = 'complete'", {V(sub.id()), V(S.id), V(surface_name + " tiles")}); };
  if (!ladder_of()) { TilesComputeOptions to; to.surface = surface_name; if (say) say(Json{{"stage", "tiles"}, {"surface", surface_name}, {"made", true}}); tiles_compute(ds, sub, to); }
  const auto ladder = ladder_of();
  const Json lp = Json::parse(text(*ladder, "params_json"), nullptr, false);
  if (!lp.is_object() || !lp.contains("depth") || !lp.contains("finest")) invalid(surface_name + " tiles: its record has no depth (a ladder made by another producer)");
  PyramidLeaves out;
  const int ladder_depth = lp["depth"].get<int>();
  out.finest = lp["finest"].get<int>();
  out.depth = depth ? *depth : ladder_depth;
  if (out.depth < 0 || out.depth > ladder_depth) invalid(who + ": --depth " + std::to_string(out.depth) + " is outside the surface's ladder (levels 0 \xe2\x80\xa6 " + std::to_string(ladder_depth) + ")");
  out.name = surface_name + " tiles L" + std::to_string(out.depth);
  out.part = sub.db().one("SELECT * FROM selection WHERE member_of_id = ? AND name = ? AND status = 'complete'", {V(text(*ladder, "id")), V(out.name)});
  if (!out.part) invalid(sub.name() + ": the ladder of " + surface_name + " has no level " + std::to_string(out.depth));
  out.record = Json::parse(text(*out.part, "params_json"), nullptr, false);
  if (!out.record.is_object() || !out.record.contains("node") || !out.record.contains("areas")) invalid(out.name + ": its record lacks the node of its arrays (a ladder made by another producer)");
  out.tree = lp.value("tree", std::string("subject"));
  out.node = out.record["node"].get<std::string>();
  out.leaves = integer(*out.part, "n_members");
  out.codes = read_vec<std::int32_t>(sub.store(), text(*out.part, "array_path"));
  out.leaf_area = read_vec<double>(sub.store(), out.record["areas"].get<std::string>());
  if (static_cast<std::int64_t>(out.codes.size()) != N || static_cast<std::int64_t>(out.leaf_area.size()) != out.leaves) invalid(out.name + ": its codes or areas do not fit the surface");
  return out;
}

void pyramid_leaves_record(SubjectWriter& sub, PyramidLeaves& leaves, const Json& gauge) {
  leaves.record["atlas"] = Json{{"tree", leaves.tree}, {"level", leaves.depth}, {"finest", leaves.finest}, {"gauge", gauge}, {"areas", leaves.record["areas"]}, {"node", leaves.node}};
  sub.db().execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(leaves.record)), V(text(*leaves.part, "id"))});
}

PyramidBuildResult pyramid_build(DatasetWriter& ds, SubjectWriter& sub, const PyramidBuildOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording.empty()) invalid("pyramid: --recording names the recording whose band power is measured");
  py::Shape shape;
  try { shape = py::shapeOf(opt.shape); } catch (const std::invalid_argument& e) { invalid(std::string("pyramid: ") + e.what()); }
  const py::Bands bands = opt.bands.empty() ? py::octaveBands() : opt.bands;
  if (opt.halo_seconds < 0) invalid("pyramid: --halo-seconds is 0 or more");
  if (opt.threads < 0) invalid("pyramid: --threads is at least 1 (0: the machine's performance cores)");
  const int threads = nxr::parallel::resolveThreads(opt.threads);
  if (opt.block_samples < 0) invalid("pyramid: --block-samples is at least 1 (0: the default of the kernel's orientation)");
  if (opt.frame_level > 0) invalid("pyramid: --frame-level is 0 (the day) or finer (negative)");
  try { py::bandGains(Eigen::VectorXd::Ones(1), bands, shape); } catch (const std::invalid_argument& e) { invalid(std::string("pyramid: ") + e.what()); }

  // ── the recording (prepare: what it is, its time line, its channels) ──
  const auto recrow = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording' AND status = 'complete'", {V(sub.id()), V(opt.recording)});
  if (!recrow) missing(sub.name() + ": no recording '" + opt.recording + "'");
  const finder::Value rv = finder::value_of(sub.db(), "field", *recrow);
  if (!rv.time || !(rv.rate > 0)) invalid(opt.recording + ": it has no time line with a rate");
  const std::string time_id = *rv.time, rec_id = text(*recrow, "id");
  const auto dims = sub.db().all("SELECT d.manifold_id AS mid FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? ORDER BY d.ordinal", {V(text(*recrow, "manifold_id"))});
  if (dims.size() != 2) invalid(opt.recording + ": a recording lies on channels x time");
  const std::string channels_id = text(dims[0], "mid");
  const double sfreq = rv.rate, t0 = rv.origin;
  const std::int64_t n_samples = integer(*sub.db().read("manifold", time_id), "n_vertices");

  // ── the kernel (constrained orientation) and its channels, in its order ──
  Row kern;
  if (opt.kernel) {
    const auto r = sub.db().one("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.kernel)});
    if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'");
    kern = *r;
  } else {
    const auto rs = sub.db().all("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND status = 'complete' AND path IS NOT NULL", {V(sub.id())});
    if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels; name one with --kernel");
    kern = rs[0];
  }
  const std::string kname = text(kern, "name"), kernel_id = text(kern, "id"), surface_id = text(kern, "to_manifold_id");
  if (text(kern, "from_manifold_id") != channels_id) invalid(kname + " is on other channels than " + opt.recording);
  const bool is_free = opt_text(kern, "to_value_type").value_or("scalar") == "vector3";          // three rows a source (x y z)
  const std::int64_t block_samples = opt.block_samples > 0 ? opt.block_samples : (is_free ? 128 : 512);
  const auto ksel = sub.db().read("selection", text(kern, "from_selection_id"));
  if (!ksel || is_null(*ksel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_vec<std::int32_t>(sub.store(), text(*ksel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(chans.size());
  std::vector<std::int64_t> kshape;
  const std::vector<float> kf = read_vec<float>(sub.store(), text(kern, "path"), &kshape);
  if (kshape.size() != 2 || kshape[1] != M || (is_free && kshape[0] % 3 != 0)) invalid(kname + ": its array is not [" + (is_free ? "3 x sources rows x " : "sources x ") + std::to_string(M) + " channels]");
  const std::int64_t N = is_free ? kshape[0] / 3 : kshape[0];
  const Eigen::MatrixXd K = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(kf.data(), kshape[0], M).cast<double>();
  const auto surf = sub.db().read("manifold", surface_id);
  if (!surf || integer(*surf, "n_vertices") != N) invalid(kname + ": " + std::to_string(N) + " sources, its target surface has another vertex count");
  const std::string surface_name = text(*surf, "name");

  // ── the leaves: the tree partition at depth D of the surface's ladder ──
  const SurfaceInput S = read_surface(sub, surface_name, true);
  const PyramidLeaves PL = pyramid_leaves(ds, sub, S, opt.depth, "pyramid", say, opt.tree);
  const std::optional<Row>& part = PL.part;
  Json prec = PL.record;
  const int D = PL.depth, finest = PL.finest;
  const std::string& part_name = PL.name, &tree_kind = PL.tree, &node_of_part = PL.node;
  const std::int64_t L = PL.leaves;
  const std::vector<std::int32_t>& codes = PL.codes;
  const std::vector<double>& leaf_area = PL.leaf_area;
  (void)leaf_area;
  lad::Mesh mesh; mesh.n = static_cast<int>(N); mesh.positions = S.vertices; mesh.faces = S.faces;
  const std::vector<double> area = lad::vertexAreas(mesh);
  const auto gauge_row = sub.db().one("SELECT id FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(surface_name + "_canonical_frames")});
  const Json gauge = gauge_row ? Json(text(*gauge_row, "id")) : Json(nullptr);
  // free orientation: the kernel in the canonical frames (the gauge); a vertex with no frame is out of the nine tensor sums and in the total
  py::FrameKernel fk;
  std::vector<double> w_tensor, w_total = area;
  Json frames_hash = Json(nullptr);
  if (is_free) {
    const StoredFrames sf = read_stored_frames(sub, surface_name, N);
    if (!sf.canonical) invalid(surface_name + " has no canonical frames: the orientation tensor is measured in them (`nxr-neuro frames compute` first)");
    std::vector<char> sing(static_cast<std::size_t>(N), 0);
    w_tensor = area;
    for (std::int64_t v = 0; v < N; ++v) {
      bool bad_row = false;
      for (int k = 0; k < 3; ++k) bad_row = bad_row || !std::isfinite(sf.e1(v, k)) || !std::isfinite(sf.e2(v, k));
      sing[static_cast<std::size_t>(v)] = bad_row;
      if (bad_row) w_tensor[static_cast<std::size_t>(v)] = 0.0;
    }
    try { fk = py::frameKernel(K, sf.e1, sf.e2, sf.n, sing, threads); } catch (const std::invalid_argument& e) { invalid(kname + ": " + e.what()); }
    frames_hash = sha256_hex(std::string(reinterpret_cast<const char*>(sf.e1.data()), static_cast<std::size_t>(sf.e1.size()) * sizeof(double)) + std::string(reinterpret_cast<const char*>(sf.e2.data()), static_cast<std::size_t>(sf.e2.size()) * sizeof(double))).substr(0, 16);
  }

  // ── the frames: the tower tiling of the recording ──
  nxr::tower::FrameTiling ft;
  try { ft = nxr::tower::makeFrameTiling(sfreq, n_samples, opt.frame_level, opt.start_s); } catch (const std::invalid_argument& e) { invalid(std::string("pyramid: ") + e.what()); }
  const double frame_s = ft.frame_s();
  const std::int64_t n_frames = ft.count;
  const int B = joint_bits(n_frames);
  if ((L << B) >= (std::int64_t(1) << 31)) invalid("pyramid: " + std::to_string(L) + " leaves x 2^" + std::to_string(B) + " frames do not fit a 32-bit joint code; use a smaller --depth");

  // ── what is loaded: one chunk by default ──
  const nxr::io::Span span = span_of(*recrow, n_samples, sfreq, t0, opt.chunks, opt.start, opt.stop);
  const std::int64_t s0 = span.s0, s1 = span.s1;
  if (s1 <= s0) invalid(opt.recording + ": the span holds no sample");
  const std::int64_t frame0 = ft.frameOfSample(s0), frames_span = ft.frameOfSample(s1 - 1) - frame0 + 1;
  const std::int64_t halo = static_cast<std::int64_t>(std::ceil(opt.halo_seconds * sfreq));
  std::optional<std::vector<std::pair<double, double>>> bad;
  if (opt.exclude_bad_segments)
    if (const auto sel = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'spans' AND status = 'complete'", {V(sub.id()), V(opt.recording + "_bad_segments")})) {
      bad = std::vector<std::pair<double, double>>();
      for (const Row& r : sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*sel, "id"))})) bad->emplace_back(std::stod(text(r, "start")), std::stod(text(r, "stop")));
    }

  // ── the names, and what a rebuild replaces ──
  // nxr's key `<recording>__<method>_<stamp>` (build.py): a Brainstorm-made kernel is named `<method>_KERNEL_<stamp>` (nxr_store.py: method, _, rest = name.partition("_KERNEL_"); stamp = rest[:11]);
  // a kernel of another name (made by another producer) has neither, so its whole name stands in (the rows stay authoritative: a reader finds the arrays through them)
  std::string kmethod, kstamp;
  if (const auto at = kname.find("_KERNEL_"); at != std::string::npos) { kmethod = kname.substr(0, at); kstamp = kname.substr(at + 8, 11); }
  const std::string key = safe_name(opt.recording + "__" + (kmethod.empty() ? kname : kmethod + "_" + kstamp));
  const std::string est_name = opt.recording + " \xc3\x97 " + kname;
  const std::string joint_name = opt.recording + " \xc3\x97 " + kname + " " + part_name + " \xc3\x97 " + g(frame_s) + " s";
  const auto old_joint = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(joint_name)});
  const auto old_est = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'source estimate' AND status = 'complete'", {V(sub.id()), V(est_name)});

  // ── the cache key (method rule 3): EVERY option that changes the stored arrays, and the identity of what it was run on; a rerun with the same key is a hit (nothing computed, nothing written) ──
  Json bj = Json::array();
  for (const auto& b : bands) bj.push_back(Json::array({b.first, b.second}));
  std::string bad_bytes;
  if (bad) for (const auto& [a, b] : *bad) { bad_bytes += g(a) + ":" + g(b) + ";"; }
  // what is stored: an array (relative to the tree's node), the measure it is (nxr's ATLAS_MEASURES), and the value an empty cell holds
  struct Out { std::string rel, measure; double fill; };
  std::vector<Out> outs;
  if (!is_free) outs = {{"power", "power_sum", 0.0}, {"env", "envelope_sum", 0.0}, {"envmax", "envelope_max", -std::numeric_limits<double>::infinity()}};
  else outs = {{"tensor/t11", "tensor_11", 0.0}, {"tensor/t22", "tensor_22", 0.0}, {"tensor/tnn", "tensor_nn", 0.0}, {"tensor/t12r", "tensor_12_re", 0.0}, {"tensor/t12i", "tensor_12_im", 0.0},
               {"tensor/t1nr", "tensor_1n_re", 0.0}, {"tensor/t1ni", "tensor_1n_im", 0.0}, {"tensor/t2nr", "tensor_2n_re", 0.0}, {"tensor/t2ni", "tensor_2n_im", 0.0}, {"tensor/total", "tensor_total", 0.0}};
  Json array_names = Json::array();
  for (const Out& o : outs) array_names.push_back(o.rel);
  CacheKey ckey;
  ckey.pipeline = "pyramid_build";
  ckey.params = Json{{"shape", opt.shape}, {"bands_hz", bj}, {"halo_seconds", opt.halo_seconds}, {"halo_samples", halo}, {"frame_level", opt.frame_level}, {"start_s", opt.start_s ? Json(*opt.start_s) : Json()},
                     {"depth", D}, {"span", nxr::io::span_record(span, sfreq, t0)}, {"chunk_width", nxr::io::chunk_width(*recrow)}, {"block_samples", block_samples},
                     {"exclude_bad_segments", opt.exclude_bad_segments}, {"orientation", is_free ? "free" : "constrained"}};
  if (tree_kind != "subject") ckey.params["tree"] = tree_kind;                  // the subject tree's key is what it always was
  ckey.input = Json{{"recording", rec_id}, {"samples", n_samples}, {"sfreq", sfreq}, {"kernel", kernel_id},
                    {"kernel_data", sha256_hex(std::string(reinterpret_cast<const char*>(kf.data()), kf.size() * sizeof(float))).substr(0, 16)},
                    {"channels", sha256_hex(std::string(reinterpret_cast<const char*>(chans.data()), chans.size() * sizeof(std::int32_t))).substr(0, 16)},
                    {"frames", frames_hash}, {"leaves", text(*part, "id")}, {"bad_segments", opt.exclude_bad_segments ? Json(sha256_hex(bad_bytes).substr(0, 16)) : Json()}};
  ckey.compute = NXR_COMPUTE_VERSION;
  // THE TWO TREES SHARE ONE ESTIMATE (nxr: one source estimate, one joint partition a tree). A build on the other tree finds the estimate made with the same options (the tree's own leaves apart) and adds its joint;
  // the cache key of each tree's build is stamped on its joint (params_json.cache), and on the estimate of the build that made it.
  auto cached_hit = [&]() -> bool {
    if (!old_joint) return false;
    if (const auto hit = find_cached(sub.db(), sub.id(), "selection", ckey, std::nullopt); hit && text(*hit, "id") == text(*old_joint, "id")) return true;
    if (old_est)
      if (const auto hit = find_cached(sub.db(), sub.id(), "field", ckey, std::nullopt); hit && text(*hit, "id") == text(*old_est, "id")) return true;
    return false;
  };
  if (cached_hit()) {
    say(Json{{"stage", "cache"}, {"hit", true}, {"recording", opt.recording}, {"kernel", kname}});
    PyramidBuildResult res;
    res.info = Json{{"estimate", est_name}, {"estimate_id", text(*old_est, "id")}, {"joint", joint_name}, {"joint_id", text(*old_joint, "id")}, {"leaves", L}, {"depth", D}, {"tree", tree_kind}, {"bands", bands.size()}, {"frames", n_frames},
                    {"frames_in_span", frames_span}, {"shape", opt.shape}, {"orientation", is_free ? "free" : "constrained"}, {"arrays", array_names}, {"layout", Json::array({"band", "leaf", "frame"})}, {"bits", B},
                    {"span", span.what}, {"cache", "hit"}};
    return res;
  }
  // an estimate already stored: this build's own if the options match apart from the tree's leaves (then it is shared), else another build's
  bool reuse_est = false;
  std::vector<std::string> other_joints;                                          // the other trees' joints of the estimate
  if (old_est) {
    Json stored = Json::parse(opt_text(*old_est, "producer_json").value_or("{}"), nullptr, false);
    Json sc = stored.is_object() && stored.contains("cache") ? stored["cache"] : Json::object();
    Json a = sc.value("params", Json::object()), b = ckey.params, ai = sc.value("input", Json::object()), bi = ckey.input;
    for (Json* j : {&a, &b}) { j->erase("depth"); j->erase("tree"); }
    for (Json* j : {&ai, &bi}) j->erase("leaves");
    reuse_est = !sc.empty() && a == b && ai == bi && sc.value("compute", std::string()) == ckey.compute;
    for (const Row& r : sub.db().all("SELECT id FROM selection WHERE subject_id = ? AND of_field_id = ? AND status = 'complete'", {V(sub.id()), V(text(*old_est, "id"))}))
      if (!old_joint || text(r, "id") != text(*old_joint, "id")) other_joints.push_back(text(r, "id"));
  }
  if (old_joint && !opt.replace) invalid(sub.name() + ": '" + joint_name + "' already holds a pyramid made with other options; pass --replace to rebuild it");
  if (old_est && !old_joint && !reuse_est && !opt.replace) invalid(sub.name() + ": '" + est_name + "' already has a pyramid made with other options; pass --replace to rebuild it");

  // ── the run: the chain over the chunks (the chunk service loops; the chain is nxr::pyramid) ──
  const std::int64_t leaves = L;
  std::vector<partition::AnalyticSums> acc;
  std::vector<partition::TensorSums> tacc;
  if (is_free) tacc.assign(bands.size(), partition::TensorSums(static_cast<int>(leaves), frames_span));
  else acc.assign(bands.size(), partition::AnalyticSums(static_cast<int>(leaves), frames_span));
  std::vector<std::int64_t> counts(static_cast<std::size_t>(frames_span), 0);
  bands_selection_impl(sub, bands, true, &ds);   // the bank and the Frequency manifold are checked now: a refusal after the chunks would throw the run's hours away
  const nxr::io::ChunkSource src = source_of(sub, *recrow, sfreq, t0, chans, bad);
  say(Json{{"stage", "run"}, {"recording", opt.recording}, {"kernel", kname}, {"span", span.what}, {"leaves", leaves}, {"bands", bands.size()}, {"frames_in_span", frames_span}, {"halo_samples", halo}});
  const nxr::io::ChunkRun cr = nxr::io::haloed_loop(src, span, halo, [&](const nxr::io::HaloedChunk& h) {
    const std::int64_t w = h.left + h.n + h.right;
    const Eigen::MatrixXd block = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(h.data.data(), M, w).transpose();   // [samples x channels]
    const py::Spectrum spec = py::spectrumOf(block, sfreq);
    const Eigen::MatrixXd gains = py::bandGains(spec.freq, bands, shape);
    const std::uint8_t* good = h.good.empty() ? nullptr : h.good.data() + h.left;
    const std::vector<std::int32_t> tile = py::frameOfSamples(ft, h.first, h.n, frame0);
    py::addSampleCounts(ft, h.first, h.n, good, frame0, counts);
    for (std::size_t b = 0; b < bands.size(); ++b) {
      const Eigen::MatrixXcd A = py::analyticBand(spec, gains.row(static_cast<Eigen::Index>(b)).transpose());
      if (is_free) py::tensorBandSums(tacc[b], fk, A.middleRows(h.left, h.n), good, codes.data(), w_tensor.data(), w_total.data(), tile.data(), block_samples, threads);
      else py::bandSums(acc[b], K, A.middleRows(h.left, h.n), good, codes.data(), area.data(), tile.data(), block_samples, threads);
    }
  });
  say(Json{{"stage", "run done"}, {"chunks", cr.visited.size()}, {"samples", cr.samples}, {"bad_samples", cr.bad_samples}});

  // ── gather: [band, leaf, frame] over the recording's whole frame axis ──
  const std::size_t NB = bands.size(), NL = static_cast<std::size_t>(leaves), NF = static_cast<std::size_t>(n_frames);
  std::vector<std::vector<double>> arrays_out(outs.size());
  for (std::size_t o = 0; o < outs.size(); ++o) {
    arrays_out[o].assign(NB * NL * NF, outs[o].fill);
    for (std::size_t b = 0; b < NB; ++b) {
      const std::vector<double>& from_arr = is_free ? *tacc[b].arrays()[o] : (o == 0 ? acc[b].power : o == 1 ? acc[b].env : acc[b].envmax);
      for (std::int64_t f = 0; f < frames_span; ++f)
        for (std::size_t l = 0; l < NL; ++l) arrays_out[o][(b * NL + l) * NF + static_cast<std::size_t>(frame0 + f)] = from_arr[static_cast<std::size_t>(f) * NL + l];
    }
  }
  std::vector<std::int64_t> samples(NF, 0);
  for (std::int64_t f = 0; f < frames_span; ++f) samples[static_cast<std::size_t>(frame0 + f)] = counts[static_cast<std::size_t>(f)];

  // ── store: rows and arrays, as nxr stores them ──
  PyramidBuildResult res;
  std::string est_id, frames_id, joint_id, bands_id;
  // the bank and the frame tiling are compositions of their own (stored once, shared by every recording and kernel)
  bands_id = bands_selection_impl(sub, bands, false, &ds);
  FrameTilingOptions fo;
  fo.recording = opt.recording; fo.frame_level = opt.frame_level; fo.start_s = opt.start_s;
  const FrameTilingResult fr = frame_tiling_compute(ds, sub, fo);
  const auto frow = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND manifold_id = ? AND status = 'complete'", {V(sub.id()), V(frame_tiling_name(opt.recording, opt.frame_level)), V(time_id)});
  if (!frow) missing("pyramid: the frame tiling was not stored");
  frames_id = text(*frow, "id");
  const Json fparams = Json::parse(text(*frow, "params_json"), nullptr, false);
  if (!fparams.is_object() || !fparams.contains("atlas") || fparams["atlas"].value("frame_code0", std::int64_t(-1)) != ft.code0 || integer(*frow, "n_members") != n_frames)
    invalid(frame_tiling_name(opt.recording, opt.frame_level) + ": a tiling already stored is anchored differently (--start-s); replace it with `frametiling compute --replace`");
  (void)fr;
  sub.compose([&] {
    if (old_joint) sub.remove("selection", text(*old_joint, "id"));
    if (old_est && !reuse_est) { for (const std::string& j : other_joints) sub.remove("selection", j); sub.remove("field", text(*old_est, "id")); }
    // the leaf partition's atlas block (nxr `_tree_partition`; the arrays w and codes are in its node already)
    { PyramidLeaves pl = PL; pyramid_leaves_record(sub, pl, gauge); }
    // cortex x time, and the SOURCE ESTIMATE on it: no bytes, its reductions are the pyramid
    const std::string product = sub.product({surface_id, time_id}, opt_text(*recrow, "acquisition_id"));
    const std::string node = atlas_node(sub.db(), product, key);
    const std::string tg = node + "/" + tree_kind;
    const std::string samples_path = node_path(sub.db(), time_id, key + "_samples");
    if (!reuse_est) sub.write_array(samples_path, view(samples, {static_cast<std::int64_t>(NF)}));
    const Shape arr{static_cast<std::int64_t>(NB), static_cast<std::int64_t>(NL), static_cast<std::int64_t>(NF)};
    for (std::size_t o = 0; o < outs.size(); ++o) sub.write_array(tg + "/" + outs[o].rel, view(arrays_out[o], arr));
    const Json warnings = Json::array({
        "the minimum-norm current is smooth: its spatial resolution is the kernel's point-spread, not the leaves'; a leaf finer than it carries no independent information",
        "a band's analytic amplitude is meaningful for a band narrow against its centre; octave bands are wide, read the powers as band-limited energy, not as an oscillation's amplitude",
        "the band filter is circular over the block read (chunk + halo): the first and last `halo` seconds of a recording carry its edge effect, as nxr's whole-record filter does at both ends",
        shape == py::Shape::Morlet ? "Morlet bands overlap their neighbours: band powers do not add to broadband (shape power_partition does)" : "power_partition crossovers are half an octave wide: a tone near an edge splits between two bands",
        is_free ? "free orientation: the tensor's principal axis is the dominant orientation of the current as the kernel's inverse made it; for a minimum-norm kernel that is partly its own prior (the kernel's orientation prior is not a measurement); a vertex with no canonical frame is in the total only"
                : "constrained orientation only: the current is the kernel's normal component; the free-orientation tensor is not stored (a free kernel gives it)"});
    FieldSpec fs;
    fs.name = est_name; fs.kind = "source estimate"; fs.manifold_id = product; fs.value_type = is_free ? "vector3" : "scalar";
    fs.cols.set("label", (kmethod.empty() ? kname : kmethod) + (is_free ? " (free)" : " (constrained)")).set("data_type", "float32").set("by_operator_id", kernel_id).set("derived_from_id", rec_id)
        .set("description", "the kernel applied to the recording, stored as its pyramid reductions only, never K x itself");
    if (const auto a = opt_text(*recrow, "acquisition_id")) fs.cols.set("acquisition_id", *a);
    fs.cols.json("producer_json", Json{{"by", "nxr-neuro pyramid build"}, {"producer", internal::producer_string("pyramid build")}, {"kernel", kname}, {"method", kmethod.empty() ? Json(nullptr) : Json(kmethod)}, {"stamp", kstamp.empty() ? Json(nullptr) : Json(kstamp)}, {"orientation", is_free ? "free" : "constrained"},
                                       {"shape", opt.shape}, {"bands_hz", bj}, {"halo_seconds", opt.halo_seconds}, {"halo_samples", halo}, {"frame_level", opt.frame_level}, {"frame_s", frame_s},
                                       {"filterbank", shape == py::Shape::Morlet ? "Morlet: Gaussian gain, unit peak, centred on (lo+hi)/2, FWHM hi-lo (analytic: positive frequencies doubled)" : "octave power partition (cos/sin crossover, half an octave), nxr band_gains"},
                                       {"depth", D}, {"leaves", leaves}, {"tree", tree_kind}, {"span", nxr::io::span_record(span, sfreq, t0)}, {"chunk_width", nxr::io::chunk_width(*recrow)},
                                       {"block_samples", block_samples}, {"threads", threads}, {"threads_from", nxr::parallel::allocation().source}, {"exclude_bad_segments", opt.exclude_bad_segments}, {"start_s", opt.start_s ? Json(*opt.start_s) : Json()}, {"cache", ckey.json()},
                                       {"bad_segments", bad ? Json{{"selection", opt.recording + "_bad_segments"}, {"n_spans", static_cast<std::int64_t>(bad->size())}, {"rule", "samples inside a bad span carry no current and no count"}} : Json(nullptr)},
                                       {"warnings", warnings}});
    if (reuse_est) est_id = text(*old_est, "id");
    else {
      est_id = sub.field(std::move(fs));
      Cols cnt;
      cnt.set("measure", "count").set("of_field_id", est_id).set("array_path", samples_path).set("computed_by", "nxr-neuro pyramid build");
      sub.measurement(frames_id, cnt);
    }
    SelectionSpec js;
    js.name = joint_name; js.type = "set"; js.manifold_id = product; js.no_elements = true;
    js.cols.set("cell", "0,1").set("of_field_id", est_id).set("n_members", static_cast<std::int64_t>(L) * n_frames)
        .set("description", "the joint cells of " + part_name + " x the " + std::to_string(n_frames) + " frames: code = (tile << " + std::to_string(B) + ") | frame")
        .json("params_json", Json{{"joint", Json{{"space", text(*part, "id")}, {"time", frames_id}, {"bits", B}, {"tiles", L}, {"frames", n_frames}, {"layout", Json::array({"band", "tile", "frame"})}}},
                                  {"atlas", Json{{"tree", tree_kind}, {"gauge", gauge}, {"node", tg}}}, {"cache", ckey.json()}});
    if (const auto a = opt_text(*recrow, "acquisition_id")) js.cols.set("acquisition_id", *a);
    for (const Out& o : outs)
      for (std::size_t b = 0; b < NB; ++b) {
        Cols c;
        c.set("measure", o.measure).set("of_field_id", est_id).set("array_path", tg + "/" + o.rel).set("within_selection_id", bands_id).set("within_code", static_cast<std::int64_t>(b)).set("computed_by", "nxr-neuro pyramid build");
        js.measurements.push_back(c);
      }
    joint_id = sub.selection(std::move(js));
  });
  res.info = Json{{"estimate", est_name}, {"estimate_id", est_id}, {"joint", joint_name}, {"joint_id", joint_id}, {"leaves", leaves}, {"depth", D}, {"tree", tree_kind}, {"bands", bands.size()}, {"frames", n_frames},
                  {"frames_in_span", frames_span}, {"shape", opt.shape}, {"orientation", is_free ? "free" : "constrained"}, {"arrays", array_names}, {"layout", Json::array({"band", "leaf", "frame"})}, {"bits", B},
                  {"span", span.what}, {"bad_samples", cr.bad_samples}, {"cache", "miss"}};
  return res;
}

FrameCells frame_cells(DatasetWriter& ds, SubjectWriter& sub, const std::string& recording, int frame_level, std::optional<double> start_s, const std::string& who) {
  if (frame_level > 0) invalid(who + ": the frame level is 0 (the day) or finer (negative)");
  const auto recrow = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording' AND status = 'complete'", {V(sub.id()), V(recording)});
  if (!recrow) missing(sub.name() + ": no recording '" + recording + "'");
  const finder::Value rv = finder::value_of(sub.db(), "field", *recrow);
  if (!rv.time || !(rv.rate > 0)) invalid(recording + ": it has no time line with a rate");
  FrameCells fc;
  fc.time_id = *rv.time;
  const std::int64_t n_samples = integer(*sub.db().read("manifold", fc.time_id), "n_vertices");
  try { fc.ft = nxr::tower::makeFrameTiling(rv.rate, n_samples, frame_level, start_s); } catch (const std::invalid_argument& e) { invalid(who + ": " + e.what()); }
  fc.n_frames = fc.ft.count;
  fc.bits = joint_bits(fc.n_frames);
  FrameTilingOptions fo;
  fo.recording = recording; fo.frame_level = frame_level; fo.start_s = start_s;
  frame_tiling_compute(ds, sub, fo);
  const auto frow = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND manifold_id = ? AND status = 'complete'", {V(sub.id()), V(frame_tiling_name(recording, frame_level)), V(fc.time_id)});
  if (!frow) missing(who + ": the frame tiling was not stored");
  fc.frames_id = text(*frow, "id");
  const Json fparams = Json::parse(opt_text(*frow, "params_json").value_or("{}"), nullptr, false);
  if (!fparams.is_object() || !fparams.contains("atlas") || fparams["atlas"].value("frame_code0", std::int64_t(-1)) != fc.ft.code0 || integer(*frow, "n_members") != fc.n_frames)
    invalid(frame_tiling_name(recording, frame_level) + ": a tiling already stored is anchored differently (--start-s); replace it with `frametiling compute --replace`");
  return fc;
}

std::string frame_sums_write(SubjectWriter& sub, const PyramidLeaves& PL, const FrameCells& fc, const std::string& joint_name, const std::string& description, std::int64_t frame0, std::int64_t frames_span,
                             const std::vector<FrameSumsSeries>& series, const std::string& computed_by, const std::optional<std::string>& acquisition_id) {
  const std::int64_t L = PL.leaves, NF = fc.n_frames;
  const std::string surface_id = text(*PL.part, "manifold_id");
  const std::string product = sub.product({surface_id, fc.time_id}, acquisition_id);
  SelectionSpec js;
  js.name = joint_name; js.type = "set"; js.manifold_id = product; js.no_elements = true;
  js.cols.set("cell", "0,1").set("n_members", L * NF).set("description", description)
      .json("params_json", Json{{"joint", Json{{"space", text(*PL.part, "id")}, {"time", fc.frames_id}, {"bits", fc.bits}, {"tiles", L}, {"frames", NF}, {"layout", Json::array({"band", "tile", "frame"})}}},
                                {"atlas", Json{{"tree", PL.tree}, {"gauge", PL.record["atlas"].value("gauge", Json(nullptr))}, {"node", PL.node}}}});
  if (acquisition_id) js.cols.set("acquisition_id", *acquisition_id);
  for (const FrameSumsSeries& s : series) {
    if (static_cast<std::int64_t>(s.counts.size()) != frames_span) invalid("frame sums: the counts do not fit the span's frames");
    // the per-frame count of the series' steps, on the frame tiling, as the build stores its samples (a measurement row `count` of the Field)
    std::vector<double> counts(static_cast<std::size_t>(NF), 0.0);
    for (std::int64_t f = 0; f < frames_span; ++f) counts[static_cast<std::size_t>(frame0 + f)] = s.counts[static_cast<std::size_t>(f)];
    const std::string cpath = PL.node + "/" + s.group + "/count";
    sub.write_array(cpath, view(counts, {NF}));
    Cols cnt;
    cnt.set("measure", "count").set("of_field_id", s.field_id).set("array_path", cpath).set("computed_by", computed_by);
    sub.measurement(fc.frames_id, cnt);
    for (const FrameSumsRow& r : s.rows) {
      if (static_cast<std::int64_t>(r.v.size()) != L * frames_span) invalid("frame sums: " + r.measure + " does not fit the leaves x the span's frames");
      std::vector<double> out(static_cast<std::size_t>(L * NF), 0.0);
      for (std::int64_t l = 0; l < L; ++l)
        for (std::int64_t f = 0; f < frames_span; ++f) out[static_cast<std::size_t>(l * NF + frame0 + f)] = r.v[static_cast<std::size_t>(l * frames_span + f)];
      const std::string path = PL.node + "/" + s.group + "/" + r.rel;
      sub.write_array(path, view(out, {1, L, NF}));
      Cols c;
      c.set("measure", r.measure).set("of_field_id", s.field_id).set("array_path", path).set("computed_by", computed_by);
      if (!r.unit.empty()) c.set("unit", r.unit);
      js.measurements.push_back(c);
    }
  }
  return sub.selection(std::move(js));
}

SpatialBandsResult pyramid_spatial_bands(DatasetWriter& ds, SubjectWriter& sub, const SpatialBandsOptions& opt) {
  const SurfaceInput S = read_surface(sub, opt.surface, true);
  const std::string surface_name = S.name;
  auto ladder_of = [&]() { return sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND manifold_id = ? AND name = ? AND type = 'ladder' AND status = 'complete'", {V(sub.id()), V(S.id), V(surface_name + " tiles")}); };
  if (!ladder_of()) { TilesComputeOptions to; to.surface = surface_name; tiles_compute(ds, sub, to); }
  const auto ladder = ladder_of();
  if (!ladder) missing(sub.name() + ": the ladder of " + surface_name + " was not stored");
  std::vector<SpatialBandEdge> edges;
  Json levels = Json::array();
  for (const Row& m : sub.db().all("SELECT * FROM selection WHERE member_of_id = ? AND status = 'complete' ORDER BY member_ordinal", {V(text(*ladder, "id"))})) {
    const Json r = Json::parse(opt_text(m, "params_json").value_or("{}"), nullptr, false);
    if (!r.is_object() || !r.contains("level") || !r.contains("lambda_min") || !r.contains("lambda_max")) invalid(surface_name + " tiles: a level's record has no lambda range (a ladder made by another producer)");
    SpatialBandEdge e;
    e.level = r["level"].get<std::int64_t>(); e.lo = r["lambda_min"].get<double>(); e.hi = r["lambda_max"].get<double>();
    edges.push_back(e);
    levels.push_back(Json{{"level", e.level}, {"lambda_min", e.lo}, {"lambda_max", e.hi}, {"wavelength_m", r.value("wavelength_m", std::nan(""))}});
  }
  // the Eigenvalue line lives in the dataset's default subject; made there when absent (additive, as the Frequency manifold is made), so a verb that needs a spatial band never asks for a separate step
  if (!find_eigenvalue(sub.db()) && text(sub.row(), "kind") != "template") {
    if (const auto def = ds.template_()) {
      SubjectWriter dsub = SubjectWriter::open(ds, text(*def, "name"));
      dsub.compose([&] { ensure_eigenvalue(dsub); });
    }
  }
  const std::string id = spatial_bands_selection(sub, surface_name, edges);
  SpatialBandsResult res;
  res.selection_id = id;
  res.edges = edges;
  {
    const Json gp = Json::parse(text(*ladder, "params_json"), nullptr, false);
    if (!gp.is_object() || !gp.contains("area_m2") || !gp.contains("pieces")) invalid(surface_name + " tiles: its record lacks the surface's area (a ladder made by another producer)");
    res.piece_area = gp["area_m2"].get<double>() / std::max(1, gp["pieces"].get<int>());
  }
  res.info = Json{{"surface", surface_name}, {"selection", spatial_bands_name(surface_name)}, {"selection_id", id}, {"levels", edges.size()}, {"bands", levels}, {"line", kEigenvalueName}, {"unit", "1/m^2"}};
  return res;
}

}  // namespace nxr::pipelines

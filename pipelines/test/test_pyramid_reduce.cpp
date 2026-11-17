// The GROUP SCOPE of the measurement pyramid (step 6): the group tree per subject, `pyramid build --tree group`, and `pyramid reduce`, on a SYNTHETIC dataset (never a real dataset):
//   a default subject (a two-hemisphere "brain" and its registered sphere, group tree depth 3) and three subjects whose surfaces and spheres differ (other meshes, rotated spheres);
//   S1: one recording, a constrained and a free kernel; S2: two recordings, two constrained kernels of one method (an older and a newer stamp); S3: one recording.
//   - the group tree: every subject's leaves at depth 3 are the same 16 tiles (the group partition, its `atlas` block, tree group), a subject's vertex takes the tile of its nearest template vertex on the sphere
//   - `pyramid build --tree group` stores nxr's rows (joint on the group partition, node `<recording>__<method>_<stamp>_atlas/group/...`); the subject tree and the group tree SHARE the estimate (one estimate, two joints)
//   - the group reduce: refusals (no members, the default subject, a subject without the group tree, two kernels of one method without a selector, another band bank); with `latest` the sums of the members,
//     cell for cell, frames collapsed (checked here against the members' own arrays), per-subject means and densities, the rows (kind `group map` / `source estimate`, pathless, function sum, contributions), a re-run replaces
//   NXR_TEST_KEEP=<folder>: the dataset is made there BEFORE the reduce (a copy, `pre`) and after it (`post`), for nxr's own reduce to run on the copy (io/test/parity/test_pyramid_reduce_parity.py)
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/frequency.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/layout.h"
#include "nxr/io/migrate.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/group.h"
#include "nxr/pipelines/pyramid.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace lad = nxr::ladder;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class T> std::vector<T> read_at(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}
const int M = 6, T = 4000;
const double sfreq = 250.0;

// the sphere turned by `a` about z (a registered sphere that is not the template's grid)
std::vector<double> turned(const std::vector<double>& s, double a) {
  std::vector<double> o(s.size());
  for (std::size_t i = 0; i + 2 < s.size(); i += 3) { o[i] = std::cos(a) * s[i] - std::sin(a) * s[i + 1]; o[i + 1] = std::sin(a) * s[i] + std::cos(a) * s[i + 1]; o[i + 2] = s[i + 2]; }
  return o;
}

struct Spec { std::string name; Brain brain; double turn; std::vector<std::string> recordings; std::vector<std::pair<std::string, bool>> kernels; unsigned seed; };

void make_subject(DatasetWriter& ds, const Spec& sp) {
  const Brain& b = sp.brain;
  const int N = b.n;
  std::mt19937 rng(sp.seed);
  std::normal_distribution<double> nd;
  SubjectWriter sub = SubjectWriter::create(ds, sp.name);
  sub.compose([&] {
    SurfaceOptions so; so.name = "cortex_pial"; so.sphere = turned(b.Sph, sp.turn);
    const std::string surf = add_surface(sub, so, b.P, b.F).id;
    const std::string cid = uuid7();
    std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0, 0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0, 0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                        [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
    ManifoldSpec tm;
    tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
    tm.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto lc = lattice_counts({T});
    if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
    GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
    DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
    tg.dimensions.push_back(dm); tm.geometry = tg;
    const std::string tid = sub.manifold(std::move(tm));
    const std::string prod = sub.product({cid, tid});
    for (const std::string& name : sp.recordings) {
      std::vector<float> xf(static_cast<std::size_t>(M) * T);
      for (int c = 0; c < M; ++c) { const double amp = 1 + 0.3 * c; for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(amp * std::sin(2 * M_PI * 10.0 * t / sfreq + c) + 0.5 * nd(rng)); }
      FieldSpec rec;
      rec.name = name; rec.path = task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, std::int64_t(2048)}; rec.array.compress = false; rec.cols.set("unit", "T");
      sub.field(std::move(rec));
    }
    std::vector<std::int32_t> chans = {0, 1, 2, 3, 4, 5};
    SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
    const std::string sel = sub.selection(std::move(ss));
    for (const auto& [kname, free] : sp.kernels) {
      const int rows = free ? 3 * N : N;
      std::vector<float> k(static_cast<std::size_t>(rows) * M);
      for (auto& x : k) x = static_cast<float>(1e-3 * nd(rng));
      OperatorSpec op;
      op.name = kname; op.path = kname; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
      op.data = view(k, {rows, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", free ? "vector3" : "scalar");
      sub.operator_(std::move(op));
    }
  });
}
// a scalar map on the subject's surface, summarised on the GROUP partition as the tile summaries are on a ladder level (nxr's maps/<name>/{w,s1,s2,n,min,max})
void add_map(SubjectWriter& sub, const Brain& b, unsigned seed, double scale) {
  const task::Subject tsub = task::Subject::of(sub);
  const auto surf = tsub.find({.cls = task::Cls::manifold, .name = "cortex_pial"});
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::vector<float> x(static_cast<std::size_t>(b.n));
  for (auto& v : x) v = static_cast<float>(scale * (1 + 0.3 * nd(rng)));
  const auto part = tsub.find({.cls = task::Cls::selection, .name = "cortex_pial group tiles L3"});
  const Json pp = part->params;
  const std::vector<std::int32_t> codes = read_at<std::int32_t>(sub, part->array_path.value());
  lad::Mesh mesh; mesh.n = b.n; mesh.positions = b.P; mesh.faces = b.F;
  const std::vector<double> area = lad::vertexAreas(mesh);
  std::vector<double> xd(x.begin(), x.end());
  const lad::ScalarSummary s = lad::scalarSummary(codes, 16, area, xd.data());
  sub.compose([&] {
    FieldSpec f;
    f.name = "map_" + sub.name(); f.path = tsub.node_path(surf->id, f.name); f.kind = "synthetic map"; f.manifold_id = surf->id; f.data = view(x, {b.n}); f.cols.set("unit", "T");
    const std::string fid = sub.field(std::move(f));
    const std::string g = pp["node"].get<std::string>() + "/maps/map_" + sub.name();
    const std::vector<std::pair<const char*, std::pair<const char*, const Eigen::VectorXd*>>> parts = {{"w", {"weight", &s.w}}, {"s1", {"sum", &s.s1}}, {"s2", {"sum_square", &s.s2}}, {"n", {"count", &s.n}}, {"min", {"min", &s.min}}, {"max", {"max", &s.max}}};
    for (const auto& [rel, mv] : parts) {
      std::vector<double> v(mv.second->data(), mv.second->data() + mv.second->size());
      sub.write_array(g + "/" + rel, view(v, {16}));
      Cols c;
      c.set("measure", mv.first).set("of_field_id", fid).set("array_path", g + "/" + rel).set("computed_by", "test");
      if (std::string(mv.first) != "weight" && std::string(mv.first) != "count") c.set("unit", "T");
      sub.measurement(part->id, c);
    }
  });
}
// the member's own collapsed array: [band, leaf] summed (max: maxed) over frames
std::vector<double> collapsed(SubjectWriter& sub, const std::string& path, bool maxop, std::vector<std::int64_t>* shape = nullptr) {
  std::vector<std::int64_t> sh;
  const std::vector<double> a = read_at<double>(sub, path, &sh);
  if (shape) *shape = {sh[0], sh[1]};
  std::vector<double> o(static_cast<std::size_t>(sh[0] * sh[1]));
  for (std::int64_t i = 0; i < sh[0] * sh[1]; ++i) { double acc = maxop ? -INFINITY : 0.0; for (std::int64_t f = 0; f < sh[2]; ++f) acc = maxop ? std::max(acc, a[static_cast<std::size_t>(i * sh[2] + f)]) : acc + a[static_cast<std::size_t>(i * sh[2] + f)]; o[static_cast<std::size_t>(i)] = acc; }
  return o;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const char* keep = std::getenv("NXR_TEST_KEEP");
  const fs::path root = keep ? fs::path(keep) / "ds" : fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_pyramid_reduce_" + std::to_string(::getpid()));
  fs::remove_all(root);
  if (keep) fs::remove_all(fs::path(keep) / "pre");
  struct Cleanup { fs::path p; bool keep; ~Cleanup() { if (!keep) fs::remove_all(p); } } cleanup{root, keep != nullptr};
  const Brain tb = brain(3, false), b1 = brain(2, false), b2 = brain(2, false, 0.15, Vector3d(2.1, 1.0, 1.5)), b3 = brain(3, false, 0.05, Vector3d(1.9, 1.1, 1.4));
  // the store format the dataset is made at: 52 (the default) or, with NXR_TEST_STORE_FORMAT=51, the folder tree (the parity test runs both)
  CreateOptions co;
  if (const char* f = std::getenv("NXR_TEST_STORE_FORMAT")) co.store_format = std::atoi(f);
  DatasetWriter ds = create_dataset(root, "demo", co);
  // the default subject: a template with the registered sphere, the Frequency manifold and the group tree at depth 3
  {
    SubjectOptions so; so.kind = "template";
    SubjectWriter def = SubjectWriter::create(ds, "@default_subject", so);
    def.compose([&] {
      nxr::bst::ensure_frequency(def);
      SurfaceOptions s; s.name = "cortex_pial"; s.sphere = tb.Sph;
      add_surface(def, s, tb.P, tb.F);
    });
    GroupTreeOptions go; go.atlas_depth = 3;
    const TilesComputeResult gr = group_tree_compute(ds, def, go);
    CHECK(gr.info.value("depth", 0) >= 3);
    std::printf("  default subject: ladder depth %d\n", gr.info.value("depth", 0));
  }
  const std::string k1 = "MN_MEG_KERNEL_261005_2149", k2 = "MN_MEG_KERNEL_261006_0101", kf = "WMNE_MEG_KERNEL_261005_2150";
  make_subject(ds, {"S1", b1, 0.20, {"task-rest_run-1"}, {{k1, false}, {kf, true}}, 11});
  make_subject(ds, {"S2", b2, -0.35, {"task-rest_run-1", "task-rest_run-2"}, {{k1, false}, {k2, false}}, 12});
  make_subject(ds, {"S3", b3, 0.05, {"task-rest_run-1"}, {{k1, false}}, 13});
  make_subject(ds, {"S4", b1, 0.0, {"task-rest_run-1"}, {{k1, false}}, 14});          // no group tree: refused as a member
  make_subject(ds, {"S5", b1, 0.1, {"task-rest_run-1"}, {{k1, false}}, 15});          // other bands: refused as a member

  auto build = [&](const std::string& subject, const std::string& rec, const std::string& kernel, const std::string& tree, std::vector<std::pair<double, double>> bands = {}) {
    SubjectWriter sub = SubjectWriter::open(ds, subject);
    if (tree == "group" && subject != "S4") { /* the group tiles are made by the build */ }
    PyramidBuildOptions o;
    o.chunks = "0";                      // the first chunk's samples, stated
    o.recording = rec; o.kernel = kernel; o.tree = tree; o.halo_seconds = 2.0; o.bands = bands;
    if (tree == "group") o.depth = 3;
    return pyramid_build(ds, sub, o);
  };
  // ── the group tree ──
  {
    SubjectWriter s1 = SubjectWriter::open(ds, "S1");
    const PyramidBuildResult r = build("S1", "task-rest_run-1", k1, "group");
    CHECK(r.info["tree"] == "group" && r.info["depth"] == 3 && r.info["leaves"] == 16 && r.info["cache"] == "miss");
    const task::Subject ts1 = task::Subject::of(s1);
    const auto part = ts1.find({.cls = task::Cls::selection, .name = "cortex_pial group tiles L3"});
    CHECK(part && part->n_members == 16);
    const Json pp = part->params;
    CHECK(pp["atlas"]["tree"] == "group" && pp["atlas"]["level"] == 3 && pp["atlas"].contains("areas") && pp["atlas"]["node"] == pp["node"]);
    const auto parent = ts1.find({.cls = task::Cls::selection, .name = "cortex_pial_group_tiles"});
    CHECK(parent && !parent->member_of_id && parent->n_members.value_or(0) >= 4);
    // the leaves are the template's tiles: a vertex's code is its nearest template vertex's (checked by hand through the sphere)
    const SurfaceInput S = read_surface(s1, std::string("cortex_pial"), true);
    SubjectWriter dsub = SubjectWriter::open(ds, "@default_subject");
    const std::vector<std::int32_t> codes = read_at<std::int32_t>(s1, part->array_path.value());
    const auto tl3 = task::Subject::of(dsub).find({.cls = task::Cls::selection, .name = "cortex_pial tiles L3"});
    const std::vector<std::int32_t> tcodes = read_at<std::int32_t>(dsub, tl3->array_path.value());
    const std::vector<double> sph = turned(b1.Sph, 0.20);
    int bad = 0;
    for (int v = 0; v < b1.n; v += 7) {
      double best = -2; int bj = -1;
      const int h = v < b1.nHalf ? 0 : 1;
      for (int j = h * tb.nHalf; j < (h + 1) * tb.nHalf; ++j) { const double d = sph[3 * v] * tb.Sph[3 * j] + sph[3 * v + 1] * tb.Sph[3 * j + 1] + sph[3 * v + 2] * tb.Sph[3 * j + 2]; if (d > best) { best = d; bj = j; } }
      if (codes[static_cast<std::size_t>(v)] != tcodes[static_cast<std::size_t>(bj)]) ++bad;
    }
    CHECK(bad == 0);
    // the subject tree on the same estimate: one estimate, two joints
    const PyramidBuildResult rs = build("S1", "task-rest_run-1", k1, "subject");
    CHECK(rs.info["tree"] == "subject" && rs.info["cache"] == "miss");
    CHECK(ts1.find_all({.cls = task::Cls::field, .kind = "source estimate"}).size() == 1);
    const auto joints = ts1.find_all({.cls = task::Cls::selection, .params_present = {"joint"}});
    CHECK(joints.size() == 2);
    std::set<std::string> trees;
    for (const task::Node& j : joints) trees.insert(j.params["atlas"]["tree"].get<std::string>());
    CHECK((trees == std::set<std::string>{"group", "subject"}));
    CHECK(build("S1", "task-rest_run-1", k1, "group").info["cache"] == "hit" && build("S1", "task-rest_run-1", k1, "subject").info["cache"] == "hit");
    CHECK(code_of([&] { SubjectWriter s = SubjectWriter::open(ds, "S1"); PyramidBuildOptions o; o.recording = "task-rest_run-1"; o.kernel = k1; o.tree = "group"; o.halo_seconds = 3.0; o.depth = 3; pyramid_build(ds, s, o); }) == "invalid_argument");
    CHECK(code_of([&] { SubjectWriter s = SubjectWriter::open(ds, "S1"); PyramidBuildOptions o; o.recording = "task-rest_run-1"; o.kernel = k1; o.tree = "group"; o.depth = 99; pyramid_build(ds, s, o); }) == "invalid_argument");
    CHECK(code_of([&] { SubjectWriter s = SubjectWriter::open(ds, "S1"); PyramidBuildOptions o; o.recording = "task-rest_run-1"; o.kernel = k1; o.tree = "bad"; pyramid_build(ds, s, o); }) == "invalid_argument");
  }
  // the free kernel needs the canonical frames (on the subject's registered sphere)
  { SubjectWriter s1 = SubjectWriter::open(ds, "S1"); FramesComputeOptions fo; frames_compute(ds, s1, fo); const PyramidBuildResult r = build("S1", "task-rest_run-1", kf, "group"); CHECK(r.info["orientation"] == "free" && r.info["arrays"].size() == 10); }
  build("S2", "task-rest_run-1", k1, "group"); build("S2", "task-rest_run-2", k1, "group"); build("S2", "task-rest_run-1", k2, "group"); build("S2", "task-rest_run-2", k2, "group");
  build("S3", "task-rest_run-1", k1, "group");
  build("S5", "task-rest_run-1", k1, "group", {{2, 4}, {4, 8}});
  { SubjectWriter s = SubjectWriter::open(ds, "S1"); add_map(s, b1, 21, 1.0); }
  { SubjectWriter s = SubjectWriter::open(ds, "S2"); add_map(s, b2, 22, 2.0); }
  { SubjectWriter s = SubjectWriter::open(ds, "S3"); add_map(s, b3, 23, 3.0); }
  // the leaves are the SAME in every subject: 16 tiles each, and the leaf areas differ (another surface)
  {
    SubjectWriter a = SubjectWriter::open(ds, "S1"), c = SubjectWriter::open(ds, "S3");
    auto n = [&](SubjectWriter& s) { return task::Subject::of(s).require({.cls = task::Cls::selection, .name = "cortex_pial group tiles L3"}, "the group tiles").n_members.value_or(0); };
    CHECK(n(a) == 16 && n(c) == 16);
  }
  if (keep) { fs::remove_all(fs::path(keep) / "pre"); fs::copy(root, fs::path(keep) / "pre", fs::copy_options::recursive); }
  const fs::path pkg_root = root.parent_path() / (root.filename().string() + "_pkg");   // the same dataset, to be split into packages: the reduce is run on both
  fs::remove_all(pkg_root);
  fs::copy(root, pkg_root, fs::copy_options::recursive);
  fs::remove(pkg_root / "demo" / ".writer.lock");

  // ── refusals (nothing is stored by them) ──
  { PyramidReduceOptions o; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }
  { PyramidReduceOptions o; o.subjects = {"@default_subject"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }
  { PyramidReduceOptions o; o.subjects = {"S1", "S1"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }
  { PyramidReduceOptions o; o.subjects = {"S1", "nobody"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "no_such_row"); }
  { PyramidReduceOptions o; o.subjects = {"S1", "S4"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }
  { PyramidReduceOptions o; o.subjects = {"S1", "S5"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }
  { PyramidReduceOptions o; o.subjects = {"S1", "S2"}; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }      // S2: two kernels of one method, no selector
  { PyramidReduceOptions o; o.subjects = {"S1"}; o.depth = 2; CHECK(code_of([&] { pyramid_reduce(ds, o); }) == "invalid_argument"); }   // no group tree at depth 2

  // ── the reduce ──
  PyramidReduceOptions ro;
  ro.subjects = {"S1", "S2", "S3"}; ro.kernels = "latest";
  const PyramidReduceResult rr = pyramid_reduce(ds, ro);
  std::printf("  reduced %d members: %s\n", rr.info["members"].get<int>(), rr.info["time"].dump().c_str());
  CHECK(rr.info["members"] == 3 && rr.info["depth"] == 3 && rr.info["leaves"] == 16);
  SubjectWriter dsub = SubjectWriter::open(ds, "@default_subject");
  const task::Subject tdsub = task::Subject::of(dsub);
  const auto dpart = tdsub.find({.cls = task::Cls::selection, .name = "cortex_pial tiles L3"});
  const Json dp = dpart->params;
  auto atlas_fields = [&] {   // the group fields: no arrays of their own, made by a reduce
    std::vector<task::Node> out;
    for (const task::Node& n : tdsub.find_all({.cls = task::Cls::field}))
      if (!n.path && n.producer.contains("atlas")) out.push_back(n);
    return out;
  };
  CHECK(dp["atlas"]["tree"] == "group" && dp["atlas"]["level"] == 3 && dp["atlas"]["members"].size() == 3);
  const std::string node = rr.info["node"].get<std::string>();
  CHECK(node == dp["atlas"]["node"]);
  // the group fields: pathless, function sum, contributions
  const auto gfs = atlas_fields();
  std::set<std::string> gnames;
  for (const task::Node& f : gfs) { gnames.insert(f.name); CHECK(f.fact("function") == "sum" && !f.path); }
  CHECK((gnames == std::set<std::string>{"group synthetic_map", "group rest__MN_MEG__constrained", "group rest__WMNE_MEG__free"}) || (std::printf("  names: %s\n", [&] { std::string s; for (auto& n : gnames) s += n + " | "; return s; }().c_str()) && false));
  for (const task::Node& f : gfs) {
    const std::string n = f.name;
    const auto cs = testing::Probe(dsub).rows("field_contribution", {{"field_id", f.id}});
    const std::size_t want = n == "group synthetic_map" ? 3 : n == "group rest__MN_MEG__constrained" ? 4 : 1;          // S2's two recordings contribute twice
    CHECK(cs.size() == want);
    CHECK(f.kind == (n == "group synthetic_map" ? "group map" : "source estimate"));
  }
  // the sums, cell for cell, against the members' own arrays (the newest kernel of S2 only: `latest`)
  {
    std::vector<double> want;
    std::int64_t nb = 0;
    double smp = 0;
    for (const auto& [sname, recs, kn] : std::vector<std::tuple<std::string, std::vector<std::string>, std::string>>{{"S1", {"task-rest_run-1"}, k1}, {"S2", {"task-rest_run-1", "task-rest_run-2"}, k2}, {"S3", {"task-rest_run-1"}, k1}}) {
      SubjectWriter m = SubjectWriter::open(ds, sname);
      for (const std::string& rec : recs) {
        const std::string key = rec + "__MN_MEG_" + kn.substr(kn.find("_KERNEL_") + 8, 11);
        const auto j = task::Subject::of(m).find({.cls = task::Cls::selection, .name_like = rec + " \xc3\x97 " + kn + " %", .params = {{"atlas.tree", "group"}}});
        CHECK(j);
        const std::string pn = j->params["atlas"]["node"].get<std::string>();
        std::vector<std::int64_t> sh;
        const std::vector<double> c = collapsed(m, pn + "/power", false, &sh);
        nb = sh[0];
        if (want.empty()) want.assign(c.size(), 0.0);
        for (std::size_t i = 0; i < c.size(); ++i) want[i] += c[i];
        const auto cnt = testing::Probe(m).rows("selection_measurement", {{"of_field_id", j->fact("of_field_id").get<std::string>()}, {"measure", "count"}});
        for (std::int64_t x : read_at<std::int64_t>(m, text(cnt.front(), "array_path"))) smp += static_cast<double>(x);
      }
    }
    const std::string g = node + "/time/rest__MN_MEG__constrained";
    std::vector<std::int64_t> sh;
    const std::vector<double> got = read_at<double>(dsub, g + "/power", &sh);
    CHECK(sh.size() == 2 && sh[0] == nb && sh[1] == 16);
    double worst = 0;
    for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(got[i] - want[i]) / (std::abs(want[i]) + 1e-300));
    std::printf("  group power vs the members' own sums: worst relative difference %.2e\n", worst);
    CHECK(want.size() == got.size() && worst < 1e-12);
    CHECK(read_at<double>(dsub, g + "/samples")[0] == smp);
    std::vector<std::int64_t> dsh;
    const std::vector<double> dens = read_at<double>(dsub, g + "/per_subject/density", &dsh);
    CHECK(dsh.size() == 3 && dsh[0] == 3 && dsh[1] == nb && dsh[2] == 16);
    for (double d : dens) CHECK(std::isfinite(d) && d > 0);
    // the envelope maximum is the maximum, not a sum
    const std::vector<double> em = read_at<double>(dsub, g + "/envmax");
    const std::vector<double> es = read_at<double>(dsub, g + "/env");
    bool ok = true;
    for (std::size_t i = 0; i < em.size(); ++i) ok = ok && em[i] > 0 && em[i] <= es[i] * 1.0000001;
    CHECK(ok);
    // the rows on the default subject's partition: one per band for each array, within the default's atlas bands
    const auto rows = testing::Probe(dsub).rows("selection_measurement", {{"selection_id", dpart->id}, {"of_field_id", tdsub.require({.cls = task::Cls::field, .name = "group rest__MN_MEG__constrained"}, "the group estimate").id}});
    CHECK(rows.size() == 3 * static_cast<std::size_t>(nb));
    const auto bands = tdsub.find({.cls = task::Cls::selection, .name = "atlas bands", .type = "spans"});
    CHECK(bands);
    std::set<std::string> measures;
    for (const Row& r : rows) { CHECK(text(r, "within_selection_id") == bands->id && !is_null(r, "array_path")); measures.insert(text(r, "measure")); }
    CHECK((measures == std::set<std::string>{"power_sum", "envelope_sum", "envelope_max"}));
  }
  // the spatial group map: sums add, extremes extreme, the per-subject mean
  {
    const std::string g = node + "/maps/synthetic_map";
    std::vector<double> w(16, 0), s1(16, 0), mx(16, -INFINITY), mn(16, INFINITY);
    std::vector<std::vector<double>> means;
    for (const char* sname : {"S1", "S2", "S3"}) {
      SubjectWriter m = SubjectWriter::open(ds, sname);
      const std::string mp = task::Subject::of(m).require({.cls = task::Cls::selection, .name = "cortex_pial group tiles L3"}, "the group tiles").params["node"].get<std::string>() + "/maps/map_" + sname;
      const auto aw = read_at<double>(m, mp + "/w"), as1 = read_at<double>(m, mp + "/s1"), amx = read_at<double>(m, mp + "/max"), amn = read_at<double>(m, mp + "/min");
      std::vector<double> mean(16);
      for (int i = 0; i < 16; ++i) { w[i] += aw[i]; s1[i] += as1[i]; mx[i] = std::max(mx[i], amx[i]); mn[i] = std::min(mn[i], amn[i]); mean[i] = aw[i] > 0 ? as1[i] / aw[i] : NAN; }
      means.push_back(mean);
    }
    const auto gw = read_at<double>(dsub, g + "/w"), gs1 = read_at<double>(dsub, g + "/s1"), gmx = read_at<double>(dsub, g + "/max"), gmn = read_at<double>(dsub, g + "/min");
    double worst = 0;
    for (int i = 0; i < 16; ++i) worst = std::max({worst, std::abs(gw[i] - w[i]), std::abs(gs1[i] - s1[i]), std::abs(gmx[i] - mx[i]), std::abs(gmn[i] - mn[i])});
    CHECK(worst < 1e-12);
    std::vector<std::int64_t> sh;
    const auto pm = read_at<double>(dsub, g + "/per_subject/mean", &sh);
    CHECK(sh == (std::vector<std::int64_t>{3, 16}));
    for (int s = 0; s < 3; ++s) for (int i = 0; i < 16; ++i) CHECK(std::isnan(means[s][i]) ? std::isnan(pm[s * 16 + i]) : std::abs(pm[s * 16 + i] - means[s][i]) < 1e-12 * (std::abs(means[s][i]) + 1));
    const auto fid = tdsub.find({.cls = task::Cls::field, .name = "group synthetic_map"});
    CHECK(fid->unit == "T");
    CHECK(testing::Probe(dsub).count("selection_measurement", {{"selection_id", dpart->id}, {"of_field_id", fid->id}}) == 6);
  }
  // a re-run REPLACES, a named subset sums only itself
  {
    const auto before = atlas_fields().size();
    pyramid_reduce(ds, ro);
    CHECK(atlas_fields().size() == before);
    PyramidReduceOptions two; two.subjects = {"S1", "S3"}; two.kernels = "MN_MEG";
    pyramid_reduce(ds, two);
    CHECK(testing::Probe(dsub).count("field_contribution", {{"field_id", tdsub.require({.cls = task::Cls::field, .name = "group rest__MN_MEG__constrained"}, "the group estimate").id}}) == 2);
    CHECK(!tdsub.find({.cls = task::Cls::field, .name = "group rest__WMNE_MEG__free"}).has_value());          // the selector left the free kernel out
    const auto dens = read_at<double>(dsub, node + "/time/rest__MN_MEG__constrained/samples");
    CHECK(dens.size() == 1);
  }
  // THE SAME REDUCE ON PACKAGES: the dataset split into a catalogue of subject packages, the members read each from its own package, the result written into a GROUP PACKAGE.
  // The group fields' arrays equal the one-database reduce's byte for byte, the rows say the same, the group package verifies, and a re-run replaces.
  {
    pyramid_reduce(ds, ro);                                                // the full cohort again (the subset above replaced it)
    SplitOptions so; so.registered_by = "test";
    split_dataset(pkg_root / "demo", so);
    const fs::path gd = pkg_root / "demo";
    CHECK(code_of([&] { pyramid_reduce_group(gd, ro, "", {}); }) == "invalid_argument");                       // the group package is named by the caller
    PyramidReduceOptions none; CHECK(code_of([&] { pyramid_reduce_group(gd, none, "grp", {}); }) == "invalid_argument");
    const PyramidReduceResult pr = pyramid_reduce_group(gd, ro, "grp", {});
    CHECK(pr.info["members"] == 3 && pr.info["depth"] == 3 && pr.info["leaves"] == 16 && pr.info["group"] == "grp");
    CHECK(pr.info["node"] == node);
    {
      DatasetWriter g = open_subject_package(gd, "grp");
      SubjectWriter gsub = SubjectWriter::open(g, "grp");
      std::size_t compared = 0;
      for (const char* what : {"/maps", "/time"}) {
        const auto keys_old = list_all_keys(dsub.store(), node + what), keys_new = list_all_keys(gsub.store(), node + what);
        CHECK(keys_old == keys_new && !keys_old.empty());
        for (const auto& k : keys_old) {
          const auto a = dsub.store().get(k), b = gsub.store().get(k);
          if (!a || !b || *a != *b) { std::printf("  differs: %s\n", k.c_str()); CHECK(false); }
          ++compared;
        }
      }
      std::printf("  group package: %zu keys equal the one-database reduce's, byte for byte\n", compared);
      const task::Subject tgsub = task::Subject::of(gsub);
      const auto fo = atlas_fields();
      const auto fn = tgsub.find_all({.cls = task::Cls::field});
      CHECK(fo.size() == 3 && fn.size() == fo.size());
      for (std::size_t i = 0; i < fo.size() && i < fn.size(); ++i) CHECK(fo[i].name == fn[i].name && fo[i].kind == fn[i].kind && fo[i].fact("function") == fn[i].fact("function"));
      std::int64_t dsub_contributions = 0;
      for (const task::Node& f : tdsub.find_all({.cls = task::Cls::field, .any_status = true})) dsub_contributions += testing::Probe(dsub).count("field_contribution", {{"field_id", f.id}});
      CHECK(testing::Probe(gsub).count("field_contribution") == dsub_contributions);
      CHECK(testing::Probe(gsub).integrity().empty());
      g.close();
    }
    CHECK(SubjectPackage::open(gd / "grp.nxr.zarr").verify().ok);
    const Catalogue cat = Catalogue::open(gd);
    CHECK(cat.find("grp") && cat.find("grp")->kind == "group" && cat.find("grp")->status == "registered");
    CHECK(group_member_status(gd, "grp").size() == 3);
    // the members' packages and the default subject's are not changed by it
    CHECK(refresh_member_status(gd).size() >= 5);
    for (const auto& e : cat.entries()) CHECK(e.status == "registered" || e.status == "replaced");
    // a re-run replaces; a subset names only its own members
    const PyramidReduceResult again = pyramid_reduce_group(gd, ro, "grp", {});
    CHECK(again.info["registered"] == "updated" || again.info["registered"] == "unchanged");
    PyramidReduceOptions sub2; sub2.subjects = {"S1", "S3"}; sub2.kernels = "MN_MEG";
    pyramid_reduce_group(gd, sub2, "grp2", {});
    CHECK(Catalogue::open(gd).find("grp2").has_value());
    CHECK(SubjectPackage::open(gd / "grp2.nxr.zarr").record().members.size() == 2);
  }
  if (keep) { pyramid_reduce(ds, ro); fs::remove_all(fs::path(keep) / "post"); fs::copy(root, fs::path(keep) / "post", fs::copy_options::recursive); }
  ds.close();
  fs::remove_all(pkg_root);
  return nxrtest::finish("test_pyramid_reduce");
}

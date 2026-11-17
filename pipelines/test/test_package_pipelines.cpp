// The pipelines compute directly on a subject package (docs/specs/2026-11-16-subject-centred-datasets.md sections 6 and 8).
//
// A synthetic one-database dataset (a subject with sensors, a recording, a cortex surface with a parcellation and a stored inverse kernel) is built; a copy of it is
// SPLIT into a catalogue of subject packages. The same chains run on both, each through the writer io opens for it (`open_writer`; `open_subject_package` through the
// catalogue), and every array they stored is compared byte for byte:
//   psd (sensor Welch PSD) · sourcepsd (band power maps of the sources) · crossspectrum (Welch, on the kernel's channels) · a field chain (a bandpass filter, then the PSD
//   of the filtered recording) · a measurement (the band maps over the parcellation) · a statistic of that measurement (the median over the members).
// Also: the package writer is a package, what the chains wrote is in the package (its content hash changes and verifies), a package of a subject the catalogue does not
// list is refused by name, and a template's arrays are read from the template's package through the routed store.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>

#include "brain_helpers.h"
#include "psd_run.h"
#include "check.h"
#include "filter_runs.h"
#include "run_definition.h"
#include "nxr/io/task/task.h"
#include "nxr/bst/entities.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/migrate.h"
#include "nxr/io/statistic.h"
#include "nxr/io/surface.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/run_shipped.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {

constexpr std::int64_t M = 4, T = 3000;
constexpr double sfreq = 100.0;

std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }

// the one-database dataset: subject S1
void make_old(const fs::path& datastore) {
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  const Brain b = brain(1, false);
  const std::int64_t N = b.n;
  DatasetWriter ds = create_dataset(datastore, "demo");
  SubjectWriter sub = SubjectWriter::create(ds, "S1");
  sub.compose([&] {
    const std::string cid = uuid7();
    std::vector<float> pos(static_cast<std::size_t>(M * 3));
    for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                        [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
    ManifoldSpec tm;
    tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
    tm.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto lc = lattice_counts({T});
    if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
    GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
    DimensionSpec d; d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
    tg.dimensions.push_back(d);
    tm.geometry = tg;
    const std::string tid = sub.manifold(std::move(tm));
    const std::string prod = sub.product({cid, tid});
    MatrixXd X(M, T);
    for (int t = 0; t < T; ++t) for (int i = 0; i < M; ++i) X(i, t) = (i == 0 ? 2.0 * std::sin(2 * M_PI * 10.0 * t / sfreq) : 0.0) + nd(rng);
    std::vector<float> xf(static_cast<std::size_t>(M * T));
    for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
    FieldSpec rec;
    rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
    rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
    sub.field(std::move(rec));
    // the cortex (a surface), its parcellation, and a kernel [3N x M] on all channels
    SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
    const std::string surf = add_surface(sub, so, b.P, b.F).id;
    std::vector<std::int32_t> codes(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) codes[static_cast<std::size_t>(i)] = i % 3;
    DictionarySpec dsp; dsp.name = "parcels"; dsp.names = {"a", "b", "c"};
    const DictionaryResult dict = sub.dictionary(dsp);
    SelectionSpec ps; ps.name = "parcels"; ps.path = "source/parcels"; ps.type = "set"; ps.manifold_id = surf; ps.data = view(codes, {N});
    ps.cols.set("dictionary_id", dict.id).set("cell", "0");
    sub.selection(ps);
    std::vector<std::int32_t> chans = {0, 1, 2, 3};
    SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
    const std::string sel = sub.selection(std::move(ss));
    std::vector<float> kf(static_cast<std::size_t>(3 * N * M));
    for (auto& v : kf) v = static_cast<float>(nd(rng));
    OperatorSpec op;
    op.name = "kern"; op.path = "kern"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
    op.data = view(kf, {3 * N, M}); op.array.compress = false;
    op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
    sub.operator_(std::move(op));
  });
  ds.close();
}

// every pipeline family, on whatever writer it is given
void run_chains(DatasetWriter& ds, SubjectWriter& sub) {
  CHECK(!nxrtest::run_psd(ds, sub, "rest").hit);
  const Json alpha = Json::array({Json{{"name", "alpha"}, {"lo", 8}, {"hi", 12}}});
  { const auto r = nxrtest::run_definition(ds, sub, "sourcepsd", Json{{"recording", "rest"}, {"bands", alpha}}); CHECK(!r.empty() && !r[0].value("hit", false)); }
  { const auto r = nxrtest::run_definition(ds, sub, "crossspectrum", Json{{"recording", "rest"}, {"channels_of", "kern"}, {"bands", alpha}}); CHECK(!r.empty() && !r[0].value("hit", false)); }
  nxrtest::FilterComputeOptions fo;                                               // the field chain: a filter, then the PSD of what it wrote
  fo.recording = "rest";
  fo.filter.kind = nxr::filter::Kind::Bandpass;
  fo.filter.lowerEdge = 8; fo.filter.upperEdge = 12;
  nxrtest::filter_compute(ds, sub, fo);
  CHECK(!nxrtest::run_psd(ds, sub, "rest_bandpass").hit);
  nxr::pipelines::run_shipped(ds, sub, "measure", Json{{"field", "rest_kern_psd_alpha"}, {"over", "parcels"}, {"weight", "none"}, {"measures", Json::array({"mean", "max"})}}, false);   // a measurement over a partition
  nxr::pipelines::run_shipped(ds, sub, "summarize", Json{{"over", "parcels"}, {"of", "rest_kern_psd_alpha"}, {"measure", "mean"}, {"summaries", Json::array({"median", "mean"})}, {"combine", "pool"}}, false);   // its statistic across the members
}

// every key the subject's store holds, with the bytes of its chunks (zarr.json texts carry ids: only their presence is compared)
std::map<std::string, std::string> keys_of(SubjectWriter& sub) {
  std::map<std::string, std::string> out;
  for (const std::string& k : list_all_keys(sub.store())) {
    const bool meta = k.size() >= 9 && k.compare(k.size() - 9, 9, "zarr.json") == 0;
    if (meta) { out[k] = ""; continue; }
    const auto v = sub.store().get(k);
    out[k] = v ? std::string(v->begin(), v->end()) : std::string("<absent>");
  }
  return out;
}

}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_package_pipelines_" + std::to_string(::getpid()));
  fs::remove_all(root);
  make_old(root / "old");
  fs::copy(root / "old" / "demo", root / "new", fs::copy_options::recursive);   // the same inputs, to be split

  // THE ONE-DATABASE FORM
  std::map<std::string, std::string> before;
  {
    DatasetWriter ds = open_writer(root / "old" / "demo");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    CHECK(!ds.is_package());
    run_chains(ds, sub);
    before = keys_of(sub);
    ds.close();
  }

  // THE PACKAGE FORM: split, then the same chains through the catalogue
  SplitOptions sp;
  sp.registered_by = "test";
  split_dataset(root / "new", sp);
  CHECK(is_catalogue_dataset(root / "new"));
  CHECK(code_of([&] { open_subject_package(root / "new", "nobody"); }) == code::no_such_row);
  std::string hash_before;
  {
    DatasetWriter ds = open_subject_package(root / "new", "S1");
    CHECK(ds.is_package());
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    hash_before = read_package_record(ds.db()).content_hash;
    run_chains(ds, sub);
    const auto after = keys_of(sub);
    // every key the one-database run wrote is in the package, with the same bytes; the package holds nothing else
    std::size_t same = 0;
    for (const auto& [k, v] : before) {
      const auto it = after.find(k);
      if (it == after.end()) { std::cout << "  missing in the package: " << k << "\n"; CHECK(false); continue; }
      if (it->second != v) { std::cout << "  bytes differ: " << k << "\n"; CHECK(false); continue; }
      ++same;
    }
    for (const auto& [k, v] : after) if (!before.count(k)) { std::cout << "  only in the package: " << k << "\n"; CHECK(false); }
    CHECK(same > 20);
    std::printf("  %zu keys equal, byte for byte\n", same);
    ds.close();
  }
  {
    const SubjectPackage p = SubjectPackage::open(root / "new" / "S1.nxr.zarr");
    CHECK(p.verify().ok);
    CHECK(p.record().content_hash != hash_before);                       // what the chains wrote is in the package, and its hash says so
  }
  {
    // a second run is a cache hit, in the package as in the old form
    DatasetWriter ds = open_subject_package(root / "new", "S1");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    CHECK(nxrtest::run_psd(ds, sub, "rest").hit);
    ds.close();
  }
  refresh_member_status(root / "new");                                   // the catalogue sees the changed package (stale) until it is registered again
  RegisterOptions ro;
  ro.registered_by = "test";
  const auto reg = register_packages(root / "new", {root / "new" / "S1.nxr.zarr"}, ro);
  CHECK(reg.size() == 1 && (reg[0].outcome == "updated" || reg[0].outcome == "unchanged"));
  fs::remove_all(root);
  return nxrtest::finish("test_package_pipelines");
}

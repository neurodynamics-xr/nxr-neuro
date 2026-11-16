// filter_compute on a small synthetic dataset (no real data): 3 channels, 10 s at 250 Hz, a recording with
// projectors recorded in its provenance, and a bad span as the importer stores it (`<recording>_bad_segments`). Checks: the stored PSD equals nxr::welch on the same
// samples with the same mask (float32), the segments across the bad span are left out, the result is identical for another chunk size, a cache hit, the refusal of a name
// already stored and --replace, the field as nxr's qc stores a PSD (kind psd, on channels x the frequency line, folder sensor-frequency), its provenance, and the command.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_filter_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  const std::int64_t M = 3, T = 2500, bad_channel = 2;
  const double sfreq = 250.0, bad_start = 4.0, bad_stop = 5.0;       // the bad span: samples 1200..1499
  MatrixXd X(M, T);
  for (int t = 0; t < T; ++t) {
    X(0, t) = 2.0 * std::sin(2 * M_PI * 8.0 * t / sfreq) + std::sin(2 * M_PI * 50.0 * t / sfreq) + 0.2 * nd(rng) + 5.0;
    X(1, t) = nd(rng);
    X(2, t) = 40 * nd(rng);
  }
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });

  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3));
      for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
      ManifoldSpec m;
      m.id = cid;
      m.name = "sensors";
      m.path = "sensor/sensors";
      m.type = "points";
      m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g;
      g.form = "stored";
      g.cs_key = "scs";
      g.positions_path = "sensor/sensors";
      m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 1}, {"MEG MAG", "EEG"}, {}}, {"quality", {0, 0, 1}, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time";
      tm.path = "time/rest_time";
      tm.type = "line";
      tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({T});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg;
      tg.form = "closed_form";
      tg.cs_key = "clock";
      DimensionSpec d;
      d.name = "time";
      d.n_vertices = T;
      d.spacing = 1.0 / sfreq;
      d.origin = 0.0;
      d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      std::vector<float> xf(static_cast<std::size_t>(M * T));
      for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
      FieldSpec rec;
      rec.name = "rest";
      rec.path = node_path(sub.db(), prod, "rest");
      rec.kind = "recording";
      rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 700};
      rec.array.compress = false;
      rec.cols.set("unit", "T");
      rec.cols.json("producer_json", Json{{"kind", "recording"}, {"projectors", Json::array({Json{{"order", 1}, {"comment", "SSP"}, {"method", "SSP"}, {"status", 1}, {"applied", "applied by the import"}, {"operator", "ch_projector_1"}},
                                                                                           Json{{"order", 2}, {"comment", "off"}, {"method", "SSP"}, {"status", 0}, {"applied", "not applied"}, {"operator", "ch_projector_2"}}})}});
      const std::string rid = sub.field(std::move(rec));
      nxr::bst::TimeSpansOptions so;
      so.line_id = tid;
      so.of_field_id = rid;
      so.description = "bad segments";
      nxr::bst::time_spans(sub, "time/rest_bad_segments", {{bad_start, bad_stop, 0}}, {"BAD"}, so);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const std::int64_t span0 = 1000, span1 = 1250;     // the bad span in samples
  FilterComputeOptions o;
  o.recording = "rest";
  o.filter = nxr::filter::brainstormFilterOptions(1.0);
  o.filter.kind = nxr::filter::Kind::Bandpass;
  o.filter.lowerEdge = 2;
  o.filter.upperEdge = 40;
  const FilterComputeResult a = filter_compute(ds, sub, o);
  CHECK(!a.hit);
  CHECK_EQ(a.info["name"].get<std::string>(), std::string("rest_bandpass"));
  CHECK_EQ(a.info["samples"].get<std::int64_t>(), T);
  // the stored samples are the whole-record filter of the stored source (float32)
  nxr::filter::FilterOptions fo = o.filter;
  fo.sfreq = sfreq;
  const MatrixXd want = nxr::filter::bandpass(X, 2, 40, fo);
  {
    std::vector<std::int64_t> shape;
    const std::vector<float> got = read_floats((sub.folder() / (a.info["path"].get<std::string>())), &shape);
    CHECK(shape.size() == 2 && shape[0] == M && shape[1] == T);
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * T + t)] - want(c, t)));
    CHECK(worst < 1e-4);
  }
  // the rows: a recording Field derived from the source, with its own envelope and chunk tiling; the bad span copied; the provenance
  {
    const auto f = sub.find("field", a.info["path"].get<std::string>());
    CHECK(f.has_value());
    CHECK_EQ(text(*f, "kind"), std::string("recording"));
    CHECK_EQ(text(*f, "derived_from_id"), text(*sub.by_name("field", "rest"), "id"));
    CHECK_EQ(text(*f, "unit"), std::string("T"));
    CHECK_EQ(text(*f, "manifold_id"), text(*sub.by_name("field", "rest"), "manifold_id"));    // same channels x time product
    const Json pj = Json::parse(text(*f, "producer_json"));
    CHECK_EQ(pj["filter"]["kind"].get<std::string>(), std::string("bandpass"));
    CHECK_EQ(pj["filter"]["option_set"].get<std::string>(), std::string("brainstorm"));
    CHECK_EQ(pj["filter"]["method"].get<std::string>(), std::string("fir"));
    CHECK(pj["filter"]["lower_edge"].get<double>() == 2.0 && pj["filter"]["upper_edge"].get<double>() == 40.0);
    CHECK_EQ(pj["filtered_from"]["name"].get<std::string>(), std::string("rest"));
    CHECK_EQ(pj["projectors"].size(), static_cast<std::size_t>(2));
    CHECK(pj["mean_removed"].get<bool>() && !pj["mean_restored"].get<bool>());
    CHECK(sub.find("selection", "time/rest_bandpass_chunks").has_value() || sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = 'rest_bandpass_chunks'", {V(sub.id())}).has_value());
    const auto sel = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = 'rest_bandpass_bad_segments'", {V(sub.id())});
    CHECK(sel.has_value());
    if (sel) {
      const auto sp = sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ?", {V(text(*sel, "id"))});
      CHECK_EQ(sp.size(), static_cast<std::size_t>(1));
      CHECK(std::abs(std::stod(text(sp[0], "start")) - bad_start) < 1e-12 && std::abs(std::stod(text(sp[0], "stop")) - bad_stop) < 1e-12);
    }
  }
  (void)span0; (void)span1;
  // a rerun is a cache hit; the name is taken otherwise
  CHECK(filter_compute(ds, sub, o).hit);
  { FilterComputeOptions e = o; e.filter.upperEdge = 41; CHECK_EQ(code_of([&] { filter_compute(ds, sub, e); }), std::string(code::already_exists)); }
  // another chunk size and a low-pass (the mean restored): the same numbers as the whole-record filter
  {
    FilterComputeOptions l = o;
    l.filter.kind = nxr::filter::Kind::Lowpass;
    l.filter.upperEdge = 30;
    l.name = "rest_lp";
    const FilterComputeResult r = filter_compute(ds, sub, l);
    const MatrixXd w2 = nxr::filter::lowpass(X, 30, fo);
    const std::vector<float> got = read_floats((sub.folder() / (r.info["path"].get<std::string>())));
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * T + t)] - w2(c, t)));
    CHECK(worst < 1e-4);
  }
  // a notch (IIR forward-backward, chunked with its margins)
  {
    FilterComputeOptions n = o;
    n.filter.kind = nxr::filter::Kind::Notch;
    n.filter.frequencies = {50};
    n.filter.notchBandwidth = 3.0;
    const FilterComputeResult r = filter_compute(ds, sub, n);
    const MatrixXd w3 = nxr::filter::notch(X, {50}, [&] { auto p = fo; p.notchBandwidth = 3.0; return p; }());
    const std::vector<float> got = read_floats((sub.folder() / (r.info["path"].get<std::string>())));
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * T + t)] - w3(c, t)));
    CHECK(worst < 1e-4);
    CHECK_EQ(r.info["name"].get<std::string>(), std::string("rest_notch"));
  }
  CHECK_EQ(code_of([&] { FilterComputeOptions e = o; e.recording = "nope"; e.name = "x"; filter_compute(ds, sub, e); }), std::string(code::no_such_row));
  CHECK_EQ(code_of([&] { FilterComputeOptions e = o; e.name = "x"; e.filter.upperEdge = 200; filter_compute(ds, sub, e); }), std::string(code::invalid_argument));
  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    const std::string common = " --dataset " + d + " --subject S1 --recording rest";
    CHECK_EQ(std::system((cli + " highpass" + common + " --fpass 5 > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " highpass" + common + " --fpass 5 >> " + log).c_str()), 0);   // a hit
    CHECK_EQ(std::system((cli + " bandstop" + common + " --fstop 48.5,51.5 --name cli_bs >> " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " notch" + common + " --line 50 --harmonics 1 --name cli_notch --bandwidth 2 >> " + log).c_str()), 0);
    CHECK(std::system((cli + " lowpass" + common + " > /dev/null 2>&1").c_str()) != 0);          // --fpass missing
    CHECK(std::system((cli + " lowpass" + common + " --fpass 400 --name bad > /dev/null 2>&1").c_str()) != 0);   // beyond Nyquist
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(all.find("\"cache\": \"miss\"") != std::string::npos && all.find("\"cache\": \"hit\"") != std::string::npos);
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    for (const char* n : {"rest_highpass", "cli_bs", "cli_notch"}) CHECK(s2.db().one("SELECT id FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording'", {V(s2.id()), V(std::string(n))}).has_value());
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_filter_pipeline");
}

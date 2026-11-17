// psd_compute on a small synthetic dataset built in a temp directory (no real data): 3 channels (one bad), a recording of known signals stored in 700-sample chunks with
// projectors recorded in its provenance, and a bad span as the importer stores it (`<recording>_bad_segments`). Checks: the stored PSD equals nxr::welch on the same
// samples with the same mask (float32), the segments across the bad span are left out, the result is identical for another chunk size, a cache hit, the refusal of a name
// already stored and --replace, the PSD as a STATISTIC (power_density by welch, from the segment partition, axes channel x the frequency line), its provenance, and the command.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/io/task/task.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/rotary.h"

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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_psd_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  const std::int64_t M = 3, T = 4000, bad_channel = 2;
  const double sfreq = 100.0, bad_start = 12.0, bad_stop = 15.0;       // the bad span: samples 1200..1499
  MatrixXd X(M, T);
  for (int t = 0; t < T; ++t) {
    X(0, t) = 2.0 * std::sin(2 * M_PI * 8.0 * t / sfreq) + 0.2 * nd(rng) + 5.0;
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
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 1}, {"MEG MAG", "EEG"}, {}}, {"unit", {0, 0, 1}, {"T", "V"}, {}}, {"quality", {0, 0, 1}, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
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
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(700)), std::make_pair(std::string("rest_b"), std::int64_t(1777))}) {   // the same samples in two chunk widths
        FieldSpec rec;
        rec.name = name;
        rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name);
        rec.kind = "recording";
        rec.manifold_id = prod;
        rec.data = view(xf, {M, T});
        rec.array.chunks = Shape{M, width};
        rec.array.compress = false;
        if (name == "rest") rec.cols.set("unit", "T");                                           // rest_b has none: its channels' units say (T, T, V: several)
        rec.cols.json("producer_json", Json{{"kind", "recording"}, {"projectors", Json::array({Json{{"order", 1}, {"comment", "SSP"}, {"method", "SSP"}, {"status", 1}, {"applied", "applied by the import"}, {"operator", "ch_projector_1"}},
                                                                                             Json{{"order", 2}, {"comment", "off"}, {"method", "SSP"}, {"status", 0}, {"applied", "not applied"}, {"operator", "ch_projector_2"}}})}});
        const std::string rid = sub.field(std::move(rec));
        nxr::bst::TimeSpansOptions so;
        so.line_id = tid;
        so.of_field_id = rid;
        so.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad_start, bad_stop, 0}}, {"BAD"}, so);
      }
      // two stored Operators over the three channels: a rotation plane [2 x 3] (rows fx, fy) and a [3 x 3] that is no pair
      std::vector<std::int32_t> chans = {0, 1, 2};
      SelectionSpec ss; ss.name = "plane_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string psel = sub.selection(std::move(ss));
      for (const auto& [oname, nrows] : {std::make_pair(std::string("plane"), std::int64_t(2)), std::make_pair(std::string("not_a_pair"), std::int64_t(3))}) {
        std::vector<float> pf = {0.5f, -0.2f, 0.1f, 0.3f, 0.8f, -0.05f, 1.0f, 1.0f, 1.0f};
        OperatorSpec op;
        op.name = oname; op.path = oname; op.kind = "rotation plane"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = nrows; op.n_cols = M;
        op.data = view(pf, {nrows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", cid).set("from_selection_id", psel);
        op.cols.json("params_json", Json{{"band", Json::array({8.0, 12.0})}});
        sub.operator_(std::move(op));
      }
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");

  // the expected PSD: nxr::welch on the stored samples, the mask from the span
  std::vector<std::uint8_t> good(T, 1);
  for (int i = 0; i < T; ++i) { const double t = i / sfreq; if (t >= bad_start && t < bad_stop) good[static_cast<std::size_t>(i)] = 0; }
  auto expect = nxr::welch::brainstormPsdOptions(sfreq, 2.0);
  const nxr::welch::WelchResult want = nxr::welch::welch(X, expect, &good);
  const nxr::welch::WelchResult unmasked = nxr::welch::welch(X, expect);

  PsdOptions o;
  o.recording = "rest";
  o.option_set = "brainstorm";                                  // the named set that reproduces Brainstorm (2 s = round(2 fs) samples, from the span's start), to the bit
  o.segment_seconds = 2.0;
  o.chunks = "all";                                             // every chunk (stored 700 wide): the whole recording, read chunk by chunk
  const PsdResult a = psd_compute(ds, sub, o);
  CHECK(!a.hit);
  CHECK_EQ(a.info["segments"].get<std::int64_t>(), want.segments);
  CHECK(a.info["segments"].get<std::int64_t>() < unmasked.segments);
  CHECK_EQ(a.info["segments_skipped"].get<std::int64_t>(), want.segmentsSkipped);
  CHECK_EQ(a.info["channels"].get<std::int64_t>(), M);                                        // all channels, the bad one too
  CHECK_EQ(a.info["frequencies"].get<std::int64_t>(), 101);
  const std::string path = a.info["path"].get<std::string>();
  CHECK_EQ(path, std::string("sensor-time/rest_statistics/rest_psd_welch"));   // a statistic beside its recording (docs/specs/2026-11-13-statistics.md)
  {
    std::vector<std::int64_t> shape;
    const std::vector<float> got = read_floats((sub.folder() / (path)), &shape);
    CHECK(shape.size() == 2 && shape[0] == M && shape[1] == 101);
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int k = 0; k < 101; ++k) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * 101 + k)] / want.psd(c, k) - 1));
    CHECK(worst < 1e-6);                                                                      // float32 storage
  }
  // THE STATISTIC (docs/specs/2026-11-13-statistics.md): power_density by welch, OF the recording, from its segment partition (the segments its members),
  // axes channel x the frequency line, its options and provenance
  {
    const StoredStatistic st = read_statistic(sub, "rest_psd_welch");
    CHECK_EQ(text(st.row, "measure"), std::string("power_density"));
    CHECK_EQ(text(st.row, "method"), std::string("welch"));
    CHECK_EQ(text(st.row, "unit"), std::string("T\xc2\xb2/Hz"));
    CHECK_EQ(text(st.row, "of_field_id"), text(*sub.by_name("field", "rest"), "id"));
    CHECK_EQ(integer(st.row, "n_members"), want.segments);
    CHECK_EQ(integer(st.row, "members_skipped"), want.segmentsSkipped);
    CHECK(st.axes.size() == 2 && st.axes[0]["name"] == "channel" && st.axes[1]["name"] == "frequency" && st.shape == (std::vector<std::int64_t>{M, 101}));
    const auto line = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::manifold, st.axes[1]["manifold_id"].get<std::string>());
    CHECK(line.has_value());
    const auto line_axis = nxr::io::task::Subject::of(sub).geometry(line->id).axes.at(0);
    CHECK_EQ(line->name, std::string("rest_freq"));
    CHECK_EQ(*line_axis.unit, std::string("Hz"));
    CHECK(std::abs(*line_axis.spacing - 0.5) < 1e-12);
    // the partition: Welch's segments as a closed-form spans Selection on the time line (200 samples, a new one every 100)
    const auto seg = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::selection, text(st.row, "partition_selection_id"));
    CHECK(seg && seg->type == "spans" && seg->name == "rest_psd_welch_segments");
    const Json tp = seg->params;
    CHECK(std::abs(tp["tiling"]["width"].get<double>() - 2.0) < 1e-12 && std::abs(tp["tiling"]["hop"].get<double>() - 1.0) < 1e-12 && tp["tiling"]["count"] == want.segments + want.segmentsSkipped);
    CHECK(st.options["window"] == "hamming" && st.options["window_form"] == "symmetric" && st.options["detrend"] == "constant" && st.options["scaling"] == "density");
    CHECK(st.span["start_sample"] == 0 && st.span["stop_sample"] == 4000);
    const Json& pj = st.producer;
    CHECK_EQ(pj["bad_segments"]["n_spans"].get<std::int64_t>(), 1);
    CHECK_EQ(pj["projectors"].size(), static_cast<std::size_t>(2));
    CHECK_EQ(pj["projectors_applied_in_samples"][0].get<std::string>(), std::string("ch_projector_1"));
    CHECK_EQ(pj["projectors_not_applied"][0].get<std::string>(), std::string("ch_projector_2"));
    CHECK(std::abs(pj["excluded_fraction"].get<double>() - 300.0 / 4000.0) < 1e-12);
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int k = 0; k < 101; ++k) worst = std::max(worst, std::abs(st.values[static_cast<std::size_t>(c * 101 + k)] / want.psd(c, k) - 1));
    CHECK(worst < 1e-6);                                                                      // float32 storage, read back through read_statistic
  }
  // the same samples stored in another chunk width (1777, read chunk by chunk): the same numbers to the last bit
  {
    PsdOptions o2 = o;
    o2.recording = "rest_b";
    const PsdResult b = psd_compute(ds, sub, o2);
    CHECK(!b.hit);
    CHECK(read_floats((sub.folder() / (a.info["path"].get<std::string>()))) == read_floats((sub.folder() / (b.info["path"].get<std::string>()))));
  }
  // a recording with no unit of its own, channels in T and V: "channel unit^2/Hz" on the row, each type's unit in the provenance
  {
    const StoredStatistic sb = read_statistic(sub, "rest_b_psd_welch");
    CHECK_EQ(text(sb.row, "unit"), std::string("channel unit\xc2\xb2/Hz"));
    CHECK_EQ(sb.producer["channel_units"]["MEG MAG"].get<std::string>(), std::string("T\xc2\xb2/Hz"));
    CHECK_EQ(sb.producer["channel_units"]["EEG"].get<std::string>(), std::string("V\xc2\xb2/Hz"));
  }
  // a hit; a name already stored with other settings is refused, then replaced on request
  CHECK(psd_compute(ds, sub, o).hit);
  {
    PsdOptions o3 = o;
    o3.welch.window = nxr::welch::Window::Hann;
    CHECK_EQ(code_of([&] { psd_compute(ds, sub, o3); }), std::string(code::already_exists));
    o3.replace = true;
    const PsdResult r = psd_compute(ds, sub, o3);
    CHECK(!r.hit && r.info["replaced"].get<bool>());
  }
  // without the bad segments the whole grid counts
  {
    PsdOptions o4 = o;
    o4.name = "rest_psd_all";
    o4.exclude_bad_segments = false;
    CHECK_EQ(psd_compute(ds, sub, o4).info["segments"].get<std::int64_t>(), unmasked.segments);
  }
  // refusals: a window shorter than one segment, a segment longer than the window
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "x"; e.chunks.reset(); e.start = 0.0; e.stop = 1.0; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  // the span is always stated: none is refused; "0" is the first chunk's samples (0..699: six 2-s segments at 50 % overlap); a chunk the recording lacks (it has 0..5),
  // a malformed one, half an explicit span, or both forms are refused
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "c0n"; e.chunks.reset(); psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  { PsdOptions e = o; e.name = "c0"; e.chunks = "0"; const PsdResult r = psd_compute(ds, sub, e); CHECK(!r.hit); CHECK_EQ(r.info["segments"].get<std::int64_t>(), 6); }
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "c1"; e.chunks = "6"; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "c2"; e.chunks = "x"; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "c3"; e.chunks.reset(); e.stop = 2.0; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "c4"; e.start = 0.0; e.stop = 2.0; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "y"; e.welch.overlap = 1.0; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { PsdOptions e = o; e.recording = "nope"; psd_compute(ds, sub, e); }), std::string(code::no_such_row));
  // THE ROTARY SPECTRUM (`--through`): the stored Operator [2 x 3] applied as z = fx.X + i fy.X, its TWO-SIDED Welch PSD over the same segments and mask, stored as a
  // statistic on axes [pair (1), signed frequency]; the chunk width does not matter; the mirrored operator (fy negated) mirrors the spectrum
  {
    Eigen::MatrixXd A(2, 3);
    A << 0.5f, -0.2f, 0.1f, 0.3f, 0.8f, -0.05f;                                              // as stored (float32)
    const Eigen::MatrixXcd z = nxr::rotary::pairSeries(A, X);
    const nxr::welch::ComplexWelchResult rw = nxr::welch::complexWelch(z, expect, &good);
    PsdOptions r; r.recording = "rest"; r.through = "plane"; r.option_set = "brainstorm"; r.segment_seconds = 2.0; r.chunks = "all";
    const PsdResult ra = psd_compute(ds, sub, r);
    CHECK(!ra.hit);
    CHECK_EQ(ra.info["path"].get<std::string>(), std::string("sensor-time/rest_statistics/rest_plane_psd_welch"));
    CHECK_EQ(ra.info["frequencies"].get<std::int64_t>(), 200);
    CHECK_EQ(ra.info["segments"].get<std::int64_t>(), rw.segments);
    CHECK_EQ(ra.info["segments"].get<std::int64_t>(), want.segments);                      // the same segment partition as the sensor PSD
    const StoredStatistic st = read_statistic(sub, "rest_plane_psd_welch");
    CHECK_EQ(text(st.row, "measure"), std::string("power_density"));
    CHECK_EQ(text(st.row, "method"), std::string("welch"));
    CHECK_EQ(text(st.row, "of_field_id"), text(*sub.by_name("field", "rest"), "id"));
    CHECK_EQ(integer(st.row, "n_members"), want.segments);
    CHECK(st.axes.size() == 2 && st.axes[0]["name"] == "pair" && st.axes[0]["size"] == 1 && st.axes[1]["name"] == "frequency" && st.shape == (std::vector<std::int64_t>{1, 200}));
    CHECK(st.options["input"]["operator"] == "plane" && st.options["input"]["band"] == Json::array({8.0, 12.0}) && st.options["sides"] == "two");
    const auto line = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::manifold, st.axes[1]["manifold_id"].get<std::string>());
    CHECK(line.has_value());
    const auto line_axis = nxr::io::task::Subject::of(sub).geometry(line->id).axes.at(0);
    CHECK_EQ(line->name, std::string("rest_freq_signed"));
    CHECK(std::abs(*line_axis.origin + 50.0) < 1e-9 && std::abs(*line_axis.spacing - 0.5) < 1e-12);   // -fs/2, 0.5 Hz
    double worst = 0;
    for (int k = 0; k < 200; ++k) worst = std::max(worst, std::abs(st.values[static_cast<std::size_t>(k)] / rw.psd(0, k) - 1));
    CHECK(worst < 1e-6);
    // chunk width (stored 1777 wide): the same numbers to the last bit
    PsdOptions rb = r; rb.recording = "rest_b";
    CHECK(!psd_compute(ds, sub, rb).hit);
    CHECK(read_floats(sub.folder() / ra.info["path"].get<std::string>()) == read_floats(sub.folder() / psd_compute(ds, sub, rb).info["path"].get<std::string>()));
    // the sense: fy negated is the mirror spectrum (P'(f) = P(-f)); +f is the operator's x -> y
    Eigen::MatrixXd Am = A; Am.row(1) *= -1;
    const nxr::welch::ComplexWelchResult mw = nxr::welch::complexWelch(nxr::rotary::pairSeries(Am, X), expect, &good);
    for (int k = 1; k < 200; ++k) CHECK(std::abs(mw.psd(0, k) / rw.psd(0, 200 - k) - 1) < 1e-9);
    CHECK(psd_compute(ds, sub, r).hit);
    CHECK_EQ(code_of([&] { PsdOptions e = r; e.through = "nope"; psd_compute(ds, sub, e); }), std::string(code::no_such_row));
    CHECK_EQ(code_of([&] { PsdOptions e = r; e.name = "x"; e.through = "not_a_pair"; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  }
  // THE TILE SET (the default since 2026-11-13, the time-tile amendment): a segment of 2^k samples, k = round(log2(2 s x 100 Hz)) = 8 (256 samples), half
  // overlapping, Brainstorm's taper, detrend and density: nxr::welch on 256-sample segments of the same samples and mask; a span in seconds snaps to level 8
  // ([3, 33) s = samples [300, 3300) -> [256, 3328)); --segment-level states the level; an overlap that leaves the hop no tile is refused, as is
  // --segment-samples that is no power of two, and --segment-level with another set
  {
    PsdOptions t; t.recording = "rest"; t.name = "tiles"; t.segment_seconds = 2.0; t.chunks = "all";
    CHECK_EQ(t.option_set, std::string("tiles"));
    const PsdResult r = psd_compute(ds, sub, t);
    auto eo = nxr::welch::brainstormPsdOptions(sfreq, 2.0);
    eo.segmentSamples = 256;
    const nxr::welch::WelchResult tw = nxr::welch::welch(X, eo, &good);
    CHECK(r.info["segment_samples"] == 256 && r.info["segments"] == tw.segments && r.info["frequencies"] == 129);
    CHECK((r.psd.psd - tw.psd).cwiseAbs().maxCoeff() < 1e-12 * tw.psd.cwiseAbs().maxCoeff());
    PsdOptions sp = t; sp.name = "tiles_span"; sp.chunks.reset(); sp.start = 3.0; sp.stop = 33.0;
    CHECK(psd_compute(ds, sub, sp).info["window"] == Json::array({256, 3328}));
    PsdOptions lv = t; lv.name = "tiles_l7"; lv.segment_seconds.reset(); lv.segment_level = 7;
    CHECK(psd_compute(ds, sub, lv).info["segment_samples"] == 128);
    CHECK_EQ(code_of([&] { PsdOptions e = t; e.name = "t1"; e.welch.overlap = 0.3; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { PsdOptions e = t; e.name = "t2"; e.segment_samples = 200; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { PsdOptions e = o; e.name = "t3"; e.segment_level = 8; psd_compute(ds, sub, e); }), std::string(code::invalid_argument));
  }

  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    CHECK_EQ(std::system((cli + " run psd --dataset " + d + " --subject S1 --recording rest --name cli_psd --set scipy --segment-seconds 2 --window hann --detrend linear --chunks 0 > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " run psd --dataset " + d + " --subject S1 --recording rest --name cli_psd --set scipy --segment-seconds 2 --window hann --detrend linear --chunks 0 >> " + log).c_str()), 0);   // a hit
    CHECK(std::system((cli + " run psd --dataset " + d + " --subject S1 --recording rest --set bogus --chunks 0 > /dev/null 2>&1").c_str()) != 0);
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(all.find("\"cache\": \"miss\"") != std::string::npos && all.find("\"cache\": \"hit\"") != std::string::npos);
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    const StoredStatistic st = read_statistic(s2, "cli_psd");
    CHECK_EQ(st.options["option_set"].get<std::string>(), std::string("scipy"));
    CHECK_EQ(st.options["window_form"].get<std::string>(), std::string("periodic"));
    CHECK_EQ(st.options["detrend"].get<std::string>(), std::string("linear"));
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_psd_pipeline");
}

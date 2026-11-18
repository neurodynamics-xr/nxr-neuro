// the `psd` and `rotary_psd` definitions, run by the runner on a small synthetic dataset built in a temp directory (no real data): 3 channels (one bad), a recording of known signals stored in 700-sample chunks with
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
#include "nxr/io/statistic.h"
#include "psd_run.h"
#include "nxr/rotary.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {
std::string refusal(const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
bool has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }
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
  auto expect = nxr::signal::spectral::brainstormPsdOptions(sfreq, 2.0);
  const nxr::signal::spectral::WelchResult want = nxr::signal::spectral::welch(X, expect, &good);
  const nxr::signal::spectral::WelchResult unmasked = nxr::signal::spectral::welch(X, expect);

  const auto stat = [&](const std::string& n) { return read_statistic(sub, n); };
  const double tol = 1e-6;                                                                      // float32 storage
  const Json two_s = Json{{"segment_seconds", 2.0}};

  // THE PRESET `brainstorm` (2 s = round(2 fs) samples from the span's start, a symmetric Hamming window): nxr::welch on the stored samples with the bad span's mask, to float32
  const nxrtest::PsdRun a = nxrtest::run_psd(ds, sub, "rest", two_s, "brainstorm");              // every chunk (stored 700 wide): the whole recording, read chunk by chunk
  CHECK(!a.hit);
  CHECK(want.segments < unmasked.segments && want.segmentsSkipped > 0);
  CHECK_EQ(a.path, std::string("sensor-time/rest_statistics/rest_psd_welch"));                    // a statistic beside its recording (docs/specs/2026-11-13-statistics.md)
  {
    std::vector<std::int64_t> shape;
    const std::vector<float> got = nxrtest::psd_values(sub, a, &shape);
    CHECK(shape.size() == 2 && shape[0] == M && shape[1] == 101);                                // all channels, the bad one too
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int k = 0; k < 101; ++k) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * 101 + k)] / want.psd(c, k) - 1));
    CHECK(worst < tol);
  }
  // THE STATISTIC: power_density by welch, OF the recording, from its segment partition (the segments its members), axes channel x the frequency line, its options and provenance
  {
    const StoredStatistic st = stat("rest_psd_welch");
    CHECK_EQ(text(st.row, "measure"), std::string("power_density"));
    CHECK_EQ(text(st.row, "method"), std::string("welch"));
    CHECK_EQ(text(st.row, "unit"), std::string("T\xc2\xb2/Hz"));
    CHECK_EQ(text(st.row, "of_field_id"), text(*sub.by_name("field", "rest"), "id"));
    CHECK_EQ(integer(st.row, "n_members"), want.segments);
    CHECK_EQ(integer(st.row, "members_skipped"), want.segmentsSkipped);
    CHECK(st.axes.size() == 2 && st.axes[0]["name"] == "channel" && st.axes[1]["name"] == "frequency" && st.shape == (std::vector<std::int64_t>{M, 101}));
    CHECK_EQ(st.axes[0]["manifold_id"].get<std::string>(), text(*sub.by_name("manifold", "sensors"), "id"));   // the channel axis lies on the recording's channels
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
    CHECK(std::abs(tp["tiling"]["origin"].get<double>()) < 1e-12 && tp["segments_skipped"] == want.segmentsSkipped);
    CHECK(st.options["window"] == "hamming" && st.options["window_form"] == "symmetric" && st.options["detrend"] == "constant" && st.options["scaling"] == "density");
    CHECK(st.options["segment_rule"] == "round" && st.options["exclude_bad_segments"] == true);
    CHECK(st.span["start_sample"] == 0 && st.span["stop_sample"] == 4000);
    CHECK(st.scaling.is_object() && st.scaling["stored"] == "density");                         // the scaling record a reader converts with
    CHECK(st.producer["pipeline"]["name"] == "psd" && st.producer["output"] == "psd" && st.producer.contains("cache"));
    CHECK(std::abs(st.producer["frequency_resolution_hz"].get<double>() - 0.5) < 1e-12 && st.producer["sfreq"] == sfreq && st.producer["segments"] == want.segments);   // what the readers ask of it
    CHECK(std::abs(st.producer["excluded_fraction"].get<double>() - 300.0 / 4000.0) < 1e-12 && st.producer["spectral_scaling"] == st.scaling);
    double worst = 0;
    for (int c = 0; c < M; ++c) for (int k = 0; k < 101; ++k) worst = std::max(worst, std::abs(st.values[static_cast<std::size_t>(c * 101 + k)] / want.psd(c, k) - 1));
    CHECK(worst < tol);
  }
  // the same samples stored in another chunk width (1777): the same numbers to the last bit; a recording with no unit of its own (channels in T and V): "channel unit^2/Hz"
  {
    const nxrtest::PsdRun b = nxrtest::run_psd(ds, sub, "rest_b", two_s, "brainstorm");
    CHECK(!b.hit);
    CHECK(nxrtest::psd_values(sub, a) == nxrtest::psd_values(sub, b));
    CHECK_EQ(text(stat("rest_b_psd_welch").row, "unit"), std::string("channel unit\xc2\xb2/Hz"));
  }
  // a rerun is a hit; the same name with other settings is refused, then replaced on request; the preset's name is in the key
  CHECK(nxrtest::run_psd(ds, sub, "rest", two_s, "brainstorm").hit);
  CHECK(has(refusal([&] { nxrtest::run_psd(ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"window", "hann"}}, "brainstorm"); }), "already exists"));
  CHECK(!nxrtest::run_psd(ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"window", "hann"}}, "brainstorm", true).hit);
  // without the bad segments the whole grid counts
  {
    const nxrtest::PsdRun r = nxrtest::run_psd(ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"name", "rest_psd_all"}, {"bad_segments", "ignore"}}, "brainstorm");
    CHECK_EQ(integer(stat("rest_psd_all").row, "n_members"), unmasked.segments);
    CHECK_EQ(integer(stat("rest_psd_all").row, "members_skipped"), static_cast<std::int64_t>(0));
    (void)r;
  }
  // the span is always stated: none is refused; "0" is the first chunk's samples (0..699: six 2-s segments at 50 % overlap); a chunk the recording lacks (it has 0..5), a malformed one,
  // half an explicit span, or both forms are refused; a span shorter than one segment is refused
  auto with_span = [&](const std::string& name, const Json& set, const std::function<void(nxr::pipelines::graph::RunOptions&)>& tweak, const std::string& preset = "brainstorm") {
    Json s2 = set;
    s2["recording"] = "rest"; s2["name"] = name;
    nxr::pipelines::graph::RunOptions ro;
    ro.preset = preset;
    tweak(ro);
    return nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load("psd"), s2, ro);
  };
  CHECK(!refusal([&] { with_span("c0n", two_s, [](auto&) {}); }).empty());
  { const auto r = with_span("c0", two_s, [](auto& ro) { ro.chunks = "0"; }); CHECK(!r[0]["hit"].get<bool>()); CHECK_EQ(integer(stat("c0").row, "n_members"), static_cast<std::int64_t>(6)); }
  CHECK(!refusal([&] { with_span("c1", two_s, [](auto& ro) { ro.chunks = "6"; }); }).empty());
  CHECK(!refusal([&] { with_span("c2", two_s, [](auto& ro) { ro.chunks = "x"; }); }).empty());
  CHECK(!refusal([&] { with_span("c3", two_s, [](auto& ro) { ro.stop = 2.0; }); }).empty());
  CHECK(!refusal([&] { with_span("c4", two_s, [](auto& ro) { ro.chunks = "0"; ro.start = 0.0; ro.stop = 2.0; }); }).empty());
  CHECK(has(refusal([&] { with_span("short", two_s, [](auto& ro) { ro.start = 0.0; ro.stop = 1.0; }); }), "fewer than one segment"));
  CHECK(!refusal([&] { nxrtest::run_psd(ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"name", "y"}, {"overlap", 1.0}}, "brainstorm"); }).empty());
  CHECK(!refusal([&] { nxrtest::run_psd(ds, sub, "nope", two_s, "brainstorm"); }).empty());
  CHECK(has(refusal([&] { nxrtest::run_psd(ds, sub, "rest", two_s, "no-such-preset"); }), "no-such-preset"));

  // THE ROTARY SPECTRUM (the definition `rotary_psd`): the stored Operator [2 x 3] applied as z = fx.X + i fy.X, its TWO-SIDED Welch PSD over the same segments and mask, stored as a
  // statistic on axes [pair (1), signed frequency]; the chunk width does not matter; the mirrored operator (fy negated) mirrors the spectrum
  {
    Eigen::MatrixXd A(2, 3);
    A << 0.5f, -0.2f, 0.1f, 0.3f, 0.8f, -0.05f;                                              // as stored (float32)
    const Eigen::MatrixXcd z = nxr::rotary::pairSeries(A, X);
    const nxr::signal::spectral::ComplexWelchResult rw = nxr::signal::spectral::complexWelch(z, expect, &good);
    const Json rset = Json{{"segment_seconds", 2.0}, {"operator", "plane"}};
    const nxrtest::PsdRun ra = nxrtest::run_definition("rotary_psd", ds, sub, "rest", rset, "brainstorm");
    CHECK(!ra.hit);
    CHECK_EQ(ra.path, std::string("sensor-time/rest_statistics/rest_plane_psd_welch"));
    const StoredStatistic st = stat("rest_plane_psd_welch");
    CHECK_EQ(text(st.row, "measure"), std::string("power_density"));
    CHECK_EQ(text(st.row, "method"), std::string("welch"));
    CHECK_EQ(text(st.row, "of_field_id"), text(*sub.by_name("field", "rest"), "id"));
    CHECK_EQ(integer(st.row, "n_members"), rw.segments);
    CHECK_EQ(integer(st.row, "n_members"), want.segments);                                   // the same segment partition as the sensor PSD
    CHECK(st.axes.size() == 2 && st.axes[0]["name"] == "pair" && st.axes[0]["size"] == 1 && st.axes[1]["name"] == "frequency" && st.shape == (std::vector<std::int64_t>{1, 200}));
    CHECK(st.axes[0]["values"][0]["name"] == "plane");
    CHECK(st.options["input"]["operator"] == "plane" && st.options["input"]["band"] == Json::array({8.0, 12.0}) && st.options["sides"] == "two");
    const auto line = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::manifold, st.axes[1]["manifold_id"].get<std::string>());
    CHECK(line.has_value());
    const auto line_axis = nxr::io::task::Subject::of(sub).geometry(line->id).axes.at(0);
    CHECK_EQ(line->name, std::string("rest_freq_signed"));
    CHECK(std::abs(*line_axis.origin + 50.0) < 1e-9 && std::abs(*line_axis.spacing - 0.5) < 1e-12);   // -fs/2, 0.5 Hz
    double worst = 0;
    for (int k = 0; k < 200; ++k) worst = std::max(worst, std::abs(st.values[static_cast<std::size_t>(k)] / rw.psd(0, k) - 1));
    CHECK(worst < 1e-5);                                                                      // z passes through float32 blocks: a little above storage precision
    // chunk width (stored 1777 wide): the same numbers to the last bit
    const nxrtest::PsdRun rb = nxrtest::run_definition("rotary_psd", ds, sub, "rest_b", rset, "brainstorm");
    CHECK(!rb.hit);
    CHECK(nxrtest::psd_values(sub, ra) == nxrtest::psd_values(sub, rb));
    // the sense: fy negated is the mirror spectrum (P'(f) = P(-f)); +f is the operator's x -> y
    Eigen::MatrixXd Am = A; Am.row(1) *= -1;
    const nxr::signal::spectral::ComplexWelchResult mw = nxr::signal::spectral::complexWelch(nxr::rotary::pairSeries(Am, X), expect, &good);
    for (int k = 1; k < 200; ++k) CHECK(std::abs(mw.psd(0, k) / rw.psd(0, 200 - k) - 1) < 1e-9);
    CHECK(nxrtest::run_definition("rotary_psd", ds, sub, "rest", rset, "brainstorm").hit);
    CHECK(!refusal([&] { nxrtest::run_definition("rotary_psd", ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"operator", "nope"}}, "brainstorm"); }).empty());
    CHECK(has(refusal([&] { nxrtest::run_definition("rotary_psd", ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"operator", "not_a_pair"}, {"name", "x"}}, "brainstorm"); }), "two rows"));
  }
  // THE PRESET `tiles` (the default since 2026-11-13, the time-tile amendment): a segment of 2^k samples, k = round(log2(2 s x 100 Hz)) = 8 (256 samples), half overlapping, Brainstorm's taper,
  // detrend and density: nxr::welch on 256-sample segments of the same samples and mask; a span in seconds snaps to level 8 ([3, 33) s = samples [300, 3300) -> [256, 3328)); the segment is stated in seconds or samples; an overlap that leaves the hop no tile is refused, as is segment_samples that is no power of two
  {
    const Json t = Json{{"segment_seconds", 2.0}, {"name", "tiles"}};
    const nxrtest::PsdRun r = nxrtest::run_psd(ds, sub, "rest", t);
    auto eo = nxr::signal::spectral::brainstormPsdOptions(sfreq, 2.0);
    eo.segmentSamples = 256;
    const nxr::signal::spectral::WelchResult tw = nxr::signal::spectral::welch(X, eo, &good);
    const StoredStatistic st = stat("tiles");
    CHECK(st.options["segment_samples"] == 256 && st.options["segment_rule"] == "tile" && integer(st.row, "n_members") == tw.segments && st.shape == (std::vector<std::int64_t>{M, 129}));
    {
      const std::vector<float> got = nxrtest::psd_values(sub, r);
      double worst = 0;
      for (int c = 0; c < M; ++c) for (int k = 0; k < 129; ++k) worst = std::max(worst, std::abs(got[static_cast<std::size_t>(c * 129 + k)] / tw.psd(c, k) - 1));
      CHECK(worst < tol);
    }
    with_span("tiles_span", Json{{"segment_seconds", 2.0}}, [](auto& ro) { ro.start = 3.0; ro.stop = 33.0; }, "tiles");
    CHECK(stat("tiles_span").span["start_sample"] == 256 && stat("tiles_span").span["stop_sample"] == 3328);
    nxrtest::run_psd(ds, sub, "rest_b", Json{{"segment_samples", 128}, {"name", "tiles_l7"}});   // (the line `rest_b_freq` of 101 bins is taken: this is the second resolution)
    CHECK(stat("tiles_l7").options["segment_samples"] == 128);
    CHECK(has(refusal([&] { nxrtest::run_psd(ds, sub, "rest", Json{{"segment_seconds", 2.0}, {"name", "t1"}, {"overlap", 0.3}}); }), "hop"));
    CHECK(has(refusal([&] { nxrtest::run_psd(ds, sub, "rest", Json{{"segment_samples", 200}, {"name", "t2"}}); }), "tile"));
    // (a level index is no parameter any more: the segment is stated in seconds or samples, and `segment_rule: tile` takes a power of two)
  }

  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    const std::string run = cli + " run psd --dataset " + d + " --subject S1 --recording rest --name cli_psd --set scipy --segment-seconds 2 --window hann --detrend linear --chunks 0";
    CHECK_EQ(std::system((run + " > " + log).c_str()), 0);
    CHECK_EQ(std::system((run + " >> " + log).c_str()), 0);                                  // a hit
    CHECK(std::system((cli + " run psd --dataset " + d + " --subject S1 --recording rest --set bogus --chunks 0 > /dev/null 2>&1").c_str()) != 0);
    CHECK(std::system((cli + " psd compute --dataset " + d + " --subject S1 --recording rest --chunks 0 > /dev/null 2>&1").c_str()) == 2 * 256);   // the old spelling is refused (exit 2)
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(has(all, "\"hit\": false") && has(all, "\"hit\": true"));
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    const StoredStatistic st = read_statistic(s2, "cli_psd");
    CHECK_EQ(st.options["window_form"].get<std::string>(), std::string("periodic"));
    CHECK_EQ(st.options["detrend"].get<std::string>(), std::string("linear"));
    CHECK_EQ(st.options["segment_rule"].get<std::string>(), std::string("round"));
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_psd_pipeline");
}

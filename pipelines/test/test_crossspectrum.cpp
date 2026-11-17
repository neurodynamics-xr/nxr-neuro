// crossspectrum compute on the synthetic two-hemisphere brain of test_source_tensor (three sensors carrying a current that turns at 10 Hz, a free kernel giving every source the
// sensors' three channels as its x y z): the stored cross-spectrum is Hermitian, complex, on its channels' Selection; read through the kernel, K C K^T's diagonal blocks
// are the stored sourcetensor tensors (one source of truth, two routes); the result does not depend on the chunk size; a re-run is a cache hit; read on a subset of
// channels it is the matching sub-matrix, and a channel it does not cover is refused; --channels-of an Operator on other channels is refused.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/task/task.h"
#include "nxr/bst/entities.h"
#include "nxr/meeg/crossspectrum.h"
#include "nxr/meeg/tensor.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/timebank.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

template <class T> static std::vector<T> read_field(SubjectWriter& sub, const std::string& name) {
  const auto r = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = name, .any_status = true});
  ArrayReader a = ArrayReader::open(sub.store(), *r->path);
  const auto raw = a.read();
  std::vector<T> out(raw.size() / sizeof(T));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

static std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }

// a band of a map statistic as its values [n x n_components]: the statistic's values [n x entries x n_components] at the entry (the slice's index)
static std::vector<double> slice_values(nxr::io::SubjectWriter& sub, const std::string& statistic, const nxr::io::task::MapSlice& m) {
  const nxr::io::StoredStatistic st = nxr::io::read_statistic(sub, statistic);
  const std::int64_t K = st.shape.at(1), C = m.n_components;
  std::vector<double> v(static_cast<std::size_t>(m.n * C));
  for (std::int64_t i = 0; i < m.n; ++i)
    for (std::int64_t c = 0; c < C; ++c) v[static_cast<std::size_t>(i * C + c)] = st.values[static_cast<std::size_t>((i * K + m.index) * C + c)];
  return v;
}

// a map of a statistic by its name (`<statistic>_<band>`), as float [n x components] (since 2026-11-13 the source maps are statistics)
static std::vector<float> map_named(nxr::io::SubjectWriter& sub, const std::string& statistic, const std::string& name) {
  const std::string surf = nxr::io::read_statistic(sub, statistic).axes[0]["manifold_id"].get<std::string>();
  for (const auto& m : nxr::io::task::Subject::of(sub).map_slices(surf))
    if (m.name == name) { const auto d = slice_values(sub, statistic, m); return std::vector<float>(d.begin(), d.end()); }
  return {};
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_crossspectrum_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(2, false);
  const int N = b.n, M = 3, T = 6000;
  const double sfreq = 250.0;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (int t = 0; t < T; ++t) {
    const double w = 2 * M_PI * 10.0 * t / sfreq;
    xf[static_cast<size_t>(t)] = static_cast<float>(std::cos(w) + 0.05 * nd(rng));
    xf[static_cast<size_t>(T + t)] = static_cast<float>(std::sin(w) + 0.05 * nd(rng));
    xf[static_cast<size_t>(2 * T + t)] = static_cast<float>(0.05 * nd(rng));
  }
  // a DELAYED PAIR for the phase slope: channel 0 a broadband source, channel 1 its copy 3 samples (12 ms) later, each with independent noise; channel 2 independent
  std::vector<float> xd(static_cast<std::size_t>(M * T));
  {
    std::vector<double> src(static_cast<std::size_t>(T + 3));
    for (auto& v : src) v = nd(rng);
    for (int t = 0; t < T; ++t) {
      xd[static_cast<size_t>(t)] = static_cast<float>(src[static_cast<size_t>(t + 3)] + 0.3 * nd(rng));
      xd[static_cast<size_t>(T + t)] = static_cast<float>(src[static_cast<size_t>(t)] + 0.3 * nd(rng));
      xd[static_cast<size_t>(2 * T + t)] = static_cast<float>(nd(rng));
    }
  }
  std::vector<float> kf(static_cast<std::size_t>(3 * N * M), 0.0f), kfix(static_cast<std::size_t>(N * M), 0.0f);
  for (int v = 0; v < N; ++v) for (int a = 0; a < 3; ++a) kf[static_cast<size_t>((3 * v + a) * M + a)] = 1.0f;
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
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
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(700)), std::make_pair(std::string("rest_b"), std::int64_t(1777)),   // the same samples in four chunk widths
                                        std::make_pair(std::string("rest_d"), std::int64_t(128)), std::make_pair(std::string("rest_e"), std::int64_t(512))}) {
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
      {                                                                   // the same samples with a bad span at 9.1-9.3 s (inside the third 4 s window)
        FieldSpec rec;
        rec.name = "rest_c"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest_c"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
        const std::string rid = sub.field(std::move(rec));
        nxr::bst::TimeSpansOptions so; so.line_id = tid; so.of_field_id = rid; so.description = "bad segments";
        nxr::bst::time_spans(sub, "time/rest_c_bad_segments", {{9.1, 9.3, 0}}, {"BAD"}, so);
      }
      {
        FieldSpec rec;
        rec.name = "rest_delay"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest_delay"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xd, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
      std::vector<std::int32_t> chans = {0, 1, 2};
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        sub.operator_(std::move(op));
      };
      kernel("free", kf, 3 * N, "vector3");
      kernel("fixed", kfix, N, "scalar");
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  CrossSpectrumOptions o;
  o.recording = "rest"; o.chunks = "all"; o.channels_of = "free"; o.segment_seconds = 4.0; o.bands = {{"alpha", 8, 12}, {"beta", 15, 29}};
  const CrossSpectrumResult c = crossspectrum_compute(ds, sub, o);
  CHECK(!c.hit && c.S.size() == 2);
  // a STATISTIC per band (docs/specs/2026-11-13-statistics.md): cross_spectral_density by welch, from the segment partition, axes channel x channel x (re, im)
  {
    const StoredStatistic st = read_statistic(sub, "rest_crossspectrum_alpha");
    CHECK(text(st.row, "measure") == "cross_spectral_density" && text(st.row, "method") == "welch" && st.shape == (std::vector<std::int64_t>{M, M, 2}));
    CHECK(st.axes.size() == 2 && st.axes[0]["name"] == "channel" && st.axes[1]["selection_id"] == st.axes[0]["selection_id"] && text(st.row, "value_type") == "complex" && integer(st.row, "n_components") == 2);
    CHECK(st.options["band"]["name"] == "alpha" && integer(st.row, "n_members") > 0);
    const auto seg = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::selection, text(st.row, "partition_selection_id"));
    CHECK(seg && seg->name == "rest_crossspectrum_segments" && seg->n_members == integer(st.row, "n_members") + integer(st.row, "members_skipped"));
    CHECK(read_statistic(sub, "rest_crossspectrum_beta").options["band"]["name"] == "beta");
    // ONE MADE BEFORE (an Operator of kind `cross-spectral density`, as stored until 2026-11-13) is still read: no migration
    const auto chsel = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "rest_crossspectrum_channels", .any_status = true});
    std::vector<double> vals(st.values);
    sub.compose([&] {
      OperatorSpec op;
      op.name = "old_csd"; op.path = "old_csd"; op.kind = "cross-spectral density"; op.from_manifold_id = *chsel->manifold_id; op.layout = "dense";
      op.n_rows = M; op.n_cols = M; op.data = view(vals, {M, M, 2}); op.array.compress = false;
      op.cols.set("to_manifold_id", *chsel->manifold_id).set("from_selection_id", chsel->id).set("to_selection_id", chsel->id)
          .set("from_value_type", "complex").set("to_value_type", "complex");
      op.cols.json("producer_json", Json{{"band_name", "alpha"}});
      sub.operator_(std::move(op));
    });
    const StoredCrossSpectrum oc = crossspectrum_read(sub, "old_csd"), nc = crossspectrum_read(sub, "rest_crossspectrum_alpha");
    CHECK(oc.S == nc.S && oc.Sq == nc.Sq && oc.channels == nc.channels);
  }
  const StoredCrossSpectrum C = crossspectrum_read(sub, "rest_crossspectrum_alpha");
  CHECK_EQ(static_cast<int>(C.channels.size()), M);
  const double herm = std::max((C.S - C.S.transpose()).cwiseAbs().maxCoeff(), (C.Sq + C.Sq.transpose()).cwiseAbs().maxCoeff());
  const double stored = std::max((C.S - c.S[0]).cwiseAbs().maxCoeff(), (C.Sq - c.Sq[0]).cwiseAbs().maxCoeff());
  // the turn (cos on channel 0, sin on channel 1): a quadrature part between them, about as large as their power
  std::printf("  Hermitian to %.2e; stored vs computed %.2e; |Sq01| / S00 = %.3f\n", herm, stored, std::abs(C.Sq(0, 1)) / C.S(0, 0));
  CHECK(herm < 1e-12 * C.S.cwiseAbs().maxCoeff() && stored == 0);
  CHECK(std::abs(C.Sq(0, 1)) > 0.5 * C.S(0, 0));
  // K C K^T's diagonal blocks are sourcetensor's tensors (stored float32)
  FramesComputeOptions fo;
  frames_compute(ds, sub, fo);
  SourcePsdOptions so;
  so.recording = "rest"; so.chunks = "all"; so.kernel = "free"; so.segment_seconds = 4.0; so.bands = {{"alpha", 8, 12}};
  sourcetensor_compute(ds, sub, so);
  const auto herm3 = map_named(sub, "rest_free_tensor", "rest_free_tensor_alpha");
  Eigen::MatrixXd K(3 * N, M);
  for (int r = 0; r < 3 * N; ++r) for (int ch = 0; ch < M; ++ch) K(r, ch) = kf[static_cast<size_t>(r * M + ch)];
  double worst = 0, scale = 0;
  double re[9], im[9];
  for (int v = 0; v < N; ++v) {
    nxr::meeg::crossTensor(K, C.S, C.Sq, v, v, re, im);
    for (int k = 0; k < 9; ++k) {
      worst = std::max({worst, std::abs(re[k] - herm3[static_cast<size_t>(18 * v + 2 * k)]), std::abs(im[k] - herm3[static_cast<size_t>(18 * v + 2 * k + 1)])});
      scale = std::max(scale, std::abs(re[k]));
    }
  }
  std::printf("  K C K^T diagonal blocks vs sourcetensor: %.2e (of %.2e)\n", worst, scale);
  CHECK(worst < 1e-6 * scale);
  // the same samples stored in another chunk width: the same result
  CrossSpectrumOptions o2 = o; o2.recording = "rest_b"; o2.name = "rest_b_crossspectrum";
  const CrossSpectrumResult c2 = crossspectrum_compute(ds, sub, o2);
  const double chunks = std::max((c2.S[0] - c.S[0]).cwiseAbs().maxCoeff(), (c2.Sq[0] - c.Sq[0]).cwiseAbs().maxCoeff()) / c.S[0].cwiseAbs().maxCoeff();
  std::printf("  chunk width 700 vs 1777: %.2e\n", chunks);
  CHECK(chunks < 1e-12);
  // a re-run is a cache hit; the default channels (good imaging ones) are the same three here
  CHECK(crossspectrum_compute(ds, sub, o).hit);
  CrossSpectrumOptions od = o; od.channels_of.reset(); od.name = "rest_crossspectrum_default";
  CHECK_EQ(static_cast<int>(crossspectrum_compute(ds, sub, od).channels.size()), M);
  // read on a subset, in another order: the matching sub-matrix; a channel not covered is refused
  const StoredCrossSpectrum sub2 = crossspectrum_read(sub, "rest_crossspectrum_alpha", std::vector<std::int32_t>{2, 0});
  CHECK(sub2.S(0, 1) == C.S(2, 0) && sub2.Sq(1, 0) == C.Sq(0, 2) && sub2.S(1, 1) == C.S(0, 0));
  bool uncovered = false;
  try { crossspectrum_read(sub, "rest_crossspectrum_alpha", std::vector<std::int32_t>{0, 5}); } catch (const Error&) { uncovered = true; }
  CHECK(uncovered);
  bool absent = false;
  CrossSpectrumOptions ob = o; ob.channels_of = "nothing"; ob.name = "rest_crossspectrum_bad";
  try { crossspectrum_compute(ds, sub, ob); } catch (const Error&) { absent = true; }
  CHECK(absent);
  // ── --resolve bins (spec 2026-11-15-band-resolved-cross-spectrum.md): the band's cross-spectrum bin by bin, from the same pass ──
  {
    CrossSpectrumOptions ox = o; ox.name = "rb"; ox.resolve = "bins";
    const CrossSpectrumResult r = crossspectrum_compute(ds, sub, ox);
    CHECK(!r.hit && r.bins_S.size() == 2 && r.bins_S[0].size() == 17 && r.bins_S[1].size() == 57);   // alpha 8-12 Hz at 0.25 Hz: 17 bins; beta 15-29: 57
    const StoredStatistic sb = read_statistic(sub, "rb_alpha_bins");
    CHECK(text(sb.row, "measure") == "cross_spectral_density" && text(sb.row, "method") == "welch" && text(sb.row, "value_type") == "complex" && sb.shape == (std::vector<std::int64_t>{M, M, 17, 2}));
    CHECK(sb.axes.size() == 3 && sb.axes[2]["name"] == "frequency" && sb.axes[2]["values"].size() == 17 && std::abs(sb.axes[2]["values"][0].get<double>() - 33 * 250.0 / 1024) < 1e-12 && sb.axes[2]["bins"] == Json::array({33, 49}));
    CHECK(sb.options["resolve"] == "bins" && sb.producer["resolve"] == "bins" && sb.producer.contains("key") == sb.producer.contains("key"));
    // the band statistic of this run is the mean of the bins, and the same as a band-resolution run's
    const StoredCrossSpectrumBins B = crossspectrum_bins_read(sub, "rb_alpha_bins");
    const StoredCrossSpectrum Cb = crossspectrum_read(sub, "rb_alpha");
    Eigen::MatrixXcd mean = Eigen::MatrixXcd::Zero(M, M);
    for (const auto& c : B.C) mean += c;
    mean /= static_cast<double>(B.C.size());
    double herm_b = 0;
    for (const auto& c : B.C) herm_b = std::max(herm_b, (c - c.adjoint()).cwiseAbs().maxCoeff() / c.cwiseAbs().maxCoeff());
    const double mean_err = std::max((mean.real() - Cb.S).cwiseAbs().maxCoeff(), (mean.imag() - Cb.Sq).cwiseAbs().maxCoeff()) / Cb.S.cwiseAbs().maxCoeff();
    const double vs_band = std::max((Cb.S - C.S).cwiseAbs().maxCoeff(), (Cb.Sq - C.Sq).cwiseAbs().maxCoeff()) / C.S.cwiseAbs().maxCoeff();
    std::printf("  bins: mean over the bins vs the band statistic %.2e; band statistic vs a band-resolution run %.2e; Hermitian %.2e\n", mean_err, vs_band, herm_b);
    CHECK(mean_err < 1e-13 && vs_band < 1e-13 && herm_b < 1e-12 && B.frequencies.size() == 17 && std::abs(B.frequencies[16] - 49 * 250.0 / 1024) < 1e-12);
    // a re-run is a hit; a band-resolution run under the same name is another key (refused without --replace, so the stored result is never silently swapped)
    CHECK(crossspectrum_compute(ds, sub, ox).hit);
    {   // the band run's cache key is the one from before the option: exactly these params (no `resolve`), so stored band statistics stay hits; bins adds it
      const Json kb = read_statistic(sub, "rest_crossspectrum_alpha").producer["cache"]["params"], kx = sb.producer["cache"]["params"];
      std::vector<std::string> names;
      for (auto it = kb.begin(); it != kb.end(); ++it) names.push_back(it.key());
      std::sort(names.begin(), names.end());
      CHECK((names == std::vector<std::string>{"bad_segments", "bands", "detrend", "name", "option_set", "overlap", "periodic", "scaling", "segment_samples", "window", "window_samples"}));
      CHECK(!kb.contains("resolve") && kx.contains("resolve") && kx["resolve"] == "bins");
    }
    // independent of the chunk size
    CrossSpectrumOptions oy = ox; oy.recording = "rest_b"; oy.name = "rb_b";
    crossspectrum_compute(ds, sub, oy);
    const StoredCrossSpectrumBins B2 = crossspectrum_bins_read(sub, "rb_b_alpha_bins");
    double cw = 0;
    for (std::size_t k = 0; k < B.C.size(); ++k) cw = std::max(cw, (B.C[k] - B2.C[k]).cwiseAbs().maxCoeff() / B.C[k].cwiseAbs().maxCoeff());
    std::printf("  bins, chunk width 700 vs 1777: %.2e\n", cw);
    CHECK(cw < 1e-12);
    // read on a subset of channels, in another order
    const StoredCrossSpectrumBins B3 = crossspectrum_bins_read(sub, "rb_alpha_bins", std::vector<std::int32_t>{2, 0});
    CHECK(B3.C[5](0, 1) == B.C[5](2, 0) && B3.C[5](1, 1) == B.C[5](0, 0));
    // refusals: the analytic estimator has no bins; a bad value names the valid ones; a band statistic is not a bins statistic
    CrossSpectrumOptions oz = ox; oz.estimator = "analytic"; oz.name = "rb_an";
    CHECK(code_of([&] { crossspectrum_compute(ds, sub, oz); }) != "");
    CrossSpectrumOptions ow = ox; ow.resolve = "fine"; ow.name = "rb_bad";
    CHECK(code_of([&] { crossspectrum_compute(ds, sub, ow); }) != "");
    CHECK(code_of([&] { crossspectrum_bins_read(sub, "rb_alpha"); }) != "");
    // THE PHASE SLOPE recovers a known delay: channel 0 leads its copy in channel 1 by 12 ms (the broadband source, plus independent noise); channel 2 is independent
    CrossSpectrumOptions od2 = ox; od2.recording = "rest_delay"; od2.name = "dl"; od2.bands = {{"wide", 5, 30}, {"alpha", 8, 12}};
    crossspectrum_compute(ds, sub, od2);
    for (const char* nm : {"dl_wide_bins", "dl_alpha_bins"}) {
      const StoredCrossSpectrumBins D = crossspectrum_bins_read(sub, nm);
      Eigen::VectorXd fv = Eigen::Map<const Eigen::VectorXd>(D.frequencies.data(), static_cast<Eigen::Index>(D.frequencies.size()));
      const nxr::meeg::PhaseSlope ps = nxr::meeg::phaseSlope(D.C, fv);
      const Eigen::MatrixXd psi = nxr::meeg::phaseSlopeIndex(D.C);
      std::printf("  %s: delay 0->1 %.3f ms (made with 12), 1->0 %.3f, residual %.3f rad, PSI %.3f\n", nm, ps.delayMs(0, 1), ps.delayMs(1, 0), ps.residual(0, 1), psi(0, 1));
      // the wide band (5-30 Hz, 101 bins) recovers the delay; 8-12 Hz holds only a few independent bins of 6 segments, so its slope is the right sign and no more
      CHECK(std::abs(ps.delayMs(1, 0) + ps.delayMs(0, 1)) < 1e-9 && psi(0, 1) > 0 && psi(1, 0) < 0 && ps.delayMs(0, 1) > 0);
      if (std::string(nm) == "dl_wide_bins") CHECK(std::abs(ps.delayMs(0, 1) - 12.0) < 1.5);
    }
  }
  // ── THE ANALYTIC ESTIMATOR (spec 2026-11-13-cycle-pattern.md, stage 1, as amended 2026-11-13: the time tiles) ──
  // At 250 Hz the bands' constant-Q windows are alpha 8-12 Hz: 256 samples (level 8), 5-14 Hz: 512 (level 9), so the series is stored at level 8; the ladder's
  // strides are alpha 1 and 5-14 Hz 2 (125 Hz). 6000 samples hold 23 whole tiles of 256 (the 24th is partial).
  {
    // THE REFERENCE: each band's analytic signal over the whole recording at the full rate, Σ z zᴴ per level-8 tile of the line
    nxr::timebank::Design dz(nxr::timebank::Family::Meyer, 2.0, nxr::timebank::Sides::Analytic);
    dz.bands = {{8, 12}, {5, 14}};
    Eigen::MatrixXd X(M, T);
    for (int ch = 0; ch < M; ++ch) for (int t = 0; t < T; ++t) X(ch, t) = xf[static_cast<std::size_t>(ch * T + t)];
    const auto z = nxr::timebank::analysisOfBlock(dz, X, sfreq, 0, static_cast<std::size_t>(T));
    auto ref = [&](int band, std::int64_t m) { const Eigen::MatrixXcd Z = z.own[static_cast<std::size_t>(band)].middleCols(m * 256, 256); return Eigen::MatrixXcd(Z * Z.adjoint()); };
    CrossSpectrumOptions oa;
    oa.recording = "rest"; oa.chunks = "all"; oa.channels_of = "free"; oa.estimator = "analytic"; oa.bands = {{"alpha", 8, 12}, {"wide", 5, 14}};
    oa.name = "rest_an";
    const CrossSpectrumResult a = crossspectrum_compute(ds, sub, oa);
    CHECK(!a.hit && a.windows.size() == 2 && a.windows[0].sum.size() == 23 && a.windows[0].level == 8);
    const CrossSpectrumWindows st = crossspectrum_windows_read(sub, "rest_an_alpha_windows");
    const CrossSpectrumWindows sw = crossspectrum_windows_read(sub, "rest_an_wide_windows");
    {   // a STATISTIC (2026-11-13): cross_spectrum_sum by analytic, complex, axes window (the kept tiles of the level, by ordinal) x channel x channel; its partition the level's tiling
      const StoredStatistic ss = read_statistic(sub, "rest_an_alpha_windows");
      CHECK(text(ss.row, "measure") == "cross_spectrum_sum" && text(ss.row, "method") == "analytic" && text(ss.row, "value_type") == "complex");
      CHECK(ss.axes.size() == 3 && ss.axes[0]["name"] == "window" && ss.axes[0]["members"].size() == 23 && ss.shape == (std::vector<std::int64_t>{23, M, M, 2}));
      const auto ws = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::selection, text(ss.row, "partition_selection_id"));
      CHECK(ws && ws->name == "rest_an_windows" && integer(ss.row, "n_members") == 23);
    }
    CHECK(st.level == 8 && st.count[0] == 256 && sw.count[0] == 128 && st.index.front() == 0 && st.index.back() == 22);
    CHECK(st.producer["decimation"] == 1 && sw.producer["decimation"] == 2 && st.producer["window_level"] == 8 && sw.producer["rate_hz"] == 125.0);
    double herm = 0, same = 0, ra = 0, rw = 0;
    for (std::size_t w = 0; w < st.sum.size(); ++w) {
      herm = std::max(herm, (st.sum[w] - st.sum[w].adjoint()).cwiseAbs().maxCoeff() / st.sum[w].cwiseAbs().maxCoeff());
      same = std::max(same, (st.sum[w] - a.windows[0].sum[w]).cwiseAbs().maxCoeff());
      const Eigen::MatrixXcd r0 = ref(0, st.index[w]), r1 = ref(1, sw.index[w]);
      ra = std::max(ra, (st.sum[w] - r0).cwiseAbs().maxCoeff() / r0.cwiseAbs().maxCoeff());
      rw = std::max(rw, (2.0 * sw.sum[w] - r1).norm() / r1.norm());
    }
    // alpha (stride 1) is the reference to the halo's truncation; 5-14 Hz kept every 2nd sample on the line's grid: x 2 is the full-rate sum (band-limited)
    std::printf("  analytic tiles vs the full-recording reference: alpha %.2e, 5-14 Hz decimated x 2 %.2e\n", ra, rw);
    CHECK(herm < 1e-12 && same == 0 && ra < 1e-4 && rw < 1e-2);
    // the windows Selection: the level's closed-form tiling of the whole line (the envelope's form)
    const auto wsel = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "rest_an_windows", .type = "spans", .any_status = true});
    CHECK(wsel.has_value());
    if (wsel) {
      const Json wp = wsel->params;
      CHECK(wp["tiling"]["count"] == 24 && std::abs(wp["tiling"]["width"].get<double>() - 256 / sfreq) < 1e-12 && wp["level"] == 8 && wsel->n_members == 24);
    }
    // the turn: channel 1 lags channel 0 by a quarter cycle, so Im C01 is about as large as C00 in every tile
    CHECK(std::abs(st.sum[8](0, 1).imag()) > 0.8 * st.sum[8](0, 0).real());
    // other chunk widths: 1777 (no power of two), 128 (finer than the tile: tiles summed across chunks) and 512 (coarser): the same tiles
    for (const char* r : {"rest_b", "rest_d", "rest_e"}) {
      CrossSpectrumOptions ob2 = oa; ob2.recording = r; ob2.name = std::string(r) + "_an";
      const CrossSpectrumResult b2 = crossspectrum_compute(ds, sub, ob2);
      double chunks = 0;
      CHECK(b2.windows[0].index == a.windows[0].index);
      for (std::size_t w = 0; w < a.windows[0].sum.size(); ++w)
        chunks = std::max(chunks, (b2.windows[0].sum[w] - a.windows[0].sum[w]).cwiseAbs().maxCoeff() / a.windows[0].sum[w].cwiseAbs().maxCoeff());
      // not to rounding: the band's response is cut at its halo (timebank::haloOf, where the tail holds under the Studio's TAIL_ENERGY = 1e-6), so a sample
      // near a chunk edge differs by that tail
      std::printf("  analytic tiles, chunk width 700 vs %s: %.2e (the halo's truncation)\n", r, chunks);
      CHECK(chunks < 1e-4);
    }
    // COARSER WINDOWS ARE SUMS AT READ: level 10 is the level-8 tiles summed by fours; window 5 (tiles 20-23) lacks tile 23 (partial): dropped
    const CrossSpectrumWindows up = crossspectrum_windows_read(sub, "rest_an_alpha_windows", std::nullopt, "stored", 10);
    CHECK(up.level == 10 && up.sum.size() == 5 && up.index.back() == 4 && up.count[0] == 1024);
    CHECK((up.sum[1] - (st.sum[4] + st.sum[5] + st.sum[6] + st.sum[7])).cwiseAbs().maxCoeff() < 1e-12 * up.sum[1].cwiseAbs().maxCoeff());
    CHECK_EQ(code_of([&] { crossspectrum_windows_read(sub, "rest_an_alpha_windows", std::nullopt, "stored", 7); }), std::string(code::invalid_argument));   // finer than stored
    // stored at level 10 directly (--level): the same windows as the roll-up
    CrossSpectrumOptions ol = oa; ol.level = 10; ol.name = "rest_l10";
    const CrossSpectrumResult l10 = crossspectrum_compute(ds, sub, ol);
    CHECK(l10.windows[0].level == 10 && l10.windows[0].index == up.index);
    double roll = 0;
    for (std::size_t w = 0; w < up.sum.size(); ++w) roll = std::max(roll, (l10.windows[0].sum[w] - up.sum[w]).cwiseAbs().maxCoeff() / up.sum[w].cwiseAbs().maxCoeff());
    CHECK(roll < 1e-12);
    // --window-seconds snaps to the nearest level: 4 s at 250 Hz is 1000 samples, level 10
    CrossSpectrumOptions ow = oa; ow.window_seconds = 4.0; ow.name = "rest_w4";
    CHECK(crossspectrum_compute(ds, sub, ow).windows[0].level == 10);
    // a bad span at 9.1-9.3 s (samples 2275-2325): tiles 8 and 9 dropped, the others the same; at level 10, window 2 (tiles 8-11) dropped
    CrossSpectrumOptions oc = oa; oc.recording = "rest_c"; oc.name = "rest_c_an";
    const CrossSpectrumResult c3 = crossspectrum_compute(ds, sub, oc);
    CHECK(c3.windows[0].sum.size() == 21 && c3.windows[0].index[7] == 7 && c3.windows[0].index[8] == 10);
    CHECK((c3.windows[0].sum[8] - a.windows[0].sum[10]).cwiseAbs().maxCoeff() < 1e-12 * a.windows[0].sum[10].cwiseAbs().maxCoeff());
    CHECK(crossspectrum_windows_read(sub, "rest_c_an_alpha_windows", std::nullopt, "stored", 10).index == (std::vector<std::int64_t>{0, 1, 3, 4}));
    // a span in seconds snaps to the tiles: [3, 9) s = samples [750, 2250) -> [768, 2304), tiles 3 to 8
    CrossSpectrumOptions oss = oa; oss.chunks.reset(); oss.start = 3.0; oss.stop = 9.0; oss.name = "rest_span";
    const CrossSpectrumResult sp = crossspectrum_compute(ds, sub, oss);
    CHECK(sp.windows[0].index == (std::vector<std::int64_t>{3, 4, 5, 6, 7, 8}));
    CHECK(sp.info["window"] == Json::array({768, 2304}));
    // THE SCALING CHOSEN AT READ (nxr::spectral): the Welch density read as power is × the band's width (bins × resolution); the analytic windows read as
    // a density are sum / (2 count) / the band-pass's noise bandwidth; and on this 10 Hz turn the two estimators' band powers of channel 0 agree (0.5,
    // the cosine's power, within 5 %); a non-linear scaling of a cross-spectrum is refused
    {
      const StoredCrossSpectrum dens = crossspectrum_read(sub, "rest_crossspectrum_alpha", std::nullopt, "density");
      const StoredCrossSpectrum pow = crossspectrum_read(sub, "rest_crossspectrum_alpha", std::nullopt, "power");
      const double width = C.producer["spectral_scaling"]["band_width_hz"].get<double>();
      CHECK(dens.S == C.S && (pow.S - width * C.S).cwiseAbs().maxCoeff() < 1e-12 * C.S.cwiseAbs().maxCoeff() && pow.unit == "channel unit\xc2\xb2" && dens.unit == "channel unit\xc2\xb2/Hz");
      const CrossSpectrumWindows ad = crossspectrum_windows_read(sub, "rest_an_alpha_windows", std::nullopt, "density");
      const CrossSpectrumWindows ap = crossspectrum_windows_read(sub, "rest_an_alpha_windows", std::nullopt, "power");
      const double bn = ad.producer["spectral_scaling"]["band_width_hz"].get<double>();
      CHECK(ad.value.size() == 23 && (ad.value[2] - st.sum[2] / (2.0 * 256 * bn)).cwiseAbs().maxCoeff() < 1e-12 * ad.value[2].cwiseAbs().maxCoeff());
      double mean_ap = 0; for (const auto& v : ap.value) mean_ap += v(0, 0).real() / static_cast<double>(ap.value.size());
      std::printf("  band power of channel 0: Welch (density x width) %.4f, analytic %.4f (the cos at 10 Hz carries 0.5; noise bandwidth %.2f Hz)\n", pow.S(0, 0), mean_ap, bn);
      CHECK(std::abs(pow.S(0, 0) / 0.5 - 1) < 0.05 && std::abs(mean_ap / 0.5 - 1) < 0.05);
      bool nl = false;
      try { crossspectrum_read(sub, "rest_crossspectrum_alpha", std::nullopt, "db"); } catch (const Error&) { nl = true; }
      CHECK(nl);
    }
    // a re-run is a cache hit; refusals: an unknown estimator, --level with Welch, a level above a chunk, Welch segments with analytic, a band too wide for
    // its stride (2-40 Hz: stride 4 from the 2 Hz edge, a 31 Hz Nyquist rate)
    CHECK(crossspectrum_compute(ds, sub, oa).hit);
    CHECK_EQ(code_of([&] { CrossSpectrumOptions e = oa; e.estimator = "multitaper"; e.name = "x1"; crossspectrum_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { CrossSpectrumOptions e = o; e.level = 10; e.name = "x2"; crossspectrum_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { CrossSpectrumOptions e = oa; e.level = 16; e.name = "x3"; crossspectrum_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { CrossSpectrumOptions e = oa; e.segment_samples = 1000; e.name = "x4"; crossspectrum_compute(ds, sub, e); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { CrossSpectrumOptions e = oa; e.bands = {{"broad", 2, 40}}; e.name = "x5"; crossspectrum_compute(ds, sub, e); }), std::string(code::invalid_argument));
  }
  fs::remove_all(root);
  return nxrtest::finish("test_crossspectrum");
}

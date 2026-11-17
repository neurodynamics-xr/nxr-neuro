// Source band power maps. (1) Brainstorm's output on synthetic data (golden/source_psd_golden.txt: bst_psd on K x, "Group in frequency bands", "relative" normalisation; the three orientations
// summed): the cross-spectrum route equals it. (2) the cross-spectrum route equals Welch run on K x directly (also with a bad span, in chunks of any size). (3) the command's pipeline on a small
// synthetic dataset: one field per band, rows and arrays together, cache hit, replace, relative.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/io/task/task.h"
#include "golden_io.h"
#include "nxr/bst/entities.h"
#include "nxr/meeg/sources.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/measure.h"
#include "nxr/io/measures.h"
#include "nxr/welch.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

static std::vector<nxr::welch::BinGroup> groups_of(const std::vector<nxr::meeg::Band>& bands, double df, std::size_t bins, bool total) {
  std::vector<nxr::welch::BinGroup> g;
  for (const auto& b : bands) g.push_back({static_cast<std::size_t>(std::ceil(b.lo / df - 1e-9)), static_cast<std::size_t>(std::min<double>(std::floor(b.hi / df + 1e-9), bins - 1)), true});
  if (total) g.push_back({0, bins - 1, false});
  return g;
}
static MatrixXd sum3(const MatrixXd& P) {
  MatrixXd out(P.rows() / 3, P.cols());
  for (Eigen::Index s = 0; s < out.rows(); ++s) out.row(s) = P.row(3 * s) + P.row(3 * s + 1) + P.row(3 * s + 2);
  return out;
}
static double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }

int main() {
  const auto bands = nxr::meeg::brainstormBands();
  {   // (1) Brainstorm
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/source_psd_golden.txt");
    CHECK(g.good());
    const double fs = g.mat("fs")(0, 0);
    const MatrixXd x = g.mat("x"), K = g.mat("K"), freqs = g.mat("freqs"), bst = g.mat("bands_bst"), rule = g.mat("bands_rule"), relb = g.mat("relative_bst");
    auto wo = nxr::welch::brainstormPsdOptions(fs, 4.0);
    const std::size_t bins = wo.segmentSamples / 2 + 1;
    CHECK_EQ(static_cast<std::size_t>(freqs.cols()), bins);
    nxr::welch::CrossWelch cw(static_cast<int>(x.rows()), wo, groups_of(bands, fs / wo.segmentSamples, bins, true));
    cw.add(x);
    const auto r = cw.result();
    std::vector<MatrixXd> S(r.S.begin(), r.S.begin() + 6);
    const MatrixXd P = nxr::meeg::sourcePower(K, S, 3, nxr::meeg::Orientation::Norm);
    const MatrixXd Pb = sum3(bst);
    std::printf("  parity with Brainstorm bands (3 orientations summed): %.2e   (Brainstorm's band function vs its documented rule: %.1e)\n", rel(P, Pb), rel(bst, rule));
    CHECK(rel(P, Pb) < 1e-6);
    // relative power on a fixed-orientation kernel (each row a source, as Brainstorm normalises each row of the file): band mean / the sum of all bins
    const MatrixXd Kf = K.topRows(50);
    const MatrixXd Pf = nxr::meeg::sourcePower(Kf, S, 1, nxr::meeg::Orientation::Norm);
    const MatrixXd Rf = nxr::meeg::relativePower(Pf, nxr::meeg::sourcePower(Kf, {r.S.back()}, 1, nxr::meeg::Orientation::Norm).col(0));
    std::printf("  parity with Brainstorm relative power (fixed kernel): %.2e\n", rel(Rf, relb.topRows(50)));
    CHECK(rel(Rf, relb.topRows(50)) < 1e-6);
  }
  {   // (2) cross-spectrum route == Welch on K x, with a bad span and odd chunks
    std::mt19937 rng(5);
    std::normal_distribution<double> nd;
    const int M = 20, N = 50;
    for (double fs : {250.0, 600.0}) {
      const int T = static_cast<int>(12 * fs);   // 12 s, one second bad
      MatrixXd x(M, T), K(3 * N, M);
      for (auto& v : x.reshaped()) v = nd(rng);
      for (auto& v : K.reshaped()) v = nd(rng);
      std::vector<std::uint8_t> good(T, 1);
      for (int i = static_cast<int>(5 * fs); i < static_cast<int>(6 * fs); ++i) good[i] = 0;
      auto wo = nxr::welch::brainstormPsdOptions(fs, 4.0);
      const std::size_t bins = wo.segmentSamples / 2 + 1;
      const double df = fs / wo.segmentSamples;
      const auto gr = groups_of(bands, df, bins, true);
      nxr::welch::CrossWelch cw(M, wo, gr);
      for (int t = 0; t < T; t += 333) {
        const int n = std::min(333, T - t);
        cw.add(x.middleCols(t, n), std::vector<std::uint8_t>(good.begin() + t, good.begin() + t + n));
      }
      const auto r = cw.result();
      const auto direct = nxr::welch::welch(K * x, wo, &good);
      CHECK_EQ(r.segments, direct.segments);
      MatrixXd want(3 * N, gr.size());
      for (std::size_t b = 0; b < gr.size(); ++b)
        { if (gr[b].mean) want.col(b) = direct.psd.middleCols(gr[b].first, gr[b].last - gr[b].first + 1).rowwise().mean(); else want.col(b) = direct.psd.rowwise().sum(); }
      const MatrixXd got = nxr::meeg::sourcePower(K, r.S, 3, nxr::meeg::Orientation::Components);
      std::printf("  cross-spectrum vs Welch on K x, %g Hz, %lld segments (%lld skipped): %.2e\n", fs, static_cast<long long>(r.segments), static_cast<long long>(r.segmentsSkipped), rel(got, want));
      CHECK(rel(got, want) < 1e-10);
    }
  }
  {   // (3) the pipeline on a small synthetic dataset
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_sourcepsd_" + std::to_string(::getpid()));
    fs::remove_all(root);
    std::mt19937 rng(9);
    std::normal_distribution<double> nd;
    const std::int64_t M = 4, N = 3, T = 2400;
    const double sfreq = 100.0;
    MatrixXd X(M, T), K(3 * N, M);
    for (auto& v : X.reshaped()) v = static_cast<float>(nd(rng));
    for (int t = 0; t < T; ++t) X.col(t) += Eigen::VectorXd::Constant(M, 3.0 * std::sin(2 * M_PI * 10.0 * t / sfreq));
    X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
    for (auto& v : K.reshaped()) v = static_cast<float>(nd(rng));
    K = K.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
    DatasetWriter ds = create_dataset(root, "demo");
    std::string surf;
    {
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
        std::vector<float> xf(static_cast<std::size_t>(M * T));
        for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
        FieldSpec rec;
        rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
        // the sources (3 points) and a kernel [3N x M] on all four channels
        const std::string sid = uuid7();
        std::vector<float> sp(static_cast<std::size_t>(N * 3));
        for (auto& p : sp) p = static_cast<float>(nd(rng));
        ManifoldSpec sm;
        sm.id = sid; sm.name = "cortex"; sm.path = "source/cortex"; sm.type = "points"; sm.n_vertices = N;
        sm.topology = {"none", 0, 1, std::nullopt, {}};
        GeometrySpec sg; sg.form = "stored"; sg.cs_key = "scs"; sg.positions_path = "source/cortex"; sm.geometry = sg;
        sm.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("source/cortex", view(sp, {N, 3}), {}, std::nullopt, "manifold", sid); };
        surf = sub.manifold(std::move(sm));
        std::vector<std::int32_t> chans = {0, 1, 2, 3};
        SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
        const std::string sel = sub.selection(std::move(ss));
        std::vector<float> kf(static_cast<std::size_t>(3 * N * M));
        for (int i = 0; i < 3 * N; ++i) for (int j = 0; j < M; ++j) kf[static_cast<std::size_t>(i * M + j)] = static_cast<float>(K(i, j));
        OperatorSpec op;
        op.name = "kern"; op.path = "kern"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
        op.data = view(kf, {3 * N, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
        sub.operator_(std::move(op));
      });
    }
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    SourcePsdOptions o;
    o.recording = "rest";
    o.chunks = "all";                                           // every chunk (stored 700 wide), against the whole-record reference
    o.option_set = "brainstorm";                                // the direct reference below is Brainstorm's exact 4 s
    o.segment_seconds = 4.0;
    o.bands = {{"alpha", 8, 12}, {"beta", 15, 29}};
    const SourcePsdResult a = sourcepsd_compute(ds, sub, o);
    CHECK(!a.hit);
    CHECK_EQ(a.info["maps"].size(), std::size_t(2));
    auto wo = nxr::welch::brainstormPsdOptions(sfreq, 4.0);
    const auto direct = nxr::welch::welch(K * X, wo);
    const double df = sfreq / wo.segmentSamples;
    const auto gr = groups_of(o.bands, df, wo.segmentSamples / 2 + 1, false);
    MatrixXd want(N, 2);
    for (int b = 0; b < 2; ++b) {
      Eigen::VectorXd m = direct.psd.middleCols(gr[b].first, gr[b].last - gr[b].first + 1).rowwise().mean();
      for (int s = 0; s < N; ++s) want(s, b) = m[3 * s] + m[3 * s + 1] + m[3 * s + 2];
    }
    std::printf("  pipeline maps vs Welch on K x: %.2e\n", rel(a.maps, want));
    CHECK(rel(a.maps, want) < 1e-10);
    // A STATISTIC read as MAPS (docs/specs/2026-11-13-statistics.md): power_density by welch, axes vertex (the cortex) x band; band k's map is `<stem>_<band>`
    {
      const StoredStatistic st = read_statistic(sub, a.info["statistic"].get<std::string>());
      CHECK(text(st.row, "measure") == "power_density" && text(st.row, "method") == "welch" && st.shape == (std::vector<std::int64_t>{N, 2}));
      CHECK(st.axes[0]["manifold_id"] == surf && st.axes[1]["values"][0]["name"] == "alpha" && st.axes[1]["values"][1]["lo_hz"] == 15.0);
      CHECK(integer(st.row, "n_members") > 0 && !is_null(st.row, "partition_selection_id"));
      const auto slices = nxr::io::task::Subject::of(sub).map_slices(surf);
      CHECK(slices.size() == 2 && slices[0].name == a.info["maps"][0].get<std::string>() && slices[1].index == 1);
      const auto beta = [&] {   // the band's map: the statistic's values [N x entries] at the slice's entry
        const std::int64_t K = st.shape.at(1);
        std::vector<double> v(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) v[static_cast<std::size_t>(i)] = st.values[static_cast<std::size_t>(i * K + slices[1].index)];
        return v;
      }();
      double worst = 0;
      for (int i = 0; i < N; ++i) worst = std::max(worst, std::abs(beta[static_cast<std::size_t>(i)] / want(i, 1) - 1));
      CHECK(worst < 1e-6);                                       // float32 storage
    }
    const SourcePsdResult b = sourcepsd_compute(ds, sub, o);
    CHECK(b.hit);
    bool refused = false;
    SourcePsdOptions o2 = o;
    o2.normalisation = "relative";
    try { sourcepsd_compute(ds, sub, o2); } catch (const Error& e) { refused = true; }   // same names, other options: refused until --replace
    CHECK(refused);
    o2.replace = true;
    const SourcePsdResult c = sourcepsd_compute(ds, sub, o2);
    CHECK(!c.hit && c.info["replaced"].get<bool>());
    CHECK(c.maps.maxCoeff() < 1.0 && c.maps.minCoeff() > 0);
    ds.close();
    fs::remove_all(root);
  }
  return nxrtest::finish("test_source_psd");
}

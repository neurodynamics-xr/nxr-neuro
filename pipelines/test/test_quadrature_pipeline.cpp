// quadrature_compute (nxr-neuro quadrature compute) on a small synthetic dataset, for a FREE kernel (three orientation rows per source: vector3 pair maps) and a FIXED one: the fraction, the energy
// map and the top pairs equal the maths of nxr/quadrature.h on the N x N cross-spectrum; one field per map; rows and arrays together; cache hit; replace.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/io/measures.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/quadrature.h"
#include "nxr/welch.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

template <int ORIENT>
static void run() {
  std::printf("== %s kernel ==\n", ORIENT == 3 ? "free" : "fixed");
  {
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_quadrature_" + std::to_string(::getpid()));
    fs::remove_all(root);
    std::mt19937 rng(9);
    std::normal_distribution<double> nd;
    const std::int64_t M = 4, N = 12, T = 2400;     // N: the vertices of an icosahedron (the sources are a real surface, so tiles can be summarised)
    const int o3 = ORIENT;   // 3 free · 1 fixed
    const double sfreq = 100.0;
    MatrixXd X(M, T), K(o3 * N, M);
    for (auto& v : X.reshaped()) v = static_cast<float>(nd(rng));
    for (int t = 0; t < T; ++t) for (int c = 0; c < M; ++c) X(c, t) += 3.0 * std::sin(2 * M_PI * 10.0 * t / sfreq - 0.5 * c);   // channel c lags channel 0 by 0.5 c rad at 10 Hz
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
        rec.name = "rest"; rec.path = node_path(sub.db(), prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
        // the sources (3 points) and a kernel [3N x M] on all four channels
        const std::string sid = uuid7();
        std::vector<double> Vx; std::vector<std::int32_t> Fx;
        icosphere(0, Vx, Fx, 0.08);
        SurfaceOptions so; so.name = "cortex"; so.sphere = Vx;
        add_surface(sub, so, Vx, Fx);
        surf = text(*sub.by_name("manifold", "cortex"), "id");
        std::vector<std::int32_t> chans = {0, 1, 2, 3};
        SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
        const std::string sel = sub.selection(std::move(ss));
        std::vector<float> kf(static_cast<std::size_t>(o3 * N * M));
        for (int i = 0; i < o3 * N; ++i) for (int j = 0; j < M; ++j) kf[static_cast<std::size_t>(i * M + j)] = static_cast<float>(K(i, j));
        OperatorSpec op;
        op.name = "kern"; op.path = "kern"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = o3 * N; op.n_cols = M;
        op.data = view(kf, {o3 * N, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", o3 == 3 ? "vector3" : "scalar");
        sub.operator_(std::move(op));
      });
    }
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    QuadratureOptions o;
    o.recording = "rest";
    o.chunks = "all";                                           // every chunk, against the whole-record reference
    o.option_set = "brainstorm";                                // the direct reference below is Brainstorm's exact 4 s
    o.segment_seconds = 4.0;
    o.bands = {{"alpha", 8, 12}, {"beta", 15, 29}};
    o.pairs = 2;
    const QuadratureResult a = quadrature_compute(ds, sub, o);
    CHECK(!a.hit);
    // direct: the same cross-spectra, then the maths of nxr/quadrature.h on the N x N matrices
    auto wo = nxr::welch::brainstormPsdOptions(sfreq, 4.0);
    const double df = sfreq / wo.segmentSamples;
    std::vector<nxr::welch::BinGroup> gr;
    for (const auto& b : o.bands) gr.push_back({static_cast<std::size_t>(std::ceil(b.lo / df - 1e-9)), static_cast<std::size_t>(std::floor(b.hi / df + 1e-9)), true});
    nxr::welch::CrossWelch cw(M, wo, gr, true);
    cw.add(X);
    const auto cr = cw.result();
    for (int b = 0; b < 2; ++b) {
      using cd = std::complex<double>;
      const Eigen::MatrixXcd Sb = cr.S[b].cast<cd>() + cd(0, 1) * cr.Sq[b].cast<cd>();
      const Eigen::MatrixXcd C = K.cast<cd>() * Sb * K.transpose().cast<cd>();
      const Eigen::MatrixXd Im = nxr::quadrature::quadratureSpectrum(C);
      Eigen::VectorXd rows = Im.array().square().matrix().rowwise().sum(), per(N);
      for (int s = 0; s < N; ++s) { per[s] = 0; for (int o_ = 0; o_ < o3; ++o_) per[s] += rows[o3 * s + o_]; }
      std::printf("  band %d: quadratureFraction %.4f (direct %.4f); energy vs direct %.2e\n", b, a.fraction[b], nxr::quadrature::quadratureFraction(C), (a.energy.col(b) - per).cwiseAbs().maxCoeff() / per.cwiseAbs().maxCoeff());
      CHECK(std::abs(a.fraction[b] - nxr::quadrature::quadratureFraction(C)) < 1e-6);
      CHECK((a.energy.col(b) - per).cwiseAbs().maxCoeff() < 1e-6 * per.cwiseAbs().maxCoeff());
      const auto ref = nxr::quadrature::quadraturePairs(nxr::quadrature::quadratureFactor(nxr::quadrature::hermitianFactor(C)), 2);
      CHECK(!a.pairs[b].empty() && a.pairs[b].size() == ref.size());
      if (!ref.empty()) {
        CHECK(std::abs(a.pairs[b][0].sigma - ref[0].sigma) < 1e-6 * ref[0].sigma);
        CHECK((a.pairs[b][0].p * a.pairs[b][0].q.transpose() - a.pairs[b][0].q * a.pairs[b][0].p.transpose() - (ref[0].p * ref[0].q.transpose() - ref[0].q * ref[0].p.transpose())).cwiseAbs().maxCoeff() < 1e-6);
      }
    }
    CHECK(a.fraction[0] > a.fraction[1]);   // the lagged 10 Hz sine is in alpha
    CHECK(a.pairs[0].size() >= 1 && a.pairs[0][0].p.size() == o3 * N);
    CHECK(a.share[0][0] > 0.5);
    // storage: energy + 2 pairs x (p, q) per band; vector3 for a free kernel; rows and arrays together
    std::size_t expected = 0;   // an energy map per band and p, q for each pair there is (3 fixed-orientation sources have one pair at most)
    for (const auto& pb : a.pairs) expected += 1 + 2 * pb.size();
    CHECK_EQ(a.info["maps"].size(), expected);
    // TWO MAP STATISTICS (2026-11-13): the energy (vertex x band, scalar) and the pairs (vertex x patterns, vector3 for a free kernel); each map read by its old name
    const StoredStatistic es = read_statistic(sub, a.info["statistics"][0].get<std::string>()), ps = read_statistic(sub, a.info["statistics"][1].get<std::string>());
    CHECK(text(es.row, "measure") == "quadrature_energy" && text(es.row, "value_type") == "scalar" && es.axes[1]["values"][0].contains("quadrature_fraction"));
    CHECK(text(ps.row, "measure") == "quadrature_pattern" && text(ps.row, "value_type") == (o3 == 3 ? "vector3" : "scalar"));
    CHECK(ps.axes[1]["values"][0]["rotary_sense"].get<int>() == 1 && ps.axes[1]["values"][0].contains("sigma"));
    int found = 0;
    for (const auto& n : a.info["maps"]) {
      const auto m = stored_map(sub, n.get<std::string>(), N);       // the map of a statistic, by the name its Field had
      CHECK(m.has_value());
      if (!m) continue;
      ++found;
      CHECK(m->cols() == ((n.get<std::string>().find("_energy_") != std::string::npos || o3 != 3) ? 1 : 3));
    }
    std::printf("  %d maps stored\n", found);
    CHECK(found >= 2 * 3);
    const QuadratureResult h = quadrature_compute(ds, sub, o);
    CHECK(h.hit);
    CHECK(std::abs(h.info["bands"][0]["quadrature_fraction"].get<double>() - a.fraction[0]) < 1e-12);
    bool refused = false;
    QuadratureOptions o2 = o;
    o2.pairs = 1;
    try { quadrature_compute(ds, sub, o2); } catch (const Error&) { refused = true; }   // same names, other options: refused until --replace
    CHECK(refused);
    o2.replace = true;
    const QuadratureResult c = quadrature_compute(ds, sub, o2);
    CHECK(!c.hit && c.info["replaced"].get<bool>());
    {   // --tiles: the energy maps and (free kernel) the pairs' patterns summarised over tiles
      QuadratureOptions o4 = o2;
      o4.tiles = true; o4.tile_max_level = 2;
      const QuadratureResult t = quadrature_compute(ds, sub, o4);
      CHECK(t.hit && t.info["tiles"]["levels"].size() == 2 && t.info["tiles"]["scalars"].size() == 2);
      {   // the energy maps are maps of a statistic's slice: each has a BYTE-LESS Field (no path), whose array-backed rows (weight, sum, ..., no `mean`) are the map's summary; the mean is derived at read
        const auto pr = sub.db().one("SELECT s.id AS id FROM selection s WHERE s.subject_id = ? AND s.name LIKE '% tiles L2' AND s.type = 'set'", {V(sub.id())});
        CHECK(pr.has_value());
        const auto bf = sub.db().all("SELECT id, name FROM field WHERE subject_id = ? AND kind = 'tile map' AND path IS NULL", {V(sub.id())});
        CHECK_EQ(static_cast<int>(bf.size()), 2);
        if (pr && !bf.empty()) {
          const std::string fid = text(bf[0], "id");
          CHECK(sub.db().one("SELECT measure FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'weight' AND array_path IS NOT NULL", {V(text(*pr, "id")), V(fid)}).has_value());
          CHECK(!sub.db().one("SELECT measure FROM selection_measurement WHERE selection_id = ? AND measure = 'mean'", {V(text(*pr, "id"))}).has_value());
          const Json q = Json{{"selectionId", text(*pr, "id")}, {"measure", "mean"}, {"ofFieldId", fid}, {"spaceLevel", 2}};
          CHECK(!query_measurements(sub.db(), sub.store(), q).empty());     // derived at read from the sums
        }
      }
      CHECK(ORIENT == 3 ? t.info["tiles"]["vectors"].size() >= 2 * 2 * 1 : t.info["tiles"]["vectors"].size() == 0);   // pairs of a free kernel are vector maps
    }
    bool bad = false;
    QuadratureOptions o3x = o; o3x.bands = {{"alpha", 8, 12}}; o3x.pairs = 0;
    try { quadrature_compute(ds, sub, o3x); } catch (const Error&) { bad = true; }
    CHECK(bad);
    ds.close();
    fs::remove_all(root);
  }
}

int main() {
  run<3>();
  run<1>();
  return nxrtest::finish("test_quadrature_pipeline");
}

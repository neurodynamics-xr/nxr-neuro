// The speed spectrum pipeline (a synthetic tone on all channels), the rows and arrays stored, a cache hit, a fixed-orientation kernel refused. The maths: compute/test/test_speedspectrum.cpp. The pipeline on a small synthetic dataset (an icosphere, its solved package, a free-orientation kernel on 20 channels):
// the stored fields equal diag(B S B^T) from the cross-spectrum route computed here from the same package, a cache hit, a fixed-orientation kernel refused.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/io/task/task.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/hodge_weights.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/speedspectrum.h"
#include "nxr/welch.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_speedspectrum_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<double> Vx;
  std::vector<std::int32_t> Fx;
  icosphere(1, Vx, Fx, 0.08);                                  // 42 vertices
  const std::int64_t M = 20, N = static_cast<std::int64_t>(Vx.size() / 3), T = 1600;
  const double sfreq = 100.0;
  MatrixXd X(M, T), K(3 * N, M), Kfix(N, M);
  for (auto& v : X.reshaped()) v = static_cast<float>(nd(rng));
  for (int t = 0; t < T; ++t) X.col(t) += Eigen::VectorXd::Constant(M, 3.0 * std::sin(2 * M_PI * 10.0 * t / sfreq));
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  for (auto& v : K.reshaped()) v = static_cast<float>(nd(rng) * 1e-3);
  K = K.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  for (auto& v : Kfix.reshaped()) v = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { add_surface(sub, {.name = "cortex"}, Vx, Fx); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  EigenSolveOptions eo;
  eo.modes = 16;
  const EigenSolveResult pk = eigen_solve(ds, sub, eo);
  const SurfaceInput surf = read_surface(sub, std::string("cortex"));
  {
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
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", std::vector<std::int64_t>(M, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(M, 0), {"good"}, {}}}, cid, "the montage families", std::nullopt,
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
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto add_kernel = [&](const std::string& name, const MatrixXd& Km, const char* value_type) {
        std::vector<float> kf(static_cast<std::size_t>(Km.rows() * M));
        for (int i = 0; i < Km.rows(); ++i) for (int j = 0; j < M; ++j) kf[static_cast<std::size_t>(i * M + j)] = static_cast<float>(Km(i, j));
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = Km.rows(); op.n_cols = M;
        op.data = view(kf, {Km.rows(), M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf.id).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", value_type);
        sub.operator_(std::move(op));
      };
      add_kernel("kern", K, "vector3");
      add_kernel("fixed", Kfix, "scalar");
    });
  }
  SpeedSpectrumOptions o;
  o.recording = "rest";
  o.chunks = "0";                      // the first chunk's samples, stated
  o.kernel = "kern";
  o.segment_seconds = 4.0;
  o.fmin = 2; o.fmax = 40;
  o.graph_family = nxr::filterbank::Family::LogItersine;
  o.family = nxr::timebank::Family::Itersine;
  o.gain = 1.0;
  o.sides = nxr::timebank::Sides::Analytic;
  o.speed.speedBins = 30;
  bool refused_family = false;
  { SpeedSpectrumOptions bad = o; bad.family.reset(); try { speedspectrum_compute(ds, sub, bad); } catch (const Error& e) { refused_family = std::string(e.what()).find("family is required") != std::string::npos; } }
  CHECK(refused_family);
  const SpeedSpectrumResult a = speedspectrum_compute(ds, sub, o);
  CHECK(!a.hit);
  CHECK_EQ(a.info["statistics"].size(), std::size_t(8));
  CHECK(a.result.parts.size() == 2);
  // tiles of a tight bank sum to the power of the bins (2-40 Hz) inside the bank's support, per kind; the histogram holds the same energy
  for (const auto& p : a.result.parts) {
    CHECK(p.power > 0);
    CHECK(p.tileEnergy.sum() <= 1.0000001 * p.power);
    const double hist = p.histogram.sum() + p.outside.sum();
    std::printf("  kind %d: power %.4g, tiles %.4g, histogram %.4g\n", p.kind, p.power, p.tileEnergy.sum(), hist);
    CHECK(std::abs(hist - p.power) / p.power < 1e-6);
    CHECK(p.quadrature.rows() == p.tileEnergy.rows());
  }
  // STATISTICS (2026-11-13): each part on the axes graph member x time member (the histogram: x speed bin), from the Welch segment partition
  {
    const StoredStatistic st = read_statistic(sub, a.info["statistics"][0].get<std::string>());
    CHECK(st.shape == (std::vector<std::int64_t>{static_cast<std::int64_t>(a.result.parts[0].tileEnergy.rows()), static_cast<std::int64_t>(a.result.parts[0].tileEnergy.cols())}));
    CHECK(text(st.row, "measure") == "power" && st.axes[0]["name"] == "graph_member" && st.axes[1]["name"] == "time_member" && !is_null(st.row, "partition_selection_id"));
    const StoredStatistic h = read_statistic(sub, a.info["statistics"][3].get<std::string>());
    CHECK(h.axes[1]["name"] == "speed_bin" && text(read_statistic(sub, a.info["statistics"][1].get<std::string>()).row, "measure") == "speed");
  }
  CHECK(speedspectrum_compute(ds, sub, o).hit);
  bool refused = false;
  SpeedSpectrumOptions fx = o;
  fx.kernel = "fixed";
  try { speedspectrum_compute(ds, sub, fx); } catch (const Error& e) { refused = std::string(e.what()).find("fixed-orientation") != std::string::npos; }
  CHECK(refused);
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_speedspectrum_pipeline");
}

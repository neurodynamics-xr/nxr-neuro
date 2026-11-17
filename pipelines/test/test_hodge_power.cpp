// Hodge-Dirac eigenmode power of source maps, the pipeline on a small synthetic dataset (an icosphere, its solved package, a free-orientation kernel on 20 channels):
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
#include "nxr/pipelines/tiles.h"
#include "nxr/welch.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_hodgepower_" + std::to_string(::getpid()));
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
  HodgePowerOptions o;
  o.recording = "rest";
  o.chunks = "all";                                             // every chunk, against the whole-record reference
  o.kernel = "kern";
  o.option_set = "brainstorm";                                // the direct reference below is Brainstorm's exact 4 s
  o.segment_seconds = 4.0;
  o.bands = {{"alpha", 8, 12}, {"beta", 15, 29}};
  const HodgePowerResult a = hodgepower_compute(ds, sub, o);
  CHECK(!a.hit);
  CHECK_EQ(a.info["maps"].size(), std::size_t(4));
  // the same from the package and the cross-spectrum, here
  const HodgeDirac& h = pk.modes;
  const std::int64_t Kn = h.n_modes();
  nxr::hodge::Basis b;
  b.P.resize(Kn, N);
  for (std::int64_t v = 0; v < N; ++v) for (std::int64_t k = 0; k < Kn; ++k) b.P(k, v) = h.eigenvectors[static_cast<std::size_t>(v * Kn + k)];
  b.mu = Eigen::Map<const Eigen::VectorXd>(h.mu.data(), Kn);
  for (auto k : h.kind) b.kind.push_back(k);
  b.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(surf.vertices.data(), N, 3);
  b.faces = Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>>(surf.faces.data(), surf.n_faces(), 3);
  for (std::int64_t i = 0; i < N; ++i) b.keep.push_back(i);
  nxr::manifold::IntrinsicSurface isurf(surf.vertices.data(), static_cast<int>(N), surf.faces.data(), static_cast<int>(surf.n_faces()));
  const std::vector<double> dm_mass = isurf.inputMixedVoronoiMass();
  b.mass = Eigen::Map<const Eigen::VectorXd>(dm_mass.data(), static_cast<Eigen::Index>(dm_mass.size()));
  const MatrixXd B = nxr::hodge::hodgeWeights(b, K);
  auto wo = nxr::welch::brainstormPsdOptions(sfreq, 4.0);
  const auto direct = nxr::welch::welch(B * X, wo);
  const double df = sfreq / wo.segmentSamples;
  MatrixXd want(2 * Kn, 2);
  const double lo[2] = {8, 15}, hi[2] = {12, 29};
  for (int j = 0; j < 2; ++j) {
    const auto f0 = static_cast<Eigen::Index>(std::ceil(lo[j] / df - 1e-9)), f1 = static_cast<Eigen::Index>(std::floor(hi[j] / df + 1e-9));
    want.col(j) = direct.psd.middleCols(f0, f1 - f0 + 1).rowwise().mean();
  }
  const double err = (a.power - want).cwiseAbs().maxCoeff() / want.cwiseAbs().maxCoeff();
  std::printf("  pipeline mode power vs Welch on B x: %.2e\n", err);
  CHECK(err < 1e-6);
  // stored: a MAP STATISTIC (2026-11-13) on the eigen axis, a band's current and flow parts its entries; each part's map read by the name its Field had
  {
    const StoredStatistic st = read_statistic(sub, a.info["statistic"].get<std::string>());
    const auto axis = sub.find("manifold", "eigenvalue/hodgeDirac");
    CHECK(axis.has_value() && st.axes[0]["manifold_id"] == text(*axis, "id") && st.shape == (std::vector<std::int64_t>{Kn, 4}) && text(st.row, "measure") == "power_density");
    CHECK(st.axes[1]["values"][0]["part"] == "current" && st.axes[1]["values"][1]["part"] == "flow");
    const auto m = stored_map(sub, a.info["maps"][0].get<std::string>(), Kn);
    CHECK(m.has_value() && m->rows() == Kn);
    double w = 0;
    if (m) for (std::int64_t k = 0; k < Kn; ++k) w = std::max(w, std::abs((*m)(k, 0) - want(k, 0)) / std::max(1e-30, std::abs(want(k, 0))));
    CHECK(w < 1e-6);
  }
  CHECK(hodgepower_compute(ds, sub, o).hit);
  bool refused = false;
  HodgePowerOptions fx = o;
  fx.kernel = "fixed";
  try { hodgepower_compute(ds, sub, fx); } catch (const Error& e) { refused = std::string(e.what()).find("fixed-orientation") != std::string::npos; }
  CHECK(refused);
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_hodge_power");
}

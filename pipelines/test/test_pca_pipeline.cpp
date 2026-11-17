// compute pca (nxr-pipelines pca.h, spec docs/specs/2026-11-16-meg-fingerprinting.md, 4.4) on synthetic statistics: a table of members x regions x frequencies is read with a named or numbered
// members axis (the other axes flattened in order), decomposed, and stored as the four statistics of method `pca` (components, scores, variance, centring); read back, the full-rank reconstruction
// is the table and the components equal nxr::pca's on the same matrix; a float32 table is stored in float32; unit-variance scaling is recorded in the centring; a re-run is a cache hit; refusals.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/io/statistic.h"
#include "nxr/pca.h"
#include "nxr/pipelines/pca.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

static std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_pca_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  const std::int64_t n = 12, R = 5, F = 4;                       // members x regions x frequencies
  std::vector<double> tab(static_cast<std::size_t>(n * R * F));
  {   // two planted directions plus noise, so the components are not degenerate
    std::vector<double> a(R * F), b(R * F);
    for (auto& v : a) v = nd(rng);
    for (auto& v : b) v = nd(rng);
    for (std::int64_t i = 0; i < n; ++i) {
      const double u = 3 * nd(rng), w = 1.5 * nd(rng);
      for (std::int64_t j = 0; j < R * F; ++j) tab[static_cast<std::size_t>(i * R * F + j)] = 10 + 0.1 * j + u * a[static_cast<std::size_t>(j)] + w * b[static_cast<std::size_t>(j)] + 0.2 * nd(rng);
    }
  }
  const std::vector<float> tab32(tab.begin(), tab.end());
  DatasetWriter ds = create_dataset(root, "demo");
  std::string fid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    std::vector<float> x(100, 1.0f);
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 99);
      const std::string tid = sub.manifold(t);
      FieldSpec f;
      f.name = "rec"; f.path = node_path(sub.db(), tid, "rec"); f.kind = "recording"; f.manifold_id = tid; f.data = view(x, {100});
      fid = sub.field(f);
      auto named = [](const char* stem, std::int64_t count) { Json v = Json::array(); for (std::int64_t i = 0; i < count; ++i) v.push_back(Json{{"name", std::string(stem) + std::to_string(i)}}); return v; };
      auto axes = [&](const std::string& first) {
        return Json::array({Json{{"name", first}, {"size", n}, {"values", named("p", n)}}, Json{{"name", "region"}, {"size", R}, {"values", named("r", R)}},
                            Json{{"name", "frequency"}, {"size", F}, {"unit", "Hz"}, {"values", Json::array({8.0, 9.0, 10.0, 11.0})}}});
      };
      StatisticSpec s;
      s.name = "tab"; s.measure = "power_density"; s.method = "welch"; s.of_field_id = fid; s.unit = "T^2/Hz"; s.n_members = n;
      s.axes = axes("person"); s.data = view(tab, {n, R, F});
      sub.statistic(s);
      StatisticSpec s32 = s;
      s32.name = "tab32"; s32.data = view(tab32, {n, R, F});
      sub.statistic(s32);
      StatisticSpec sc = s;                                       // a complex statistic
      std::vector<double> c(static_cast<std::size_t>(n * 2), 1.0);
      sc.name = "cplx"; sc.value_type = "complex"; sc.axes = Json::array({Json{{"name", "person"}, {"size", n}, {"values", named("p", n)}}}); sc.data = view(c, {n, 2});
      sub.statistic(sc);
      StatisticSpec sn = s;                                       // a table with a NaN
      std::vector<double> nan_tab = tab;
      nan_tab[7] = std::nan("");
      sn.name = "tabnan"; sn.data = view(nan_tab, {n, R, F});
      sub.statistic(sn);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  auto matrix_of = [&](int members_axis) {                       // the table as members x features, by hand
    MatrixXd X;
    if (members_axis == 0) { X.resize(n, R * F); for (int i = 0; i < n; ++i) for (int j = 0; j < R * F; ++j) X(i, j) = tab[static_cast<std::size_t>(i * R * F + j)]; }
    else { X.resize(R, n * F); for (int r = 0; r < R; ++r) for (int i = 0; i < n; ++i) for (int f = 0; f < F; ++f) X(r, i * F + f) = tab[static_cast<std::size_t>((i * R + r) * F + f)]; }
    return X;
  };

  // members on axis 0 (by name): the decomposition of the 12 x 20 table
  PcaComputeOptions o;
  o.of = "tab"; o.members = "person"; o.name = "tab_pca";
  const PcaComputeResult r = pca_compute(ds, sub, o);
  CHECK(!r.hit && r.info["members"] == n && r.info["features"] == R * F && r.info["rank"] == n - 1);   // 11 components after centring 12 members
  {
    const nxr::pca::PcaResult want = nxr::pca::pca(matrix_of(0));
    const StoredPca s = pca_read(sub, "tab_pca");
    const double ec = (s.components - want.components).cwiseAbs().maxCoeff(), es = (s.scores - want.scores).cwiseAbs().maxCoeff() / want.scores.cwiseAbs().maxCoeff();
    const double ev = (s.explained_variance - want.explainedVariance).cwiseAbs().maxCoeff() / want.explainedVariance.maxCoeff();
    std::printf("  members x features: components %.2e, scores %.2e, variance %.2e\n", ec, es, ev);
    CHECK(ec < 1e-13 && es < 1e-13 && ev < 1e-13 && s.feature_shape == (std::vector<std::int64_t>{R, F}));
    const MatrixXd back = nxr::pca::reconstruct(s.scores, s.components, s.mean, s.scale, static_cast<int>(s.components.rows()));
    CHECK((back - matrix_of(0)).cwiseAbs().maxCoeff() < 1e-10);
    CHECK(std::abs(s.explained_variance_ratio.sum() - 1.0) < 1e-12 && s.options["scale"] == "none" && s.options["rank"] == n - 1);
    const StoredStatistic comp = read_statistic(sub, "tab_pca"), sc = read_statistic(sub, "tab_pca_scores"), var = read_statistic(sub, "tab_pca_variance"), cen = read_statistic(sub, "tab_pca_centring");
    CHECK(text(comp.row, "method") == "pca" && text(comp.row, "measure") == "principal_components" && comp.shape == (std::vector<std::int64_t>{n - 1, R, F}));
    CHECK(comp.axes[0]["name"] == "component" && comp.axes[1]["name"] == "region" && comp.axes[2]["name"] == "frequency");
    CHECK(sc.shape == (std::vector<std::int64_t>{n, n - 1}) && sc.axes[0]["name"] == "person" && text(sc.row, "unit") == "T^2/Hz" && text(sc.row, "measure") == "principal_scores");
    CHECK(var.shape == (std::vector<std::int64_t>{n - 1, 3}) && cen.shape == (std::vector<std::int64_t>{2, R, F}) && text(cen.row, "of_statistic_id") == text(read_statistic(sub, "tab").row, "id"));
    CHECK(ArrayReader::open(sub.store(), text(comp.row, "path")).dtype() == DType::Float64);
    // the sign convention holds in what is stored
    bool sign_ok = true;
    for (int c = 0; c < s.components.rows(); ++c) { Eigen::Index at = 0; s.components.row(c).cwiseAbs().maxCoeff(&at); sign_ok = sign_ok && s.components(c, at) > 0; }
    CHECK(sign_ok);
    CHECK(pca_compute(ds, sub, o).hit);                           // a re-run is a hit
  }
  // members given by position, on the middle axis: 5 members (regions), 48 features (members x frequencies)
  {
    PcaComputeOptions p2; p2.of = "tab"; p2.members = "1"; p2.name = "tab_pca_region";
    pca_compute(ds, sub, p2);
    const nxr::pca::PcaResult want = nxr::pca::pca(matrix_of(1));
    const StoredPca s = pca_read(sub, "tab_pca_region");
    CHECK(s.components.rows() == R - 1 && s.components.cols() == n * F && s.feature_shape == (std::vector<std::int64_t>{n, F}));
    CHECK((s.components - want.components).cwiseAbs().maxCoeff() < 1e-13 && (s.scores - want.scores).cwiseAbs().maxCoeff() < 1e-10);
  }
  // unit variance: recorded in the centring (the divisor), and the result is nxr::pca's with that option
  {
    PcaComputeOptions p3 = o; p3.name = "tab_pca_uv"; p3.scale = "unit-variance";
    pca_compute(ds, sub, p3);
    nxr::pca::PcaOptions po; po.scale = nxr::pca::Scale::UnitVariance;
    const nxr::pca::PcaResult want = nxr::pca::pca(matrix_of(0), po);
    const StoredPca s = pca_read(sub, "tab_pca_uv");
    CHECK((s.scale - want.scale).cwiseAbs().maxCoeff() < 1e-13 && s.scale.minCoeff() > 0 && (s.components - want.components).cwiseAbs().maxCoeff() < 1e-12 && s.options["scale"] == "unit-variance");
    CHECK(code_of([&] { pca_compute(ds, sub, p3); }) == "" );   // same options: a hit, not an error
    // a different option under the same name is refused until --replace
    PcaComputeOptions p4 = p3; p4.scale = "none";
    CHECK(code_of([&] { pca_compute(ds, sub, p4); }) != "");
    p4.replace = true;
    CHECK(!pca_compute(ds, sub, p4).hit);
  }
  // storage precision follows the input: a float32 table gives float32 components, scores and centring; the variance table stays float64
  {
    PcaComputeOptions p5; p5.of = "tab32"; p5.members = "person"; p5.name = "tab32_pca";
    pca_compute(ds, sub, p5);
    CHECK(ArrayReader::open(sub.store(), text(read_statistic(sub, "tab32_pca").row, "path")).dtype() == DType::Float32);
    CHECK(ArrayReader::open(sub.store(), text(read_statistic(sub, "tab32_pca_scores").row, "path")).dtype() == DType::Float32);
    CHECK(ArrayReader::open(sub.store(), text(read_statistic(sub, "tab32_pca_centring").row, "path")).dtype() == DType::Float32);
    CHECK(ArrayReader::open(sub.store(), text(read_statistic(sub, "tab32_pca_variance").row, "path")).dtype() == DType::Float64);
    const StoredPca s = pca_read(sub, "tab32_pca");
    const MatrixXd back = nxr::pca::reconstruct(s.scores, s.components, s.mean, s.scale, static_cast<int>(s.components.rows()));
    CHECK((back - matrix_of(0)).cwiseAbs().maxCoeff() < 1e-3);   // float32 storage: ~1e-6 relative
  }
  // refusals, each with its reason
  {
    PcaComputeOptions b; b.of = "tab"; b.name = "x";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);                       // neither form
    b.members = "person"; b.over = "parcels";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);                       // both
    b.over.reset(); b.members = "nonesuch";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);
    b.members = "7";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);
    b.members = "person"; b.scale = "z-score";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);
    b.scale = "none"; b.of = "cplx";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);                       // complex
    b.of = "tabnan";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) == code::invalid_argument);                       // a NaN
    b.of = "absent";
    CHECK(code_of([&] { pca_compute(ds, sub, b); }) != "");
    CHECK(code_of([&] { pca_read(sub, "tab"); }) == code::invalid_argument);                          // not a pca
  }
  ds.close();
  return nxrtest::finish("test_pca_pipeline");
}

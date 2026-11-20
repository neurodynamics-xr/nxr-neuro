// THE SOURCE MODEL OF MEG FINGERPRINTING end to end on a small synthetic dataset (no real data): a head model (a random gain on a cortex icosphere, 16 MEG channels of two types), a
// recording and an empty room, the data and noise covariances (`run noisecov`), the LCMV kernel (`run lcmv`), the span's covariance and the region map (`run region_pca`) over 4 regions.
// Checks the wiring and the record, the maths being held to Brainstorm by meeg_test_beamformer_parity: the stored kernel is nxr::meeg::lcmvKernel's on the same inputs (oriented by the
// cortex's normals), stored as a fixed-orientation kernel with its whiteners and the method's citation; the region map and offsets are the meeg functions' on the stored kernel and
// covariance, the region signal is the first principal component of the region's sources, and the result does not depend on how the recording is chunked; the refusals name their reason.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <random>

#include "check.h"
#include "meeg_runs.h"
#include "nxr/bst/entities.h"
#include "nxr/geometry/frames.h"
#include "nxr/io/task/task.h"
#include "nxr/meeg/beamformer.h"
#include "nxr/meeg/scouts.h"

using namespace nxr::io;
using namespace nxrtest;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {
void icosphere1(double radius, std::vector<double>& V, std::vector<std::int32_t>& F) {
  const double t = (1 + std::sqrt(5.0)) / 2;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  const int tri[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                          {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  V.clear(); F.clear();
  for (auto& b : base) V.insert(V.end(), b, b + 3);
  for (auto& f : tri) F.insert(F.end(), f, f + 3);
  std::map<std::pair<int, int>, int> mid;
  auto midpoint = [&](int a, int b) {
    auto k = std::make_pair(std::min(a, b), std::max(a, b));
    auto it = mid.find(k);
    if (it != mid.end()) return it->second;
    const int id = static_cast<int>(V.size() / 3);
    for (int c = 0; c < 3; ++c) V.push_back((V[3 * a + c] + V[3 * b + c]) / 2);
    return mid[k] = id;
  };
  std::vector<std::int32_t> G;
  for (std::size_t f = 0; f < F.size(); f += 3) {
    const int a = F[f], b = F[f + 1], c = F[f + 2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
    for (int q : {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca}) G.push_back(q);
  }
  F = G;
  for (std::size_t i = 0; i < V.size(); i += 3) {
    const double n = std::sqrt(V[i] * V[i] + V[i + 1] * V[i + 1] + V[i + 2] * V[i + 2]);
    for (int c = 0; c < 3; ++c) V[i + static_cast<std::size_t>(c)] *= radius / n;
  }
}
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception& e) { return std::string("other: ") + e.what(); } return ""; }
double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
MatrixXd stored_operator(SubjectWriter& sub, const std::string& name) { return read_matrix(sub, text(*sub.by_name("operator", name), "path")); }
MatrixXd stored_statistic(SubjectWriter& sub, const std::string& name) {
  const StoredStatistic st = read_statistic(sub, name);
  const std::int64_t r = st.shape.at(0), c = st.shape.size() > 1 ? st.shape[1] : 1;
  MatrixXd m(r, c);
  for (std::int64_t i = 0; i < r; ++i) for (std::int64_t j = 0; j < c; ++j) m(i, j) = st.values[static_cast<std::size_t>(i * c + j)];
  return m;
}
Json producer_of(SubjectWriter& sub, const char* table, const std::string& name) { return Json::parse(opt_text(*sub.by_name(table, name), "producer_json").value_or("{}"), nullptr, false); }
bool cites(const Json& producer) {
  if (!producer.contains("references")) return false;
  const std::string all = producer["references"].dump();
  return all.find("Van Veen") != std::string::npos && all.find("Castanheira") != std::string::npos && all.find("2018-10-12") != std::string::npos;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_lcmv_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(23);
  std::normal_distribution<double> nd;

  std::vector<double> V;
  std::vector<std::int32_t> F;
  icosphere1(0.07, V, F);                                      // 42 sources
  const std::int64_t nv = static_cast<std::int64_t>(V.size() / 3), M = 16, T = 4000;
  const double sfreq = 250.0;
  std::vector<std::string> types;
  for (int i = 0; i < M; ++i) types.push_back(i < 8 ? "MEG MAG" : "MEG GRAD");
  const nxr::geometry::Mat3 nrm = nxr::geometry::vertexNormalsPatch(V.data(), static_cast<int>(nv), F.data(), static_cast<int>(F.size() / 3));
  MatrixXd normals(nv, 3);
  for (int i = 0; i < nv; ++i) normals.row(i) = Eigen::RowVector3d(nrm(i, 0), nrm(i, 1), nrm(i, 2)).normalized();
  MatrixXd gain(M, 3 * nv);
  for (int i = 0; i < M; ++i) for (int j = 0; j < 3 * nv; ++j) gain(i, j) = (i < 8 ? 1e-13 : 5e-13) * nd(rng);
  gain = gain.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });   // the stored float32 gain
  // the recording: every source along its normal with its own time course (one planted at 10 Hz), plus channel noise; the empty room: channel noise only
  MatrixXd S(nv, T), X(M, T), E(M, T);
  for (int k = 0; k < nv; ++k) for (int t = 0; t < T; ++t) S(k, t) = 1e-9 * nd(rng) + (k == 5 ? 2e-8 * std::sin(2 * M_PI * 10 * t / sfreq) : 0.0);
  MatrixXd Gn(M, nv);
  for (int k = 0; k < nv; ++k) Gn.col(k) = gain.middleCols(3 * k, 3) * normals.row(k).transpose();
  X = Gn * S;
  for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) { X(i, t) += (i < 8 ? 1e-13 : 5e-13) * (0.5 * nd(rng) + 0.01 * i); E(i, t) = (i < 8 ? 1e-13 : 5e-13) * nd(rng) * (1 + 0.05 * i); }
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  E = E.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  // the regions: four, by the signs of x and z
  std::vector<std::int64_t> region(static_cast<std::size_t>(nv));
  for (int k = 0; k < nv; ++k) region[static_cast<std::size_t>(k)] = (V[3 * k] > 0 ? 1 : 0) + (V[3 * k + 2] > 0 ? 2 : 0);

  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      add_surface(sub, {.name = "cortex"}, V, F);
      const std::string surface_id = text(*sub.by_name("manifold", "cortex"), "id");
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3));
      for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g;
      g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors";
      m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      std::vector<std::int64_t> tcode, qcode(static_cast<std::size_t>(M), 0);
      for (const auto& t : types) tcode.push_back(t == "MEG MAG" ? 0 : 1);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", tcode, {"MEG MAG", "MEG GRAD"}, {}}, {"quality", qcode, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      std::vector<float> gf(static_cast<std::size_t>(M * 3 * nv));
      for (int i = 0; i < M; ++i) for (int j = 0; j < 3 * nv; ++j) gf[static_cast<std::size_t>(i * 3 * nv + j)] = static_cast<float>(gain(i, j));
      OperatorSpec op;
      op.name = "headmodel"; op.path = "headmodel"; op.kind = "forward"; op.from_manifold_id = surface_id; op.layout = "dense";
      op.n_rows = M; op.n_cols = 3 * nv; op.data = view(gf, {M, 3 * nv}); op.array.compress = true;
      op.cols.set("to_manifold_id", cid).set("from_value_type", "vector3").set("to_value_type", "scalar").set("description", "the leadfield");
      sub.operator_(std::move(op));
      nxr::bst::LabellingOptions lo;
      lo.manifold_id = surface_id; lo.dictionary = "regions"; lo.scope = "subject";
      nxr::bst::labelling(sub, "selection/regions", region, {"back-bottom", "front-bottom", "back-top", "front-top"}, lo);
      // the recordings on channels x time: the run twice, stored in chunks of 700 and of 1100 samples, and the empty room
      auto add_recording = [&](const std::string& name, const MatrixXd& Y, std::int64_t chunk) {
        ManifoldSpec tm;
        tm.name = name + "_time"; tm.path = "time/" + name + "_time"; tm.type = "line"; tm.n_vertices = T;
        tm.topology = {"lattice", 1, 1, std::nullopt, {}};
        const auto lc = lattice_counts({T});
        if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
        GeometrySpec tg;
        tg.form = "closed_form"; tg.cs_key = "clock";
        DimensionSpec d;
        d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
        tg.dimensions.push_back(d);
        tm.geometry = tg;
        const std::string tid = sub.manifold(std::move(tm));
        const std::string prod = sub.product({cid, tid});
        std::vector<float> xf(static_cast<std::size_t>(M * T));
        for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(Y(i, t));
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, chunk}; rec.array.compress = false;
        sub.field(std::move(rec));
      };
      add_recording("rest", X, 700);
      add_recording("rest_b", X, 1100);
      add_recording("emptyroom", E, 700);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  namespace g = nxr::pipelines::graph;

  // ── the covariances (step 11: the data covariance is the same statistic as the noise covariance, made from the run) ──
  NoiseCovOptions dc;
  dc.recording = "rest"; dc.name = "rest_data_cov"; dc.covariance.divisor = nxr::stats::CovarianceDivisor::SamplesMinusOne;
  dc.start = 0.0; dc.stop = static_cast<double>(T) / sfreq;
  const NoiseCovResult cd = noisecov_compute(ds, sub, dc);
  NoiseCovOptions nc = dc;
  nc.recording = "emptyroom"; nc.name = "emptyroom_noise_cov";
  const NoiseCovResult cn = noisecov_compute(ds, sub, nc);
  CHECK_EQ(static_cast<std::int64_t>(cd.channels.size()), M);

  // ── step 14: the LCMV kernel ──
  const Json lset{{"data_cov", "rest_data_cov"}, {"noise_cov", "emptyroom_noise_cov"}, {"name", "LCMV"}};
  const auto l1 = g::run(ds, sub, g::load("lcmv"), lset, span_options(std::nullopt, std::nullopt, std::nullopt, false));
  CHECK(!l1.empty() && !l1[0].value("hit", true));
  const MatrixXd K = stored_operator(sub, "LCMV");
  const nxr::meeg::LcmvKernel ref = nxr::meeg::lcmvKernel(gain, normals, cd.covariance, cn.covariance, types);
  CHECK(K.rows() == nv && K.cols() == M);
  CHECK(rel(K, ref.K) < 1e-5);                                  // stored as float32
  {
    const auto row = sub.by_name("operator", "LCMV");
    CHECK_EQ(text(*row, "kind"), std::string("inverse kernel"));
    CHECK_EQ(text(*row, "to_value_type"), std::string("scalar"));
    const Json pj = Json::parse(text(*row, "params_json"));
    CHECK_EQ(pj["inverse"]["measure"].get<std::string>(), std::string("LCMV"));
    CHECK_EQ(pj["inverse"]["orientation"].get<std::string>(), std::string("fixed"));
    CHECK_EQ(pj["inverse"]["noise"]["method"].get<std::string>(), std::string("median"));
    CHECK(cites(producer_of(sub, "operator", "LCMV")));
    const auto wn = sub.by_name("operator", "LCMV_whitener"), wd = sub.by_name("operator", "LCMV_data_whitener");
    CHECK(wn.has_value() && wd.has_value());
    CHECK_EQ(text(*wn, "of_statistic_id"), text(*sub.by_name("statistic", "emptyroom_noise_cov"), "id"));
    CHECK_EQ(text(*wd, "of_statistic_id"), text(*sub.by_name("statistic", "rest_data_cov"), "id"));
    CHECK(rel(stored_operator(sub, "LCMV_data_whitener"), ref.dataWhitener) < 1e-12);
  }
  CHECK(g::run(ds, sub, g::load("lcmv"), lset, span_options(std::nullopt, std::nullopt, std::nullopt, false))[0].value("hit", false));   // a second call hits the cache
  {
    Json other = lset;
    other["regularisation"] = "none";
    other["name"] = "LCMV_none";
    g::run(ds, sub, g::load("lcmv"), other, span_options(std::nullopt, std::nullopt, std::nullopt, false));
    nxr::meeg::LcmvOptions o;
    o.covariance.regularisation = nxr::meeg::TruncatedRegularisation::None;
    CHECK(rel(stored_operator(sub, "LCMV_none"), nxr::meeg::lcmvKernel(gain, normals, cd.covariance, cn.covariance, types, o).K) < 1e-5);
  }
  {   // the corner-angle normals: a named option, recorded in the provenance (and the cache key: a new result)
    Json other = lset;
    other["normals"] = "corner_angle";
    other["name"] = "LCMV_corner";
    CHECK(!g::run(ds, sub, g::load("lcmv"), other, span_options(std::nullopt, std::nullopt, std::nullopt, false))[0].value("hit", true));
    const nxr::geometry::Mat3 ca = nxr::geometry::vertexNormals(V.data(), static_cast<int>(nv), F.data(), static_cast<int>(F.size() / 3));
    MatrixXd cn3(nv, 3);
    for (int i = 0; i < nv; ++i) cn3.row(i) = Eigen::RowVector3d(ca(i, 0), ca(i, 1), ca(i, 2)).normalized();
    CHECK(rel(stored_operator(sub, "LCMV_corner"), nxr::meeg::lcmvKernel(gain, cn3, cd.covariance, cn.covariance, types).K) < 1e-5);
    CHECK_EQ(producer_of(sub, "operator", "LCMV_corner")["computed"]["normals"].get<std::string>(), std::string("corner_angle"));
    CHECK_EQ(producer_of(sub, "operator", "LCMV")["computed"]["normals"].get<std::string>(), std::string("brainstorm"));
  }
  CHECK(!code_of([&] { g::run(ds, sub, g::load("lcmv"), Json{{"noise_cov", "emptyroom_noise_cov"}}, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());   // no data covariance: refused
  {   // a noise covariance that does not cover the run's good channels: refused (both covariances are read on them)
    g::run(ds, sub, g::load("noisecov"), Json{{"recording", "emptyroom"}, {"name", "emptyroom_mag_cov"}, {"channel_types", Json::array({"MEG MAG"})}},
           span_options(std::nullopt, 0.0, static_cast<double>(T) / sfreq, false));
    CHECK(!code_of([&] { g::run(ds, sub, g::load("lcmv"), Json{{"data_cov", "rest_data_cov"}, {"noise_cov", "emptyroom_mag_cov"}, {"name", "LCMV_mag"}}, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());
    CHECK(!sub.by_name("operator", "LCMV_mag").has_value());
  }

  // ── conventions mne (MNE-Python's make_lcmv, held to MNE-Python by meeg_test_beamformer_mne_parity): the stored kernel is the meeg functions' on the same inputs ──
  {
    namespace m = nxr::meeg;
    const m::RankWhitening wn = m::rankWhitener(cn.covariance, types, {static_cast<int>(M)});   // one modality (MEG), no projectors: rank 'info' is every channel
    const Json mset{{"data_cov", "rest_data_cov"}, {"noise_cov", "emptyroom_noise_cov"}, {"name", "LCMV_mne"}, {"conventions", "mne"}};
    CHECK(!g::run(ds, sub, g::load("lcmv"), mset, span_options(std::nullopt, std::nullopt, std::nullopt, false))[0].value("hit", true));
    const MatrixXd Kf = stored_operator(sub, "LCMV_mne");
    const m::MneLcmvKernel free = m::mneLcmvKernel(gain, normals, cd.covariance, wn.W, static_cast<int>(M));   // free orientation, unit-noise-gain-invariant, reg 0.05
    CHECK(Kf.rows() == 3 * nv && Kf.cols() == M);
    CHECK(rel(Kf, free.K) < 1e-5);                              // stored as float32
    CHECK_EQ(text(*sub.by_name("operator", "LCMV_mne"), "to_value_type"), std::string("vector3"));
    CHECK(rel(stored_operator(sub, "LCMV_mne_whitener"), wn.W) < 1e-12);
    CHECK(!sub.by_name("operator", "LCMV_mne_data_whitener").has_value());   // MNE-Python does not whiten by the data covariance
    const Json pj = producer_of(sub, "operator", "LCMV_mne");
    CHECK_EQ(pj["computed"]["options"]["weight_norm"].get<std::string>(), std::string("unit-noise-gain-invariant"));
    CHECK_EQ(pj["computed"]["options"]["pick_ori"].get<std::string>(), std::string("free"));
    CHECK_EQ(pj["computed"]["rank"].get<int>(), static_cast<int>(M));
    CHECK(pj["references"].dump().find("make_lcmv") != std::string::npos);
    // max-power with the neural activity index: a new result (the options are in the cache key), the meeg function's
    Json mp = mset;
    mp["name"] = "LCMV_mne_mp"; mp["pick_ori"] = "max-power"; mp["weight_norm"] = "nai"; mp["reg"] = 0.1;
    CHECK(!g::run(ds, sub, g::load("lcmv"), mp, span_options(std::nullopt, std::nullopt, std::nullopt, false))[0].value("hit", true));
    m::MneLcmvOptions o;
    o.orientation = m::LcmvOrientation::MaxPower; o.weightNorm = m::LcmvWeightNorm::Nai; o.reg = 0.1;
    CHECK(rel(stored_operator(sub, "LCMV_mne_mp"), m::mneLcmvKernel(gain, normals, cd.covariance, wn.W, static_cast<int>(M), o).K) < 1e-5);
    CHECK_EQ(text(*sub.by_name("operator", "LCMV_mne_mp"), "to_value_type"), std::string("scalar"));
    // refused: no noise covariance with two sensor types (as MNE-Python); an option of the other conventions; brainstorm without a noise covariance
    CHECK(!code_of([&] { g::run(ds, sub, g::load("lcmv"), Json{{"data_cov", "rest_data_cov"}, {"conventions", "mne"}, {"name", "LCMV_adhoc"}}, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());
    CHECK(!code_of([&] { Json b = lset; b["name"] = "LCMV_bad"; b["pick_ori"] = "normal"; g::run(ds, sub, g::load("lcmv"), b, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());
    CHECK(!code_of([&] { Json b = mset; b["name"] = "LCMV_bad"; b["regularisation"] = "ridge"; g::run(ds, sub, g::load("lcmv"), b, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());
    CHECK(!code_of([&] { g::run(ds, sub, g::load("lcmv"), Json{{"data_cov", "rest_data_cov"}, {"name", "LCMV_bad"}}, span_options(std::nullopt, std::nullopt, std::nullopt, false)); }).empty());
    CHECK(!sub.by_name("operator", "LCMV_bad").has_value() && !sub.by_name("operator", "LCMV_adhoc").has_value());

    // ── a BAND-LIMITED data covariance by composition: bandpass (8-12 Hz) -> noisecov on its output -> lcmv (mne, normal orientation) ──
    g::RunOptions all;
    all.chunks = "all";
    g::run(ds, sub, g::load("bandpass"), Json{{"recording", "rest"}, {"lower", 8.0}, {"upper", 12.0}, {"remove_mean", false}, {"name", "rest_alpha"}}, all);
    g::run(ds, sub, g::load("noisecov"), Json{{"recording", "rest_alpha"}, {"name", "rest_alpha_cov"}}, all);
    const MatrixXd Ca = stored_statistic(sub, "rest_alpha_cov");
    CHECK(Ca.rows() == M && Ca.trace() < 0.5 * cd.covariance.trace());   // the band keeps a fraction of the broadband channel noise
    Json band = mset;
    band["data_cov"] = "rest_alpha_cov"; band["name"] = "LCMV_alpha"; band["pick_ori"] = "normal";
    g::run(ds, sub, g::load("lcmv"), band, span_options(std::nullopt, std::nullopt, std::nullopt, false));
    m::MneLcmvOptions on;
    on.orientation = m::LcmvOrientation::Normal;
    const MatrixXd Kb = stored_operator(sub, "LCMV_alpha");
    CHECK(rel(Kb, m::mneLcmvKernel(gain, normals, Ca, wn.W, static_cast<int>(M), on).K) < 1e-5);
    // one sensor type and no noise covariance: MNE-Python's identity covariance; no whitener is stored
    g::run(ds, sub, g::load("noisecov"), Json{{"recording", "rest_alpha"}, {"name", "rest_alpha_mag_cov"}, {"channel_types", Json::array({"MEG MAG"})}}, all);
    g::run(ds, sub, g::load("lcmv"), Json{{"data_cov", "rest_alpha_mag_cov"}, {"conventions", "mne"}, {"name", "LCMV_alpha_mag"}}, span_options(std::nullopt, std::nullopt, std::nullopt, false));
    const MatrixXd Cm = stored_statistic(sub, "rest_alpha_mag_cov");
    CHECK(rel(stored_operator(sub, "LCMV_alpha_mag"), m::mneLcmvKernel(gain.topRows(8), normals, Cm, MatrixXd::Identity(8, 8), 8).K) < 1e-5);
    CHECK(!sub.by_name("operator", "LCMV_alpha_mag_whitener").has_value());
  }

  // ── step 15: the region map over a span ──
  const double t0 = 2.0, t1 = 12.0;
  const std::int64_t s0 = 500, s1 = 3000;
  auto region_run = [&](const std::string& rec, const std::string& name) {
    NoiseCovOptions sc;
    sc.recording = rec; sc.name = rec + "_span_cov"; sc.start = t0; sc.stop = t1;
    noisecov_compute(ds, sub, sc);
    const Json rset{{"recording", rec}, {"kernel", "LCMV"}, {"covariance", rec + "_span_cov"}, {"partition", "regions"}, {"name", name}};
    return g::run(ds, sub, g::load("region_pca"), rset, span_options(std::nullopt, t0, t1, false));
  };
  const auto r1 = region_run("rest", "REG");
  CHECK(!r1.empty());
  const MatrixXd W = stored_operator(sub, "REG");
  const auto offsets = [&](const std::string& name) {
    const Json regions = producer_of(sub, "operator", name)["regions"];
    MatrixXd o(static_cast<Eigen::Index>(regions.size()), 1);
    for (std::size_t r = 0; r < regions.size(); ++r) o(static_cast<Eigen::Index>(r), 0) = regions[r].value("offset", 0.0);
    return o;
  };
  const MatrixXd off = offsets("REG");
  CHECK(W.rows() == 4 && W.cols() == M);
  CHECK(off.rows() == 4);
  {
    const auto row = sub.by_name("operator", "REG");
    CHECK_EQ(text(*row, "kind"), std::string("region map"));
    CHECK_EQ(text(*row, "to_selection_id"), text(*sub.by_name("selection", "regions"), "id"));
    CHECK(cites(producer_of(sub, "operator", "REG")));
  }
  // the meeg functions on the stored kernel and the span's covariance, and the region signal as the first principal component of the region's sources
  const MatrixXd C = stored_statistic(sub, "rest_span_cov");
  const MatrixXd Xs = X.middleCols(s0, s1 - s0);
  double worst = 0, worstPc = 0;
  for (int r = 0; r < 4; ++r) {
    std::vector<int> iv;
    for (int k = 0; k < nv; ++k) if (region[static_cast<std::size_t>(k)] == r) iv.push_back(k);
    MatrixXd Kr(static_cast<Eigen::Index>(iv.size()), M), Or(static_cast<Eigen::Index>(iv.size()), 3);
    for (std::size_t i = 0; i < iv.size(); ++i) { Kr.row(static_cast<Eigen::Index>(i)) = K.row(iv[i]); Or.row(static_cast<Eigen::Index>(i)) = normals.row(iv[i]); }
    const Eigen::VectorXd flip = nxr::meeg::scoutSignFlip(Or);
    const nxr::meeg::RegionComponent rc = nxr::meeg::regionComponent(Kr, C, flip);
    const Eigen::RowVectorXd z = rc.component * Xs, p = rc.peak * Xs;
    const double s = nxr::meeg::regionComponentSign(p.maxCoeff(), p.minCoeff(), p.mean(), z.maxCoeff(), z.minCoeff(), z.mean());
    worst = std::max({worst, rel(W.row(r), s * rc.component), std::abs(off(r, 0) - (p.mean() - s * z.mean())) / std::abs(p.mean())});
    // the signal y = map x + offset is the region's first principal component (up to its sign), with the peak source's mean
    const Eigen::RowVectorXd y = (W.row(r) * Xs).array() + off(r, 0);
    MatrixXd Fr = flip.asDiagonal() * Kr * Xs;
    Fr.colwise() -= Fr.rowwise().mean();
    const Eigen::JacobiSVD<MatrixXd> svd(Fr, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::RowVectorXd pc = svd.singularValues()(0) * svd.matrixV().col(0).transpose();
    const Eigen::RowVectorXd yc = y.array() - y.mean();
    worstPc = std::max(worstPc, std::min(rel(yc, pc), rel(yc, -pc)));
  }
  std::printf("  region map against the meeg functions: %.2e; region signal against the first principal component: %.2e\n", worst, worstPc);
  CHECK(worst < 1e-4);                                          // the map is stored as float32
  CHECK(worstPc < 1e-4);
  // the result does not depend on the chunking: the same samples stored in chunks of 1100
  region_run("rest_b", "REG_b");
  const double ec = rel(stored_operator(sub, "REG_b"), W), eo = rel(offsets("REG_b"), off);
  std::printf("  chunks of 700 against 1100: map %.2e, offsets %.2e\n", ec, eo);
  CHECK(ec < 1e-6 && eo < 1e-6);
  // refused: a covariance over another span than the run's; a free-orientation kernel
  {
    NoiseCovOptions sc;
    sc.recording = "rest"; sc.name = "rest_other_cov"; sc.start = 1.0; sc.stop = t1;
    noisecov_compute(ds, sub, sc);
    const Json bad{{"recording", "rest"}, {"kernel", "LCMV"}, {"covariance", "rest_other_cov"}, {"partition", "regions"}, {"name", "REG_bad"}};
    CHECK(!code_of([&] { g::run(ds, sub, g::load("region_pca"), bad, span_options(std::nullopt, t0, t1, false)); }).empty());
    CHECK(!sub.by_name("operator", "REG_bad").has_value());
    InverseComputeOptions io;
    io.noise_cov = "emptyroom_noise_cov"; io.name = "MN";
    inverse_compute(ds, sub, io);
    const Json free{{"recording", "rest"}, {"kernel", "MN"}, {"covariance", "rest_span_cov"}, {"partition", "regions"}, {"name", "REG_free"}};
    const std::string why = code_of([&] { g::run(ds, sub, g::load("region_pca"), free, span_options(std::nullopt, t0, t1, false)); });
    CHECK(!why.empty());
  }
  fs::remove_all(root);
  return nxrtest::finish("test_lcmv_pipeline");
}

// MEG BRAIN FINGERPRINTING END TO END on a synthetic cohort (no real data): the run order of `fingerprint_features.yaml` for each of four subjects, then the cohort statistics (`run fingerprint`).
// Each subject: a head model (a random gain on a cortex icosphere, 48 MEG channels of two types), four regions, a recording of 120 s at 100 Hz and an empty room. Its regions' sources carry an
// alpha rhythm (its peak frequency the subject's own) and a 20 Hz rhythm with amplitudes of its own (the planted identity of its spectrum) and slow amplitude modulations shared between regions in a pattern of its own (the planted identity of
// its connectome), plus sensor noise. Per band (alpha 8-13 Hz, beta 13-30 Hz): the band-pass of the run and the empty room (Brainstorm 2018's design), the data and noise covariances, the LCMV kernel;
// per half (A = [5 s, floor(T/2)], B = [ceil(T/2), T - 5 s]): the span's covariance, the region map, `fingerprint_band`; per half `fingerprint_features` (two bands: their cuts 8-12.5 and 13-29.5 Hz).
// Checks: the features' shapes, cuts and provenance (the method's citation); the features are the bands' statistics cut and stacked; the cohort's spectral and connectome features are identified both
// ways (accuracy 1); `fingerprint_band` does not depend on the recording's chunking (the same samples stored in chunks of 700 and 1100); the refusals name their reason. Synthetic data only.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <random>

#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;
namespace face = nxr::pipelines::face;
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
std::string refusal(const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
g::RunOptions span(std::optional<double> a, std::optional<double> b) {
  g::RunOptions o;
  o.chunks = (a || b) ? std::nullopt : std::optional<std::string>("all");
  o.start = a; o.stop = b;
  return o;
}
MatrixXd statistic(SubjectWriter& sub, const std::string& name, std::vector<std::int64_t>* shape = nullptr, Json* producer = nullptr) {
  const StoredStatistic st = read_statistic(sub, name);
  if (shape) *shape = st.shape;
  if (producer) *producer = st.producer;
  const std::int64_t r = st.shape.at(0), c = static_cast<std::int64_t>(st.values.size()) / r;
  MatrixXd m(r, c);
  for (std::int64_t i = 0; i < r; ++i) for (std::int64_t j = 0; j < c; ++j) m(i, j) = st.values[static_cast<std::size_t>(i * c + j)];
  return m;
}
bool cites(const Json& producer) {
  const std::string all = producer.dump();
  return all.find("Castanheira") != std::string::npos && all.find("2018-10-12") != std::string::npos;
}
double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }

constexpr int kSubjects = 4, kM = 48, kRegions = 4;
constexpr std::int64_t kT = 12000;
constexpr double kFs = 100.0;
struct Band { std::string name; double lo, hi; };
const std::vector<Band> kBands = {{"alpha", 8, 13}, {"beta", 13, 30}};

// one subject of the cohort, made in its own rows; `chunks` the stored chunk width of its run (and a second copy in chunks of 1100 when `copy`)
void make_subject(DatasetWriter& ds, const std::string& name, int who, bool copy) {
  std::mt19937 rng(100 + who);
  std::normal_distribution<double> nd;
  std::vector<double> V;
  std::vector<std::int32_t> F;
  icosphere1(0.07, V, F);
  const std::int64_t nv = static_cast<std::int64_t>(V.size() / 3);
  std::vector<std::string> types;
  for (int i = 0; i < kM; ++i) types.push_back(i < kM / 2 ? "MEG MAG" : "MEG GRAD");
  MatrixXd gain(kM, 3 * nv);
  std::mt19937 grng(7);                                          // one head model for all (anatomy is not what is identified here)
  for (int i = 0; i < kM; ++i) for (int j = 0; j < 3 * nv; ++j) gain(i, j) = (i < kM / 2 ? 1e-13 : 5e-13) * nd(grng);
  gain = gain.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  std::vector<std::int64_t> region(static_cast<std::size_t>(nv));
  for (int k = 0; k < nv; ++k) region[static_cast<std::size_t>(k)] = (V[3 * k] > 0 ? 1 : 0) + (V[3 * k + 2] > 0 ? 2 : 0);
  // the planted identity: each region's 10 and 20 Hz amplitudes, and which of two slow modulators each region's rhythms follow (the envelope pattern)
  std::mt19937 id(1000 + who);
  std::uniform_real_distribution<double> ua(0.2, 2.0);
  const int patterns[kSubjects][kRegions] = {{0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 2, 3}};   // the three pairings of four regions, and none
  double a10[kRegions], a20[kRegions];
  for (int r = 0; r < kRegions; ++r) { a10[r] = ua(id); a20[r] = ua(id); }
  // each source's rhythms: three sinusoids of its own around the subject's alpha peak and around 20 Hz (random phases: the sources are not coherent, which a beamformer would cancel), their amplitude
  // the region's, modulated by one of four slow modulators (a pattern of the subject's own), plus the source's own noise
  MatrixXd S(nv, kT);
  std::uniform_real_distribution<double> uf(-1.0, 1.0), up(0.0, 2 * M_PI);
  for (int k = 0; k < nv; ++k) {
    const int r = static_cast<int>(region[static_cast<std::size_t>(k)]), p = patterns[who][r];
    double f10[3], f20[3], p10[3], p20[3];
    for (int c = 0; c < 3; ++c) { f10[c] = 9 + 0.7 * who + 1.3 * (c - 1) + 0.1 * uf(rng); f20[c] = 20 + 2.5 * (c - 1) + 0.2 * uf(rng); p10[c] = up(rng); p20[c] = up(rng); }   // their beats (above 1 Hz) faster than the modulators
    for (std::int64_t t = 0; t < kT; ++t) {
      const double s = t / kFs;
      const double m = 1 + 0.95 * std::sin(2 * M_PI * (0.13 + 0.08 * p) * s + p);   // the region's slow modulator (0.13, 0.21, 0.29 or 0.37 Hz)
      double a = 0, b = 0;
      for (int c = 0; c < 3; ++c) { a += std::sin(2 * M_PI * f10[c] * s + p10[c]); b += std::sin(2 * M_PI * f20[c] * s + p20[c]); }
      S(k, t) = 1e-9 * (0.3 * nd(rng) + m * (a10[r] * a + a20[r] * b));
    }
  }
  MatrixXd Gn(kM, nv);
  for (int k = 0; k < nv; ++k) {
    const Eigen::Vector3d n(V[3 * k], V[3 * k + 1], V[3 * k + 2]);
    Gn.col(k) = gain.middleCols(3 * k, 3) * n.normalized();
  }
  MatrixXd X = Gn * S, E(kM, kT);
  const double rms = std::sqrt(X.squaredNorm() / static_cast<double>(X.size()));          // the sensor noise and the empty room at a tenth of the brain signal's size
  for (int i = 0; i < kM; ++i) for (std::int64_t t = 0; t < kT; ++t) { X(i, t) += 0.1 * rms * nd(rng); E(i, t) = 0.1 * rms * nd(rng) * (1 + 0.05 * i); }

  SubjectWriter sub = SubjectWriter::open(ds, name);
  sub.compose([&] {
    add_surface(sub, {.name = "cortex"}, V, F);
    const std::string surface_id = text(*sub.by_name("manifold", "cortex"), "id");
    const std::string cid = uuid7();
    std::vector<float> pos(static_cast<std::size_t>(kM * 3));
    for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = kM;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec gs;
    gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors";
    m.geometry = gs;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {kM, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    std::vector<std::int64_t> tcode, qcode(static_cast<std::size_t>(kM), 0);
    for (const auto& t : types) tcode.push_back(t == "MEG MAG" ? 0 : 1);
    nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", tcode, {"MEG MAG", "MEG GRAD"}, {}}, {"quality", qcode, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                        [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
    std::vector<float> gf(static_cast<std::size_t>(kM * 3 * nv));
    for (int i = 0; i < kM; ++i) for (int j = 0; j < 3 * nv; ++j) gf[static_cast<std::size_t>(i * 3 * nv + j)] = static_cast<float>(gain(i, j));
    OperatorSpec op;
    op.name = "headmodel"; op.path = "headmodel"; op.kind = "forward"; op.from_manifold_id = surface_id; op.layout = "dense";
    op.n_rows = kM; op.n_cols = 3 * nv; op.data = view(gf, {kM, 3 * nv}); op.array.compress = true;
    op.cols.set("to_manifold_id", cid).set("from_value_type", "vector3").set("to_value_type", "scalar").set("description", "the leadfield");
    sub.operator_(std::move(op));
    nxr::bst::LabellingOptions lo;
    lo.manifold_id = surface_id; lo.dictionary = "regions"; lo.scope = "subject";
    nxr::bst::labelling(sub, "selection/regions", region, {"back-bottom", "front-bottom", "back-top", "front-top"}, lo);
    auto add_recording = [&](const std::string& rname, const MatrixXd& Y, std::int64_t chunk) {
      ManifoldSpec tm;
      tm.name = rname + "_time"; tm.path = "time/" + rname + "_time"; tm.type = "line"; tm.n_vertices = kT;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({kT});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg;
      tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec d;
      d.name = "time"; d.n_vertices = kT; d.spacing = 1.0 / kFs; d.origin = 0.0; d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      std::vector<float> xf(static_cast<std::size_t>(kM * kT));
      for (int i = 0; i < kM; ++i) for (std::int64_t t = 0; t < kT; ++t) xf[static_cast<std::size_t>(i * kT + t)] = static_cast<float>(Y(i, t));
      FieldSpec rec;
      rec.name = rname; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, rname); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {kM, kT}); rec.array.chunks = Shape{kM, chunk}; rec.array.compress = false;
      sub.field(std::move(rec));
    };
    add_recording("rest", X, 700);
    add_recording("emptyroom", E, 700);
    if (copy) add_recording("rest_1100", X, 1100);
  });
}

// THE RUN ORDER of fingerprint_features.yaml, one subject (two bands, two halves)
void run_subject(DatasetWriter& ds, const std::string& name) {
  SubjectWriter sub = SubjectWriter::open(ds, name);
  const double T = static_cast<double>(kT - 1) / kFs;                                   // the run's last time
  const std::pair<double, double> halves[2] = {{5.0, std::floor(T / 2)}, {std::ceil(T / 2), T - 5.0}};
  const char* stems[2] = {"fp_A", "fp_B"};
  for (const Band& b : kBands) {
    for (const char* r : {"rest", "emptyroom"})
      g::run(ds, sub, g::load("bandpass"), Json{{"recording", r}, {"lower", b.lo}, {"upper", b.hi}, {"design", "brainstorm_2018"}, {"remove_mean", false}, {"name", std::string(r) + "_" + b.name}}, span(std::nullopt, std::nullopt));
    g::run(ds, sub, g::load("noisecov"), Json{{"recording", "rest_" + b.name}, {"divisor", "samples_minus_one"}, {"name", "rest_" + b.name + "_data_cov"}}, span(std::nullopt, std::nullopt));
    g::run(ds, sub, g::load("noisecov"), Json{{"recording", "emptyroom_" + b.name}, {"divisor", "samples_minus_one"}, {"name", "emptyroom_" + b.name + "_noise_cov"}}, span(std::nullopt, std::nullopt));
    g::run(ds, sub, g::load("lcmv"), Json{{"data_cov", "rest_" + b.name + "_data_cov"}, {"noise_cov", "emptyroom_" + b.name + "_noise_cov"}, {"name", "lcmv_" + b.name}}, span(std::nullopt, std::nullopt));
    for (int h = 0; h < 2; ++h) {
      const auto [s0, s1] = halves[h];
      const std::string map = std::string(stems[h]) + "_" + b.name, rec = "rest_" + b.name;
      g::run(ds, sub, g::load("noisecov"), Json{{"recording", rec}, {"name", rec + "_" + stems[h] + "_cov"}}, span(s0, s1));
      g::run(ds, sub, g::load("region_pca"), Json{{"recording", rec}, {"kernel", "lcmv_" + b.name}, {"covariance", rec + "_" + stems[h] + "_cov"}, {"partition", "regions"}, {"name", map}}, span(s0, s1));
      g::run(ds, sub, g::load("fingerprint_band"), Json{{"recording", rec}, {"map", map}, {"lower", b.lo}, {"upper", b.hi}}, span(s0, s1));
    }
  }
  for (int h = 0; h < 2; ++h)
    g::run(ds, sub, g::load("fingerprint_features"), Json{{"recording", "rest"}, {"stem", stems[h]}, {"band_names", Json::array({"alpha", "beta"})}, {"bands", Json::array({"8-12.5 Hz", "13-29.5 Hz"})}},
           span(std::nullopt, std::nullopt));
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_fpfeatures_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = root / "cohort";
  task::CreateDatasetRequest cd;
  cd.datastore = root.string(); cd.name = "cohort";
  task::create_dataset(cd);
  task::create_subject({folder.string(), "group", "template", "test"});
  std::vector<std::string> names;
  for (int i = 0; i < kSubjects; ++i) {
    names.push_back("S" + std::to_string(i + 1));
    task::create_subject({folder.string(), names.back(), "subject", "test"});
  }
  try {
    {
    DatasetWriter ds = open_writer(folder);
    for (int i = 0; i < kSubjects; ++i) make_subject(ds, names[static_cast<std::size_t>(i)], i, i == 0);
    for (const auto& s : names) run_subject(ds, s);

    // ── one subject's features: shapes, cuts, the bands' statistics, provenance ──
    SubjectWriter s1 = SubjectWriter::open(ds, names[0]);
    std::vector<std::int64_t> shape;
    Json prod;
    const MatrixXd spectral = statistic(s1, "fp_A_spectral", &shape, &prod);
    CHECK((shape == std::vector<std::int64_t>{kRegions, 44}));                              // alpha 8-12.5 Hz: 10 bins, beta 13-29.5 Hz: 34 bins (0.5 Hz)
    CHECK(cites(prod));
    const MatrixXd alpha = statistic(s1, "fp_A_alpha_spectrum", &shape);
    CHECK((shape == std::vector<std::int64_t>{kRegions, 101}));                             // 2 s at 100 Hz: 0 to 50 Hz in 0.5 Hz bins
    const MatrixXd beta = statistic(s1, "fp_A_beta_spectrum");
    CHECK(rel(spectral.leftCols(10), alpha.middleCols(16, 10)) < 1e-6 && rel(spectral.rightCols(34), beta.middleCols(26, 34)) < 1e-6);
    // the planted rhythm: every region's alpha spectrum peaks at one of S1's alpha carriers (9 and 10.3 Hz: bins 17 to 21)
    bool peaks = true;
    for (int r = 0; r < kRegions; ++r) { Eigen::Index k; alpha.row(r).maxCoeff(&k); std::printf("  S1 region %d: the alpha spectrum peaks at %.1f Hz\n", r, 0.5 * static_cast<double>(k)); peaks = peaks && k >= 17 && k <= 21; }
    CHECK(peaks);
    const MatrixXd conn = statistic(s1, "fp_A_connectome", &shape, &prod);
    CHECK((shape == std::vector<std::int64_t>{kRegions, kRegions, 2}));
    CHECK(cites(prod));
    const MatrixXd aec_a = statistic(s1, "fp_A_alpha_aec");
    bool stacked = true;
    for (int i = 0; i < kRegions; ++i) for (int j = 0; j < kRegions; ++j) stacked = stacked && std::abs(conn(i, j * 2) - aec_a(i, j)) < 1e-12 && std::abs(aec_a(i, i) - 1) < 1e-12;
    CHECK(stacked);
    std::printf("  S1 alpha AEC, half A:\n");
    for (int i = 0; i < kRegions; ++i) std::printf("    %+.3f %+.3f %+.3f %+.3f\n", aec_a(i, 0), aec_a(i, 1), aec_a(i, 2), aec_a(i, 3));

    // ── chunk-size independence of the spectra and the AEC: the same run stored in chunks of 700 and of 1100 ──
    {
      const double s0 = 5.0, e0 = 59.0;
      g::run(ds, s1, g::load("fingerprint_band"), Json{{"recording", "rest"}, {"map", "fp_A_alpha"}, {"lower", 8.0}, {"upper", 13.0}, {"name", "chunks_700"}}, span(s0, e0));
      g::run(ds, s1, g::load("fingerprint_band"), Json{{"recording", "rest_1100"}, {"map", "fp_A_alpha"}, {"lower", 8.0}, {"upper", 13.0}, {"name", "chunks_1100"}}, span(s0, e0));
      const double es = rel(statistic(s1, "chunks_1100_spectrum"), statistic(s1, "chunks_700_spectrum"));
      const double ea = (statistic(s1, "chunks_1100_aec") - statistic(s1, "chunks_700_aec")).cwiseAbs().maxCoeff();
      std::printf("  chunks of 700 against 1100: spectrum %.2e, AEC %.2e\n", es, ea);
      CHECK(es < 1e-6 && ea < 1e-6);
      // the paper's reading: one band-pass only (named option), a new result
      g::run(ds, s1, g::load("fingerprint_band"), Json{{"recording", "rest_alpha"}, {"map", "fp_A_alpha"}, {"lower", 8.0}, {"upper", 13.0}, {"single_bandpass", true}, {"name", "single"}}, span(s0, e0));
      const double d = (statistic(s1, "single_aec") - aec_a).cwiseAbs().maxCoeff();
      std::printf("  --single-bandpass against the release's double band-pass: AEC max |difference| %.2e\n", d);
      CHECK(d > 0 && d < 0.2);
    }
    // refused: a map that is not a region map's name; a band set of another length than its names
    CHECK(!refusal([&] { g::run(ds, s1, g::load("fingerprint_band"), Json{{"recording", "rest_alpha"}, {"map", "nope"}, {"lower", 8.0}, {"upper", 13.0}}, span(5.0, 59.0)); }).empty());
    CHECK(!refusal([&] { g::run(ds, s1, g::load("fingerprint_features"), Json{{"recording", "rest"}, {"stem", "fp_A"}, {"band_names", Json::array({"alpha", "beta"})}, {"bands", Json::array({"8-12.5 Hz"})}}, span(std::nullopt, std::nullopt)); }).empty());

    }   // the writer is closed: the group run opens the dataset itself
    // ── the cohort: the release's validation half (B) identified against its training half (A) ──
    for (const auto& [feature, vector] : std::vector<std::pair<std::string, std::string>>{{"spectral", "flatten"}, {"connectome", "upper-triangle"}}) {
      face::GroupRunRequest rq;
      rq.pipeline = "fingerprint";
      rq.subjects = names;
      rq.parameters = Json{{"target", "fp_B_" + feature}, {"database", "fp_A_" + feature}, {"vector", vector}, {"name", "fp_" + feature}};
      face::run_group(folder, rq);
      DatasetWriter ds = open_writer(folder);
      SubjectWriter gs = SubjectWriter::open(ds, "group");
      const MatrixXd acc = statistic(gs, "fp_" + feature + "_accuracy");
      const double* q = acc.data();                                                         // [3]: target from database, database from target, chance
      CHECK(acc.size() == 3);
      if (acc.size() != 3) continue;
      std::printf("  %s: accuracy %.2f (target from database), %.2f (database from target), chance %.2f\n", feature.c_str(), q[0], q[1], q[2]);
      CHECK(q[0] == 1.0 && q[1] == 1.0);
    }
  } catch (const std::exception& e) {
    std::printf("  refused: %s\n", e.what());
    CHECK(false);
  }
  return nxrtest::finish("test_fingerprint_features_pipeline");
}

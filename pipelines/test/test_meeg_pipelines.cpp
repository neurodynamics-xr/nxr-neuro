// noisecov_compute and inverse_compute on a small synthetic dataset built in a temp directory (no real data): a cortex icosphere, 12 channels (MEG MAG, MEG GRAD, EEG)
// with one bad channel, a head model (random gain), a recording of known second moments read in odd-sized chunks. Checks: the covariance equals the direct one,
// blocks and the divisor, the window, the channels left out, a cache hit and --replace; the kernel equals the library's on the same inputs (EEG average-referenced),
// is stored as the importer stores one (kind, shape, selection, params, noise covariance and whitener beside it, the pair), and a second call hits.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include "check.h"
#include "nxr/io/task/task.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
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

std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
template <class T>
std::vector<T> read_typed(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_meeg_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;

  std::vector<double> V;
  std::vector<std::int32_t> F;
  icosphere1(0.07, V, F);                                       // 42 sources
  const std::int64_t nv = static_cast<std::int64_t>(V.size() / 3);
  const std::vector<std::string> types = {"MEG MAG", "MEG MAG", "MEG MAG", "MEG MAG", "MEG GRAD", "MEG GRAD", "MEG GRAD", "MEG GRAD", "EEG", "EEG", "EEG", "EEG"};
  const std::int64_t M = 12, T = 3000, bad = 2;
  const double sfreq = 100.0;
  MatrixXd gain(M, 3 * nv), X(M, T);
  for (int i = 0; i < M; ++i) {
    const double s = types[static_cast<std::size_t>(i)] == "MEG MAG" ? 1e-13 : types[static_cast<std::size_t>(i)] == "MEG GRAD" ? 1e-12 : 1e-6;
    for (int j = 0; j < 3 * nv; ++j) gain(i, j) = s * nd(rng);
    for (int t = 0; t < T; ++t) X(i, t) = s * (nd(rng) * (1 + 0.1 * i) + 0.5 * nd(rng) * (i % 3 == 0)) + 50 * s * (i + 1);   // a DC offset per channel
  }
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });   // the stored values
  const MatrixXd gain32 = gain.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });

  DatasetWriter ds = create_dataset(root, "demo");
  std::string fwd_id;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      add_surface(sub, {.name = "cortex"}, V, F);
      const std::string surface_id = text(*sub.by_name("manifold", "cortex"), "id");
      // the channels: a points manifold with its type and quality labellings
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
      std::vector<std::int64_t> tcode, qcode;
      std::vector<std::string> tl = {"MEG MAG", "MEG GRAD", "EEG"};
      for (const auto& t : types) tcode.push_back(t == "MEG MAG" ? 0 : t == "MEG GRAD" ? 1 : 2);
      for (int i = 0; i < M; ++i) qcode.push_back(i == bad ? 1 : 0);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", tcode, tl, {}}, {"quality", qcode, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      // the head model: the gain, channels <- cortex (vector3), and its grid
      std::vector<float> gf(static_cast<std::size_t>(M * 3 * nv));
      for (int i = 0; i < M; ++i) for (int j = 0; j < 3 * nv; ++j) gf[static_cast<std::size_t>(i * 3 * nv + j)] = static_cast<float>(gain(i, j));
      OperatorSpec op;
      op.name = "headmodel";
      op.path = "headmodel";
      op.kind = "forward";
      op.from_manifold_id = surface_id;
      op.layout = "dense";
      op.n_rows = M;
      op.n_cols = 3 * nv;
      op.data = view(gf, {M, 3 * nv});
      op.array.compress = true;
      op.cols.set("to_manifold_id", cid).set("from_value_type", "vector3").set("to_value_type", "scalar").set("description", "the leadfield");
      fwd_id = sub.operator_(std::move(op));
      std::vector<float> loc(V.begin(), V.end());
      FieldSpec fs;
      fs.name = "headmodel_GridLoc";
      fs.path = fs.name;
      fs.kind = "GridLoc";
      fs.manifold_id = surface_id;
      fs.value_type = "vector3";
      fs.data = view(loc, {nv, 3});
      fs.cols.set("of_operator_id", fwd_id).set("unit", "m");
      sub.field(std::move(fs));
      // the recording on channels x time, stored in chunks of 700 samples
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
      FieldSpec rec;
      rec.name = "rest";
      rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest");
      rec.kind = "recording";
      rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 700};
      rec.array.compress = false;
      sub.field(std::move(rec));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");

  // ── the noise covariance ──
  std::vector<int> good;
  for (int i = 0; i < M; ++i) if (i != bad) good.push_back(i);
  auto direct = [&](std::int64_t s0, std::int64_t s1, std::int64_t block) {
    const int m = static_cast<int>(good.size());
    MatrixXd S = MatrixXd::Zero(m, m);
    int nb = 0;
    for (std::int64_t b = s0; b < s1; b += block) {
      const std::int64_t e = std::min(b + block, s1);
      MatrixXd Y(m, e - b);
      for (int i = 0; i < m; ++i) Y.row(i) = X.row(good[static_cast<std::size_t>(i)]).segment(b, e - b);
      Y.colwise() -= Y.rowwise().mean();
      S += Y * Y.transpose();
      ++nb;
    }
    return std::make_pair(S, nb);
  };
  NoiseCovOptions no;
  no.recording = "rest";
  no.start = 2.0;
  no.stop = 25.0;
  const NoiseCovResult a = noisecov_compute(ds, sub, no);
  CHECK(!a.hit);
  CHECK_EQ(a.channels.size(), static_cast<std::size_t>(M - 1));            // the bad channel is out
  CHECK_EQ(a.info["samples"].get<std::int64_t>(), 2300);
  CHECK_EQ(a.info["blocks"].get<int>(), 1);
  {
    const auto [S, nb] = direct(200, 2500, 2300);
    CHECK_EQ(nb, 1);
    CHECK(rel(a.covariance, S / 2299.0) < 1e-10);
  }
  const NoiseCovResult b = noisecov_compute(ds, sub, no);
  CHECK(b.hit);
  CHECK(rel(b.covariance, a.covariance) < 1e-15);
  CHECK_EQ(b.info["samples"].get<std::int64_t>(), 2300);
  // stored as a STATISTIC: measure covariance, float64 [M x M] on the channels' Selection of indices, its blocks the members and its divisor the degrees of freedom
  {
    const auto row = sub.db().one("SELECT * FROM statistic WHERE subject_id = ? AND name = ?", {nxr::io::V(sub.id()), nxr::io::V("rest_noise_cov")});
    CHECK(row.has_value());
    CHECK_EQ(text(*row, "measure"), std::string("covariance"));
    CHECK_EQ(text(*row, "method"), std::string("sample"));
    CHECK(!is_null(*row, "of_field_id"));
    std::vector<std::int64_t> sh;
    const auto d = read_typed<double>(sub.folder() / text(*row, "path"), &sh);
    CHECK(sh.size() == 2 && sh[0] == M - 1 && sh[1] == M - 1);
    const Json pj = Json::parse(text(*row, "producer_json"));
    CHECK_EQ(pj["n_samples"].get<std::int64_t>(), 2300);
    CHECK_EQ(pj["window"]["start_sample"].get<std::int64_t>(), 200);
    const auto sel = sub.find("selection", "rest_noise_cov_channels");
    CHECK(sel.has_value());
    const Json axes = Json::parse(text(*row, "axes_json"));
    CHECK_EQ(axes[0]["selection_id"].get<std::string>(), text(*sel, "id"));
    CHECK(!is_null(*row, "dof"));
  }
  // blocks of 10 s: two blocks and a third of 3 s; each its own mean; N - B
  NoiseCovOptions nb = no;
  nb.name = "rest_blocks";
  nb.start.reset();
  nb.stop.reset();
  nb.chunks = "all";                                            // every chunk (stored 700 wide): the whole recording, read chunk by chunk
  nb.block_seconds = 10.0;
  const NoiseCovResult c = noisecov_compute(ds, sub, nb);
  {
    const auto [S, nbk] = direct(0, T, 1000);
    CHECK_EQ(c.info["blocks"].get<int>(), nbk);
    CHECK_EQ(c.info["samples"].get<std::int64_t>(), T);
    CHECK(rel(c.covariance, S / static_cast<double>(T - nbk)) < 1e-10);
  }
  // another divisor is another result under the same name: refused, then replaced with the old array moved aside
  nb.covariance.divisor = nxr::meeg::CovarianceDivisor::Samples;
  CHECK_EQ(code_of([&] { noisecov_compute(ds, sub, nb); }), std::string(code::already_exists));
  nb.replace = true;
  const NoiseCovResult c2 = noisecov_compute(ds, sub, nb);
  CHECK(!c2.hit && c2.info["replaced"].get<bool>());
  CHECK(fs::exists(ds.root() / ("_replaced-" + now_utc(false).substr(0, 10))));
  CHECK(rel(c2.covariance, direct(0, T, 1000).first / static_cast<double>(T)) < 1e-10);
  // a window with no samples, an unknown recording
  NoiseCovOptions empty = no;
  empty.name = "x";
  empty.start = 29.999;
  empty.stop = 30.0;
  CHECK_EQ(code_of([&] { noisecov_compute(ds, sub, empty); }), std::string(code::invalid_argument));
  empty.recording = "nothing";
  CHECK_EQ(code_of([&] { noisecov_compute(ds, sub, empty); }), std::string(code::no_such_row));

  // ── the kernel ──
  InverseComputeOptions io;
  io.noise_cov = "rest_noise_cov";
  io.name = "MN";
  const InverseComputeResult k1 = inverse_compute(ds, sub, io);
  CHECK(!k1.hit);
  CHECK_EQ(k1.info["channels"].get<int>(), static_cast<int>(M - 1));
  {
    // the library's kernel on the same inputs, EEG average-referenced
    const int m = static_cast<int>(good.size());
    MatrixXd G(m, 3 * nv);
    std::vector<std::string> ty;
    for (int i = 0; i < m; ++i) { G.row(i) = gain32.row(good[static_cast<std::size_t>(i)]); ty.push_back(types[static_cast<std::size_t>(good[static_cast<std::size_t>(i)])]); }
    MatrixXd A = MatrixXd::Identity(m, m);
    std::vector<int> eeg;
    for (int i = 0; i < m; ++i) if (ty[static_cast<std::size_t>(i)] == "EEG") eeg.push_back(i);
    for (int p : eeg) for (int q : eeg) A(p, q) -= 1.0 / static_cast<double>(eeg.size());
    const MatrixXd Cc = A * a.covariance * A.transpose();
    const nxr::meeg::Kernel ref = nxr::meeg::unconstrainedKernel(A * G, Cc, ty);
    std::vector<std::int64_t> sh;
    const auto kf = read_typed<float>((sub.folder() / ("MN")), &sh);
    CHECK(sh.size() == 2 && sh[0] == 3 * nv && sh[1] == m);
    MatrixXd K(sh[0], sh[1]);
    for (int i = 0; i < sh[0]; ++i) for (int j = 0; j < sh[1]; ++j) K(i, j) = kf[static_cast<std::size_t>(i * sh[1] + j)];
    CHECK(rel(K, ref.K) < 1e-5);                                // stored as float32
    CHECK_EQ(k1.whitener_rank, ref.whitenerRank);
    CHECK(std::abs(k1.lambda / ref.lambda - 1) < 1e-9);
  }
  {
    // stored as the importer stores a kernel
    const auto row = sub.find("operator", "MN");
    CHECK(row.has_value());
    CHECK_EQ(text(*row, "kind"), std::string("inverse kernel"));
    CHECK_EQ(text(*row, "to_value_type"), std::string("vector3"));
    CHECK_EQ(text(*row, "inverts_id"), fwd_id);
    CHECK_EQ(text(*row, "inverse_kind"), std::string("regularized"));
    const Json pj = Json::parse(text(*row, "params_json"));
    CHECK_EQ(pj["inverse"]["measure"].get<std::string>(), std::string("MNE"));
    CHECK_EQ(pj["inverse"]["orientation"].get<std::string>(), std::string("free"));
    CHECK(std::abs(pj["inverse"]["depth"]["exponent"].get<double>() - 0.5) < 1e-15);
    CHECK(std::abs(pj["inverse"]["snr"].get<double>() - 3) < 1e-15);
    CHECK(std::abs(pj["inverse"]["noise"]["reg"].get<double>() - 0.1) < 1e-15);
    const Json prod = Json::parse(text(*row, "producer_json"));
    CHECK(prod["producer"].get<std::string>().rfind("nxr-neuro", 0) == 0);
    CHECK_EQ(prod["units"].get<std::string>(), std::string("A.m"));
    for (const char* n : {"MN_channels", "MN_whitener", "MN_source_rr", "MN_source_nn"})
      CHECK(sub.find(std::string(n).find("_channels") != std::string::npos ? "selection" : std::string(n).find("source") != std::string::npos ? "field" : "operator", n).has_value());
    const auto kc = sub.db().one("SELECT * FROM statistic WHERE subject_id = ? AND name = ?", {nxr::io::V(sub.id()), nxr::io::V("MN_noise_cov")});
    CHECK(kc.has_value());
    CHECK_EQ(text(*kc, "measure"), std::string("covariance"));
    const auto wh = sub.find("operator", "MN_whitener");
    CHECK(wh.has_value());
    CHECK_EQ(text(*wh, "kind"), std::string("whitener"));
    CHECK_EQ(text(*wh, "of_statistic_id"), text(*kc, "id"));       // the whitener is an operator derived from the covariance statistic
  }
  CHECK(inverse_compute(ds, sub, io).hit);
  io.kernel.depth.enabled = false;
  CHECK_EQ(code_of([&] { inverse_compute(ds, sub, io); }), std::string(code::already_exists));
  // a kernel straight from a recording: the covariance is computed and stored on the way; MNE's conventions as one switch
  InverseComputeOptions ir;
  ir.from_recording = no;
  ir.from_recording->name = "rest_cov2";
  ir.from_recording->start = 0;
  ir.from_recording->stop = 20.0;
  ir.name = "MN_mne";
  ir.mne_conventions = true;
  const InverseComputeResult k2 = inverse_compute(ds, sub, ir);
  CHECK(!k2.hit);
  CHECK(sub.db().one("SELECT 1 FROM statistic WHERE subject_id = ? AND name = ?", {nxr::io::V(sub.id()), nxr::io::V("rest_cov2")}).has_value());
  CHECK(sub.find("operator", "MN_mne").has_value());
  CHECK_EQ(code_of([&] { InverseComputeOptions bad_opt; bad_opt.name = "z"; inverse_compute(ds, sub, bad_opt); }), std::string(code::invalid_argument));

  // the commands: the same jobs through the nxr-neuro command line (JSON lines out), then read back
  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    CHECK_EQ(std::system((cli + " run noisecov --dataset " + d + " --subject S1 --recording rest --name cli_cov --start 1 --stop 20 > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " run inverse --dataset " + d + " --subject S1 --noise-cov cli_cov --name MN_cli --snr 2 --all >> " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " run inverse --dataset " + d + " --subject S1 --noise-cov cli_cov --name MN_cli --snr 2 --all >> " + log).c_str()), 0);   // a hit
    CHECK(std::system((cli + " run inverse --dataset " + d + " --subject S1 > /dev/null 2>&1").c_str()) != 0);                                    // neither a covariance nor a recording
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(all.find("\"cache\": \"miss\"") != std::string::npos && all.find("\"cache\": \"hit\"") != std::string::npos);
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    CHECK(s2.find("operator", "cli_cov").has_value());
    const auto kr = s2.find("operator", "MN_cli");
    CHECK(kr.has_value());
    CHECK(std::abs(Json::parse(text(*kr, "params_json"))["inverse"]["snr"].get<double>() - 2) < 1e-15);
    d2.close();
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_meeg_pipelines");
}

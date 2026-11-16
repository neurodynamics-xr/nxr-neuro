// headmodel_compute on a small synthetic dataset built in a temp directory (no real data): a cortex, inner skull and scalp (icospheres about a centre), 10 channels in the
// layout the Brainstorm importer writes (3 magnetometers of four integration points, 2 planar gradiometers of two coils, 5 electrodes; per-channel orientations; the coils
// manifold, its orientations and the csr coil weights), and a recording. Checks: the gain equals the library's on the same inputs (spheres fitted to the surfaces), it is
// stored as an imported head model is (kind, shapes, value types, producer, params, GridLoc), a second call hits, a missing surface is refused naming the subject's surfaces,
// and the CHAIN headmodel compute -> noisecov compute -> inverse compute runs on the computed head model, through the library and through the nxr-neuro command.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;
using Eigen::Vector3d;

namespace {
void icosphere1(double radius, const Vector3d& centre, std::vector<double>& V, std::vector<std::int32_t>& F) {
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
    for (int c = 0; c < 3; ++c) V[i + static_cast<std::size_t>(c)] = centre(c) + V[i + static_cast<std::size_t>(c)] * radius / n * (1 + (i % 7 == 0 ? 0.03 : 0.0));   // a bump or two: not a perfect sphere
  }
}
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
std::string kind_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_headmodel_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  const Vector3d c(0.003, -0.002, 0.045);
  const std::int64_t M = 10, T = 2000;
  const double sfreq = 100.0;
  const std::vector<std::string> types = {"MEG MAG", "MEG MAG", "MEG MAG", "MEG GRAD", "MEG GRAD", "EEG", "EEG", "EEG", "EEG", "EEG"};

  // the sensor points, per channel, as the importer stores them
  std::vector<Vector3d> pos(M), ori(M);
  std::vector<std::vector<Vector3d>> cpts(M);
  std::vector<std::vector<double>> cwts(M);
  for (int i = 0; i < M; ++i) {
    const double ph = 0.5 + 0.35 * i, th = 1.7 * i;
    const double r = types[static_cast<std::size_t>(i)] == "EEG" ? 0.091 : 0.115;
    const Vector3d u(std::sin(ph) * std::cos(th), std::sin(ph) * std::sin(th), std::cos(ph));
    pos[static_cast<std::size_t>(i)] = c + r * u;
    ori[static_cast<std::size_t>(i)] = u;
    const Vector3d x = u.cross(Vector3d(0.3, 0.5, 0.8)).normalized(), y = u.cross(x);
    auto& P = cpts[static_cast<std::size_t>(i)];
    auto& W = cwts[static_cast<std::size_t>(i)];
    if (types[static_cast<std::size_t>(i)] == "MEG MAG") for (int k = 0; k < 4; ++k) { P.push_back(pos[static_cast<std::size_t>(i)] + 0.01 * ((k & 1 ? 1 : -1) * x + (k & 2 ? 1 : -1) * y)); W.push_back(0.25); }
    else if (types[static_cast<std::size_t>(i)] == "MEG GRAD") { P = {pos[static_cast<std::size_t>(i)] + 0.008 * x, pos[static_cast<std::size_t>(i)] - 0.008 * x}; W = {62.5, -62.5}; }
    else { P = {pos[static_cast<std::size_t>(i)]}; W = {1.0}; }
  }

  DatasetWriter ds = create_dataset(root, "demo");
  auto build = [&](const std::string& subject, bool with_skull) {
    SubjectWriter sub = SubjectWriter::create(ds, subject);
    sub.compose([&] {
      std::vector<double> V;
      std::vector<std::int32_t> F;
      icosphere1(0.06, c, V, F);
      add_surface(sub, {.name = "cortex"}, V, F);
      if (with_skull) { icosphere1(0.078, c, V, F); add_surface(sub, {.name = "innerskull", .primary = false}, V, F); }
      icosphere1(0.092, c, V, F);
      add_surface(sub, {.name = "scalp", .primary = false}, V, F);
      // the channels manifold, its labellings and the full geometry
      const std::string cid = uuid7();
      std::vector<float> p32;
      for (const auto& p : pos) for (int k = 0; k < 3; ++k) p32.push_back(static_cast<float>(p(k)));
      auto points = [&](const std::string& id, const std::string& name, const std::vector<float>& a, std::int64_t n) {
        ManifoldSpec m;
        m.id = id;
        m.name = name;
        m.path = "sensor/" + name;
        m.type = "points";
        m.n_vertices = n;
        m.topology = {"none", 0, 1, std::nullopt, {}};
        GeometrySpec g;
        g.form = "stored";
        g.cs_key = "scs";
        g.positions_path = "sensor/" + name;
        m.geometry = g;
        m.populate = [&sub, a, n, id, name](nxr::io::Store&, const std::string&) { sub.write_array("sensor/" + name, view(a, {n, 3}), {}, std::nullopt, "manifold", id); };
        sub.manifold(std::move(m));
      };
      points(cid, "sensors", p32, M);
      std::vector<std::int64_t> tcode, qcode(M, 0);
      for (const auto& t : types) tcode.push_back(t == "MEG MAG" ? 0 : t == "MEG GRAD" ? 1 : 2);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", tcode, {"MEG MAG", "MEG GRAD", "EEG"}, {}}, {"quality", qcode, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      auto vec3 = [&](const std::string& name, const std::string& manifold, const std::vector<float>& a, std::int64_t n) {
        FieldSpec f;
        f.name = name;
        f.path = "sensor/" + name;
        f.kind = "sensor orientation";
        f.manifold_id = manifold;
        f.value_type = "vector3";
        f.data = view(a, {n, 3});
        sub.field(std::move(f));
      };
      std::vector<float> o32;
      for (const auto& o : ori) for (int k = 0; k < 3; ++k) o32.push_back(static_cast<float>(o(k)));
      vec3("sensors_orientation", cid, o32, M);
      std::vector<float> cp, co, cw;
      std::vector<std::int32_t> indptr(1, 0), indices;
      for (int i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < cpts[static_cast<std::size_t>(i)].size(); ++k) {
          for (int d = 0; d < 3; ++d) { cp.push_back(static_cast<float>(cpts[static_cast<std::size_t>(i)][k](d))); co.push_back(static_cast<float>(ori[static_cast<std::size_t>(i)](d))); }
          cw.push_back(static_cast<float>(cwts[static_cast<std::size_t>(i)][k]));
          indices.push_back(static_cast<std::int32_t>(indices.size()));
        }
        indptr.push_back(static_cast<std::int32_t>(indices.size()));
      }
      const std::int64_t P = static_cast<std::int64_t>(cw.size());
      const std::string coil_id = uuid7();
      points(coil_id, "sensors_coils", cp, P);
      vec3("sensors_coils_orientation", coil_id, co, P);
      OperatorSpec op;
      op.name = "sensors_coil_weights";
      op.path = op.name;
      op.kind = "coil weights";
      op.from_manifold_id = coil_id;
      op.layout = "csr";
      op.n_rows = M;
      op.n_cols = P;
      op.cols.set("to_manifold_id", cid).set("nnz", P);
      op.sparse = SparseArrays{{{"indptr", view(indptr, {M + 1})}, {"indices", view(indices, {P})}, {"data", view(cw, {P})}}};
      sub.operator_(std::move(op));
      if (!with_skull) return;
      // the recording on channels x time
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
      for (int i = 0; i < M; ++i) {
        const double s = types[static_cast<std::size_t>(i)] == "MEG MAG" ? 1e-13 : types[static_cast<std::size_t>(i)] == "MEG GRAD" ? 1e-12 : 1e-6;
        for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(s * nd(rng) * (1 + 0.1 * i));
      }
      FieldSpec rec;
      rec.name = "rest";
      rec.path = node_path(sub.db(), prod, "rest");
      rec.kind = "recording";
      rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 500};
      rec.array.compress = false;
      sub.field(std::move(rec));
    });
  };
  build("S1", true);
  build("S2", false);          // no inner skull

  // ── the library's gain on the same inputs ──
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const SurfaceInput cortex = read_surface(sub, "cortex"), skull = read_surface(sub, "innerskull"), scalp = read_surface(sub, "scalp");
  auto mat = [](const SurfaceInput& s) { MatrixXd P(s.n_vertices(), 3); for (int i = 0; i < s.n_vertices(); ++i) for (int k = 0; k < 3; ++k) P(i, k) = s.vertices[static_cast<std::size_t>(3 * i + k)]; return P; };
  const MatrixXd src = mat(cortex);
  auto coils_for = [&](const std::vector<int>& chans, bool project, const nxr::meeg::Sphere& eeg_sphere) {
    nxr::meeg::Coils k;
    std::vector<Vector3d> p, o;
    std::vector<double> w;
    for (std::size_t q = 0; q < chans.size(); ++q)
      for (std::size_t j = 0; j < cpts[static_cast<std::size_t>(chans[q])].size(); ++j) {
        // as stored (float32)
        Vector3d pp = cpts[static_cast<std::size_t>(chans[q])][j].cast<float>().cast<double>();
        if (project) pp = eeg_sphere.centre + eeg_sphere.radius * (pp - eeg_sphere.centre).normalized();
        p.push_back(pp);
        o.push_back(Vector3d(ori[static_cast<std::size_t>(chans[q])].cast<float>().cast<double>()).normalized());   // as stored (float32), then a unit vector
        w.push_back(static_cast<double>(static_cast<float>(cwts[static_cast<std::size_t>(chans[q])][j])));
        k.channel.push_back(static_cast<int>(q));
      }
    k.channels = static_cast<int>(chans.size());
    k.position.resize(p.size(), 3); k.orientation.resize(p.size(), 3); k.weight.resize(p.size());
    for (std::size_t q = 0; q < p.size(); ++q) { k.position.row(q) = p[q].transpose(); k.orientation.row(q) = o[q].transpose(); k.weight(q) = w[q]; }
    return k;
  };
  const std::vector<int> megc = {0, 1, 2, 3, 4}, eegc = {5, 6, 7, 8, 9};
  const nxr::meeg::Sphere es = nxr::meeg::fitSphere(mat(scalp));
  const nxr::meeg::Coils mk = coils_for(megc, false, es), ek = coils_for(eegc, true, es);
  const auto os = nxr::meeg::sensorSpheres(mat(skull), mk.position, mk.weight, mk.channel, mk.channels);
  std::vector<Vector3d> centres;
  for (const auto& s : os) centres.push_back(s.centre);
  MatrixXd expect = MatrixXd::Zero(M, 3 * src.rows());
  expect.topRows(5) = nxr::meeg::sarvasGain(src, mk, centres);
  expect.bottomRows(5) = nxr::meeg::eegSphereGain(src, ek, es.centre);

  HeadModelOptions ho;
  ho.name = "hm";
  const HeadModelResult a = headmodel_compute(ds, sub, ho);
  CHECK(!a.hit);
  CHECK(rel(a.gain, expect) < 1e-12);
  CHECK(a.info["meg"].get<std::string>() == "overlapping spheres" && a.info["eeg"].get<std::string>() == "three shells");
  {
    // stored as the importer stores a head model
    const auto row = sub.find("operator", "hm");
    CHECK(row.has_value());
    CHECK_EQ(text(*row, "kind"), std::string("forward"));
    CHECK_EQ(text(*row, "from_value_type"), std::string("vector3"));
    CHECK_EQ(text(*row, "to_value_type"), std::string("scalar"));
    CHECK_EQ(text(*row, "from_manifold_id"), cortex.id);
    std::vector<std::int64_t> sh;
    const auto gf = read_typed<float>((sub.folder() / ("hm")), &sh);
    CHECK(sh.size() == 2 && sh[0] == M && sh[1] == 3 * src.rows());
    MatrixXd G(sh[0], sh[1]);
    for (int i = 0; i < sh[0]; ++i) for (int j = 0; j < sh[1]; ++j) G(i, j) = gf[static_cast<std::size_t>(i * sh[1] + j)];
    CHECK(rel(G, expect) < 1e-6);                                        // float32
    const Json pj = Json::parse(text(*row, "producer_json"));
    CHECK(pj["producer"].get<std::string>().rfind("nxr-neuro", 0) == 0);
    CHECK_EQ(pj["units"].get<std::string>(), std::string("m"));
    CHECK_EQ(pj["n_sources"].get<std::int64_t>(), src.rows());
    CHECK_EQ(pj["meg_method"].get<std::string>(), std::string("os_meg"));
    CHECK_EQ(pj["eeg_method"].get<std::string>(), std::string("eeg_3sphereberg"));
    const Json pr = Json::parse(text(*row, "params_json"))["forward"];
    CHECK_EQ(pr["options"]["meg"].get<std::string>(), std::string("overlapping spheres"));
    CHECK(pr["computed"]["meg"]["spheres"].size() == 5);
    CHECK(std::abs(pr["computed"]["eeg"]["sphere"]["radius"].get<double>() - es.radius) < 1e-12);
    CHECK(pr["computed"]["eeg"]["berg_scherg"]["magnitude"].size() == 3);
    CHECK(std::abs(pr["options"]["sensor_sphere"]["weight_exponent"].get<double>() - 4) < 1e-15);
    const auto gl = sub.find("field", "hm_GridLoc");
    CHECK(gl.has_value());
    CHECK_EQ(text(*gl, "of_operator_id"), text(*row, "id"));
  }
  CHECK(headmodel_compute(ds, sub, ho).hit);
  ho.eeg = false;                                                         // another model under the same name: refused, then replaced
  CHECK_EQ(kind_of([&] { headmodel_compute(ds, sub, ho); }), std::string(code::already_exists));
  ho.replace = true;
  ho.meg = MegModel::SingleSphere;
  const HeadModelResult b = headmodel_compute(ds, sub, ho);
  CHECK(!b.hit && b.info["replaced"].get<bool>());
  {
    const nxr::meeg::Sphere s = nxr::meeg::fitSphere(mat(skull));
    CHECK(rel(b.gain.topRows(5), nxr::meeg::sarvasGain(src, mk, s.centre)) < 1e-12);
    CHECK(b.gain.bottomRows(5).cwiseAbs().maxCoeff() == 0);                // no EEG asked for: zero rows
  }
  // refusals
  {
    SubjectWriter s2 = SubjectWriter::open(ds, "S2");
    HeadModelOptions h2;
    const std::string m = code_of([&] { headmodel_compute(ds, s2, h2); });
    CHECK(m.find("inner skull") != std::string::npos && m.find("'cortex'") != std::string::npos && m.find("'scalp'") != std::string::npos && m.find("--meg-surface") != std::string::npos);
    h2.meg_surface = "scalp";                                              // naming another surface is allowed
    h2.name = "hm2";
    CHECK(!headmodel_compute(ds, s2, h2).hit);
    h2.meg_surface = "nothing";
    h2.name = "hm3";
    CHECK_EQ(kind_of([&] { headmodel_compute(ds, s2, h2); }), std::string(code::no_such_row));
    HeadModelOptions none;
    none.meg = MegModel::None;
    none.eeg = false;
    CHECK_EQ(kind_of([&] { headmodel_compute(ds, s2, none); }), std::string(code::invalid_argument));
  }

  // ── the chain: head model -> noise covariance -> kernel, on the computed head model ──
  {
    HeadModelOptions h;
    h.name = "headmodel";
    const HeadModelResult g = headmodel_compute(ds, sub, h);
    NoiseCovOptions no;
    no.recording = "rest";
    no.chunks = "0";                     // the first chunk's samples, stated
    const NoiseCovResult nc = noisecov_compute(ds, sub, no);
    CHECK_EQ(nc.channels.size(), static_cast<std::size_t>(M));
    InverseComputeOptions io;
    io.forward = "headmodel";
    io.noise_cov = "rest_noise_cov";
    io.name = "MN";
    const InverseComputeResult k = inverse_compute(ds, sub, io);
    CHECK(!k.hit && k.whitener_rank > 0);
    std::vector<std::int64_t> sh;
    const auto kf = read_typed<float>((sub.folder() / ("MN")), &sh);
    CHECK(sh.size() == 2 && sh[0] == 3 * src.rows() && sh[1] == M);
    bool finite = true;
    for (float v : kf) finite = finite && std::isfinite(v);
    CHECK(finite);
    CHECK_EQ(text(*sub.find("operator", "MN"), "inverts_id"), text(*sub.find("operator", "headmodel"), "id"));
    CHECK(rel(g.gain, g.gain) == 0);
  }
  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    CHECK_EQ(std::system((cli + " headmodel compute --dataset " + d + " --subject S1 --name hm_cli --meg single --sensor-weight-exponent 3 --eeg-series --eeg-series-terms 100 > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " headmodel compute --dataset " + d + " --subject S1 --name hm_cli --meg single --sensor-weight-exponent 3 --eeg-series --eeg-series-terms 100 >> " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " inverse compute --dataset " + d + " --subject S1 --forward hm_cli --noise-cov rest_noise_cov --name MN_cli >> " + log).c_str()), 0);
    CHECK(std::system((cli + " headmodel compute --dataset " + d + " --subject S2 --name x --channels nothing > /dev/null 2>&1").c_str()) != 0);
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(all.find("\"cache\": \"miss\"") != std::string::npos && all.find("\"cache\": \"hit\"") != std::string::npos);
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    const auto row = s2.find("operator", "hm_cli");
    CHECK(row.has_value());
    CHECK_EQ(Json::parse(text(*row, "params_json"))["forward"]["options"]["eeg_sphere"]["approximation"].get<std::string>(), std::string("series"));
    CHECK(s2.find("operator", "MN_cli").has_value());
    d2.close();
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_headmodel");
}

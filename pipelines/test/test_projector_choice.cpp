// THE CHOICE OF PROJECTORS (`--projectors active | none | NAME[,NAME...]`) and a projector's status (`edit operator --status`), on a synthetic recording of 10 magnetometers with three
// active SSP projectors, read across two chunks. Checks:
//  - read field: a named pair is P_AB x, active is P_ABC x (unchanged), an unknown name is refused naming the valid ones; the result does not depend on the chunk grid (two chunks);
//  - noisecov: a named pair is P_AB C P_AB^T of the covariance read as recorded, differs from all three, is recorded (option, applied) and is another cached result;
//  - inverse (Brainstorm and MNE conventions): a named pair changes the covariance the kernel is computed with (P_AB C P_AB^T) and, for MNE, the kernel's end projector (K u = 0 for
//    the named ones only); the provenance names the pair;
//  - edit operator: a projector set to 0 is no longer applied by active (read, noisecov, the inverse's default), the cache keys change; set back to 1 every result is the one before,
//    byte for byte; the operator's provenance holds both edits; another kind of operator and a missing name are refused.
// Synthetic data only.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>

#include <Eigen/SVD>

#include "check.h"
#include "meeg_runs.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/projectors.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
using namespace nxrtest;
namespace fs = std::filesystem;
namespace face = nxr::pipelines::face;
using Eigen::MatrixXd;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
// the SSP operator of the vectors given: I - U U^T, U an orthonormal basis of their span
MatrixXd ssp(const std::vector<Eigen::VectorXd>& us) {
  const Eigen::Index m = us.front().size();
  MatrixXd S(m, static_cast<Eigen::Index>(us.size()));
  for (std::size_t j = 0; j < us.size(); ++j) S.col(static_cast<Eigen::Index>(j)) = us[j];
  Eigen::JacobiSVD<MatrixXd> svd(S, Eigen::ComputeThinU);
  const MatrixXd U = svd.matrixU();
  return MatrixXd::Identity(m, m) - U * U.transpose();
}
MatrixXd floats(const task::CommandArray& a) {
  const Eigen::Index r = a.shape.at(0), c = a.shape.at(1);
  MatrixXd m(r, c);
  for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) { float v; std::memcpy(&v, a.bytes.data() + 4 * (i * c + j), 4); m(i, j) = v; }
  return m;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_projector_choice_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  const std::int64_t M = 10, T = 40000, nv = 12;                          // two chunks of 2^15 samples
  const double sfreq = 1000.0, s = 1e-13;
  std::vector<double> V;
  std::vector<std::int32_t> F;
  {                                                                        // an icosahedron: 12 sources
    const double t = (1 + std::sqrt(5.0)) / 2;
    const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    const int tri[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    for (auto& b : base) for (int c = 0; c < 3; ++c) V.push_back(0.07 * b[c] / std::sqrt(1 + t * t));
    for (auto& f : tri) F.insert(F.end(), f, f + 3);
  }
  MatrixXd gain(M, 3 * nv), X(M, T);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < 3 * nv; ++j) gain(i, j) = s * nd(rng);
    for (int t = 0; t < T; ++t) X(i, t) = s * nd(rng) * (1 + 0.1 * i);
  }
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  std::vector<Eigen::VectorXd> u(3, Eigen::VectorXd(M));
  for (auto& v : u) { for (int i = 0; i < M; ++i) v(i) = nd(rng); v /= v.norm(); }
  std::vector<std::string> names;

  {
    DatasetWriter ds = create_dataset(root, "demo");
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
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", std::vector<std::int64_t>(M, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(M, 0), {"good", "bad"}, {}}}, cid,
                          "the montage families", std::nullopt, [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      std::vector<float> gf(static_cast<std::size_t>(M * 3 * nv));
      for (int i = 0; i < M; ++i) for (int j = 0; j < 3 * nv; ++j) gf[static_cast<std::size_t>(i * 3 * nv + j)] = static_cast<float>(gain(i, j));
      OperatorSpec op;
      op.name = "headmodel"; op.path = "headmodel"; op.kind = "forward"; op.from_manifold_id = surface_id; op.layout = "dense"; op.n_rows = M; op.n_cols = 3 * nv;
      op.data = view(gf, {M, 3 * nv});
      op.cols.set("to_manifold_id", cid).set("from_value_type", "vector3").set("to_value_type", "scalar").set("description", "the leadfield");
      const std::string fwd_id = sub.operator_(std::move(op));
      std::vector<float> loc(V.begin(), V.end());
      FieldSpec gl;
      gl.name = "headmodel_GridLoc"; gl.path = gl.name; gl.kind = "GridLoc"; gl.manifold_id = surface_id; gl.value_type = "vector3"; gl.data = view(loc, {nv, 3});
      gl.cols.set("of_operator_id", fwd_id).set("unit", "m");
      sub.field(std::move(gl));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
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
      for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
      FieldSpec rec;
      rec.name = "rest"; rec.path = task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 4096};
      sub.field(std::move(rec));
      std::vector<nxr::bst::ProjectorEntry> es;
      for (int k = 0; k < 3; ++k) {
        nxr::bst::ProjectorEntry e;
        e.order = k + 1; e.comment = "ssp " + std::to_string(k + 1); e.method = "SSP"; e.kind = "ssp"; e.status = 1; e.m = static_cast<std::size_t>(M); e.k = 1;
        e.components.assign(u[static_cast<std::size_t>(k)].data(), u[static_cast<std::size_t>(k)].data() + M); e.mask = {true};
        es.push_back(e);
      }
      names = nxr::bst::export_projectors(sub, es, "sensors", cid, "");
    });
    ds.close();
  }
  CHECK_EQ(names.size(), std::size_t(3));
  const std::string D = (root / "demo").string(), A = names[0], B = names[1], C = names[2];
  const MatrixXd Pab = ssp({u[0], u[1]}), Pabc = ssp({u[0], u[1], u[2]});
  // the in-process runs (the definitions through the runner), each on a writer opened for it and closed after
  auto with = [&](const std::function<void(DatasetWriter&, SubjectWriter&)>& f) {
    DatasetWriter ds = open_writer(root / "demo");
    {
      SubjectWriter sub = SubjectWriter::open(ds, "S1");
      f(ds, sub);
    }
    ds.close();
  };

  // ── read field ──
  auto read = [&](const std::string& projectors) {
    face::ReadNodeRequest r;
    r.dataset = D; r.subject = "S1"; r.name = "rest"; r.all = true; r.projectors = projectors;
    return face::read_field(r);
  };
  const task::CommandResult r_ab = read(A + "," + B), r_all = read("active"), r_none = read("none");
  CHECK(rel(floats(r_ab.arrays.at(0)), Pab * X) < 1e-6);                  // exactly A and B, across the two chunks
  CHECK(rel(floats(r_all.arrays.at(0)), Pabc * X) < 1e-6);                // active: the three (unchanged)
  CHECK(rel(floats(r_none.arrays.at(0)), X) == 0.0);
  CHECK(r_ab.result["projection"]["applied"] == Json::array({A, B}));
  CHECK(r_ab.result["projection"]["chosen"] == Json::array({A, B}));
  CHECK(r_all.result["projection"]["applied"] == Json::array({A, B, C}));
  {   // --projection-only: the matrix itself, over every channel, no sample read; it is what a read applies
    auto projection = [&](const std::string& projectors) {
      face::ReadNodeRequest r;
      r.dataset = D; r.subject = "S1"; r.name = "rest"; r.projectors = projectors; r.projection_only = true;
      return face::read_field(r);
    };
    const task::CommandResult p_all = projection("active"), p_ab = projection(A + "," + B), p_none = projection("none");
    auto doubles = [](const task::CommandArray& a) {
      MatrixXd m(a.shape.at(0), a.shape.at(1));
      for (Eigen::Index i = 0; i < m.rows(); ++i) for (Eigen::Index j = 0; j < m.cols(); ++j) std::memcpy(&m(i, j), a.bytes.data() + 8 * (i * m.cols() + j), 8);
      return m;
    };
    CHECK(p_all.arrays.size() == 1 && p_all.arrays.at(0).name == "projection");
    CHECK(rel(doubles(p_all.arrays.at(0)), Pabc) < 1e-12);
    CHECK(rel(doubles(p_ab.arrays.at(0)), Pab) < 1e-12);
    CHECK(rel(doubles(p_all.arrays.at(0)) * X, floats(r_all.arrays.at(0))) < 1e-6);   // composing it with the raw samples gives the active read
    CHECK(p_none.arrays.empty());
    CHECK(p_all.result["projection"]["applied"] == Json::array({A, B, C}));
  }
  {
    const std::string why = thrown([&] { read(A + ",nothing"); });
    CHECK(has(why, "'nothing' is not one of the channels' projectors") && has(why, C));   // refused, naming the valid ones
    CHECK(has(thrown([&] { read(","); }), "invalid_argument"));
  }

  // ── noisecov ──
  NoiseCovOptions no;
  no.recording = "rest"; no.chunks = "all";
  NoiseCovResult c_all, c_ab, c_none;
  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    NoiseCovOptions o = no;
    o.name = "cov_all"; c_all = noisecov_compute(ds, sub, o);
    o.name = "cov_none"; o.projectors = Projectors::none; c_none = noisecov_compute(ds, sub, o);
    o.name = "cov_ab"; o.projectors = Projectors::named({A, B}); c_ab = noisecov_compute(ds, sub, o);
    CHECK(noisecov_compute(ds, sub, o).hit);                              // the same choice again: a hit
    o.name = "cov_all"; o.replace = false;                                // the pair under the name of all three: another result, refused unless replaced
    CHECK_EQ(thrown([&] { noisecov_compute(ds, sub, o); }).substr(0, 14), std::string(code::already_exists));
  });
  CHECK(rel(c_ab.covariance, Pab * c_none.covariance * Pab.transpose()) < 1e-8);     // the sums over 40000 samples, projected before or after: double rounding
  CHECK(rel(c_all.covariance, Pabc * c_none.covariance * Pabc.transpose()) < 1e-8);
  CHECK(rel(c_ab.covariance, c_all.covariance) > 1e-3);
  CHECK(c_ab.info["run"]["projectors"]["option"] == Json(A + "," + B));
  CHECK(c_ab.info["run"]["projectors"]["applied"] == Json::array({A, B}));

  // ── inverse ──
  struct Kernel { MatrixXd K, C; Json info; bool hit = false; };
  auto kernel = [&](SubjectWriter& sub, DatasetWriter& ds, const std::string& cov, const std::string& name, bool mne, const Projectors& p) {
    InverseComputeOptions o;
    o.noise_cov = cov; o.name = name; o.mne_conventions = mne; o.projectors = p;
    const InverseComputeResult r = inverse_compute(ds, sub, o);
    Kernel k;
    k.hit = r.hit;
    k.info = r.info;
    k.K = read_matrix(sub, text(*sub.by_name("operator", name), "path"));
    const StoredStatistic st = read_statistic(sub, name + "_noise_cov");
    k.C.resize(M, M);
    for (int i = 0; i < M; ++i) for (int j = 0; j < M; ++j) k.C(i, j) = st.values[static_cast<std::size_t>(i * M + j)];
    return k;
  };
  Kernel kb_all, kb_ab, km_all, km_ab;
  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    kb_all = kernel(sub, ds, "cov_none", "K_bst_all", false, Projectors::active);
    kb_ab = kernel(sub, ds, "cov_none", "K_bst_ab", false, Projectors::named({A, B}));
    km_all = kernel(sub, ds, "cov_none", "K_mne_all", true, Projectors::active);
    km_ab = kernel(sub, ds, "cov_none", "K_mne_ab", true, Projectors::named({A, B}));
    CHECK(has(thrown([&] { kernel(sub, ds, "cov_none", "K_bad", false, Projectors::named({"nothing"})); }), "is not one of the channels' projectors"));
  });
  // Brainstorm: P on the gain and both sides of the covariance
  CHECK(rel(kb_ab.C, Pab * c_none.covariance * Pab.transpose()) < 1e-10);
  CHECK(rel(kb_all.C, Pabc * c_none.covariance * Pabc.transpose()) < 1e-10);
  {
    const MatrixXd G = gain.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
    const std::vector<std::string> ty(static_cast<std::size_t>(M), "MEG MAG");
    CHECK(rel(kb_ab.K, nxr::meeg::unconstrainedKernel(Pab * G, Pab * c_none.covariance * Pab.transpose(), ty).K) < 1e-5);
    CHECK(rel(kb_all.K, nxr::meeg::unconstrainedKernel(Pabc * G, Pabc * c_none.covariance * Pabc.transpose(), ty).K) < 1e-5);
  }
  CHECK(kb_ab.info["computed"]["projectors"]["applied"] == Json::array({A, B}));
  CHECK(kb_ab.info["computed"]["projectors"]["choice"]["option"] == Json(A + "," + B));
  CHECK(kb_all.info["computed"]["projectors"]["applied"] == Json::array({A, B, C}));
  // MNE: the covariance P C P^T and the kernel ending with P: K u = 0 for the projectors applied, not for one left out
  CHECK(rel(km_ab.C, Pab * c_none.covariance * Pab.transpose()) < 1e-10);
  auto kills = [&](const MatrixXd& K, int k) { return (K * u[static_cast<std::size_t>(k)]).cwiseAbs().maxCoeff() < 1e-5 * K.cwiseAbs().rowwise().sum().maxCoeff(); };
  CHECK(kills(km_ab.K, 0) && kills(km_ab.K, 1) && !kills(km_ab.K, 2));
  CHECK(kills(km_all.K, 0) && kills(km_all.K, 1) && kills(km_all.K, 2));
  CHECK(km_ab.info["computed"]["projectors"]["kernel"] == Json::array({A, B}));
  CHECK(km_ab.info["computed"]["projectors"]["choice"]["option"] == Json(A + "," + B));

  // ── edit operator: C inactive, then active again ──
  auto edit = [&](const std::string& name, const std::string& status, const std::string& note) {
    task::EditOperatorRequest e;
    e.dataset = D; e.subject = "S1"; e.name = name; e.status = status; e.note = note; e.by = "a tester"; e.created_by = "test";
    return task::edit_operator(e);
  };
  const task::CommandResult e0 = edit(C, "0", "not the artefact it is named for");
  CHECK(e0.result["changed"].get<bool>());
  CHECK(!edit(C, "0", "again").result["changed"].get<bool>());             // already 0: nothing appended
  {
    const task::CommandResult r = read("active");
    CHECK(rel(floats(r.arrays.at(0)), Pab * X) < 1e-6);                  // active no longer applies C
    CHECK(r.result["projection"]["applied"] == Json::array({A, B}));
    CHECK(floats(read(A + "," + B + "," + C).arrays.at(0)) == floats(r_all.arrays.at(0)));   // named: applied whatever its status
  }
  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    NoiseCovOptions o = no;
    o.name = "cov_all";                                                   // the key changed (C no longer applied): another result, refused unless replaced
    CHECK_EQ(thrown([&] { noisecov_compute(ds, sub, o); }).substr(0, 14), std::string(code::already_exists));
    o.name = "cov_active_ab";
    const NoiseCovResult c = noisecov_compute(ds, sub, o);
    CHECK(!c.hit);
    CHECK(rel(c.covariance, c_ab.covariance) < 1e-12);
    CHECK(c.info["run"]["projectors"]["applied"] == Json::array({A, B}));
    CHECK(has(thrown([&] { kernel(sub, ds, "cov_none", "K_bst_all", false, Projectors::active); }), code::already_exists));   // the kernel's key changed too
    const Kernel k = kernel(sub, ds, "cov_none", "K_bst_active_ab", false, Projectors::active);
    CHECK(!k.hit);
    CHECK(k.K == kb_ab.K);                                                // the default no longer applies C: the kernel of the pair, byte for byte
    CHECK(k.info["computed"]["projectors"]["applied"] == Json::array({A, B}));
  });
  const task::CommandResult e1 = edit(C, "1", "restored");
  CHECK(e1.result["changed"].get<bool>());
  CHECK(floats(read("active").arrays.at(0)) == floats(r_all.arrays.at(0)));   // byte for byte as before
  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    NoiseCovOptions o = no;
    o.name = "cov_all";
    const NoiseCovResult c = noisecov_compute(ds, sub, o);
    CHECK(c.hit);                                                         // the key of before: the result of before
    CHECK(c.covariance == c_all.covariance);
    const Kernel k = kernel(sub, ds, "cov_none", "K_bst_all", false, Projectors::active);
    CHECK(k.hit);
    CHECK(k.K == kb_all.K);
    const auto op = sub.by_name("operator", C);
    const Json pj = Json::parse(opt_text(*op, "producer_json").value_or("{}"));
    CHECK(pj["edits"].size() == 2);
    CHECK(pj["edits"][0]["from"] == 1 && pj["edits"][0]["to"] == 0 && pj["edits"][0]["by"] == "a tester" && pj["edits"][0]["note"] == "not the artefact it is named for");
    CHECK(pj["edits"][1]["from"] == 0 && pj["edits"][1]["to"] == 1 && pj["edits"][1]["note"] == "restored");
    CHECK(Json::parse(opt_text(*op, "params_json").value_or("{}"))["status"] == 1);
  });
  CHECK(has(thrown([&] { edit("headmodel", "0", ""); }), "projector operators only"));
  CHECK(has(thrown([&] { edit("nothing", "0", ""); }), "no_such_row"));
  CHECK(has(thrown([&] { edit(C, "2", ""); }), "0 inactive or 1 active"));

  // ── the faces' run (Python, JavaScript): `projectors` among the parameters, a list or comma text; a run over a time range, its rerun (a hit) and a dry run of the whole recording
  {
    face::RunRequest rq;
    rq.pipeline = "psd";
    rq.parameters = Json{{"recording", "rest"}, {"segment_seconds", 2.0}, {"projectors", Json::array({A, B})}};
    rq.from = 0.0; rq.to = 20.0;
    const Json r = face::run(D, "S1", rq);
    CHECK_EQ(r["cache"].get<std::string>(), std::string("computed"));
    CHECK_EQ(face::run(D, "S1", rq)["cache"].get<std::string>(), std::string("hit"));
    rq.parameters["projectors"] = A + "," + B + "," + C;                 // comma text: another choice, another result (refused under the same name unless replaced)
    CHECK(has(thrown([&] { face::run(D, "S1", rq); }), "already_exists"));
    rq.parameters["projectors"] = 3;
    CHECK(has(thrown([&] { face::run(D, "S1", rq); }), "projectors is active, none"));
    face::RunRequest dry;
    dry.pipeline = "psd";
    dry.parameters = Json{{"recording", "rest"}};
    dry.all = true;
    dry.dry_run = true;
    const Json d = face::run(D, "S1", dry);
    CHECK(!d["runs"].empty() && d["runs"][0]["cache"] == "planned");
  }

  fs::remove_all(root);
  return nxrtest::finish("test_projector_choice");
}

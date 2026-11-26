// A NOISE COVARIANCE OF ANOTHER CHANNEL SET (`run inverse --noise-cov`), matched to the head model's channels by name when the runner binds it. A synthetic subject with three channel sets:
// the run's (10 magnetometers C0..C9, three active SSP projectors, the head model's), an empty room's (the same names in another order plus one extra channel, no projectors) and a short one
// (C3 missing). Checks:
//  - the kernel from the empty room's covariance equals the kernel from the same covariance stored on the run's channel set, to rounding (none and active projectors, Brainstorm and MNE);
//  - the projectors applied are the run's channel set's (active: all three), not the empty room's (none);
//  - the provenance records the match (the covariance's channel set, the mapping by name, the dropped channel), and the cache key holds it (a rerun is a hit);
//  - a covariance missing a channel of the head model is refused, naming it.
// Synthetic data only.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <random>

#include "check.h"
#include "meeg_runs.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/projectors.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"

using namespace nxr::io;
using namespace nxrtest;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_channels_by_name_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  const std::int64_t M = 10, T = 6000, nv = 12;
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
  MatrixXd gain(M, 3 * nv), X(M, T), extra(1, T);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < 3 * nv; ++j) gain(i, j) = s * nd(rng);
    for (int t = 0; t < T; ++t) X(i, t) = s * nd(rng) * (1 + 0.1 * i);
  }
  for (int t = 0; t < T; ++t) extra(0, t) = s * nd(rng);
  std::vector<std::string> run_names;
  for (int i = 0; i < M; ++i) run_names.push_back("C" + std::to_string(i));
  // the empty room's channels: the run's in another order, with an extra one in the middle; the short set: C3 missing
  const std::vector<std::string> er_names = {"C7", "C2", "C9", "C0", "X1", "C5", "C1", "C8", "C3", "C6", "C4"};
  const std::vector<std::string> short_names = {"C9", "C8", "C7", "C6", "C5", "C4", "C2", "C1", "C0"};
  std::vector<Eigen::VectorXd> u(3, Eigen::VectorXd(M));
  for (auto& v : u) { for (int i = 0; i < M; ++i) v(i) = nd(rng); v /= v.norm(); }
  std::vector<std::string> projs;

  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      add_surface(sub, {.name = "cortex"}, V, F);
      const std::string surface_id = text(*sub.by_name("manifold", "cortex"), "id");
      // a channel set with its names, types and a recording on it (the rows of the run's X by name; X1 the extra row)
      auto channel_set = [&](const std::string& name, const std::vector<std::string>& names, const std::string& recording) {
        const std::int64_t n = static_cast<std::int64_t>(names.size());
        const std::string cid = uuid7();
        std::vector<float> pos(static_cast<std::size_t>(n * 3));
        for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
        ManifoldSpec m;
        m.id = cid; m.name = name; m.path = "sensor/" + name; m.type = "points"; m.n_vertices = n;
        m.topology = {"none", 0, 1, std::nullopt, {}};
        GeometrySpec g;
        g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/" + name;
        m.geometry = g;
        m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/" + name, view(pos, {n, 3}), {}, std::nullopt, "manifold", cid); };
        sub.manifold(std::move(m));
        std::vector<std::int64_t> number(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) number[static_cast<std::size_t>(i)] = i;
        nxr::bst::LabellingOptions lo;
        lo.manifold_id = cid; lo.dictionary = name + "_names"; lo.description = "the channel names";
        nxr::bst::labelling(sub, "sensor/" + name + "_names", number, names, lo);
        nxr::bst::label_set(sub, "sensor/" + name + "_labels", {{"type", std::vector<std::int64_t>(n, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(n, 0), {"good", "bad"}, {}}}, cid,
                            "the montage families", std::nullopt, [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
        ManifoldSpec tm;
        tm.name = recording + "_time"; tm.path = "time/" + recording + "_time"; tm.type = "line"; tm.n_vertices = T;
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
        std::vector<float> xf(static_cast<std::size_t>(n * T));
        for (std::int64_t i = 0; i < n; ++i) {
          const auto it = std::find(run_names.begin(), run_names.end(), names[static_cast<std::size_t>(i)]);
          for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(it == run_names.end() ? extra(0, t) : X(it - run_names.begin(), t));
        }
        FieldSpec rec;
        rec.name = recording; rec.path = task::Subject::of(sub).node_path(prod, recording); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {n, T});
        rec.array.chunks = Shape{n, 4096};
        sub.field(std::move(rec));
        return cid;
      };
      const std::string cid = channel_set("sensors", run_names, "rest");
      channel_set("emptyroom_sensors", er_names, "emptyroom");
      channel_set("short_sensors", short_names, "short");
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
      std::vector<nxr::bst::ProjectorEntry> es;
      for (int k = 0; k < 3; ++k) {
        nxr::bst::ProjectorEntry e;
        e.order = k + 1; e.comment = "ssp " + std::to_string(k + 1); e.method = "SSP"; e.kind = "ssp"; e.status = 1; e.m = static_cast<std::size_t>(M); e.k = 1;
        e.components.assign(u[static_cast<std::size_t>(k)].data(), u[static_cast<std::size_t>(k)].data() + M); e.mask = {true};
        es.push_back(e);
      }
      projs = nxr::bst::export_projectors(sub, es, "sensors", cid, "");
    });
    ds.close();
  }
  CHECK_EQ(projs.size(), std::size_t(3));
  auto with = [&](const std::function<void(DatasetWriter&, SubjectWriter&)>& f) {
    DatasetWriter ds = open_writer(root / "demo");
    {
      SubjectWriter sub = SubjectWriter::open(ds, "S1");
      f(ds, sub);
    }
    ds.close();
  };

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
    const Eigen::Index n = static_cast<Eigen::Index>(std::sqrt(static_cast<double>(st.values.size())));
    k.C.resize(n, n);
    for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) k.C(i, j) = st.values[static_cast<std::size_t>(i * n + j)];
    return k;
  };
  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    NoiseCovOptions o;
    o.chunks = "all"; o.projectors = Projectors::none;
    o.recording = "rest"; o.name = "cov_rest"; noisecov_compute(ds, sub, o);
    o.recording = "emptyroom"; o.name = "cov_er"; const NoiseCovResult er = noisecov_compute(ds, sub, o);
    CHECK_EQ(er.channels.size(), std::size_t(11));                         // the empty room's own 11 channels, X1 among them
    o.recording = "short"; o.name = "cov_short"; noisecov_compute(ds, sub, o);
  });

  with([&](DatasetWriter& ds, SubjectWriter& sub) {
    for (const bool mne : {false, true})
      for (const Projectors& p : {Projectors::none, Projectors::active}) {
        const std::string tag = std::string(mne ? "mne" : "bst") + (p.choice == Projectors::Choice::none ? "_none" : "_active");
        const Kernel same = kernel(sub, ds, "cov_rest", "K_rest_" + tag, mne, p);
        const Kernel other = kernel(sub, ds, "cov_er", "K_er_" + tag, mne, p);
        const double ek = rel(other.K, same.K), ec = rel(other.C, same.C);
        std::printf("  %-10s  kernel %.2e  covariance %.2e\n", tag.c_str(), ek, ec);
        CHECK(ek < 1e-6);                                                  // the kernels are stored as float32
        CHECK(ec < 1e-12);
        CHECK(other.info["computed"]["projectors"] == same.info["computed"]["projectors"]);   // the run's channel set's projectors, handed alike
        if (!mne && p.choice == Projectors::Choice::active) CHECK(other.info["computed"]["projectors"]["applied"] == Json::array({projs[0], projs[1], projs[2]}));
        if (mne && p.choice == Projectors::Choice::active) CHECK(other.info["computed"]["projectors"]["kernel"] == Json::array({projs[0], projs[1], projs[2]}));
        CHECK(!same.info["computed"].contains("noise_cov_channels"));      // on the head model's own set: nothing matched
        const Json& m = other.info["computed"]["noise_cov_channels"];
        CHECK(m["by"] == "name" && m["channels"] == "emptyroom_sensors" && m["matched_to"] == "sensors");
        CHECK(m["dropped"] == Json::array({"X1"}));
        CHECK(m["mapping"].size() == std::size_t(M));
        for (int i = 0; i < M; ++i) {
          CHECK(m["mapping"][static_cast<std::size_t>(i)]["name"] == run_names[static_cast<std::size_t>(i)] && m["mapping"][static_cast<std::size_t>(i)]["to"] == i);
          CHECK(m["mapping"][static_cast<std::size_t>(i)]["from"] == std::find(er_names.begin(), er_names.end(), run_names[static_cast<std::size_t>(i)]) - er_names.begin());
        }
      }
    // the projectors change the kernel (they are applied, the run's)
    CHECK(rel(kernel(sub, ds, "cov_er", "K_er_bst_active", false, Projectors::active).K, kernel(sub, ds, "cov_er", "K_er_bst_none", false, Projectors::none).K) > 1e-3);
    CHECK(kernel(sub, ds, "cov_er", "K_er_bst_active", false, Projectors::active).hit);   // the same request again: a hit
    // a covariance missing one of the head model's channels: refused, naming it
    const std::string why = thrown([&] { kernel(sub, ds, "cov_short", "K_short", false, Projectors::active); });
    std::printf("  missing: %s\n", why.c_str());
    CHECK(has(why, "does not hold 1 channel(s) of sensors") && has(why, ": C3"));
  });

  fs::remove_all(root);
  return nxrtest::finish("test_channels_by_name");
}

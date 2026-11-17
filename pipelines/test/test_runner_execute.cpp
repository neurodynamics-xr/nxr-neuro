// The runner's execution on a small synthetic subject (no real data): the band_power_test fixture definition (pipelines/test/definitions/, not shipped) run through io's chunk service, its band power
// equal to a direct computation on the whole recording (the band's taps over every sample, the kernel, |x|², the mean over the span, the bad span left out);
// the same result from a recording stored in another chunk width (the halo makes each chunk exact); chunk 0 by default; the field stored on the kernel's
// target with its provenance; a rerun a cache hit; another key refused, then replaced with --replace (moved aside); the analytic chain's power equal to
// re² + im² of the band's even and odd taps; the phase current stored as a vector3 field; and the command.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/filterbank.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/runner.h"
#include "nxr/taps.h"

using namespace nxr::io;
static const std::string kBandPower = std::string(NXR_TEST_DEFINITIONS_DIR) + "/band_power_test.yaml";   // the test fixture definition (not shipped)
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
double rel(const std::vector<float>& a, const std::vector<double>& b) {
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]) / std::max(1e-30, std::abs(b[i])));
  return a.size() == b.size() ? worst : 1e30;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_runner_execute_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(2, false);
  const int N = b.n, M = 3, T = 9000;
  const double sfreq = 250.0;
  double bad0 = 20.0, bad1 = 21.0;                   // the bad span: samples 5000..5249
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (int c = 0; c < M; ++c)
    for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c * T + t)] = static_cast<float>(std::sin(2 * M_PI * (9.0 + c) * t / sfreq + c) + 0.3 * nd(rng));
  std::vector<float> kf(static_cast<std::size_t>(3 * N * M)), kfix(static_cast<std::size_t>(N * M));
  for (auto& k : kf) k = static_cast<float>(nd(rng));
  for (auto& k : kfix) k = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {   // the same samples, two chunk widths
        FieldSpec rec;
        rec.name = name; rec.path = node_path(sub.db(), prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        const std::string rid = sub.field(std::move(rec));
        nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      }
      std::vector<std::int32_t> chans = {0, 1, 2};
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to, const char* method) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        op.cols.json("params_json", Json{{"inverse", Json{{"measure", method}}}});
        sub.operator_(std::move(op));
      };
      kernel("free", kf, 3 * N, "vector3", "MNE");
      kernel("fixed", kfix, N, "scalar", "dSPM");
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");

  // THE DIRECT COMPUTATION on the whole recording: the band's taps over every sample (reflected at the recording's ends), the kernel, |x|² summed over a
  // source's three rows, the mean over [s0, s1) without the bad samples
  auto direct = [&](const std::vector<float>& K, int rows, int per, double lo, double hi, std::int64_t s0, std::int64_t s1, bool analytic) {
    const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, lo, hi); };
    const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
    const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false);
    const auto im = analytic ? nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true) : std::vector<float>();
    std::vector<double> out(static_cast<std::size_t>(rows / per), 0.0);
    std::int64_t n = 0;
    for (std::int64_t t = s0; t < s1; ++t) {
      const double sec = t / sfreq;
      if (sec >= bad0 && sec < bad1) continue;
      ++n;
      for (int r = 0; r < rows; ++r) {
        double a = 0, bb = 0;
        for (int c = 0; c < M; ++c) {
          a += static_cast<double>(K[static_cast<std::size_t>(r * M + c)]) * re[static_cast<std::size_t>(c * T + t)];
          if (analytic) bb += static_cast<double>(K[static_cast<std::size_t>(r * M + c)]) * im[static_cast<std::size_t>(c * T + t)];
        }
        const double ar = static_cast<float>(a), br = static_cast<float>(bb);   // the kernel's output is float32, as the block's
        out[static_cast<std::size_t>(r / per)] += static_cast<float>(ar * ar + br * br);
      }
    }
    for (auto& v : out) v /= static_cast<double>(n);
    return out;
  };

  // 1. the fixture definition, every chunk: one field a recording, on the kernel's target, equal to the direct computation; both chunk widths agree
  const g::Definition bp = g::load(kBandPower);
  g::RunOptions all;
  all.chunks = "all";
  const auto r1 = g::run(ds, sub, bp, Json::object(), all);
  CHECK_EQ(static_cast<int>(r1.size()), 2);
  const auto ref = direct(kf, 3 * N, 3, 8, 12, 0, T, false);
  std::vector<float> got[2];
  for (int i = 0; i < 2 && i < static_cast<int>(r1.size()); ++i) {
    const Json& st = r1[static_cast<std::size_t>(i)]["stored"][0];
    std::vector<std::int64_t> shape;
    got[i] = read_floats((sub.folder() / (st["path"].get<std::string>())), &shape);
    const auto row = sub.db().read("field", st["id"].get<std::string>());
    CHECK(row && text(*row, "manifold_id") == surf && text(*row, "kind") == "band power" && text(*row, "value_type") == "scalar" && shape == std::vector<std::int64_t>{N});
    const Json pj = Json::parse(text(*row, "producer_json"));
    CHECK(pj["pipeline"]["name"] == "band_power_test" && pj["parameters"]["band"] == Json::array({8.0, 12.0}) && pj["inputs"]["kernel"]["name"] == "free" &&
          pj["run"]["bad_samples"].get<std::int64_t>() == 250 && pj.contains("cache"));
    std::printf("  %s: %zu values, %lld blocks, halo %lld, worst relative difference to the direct computation %.2e\n", r1[static_cast<std::size_t>(i)]["anchor"].get<std::string>().c_str(),
                got[i].size(), static_cast<long long>(pj["run"]["blocks"].get<std::int64_t>()), static_cast<long long>(pj["run"]["halo"].get<std::int64_t>()), rel(got[i], ref));
    CHECK(rel(got[i], ref) < 1e-5);
  }
  CHECK(rel(got[0], std::vector<double>(got[1].begin(), got[1].end())) < 1e-6);   // the chunk width does not change the result

  // 2. a rerun: a cache hit; the default span (chunk 0) is another key: refused, then replaced (moved aside)
  const auto r2 = g::run(ds, sub, bp, Json::object(), all);
  CHECK(r2.size() == 2 && r2[0]["hit"] == true && r2[1]["hit"] == true);
  std::string why;
  try { g::run(ds, sub, bp, Json::object(), [] { g::RunOptions o; o.chunks = "0"; return o; }()); } catch (const Error& e) { why = e.code(); }
  CHECK_EQ(why, std::string(code::already_exists));
  g::RunOptions zero;
  zero.chunks = "0";
  zero.replace = true;
  const auto r3 = g::run(ds, sub, bp, Json::object(), zero);
  CHECK(r3.size() == 2 && r3[0]["hit"] == false && r3[0]["replaced"] == true && r3[0]["run"]["span"]["chunks_read"] == Json::array({0, 0}));
  const auto ref0 = direct(kf, 3 * N, 3, 8, 12, 0, 4096, false);
  CHECK(rel(read_floats((sub.folder() / (r3[0]["stored"][0]["path"].get<std::string>()))), ref0) < 1e-5);
  CHECK(!fs::is_empty(root / "demo"));
  bool aside = false;
  for (const auto& e : fs::directory_iterator(root)) if (e.path().filename().string().rfind("_replaced-", 0) == 0) aside = true;
  CHECK(aside);

  // 3. the analytic chain on the fixed kernel: power = re² + im² of the band's even and odd taps; the phase current a vector3 field
  const std::string phase = R"(pipeline: phase_test
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: $kernel}}
parameters: {band: [8.0, 12.0], kernel: fixed}
steps:
  chunks:
    over: recording
    do:
      - {step: analytic, in: {x: recording}, with: {band: $band}, out: z}
      - {step: apply_kernel, in: {x: z, kernel: kernel}, out: j}
      - {step: power, in: {x: j}, out: p}
      - {step: phase_current, in: {x: j}, out: h}
      - {step: measure, in: {x: p}, with: {bad_segments: ignore}, out: power}
      - {step: measure, in: {x: h}, with: {bad_segments: ignore}, out: current}
outputs:
  power: {kind: band power}
  current: {kind: phase current}
)";
  const g::Definition pd = g::parse(parse_yaml(phase, "phase.yaml"), "phase.yaml", phase);
  const auto r4 = g::run(ds, sub, pd, Json::object(), all);
  CHECK(r4.size() == 1);
  if (r4.size() == 1) {
    bad0 = bad1;   // ignore: no sample left out (the direct computation's bad span emptied)
    const auto refa = direct(kfix, N, 1, 8, 12, 0, T, true);
    CHECK(rel(read_floats((sub.folder() / (r4[0]["stored"][0]["path"].get<std::string>()))), refa) < 1e-5);
    std::vector<std::int64_t> shape;
    const auto cur = read_floats((sub.folder() / (r4[0]["stored"][1]["path"].get<std::string>())), &shape);
    bool finite = true;
    for (float v : cur) finite = finite && std::isfinite(v);
    const auto row = sub.db().read("field", r4[0]["stored"][1]["id"].get<std::string>());
    CHECK(finite && shape == (std::vector<std::int64_t>{N, 3}) && row && text(*row, "value_type") == "vector3");
  }
  // a free kernel's ℍ current through the form nx (n·x, linear: the phase kept), then its power: runs and stores a scalar map
  const std::string fphase = R"(pipeline: form_test
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}
steps:
  chunks:
    over: recording
    do:
      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}
      - {step: apply_kernel, in: {x: z, kernel: kernel}, out: j}
      - {step: form, in: {x: j}, with: {form: nx}, out: n}
      - {step: power, in: {x: n}, out: p}
      - {step: measure, in: {x: p}, with: {bad_segments: ignore}, out: power}
outputs:
  power: {kind: band power}
)";
  const auto r5 = g::run(ds, sub, g::parse(parse_yaml(fphase, "form.yaml"), "form.yaml", fphase), Json::object(), [] { g::RunOptions o; o.chunks = "0"; return o; }());
  CHECK(r5.size() == 1 && r5[0]["stored"].size() == 1 && read_floats((sub.folder() / (r5[0]["stored"][0]["path"].get<std::string>()))).size() == static_cast<std::size_t>(N));

#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    ds.close();
    CHECK_EQ(std::system((cli + " run " + kBandPower + " --dataset " + d + " --subject S1 --chunks 0 > " + log).c_str()), 0);        // chunk 0, as stored last: a hit
    CHECK_EQ(std::system((cli + " run " + kBandPower + " --dataset " + d + " --subject S1 --band '[15, 29]' --all --replace >> " + log).c_str()), 0);
    CHECK(std::system((cli + " run " + kBandPower + " --dataset " + d + " --subject S1 --band '[15, 29]' --chunks 0 > /dev/null 2>&1").c_str()) != 0);   // another span: refused
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_runner_execute");
}

// The runner's plan on a small synthetic subject (no real data): YAML read as nxr reads it (PyYAML's safe_load types); a definition's structure checked
// (nxr's refusals, each named); the inputs found by the model (the anchor once per recording, an optional input absent, a `where` on params_json with a
// $parameter, an ambiguous match refused); every step's arrows checked before any data is read (Nyquist, D106, a phase of a real field or of an envelope,
// a form of the wrong type, a measure of a complex field); what each step makes (manifolds, value type, ℝ/ℂ); and the command (`run --dry-run`, `--list`).
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::io;
static const std::string kBandPower = std::string(NXR_TEST_DEFINITIONS_DIR) + "/band_power_test.yaml";   // the test fixture definition (not shipped)
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
g::Definition def(const std::string& yaml) { return g::parse(parse_yaml(yaml, "test.yaml"), "test.yaml", yaml); }

// a phase chain on the free kernel; `body` the steps of the loop
std::string chain(const std::string& body, const std::string& outputs = "out: {kind: statistic}") {
  return "pipeline: t\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}\n"
         "steps:\n  chunks:\n    over: recording\n    do:\n" + body + "outputs: {" + outputs + "}\n";
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);

  // 1. YAML, typed as PyYAML's safe_load types it
  const Json y = parse_yaml("a: 8\nb: 8.5\nc: [15, 29]\nd: dSPM\ne: '8'\nf: ~\ng: true\nh: no\ni: 1e3\nj: .inf\nk: &x {p: 1}\nl: *x\nm: \"A²·m²\"\n");
  CHECK(y["a"] == 8 && y["a"].is_number_integer() && y["b"] == 8.5 && y["c"] == Json::array({15, 29}) && y["d"] == "dSPM" && y["e"] == "8" && y["f"].is_null());
  CHECK(y["g"] == true && y["h"] == false && y["i"] == 1000.0 && std::isinf(y["j"].get<double>()) && y["l"]["p"] == 1 && y["m"] == "A²·m²");
  CHECK(has(refusal([] { parse_yaml("a: [1, 2\n", "bad.yaml"); }), "bad.yaml"));

  // 2. THE STRUCTURE (definition.cpp: nxr's parse), each refusal named
  const std::string ok = chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\n");
  CHECK(refusal([&] { def(ok); }).empty());
  CHECK(has(refusal([&] { def(ok + "extra: 1\n"); }), "unknown section(s) extra"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}\n")); }), "nothing leaves the loop without an accumulate step"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {y: recording}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\n")); }), "'y' is not one of its ports"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: nothing}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\n")); }), "neither an input nor an earlier step's output"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: $nope}, out: out}\n")); }), "$nope is not a parameter"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {window: 2}, out: out}\n")); }), "'window' is not one of its parameters"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: [p, q]}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\n")); }), "it makes 1 thing(s)"));
  CHECK(has(refusal([&] { def(chain("      - {node: signal.field_form, in: {x: recording}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\n", "nope: {kind: test}")); }), "outputs.nope: not made by any step"));
  CHECK(has(refusal([&] { def("pipeline: t\ninputs: {r: {field: recording}}\nsteps:\n  after:\n    - {node: signal.field_form, in: {x: r}, with: {form: power}, out: out}\noutputs: {out: {kind: statistic}}\n"); }), "a chunk step cannot be in 'after'"));
  CHECK(has(refusal([&] { def("pipeline: t\ninputs:\n  k: {operator: inverse kernel, from: r.manifold.0}\n  r: {field: recording}\nsteps: {}\noutputs: {out: {kind: statistic}}\n"); }), "relations start from an input declared BEFORE"));
  CHECK(has(refusal([] { g::step("mean_square"); }), "no pipeline step 'mean_square'"));
  CHECK(g::load(kBandPower).anchor() == std::optional<std::string>("recording"));

  // 3. A SUBJECT: a cortex, three sensors, two recordings on one time line, a free (MNE) and a fixed (dSPM) kernel, rest's bad segments
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_runner_plan_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(2, false);
  const int N = b.n, M = 3, T = 600;
  const double sfreq = 250.0;
  std::vector<float> xf(static_cast<std::size_t>(M * T), 0.0f), kf(static_cast<std::size_t>(3 * N * M), 0.0f), kfix(static_cast<std::size_t>(N * M), 0.0f);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string sid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sid = sub.id();
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string surf = add_surface(sub, so, b.P, b.F).id;
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
      std::string rid;
      for (const char* name : {"rest", "rest_b"}) {
        FieldSpec rec;
        rec.name = name; rec.path = node_path(sub.db(), prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.compress = false; rec.cols.set("unit", "T");
        const std::string id = sub.field(std::move(rec));
        if (std::string(name) == "rest") rid = id;
      }
      nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
      nxr::bst::time_spans(sub, "time/rest_bad_segments", {{1.0, 1.5, 0}}, {"BAD"}, bo);
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to, const char* method) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_value_type", "scalar").set("to_value_type", to);
        op.cols.json("params_json", Json{{"inverse", Json{{"measure", method}}}});
        sub.operator_(std::move(op));
      };
      kernel("free", kf, 3 * N, "vector3", "MNE");
      kernel("fixed", kfix, N, "scalar", "dSPM");
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");

  // 4. THE FIXTURE DEFINITION (pipelines/test/definitions/band_power_test.yaml, not shipped): one run per recording; rest's bad segments found, rest_b's absent (optional); the MNE kernel by its params_json
  const g::Definition bp = g::load(kBandPower);
  const auto runs = g::plan_runs(bp, sub.db(), sid, Json::object());
  CHECK_EQ(static_cast<int>(runs.size()), 2);
  if (runs.size() == 2) {
    const g::Plan& P = runs[0];
    CHECK(text(*P.found.at("recording").row, "name") == "rest" && P.found.count("bad") && !runs[1].found.count("bad") && text(*P.found.at("kernel").row, "name") == "free");
    const g::Value& src = P.values.at("sources");
    CHECK(src.value_type == "vector3" && src.chunked && !src.complex && src.manifolds.size() == 2 && src.manifolds[1] == *P.found.at("recording").time);
    const g::Value& bpow = P.values.at("band_power");
    CHECK(!bpow.manifolds.empty() && bpow.manifolds[0] == P.found.at("kernel").manifolds[1] && bpow.value_type == "scalar" && !bpow.chunked && !bpow.time);
    const std::string txt = P.text(sub.db());
    CHECK(has(txt, "pipeline band_power_test@1 — recording 'rest'") && has(txt, "output band_power  a statistic"));
    std::cout << txt;
  }
  // --set: dSPM picks the fixed kernel (a scalar source); an unknown parameter and a band past Nyquist are refused
  const auto d2 = g::plan_runs(bp, sub.db(), sid, Json{{"method", "dSPM"}});
  CHECK(d2.size() == 2 && d2[0].values.at("sources").value_type == "scalar");
  CHECK(has(refusal([&] { g::plan_runs(bp, sub.db(), sid, Json{{"bnad", 1}}); }), "--set bnad: not parameters of band_power_test"));
  CHECK(has(refusal([&] { g::plan_runs(bp, sub.db(), sid, Json{{"band", Json::array({100, 200})}}); }), "reaches past Nyquist"));
  CHECK(has(refusal([&] { g::plan_runs(bp, sub.db(), sid, Json{{"method", "sLORETA"}}); }), "inputs.kernel (for recording 'rest'): nothing in the subject matches"));
  // a definition NAMES its recording and kernel (`recording` / `kernel` parameters; `none` = no filter): one of two recordings plans one run; a kernel by name
  const auto one = g::plan_runs(bp, sub.db(), sid, Json{{"recording", "rest_b"}});
  CHECK(one.size() == 1 && text(*one[0].found.at("recording").row, "name") == "rest_b");
  const auto none_named = g::plan_runs(bp, sub.db(), sid, Json{{"recording", "none"}, {"kernel", "none"}});
  CHECK_EQ(static_cast<int>(none_named.size()), 2);
  const auto fixed = g::plan_runs(bp, sub.db(), sid, Json{{"recording", "rest"}, {"kernel", "fixed"}, {"method", "dSPM"}});
  CHECK(fixed.size() == 1 && text(*fixed[0].found.at("kernel").row, "name") == "fixed" && fixed[0].values.at("sources").value_type == "scalar");
  const auto freek = g::plan_runs(bp, sub.db(), sid, Json{{"kernel", "free"}});
  CHECK(freek.size() == 2 && text(*freek[0].found.at("kernel").row, "name") == "free");
  CHECK(has(refusal([&] { g::plan_runs(bp, sub.db(), sid, Json{{"recording", "nope"}}); }), "inputs.recording"));
  CHECK(has(refusal([&] { g::plan_runs(bp, sub.db(), sid, Json{{"kernel", "fixed"}}); }), "inputs.kernel"));   // a named kernel is also of the `method` (MNE): name its method too
  // every shipped definition takes them
  for (const auto& f : fs::directory_iterator(NXR_SHIPPED_DEFINITIONS_DIR)) {
    if (f.is_directory() || f.path().extension() != ".yaml") continue;   // the `pending` folder holds definitions not shipped
    const g::Definition sd = g::load(f.path().string());
    if (!sd.inputs.contains("recording") || !sd.inputs["recording"].value("each", false)) continue;   // the definitions that run once per recording
    CHECK(sd.parameters.contains("recording") && sd.parameters["recording"] == "none");
    CHECK(!sd.inputs.contains("kernel") || (sd.parameters.contains("kernel") && (sd.parameters["kernel"] == "none" || sd.parameters["kernel"] == "required"
          || sd.parameters["kernel"].is_null())));   // a required kernel is the caller's choice; so is an optional one (null: no kernel unless the caller names one)
  }
  // an ambiguous input is refused, naming the matches
  CHECK(has(refusal([&] { g::plan_runs(def("pipeline: t\ninputs:\n  r: {field: recording}\n  k: {operator: inverse kernel, from: r.manifold.0}\nsteps:\n  chunks:\n    over: r\n    do:\n"
                                           "      - {node: signal.field_form, in: {x: r}, with: {form: power}, out: p}\n      - {node: measurement.estimate, in: {x: p}, with: {method: sample}, out: out}\noutputs: {out: {kind: statistic}}\n"), sub.db(), sid, Json::object()); }),
            "inputs.r: 2 match (rest, rest_b)"));

  // 5. THE PHASE CHAIN: band → analytic → kernel → phase current (vector3 ℝ on the cortex × time) → measure (out of the loop)
  const std::string phase = chain("      - {node: signal.band_filter, in: {x: recording}, with: {band: [8, 12]}, out: xb}\n      - {node: signal.analytic, in: {x: xb}, out: z}\n"
                                  "      - {node: meeg.apply_kernel, in: {x: z, kernel: kernel}, out: j}\n      - {node: signal.field_form, in: {x: j}, with: {form: phase_current}, out: h}\n"
                                  "      - {node: signal.gradient_power, in: {x: j}, out: q}\n      - {node: measurement.estimate, in: {x: h}, with: {method: sample}, out: out}\n");
  const auto pr = g::plan_runs(def(phase), sub.db(), sid, Json::object());
  CHECK(pr.size() == 1);
  if (pr.size() == 1) {
    CHECK(pr[0].values.at("z").complex && pr[0].values.at("j").complex && pr[0].values.at("j").value_type == "vector3");
    CHECK(!pr[0].values.at("h").complex && pr[0].values.at("h").value_type == "vector3" && pr[0].values.at("q").value_type == "scalar" && pr[0].values.at("out").manifolds.size() == 1);
  }
  auto plan_of = [&](const std::string& body) { const std::string r = refusal([&] { g::plan_runs(def(chain(body)), sub.db(), sid, Json::object()); }); return r; };
  // a real field has no phase
  CHECK(has(plan_of("      - {node: meeg.apply_kernel, in: {x: recording, kernel: kernel}, out: j}\n      - {node: signal.field_form, in: {x: j}, with: {form: phase_current}, out: h}\n      - {node: measurement.estimate, in: {x: h}, with: {method: sample}, out: out}\n"),
            "x must be complex"));
  // the phase current's discretisation: a named option of that form only (vertex_gradient, the reference, or intrinsic_edges)
  {
    const std::string zj = "      - {node: signal.analytic, in: {x: recording}, out: z}\n      - {node: meeg.apply_kernel, in: {x: z, kernel: kernel}, out: j}\n";
    CHECK(has(plan_of(zj + "      - {node: signal.field_form, in: {x: j}, with: {form: phase_current, discretisation: cotan}, out: h}\n      - {node: measurement.estimate, in: {x: h}, with: {method: sample}, out: out}\n"),
              "discretisation"));
    CHECK(has(plan_of(zj + "      - {node: signal.field_form, in: {x: j}, with: {form: power, discretisation: intrinsic_edges}, out: h}\n      - {node: measurement.estimate, in: {x: h}, with: {method: sample}, out: out}\n"),
              "discretisation"));
  }
  // D106: no linear map after the envelope
  CHECK(has(plan_of("      - {node: signal.envelope, in: {x: recording}, out: e}\n      - {node: meeg.apply_kernel, in: {x: e, kernel: kernel}, out: j}\n      - {node: measurement.estimate, in: {x: j}, with: {method: sample}, out: out}\n"),
            "(D106)"));
  const std::string head = "      - {node: signal.analytic, in: {x: recording}, out: z}\n      - {node: meeg.apply_kernel, in: {x: z, kernel: kernel}, out: j}\n";
  // a form: |q| of the current is its envelope (no phase); nx keeps it; a form the type lacks is refused with the type's forms
  CHECK(has(plan_of(head + "      - {node: signal.form, in: {x: j}, with: {form: absq}, out: a}\n      - {node: signal.field_form, in: {x: a}, with: {form: phase_current}, out: h}\n      - {node: measurement.estimate, in: {x: h}, with: {method: sample}, out: out}\n"),
            "x must be complex"));
  CHECK(plan_of(head + "      - {node: signal.form, in: {x: j}, with: {form: nx}, out: a}\n      - {node: signal.field_form, in: {x: a}, with: {form: phase_current}, out: h}\n      - {node: signal.field_form, in: {x: a}, with: {form: power}, out: q}\n      - {node: measurement.estimate, in: {x: q}, with: {method: sample}, out: out}\n").empty());
  CHECK(has(plan_of(head + "      - {node: signal.form, in: {x: j}, with: {form: f}, out: a}\n      - {node: measurement.estimate, in: {x: a}, with: {method: sample}, out: out}\n"), "'f' is not a form of a ℍ field"));
  // a reduction of a complex field is refused
  CHECK(has(plan_of(head + "      - {node: measurement.estimate, in: {x: j}, with: {method: sample}, out: out}\n"), "x is complex"));

#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string(), log = (root / "cli.log").string();
    ds.close();
    CHECK_EQ(std::system((cli + " run " + kBandPower + " --dataset " + d + " --subject S1 --method dSPM --band '[15, 29]' --dry-run > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " list pipelines >> " + log).c_str()), 0);
    CHECK(std::system((cli + " run " + kBandPower + " --dataset " + d + " --subject S1 --band '[1, 200]' --dry-run > /dev/null 2>&1").c_str()) != 0);   // past Nyquist
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_runner_plan");
}

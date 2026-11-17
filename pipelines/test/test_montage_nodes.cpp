// The montage node (pipelines/src/montage_steps.cpp) run through the dispatch point on a synthetic sphere: it equals the compute function it wraps (nxr::montage::tenFive), the labels and the
// 10-10 and 10-20 subsets travel in the output's meta, precision follows the input, parameters and inputs are refused with their reason, and the node declares its traits.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include "check.h"
#include "nxr/io/subject_package.h"
#include "nxr/montage.h"
#include "nxr/pipelines/dispatch.h"
#include "nxr/pipelines/registry.h"
#include "synthetic_meshes.h"

namespace g = nxr::pipelines::graph;
using nxr::io::Json;

namespace {
g::Arrays run(g::Bound& bound, const Json& with, const g::Arrays& ins) {
  const g::StepSpec& node = g::step("eeg_montage_1005");
  Json p = node.parameters;
  for (auto it = with.begin(); it != with.end(); ++it) p[it.key()] = it.value();
  g::ExecContext ctx{bound, [](const Json&) {}, {}};
  g::Values planned;
  g::NodeCall prep = g::NodeCall::prepare(planned, p, ctx);
  g::dispatch(node, prep);
  g::NodeCall call = g::NodeCall::run(prep.made_state, ins, ctx);
  g::dispatch(node, call);
  return call.made_arrays;
}
std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
g::Array array(const std::vector<double>& v, std::int64_t rows, bool f32 = false) {
  g::Array a; a.rows = rows; a.cols = 3; a.data = v;
  a.meta = Json{{"precision", f32 ? "float32" : "float64"}};
  return a;
}
}  // namespace

int main() {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / ("nxr_pipelines_test_montage_nodes_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  nxr::io::CreatePackageOptions opt;
  opt.home_dataset = "demo";
  nxr::io::DatasetWriter pkg = nxr::io::create_package(root / "S.nxr.zarr", "S", opt);
  nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(pkg, "S");
  g::Bound db(sub, &pkg);

  const g::StepSpec& s = g::step("eeg_montage_1005");
  CHECK(s.traits.declared());
  CHECK(!s.inputs.empty() && !s.outputs.empty() && !s.description.empty() && !s.matches.empty());
  CHECK(s.run != nullptr && s.prepare != nullptr && s.plan != nullptr);
  CHECK(g::undeclared_traits().empty());

  Mesh m = icosphere(4);
  const double R = 0.095;
  for (auto& x : m.V) x *= R;
  std::vector<double> faces(m.F.begin(), m.F.end());
  const std::vector<double> lm = {0, R, 0, 0, -R, 0, -R, 0, 0, R, 0, 0};   // nasion, inion, LPA, RPA
  g::Arrays ins;
  ins["scalp_vertices"] = array(m.V, m.nV());
  ins["scalp_faces"] = array(faces, m.nF());
  ins["landmarks"] = array(lm, 4);
  const g::Arrays out = run(db, Json::object(), ins);

  // equal to the function it wraps
  nxr::montage::Landmarks L;
  L.nasion = {0, R, 0}; L.inion = {0, -R, 0}; L.lpa = {-R, 0, 0}; L.rpa = {R, 0, 0};
  const auto direct = nxr::montage::tenFive(m.V, m.F, L);
  const g::Array& pos = out.at("positions");
  CHECK_EQ(pos.rows, std::int64_t(345));
  CHECK_EQ(pos.cols, std::int64_t(3));
  CHECK(pos.data == direct.positions);
  CHECK(pos.meta["labels"].size() == 345 && pos.meta["labels"][0] == direct.labels[0]);
  CHECK(pos.meta["systems"]["10-10"].size() == 77 && pos.meta["systems"]["10-20"].size() == 21);
  CHECK(pos.meta["old_names"]["T3"] == "T7");
  CHECK(pos.meta["warnings"].size() == 2);

  // precision follows the input
  g::Arrays f32 = ins;
  f32["scalp_vertices"] = array(m.V, m.nV(), true);
  f32["landmarks"] = array(lm, 4, true);
  const g::Arrays out32 = run(db, Json::object(), f32);
  const g::Array& p32 = out32.at("positions");
  CHECK(p32.meta["precision"] == "float32");
  double worst = 0;
  for (std::size_t i = 0; i < p32.data.size(); ++i) { CHECK(p32.data[i] == static_cast<double>(static_cast<float>(p32.data[i]))); worst = std::max(worst, std::fabs(p32.data[i] - pos.data[i])); }
  CHECK(worst < 1e-5);

  // refusals name their reason
  CHECK(has(refusal([&] { run(db, Json{{"tolerance", 0}}, ins); }), "tolerance"));
  CHECK(has(refusal([&] { run(db, Json{{"max_iterations", 0}}, ins); }), "max_iterations"));
  g::Arrays mixed = ins;
  mixed["landmarks"] = array(lm, 4, true);
  CHECK(has(refusal([&] { run(db, Json::object(), mixed); }), "different precision"));
  g::Arrays shape = ins;
  shape["landmarks"] = array({0, 1, 0}, 1);
  CHECK(has(refusal([&] { run(db, Json::object(), shape); }), "landmarks [4 x 3]"));
  CHECK(has(refusal([&] { s.plan(g::Values{}, Json::object(), g::PlanContext{db}); }), "missing"));

  std::filesystem::remove_all(root);
  return nxrtest::finish("montage_nodes");
}

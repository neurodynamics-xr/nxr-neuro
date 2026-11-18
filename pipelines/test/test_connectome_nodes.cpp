// The connectome's registered nodes (pipelines/src/connectome_steps.cpp) run through the dispatch point on synthetic arrays: each equals the compute function it wraps, the chain in the
// spec's order (ends → gate → smoothed memberships → counts over a partition; and the materialised vertex connectome → weights → pruning → largest component) equals a direct sum
// over the partition and the dense reference, precision follows the input, parameters and inputs are refused with their reason, and every node declares its traits.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include "check.h"
#include "connectome_helpers.h"
#include "nxr/meeg/connectome.h"
#include "nxr/io/subject_package.h"
#include "nxr/geometry/surface_maps.h"
#include "nxr/pipelines/dispatch.h"
#include "nxr/pipelines/registry.h"

namespace g = nxr::pipelines::graph;
using nxr::io::Json;
using Dense = Eigen::MatrixXd;

namespace {

g::Arrays run(g::Bound& bound, const std::string& name, const Json& with, const g::Arrays& ins) {
  const g::StepSpec& node = g::step(name);
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

g::Array dense(const std::vector<double>& v, std::int64_t rows, std::int64_t cols, bool f32 = false) {
  g::Array a; a.rows = rows; a.cols = cols; a.data = v;
  a.meta = Json{{"precision", f32 ? "float32" : "float64"}};
  return a;
}
g::Array whole(const std::vector<std::int32_t>& v) { g::Array a; a.rows = static_cast<std::int64_t>(v.size()); a.cols = 1; a.data.assign(v.begin(), v.end()); return a; }
Dense from_coo(const g::Array& a) {
  Dense d = Dense::Zero(a.meta["shape"][0].get<int>(), a.meta["shape"][1].get<int>());
  for (std::int64_t i = 0; i < a.rows; ++i) d(static_cast<int>(a.data[static_cast<std::size_t>(3 * i)]), static_cast<int>(a.data[static_cast<std::size_t>(3 * i + 1)])) += a.data[static_cast<std::size_t>(3 * i + 2)];
  return d;
}
}  // namespace

int main() {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / ("nxr_pipelines_test_connectome_nodes_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  nxr::io::CreatePackageOptions opt;
  opt.home_dataset = "demo";
  nxr::io::DatasetWriter pkg = nxr::io::create_package(root / "S.nxr.zarr", "S", opt);
  nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(pkg, "S");
  g::Bound db(sub, &pkg);

  std::vector<double> vertexPositions; std::vector<int32_t> triangles;
  icosphere(2, vertexPositions, triangles, 0.08);
  const int nV = static_cast<int>(vertexPositions.size() / 3);
  ctest::Lcg gen(5);
  const auto b = ctest::makeBundle(gen, 200, 5, 0.08, 5, 5);
  std::vector<double> facesd(triangles.begin(), triangles.end());
  const g::Array positions = dense(vertexPositions, nV, 3), faces = dense(facesd, static_cast<std::int64_t>(triangles.size() / 3), 3);

  // every connectome node is registered with traits, parameters and ports
  for (const char* n : {"fiber_ends", "nearest_vertex", "end_gate", "registration_match", "carry_ends", "smooth_memberships", "partition_counts", "end_connectome", "connection_weights", "prune_below", "largest_component"}) {
    const g::StepSpec& s = g::step(n);
    CHECK(s.traits.declared());
    CHECK(!s.inputs.empty() && !s.outputs.empty() && !s.description.empty() && !s.matches.empty());
    CHECK(s.run != nullptr && s.prepare != nullptr && s.plan != nullptr);
  }
  CHECK(g::undeclared_traits().empty());

  // the ends, the gate
  const auto ends = run(db, "fiber_ends", Json{{"points_per_fiber", 5}}, {{"fibers", dense(b.points, 200 * 5, 3)}});
  CHECK_EQ(ends.at("end_points").rows, std::int64_t(400));
  const auto near = run(db, "nearest_vertex", Json::object(), {{"points", ends.at("end_points")}, {"positions", positions}});
  const auto want = nxr::geometry::nearest::nearestVertices<double>(vertexPositions, ends.at("end_points").data);
  CHECK(std::vector<std::int32_t>(near.at("vertex").data.begin(), near.at("vertex").data.end()) == want.vertex);
  CHECK(near.at("distance").data == want.distance);
  const auto none = run(db, "end_gate", Json{{"max_end_distance_mm", "none"}}, {{"end_distance", near.at("distance")}});
  CHECK_EQ(none.at("keep").meta["n_left_out"].get<int>(), 0);
  const auto five = run(db, "end_gate", Json::object(), {{"end_distance", near.at("distance")}});   // the default, 5 mm
  const auto tight = run(db, "end_gate", Json{{"max_end_distance_mm", 0.5}}, {{"end_distance", near.at("distance")}});
  CHECK(tight.at("keep").meta["n_left_out"].get<int>() > 0 && tight.at("keep").meta["n_left_out"].get<int>() <= 200);
  CHECK_EQ(five.at("keep").meta["max_end_distance_mm"].get<double>(), 5.0);

  // counts over a partition: the chain equals a direct sum over the partition of the smoothed vertex matrix
  std::vector<std::int32_t> label(static_cast<std::size_t>(nV));
  for (int v = 0; v < nV; ++v) label[static_cast<std::size_t>(v)] = (vertexPositions[static_cast<std::size_t>(3 * v)] < 0 ? 0 : 1) + 2 * (vertexPositions[static_cast<std::size_t>(3 * v + 2)] < 0 ? 0 : 1);
  const auto ref = ctest::referenceEnds(vertexPositions, b);
  for (int hops : {0, 3}) {
    const auto smoothedMemberships = run(db, "smooth_memberships", Json{{"hops", hops}}, {{"label", whole(label)}, {"faces", faces}});
    const auto countsOut = run(db, "partition_counts", Json::object(), {{"end_vertex", near.at("vertex")}, {"keep", none.at("keep")}, {"memberships", smoothedMemberships.at("memberships")}});
    const Dense smoothing = ctest::referenceSmoother(triangles, nV, hops);
    const Dense smoothedWeights = smoothing * ctest::referenceCounts(ref, nV) * smoothing.transpose();
    Dense sum = Dense::Zero(4, 4);
    for (int u = 0; u < nV; ++u) for (int v = 0; v < nV; ++v) sum(label[static_cast<std::size_t>(u)], label[static_cast<std::size_t>(v)]) += smoothedWeights(u, v);
    CHECK((from_coo(countsOut.at("counts")) - sum).cwiseAbs().maxCoeff() < 1e-10);
  }
  // the gate leaves fibers out of the counts
  {
    const auto smoothedMemberships = run(db, "smooth_memberships", Json{{"hops", 0}}, {{"label", whole(label)}, {"faces", faces}});
    const auto all = run(db, "partition_counts", Json::object(), {{"end_vertex", near.at("vertex")}, {"memberships", smoothedMemberships.at("memberships")}});   // keep is optional
    const auto gated = run(db, "partition_counts", Json::object(), {{"end_vertex", near.at("vertex")}, {"keep", tight.at("keep")}, {"memberships", smoothedMemberships.at("memberships")}});
    CHECK(from_coo(gated.at("counts")).sum() < from_coo(all.at("counts")).sum());
  }

  // the materialised vertex connectome: counts → weights → pruning → largest component equals the compute functions
  {
    const auto countsOut = run(db, "end_connectome", Json{{"n_vertices", nV}}, {{"end_vertex", near.at("vertex")}, {"keep", none.at("keep")}});
    CHECK((from_coo(countsOut.at("fiber_counts")) - ctest::withoutDiag(ctest::referenceCounts(ref, nV))).cwiseAbs().maxCoeff() == 0);
    const auto smoothedWeights = run(db, "connection_weights", Json{{"smooth_hops", 3}}, {{"fiber_counts", countsOut.at("fiber_counts")}, {"faces", faces}});
    const auto Pr = run(db, "prune_below", Json{{"prune_below", 0.01}}, {{"connection_weights", smoothedWeights.at("connection_weights")}});
    const Dense smoothing = ctest::referenceSmoother(triangles, nV, 3);
    Dense refW = ctest::withoutDiag(smoothing * ctest::withoutDiag(ctest::referenceCounts(ref, nV)) * smoothing.transpose());
    CHECK((from_coo(smoothedWeights.at("connection_weights")) - refW).cwiseAbs().maxCoeff() < 1e-12);
    const double t = 0.01 * refW.maxCoeff();
    for (int i = 0; i < nV; ++i) for (int j = 0; j < nV; ++j) if (refW(i, j) < t) refW(i, j) = 0;
    CHECK((from_coo(Pr.at("connection_weights")) - refW).cwiseAbs().maxCoeff() < 1e-12);
    const auto L = run(db, "largest_component", Json::object(), {{"connection_weights", Pr.at("connection_weights")}});
    const auto lab = ctest::referenceComponents(refW);
    CHECK(std::vector<int>(L.at("label").data.begin(), L.at("label").data.end()) == lab);
    CHECK(!L.at("vertices").data.empty());
  }

  // template fibers on the subject: the match, the ends carried
  {
    std::vector<double> cv; std::vector<int32_t> cf;
    icosphere(1, cv, cf);
    g::Array tp = dense({0, 42}, 1, 2), sp = dense({0, static_cast<double>(nV)}, 1, 2);
    const auto m = run(db, "registration_match", Json::object(), {{"template_sphere", dense(cv, 42, 3)}, {"template_pieces", tp}, {"subject_sphere", dense(vertexPositions, nV, 3)}, {"subject_pieces", sp}});
    CHECK_EQ(m.at("match").rows, std::int64_t(42));
    const auto carried = run(db, "carry_ends", Json::object(), {{"end_vertex", whole({0, 5, 41})}, {"match", m.at("match")}});
    CHECK(carried.at("end_vertex").data[1] == m.at("match").data[5]);
  }

  // precision follows the input; mixed precision is refused
  {
    std::vector<double> pf(vertexPositions);
    const auto e32 = run(db, "fiber_ends", Json{{"points_per_fiber", 5}}, {{"fibers", dense(b.points, 1000, 3, true)}});
    CHECK_EQ(e32.at("end_points").meta["precision"].get<std::string>(), std::string("float32"));
    const auto n32 = run(db, "nearest_vertex", Json::object(), {{"points", e32.at("end_points")}, {"positions", dense(vertexPositions, nV, 3, true)}});
    CHECK_EQ(n32.at("distance").meta["precision"].get<std::string>(), std::string("float32"));
    CHECK(has(refusal([&] { run(db, "nearest_vertex", Json::object(), {{"points", e32.at("end_points")}, {"positions", positions}}); }), "different precision"));
  }

  // refusals say their reason
  CHECK(has(refusal([&] { run(db, "fiber_ends", Json::object(), {{"fibers", dense(b.points, 1000, 3)}}); }), "points_per_fiber"));
  CHECK(has(refusal([&] { run(db, "end_gate", Json{{"max_end_distance_mm", -1}}, {{"end_distance", near.at("distance")}}); }), "max_end_distance_mm"));
  CHECK(has(refusal([&] { run(db, "end_connectome", Json::object(), {{"end_vertex", near.at("vertex")}}); }), "n_vertices"));
  CHECK(has(refusal([&] { run(db, "connection_weights", Json::object(), {{"fiber_counts", dense({1}, 1, 1)}, {"faces", faces}}); }), "COO"));
  CHECK(has(refusal([&] { g::step("partition_counts").plan(g::Values{}, Json::object(), g::PlanContext{db}); }), "missing"));

  std::filesystem::remove_all(root);
  return nxrtest::finish("connectome_nodes");
}

// nxr_graph — `compute`'s types (spec docs/specs/2026-11-17-type-system.md, section 10, step T4): arrays get anonymous types from their shape and the caller's labels, a node or a graph
// runs on them through the same checker as `run`, a partition is an integer array, a tangent array needs its gauge, a sidecar returns a type exactly. Synthetic arrays; no io, no file.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

#include "nxr/graph/compute.h"

namespace g = nxr::graph;
using g::Json;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

bool refused(const std::function<void()>& f, const std::vector<std::string>& parts) {
  try { f(); } catch (const std::invalid_argument& e) {
    bool ok = true;
    for (const auto& part : parts) if (std::string(e.what()).find(part) == std::string::npos) { ok = false; std::printf("  refused, but with: %s\n  (wanted '%s')\n", e.what(), part.c_str()); }
    return ok;
  }
  std::printf("  not refused (wanted '%s')\n", parts.empty() ? "" : parts[0].c_str());
  return false;
}

g::Block shaped(const g::Block& in, std::int64_t keep, std::int64_t rows, bool cplx = false) {
  g::Block o;
  o.rows = rows; o.left = std::min(keep, in.left); o.right = std::min(keep, in.right);
  o.width = o.left + in.own() + o.right;
  o.re.assign(static_cast<std::size_t>(o.rows * o.width), 0.f);
  if (cplx) o.im = o.re;
  return o;
}
g::NodeTraits pointwise() { return g::node_traits(g::TimeBehaviour::pointwise, g::Precision::float32, g::OutputPrecision::follows_input, "each column on its own"); }

// three toys carrying the built-in signatures of `analytic`, `field.form` (power) and `apply_kernel` (the signature is the type rule; the node only computes)
g::Registry toys() {
  g::Registry r;
  g::NodeSpec a; a.name = "analytic"; a.kind = g::Kind::chunk; a.traits = pointwise(); a.inputs = {{"x", ""}}; a.outputs = {{"y", ""}};
  a.parameters = {{"band", Json(), "", "Hz", {}, true}};
  a.chunk = [](const std::any&, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const g::Block& x = in.at("x"); g::Block y = shaped(x, keep, x.rows, true);
    for (std::int64_t i = 0; i < y.rows * y.width; ++i) { y.re[static_cast<std::size_t>(i)] = x.re[static_cast<std::size_t>(i)]; y.im[static_cast<std::size_t>(i)] = 2 * x.re[static_cast<std::size_t>(i)]; }
    return g::Blocks{{"y", y}};
  };
  r.add(a);
  g::NodeSpec p; p.name = "field.form"; p.parameters = {{"form", Json("power"), "", "", {}, false}}; p.kind = g::Kind::chunk; p.traits = pointwise(); p.inputs = {{"x", ""}}; p.outputs = {{"y", ""}};
  p.chunk = [](const std::any&, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const g::Block& x = in.at("x"); g::Block y = shaped(x, keep, x.rows);
    for (std::int64_t i = 0; i < y.rows * y.width; ++i) { const float a = x.re[static_cast<std::size_t>(i)], b = x.im[static_cast<std::size_t>(i)]; y.re[static_cast<std::size_t>(i)] = a * a + b * b; }
    return g::Blocks{{"y", y}};
  };
  r.add(p);
  g::NodeSpec k; k.name = "apply_kernel"; k.kind = g::Kind::chunk; k.traits = pointwise(); k.linear_map = true;
  k.inputs = {{"x", ""}, {"kernel", ""}, {"tiles", "", true}, {"pre_operator", "", true}}; k.outputs = {{"y", ""}};
  k.prepare = [](const g::Types&, const g::Arrays& c, const Json&, g::Shared&) { return std::any(c.at("kernel")); };
  k.chunk = [](const std::any& prepared, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const g::Array& K = std::any_cast<const g::Array&>(prepared);
    const g::Block& x = in.at("x"); g::Block y = shaped(x, keep, K.rows);
    for (std::int64_t r = 0; r < K.rows; ++r)
      for (std::int64_t j = 0; j < y.width; ++j) {
        double s = 0;
        for (std::int64_t c = 0; c < K.cols; ++c) s += K.data[static_cast<std::size_t>(r * K.cols + c)] * x.re[static_cast<std::size_t>(c * x.width + x.left - y.left + j)];
        y.re[static_cast<std::size_t>(r * y.width + j)] = static_cast<float>(s);
      }
    return g::Blocks{{"y", y}};
  };
  r.add(k);
  CHECK(r.unmatched_signatures().empty());
  return r;
}

g::Array mat(std::int64_t rows, std::int64_t cols, double base = 1.0) {
  g::Array a; a.rows = rows; a.cols = cols; a.data.resize(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < a.data.size(); ++i) a.data[i] = base + static_cast<double>(i % 5);
  return a;
}
g::ComputeInput field_in(std::int64_t rows, std::int64_t cols, std::vector<std::string> on, std::optional<double> sfreq = 100.0) {
  g::ComputeInput i; i.array = mat(rows, cols); i.meta.on = std::move(on); i.meta.sfreq = sfreq;
  return i;
}
g::ComputeInput kernel_in(std::int64_t rows, std::int64_t cols, const std::string& from, const std::string& to) {
  g::ComputeInput i; i.array = mat(rows, cols, 0.5); i.meta.cls = "operator"; i.meta.from = from; i.meta.to = to;
  return i;
}

void test_labels() {
  const g::Registry reg = toys();
  // arrays that share a label lie on one manifold: the kernel reads x's channels, and the output is on the kernel's target, a time line the same as x's
  auto out = g::compute_node("apply_kernel", {{"x", field_in(4, 20, {"sensors", "time"})}, {"kernel", kernel_in(3, 4, "sensors", "cortex")}}, Json::object(), reg);
  CHECK(out.arrays.at("y").rows == 3 && out.arrays.at("y").cols == 20);
  const std::string printed = g::print(out.types.at("y"));
  CHECK(printed.find("field<") == 0 && printed.find("#") != std::string::npos);
  // the numbers are the node's: y[0][0] = sum_c K[0][c] x[c][0]
  double want = 0; for (int c = 0; c < 4; ++c) want += (0.5 + (c % 5)) * (1.0 + ((c * 20) % 5));
  CHECK(std::fabs(out.arrays.at("y").data[0] - want) < 1e-4);
  // unlabelled: the kernel's columns are not x's rows (two anonymous manifolds): refused before any data, naming the port and both types
  CHECK(refused([&] { g::compute_node("apply_kernel", {{"x", field_in(4, 20, {})}, {"kernel", kernel_in(3, 4, "", "")}}, Json::object(), reg); }, {"port `kernel`", "#"}));
  // a label met at two sizes: refused naming it
  CHECK(refused([&] { g::compute_node("apply_kernel", {{"x", field_in(4, 20, {"sensors", "time"})}, {"kernel", kernel_in(3, 5, "sensors", "cortex")}}, Json::object(), reg); },
                {"label 'sensors'", "4 elements", "5"}));
  // a node that is not a node, an input that is not a port, a port not given
  CHECK(refused([&] { g::compute_node("nothing.here", {{"x", field_in(4, 20, {})}}, Json::object(), reg); }, {"nothing.here"}));
  CHECK(refused([&] { g::compute_node("field.form", {{"z", field_in(4, 20, {})}}, Json{{"form", "power"}}, reg); }, {"no input port 'z'"}));
  CHECK(refused([&] { g::compute_node("apply_kernel", {{"x", field_in(4, 20, {})}}, Json::object(), reg); }, {"required input port 'kernel'"}));
}

void test_chain() {
  const g::Registry reg = toys();
  // a chain without labels: analytic -> power, both the variable M: no label needed
  const g::Graph chain = g::graph_from_json(Json::parse(R"({"name":"chain","inputs":["x"],"outputs":["p"],
      "chunks":[{"node":"analytic","in":{"x":"x"},"out":"z"},{"node":"field.form","with":{"form":"power"},"in":{"x":"z"},"out":"p"}]})"));
  auto r = g::compute(chain, {{"x", field_in(2, 16, {})}}, Json::object(), reg);
  CHECK(r.arrays.at("p").rows == 2 && std::fabs(r.arrays.at("p").data[0] - 5.0) < 1e-5);     // x = 1: |1 + 2i|^2
  CHECK(g::print(r.types.at("p")).find("power") != std::string::npos);
  // the real map before the analytic signal is refused: power of a real field
  const g::Graph bad = g::graph_from_json(Json::parse(R"({"name":"bad","inputs":["x"],"outputs":["p"],"chunks":[{"node":"field.form","with":{"form":"power"},"in":{"x":"x"},"out":"p"}]})"));
  CHECK(refused([&] { g::compute(bad, {{"x", field_in(2, 16, {})}}, Json::object(), reg); }, {"chunks[0] field.form", "port `x`", "is real", "analytic"}));
  // D106 from arrays: a linear map after a non-linear node
  const g::Graph d106 = g::graph_from_json(Json::parse(R"({"name":"d106","inputs":["x","K"],"outputs":["y"],
      "chunks":[{"node":"analytic","in":{"x":"x"},"out":"z"},{"node":"field.form","with":{"form":"power"},"in":{"x":"z"},"out":"p"},{"node":"apply_kernel","in":{"x":"p","kernel":"K"},"out":"y"}]})"));
  CHECK(refused([&] { g::compute(d106, {{"x", field_in(4, 16, {"sensors", "time"})}, {"K", kernel_in(3, 4, "sensors", "cortex")}}, Json::object(), reg); }, {"apply_kernel", "non-linear", "D106"}));
  // the same chain checks without running when only the check is asked for
  const g::Checked ck = g::check_arrays(chain, {{"x", field_in(2, 16, {})}}, Json::object(), reg);
  CHECK(ck.values.at("z").value == g::ValueType::complex && ck.values.at("p").nonlinear);
}

void test_partition_gauge_sidecar() {
  g::ArrayTyper typer;
  g::ArrayMeta pm; pm.cls = "partition"; pm.on = {"time"};
  g::Array members; members.rows = 10; members.cols = 1; members.data = {0, 0, 1, 1, -1, 2, 2, 2, 1, -1};
  const g::PortType pt = typer.type_of("cells", members, pm);
  CHECK(pt.cls == g::PortClass::selection && pt.sel && pt.sel->partition && pt.sel->type == "set" && g::print(pt).find("partition") != std::string::npos);
  members.data[3] = 1.5;
  CHECK(refused([&] { typer.type_of("cells", members, pm); }, {"integer member index", "-1"}));
  members.data[3] = -2;
  CHECK(refused([&] { typer.type_of("cells", members, pm); }, {"integer member index"}));
  // G5: a complex tangent array needs its gauge; a vector3 value needs none
  g::ArrayMeta tm; tm.value = "complex"; tm.spin = 1; tm.on = {"cortex"};
  g::Array tangent = mat(6, 1);
  CHECK(refused([&] { typer.type_of("flow", tangent, tm); }, {"tangent array", "gauge"}));
  tm.gauge = "geometry-central@8579865/vertex:cortex:abc";
  CHECK(typer.type_of("flow", tangent, tm).gauge == tm.gauge);
  g::ArrayMeta vm; vm.value = "vector3"; vm.spin = 1;
  CHECK(typer.type_of("v", mat(6, 1, 1), vm).value == g::ValueType::vector3);
  // an unknown class, a quantity outside the closed vocabulary
  g::ArrayMeta bad; bad.cls = "table";
  CHECK(refused([&] { typer.type_of("t", tangent, bad); }, {"class 'table'"}));
  g::ArrayMeta q; q.quantity = "banana";
  CHECK(refused([&] { typer.type_of("t", tangent, q); }, {"closed vocabulary"}));

  // the sidecar: printed types, read back; two files' `#1` are never one manifold
  const g::Registry reg = toys();
  auto a = g::compute_node("analytic", {{"x", field_in(4, 20, {"sensors", "time"})}}, Json::object(), reg);
  auto b = g::compute_node("analytic", {{"x", field_in(4, 20, {"sensors", "time"})}}, Json::object(), reg);
  CHECK(a.sidecar["types"].contains("y") && a.sidecar["nxr_types"] == 1);
  g::PortType ta = g::type_from_sidecar(a.sidecar, "y");
  g::PortType ta2 = g::type_from_sidecar(a.sidecar, "y");
  CHECK(g::same(ta.domain, ta2.domain) && ta.value == g::ValueType::complex && ta.typed);
  Json other = b.sidecar; other["run"] = "other-run";
  CHECK(!g::same(ta.domain, g::type_from_sidecar(other, "y").domain));
  // an array read back with its type runs through the checker as typed: power accepts it, the labels are not asked for
  g::ComputeInput back; back.array = a.arrays.at("y"); back.type = ta;
  auto p = g::compute_node("field.form", {{"x", back}}, Json{{"form", "power"}}, reg);
  CHECK(p.arrays.at("y").rows == 4);
  CHECK(refused([&] { g::type_from_sidecar(a.sidecar, "nope"); }, {"no type for 'nope'"}));
}

}  // namespace

int main() {
  test_labels();
  test_chain();
  test_partition_gauge_sidecar();
  if (failures) { std::printf("test_compute_types: %d failure(s)\n", failures); return 1; }
  std::printf("test_compute_types: ok\n");
  return 0;
}

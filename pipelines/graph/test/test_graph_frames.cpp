// nxr_graph — the gauge in the executor (G5; spec docs/specs/2026-11-17-geometry-central-switch.md, section 5): two gauges never meet, and frame.rotate, frame.embed and
// frame.project as pure nodes, checked on types and run on blocks against closed forms.
#include <cmath>
#include <complex>
#include <memory>
#include <cstdio>
#include <functional>
#include <string>

#include "nxr/graph/frame_nodes.h"

namespace g = nxr::graph;
using g::Json;
using VT = g::ValueType;

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

const auto time = g::time_line("rest", 100.0);
const auto cortex = g::manifold_id("cortex", "surf", g::ManifoldKind::surface);

g::Domain dom(std::vector<g::ManifoldRef> f) { g::Domain d; d.factors = std::move(f); return d; }
g::PortType field_t(std::vector<g::ManifoldRef> f, VT v, int spin = 0, const std::string& gauge = "", std::int64_t rows = 3) {
  g::PortType t;
  const bool none = f.size() == 1;
  t.typed = true; t.cls = g::PortClass::field; t.domain = dom(std::move(f)); t.value = v; t.quantity = "current"; t.rows = rows;
  t.axis = none ? g::Axis::none : g::Axis::samples; t.sfreq = 100.0;
  t.spin = spin; t.gauge = gauge;
  return t;
}
g::PortType map_t(VT v, int spin = 0, const std::string& gauge = "") {
  g::PortType t = field_t({cortex}, v, spin, gauge);
  t.axis = g::Axis::none; t.sfreq.reset();
  return t;
}

g::Types use(const std::string& node, const g::Types& in, const Json& params = Json::object()) {
  const g::Signature* s = g::builtin_signature(node);
  if (!s) { std::printf("  no signature for %s\n", node.c_str()); ++failures; return {}; }
  g::TypeUse tu;
  tu.label = "chunks[0] " + node;
  tu.chunk_node = true;
  tu.in = in;
  for (const auto& [port, _] : in) tu.in_names[port] = port;
  tu.out_names["y"] = "y";
  return g::type_use(node, *s, tu, params);
}

void test_types() {
  const std::string gA = "gc@8579865/vertex:surf:aaaa0001", gB = "gc@8579865/vertex:surf:bbbb0002";
  const auto x = field_t({cortex, time}, VT::complex, 1, gA);
  const auto to = map_t(VT::real);
  const auto y = use("frame.rotate", {{"x", x}, {"to", to}}).at("y");
  CHECK(y.spin == 1 && y.gauge == "frame(to)@" + gA);
  CHECK(refused([&] { use("frame.rotate", {{"x", field_t({cortex, time}, VT::complex)}, {"to", to}}); }, {"has no spin", "`spin`"}));
  CHECK(use("frame.rotate", {{"x", field_t({cortex, time}, VT::complex, 0, "")}, {"to", to}}, Json{{"spin", 2}}).at("y").spin == 2);
  CHECK(refused([&] { use("frame.rotate", {{"x", x}, {"to", map_t(VT::real, 1, gA)}}); }, {"angle is a scalar"}));
  // an embedded value is gauge-free; its projection is in the grid's gauge
  const auto grid = map_t(VT::complex, 1, gA);
  const auto v = use("frame.embed", {{"x", x}, {"grid", grid}}).at("y");
  CHECK(v.value == VT::vector3 && v.spin == 0 && v.gauge.empty());
  const auto z = use("frame.project", {{"x", field_t({cortex, time}, VT::vector3)}, {"grid", grid}}).at("y");
  CHECK(z.value == VT::complex && z.spin == 1 && z.gauge == gA);
  // two gauges meet: refused, naming both and frame.rotate
  CHECK(refused([&] { use("frame.embed", {{"x", x}, {"grid", map_t(VT::complex, 1, gB)}}); }, {"two gauges never meet", gA, gB, "frame.rotate"}));
  CHECK(refused([&] { use("frame.embed", {{"x", field_t({cortex, time}, VT::complex, 2, gA)}, {"grid", grid}}); }, {"is spin 2", "values of different spin"}));
  CHECK(refused([&] { use("frame.embed", {{"x", field_t({cortex, time}, VT::complex, 0, "")}, {"grid", grid}}); }, {"only a tangent vector (spin 1)"}));
  // a field that predates the gauge record is refused
  CHECK(refused([&] { use("frame.embed", {{"x", field_t({cortex, time}, VT::complex, 1, "")}, {"grid", grid}}); }, {"records no gauge"}));
}

g::Block complex_block(std::int64_t rows, std::int64_t width) {
  g::Block b;
  b.rows = rows; b.width = width;
  b.re.resize(static_cast<std::size_t>(rows * width)); b.im.resize(b.re.size());
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t j = 0; j < width; ++j) { b.re[static_cast<std::size_t>(r * width + j)] = 1.f + float(r) + 0.5f * float(j); b.im[static_cast<std::size_t>(r * width + j)] = -2.f + 0.25f * float(r * j); }
  return b;
}

void test_run() {
  g::Registry r;
  g::register_frame_nodes(r);
  CHECK(r.unmatched_signatures().empty());
  CHECK(r.at("frame.rotate").signature.has_value());
  const std::int64_t V = 4, W = 5;
  // a grid: e1, e2 are an orthonormal pair in the plane z = const rotated by a per-vertex angle about z; the normal is z
  g::Array grid; grid.rows = V; grid.cols = 3; grid.comps = 2; grid.data.assign(static_cast<std::size_t>(V * 3 * 2), 0.0);
  for (std::int64_t v = 0; v < V; ++v) {
    const double a = 0.3 + 0.7 * double(v);
    const double e1[3] = {std::cos(a), std::sin(a), 0}, e2[3] = {-std::sin(a), std::cos(a), 0};
    for (int c = 0; c < 3; ++c) { grid.data[static_cast<std::size_t>((v * 3 + c) * 2)] = e1[c]; grid.data[static_cast<std::size_t>((v * 3 + c) * 2 + 1)] = e2[c]; }
  }
  g::Block x = complex_block(V, W);
  // frame.rotate: y = x e^(-i k to), to the double angle for k = 2
  g::Array to; to.rows = V; to.cols = 1; to.data = {0.0, 0.5, -1.0, 2.0};
  for (int k : {1, 2}) {
    g::PortType xt = field_t({cortex, time}, VT::complex, k, "gA");
    const auto& n = r.at("frame.rotate");
    const auto prepared = n.prepare({{"x", xt}, {"to", map_t(VT::real)}}, {{"to", to}}, Json{{"spin", 0}}, *std::make_unique<g::Shared>());
    const auto out = n.chunk(prepared, {{"x", x}}, 0, g::Place{}).at("y");
    double worst = 0;
    for (std::int64_t v = 0; v < V; ++v)
      for (std::int64_t j = 0; j < W; ++j) {
        const std::complex<double> z(x.re[static_cast<std::size_t>(v * W + j)], x.im[static_cast<std::size_t>(v * W + j)]);
        const std::complex<double> w = z * std::polar(1.0, -double(k) * to.data[static_cast<std::size_t>(v)]);
        worst = std::max(worst, std::abs(w - std::complex<double>(out.re[static_cast<std::size_t>(v * W + j)], out.im[static_cast<std::size_t>(v * W + j)])));
      }
    CHECK(worst < 1e-5);
  }
  // embed then project is the identity on the tangent plane; the embedded vector is tangent (z component 0) and has the length |z|
  g::Shared sh;
  const g::PortType xt = field_t({cortex, time}, VT::complex, 1, "gA");
  const auto emb = r.at("frame.embed"), proj = r.at("frame.project");
  const auto pe = emb.prepare({{"x", xt}, {"grid", map_t(VT::complex, 1, "gA")}}, {{"grid", grid}}, Json::object(), sh);
  const g::Block ev = emb.chunk(pe, {{"x", x}}, 0, g::Place{}).at("y");
  CHECK(ev.rows == 3 * V && ev.width == W && !ev.complex());
  double tangent = 0, length = 0;
  for (std::int64_t v = 0; v < V; ++v)
    for (std::int64_t j = 0; j < W; ++j) {
      tangent = std::max(tangent, std::abs(double(ev.re[static_cast<std::size_t>((3 * v + 2) * W + j)])));
      const double a = ev.re[static_cast<std::size_t>(3 * v * W + j)], b = ev.re[static_cast<std::size_t>((3 * v + 1) * W + j)];
      const double zr = x.re[static_cast<std::size_t>(v * W + j)], zi = x.im[static_cast<std::size_t>(v * W + j)];
      length = std::max(length, std::abs(std::hypot(a, b) - std::hypot(zr, zi)));
    }
  CHECK(tangent < 1e-6 && length < 1e-5);
  const auto pp = proj.prepare({{"x", field_t({cortex, time}, VT::vector3, 0, "", 3 * V)}, {"grid", map_t(VT::complex, 1, "gA")}}, {{"grid", grid}}, Json::object(), sh);
  const g::Block back = proj.chunk(pp, {{"x", ev}}, 0, g::Place{}).at("y");
  double worst = 0;
  for (std::size_t i = 0; i < x.re.size(); ++i) worst = std::max({worst, std::abs(double(back.re[i] - x.re[i])), std::abs(double(back.im[i] - x.im[i]))});
  CHECK(back.rows == V && worst < 1e-5);
  // the plan: the shapes it gives and the refusal of a wrong grid
  CHECK(refused([&] { { auto gt = map_t(VT::complex, 1, "gA"); gt.rows = 5; emb.plan({{"x", xt}, {"grid", gt}}, Json::object()); }; }, {"`grid` has 5 rows"}));
  g::Array bad = grid; bad.cols = 2;
  CHECK(refused([&] { emb.prepare({{"x", xt}, {"grid", map_t(VT::complex, 1, "gA")}}, {{"grid", bad}}, Json::object(), sh); }, {"vertexGrid"}));
}

}  // namespace

int main() {
  test_types();
  test_run();
  if (failures) { std::printf("%d check(s) failed\n", failures); return 1; }
  std::printf("graph frames: ok\n");
  return 0;
}

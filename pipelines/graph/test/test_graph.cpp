// nxr_graph — the executor alone, on synthetic arrays and toy nodes (spec docs/specs/2026-11-17-compute-and-run.md, section 9). No io, no file, no subject.
//   1. check refuses each bad arrow naming the use, the port and the reason;
//   2. the halo is composed from the nodes' traits (the sum rule of the chain, the max rule of a branch);
//   3. the result does not depend on the block size: `apply` on the whole array, the same graph over blocks of several widths and sub-block widths, and an
//      independent reference computed on the whole array all agree;
//   4. state is carried across blocks (a running sum, an accumulate node), blocks out of time order are refused, and a stateful node is refused where blocks may be parallel;
//   5. a retiming node: the nodes after it run on frames, in blocks of frames, with the same result.
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>

#include "nxr/graph/execution.h"

namespace g = nxr::graph;
using g::Json;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

bool refused(const std::function<void()>& f, const std::string& part) {
  try { f(); } catch (const std::invalid_argument& e) {
    if (std::string(e.what()).find(part) != std::string::npos) return true;
    std::printf("  refused, but with: %s (wanted '%s')\n", e.what(), part.c_str());
    return false;
  }
  std::printf("  not refused (wanted '%s')\n", part.c_str());
  return false;
}

// ── toy nodes ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
g::PortType stream_of(std::int64_t rows, g::ValueType v = g::ValueType::real) {
  g::PortType t; t.value = v; t.rows = rows; t.axis = g::Axis::samples; t.sfreq = 100.0; t.unit = "T"; return t;
}
g::PortType array_of(std::int64_t rows) { g::PortType t; t.cls = g::PortClass::constant; t.rows = rows; t.unit = "T"; return t; }

// the block a node returns: its own columns and min(keep, what the input held) of halo on each side (the contract of g::Planned / NodeSpec::chunk)
g::Block shaped(const g::Block& in, std::int64_t keep) {
  g::Block o;
  o.rows = in.rows; o.left = std::min(keep, in.left); o.right = std::min(keep, in.right);
  o.width = o.left + in.own() + o.right;
  o.re.assign(static_cast<std::size_t>(o.rows * o.width), 0.f);
  return o;
}
float& at(g::Block& b, std::int64_t r, std::int64_t j) { return b.re[static_cast<std::size_t>(r * b.width + j)]; }
float at(const g::Block& b, std::int64_t r, std::int64_t j) { return b.re[static_cast<std::size_t>(r * b.width + j)]; }

// a centred mean of 2r + 1 columns, an edge repeated: the one function the node and the reference both use (the blocks' halo bookkeeping is what the tests exercise)
float smooth_at(const float* row, std::int64_t n, std::int64_t p, std::int64_t r) {
  double s = 0;
  for (std::int64_t d = -r; d <= r; ++d) s += row[std::clamp<std::int64_t>(p + d, 0, n - 1)];
  return static_cast<float>(s / static_cast<double>(2 * r + 1));
}

g::NodeTraits pointwise() { return g::node_traits(g::TimeBehaviour::pointwise, g::Precision::float32, g::OutputPrecision::follows_input, "each column on its own"); }

g::NodeSpec scale_node() {
  g::NodeSpec s;
  s.name = "toy.scale"; s.kind = g::Kind::chunk; s.traits = pointwise();
  s.inputs = {{"x", "signal"}}; s.outputs = {{"y", "scaled"}};
  s.parameters = {{"a", 1.0, "the gain", "", {}}};
  s.plan = [](const g::Types& in, const Json&) {
    if (in.at("x").value != g::ValueType::real) g::refuse("x must be real, it is " + std::string(g::to_string(in.at("x").value)));
    return g::Planned{{{"y", in.at("x")}}, 0};
  };
  s.prepare = [](const g::Types&, const g::Arrays&, const Json& p, g::Shared&) { return std::any(static_cast<float>(p.at("a").get<double>())); };
  s.chunk = [](const std::any& prepared, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const float a = std::any_cast<float>(prepared);
    const g::Block& x = in.at("x");
    g::Block y = shaped(x, keep);
    for (std::int64_t r = 0; r < y.rows; ++r)
      for (std::int64_t j = 0; j < y.width; ++j) at(y, r, j) = a * at(x, r, x.left - y.left + j);
    return g::Blocks{{"y", std::move(y)}};
  };
  return s;
}
g::NodeSpec smooth_node() {
  g::NodeSpec s;
  s.name = "toy.smooth"; s.kind = g::Kind::chunk;
  s.traits = g::node_traits(g::TimeBehaviour::windowed, g::Precision::float64, g::OutputPrecision::follows_input, "a centred mean", "radius samples each side");
  s.inputs = {{"x", "signal"}}; s.outputs = {{"y", "smoothed"}};
  s.parameters = {{"radius", 1, "half the window", "samples", {}}};
  s.plan = [](const g::Types& in, const Json& p) { return g::Planned{{{"y", in.at("x")}}, p.at("radius").get<std::int64_t>()}; };
  s.prepare = [](const g::Types&, const g::Arrays&, const Json& p, g::Shared&) { return std::any(p.at("radius").get<std::int64_t>()); };
  s.chunk = [](const std::any& prepared, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const std::int64_t r = std::any_cast<std::int64_t>(prepared);
    const g::Block& x = in.at("x");
    g::Block y = shaped(x, keep);
    for (std::int64_t row = 0; row < y.rows; ++row)
      for (std::int64_t j = 0; j < y.width; ++j) at(y, row, j) = smooth_at(&x.re[static_cast<std::size_t>(row * x.width)], x.width, x.left - y.left + j, r);
    return g::Blocks{{"y", std::move(y)}};
  };
  return s;
}
g::NodeSpec add_const_node() {   // a streamed input and a constant one (read once, in prepare)
  g::NodeSpec s;
  s.name = "toy.add_bias"; s.kind = g::Kind::chunk; s.traits = pointwise();
  s.inputs = {{"x", "signal"}, {"bias", "one value per row"}}; s.outputs = {{"y", "x plus bias"}};
  s.plan = [](const g::Types& in, const Json&) {
    if (in.at("bias").rows != in.at("x").rows) g::refuse("bias has " + std::to_string(in.at("bias").rows) + " rows, x has " + std::to_string(in.at("x").rows));
    return g::Planned{{{"y", in.at("x")}}, 0};
  };
  s.prepare = [](const g::Types&, const g::Arrays& c, const Json&, g::Shared&) { return std::any(c.at("bias").data); };
  s.chunk = [](const std::any& prepared, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const auto& bias = std::any_cast<const std::vector<double>&>(prepared);
    const g::Block& x = in.at("x");
    g::Block y = shaped(x, keep);
    for (std::int64_t r = 0; r < y.rows; ++r)
      for (std::int64_t j = 0; j < y.width; ++j) at(y, r, j) = at(x, r, x.left - y.left + j) + static_cast<float>(bias[static_cast<std::size_t>(r)]);
    return g::Blocks{{"y", std::move(y)}};
  };
  s.record = [](const std::any& prepared) { return Json{{"bias_rows", static_cast<std::int64_t>(std::any_cast<const std::vector<double>&>(prepared).size())}}; };
  return s;
}
g::NodeSpec square_node() {   // non-linear
  g::NodeSpec s = scale_node();
  s.name = "toy.square"; s.parameters.clear();
  s.plan = [](const g::Types& in, const Json&) { g::PortType t = in.at("x"); t.nonlinear = true; return g::Planned{{{"y", t}}, 0}; };
  s.prepare = nullptr;
  s.chunk = [](const std::any&, const g::Blocks& in, std::int64_t keep, const g::Place&) {
    const g::Block& x = in.at("x");
    g::Block y = shaped(x, keep);
    for (std::int64_t r = 0; r < y.rows; ++r)
      for (std::int64_t j = 0; j < y.width; ++j) { const float v = at(x, r, x.left - y.left + j); at(y, r, j) = v * v; }
    return g::Blocks{{"y", std::move(y)}};
  };
  return s;
}
g::NodeSpec linear_node() { g::NodeSpec s = scale_node(); s.name = "toy.linear"; s.linear_map = true; return s; }
g::NodeSpec cumsum_node() {   // STATEFUL: a running sum over time, per row; no halo of its own to give (its output is its own columns)
  g::NodeSpec s;
  s.name = "toy.cumsum"; s.kind = g::Kind::chunk;
  s.traits = g::node_traits(g::TimeBehaviour::stateful, g::Precision::float64, g::OutputPrecision::follows_input, "a running sum carries the sum forward");
  s.inputs = {{"x", "signal"}}; s.outputs = {{"y", "running sum"}};
  s.plan = [](const g::Types& in, const Json&) { return g::Planned{{{"y", in.at("x")}}, 0}; };
  s.start = [](const std::any&) { return std::any(std::vector<double>{}); };
  s.step = [](const std::any&, std::any state, const g::Blocks& in, std::int64_t, const g::Place&) {
    auto carry = std::any_cast<std::vector<double>>(std::move(state));
    const g::Block& x = in.at("x");
    carry.resize(static_cast<std::size_t>(x.rows), 0.0);
    g::Block y; y.rows = x.rows; y.width = x.own(); y.re.resize(static_cast<std::size_t>(y.rows * y.width));
    for (std::int64_t r = 0; r < x.rows; ++r)
      for (std::int64_t j = 0; j < y.width; ++j) { carry[static_cast<std::size_t>(r)] += at(x, r, x.left + j); at(y, r, j) = static_cast<float>(carry[static_cast<std::size_t>(r)]); }
    return std::make_pair(g::Blocks{{"y", std::move(y)}}, std::any(std::move(carry)));
  };
  return s;
}
struct Sums { std::vector<double> sum; std::int64_t n = 0; };
g::NodeSpec power_sum_node() {   // ACCUMULATE: the sum of squares and the count over the time partition handed (a reduction), finished as arrays
  g::NodeSpec s;
  s.name = "toy.power_sum"; s.kind = g::Kind::accumulate;
  s.traits = g::node_traits(g::TimeBehaviour::stateful, g::Precision::float64, g::OutputPrecision::float64, "sums in order");
  s.inputs = {{"x", "signal"}}; s.outputs = {{"total", "sum of squares per row"}, {"count", "samples folded"}};
  s.plan = [](const g::Types& in, const Json&) { return g::Planned{{{"total", array_of(in.at("x").rows)}, {"count", array_of(1)}}, 0}; };
  s.start = [](const std::any&) { return std::any(Sums{}); };
  s.step = [](const std::any&, std::any state, const g::Blocks& in, std::int64_t, const g::Place&) {
    auto a = std::any_cast<Sums>(std::move(state));
    const g::Block& x = in.at("x");
    a.sum.resize(static_cast<std::size_t>(x.rows), 0.0);
    for (std::int64_t r = 0; r < x.rows; ++r)
      for (std::int64_t j = 0; j < x.own(); ++j) a.sum[static_cast<std::size_t>(r)] += static_cast<double>(at(x, r, x.left + j)) * at(x, r, x.left + j);
    a.n += x.own();
    return std::make_pair(g::Blocks{}, std::any(std::move(a)));
  };
  s.finish = [](const std::any&, std::any state) {
    auto a = std::any_cast<Sums>(std::move(state));
    g::Arrays out;
    out["total"] = g::Array{static_cast<std::int64_t>(a.sum.size()), 1, a.sum, Json::object(), 1};
    out["count"] = g::Array{1, 1, {static_cast<double>(a.n)}, Json::object(), 1};
    return out;
  };
  return s;
}
g::NodeSpec mean_node() {   // ARRAY node, in `after`
  g::NodeSpec s;
  s.name = "toy.mean"; s.kind = g::Kind::array;
  s.traits = g::node_traits(g::TimeBehaviour::pointwise, g::Precision::float64, g::OutputPrecision::float64, "total over count");
  s.inputs = {{"total", ""}, {"count", ""}}; s.outputs = {{"mean", ""}};
  s.plan = [](const g::Types& in, const Json&) { return g::Planned{{{"mean", in.at("total")}}, 0}; };
  s.run = [](const std::any&, const g::Arrays& in) {
    g::Array m = in.at("total");
    for (double& v : m.data) v /= in.at("count").data.at(0);
    return g::Arrays{{"mean", m}};
  };
  return s;
}
g::NodeSpec twice_node() {   // ARRAY node, in `before`
  g::NodeSpec s = mean_node();
  s.name = "toy.twice"; s.inputs = {{"k", ""}}; s.outputs = {{"k2", ""}};
  s.plan = [](const g::Types& in, const Json&) { return g::Planned{{{"k2", in.at("k")}}, 0}; };
  s.run = [](const std::any&, const g::Arrays& in) { g::Array m = in.at("k"); for (double& v : m.data) v *= 2; return g::Arrays{{"k2", m}}; };
  return s;
}
constexpr std::int64_t kF = 8;   // the frame length of the retiming node
g::NodeSpec retime_node() {      // a RETIMING node: the mean of kF consecutive samples is a frame; the block's columns are whole frames
  g::NodeSpec s;
  s.name = "toy.frames"; s.kind = g::Kind::chunk; s.traits = pointwise();
  s.inputs = {{"x", ""}}; s.outputs = {{"f", ""}};
  s.plan = [](const g::Types& in, const Json&) { g::PortType t = in.at("x"); t.axis = g::Axis::frames; t.sfreq = *t.sfreq / kF; return g::Planned{{{"f", t}}, 0}; };
  s.chunk = [](const std::any&, const g::Blocks& in, std::int64_t, const g::Place& at_) {
    const g::Block& x = in.at("x");
    if (at_.first % kF || at_.n % kF) g::refuse("toy.frames: the block must hold whole frames");
    g::Block f; f.rows = x.rows; f.width = at_.n / kF; f.re.resize(static_cast<std::size_t>(f.rows * f.width));
    for (std::int64_t r = 0; r < x.rows; ++r)
      for (std::int64_t j = 0; j < f.width; ++j) { double s2 = 0; for (std::int64_t k = 0; k < kF; ++k) s2 += at(x, r, x.left + j * kF + k); at(f, r, j) = static_cast<float>(s2 / kF); }
    return g::Blocks{{"f", std::move(f)}};
  };
  s.frame_place = [](const std::any&, const g::Place& p) { g::FramePlace fp; fp.first = p.first / kF; fp.n = p.n / kF; return fp; };
  return s;
}

g::Registry toys() {
  g::Registry r;
  for (auto n : {scale_node(), smooth_node(), add_const_node(), square_node(), linear_node(), cumsum_node(), power_sum_node(), mean_node(), twice_node(), retime_node()}) r.add(std::move(n));
  g::NodeSpec bad = scale_node(); bad.name = "toy.undeclared"; bad.traits = {}; r.add(bad);
  return r;
}

g::Use use(const std::string& node, std::map<std::string, std::string> in, std::vector<std::string> out, Json with = Json::object()) {
  g::Use u; u.node = node; u.in = std::move(in); u.out = std::move(out); u.with = std::move(with); return u;
}

constexpr std::int64_t R = 3, N = 203;
std::vector<float> signal(std::int64_t rows, std::int64_t n) {
  std::vector<float> x(static_cast<std::size_t>(rows * n));
  for (std::int64_t r = 0; r < rows; ++r) for (std::int64_t t = 0; t < n; ++t) x[static_cast<std::size_t>(r * n + t)] = static_cast<float>(std::sin(0.13 * (r + 1) * t) + 0.01 * t - 0.5 * r);
  return x;
}
g::Array array_of_rows(const std::vector<float>& x, std::int64_t rows, std::int64_t n) {
  g::Array a; a.rows = rows; a.cols = n; a.data.assign(x.begin(), x.end()); return a;
}

// the caller's loop over a recording (the executor never loops): blocks of `width` own columns, each with the halo the graph asks for, clamped at the recording's ends
struct Result { std::vector<std::vector<float>> stream; g::Arrays arrays; std::string record; };
Result run_blocks(const g::Checked& c, const std::vector<float>& x, std::int64_t rows, std::int64_t n, std::int64_t width, const g::Arrays& constants = {}, const g::Blocking& blocking = {},
                  std::vector<std::int64_t> order = {}) {
  g::Execution ex(c, constants, {}, blocking);
  const std::int64_t H = ex.halo();
  std::map<std::string, std::vector<float>> acc;
  std::vector<std::int64_t> firsts = order;
  if (firsts.empty()) for (std::int64_t f = 0; f < n; f += width) firsts.push_back(f);
  Result res;
  for (const std::int64_t first : firsts) {
    const std::int64_t own = std::min(width, n - first), left = std::min(H, first), right = std::min(H, n - first - own);
    g::Block b; b.rows = rows; b.width = left + own + right; b.left = left; b.right = right;
    b.re.resize(static_cast<std::size_t>(rows * b.width));
    for (std::int64_t r = 0; r < rows; ++r) for (std::int64_t j = 0; j < b.width; ++j) b.re[static_cast<std::size_t>(r * b.width + j)] = x[static_cast<std::size_t>(r * n + first - left + j)];
    std::vector<std::uint8_t> good(static_cast<std::size_t>(own), 1);
    g::Blocks out = ex.block({{c.inputs.at(0), std::move(b)}}, g::Place{first, own, &good});
    for (auto& [name, blk] : out) {
      auto& v = acc[name];
      CHECK(blk.left == 0 && blk.right == 0);
      if (v.empty()) v.assign(static_cast<std::size_t>(blk.rows * 0), 0.f);
      // rows are kept as consecutive row-major pieces: store [row][piece] by appending per row later; here keep a flat per-block list
      for (std::int64_t r = 0; r < blk.rows; ++r) v.insert(v.end(), blk.re.begin() + r * blk.width, blk.re.begin() + (r + 1) * blk.width);
      acc[name + "#w"].push_back(static_cast<float>(blk.width));
      acc[name + "#rows"] = {static_cast<float>(blk.rows)};
    }
  }
  res.arrays = ex.finish();
  res.record = ex.record().dump();
  // reassemble [row][time] from the per-block row-major pieces
  for (const auto& name : c.outputs) {
    if (!acc.count(name)) continue;
    const std::int64_t orows = static_cast<std::int64_t>(acc[name + "#rows"][0]);
    std::vector<float> full; std::vector<std::vector<float>> by_row(static_cast<std::size_t>(orows));
    std::size_t at_ = 0;
    for (const float w : acc[name + "#w"]) for (std::int64_t r = 0; r < orows; ++r) { by_row[static_cast<std::size_t>(r)].insert(by_row[static_cast<std::size_t>(r)].end(), acc[name].begin() + at_, acc[name].begin() + at_ + static_cast<std::size_t>(w)); at_ += static_cast<std::size_t>(w); }
    for (auto& v : by_row) full.insert(full.end(), v.begin(), v.end());
    res.stream.push_back(std::move(full));
  }
  return res;
}

// ── the graphs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
g::Graph chain_graph() {
  g::Graph gr; gr.name = "chain";
  gr.inputs = {"x"};
  gr.parameters = Json{{"gain", Json{{"default", 0.5}, {"meaning", "the gain"}}}, {"wide", 2}};
  gr.chunks = {use("toy.smooth", {{"x", "x"}}, {"a"}, Json{{"radius", "$wide"}}), use("toy.scale", {{"x", "a"}}, {"b"}, Json{{"a", "$gain"}}), use("toy.smooth", {{"x", "b"}}, {"c"}, Json{{"radius", 1}}),
               use("toy.cumsum", {{"x", "c"}}, {"d"}), use("toy.power_sum", {{"x", "c"}}, {"total", "count"})};
  gr.after = {use("toy.mean", {{"total", "total"}, {"count", "count"}}, {"mean"})};
  gr.outputs = {"d", "total", "mean"};
  return gr;
}
// the same computation, whole array, plain loops
void reference_chain(const std::vector<float>& x, float gain, int wide, std::vector<float>& d, std::vector<double>& total, double& count) {
  d.assign(x.size(), 0.f); total.assign(R, 0.0);
  for (std::int64_t r = 0; r < R; ++r) {
    const float* row = &x[static_cast<std::size_t>(r * N)];
    std::vector<float> a(N), b(N), cc(N);
    for (std::int64_t t = 0; t < N; ++t) a[t] = smooth_at(row, N, t, wide);
    for (std::int64_t t = 0; t < N; ++t) b[t] = gain * a[t];
    for (std::int64_t t = 0; t < N; ++t) cc[t] = smooth_at(b.data(), N, t, 1);
    double carry = 0;
    for (std::int64_t t = 0; t < N; ++t) { carry += cc[t]; d[static_cast<std::size_t>(r * N + t)] = static_cast<float>(carry); total[r] += static_cast<double>(cc[t]) * cc[t]; }
  }
  count = N;
}

void test_check_refusals() {
  const g::Registry reg = toys();
  const g::Types in{{"x", stream_of(R)}};
  auto base = [] { g::Graph gr; gr.inputs = {"x"}; gr.outputs = {"y"}; return gr; };
  auto chk = [&](const g::Graph& gr, const g::Types& t = {{"x", stream_of(R)}}, const Json& p = Json::object(), const g::DispatchContext& ctx = {}) { return g::check(gr, t, p, reg, ctx); };

  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(chk(gr).values.at("y").axis == g::Axis::samples); }   // the good case
  { g::Graph gr = base(); gr.chunks = {use("toy.smoth", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "unknown node 'toy.smoth'")); CHECK(refused([&] { chk(gr); }, "toy.smooth")); }   // near names
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "nope"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "port 'x' reads 'nope'")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"z", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "no input port 'z'")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {}, {"y"})}; CHECK(refused([&] { chk(gr); }, "input port 'x' is not wired")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"}, Json{{"a", "$gone"}})}; CHECK(refused([&] { chk(gr); }, "$gone")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"}, Json{{"a", "loud"}})}; CHECK(refused([&] { chk(gr); }, "parameter 'a' is a string")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"}, Json{{"b", 1}})}; CHECK(refused([&] { chk(gr); }, "no parameter 'b'")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr, {{"x", stream_of(R, g::ValueType::complex)}}); }, "x must be real")); CHECK(refused([&] { chk(gr, {{"x", stream_of(R, g::ValueType::complex)}}); }, "chunks[0] toy.scale")); }   // the use and the reason
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr, {}); }, "has no type")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"x"})}; CHECK(refused([&] { chk(gr); }, "already exists")); }
  { g::Graph gr = base(); gr.outputs = {"missing"}; CHECK(refused([&] { chk(gr); }, "output 'missing'")); }
  { g::Graph gr = base(); gr.after = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "a chunk node belongs in `chunks`, not in `after`")); }    // a chunk node in after
  { g::Graph gr = base(); gr.before = {use("toy.twice", {{"k", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "takes the streamed value 'x'")); }                      // a block into a before node
  { g::Graph gr = base(); gr.chunks = {use("toy.power_sum", {{"x", "x"}}, {"t", "c"}), use("toy.twice", {{"k", "t"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "an array node belongs in `before` or `after`")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.scale", {{"x", "x"}}, {"a"})}; gr.after = {use("toy.mean", {{"total", "a"}, {"count", "a"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "an `after` node runs once")); }   // a block into after
  { g::Graph gr = base(); gr.chunks = {use("toy.undeclared", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "declares no (or incomplete) traits")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.square", {{"x", "x"}}, {"a"}), use("toy.linear", {{"x", "a"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "D106")); }                 // a linear map after a non-linear node
  { g::Graph gr = base(); gr.chunks = {use("toy.square", {{"x", "x"}}, {"a"}), use("toy.scale", {{"x", "a"}}, {"b"}), use("toy.linear", {{"x", "b"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "D106")); }   // carried forward
  { g::Graph gr = base(); gr.chunks = {use("toy.cumsum", {{"x", "x"}}, {"y"})}; g::DispatchContext par; par.block_parallel = true; CHECK(refused([&] { chk(gr, in, Json::object(), par); }, "cannot run on blocks in parallel")); }
  { g::Graph gr = base(); gr.inputs = {"x", "bias"}; gr.chunks = {use("toy.add_bias", {{"x", "x"}, {"bias", "bias"}}, {"y"})};
    CHECK(refused([&] { chk(gr, {{"x", stream_of(R)}, {"bias", array_of(R + 1)}}); }, "bias has 4 rows, x has 3")); }
  { g::Graph gr = base(); gr.parameters = Json{{"gain", nullptr}}; gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "has no default and was not set")); }
  { g::Graph gr = base(); gr.parameters = Json{{"gain", 1.0}}; gr.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr, in, Json{{"other", 1}}); }, "unknown parameter 'other'")); CHECK(refused([&] { chk(gr, in, Json{{"gain", "x"}}); }, "parameter 'gain' is a string")); }
  // a retiming node: no node after it may need a halo, and none reads the stream of samples
  { g::Graph gr = base(); gr.chunks = {use("toy.frames", {{"x", "x"}}, {"f"}), use("toy.smooth", {{"x", "f"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "after the retiming node")); }
  { g::Graph gr = base(); gr.chunks = {use("toy.frames", {{"x", "x"}}, {"f"}), use("toy.scale", {{"x", "x"}}, {"y"})}; CHECK(refused([&] { chk(gr); }, "reads the stream of samples")); }
  // the JSON form
  { Json doc = Json{{"inputs", {"x"}}, {"outputs", {"y"}}, {"chunks", {Json{{"node", "toy.scale"}, {"in", {{"x", "x"}}}, {"out", "y"}}}}};
    CHECK(chk(g::graph_from_json(doc)).nodes.size() == 1);
    doc["chunks"][0]["bogus"] = 1; CHECK(refused([&] { g::graph_from_json(doc); }, "unknown key 'bogus'")); }
}

void test_halo() {
  const g::Registry reg = toys();
  const g::Checked c = g::check(chain_graph(), {{"x", stream_of(R)}}, Json::object(), reg);
  // smooth(2) -> scale -> smooth(1) -> cumsum; power_sum reads smooth(1): keep(smooth_2) = 0 + keep(scale) ... the sum rule of the chain
  CHECK(c.loop.size() == 5);
  CHECK(c.nodes[c.loop[0]].planned.halo == 2 && c.nodes[c.loop[0]].keep == 1);   // smooth r=2 must leave smooth r=1's halo
  CHECK(c.nodes[c.loop[1]].keep == 1);                                            // scale passes it on
  CHECK(c.nodes[c.loop[2]].planned.halo == 1 && c.nodes[c.loop[2]].keep == 0);   // read only by a stateful chunk node (halo 0) and an accumulate node
  CHECK(c.halo == 3);
  CHECK(c.tail == 3);                                                             // the pointwise tail starts after the last node needing a halo
  CHECK(c.has_state);
  CHECK(c.text().find("loop halo 3") != std::string::npos);
  // a branch: the loop's halo is the max over the readers of the input (h + keep)
  g::Graph b; b.inputs = {"x"}; b.outputs = {"c", "d"};
  b.chunks = {use("toy.smooth", {{"x", "x"}}, {"a"}, Json{{"radius", 3}}), use("toy.smooth", {{"x", "x"}}, {"d"}, Json{{"radius", 1}}), use("toy.smooth", {{"x", "a"}}, {"c"}, Json{{"radius", 2}})};
  const g::Checked cb = g::check(b, {{"x", stream_of(R)}}, Json::object(), reg);
  CHECK(cb.nodes[cb.loop[0]].keep == 2 && cb.nodes[cb.loop[1]].keep == 0 && cb.nodes[cb.loop[2]].keep == 0);
  CHECK(cb.halo == 5);
  // a node that only maps (pointwise) adds none
  g::Graph p; p.inputs = {"x"}; p.outputs = {"y"}; p.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})};
  CHECK(g::check(p, {{"x", stream_of(R)}}, Json::object(), reg).halo == 0);
}

void test_block_independence() {
  const g::Registry reg = toys();
  const std::vector<float> x = signal(R, N);
  const g::Checked c = g::check(chain_graph(), {{"x", stream_of(R)}}, Json::object(), reg);
  std::vector<float> d; std::vector<double> total; double count;
  reference_chain(x, 0.5f, 2, d, total, count);

  // the whole array handed as ONE block equals the plain reference
  const g::Arrays whole = g::apply(c, {{"x", array_of_rows(x, R, N)}});
  CHECK(whole.at("d").data.size() == d.size());
  bool exact = whole.at("d").data.size() == d.size();
  for (std::size_t i = 0; exact && i < d.size(); ++i) exact = whole.at("d").data[i] == static_cast<double>(d[i]);
  CHECK(exact);
  for (std::int64_t r = 0; r < R; ++r) { CHECK(std::fabs(whole.at("total").data[r] - total[r]) <= 1e-12 * total[r]); CHECK(std::fabs(whole.at("mean").data[r] - total[r] / count) <= 1e-12 * total[r]); }

  // the same graph over blocks of several widths (1, odd, a power of two, the whole, more than the whole) and sub-block widths: the same values
  for (const std::int64_t width : {1, 7, 64, 203, 500}) {
    for (const std::int64_t sub : {0, 1, 5, 1024}) {
      const Result res = run_blocks(c, x, R, N, width, {}, g::Blocking{sub, 256});
      CHECK(res.stream.size() == 1);
      CHECK(res.stream.at(0) == d);                                    // every column, exactly
      for (std::int64_t r = 0; r < R; ++r) {
        CHECK(std::fabs(res.arrays.at("total").data[r] - total[r]) <= 1e-12 * total[r]);        // folds: to the rounding of the sums
        CHECK(std::fabs(res.arrays.at("mean").data[r] - total[r] / count) <= 1e-12 * total[r]);
      }
    }
  }
  // parameters reach the nodes: another gain, another window
  const g::Checked c2 = g::check(chain_graph(), {{"x", stream_of(R)}}, Json{{"gain", 2.0}, {"wide", 4}}, reg);
  reference_chain(x, 2.0f, 4, d, total, count);
  CHECK(c2.halo == 5);
  CHECK(run_blocks(c2, x, R, N, 13).stream.at(0) == d);
}

void test_constants_and_before() {
  const g::Registry reg = toys();
  const std::vector<float> x = signal(R, N);
  g::Graph gr; gr.inputs = {"x", "k"}; gr.outputs = {"y", "k2"};
  gr.before = {use("toy.twice", {{"k", "k"}}, {"k2"})};
  gr.chunks = {use("toy.add_bias", {{"x", "x"}, {"bias", "k2"}}, {"y"})};   // a constant made by a `before` node, bound once in prepare
  const g::Checked c = g::check(gr, {{"x", stream_of(R)}, {"k", array_of(R)}}, Json::object(), reg);
  g::Array k; k.rows = R; k.cols = 1; k.data = {1.0, 2.0, 3.0};
  const Result res = run_blocks(c, x, R, N, 50, {{"k", k}});
  bool ok = res.stream.at(0).size() == x.size();
  for (std::int64_t r = 0; ok && r < R; ++r) for (std::int64_t t = 0; t < N; ++t) ok = ok && res.stream.at(0)[static_cast<std::size_t>(r * N + t)] == x[static_cast<std::size_t>(r * N + t)] + static_cast<float>(2 * (r + 1));
  CHECK(ok);
  CHECK(res.arrays.at("k2").data == (std::vector<double>{2.0, 4.0, 6.0}));
  CHECK(res.record.find("bias_rows") != std::string::npos);                  // what preparation resolved is recorded
  CHECK(refused([&] { g::Execution e(c, {}); }, "the constants must hold it"));
  CHECK(refused([&] { g::Execution e(c, {{"k", k}, {"x", k}}); }, "is streamed"));
}

void test_state() {
  const g::Registry reg = toys();
  const std::vector<float> x = signal(R, N);
  g::Graph gr; gr.inputs = {"x"}; gr.outputs = {"d", "total"};
  gr.chunks = {use("toy.cumsum", {{"x", "x"}}, {"d"}), use("toy.power_sum", {{"x", "x"}}, {"total", "count"})};
  const g::Checked c = g::check(gr, {{"x", stream_of(R)}}, Json::object(), reg);
  const Result one = run_blocks(c, x, R, N, N), many = run_blocks(c, x, R, N, 9);
  CHECK(one.stream.at(0) == many.stream.at(0));                              // the carry crosses every block edge
  double s = 0;
  for (std::int64_t t = 0; t < N; ++t) s += x[static_cast<std::size_t>(t)];
  CHECK(std::fabs(many.stream.at(0)[N - 1] - static_cast<float>(s)) <= 1e-4f);
  for (std::int64_t r = 0; r < R; ++r) CHECK(std::fabs(many.arrays.at("total").data[r] - one.arrays.at("total").data[r]) <= 1e-12 * one.arrays.at("total").data[r]);
  // out of time order: refused (a block whose first sample is not the one after the last)
  CHECK(refused([&] { run_blocks(c, x, R, N, 50, {}, {}, std::vector<std::int64_t>{50, 0}); }, "blocks must meet in time order"));
  CHECK(refused([&] { g::Execution e(c, {}); e.finish(); e.block({}, g::Place{0, 1, nullptr}); }, "after finish"));
  // a block that does not hold its streamed input, or whose own columns do not add up, is refused
  CHECK(refused([&] { g::Execution e(c, {}); e.block({}, g::Place{0, 1, nullptr}); }, "is missing"));
  // a graph with no state does not care about order
  g::Graph p; p.inputs = {"x"}; p.outputs = {"y"}; p.chunks = {use("toy.scale", {{"x", "x"}}, {"y"})};
  const g::Checked cp = g::check(p, {{"x", stream_of(R)}}, Json::object(), reg);
  CHECK(run_blocks(cp, x, R, N, 50, {}, {}, std::vector<std::int64_t>{150, 0, 200, 100, 50}).stream.at(0).size() == x.size());
}

void test_retime() {
  const g::Registry reg = toys();
  constexpr std::int64_t n = 240;
  const std::vector<float> x = signal(R, n);
  g::Graph gr; gr.inputs = {"x"}; gr.outputs = {"g", "total"};
  gr.chunks = {use("toy.frames", {{"x", "x"}}, {"f"}), use("toy.scale", {{"x", "f"}}, {"g"}, Json{{"a", 2.0}}), use("toy.power_sum", {{"x", "g"}}, {"total", "count"})};
  const g::Checked c = g::check(gr, {{"x", stream_of(R)}}, Json::object(), reg);
  CHECK(c.retime && *c.retime == 0 && c.halo == 0);
  CHECK(c.values.at("g").axis == g::Axis::frames && std::fabs(*c.values.at("g").sfreq - 100.0 / kF) < 1e-12);
  // reference: frames of 8 samples, scaled, the sum of squares
  std::vector<float> ref(static_cast<std::size_t>(R * n / kF));
  std::vector<double> tot(R, 0.0);
  for (std::int64_t r = 0; r < R; ++r) for (std::int64_t j = 0; j < n / kF; ++j) {
    double s = 0; for (std::int64_t k = 0; k < kF; ++k) s += x[static_cast<std::size_t>(r * n + j * kF + k)];
    const float f = 2.0f * static_cast<float>(s / kF);
    ref[static_cast<std::size_t>(r * (n / kF) + j)] = f; tot[r] += static_cast<double>(f) * f;
  }
  for (const std::int64_t width : {8, 24, 240}) for (const std::int64_t frames : {0, 1, 2, 7}) {
    const Result res = run_blocks(c, x, R, n, width, {}, g::Blocking{1024, frames});
    CHECK(res.stream.at(0) == ref);
    for (std::int64_t r = 0; r < R; ++r) CHECK(std::fabs(res.arrays.at("total").data[r] - tot[r]) <= 1e-12 * tot[r]);
  }
  const g::Arrays whole = g::apply(c, {{"x", array_of_rows(x, R, n)}});
  CHECK(whole.at("g").cols == n / kF && whole.at("total").data.size() == static_cast<std::size_t>(R));
}

void test_apply_conversions() {
  const g::Registry reg = toys();
  g::Graph p; p.inputs = {"x"}; p.outputs = {"y"}; p.chunks = {use("toy.scale", {{"x", "x"}}, {"y"}, Json{{"a", 3.0}})};
  const g::Checked c = g::check(p, {{"x", stream_of(2)}}, Json::object(), reg);
  g::Array a; a.rows = 2; a.cols = 3; a.data = {1, 2, 3, 4, 5, 6};
  const g::Arrays y = g::apply(c, {{"x", a}});
  CHECK(y.at("y").rows == 2 && y.at("y").cols == 3 && y.at("y").data == (std::vector<double>{3, 6, 9, 12, 15, 18}));
  CHECK(refused([&] { g::apply(c, {}); }, "was not handed"));
  g::Array wrong = a; wrong.comps = 2; wrong.data.resize(12);
  CHECK(refused([&] { g::apply(c, {{"x", wrong}}); }, "component(s) per element"));
}

}  // namespace

int main() {
  test_check_refusals();
  test_halo();
  test_block_independence();
  test_constants_and_before();
  test_state();
  test_retime();
  test_apply_conversions();
  if (failures == 0) std::printf("PASS test_graph\n");
  else std::printf("FAIL test_graph (%d check(s) failed)\n", failures);
  return failures ? 1 : 0;
}

// nxr-graph — the gauge's nodes: frame.rotate, frame.embed, frame.project (frame_nodes.h).
#include "nxr/graph/frame_nodes.h"

#include <cmath>

namespace nxr::graph {

namespace {

NodeTraits traits() {
  return node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "each element and column on its own; the arithmetic is in double and the result is rounded to the input's precision");
}

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

// a block of the shape of `in` with `rows` rows (the halo kept as it is)
Block shaped(const Block& in, std::int64_t rows, std::int64_t keep, bool complex) {
  Block o;
  o.rows = rows;
  o.left = std::min(keep, in.left);
  o.right = std::min(keep, in.right);
  o.width = o.left + in.own() + o.right;
  o.re.assign(static_cast<std::size_t>(rows * o.width), 0.f);
  if (complex) o.im.assign(o.re.size(), 0.f);
  return o;
}

std::size_t ix(std::int64_t r, std::int64_t w, std::int64_t j) { return static_cast<std::size_t>(r * w + j); }

struct Grid { std::vector<double> e1, e2; std::int64_t V = 0; };   // e1, e2: [V x 3]
Grid grid_of(const Array& a, const std::string& who) {
  need(a.cols == 3 && a.comps == 2 && static_cast<std::int64_t>(a.data.size()) == a.rows * 3 * 2, who + ": `grid` is the surface's vertexGrid: rows V, 3 columns, 2 components (e1, e2)");
  Grid g;
  g.V = a.rows;
  g.e1.resize(static_cast<std::size_t>(a.rows * 3));
  g.e2.resize(g.e1.size());
  for (std::int64_t v = 0; v < a.rows; ++v)
    for (int c = 0; c < 3; ++c) {
      g.e1[static_cast<std::size_t>(v * 3 + c)] = a.data[static_cast<std::size_t>((v * 3 + c) * 2)];
      g.e2[static_cast<std::size_t>(v * 3 + c)] = a.data[static_cast<std::size_t>((v * 3 + c) * 2 + 1)];
    }
  return g;
}

}  // namespace

NodeSpec frame_rotate_node() {
  NodeSpec s;
  s.name = "frame.rotate";
  s.kind = Kind::chunk;
  s.traits = traits();
  s.inputs = {{"x", "a tangent value, complex, spin k, in some gauge"}, {"to", "the angle of the target reference in that gauge, radians, one per element"}};
  s.outputs = {{"y", "x in the target frame"}};
  s.parameters = {{"spin", Json(0), "the spin k, only when the input names none (a tangent vector 1, a line field 2)", "", {}}};
  s.matches = "an analytic rotation (tests): y = x e^(-i k to)";
  s.description = "x e^(-i k to) per element: the components of a spin-k value on the frame whose reference lies at angle `to` in x's gauge (north's gauge angle gives (north, west) for k = 1; for k = 2 a line field's double angle is measured from north).";
  s.plan = [](const Types& in, const Json& p) {
    const PortType& x = in.at("x");
    need(x.value == ValueType::complex, "frame.rotate: x must be complex, it is " + std::string(to_string(x.value)));
    need(in.at("to").value == ValueType::real, "frame.rotate: `to` must be real (radians)");
    need(in.at("to").rows == x.rows, "frame.rotate: `to` has " + std::to_string(in.at("to").rows) + " elements, x has " + std::to_string(x.rows) + " rows");
    PortType y = x;
    if (y.spin == 0 && p.contains("spin") && p["spin"].is_number_integer()) y.spin = p["spin"].get<int>();
    return Planned{{{"y", y}}, 0};
  };
  s.prepare = [](const Types& in, const Arrays& c, const Json& p, Shared&) {
    const int k = in.at("x").spin != 0 ? in.at("x").spin : p.value("spin", 0);
    need(k != 0, "frame.rotate: x has no spin; name it as the parameter `spin`");
    need(c.count("to") != 0, "frame.rotate: `to` is not given");
    return std::any(std::make_pair(k, c.at("to").data));
  };
  s.chunk = [](const std::any& prepared, const Blocks& in, std::int64_t keep, const Place&) {
    const auto& [k, to] = std::any_cast<const std::pair<int, std::vector<double>>&>(prepared);
    const Block& x = in.at("x");
    need(x.complex(), "frame.rotate: the block is not complex");
    need(static_cast<std::size_t>(x.rows) == to.size(), "frame.rotate: `to` has " + std::to_string(to.size()) + " elements, the block " + std::to_string(x.rows) + " rows");
    Block y = shaped(x, x.rows, keep, true);
    for (std::int64_t r = 0; r < y.rows; ++r) {
      const double th = -static_cast<double>(k) * to[static_cast<std::size_t>(r)];
      const double c = std::cos(th), s = std::sin(th);
      for (std::int64_t j = 0; j < y.width; ++j) {
        const std::size_t src = ix(r, x.width, x.left - y.left + j);
        const double a = x.re[src], b = x.im[src];
        y.re[ix(r, y.width, j)] = static_cast<float>(a * c - b * s);
        y.im[ix(r, y.width, j)] = static_cast<float>(a * s + b * c);
      }
    }
    return Blocks{{"y", std::move(y)}};
  };
  return s;
}

NodeSpec frame_embed_node() {
  NodeSpec s;
  s.name = "frame.embed";
  s.kind = Kind::chunk;
  s.traits = traits();
  s.inputs = {{"x", "a tangent vector, complex, spin 1"}, {"grid", "the surface's vertexGrid (e1 + i e2), in x's gauge"}};
  s.outputs = {{"y", "the embedded 3-vector, three consecutive rows per vertex"}};
  s.matches = "an analytic embedding (tests): y = a e1 + b e2";
  s.description = "z = a + ib at a vertex becomes the 3-vector a e1 + b e2 (tangent to the surface, gauge-free).";
  s.plan = [](const Types& in, const Json&) {
    const PortType& x = in.at("x");
    need(x.value == ValueType::complex, "frame.embed: x must be complex, it is " + std::string(to_string(x.value)));
    need(in.at("grid").rows == x.rows, "frame.embed: `grid` has " + std::to_string(in.at("grid").rows) + " rows, x has " + std::to_string(x.rows));
    PortType y = x;
    y.value = ValueType::vector3; y.complex_parts = false; y.spin = 0; y.gauge.clear(); y.components = 3; y.rows = x.rows * 3;
    return Planned{{{"y", y}}, 0};
  };
  s.prepare = [](const Types&, const Arrays& c, const Json&, Shared&) {
    need(c.count("grid") != 0, "frame.embed: `grid` is not given");
    return std::any(grid_of(c.at("grid"), "frame.embed"));
  };
  s.chunk = [](const std::any& prepared, const Blocks& in, std::int64_t keep, const Place&) {
    const Grid& g = std::any_cast<const Grid&>(prepared);
    const Block& x = in.at("x");
    need(x.complex() && x.rows == g.V, "frame.embed: the block has " + std::to_string(x.rows) + " rows, the grid " + std::to_string(g.V) + " vertices");
    Block y = shaped(x, 3 * x.rows, keep, false);
    for (std::int64_t v = 0; v < x.rows; ++v)
      for (std::int64_t j = 0; j < y.width; ++j) {
        const std::size_t src = ix(v, x.width, x.left - y.left + j);
        const double a = x.re[src], b = x.im[src];
        for (int c = 0; c < 3; ++c)
          y.re[ix(3 * v + c, y.width, j)] = static_cast<float>(a * g.e1[static_cast<std::size_t>(v * 3 + c)] + b * g.e2[static_cast<std::size_t>(v * 3 + c)]);
      }
    return Blocks{{"y", std::move(y)}};
  };
  return s;
}

NodeSpec frame_project_node() {
  NodeSpec s;
  s.name = "frame.project";
  s.kind = Kind::chunk;
  s.traits = traits();
  s.inputs = {{"x", "an embedded 3-vector, three consecutive rows per vertex"}, {"grid", "the surface's vertexGrid (e1 + i e2)"}};
  s.outputs = {{"y", "the tangent vector in the grid's gauge, complex, spin 1"}};
  s.matches = "an analytic projection (tests): z = (v . e1) + i (v . e2)";
  s.description = "The inverse of frame.embed on the tangent plane: z = (v . e1) + i (v . e2); the part of v along the normal is dropped.";
  s.plan = [](const Types& in, const Json&) {
    const PortType& x = in.at("x");
    need(x.value == ValueType::vector3, "frame.project: x must be a vector3 value, it is " + std::string(to_string(x.value)));
    need(x.rows == 3 * in.at("grid").rows, "frame.project: x has " + std::to_string(x.rows) + " rows, the grid " + std::to_string(in.at("grid").rows) + " vertices (three rows each)");
    PortType y = x;
    y.value = ValueType::complex; y.complex_parts = false; y.components = 1; y.rows = x.rows / 3;
    y.spin = in.at("grid").spin; y.gauge = in.at("grid").gauge;
    return Planned{{{"y", y}}, 0};
  };
  s.prepare = [](const Types&, const Arrays& c, const Json&, Shared&) {
    need(c.count("grid") != 0, "frame.project: `grid` is not given");
    return std::any(grid_of(c.at("grid"), "frame.project"));
  };
  s.chunk = [](const std::any& prepared, const Blocks& in, std::int64_t keep, const Place&) {
    const Grid& g = std::any_cast<const Grid&>(prepared);
    const Block& x = in.at("x");
    need(x.rows == 3 * g.V, "frame.project: the block has " + std::to_string(x.rows) + " rows, the grid " + std::to_string(g.V) + " vertices");
    Block y = shaped(x, g.V, keep, true);
    for (std::int64_t v = 0; v < g.V; ++v)
      for (std::int64_t j = 0; j < y.width; ++j) {
        double a = 0, b = 0;
        for (int c = 0; c < 3; ++c) {
          const double w = x.re[ix(3 * v + c, x.width, x.left - y.left + j)];
          a += w * g.e1[static_cast<std::size_t>(v * 3 + c)];
          b += w * g.e2[static_cast<std::size_t>(v * 3 + c)];
        }
        y.re[ix(v, y.width, j)] = static_cast<float>(a);
        y.im[ix(v, y.width, j)] = static_cast<float>(b);
      }
    return Blocks{{"y", std::move(y)}};
  };
  return s;
}

void register_frame_nodes(Registry& r) {
  r.add(frame_rotate_node());
  r.add(frame_embed_node());
  r.add(frame_project_node());
}

}  // namespace nxr::graph

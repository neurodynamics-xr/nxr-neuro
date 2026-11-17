// nxr::io::fibers — fiber bundles and their ends in a subject or template package (nxr/io/fibers.h).
#include "nxr/io/fibers.h"

#include <algorithm>
#include <cstring>

#include "nxr/io/array_reader.h"
#include "nxr/io/errors.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::io::fibers {

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
template <class S> const char* precision_of() { return sizeof(S) == 4 ? "float32" : "float64"; }
std::string curve_path(const std::string& name) { return folder_for("curve") + "/" + sanitize_node_name(name); }
}  // namespace

template <class S>
Stored store_bundle(SubjectWriter& sub, const Bundle<S>& b) {
  if (b.n_fibers < 1) refuse("a fiber bundle holds at least one fiber");
  if (b.points_per_fiber < 2) refuse("a fiber has at least 2 points");
  const std::int64_t rows = b.n_fibers * b.points_per_fiber;
  if (static_cast<std::int64_t>(b.points.size()) != 3 * rows) refuse("the bundle's points are [" + std::to_string(rows) + " x 3] (" + std::to_string(b.n_fibers) + " fibers of " + std::to_string(b.points_per_fiber) + " points)");
  if (!b.colors.empty() && static_cast<std::int64_t>(b.colors.size()) != 3 * rows) refuse("the bundle's colors are [" + std::to_string(rows) + " x 3]");
  const std::string name = sanitize_node_name(b.name);
  if (name.empty()) refuse("a bundle needs a name");
  if (sub.by_name("manifold", name)) refuse("the subject already has a manifold named '" + name + "'");
  const std::string node = curve_path(name);
  ManifoldSpec ms;
  ms.name = name;
  ms.path = node;
  ms.type = "curve";
  ms.n_vertices = rows;
  ms.topology = {"path", 1, static_cast<int>(b.n_fibers), std::nullopt, {}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = node + "/points";
  ms.geometry = g;
  ms.cols.set("n_edges", b.n_fibers * (b.points_per_fiber - 1));
  if (!b.comment.empty()) ms.cols.set("comment", b.comment); else ms.cols.null("comment");
  if (!b.source.empty()) ms.cols.set("source", b.source);
  Json producer = b.producer.is_object() ? b.producer : Json::object();
  producer["points_per_component"] = b.points_per_fiber;
  producer["precision"] = precision_of<S>();
  producer["description"] = std::to_string(b.n_fibers) + " fibers of " + std::to_string(b.points_per_fiber) + " points each (fiber-major), the subject's SCS in metres";
  if (!b.origin.is_null()) producer["origin"] = b.origin;
  ms.cols.json("producer_json", producer);
  ms.populate = [&](Store&, const std::string&) {
    ArrayOptions ao;
    ao.compress = true;
    sub.write_array(node + "/points", view(b.points, {rows, 3}), ao);
    if (!b.colors.empty()) sub.write_array(node + "/colors", view(b.colors, {rows, 3}), ao);
  };
  Stored out;
  out.id = sub.manifold(std::move(ms));
  out.name = name;
  out.path = node;
  out.n_fibers = b.n_fibers;
  out.points_per_fiber = b.points_per_fiber;
  out.precision = precision_of<S>();
  return out;
}

namespace {
task::Node require_curve(const task::Subject& s, const std::string& name, bool pending_too = false) {
  task::Want w;
  w.cls = task::Cls::manifold;
  w.name = sanitize_node_name(name);
  w.type = "curve";
  w.any_status = pending_too;   // while a writer is composing, the bundle it just stored is not complete yet
  return s.require(w, "the fiber bundle '" + name + "'");
}
std::int64_t points_per_fiber(const task::Node& n) {
  if (!n.producer.contains("points_per_component")) refuse("the curve '" + n.name + "' does not say its points per fiber");
  return n.producer["points_per_component"].get<std::int64_t>();
}
}  // namespace

std::string bundle_precision(const task::Subject& subject, const Store& store, const std::string& name) {
  const task::Node n = require_curve(subject, name);
  const auto rd = ArrayReader::open(store, *n.path + "/points");
  return rd.dtype() == DType::Float32 ? "float32" : "float64";
}

template <class S>
Bundle<S> read_bundle(const task::Subject& subject, const Store& store, const std::string& name) {
  const task::Node n = require_curve(subject, name);
  const auto rd = ArrayReader::open(store, *n.path + "/points");
  if ((rd.dtype() == DType::Float32) != (sizeof(S) == 4)) refuse("the bundle '" + name + "' is stored in " + (rd.dtype() == DType::Float32 ? "float32" : "float64") + ", not " + precision_of<S>() + " (precision follows the input: convert it explicitly)");
  Bundle<S> b;
  b.name = n.name;
  b.points_per_fiber = points_per_fiber(n);
  const auto bytes = rd.read();
  b.points.resize(bytes.size() / sizeof(S));
  std::memcpy(b.points.data(), bytes.data(), bytes.size());
  b.n_fibers = static_cast<std::int64_t>(b.points.size() / 3) / b.points_per_fiber;
  b.source = n.fact("source").is_string() ? n.fact("source").get<std::string>() : "";
  if (n.producer.contains("origin")) b.origin = n.producer["origin"];
  b.producer = n.producer;
  return b;
}

template <class S>
void store_ends(SubjectWriter& sub, const std::string& bundle, const std::string& end_surface, const Ends<S>& ends, const Json& producer) {
  const task::Subject subject = task::Subject::of(sub);
  const task::Node curve = require_curve(subject, bundle, true);
  const std::int64_t nP = points_per_fiber(curve), nfp = *curve.n_vertices;
  const std::int64_t nF = nfp / nP;
  if (static_cast<std::int64_t>(ends.vertex.size()) != 2 * nF || ends.distance.size() != ends.vertex.size()) refuse("the ends of '" + bundle + "' are 2 x " + std::to_string(nF) + " (vertices and distances)");
  const auto surf = sub.find("manifold", "source/" + sanitize_node_name(end_surface));
  if (!surf) refuse("the subject has no surface '" + end_surface + "' to place the ends on");
  const std::int64_t nv = integer(*surf, "n_vertices");
  const std::string op_name = curve.name + "_ends_" + sanitize_node_name(end_surface);
  if (sub.by_name("operator", op_name)) refuse("the ends of '" + bundle + "' on '" + end_surface + "' are already stored");
  for (const auto v : ends.vertex) if (v < 0 || v >= nv) refuse("an end's vertex " + std::to_string(v) + " is outside the surface's " + std::to_string(nv) + " vertices");
  // CSR over the surface's vertices: the end rows (first point f·nP, last point f·nP + nP − 1) of the vertex each end is on, ascending
  std::vector<std::int64_t> rows(static_cast<std::size_t>(2 * nF));
  for (std::int64_t f = 0; f < nF; ++f) { rows[static_cast<std::size_t>(2 * f)] = f * nP; rows[static_cast<std::size_t>(2 * f + 1)] = f * nP + nP - 1; }
  std::vector<std::int64_t> order(rows.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::int64_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t c) {
    const auto va = ends.vertex[static_cast<std::size_t>(a)], vc = ends.vertex[static_cast<std::size_t>(c)];
    return va != vc ? va < vc : rows[static_cast<std::size_t>(a)] < rows[static_cast<std::size_t>(c)];
  });
  std::vector<std::int64_t> indptr(static_cast<std::size_t>(nv + 1), 0), indices(order.size());
  for (const auto v : ends.vertex) ++indptr[static_cast<std::size_t>(v) + 1];
  for (std::int64_t i = 0; i < nv; ++i) indptr[static_cast<std::size_t>(i) + 1] += indptr[static_cast<std::size_t>(i)];
  for (std::size_t i = 0; i < order.size(); ++i) indices[i] = rows[static_cast<std::size_t>(order[i])];
  const std::vector<float> ones(order.size(), 1.0f);
  const std::int64_t nnz = static_cast<std::int64_t>(order.size());
  const std::string dist_path = *curve.path + "/end_distance_" + sanitize_node_name(end_surface);
  OperatorSpec os;
  os.name = op_name;
  os.path = op_name;
  os.kind = "fibre ends";
  os.from_manifold_id = curve.id;
  os.layout = "csr";
  os.n_rows = nv;
  os.n_cols = nfp;
  os.cols.set("to_manifold_id", text(*surf, "id")).set("nnz", nnz).set("description", "each fiber's two ends -> the nearest vertex of " + end_surface + " (1 per end; no gate: the distances are in " + dist_path + ")");
  Json prod = producer.is_object() ? producer : Json::object();
  prod["method"] = "nearest vertex, Euclidean, SCS metres";
  prod["end_surface"] = end_surface;
  prod["end_distance_array"] = dist_path;
  prod["order"] = "fiber f: first end at 2f, last end at 2f + 1";
  prod["precision"] = precision_of<S>();
  if (curve.producer.contains("origin")) prod["origin"] = derived_origin(curve.producer, curve.name);
  os.cols.json("producer_json", prod);
  os.sparse = SparseArrays{{{"indptr", view(indptr, {nv + 1})}, {"indices", view(indices, {nnz})}, {"data", view(ones, {nnz})}}};
  sub.operator_(std::move(os));
  ArrayOptions ao;
  ao.compress = true;
  sub.write_array(dist_path, view(ends.distance, {2 * nF}), ao);
}

template <class S>
Ends<S> read_ends(const task::Subject& subject, const Store& store, const std::string& bundle, const std::string& end_surface) {
  const task::Node curve = require_curve(subject, bundle);
  const std::int64_t nP = points_per_fiber(curve), nF = *curve.n_vertices / nP;
  task::Want w;
  w.cls = task::Cls::operator_;
  w.name = curve.name + "_ends_" + sanitize_node_name(end_surface);
  const task::Node op = subject.require(w, "the ends of '" + bundle + "' on '" + end_surface + "'");
  const auto ip = ArrayReader::open(store, *op.path + "/indptr"), ix = ArrayReader::open(store, *op.path + "/indices");
  auto ints = [](const ArrayReader& r) {
    const auto b = r.read();
    std::vector<std::int64_t> v(b.size() / r.itemsize());
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (r.dtype() == DType::Int32) { std::int32_t x; std::memcpy(&x, b.data() + 4 * i, 4); v[i] = x; }
      else { std::int64_t x; std::memcpy(&x, b.data() + 8 * i, 8); v[i] = x; }
    }
    return v;
  };
  const auto indptr = ints(ip), indices = ints(ix);
  Ends<S> e;
  e.vertex.assign(static_cast<std::size_t>(2 * nF), -1);
  for (std::size_t v = 0; v + 1 < indptr.size(); ++v)
    for (std::int64_t k = indptr[v]; k < indptr[v + 1]; ++k) {
      const std::int64_t row = indices[static_cast<std::size_t>(k)], f = row / nP;
      e.vertex[static_cast<std::size_t>(2 * f + (row % nP == 0 ? 0 : 1))] = static_cast<std::int32_t>(v);
    }
  const auto dr = ArrayReader::open(store, *curve.path + "/end_distance_" + sanitize_node_name(end_surface));
  if ((dr.dtype() == DType::Float32) != (sizeof(S) == 4)) refuse("the end distances are stored in " + std::string(dr.dtype() == DType::Float32 ? "float32" : "float64") + ", not " + precision_of<S>());
  const auto bytes = dr.read();
  e.distance.resize(bytes.size() / sizeof(S));
  std::memcpy(e.distance.data(), bytes.data(), bytes.size());
  return e;
}

Json derived_origin(const Json& bundle_producer, const std::string& bundle_name) {
  if (!bundle_producer.is_object() || !bundle_producer.contains("origin") || bundle_producer["origin"].is_null()) return Json();
  Json o = bundle_producer["origin"];
  o["derived_from"] = bundle_name;
  const bool share_alike = o.contains("licence") && o["licence"].is_object() && o["licence"].value("share_alike", false);
  if (share_alike) o["derived_notice"] = "derived from " + bundle_name + ": shared under the same licence (ShareAlike) and with the same attribution";
  return o;
}

template Stored store_bundle(SubjectWriter&, const Bundle<float>&);
template Stored store_bundle(SubjectWriter&, const Bundle<double>&);
template Bundle<float> read_bundle(const task::Subject&, const Store&, const std::string&);
template Bundle<double> read_bundle(const task::Subject&, const Store&, const std::string&);
template void store_ends(SubjectWriter&, const std::string&, const std::string&, const Ends<float>&, const Json&);
template void store_ends(SubjectWriter&, const std::string&, const std::string&, const Ends<double>&, const Json&);
template Ends<float> read_ends(const task::Subject&, const Store&, const std::string&, const std::string&);
template Ends<double> read_ends(const task::Subject&, const Store&, const std::string&, const std::string&);

}  // namespace nxr::io::fibers

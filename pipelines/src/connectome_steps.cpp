// nxr-pipelines — the registered NODES of the structural connectome (spec docs/specs/2026-11-16-structural-connectome.md, section 4). Each is a pure function of arrays from
// compute/ (nxr/connectome.h, nxr/nearest.h, nxr/congruence.h) with its ports, parameters and traits; none reads a file, a subject or a database, and none has a time axis, so
// they run once, before the chunk loop (kind `setup`; the connectome has no recording).
//
//   fiber_ends          fibers → end_points                      the first and last point of each fiber
//   nearest_vertex      points, positions → vertex, distance     the nearest vertex of each point and its distance (a k-d tree)
//   end_gate            end_distance → keep                      the fibers whose both ends are within `max_end_distance_mm` of the end surface (`none`: all)
//   registration_match  template/subject spheres → match         template vertex → subject vertex through the registration (an empty column goes to its NEAREST subject vertex)
//   carry_ends          end_vertex, match → end_vertex           the ends of a template's fibers on the subject
//   smooth_memberships  label, faces → memberships               one-hot memberships smoothed over the mesh (the smoothing pushed onto the memberships)
//   partition_counts    end_vertex, keep, memberships → counts   the fiber counts over a partition (a measurement over the partition: fibers are the elements)
//   end_connectome      end_vertex, keep → fiber_counts          the materialised vertex matrix (on request)
//   connection_weights  fiber_counts, faces → connection_weights   the smoothed matrix smoothing · counts · smoothingᵀ, diagonal removed (sparse congruence)
//   prune_below         connection_weights → connection_weights  entries below a fraction of the strongest removed
//   largest_component   connection_weights → vertices, label     the biggest connected component's vertices
//
// ARRAY CONVENTIONS (an `Array` is [rows x cols] doubles with a `meta`): a dense array is as it is; a SPARSE matrix is COO, [nnz x 3] rows (row, column, value) with
// meta {"form": "coo", "shape": [rows, cols]}; an index is a whole number stored as a double. Precision follows the input: meta {"precision": "float32" | "float64"} (default
// float64) on the primary input sets the precision the node computes in and the one it gives back; no node casts silently, and two inputs of different precision are refused.
//
// What they cannot resolve: see the method's section in docs/reference.md (the ends of tractography are not the fibers' cortical terminations; smoothing spreads ends over
// the mesh and is not conserving; a template's fibers on a subject carry the template's anatomy through a spherical correspondence).
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nxr/connectome.h"
#include "nxr/congruence.h"
#include "nxr/nearest.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

bool is_f32(const Array& a) { return a.meta.is_object() && a.meta.value("precision", std::string("float64")) == "float32"; }
std::string precision_name(bool f32) { return f32 ? "float32" : "float64"; }
void same_precision(const std::string& node, const Array& a, const Array& b) {
  need(is_f32(a) == is_f32(b), node + ": the inputs are in different precision (" + precision_name(is_f32(a)) + " and " + precision_name(is_f32(b)) + "): convert one explicitly");
}

template <class Scalar> std::vector<Scalar> values(const Array& a) { return std::vector<Scalar>(a.data.begin(), a.data.end()); }
std::vector<std::int32_t> indices(const Array& a) {
  std::vector<std::int32_t> v(a.data.size());
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::int32_t>(std::llround(a.data[i]));
  return v;
}
template <class Scalar> Array dense(const std::vector<Scalar>& v, std::int64_t rows, std::int64_t cols, bool f32) {
  Array a;
  a.rows = rows; a.cols = cols;
  a.data.assign(v.begin(), v.end());
  a.meta = Json{{"precision", precision_name(f32)}};
  return a;
}
Array whole(const std::vector<std::int32_t>& v) {
  Array a;
  a.rows = static_cast<std::int64_t>(v.size()); a.cols = 1;
  a.data.assign(v.begin(), v.end());
  return a;
}
template <class Scalar> connectome::Sparse<Scalar> from_coo(const Array& a, const std::string& node) {
  need(a.meta.is_object() && a.meta.value("form", std::string()) == "coo" && a.meta.contains("shape") && a.cols == 3, node + ": a sparse input is COO: [nnz x 3] (row, column, value) with meta {form: coo, shape}");
  std::vector<Eigen::Triplet<Scalar>> t;
  for (std::int64_t i = 0; i < a.rows; ++i)
    t.emplace_back(static_cast<int>(std::llround(a.data[static_cast<std::size_t>(3 * i)])), static_cast<int>(std::llround(a.data[static_cast<std::size_t>(3 * i + 1)])), static_cast<Scalar>(a.data[static_cast<std::size_t>(3 * i + 2)]));
  connectome::Sparse<Scalar> m(a.meta["shape"][0].get<int>(), a.meta["shape"][1].get<int>());
  m.setFromTriplets(t.begin(), t.end());
  return m;
}
template <class Scalar> Array to_coo(const connectome::Sparse<Scalar>& m, bool f32) {
  Array a;
  a.cols = 3;
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename connectome::Sparse<Scalar>::InnerIterator it(m, k); it; ++it) { a.data.push_back(static_cast<double>(it.row())); a.data.push_back(static_cast<double>(it.col())); a.data.push_back(static_cast<double>(it.value())); }
  a.rows = static_cast<std::int64_t>(a.data.size() / 3);
  a.meta = Json{{"form", "coo"}, {"shape", Json::array({m.rows(), m.cols()})}, {"precision", precision_name(f32)}};
  return a;
}

Value array_value(const Json& facts) {
  Value v;
  v.cls = "array";
  v.facts = facts;
  return v;
}
void present(const Values& ins, const std::string& node, std::initializer_list<const char*> ports) {
  for (const char* p : ports) need(ins.count(p) != 0, node + ": the input `" + p + "` is missing");
}
const std::vector<std::string> kWarnings = {
    "tractography ends are not the fibers' cortical terminations: streamlines stop at the grey/white boundary",
    "an end's vertex is a location on the chosen surface, not a depth; smoothing spreads ends over the mesh and is not conserving"};

}  // namespace

void add_connectome_steps(std::vector<StepSpec>& r) {
  auto traits = [](const std::string& why) {
    return node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "no time axis: runs once on arrays; " + why);
  };
  auto simple = [&](const std::string& name, std::vector<std::string> ins, std::vector<std::string> optional, std::vector<std::string> outs, Json params, const std::string& matches,
                    const std::string& description, const std::string& why) {
    StepSpec s;
    s.name = name; s.kind = "setup";
    s.traits = traits(why);
    s.inputs = std::move(ins); s.optional = std::move(optional); s.outputs = std::move(outs);
    s.parameters = std::move(params);
    s.matches = matches;
    s.description = description;
    return s;
  };
  auto plan_arrays = [](const std::string& node, std::vector<std::string> ports, std::vector<std::string> outs, const std::vector<std::string>& optional = {}) {
    return [node, ports, outs, optional](const Values& ins, const Json&, const PlanContext&) {
      for (const auto& p : ports) if (std::find(optional.begin(), optional.end(), p) == optional.end()) need(ins.count(p) != 0, node + ": the input `" + p + "` is missing");
      Values out;
      for (const auto& o : outs) out[o] = array_value(Json{{"node", node}, {"port", o}});
      return out;
    };
  };
  // ── fiber_ends ──
  {
    StepSpec s = simple("fiber_ends", {"fibers"}, {}, {"end_points"}, Json{{"points_per_fiber", nullptr}}, "nxr::connectome::fiberEnds; the Studio's connectome.ts ends",
                        "The first and the last point of each fiber: fibers [nF·nP x 3] (fiber-major, SCS metres) → end_points [2nF x 3] (row f: the first end; row nF + f: the last). Parameter points_per_fiber (nP, "
                        "points, no default; >= 2).", "indexing only");
    s.plan = plan_arrays("fiber_ends", {"fibers"}, {"end_points"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["points_per_fiber"].is_number_integer() && p["points_per_fiber"].get<int>() >= 2, "fiber_ends: points_per_fiber is the points of each fiber, a whole number >= 2");
      return std::any(p["points_per_fiber"].get<std::int64_t>());
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const std::int64_t nP = std::any_cast<std::int64_t>(st);
      const Array& f = ins.at("fibers");
      need(f.cols == 3 && f.rows % nP == 0, "fiber_ends: fibers are [nF·" + std::to_string(nP) + " x 3]");
      Arrays out;
      if (is_f32(f)) out["end_points"] = dense<float>(connectome::fiberEnds<float>(values<float>(f), nP), 2 * f.rows / nP, 3, true);
      else out["end_points"] = dense<double>(connectome::fiberEnds<double>(values<double>(f), nP), 2 * f.rows / nP, 3, false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── nearest_vertex ──
  {
    StepSpec s = simple("nearest_vertex", {"points", "positions"}, {}, {"vertex", "distance"}, Json::object(), "nxr::nearest::nearestVertices (k-d tree, Bentley 1975)",
                        "The vertex of `positions` [vertices x 3] nearest each of `points` [n x 3] (Euclidean; the lowest index on a tie) → vertex [n] and distance [n] in the points' unit (metres for SCS). The distance is "
                        "kept whatever gate follows: it does not depend on one.", "each point is looked up on its own");
    s.plan = plan_arrays("nearest_vertex", {"points", "positions"}, {"vertex", "distance"});
    s.prepare = [](const Values&, const Json&, ExecContext&) { return std::any(); };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) {
      const Array &q = ins.at("points"), &pos = ins.at("positions");
      same_precision("nearest_vertex", q, pos);
      need(q.cols == 3 && pos.cols == 3, "nearest_vertex: points and positions are [n x 3]");
      Arrays out;
      if (is_f32(q)) { const auto r = nxr::nearest::nearestVertices<float>(values<float>(pos), values<float>(q)); out["vertex"] = whole(r.vertex); out["distance"] = dense<float>(r.distance, q.rows, 1, true); }
      else { const auto r = nxr::nearest::nearestVertices<double>(values<double>(pos), values<double>(q)); out["vertex"] = whole(r.vertex); out["distance"] = dense<double>(r.distance, q.rows, 1, false); }
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── end_gate ──
  {
    StepSpec s = simple("end_gate", {"end_distance"}, {}, {"keep"}, Json{{"max_end_distance_mm", 5}},
                        "the spec's distance gate (decided 2026-11-16); `none` reproduces tess_connectome",
                        "The distance gate: end_distance [2nF] (metres, block order) → keep [nF], 1 for a fiber with both ends within max_end_distance_mm (millimetres; default 5; \"none\" keeps every fiber, as "
                        "the reference does), 0 for a fiber left out. The number left out is in keep's meta (`n_left_out`, `n_fibers`, `max_end_distance_mm`) for the report and the provenance.",
                        "each fiber is gated on its own two distances");
    s.plan = plan_arrays("end_gate", {"end_distance"}, {"keep"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      const Json& g = p["max_end_distance_mm"];
      need((g.is_string() && g == "none") || (g.is_number() && g.get<double>() >= 0), "end_gate: max_end_distance_mm is a distance in millimetres >= 0, or \"none\"");
      return std::any(g);
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json g = std::any_cast<Json>(st);
      const Array& d = ins.at("end_distance");
      need(d.rows % 2 == 0, "end_gate: the ends of a bundle are two a fiber");
      std::vector<std::uint8_t> keep;
      if (g.is_string()) keep = connectome::keepAll(d.rows / 2);
      else if (is_f32(d)) keep = connectome::withinGate<float>(values<float>(d), static_cast<float>(g.get<double>() * 1e-3));
      else keep = connectome::withinGate<double>(values<double>(d), g.get<double>() * 1e-3);
      Array a;
      a.rows = static_cast<std::int64_t>(keep.size()); a.cols = 1;
      std::int64_t out = 0;
      for (auto k : keep) { a.data.push_back(k); out += !k; }
      a.meta = Json{{"n_fibers", a.rows}, {"n_left_out", out}, {"max_end_distance_mm", g}};
      return Arrays{{"keep", a}};
    };
    r.push_back(std::move(s));
  }
  // ── registration_match ──
  {
    StepSpec s = simple("registration_match", {"template_sphere", "template_pieces", "subject_sphere", "subject_pieces"}, {}, {"match"}, Json{{"neighbours", 8}},
                        "nxr::connectome::registrationMatch (Shepard weights, Franke & Nielson 1980; the quirk of the reference fixed)",
                        "The match of each TEMPLATE vertex to a SUBJECT vertex through the registered spheres, piece by piece (left to left, right to right): the K = `neighbours` (default 8) nearest template "
                        "vertices of each subject vertex with modified Shepard weights, then, for each template vertex, the subject vertex with the largest weight (the lowest index on a tie). A template "
                        "vertex no subject vertex takes weight from goes to its NEAREST subject vertex on the sphere, never to vertex 0 (the reference's defect, fixed 2026-11-16). Pieces are [pieces x 2] "
                        "(first vertex, count). `match` meta carries `empty_columns`.", "a vertex is matched on its own");
    s.plan = plan_arrays("registration_match", {"template_sphere", "template_pieces", "subject_sphere", "subject_pieces"}, {"match"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["neighbours"].is_number_integer() && p["neighbours"].get<int>() >= 2, "registration_match: neighbours is a whole number >= 2");
      return std::any(p["neighbours"].get<int>());
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const int k = std::any_cast<int>(st);
      const Array &ts = ins.at("template_sphere"), &ss = ins.at("subject_sphere");
      same_precision("registration_match", ts, ss);
      auto pieces = [](const Array& a) {
        need(a.cols == 2, "registration_match: pieces are [pieces x 2] (first vertex, count)");
        std::vector<nxr::registration::Piece> p;
        for (std::int64_t i = 0; i < a.rows; ++i) p.push_back({static_cast<std::int64_t>(std::llround(a.data[static_cast<std::size_t>(2 * i)])), static_cast<std::int64_t>(std::llround(a.data[static_cast<std::size_t>(2 * i + 1)]))});
        return p;
      };
      connectome::MatchReport rep;
      const auto tp = pieces(ins.at("template_pieces")), sp = pieces(ins.at("subject_pieces"));
      std::vector<std::int32_t> m = is_f32(ts) ? connectome::registrationMatch<float>(values<float>(ts), tp, values<float>(ss), sp, k, &rep) : connectome::registrationMatch<double>(values<double>(ts), tp, values<double>(ss), sp, k, &rep);
      Array a = whole(m);
      a.meta = Json{{"empty_columns", rep.empty_columns}};
      return Arrays{{"match", a}};
    };
    r.push_back(std::move(s));
  }
  // ── carry_ends ──
  {
    StepSpec s = simple("carry_ends", {"end_vertex", "match"}, {}, {"end_vertex"}, Json::object(), "nxr::connectome::carryEnds", "The subject vertex of each end of a template's fibers: match[end_vertex].",
                        "each end is carried on its own");
    s.plan = plan_arrays("carry_ends", {"end_vertex", "match"}, {"end_vertex"});
    s.prepare = [](const Values&, const Json&, ExecContext&) { return std::any(); };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) { return Arrays{{"end_vertex", whole(connectome::carryEnds(indices(ins.at("end_vertex")), indices(ins.at("match"))))}}; };
    r.push_back(std::move(s));
  }
  // ── smooth_memberships ──
  {
    StepSpec s = simple("smooth_memberships", {"label", "faces"}, {}, {"memberships"}, Json{{"regions", nullptr}, {"hops", 3}}, "nxr::connectome::memberships, smoothingStep, smoothMemberships",
                        "The smoothing pushed onto a partition's memberships: label [vertices] (the region ordinal of each vertex, -1 in none) and faces [triangles x 3] → memberships [vertices x regions] (COO), the one-hot memberships "
                        "averaged over the mesh `hops` times (default 3, the reference's; 0: one-hot). `regions` (the number of regions, count) defaults to the largest label + 1. A smoothed membership is a fractional "
                        "membership: how much of the region a vertex's neighbourhood reaches. The vertex matrix is never formed.", "a vertex's membership depends on its neighbourhood only");
    s.plan = plan_arrays("smooth_memberships", {"label", "faces"}, {"memberships"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["hops"].is_number_integer() && p["hops"].get<int>() >= 0, "smooth_memberships: hops is a whole number >= 0 (0: no smoothing)");
      need(p["regions"].is_null() || (p["regions"].is_number_integer() && p["regions"].get<int>() >= 1), "smooth_memberships: regions is the number of regions, a whole number >= 1 (default: the largest label + 1)");
      return std::any(p);
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json p = std::any_cast<Json>(st);
      const Array& lab = ins.at("label");
      auto label = indices(lab);
      std::int64_t nRegions = p["regions"].is_null() ? 0 : p["regions"].get<std::int64_t>();
      if (p["regions"].is_null()) for (auto l : label) nRegions = std::max<std::int64_t>(nRegions, l + 1);
      const auto faces = indices(ins.at("faces"));
      Arrays out;
      const bool f32 = is_f32(lab);
      if (f32) out["memberships"] = to_coo<float>(connectome::smoothMemberships<float>(connectome::smoothingStep<float>(faces, static_cast<std::int64_t>(label.size())), connectome::memberships<float>(label, nRegions), p["hops"].get<int>()), true);
      else out["memberships"] = to_coo<double>(connectome::smoothMemberships<double>(connectome::smoothingStep<double>(faces, static_cast<std::int64_t>(label.size())), connectome::memberships<double>(label, nRegions), p["hops"].get<int>()), false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── partition_counts ──
  {
    StepSpec s = simple("partition_counts", {"end_vertex", "keep", "memberships"}, {"keep"}, {"counts"}, Json::object(), "nxr::connectome::partitionCounts (the Studio's connectome.ts tileConnectome / frameConnectome)",
                        "The fiber counts over a partition: end_vertex [2nF] (block order), keep [nF] (the gate; optional: every fiber) and the (smoothed) memberships [vertices x regions] → counts [regions x regions] (COO, symmetric): for each "
                        "region pair the sum over the kept fibers of (membership of the first end in a)(membership of the second end in b) + the same with the ends swapped. A fiber with both ends in one region "
                        "adds 2 at that region's diagonal (both orders, as the vertex matrix has them): the within-region count. Units: fibers (fractional when smoothed). A measurement over the partition: the "
                        "fibers are its elements, the region pairs its members.", "the fibers are summed in any order");
    s.plan = plan_arrays("partition_counts", {"end_vertex", "keep", "memberships"}, {"counts"}, {"keep"});
    s.prepare = [](const Values&, const Json&, ExecContext&) { return std::any(); };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) {
      const Array& m = ins.at("memberships");
      std::vector<std::uint8_t> keep;
      if (ins.count("keep")) for (double k : ins.at("keep").data) keep.push_back(k != 0 ? 1 : 0);
      const auto ends = indices(ins.at("end_vertex"));
      Arrays out;
      if (is_f32(m)) out["counts"] = to_coo<float>(connectome::partitionCounts<float>(ends, keep, from_coo<float>(m, "partition_counts")), true);
      else out["counts"] = to_coo<double>(connectome::partitionCounts<double>(ends, keep, from_coo<double>(m, "partition_counts")), false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── end_connectome ──
  {
    StepSpec s = simple("end_connectome", {"end_vertex", "keep"}, {"keep"}, {"fiber_counts"}, Json{{"n_vertices", nullptr}}, "nxr::connectome::endConnectome",
                        "The materialised vertex matrix of fiber counts [vertices x vertices] (COO, symmetric): each kept fiber adds 1 at its two end vertices in both orders; a fiber with both ends on one vertex leaves no entry. "
                        "Only on request (--over vertices --store): a partition result never forms it. Parameter n_vertices (the vertex count, no default).", "the fibers are summed in any order");
    s.plan = plan_arrays("end_connectome", {"end_vertex", "keep"}, {"fiber_counts"}, {"keep"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["n_vertices"].is_number_integer() && p["n_vertices"].get<int>() >= 1, "end_connectome: n_vertices is the end surface's vertex count, a whole number >= 1");
      return std::any(p["n_vertices"].get<std::int64_t>());
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const std::int64_t nVertices = std::any_cast<std::int64_t>(st);
      std::vector<std::uint8_t> keep;
      if (ins.count("keep")) for (double k : ins.at("keep").data) keep.push_back(k != 0 ? 1 : 0);
      const bool f32 = is_f32(ins.at("end_vertex"));
      Arrays out;
      if (f32) out["fiber_counts"] = to_coo<float>(connectome::endConnectome<float>(indices(ins.at("end_vertex")), keep, nVertices), true);
      else out["fiber_counts"] = to_coo<double>(connectome::endConnectome<double>(indices(ins.at("end_vertex")), keep, nVertices), false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── connection_weights ──
  {
    StepSpec s = simple("connection_weights", {"fiber_counts", "faces"}, {}, {"connection_weights"}, Json{{"smooth_hops", 3}}, "nxr::connectome::connectionWeights (sparse congruence: smoothing · counts · smoothingᵀ, nxr::algebra::sparseCongruence)",
                        "The connection weights: the fiber counts smoothed over the mesh, smoothing · counts · smoothingᵀ, the smoothing being the mesh's smoothing step raised to smooth_hops (hops, default 3, the reference's; 0: none), the "
                        "diagonal removed. Materialised vertex connectome only. Memory: about 37 non-zeros a row at 3 hops on a regular mesh; the smoothed matrix up to nF times the square of that.",
                        "a matrix product on the whole connectome");
    s.plan = plan_arrays("connection_weights", {"fiber_counts", "faces"}, {"connection_weights"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["smooth_hops"].is_number_integer() && p["smooth_hops"].get<int>() >= 0, "connection_weights: smooth_hops is a whole number >= 0");
      return std::any(p["smooth_hops"].get<int>());
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const int hops = std::any_cast<int>(st);
      const Array& c = ins.at("fiber_counts");
      const auto faces = indices(ins.at("faces"));
      Arrays out;
      if (is_f32(c)) out["connection_weights"] = to_coo<float>(connectome::connectionWeights<float>(from_coo<float>(c, "connection_weights"), faces, hops), true);
      else out["connection_weights"] = to_coo<double>(connectome::connectionWeights<double>(from_coo<double>(c, "connection_weights"), faces, hops), false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── prune_below ──
  {
    StepSpec s = simple("prune_below", {"connection_weights"}, {}, {"connection_weights"}, Json{{"prune_below", 1e-4}}, "nxr::connectome::pruneBelow",
                        "Weak connection weights removed from the pattern: entries below prune_below (a fraction of the strongest weight, default 1e-4, the reference's; 0: none). Materialised vertex connectome only.",
                        "a threshold on the whole matrix (its strongest entry is read first)");
    s.plan = plan_arrays("prune_below", {"connection_weights"}, {"connection_weights"});
    s.prepare = [](const Values&, const Json& p, ExecContext&) {
      need(p["prune_below"].is_number() && p["prune_below"].get<double>() >= 0, "prune_below: prune_below is a fraction of the strongest weight, >= 0");
      return std::any(p["prune_below"].get<double>());
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const double f = std::any_cast<double>(st);
      const Array& w = ins.at("connection_weights");
      Arrays out;
      if (is_f32(w)) out["connection_weights"] = to_coo<float>(connectome::pruneBelow<float>(from_coo<float>(w, "prune_below"), static_cast<float>(f)), true);
      else out["connection_weights"] = to_coo<double>(connectome::pruneBelow<double>(from_coo<double>(w, "prune_below"), f), false);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── largest_component ──
  {
    StepSpec s = simple("largest_component", {"connection_weights"}, {}, {"vertices", "label"}, Json::object(), "nxr::connectome::componentsOf, largestComponent",
                        "The connected components of the graph whose edges are the non-zero connection weights, over all the vertices (an isolated vertex is its own component), numbered by their lowest vertex: "
                        "label [vertices], and the vertices of the biggest (the lowest-numbered on a tie) → vertices. The connection weights themselves are not restricted to it. Materialised vertex connectome only.",
                        "a graph property of the whole connectome");
    s.plan = plan_arrays("largest_component", {"connection_weights"}, {"vertices", "label"});
    s.prepare = [](const Values&, const Json&, ExecContext&) { return std::any(); };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) {
      const Array& w = ins.at("connection_weights");
      const auto label = is_f32(w) ? connectome::componentsOf<float>(from_coo<float>(w, "largest_component")) : connectome::componentsOf<double>(from_coo<double>(w, "largest_component"));
      std::vector<std::int64_t> sizes;
      Array v = whole(connectome::largestComponent(label, &sizes));
      v.meta = Json{{"component_sizes", sizes}};
      return Arrays{{"vertices", v}, {"label", whole(label)}};
    };
    r.push_back(std::move(s));
  }
  (void)kWarnings;
}

}  // namespace nxr::pipelines::graph

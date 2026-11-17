// nxr-pipelines — THE MEASUREMENT VERBS AS REGISTERED STEPS: what the hand-wired `measure compute` (over a time tiling, over a partition of a surface) and `scaleselection compute` computed, as steps that read through
// the run's bound services and hand back described arrays the definitions store (declared outputs: kind measurement, selection, statistic). Their maths is the pure functions of compute/ (nxr::partition,
// nxr::extent, nxr::scale); the YAML definitions in pipelines/definitions/ (measure_over_time, measure, scaleselection) are the verbs.
//
//   window_measures    accumulate  a time-resolved field + the windows of a time tiling  ->  measured: [rows x windows] a measure (mean, mean_square, max, min, abs_max) and the sample count
//   partition_measures setup       a stored field (scalar, hermitian3) + a partition of its surface  ->  values: [members x measures], sizes: [members x (n_elements, area)]
//   scale_selection    setup       a recording and the stored pairs of cortical_event_pairs  ->  events (a spans Selection), scale (their measurements), links (a statistic)
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>

#include <Eigen/Dense>

#include "nxr/extent.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/io/finder.h"
#include "nxr/ladder.h"
#include "nxr/partition.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/scale.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using nxr::io::text;

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Value measurement_value(const Value& over, const std::string& measure, const std::vector<std::string>& manifolds) {
  Value v;
  v.cls = "measurement";
  v.manifolds = manifolds;
  v.facts = Json{{"measure", measure}, {"over", Json{{"selection", over.row ? text(*over.row, "name") : std::string("the partition")}}}};
  return v;
}
std::vector<std::string> strings_of(const Json& j, const std::string& what) {
  need(j.is_array(), what + " is a list of names");
  std::vector<std::string> out;
  for (const auto& x : j) { need(x.is_string(), what + " is a list of names"); out.push_back(x.get<std::string>()); }
  return out;
}

// ── window_measures ─────────────────────────────────────────────────────────────────────────────────────────────
// A TIME-RESOLVED FIELD MEASURED OVER THE WINDOWS OF A TIME TILING (the hand-wired `measure compute --over` it replaces): per row and window, the Studio's reduceTiles (nxr::partition::reduceTiles) folded across the chunks
// the chunk service hands over: each chunk split at the windows' edges, each piece reduced a row at a time, folded into its window (minima and maxima as the Studio keeps them, float32; sums in double). The span is
// the caller's. A window is measured only when the span covers it whole (a window cut by the span's edge is not the tiling's window): NaN outside the span and for a window it cuts. `count` (the samples a value is
// over: the roll-up weight) is always the last measure. Bad segments are not left out.
const std::vector<std::string> kTimeDefault = {"mean", "mean_square", "max", "min"};
const std::vector<std::string> kTimeKnown = {"mean", "mean_square", "max", "min", "abs_max"};

struct TimeState {
  std::vector<std::string> comps;             // the measures, then `count`
  std::int64_t width = 0, count = 0, samples = 0;
  std::string window_name;
  std::optional<std::string> unit;
};
struct TimeAcc {
  std::int64_t rows = 0, first = -1, stop = 0;
  std::vector<float> mn, mx;
  std::vector<double> sm, sq, ct;
};

StepSpec window_measures_step() {
  StepSpec s;
  s.name = "window_measures";
  s.kind = "accumulate";
  s.inputs = {"x", "over"};
  s.outputs = {"measured"};
  s.parameters = Json{{"measures", Json::array({"mean", "mean_square", "max", "min"})}};
  s.matches = "Studio backend/src/db/measurements.ts reduceTiles (compute nxr::partition::reduceTiles), folded chunk by chunk over the windows of a time tiling";
  s.description = "a time-resolved field reduced per row over the windows of a time tiling (a constant-Q ladder's level, or a closed-form spans tiling such as the chunk grid): mean, mean_square, max, min, abs_max, "
                  "and the samples a value is over (`count`, always last); NaN for a window the span does not cover whole";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::stateful, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::follows_input,
                                     "sums and sums of squares are added in double over a row's run and narrowed once (the Studio's float32 arrays); the window folded across chunks is the whole-recording arithmetic", {});
  s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
    const Value& x = ins.at("x");
    const Value& over = ins.at("over");
    need(x.cls == "field" && x.time && x.chunked, "window_measures: x must be a time-resolved field (… × time), inside the chunk loop: name the field with its time line");
    need(!x.complex && x.value_type == "scalar", "window_measures: x is " + std::string(x.complex ? "complex" : x.value_type) + "; measure a real scalar series");
    need(over.cls == "selection" && over.row, "window_measures: over is the time tiling (a selection)");
    const std::string type = text(*over.row, "type");
    need(type != "ladder", "window_measures: " + text(*over.row, "name") + " is a ladder: name one of its levels (<ladder name> L<k>)");
    need(type == "spans", "window_measures: " + text(*over.row, "name") + " is a " + type + ", not a tiling of time (spans)");
    need(!over.manifolds.empty() && over.manifolds[0] == *x.time, "window_measures: " + text(*over.row, "name") + " is not on x's time line");
    const Json pj = Json::parse(nxr::io::opt_text(*over.row, "params_json").value_or("{}"), nullptr, false);
    need(pj.is_object() && pj.contains("tiling"), "window_measures: " + text(*over.row, "name") + " has no closed-form tiling");
    const auto ms = strings_of(p["measures"], "window_measures: measures");
    need(!ms.empty(), "window_measures: measures names at least one of mean, mean_square, max, min, abs_max");
    for (const auto& m : ms) need(std::find(kTimeKnown.begin(), kTimeKnown.end(), m) != kTimeKnown.end(), "window_measures: measure '" + m + "': over time one of mean, mean_square, max, min, abs_max");
    (void)c;
    return Values{{"measured", measurement_value(over, "window_measures", {*x.time})}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    const Value& x = ins.at("x");
    const Value& over = ins.at("over");
    TimeState st;
    st.comps = strings_of(p["measures"], "window_measures: measures");
    st.comps.push_back("count");
    st.window_name = text(*over.row, "name");
    st.unit = x.unit;
    const Json pj = Json::parse(text(*over.row, "params_json"));
    const Json& tl = pj["tiling"];
    const auto line = ctx.bound.manifold(*x.time);
    need(line.has_value(), "window_measures: x's time line is not among the subject's rows");
    st.samples = nxr::io::integer(*line, "n_vertices");
    st.width = static_cast<std::int64_t>(std::llround(tl["width"].get<double>() * x.rate));
    const std::int64_t first = static_cast<std::int64_t>(std::llround((tl["origin"].get<double>() - x.origin) * x.rate));
    st.count = tl["count"].get<std::int64_t>();
    need(st.width >= 1 && std::llround(tl["hop"].get<double>() * x.rate) == st.width && first == 0,
         st.window_name + ": only contiguous windows from the line's start (hop = width, origin = the line's) are measured");
    return std::any(std::make_shared<TimeState>(std::move(st)));
  };
  s.start = [](const std::any&) { return std::any(TimeAcc{}); };
  s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
    const TimeState& st = *std::any_cast<const std::shared_ptr<TimeState>&>(st_);
    TimeAcc& acc = std::any_cast<TimeAcc&>(acc_);
    const Block& b = ins.at("x");
    const std::size_t W = static_cast<std::size_t>(st.count);
    if (acc.rows == 0) {
      acc.rows = b.rows;
      acc.mn.assign(static_cast<std::size_t>(b.rows) * W, std::numeric_limits<float>::infinity());
      acc.mx.assign(static_cast<std::size_t>(b.rows) * W, -std::numeric_limits<float>::infinity());
      acc.sm.assign(static_cast<std::size_t>(b.rows) * W, 0.0); acc.sq = acc.sm; acc.ct = acc.sm;
    } else if (b.rows != acc.rows) {
      refuse("window_measures: the rows of x changed between chunks");
    }
    if (acc.first < 0) acc.first = place.first;
    acc.stop = place.first + place.n;
    for (std::int64_t a = place.first; a < place.first + place.n;) {
      const std::int64_t k = a / st.width, e = std::min(place.first + place.n, (k + 1) * st.width), len = e - a;
      if (k >= st.count) break;
      std::vector<float> piece(static_cast<std::size_t>(len));
      for (std::int64_t r = 0; r < b.rows; ++r) {
        const float* row = b.re.data() + r * b.width + b.left + (a - place.first);
        std::copy(row, row + len, piece.begin());
        const auto t = nxr::partition::reduceTiles(piece.data(), 1, static_cast<std::size_t>(len), static_cast<std::size_t>(len), 1);
        const std::size_t i = static_cast<std::size_t>(r) * W + static_cast<std::size_t>(k);
        acc.mn[i] = std::min(acc.mn[i], t.min[0]);
        acc.mx[i] = std::max(acc.mx[i], t.max[0]);
        acc.sm[i] += t.sum[0];
        acc.sq[i] += t.sumSq[0];
        acc.ct[i] += t.count[0];
      }
      a = e;
    }
  };
  s.finish = [](const std::any& st_, std::any& acc_) {
    const TimeState& st = *std::any_cast<const std::shared_ptr<TimeState>&>(st_);
    TimeAcc& acc = std::any_cast<TimeAcc&>(acc_);
    if (acc.first < 0) refuse("window_measures: the span holds no sample");
    const std::size_t W = static_cast<std::size_t>(st.count), C = st.comps.size(), R = static_cast<std::size_t>(acc.rows);
    const std::int64_t k0 = acc.first / st.width, k1 = std::min<std::int64_t>(st.count - 1, (acc.stop - 1) / st.width);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Array y;
    y.rows = static_cast<std::int64_t>(R * W); y.cols = static_cast<std::int64_t>(C);
    y.data.assign(R * W * C, static_cast<double>(nan));
    std::int64_t w0 = -1, w1 = -1;                                         // the windows measured whole
    for (std::size_t r = 0; r < R; ++r)
      for (std::int64_t k = k0; k <= k1; ++k) {
        const std::size_t i = r * W + static_cast<std::size_t>(k);
        const double n = acc.ct[i];
        if (!(n > 0) || n != static_cast<double>(std::min(st.width, st.samples - k * st.width))) continue;
        if (w0 < 0) w0 = k;
        w1 = k;
        for (std::size_t c = 0; c < C; ++c) {
          const std::string& m = st.comps[c];
          const float v = m == "mean" ? static_cast<float>(acc.sm[i] / n) : m == "mean_square" ? static_cast<float>(acc.sq[i] / n)
                        : m == "max" ? acc.mx[i] : m == "min" ? acc.mn[i] : m == "abs_max" ? std::max(std::fabs(acc.mn[i]), std::fabs(acc.mx[i])) : static_cast<float>(n);
          y.data[i * C + c] = static_cast<double>(v);
        }
      }
    Json ms = Json::array();
    for (std::size_t c = 0; c < C; ++c) {
      Json one{{"measure", st.comps[c]}};
      if (st.comps[c] == "mean_square" && st.unit) one["unit"] = "(" + *st.unit + ")^2";
      else if (st.comps[c] != "count" && st.unit) one["unit"] = *st.unit;
      ms.push_back(one);
    }
    y.meta["describe"] = Json{{"measures", ms}, {"values_shape", Json::array({static_cast<std::int64_t>(R), static_cast<std::int64_t>(W)})}, {"values_f32", true}, {"n_members", st.count},
                              {"description", "per row and window of " + st.window_name + ": " + std::to_string(C - 1) + " measures and the sample count"},
                              {"windows_measured", w0 < 0 ? Json(nullptr) : Json::array({w0, w1})}};
    return Arrays{{"measured", std::move(y)}};
  };
  return s;
}

// ── partition_measures ──────────────────────────────────────────────────────────────────────────────────────────
// A STORED FIELD MEASURED OVER A PARTITION OF ITS SURFACE (the hand-wired `measure compute` it replaces): the partition's members (a `set` Selection with a code per vertex, a parcellation or a level of a tile
// ladder) get the field's measures (nxr::partition, the Studio's measureOverPartition) as additive ingredients, and the members' own sizes (n_elements, and `area` when the weights are areas).
//
//   scalar field      mean · max · min · mean_square · abs_max · sum (the defaults: the first five) and median (exact, unweighted: only with weight none); `gradient` adds the registry's gradient_* sums in the canonical frames
//   hermitian3 field  the registry's tensor_* sums in the canonical frames
// The WEIGHT is stated, never assumed: none (1 a vertex: the plain vertex mean), barycentric (a third of each incident triangle), voronoi (the mixed Voronoi area of the mollified input). They differ by tens of percent per
// region on a real cortex. A member with no value of a measure holds NaN (the hand-wired verb wrote no row).
const std::vector<std::string> kTensor = {"tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im", "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total"};
const std::vector<std::string> kGrad = {"gradient_weight", "gradient_1", "gradient_2", "gradient_n", "gradient_11", "gradient_22", "gradient_12", "gradient_magnitude"};

std::optional<std::string> unit_of(const std::string& measure, const std::optional<std::string>& u, bool areas) {
  if (measure == "area") return std::string("m\xc2\xb2");
  if (measure == "n_elements") return std::nullopt;
  if (!u || u->empty()) return std::nullopt;
  if (measure == "mean_square") return "(" + *u + ")\xc2\xb2";
  if (measure == "sum") return areas ? *u + "\xc2\xb7m\xc2\xb2" : *u;
  return u;
}

struct PartitionState { Value x, labelling; Json p; };

StepSpec partition_measures_step() {
  StepSpec s;
  s.name = "partition_measures";
  s.kind = "setup";
  s.inputs = {"x", "labelling"};
  s.outputs = {"values", "sizes"};
  s.parameters = Json{{"weight", nullptr}, {"measures", nullptr}, {"gradient", false}};
  s.matches = "Studio measureOverPartition (compute nxr::partition::accumulate, medians, measure); nxr measure.vectors.vertex_gradient and the registry's gradient_* / tensor_* sums";
  s.description = "a stored scalar or hermitian3 field measured over the members of a partition of its surface, the weight stated (none, barycentric, voronoi); and the members' own sizes";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from a stored field");
  s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
    const Value& x = ins.at("x");
    const Value& lab = ins.at("labelling");
    need(p["weight"].is_string() && (p["weight"] == "none" || p["weight"] == "barycentric" || p["weight"] == "voronoi"),
         "partition_measures: weight is one of none, barycentric, voronoi (it changes the numbers, so it is always stated)");
    need(x.cls == "field" && x.row && !x.chunked && x.manifolds.size() == 1 && !x.time, "partition_measures: x is a stored field on a surface (no time line): name it with the field parameter");
    need(x.value_type == "scalar" || x.value_type == "hermitian3", "partition_measures: " + text(*x.row, "name") + " is a " + x.value_type + " field (scalar and hermitian3 are measured)");
    need(!x.complex, "partition_measures: " + text(*x.row, "name") + " is complex (scalar and hermitian3 are measured)");
    need(lab.cls == "selection" && lab.row && text(*lab.row, "type") == "set" && !nxr::io::is_null(*lab.row, "array_path"), "partition_measures: the labelling is a set Selection with a code per vertex");
    need(!lab.manifolds.empty() && lab.manifolds[0] == x.manifolds[0], "partition_measures: the labelling is not on the surface of " + text(*x.row, "name"));
    need(c.bound.manifold_type(x.manifolds[0]) == "surface", "partition_measures: " + text(*x.row, "name") + " does not lie on a surface");
    if (!p["measures"].is_null()) {
      const auto ms = strings_of(p["measures"], "partition_measures: measures");
      for (const auto& m : ms)
        need(m == "median" || (nxr::partition::known_measure(m) && m != "area" && m != "n_elements"), "partition_measures: measure " + m + " is not one this producer writes per field (mean, mean_square, max, min, abs_max, sum, median)");
      if (std::find(ms.begin(), ms.end(), "median") != ms.end())
        need(p["weight"] == "none", "partition_measures: median: a weighted median is not built; the median is exact and unweighted, so it needs weight none");
    }
    return Values{{"values", measurement_value(lab, "partition_measures", x.manifolds)}, {"sizes", measurement_value(lab, "members_own", x.manifolds)}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(std::make_shared<PartitionState>(PartitionState{ins.at("x"), ins.at("labelling"), p})); };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const PartitionState& st = *std::any_cast<const std::shared_ptr<PartitionState>&>(state);
    const Json& p = st.p;
    const std::string weight = p["weight"].get<std::string>(), fname = text(*st.x.row, "name");
    auto say_warn = [&](const std::string& w) { if (ctx.say) ctx.say(Json{{"stage", "warning"}, {"warning", w}}); };
    std::vector<std::string> measures = p["measures"].is_null() ? nxr::partition::field_measures() : strings_of(p["measures"], "partition_measures: measures");
    const bool want_median = std::find(measures.begin(), measures.end(), "median") != measures.end();
    // THE PARTITION and its surface
    const auto man = ctx.bound.manifold(st.x.manifolds[0]);
    const SurfaceInput S = ctx.bound.surface(text(*man, "name"), true);
    const std::int64_t nV = S.n_vertices();
    std::vector<std::int32_t> codes = ctx.bound.array_int32(text(*st.labelling.row, "array_path"));
    need(static_cast<std::int64_t>(codes.size()) == nV, text(*st.labelling.row, "name") + ": a partition has a code per vertex (" + std::to_string(nV) + "), this one has " + std::to_string(codes.size()));
    const int members = codes.empty() ? 0 : *std::max_element(codes.begin(), codes.end()) + 1;
    need(members > 0, text(*st.labelling.row, "name") + ": the partition has no member");
    // THE WEIGHTS
    std::vector<double> w;
    if (weight == "barycentric") {
      nxr::ladder::Mesh m; m.positions = S.vertices; m.faces = S.faces; m.n = static_cast<int>(nV);
      w = nxr::ladder::vertexAreas(m);
    } else if (weight == "voronoi") {
      nxr::manifold::IntrinsicSurface isurf(S.vertices.data(), static_cast<int>(nV), S.faces.data(), static_cast<int>(S.n_faces()));
      w = isurf.inputMixedVoronoiMass();                                   // the mollified input's mixed-Voronoi mass
    }
    const bool areas = !w.empty();
    const double* wp = areas ? w.data() : nullptr;
    // THE MEMBERS themselves (geometric: no field)
    const nxr::partition::Sums geo = nxr::partition::accumulate(codes.data(), codes.size(), members, nullptr, wp);
    if (geo.unassigned) say_warn(std::to_string(geo.unassigned) + " vertices have a negative code: in no member, left out");
    std::int64_t empty = 0;
    for (int k = 0; k < members; ++k) if (geo.count[static_cast<std::size_t>(k)] == 0) ++empty;
    if (empty) say_warn(std::to_string(empty) + " codes below " + std::to_string(members) + " have no vertex: their means are written as 0, as the reference does");
    std::vector<std::string> geo_ms = {"n_elements"};
    if (areas) geo_ms.push_back("area");
    Array sz;
    sz.rows = members; sz.cols = static_cast<std::int64_t>(geo_ms.size());
    sz.data.assign(static_cast<std::size_t>(sz.rows * sz.cols), kNaN);
    Json szm = Json::array();
    for (std::size_t c = 0; c < geo_ms.size(); ++c) {
      for (int k = 0; k < members; ++k) sz.data[static_cast<std::size_t>(k) * geo_ms.size() + c] = nxr::partition::measure(geo, k, geo_ms[c]);
      const auto u = unit_of(geo_ms[c], std::nullopt, areas);
      szm.push_back(u ? Json{{"measure", geo_ms[c]}, {"unit", *u}} : Json{{"measure", geo_ms[c]}});
    }
    sz.meta["describe"] = Json{{"measures", szm}, {"n_members", members}, {"description", "the size of each member of " + text(*st.labelling.row, "name") + " (weights: " + weight + ")"}};

    std::vector<std::string> cols;
    Json colj = Json::array();
    std::vector<double> data;                                              // [members x cols]
    const std::optional<std::string> funit = st.x.unit;
    std::optional<StoredFrames> frames;
    auto canonical = [&](const char* what) {
      frames = ctx.bound.stored_frames(S.name, nV);
      need(frames->canonical, S.name + " has no canonical frames: " + what + " is measured in them (`nxr-neuro run frames` first)");
    };
    if (st.x.value_type == "hermitian3") {
      // A TENSOR field ([nV x 3 x 3 x 2]): per member, the area-weighted sums of its components in the canonical frames, P = E T E^T with E's rows e1 (north), e2 (west), n = e1 x e2: 11 22 nn, 12 1n 2n
      // (real and imaginary), and the total (the trace, over EVERY vertex of the member; the components over the frames' regular vertices only)
      canonical("a tensor");
      std::vector<std::int64_t> shape;
      const std::vector<float> T = ctx.bound.array_float(text(*st.x.row, "path"), &shape);
      need(static_cast<std::int64_t>(T.size()) == nV * 18, fname + ": a hermitian3 map is [" + std::to_string(nV) + " x 3 x 3 x 2]");
      std::vector<std::array<double, 10>> sums(static_cast<std::size_t>(members));
      for (auto& a : sums) a.fill(0.0);
      std::vector<double> mw(static_cast<std::size_t>(members), 0.0);
      std::int64_t singular = 0;
      for (std::int64_t v = 0; v < nV; ++v) {
        const int k = codes[static_cast<std::size_t>(v)];
        if (k < 0) continue;
        const double wv = wp ? wp[v] : 1.0;
        auto re = [&](int i, int j) { return static_cast<double>(T[static_cast<std::size_t>(18 * v + 2 * (3 * i + j))]); };
        auto im = [&](int i, int j) { return static_cast<double>(T[static_cast<std::size_t>(18 * v + 2 * (3 * i + j) + 1)]); };
        mw[static_cast<std::size_t>(k)] += wv;
        sums[static_cast<std::size_t>(k)][9] += wv * (re(0, 0) + re(1, 1) + re(2, 2));
        const Eigen::Vector3d e1(frames->e1(v, 0), frames->e1(v, 1), frames->e1(v, 2)), e2(frames->e2(v, 0), frames->e2(v, 1), frames->e2(v, 2));
        if (!e1.allFinite() || !e2.allFinite()) { ++singular; continue; }
        const Eigen::Vector3d n = e1.cross(e2);
        const Eigen::Vector3d E[3] = {e1, e2, n};
        auto P = [&](int a, int b, bool imag) { double x = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) x += E[a][i] * (imag ? im(i, j) : re(i, j)) * E[b][j]; return x; };
        const double c[9] = {P(0, 0, false), P(1, 1, false), P(2, 2, false), P(0, 1, false), P(0, 1, true), P(0, 2, false), P(0, 2, true), P(1, 2, false), P(1, 2, true)};
        for (int i = 0; i < 9; ++i) sums[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] += wv * c[i];
      }
      if (singular) say_warn(fname + ": " + std::to_string(singular) + " vertices where the canonical frames are singular: in tensor_total, not in the components");
      cols = kTensor;
      data.assign(static_cast<std::size_t>(members) * cols.size(), kNaN);
      for (int k = 0; k < members; ++k)
        if (mw[static_cast<std::size_t>(k)] > 0) for (std::size_t i = 0; i < cols.size(); ++i) data[static_cast<std::size_t>(k) * cols.size() + i] = sums[static_cast<std::size_t>(k)][i];
      const auto u = unit_of("sum", funit, areas);
      for (const auto& c : cols) colj.push_back(u ? Json{{"measure", c}, {"unit", *u}} : Json{{"measure", c}});
    } else {
      std::vector<std::int64_t> shape;
      const std::vector<double> x = ctx.bound.array_double(text(*st.x.row, "path"), &shape);
      need(shape.size() == 1 && static_cast<std::int64_t>(x.size()) == nV, fname + ": a value per vertex is [" + std::to_string(nV) + "], this array has another shape");
      const nxr::partition::Sums sm = nxr::partition::accumulate(codes.data(), codes.size(), members, x.data(), wp);
      const std::vector<double> med = want_median ? nxr::partition::medians(codes.data(), codes.size(), members, x.data()) : std::vector<double>();
      std::int64_t bad = 0;
      for (auto c : sm.nonfinite) bad += c;
      if (bad) say_warn(fname + ": " + std::to_string(bad) + " non-finite values; the members holding them are not written");
      cols = measures;
      std::vector<std::vector<double>> columns;
      for (const auto& m : measures) {
        std::vector<double> col(static_cast<std::size_t>(members), kNaN);
        for (int k = 0; k < members; ++k) col[static_cast<std::size_t>(k)] = m == "median" ? med[static_cast<std::size_t>(k)] : nxr::partition::measure(sm, k, m);
        columns.push_back(std::move(col));
        const auto u = unit_of(m, funit, areas);
        colj.push_back(u ? Json{{"measure", m}, {"unit", *u}} : Json{{"measure", m}});
      }
      if (p.value("gradient", false)) {
        // THE GRADIENT of a scalar field (nxr's vertex_gradient and vector_stats): per member, over the frames' regular vertices, the registry's gradient_weight Σa, gradient_1 · gradient_2 · gradient_n Σa·(∇x)_c,
        // gradient_11 · gradient_22 · gradient_12 Σa·(∇x)_c (∇x)_c', gradient_magnitude Σa·|∇x|_tangent
        canonical("a gradient");
        std::vector<Eigen::Vector3d> acc(static_cast<std::size_t>(nV), Eigen::Vector3d::Zero());
        std::vector<double> wsum(static_cast<std::size_t>(nV), 0.0);
        auto pos = [&](std::int32_t v) { return Eigen::Vector3d(S.vertices[static_cast<std::size_t>(3 * v)], S.vertices[static_cast<std::size_t>(3 * v + 1)], S.vertices[static_cast<std::size_t>(3 * v + 2)]); };
        for (std::int64_t f = 0; f < S.n_faces(); ++f) {
          const std::int32_t i[3] = {S.faces[static_cast<std::size_t>(3 * f)], S.faces[static_cast<std::size_t>(3 * f + 1)], S.faces[static_cast<std::size_t>(3 * f + 2)]};
          const Eigen::Vector3d p0 = pos(i[0]), p1 = pos(i[1]), p2 = pos(i[2]);
          const Eigen::Vector3d nn = (p1 - p0).cross(p2 - p0);
          const double A2 = nn.norm();
          if (!(A2 > 0)) continue;
          const Eigen::Vector3d nh = nn / A2;
          // ∇φ_k = n × (the edge opposite k) / (2A)
          const Eigen::Vector3d g = (nh.cross(p2 - p1) * x[static_cast<std::size_t>(i[0])] + nh.cross(p0 - p2) * x[static_cast<std::size_t>(i[1])] + nh.cross(p1 - p0) * x[static_cast<std::size_t>(i[2])]) / A2;
          for (int k = 0; k < 3; ++k) { acc[static_cast<std::size_t>(i[k])] += g * (A2 / 2); wsum[static_cast<std::size_t>(i[k])] += A2 / 2; }
        }
        std::vector<std::array<double, 8>> gs(static_cast<std::size_t>(members));
        for (auto& a : gs) a.fill(0.0);
        std::int64_t singular = 0;
        for (std::int64_t v = 0; v < nV; ++v) {
          const int k = codes[static_cast<std::size_t>(v)];
          if (k < 0) continue;
          const Eigen::Vector3d e1(frames->e1(v, 0), frames->e1(v, 1), frames->e1(v, 2)), e2(frames->e2(v, 0), frames->e2(v, 1), frames->e2(v, 2));
          if (!e1.allFinite() || !e2.allFinite() || !(wsum[static_cast<std::size_t>(v)] > 0)) { ++singular; continue; }
          const Eigen::Vector3d n = e1.cross(e2);
          Eigen::Vector3d gv = acc[static_cast<std::size_t>(v)] / wsum[static_cast<std::size_t>(v)];
          gv -= gv.dot(n) * n;                                             // on the vertex's tangent plane
          const double c1 = gv.dot(e1), c2 = gv.dot(e2), cn = gv.dot(n), a = wp ? wp[v] : 1.0;
          auto& g = gs[static_cast<std::size_t>(k)];
          g[0] += a; g[1] += a * c1; g[2] += a * c2; g[3] += a * cn; g[4] += a * c1 * c1; g[5] += a * c2 * c2; g[6] += a * c1 * c2; g[7] += a * std::hypot(c1, c2);
        }
        if (singular) say_warn(fname + " (its gradient): " + std::to_string(singular) + " vertices where the canonical frames are singular, left out");
        const std::string u = funit.value_or("");
        for (std::size_t i = 0; i < kGrad.size(); ++i) {
          std::vector<double> col(static_cast<std::size_t>(members), kNaN);
          for (int k = 0; k < members; ++k) if (gs[static_cast<std::size_t>(k)][0] > 0) col[static_cast<std::size_t>(k)] = gs[static_cast<std::size_t>(k)][i];
          columns.push_back(std::move(col));
          cols.push_back(kGrad[i]);
          std::string unit;
          if (i == 0) unit = areas ? "m\xc2\xb2" : "";
          else if (!u.empty()) unit = (i >= 4 && i <= 6) ? "(" + u + "/m)\xc2\xb2" + (areas ? "\xc2\xb7m\xc2\xb2" : "") : "(" + u + "/m)" + (areas ? "\xc2\xb7m\xc2\xb2" : "");
          colj.push_back(unit.empty() ? Json{{"measure", kGrad[i]}} : Json{{"measure", kGrad[i]}, {"unit", unit}});
        }
      }
      data.assign(static_cast<std::size_t>(members) * cols.size(), kNaN);
      for (std::size_t c = 0; c < cols.size(); ++c) for (int k = 0; k < members; ++k) data[static_cast<std::size_t>(k) * cols.size() + c] = columns[c][static_cast<std::size_t>(k)];
    }
    Array vals;
    vals.rows = members; vals.cols = static_cast<std::int64_t>(cols.size());
    vals.data = std::move(data);
    vals.meta["describe"] = Json{{"measures", colj}, {"n_members", members},
                                 {"description", fname + " over the members of " + text(*st.labelling.row, "name") + " (weights: " + weight + ")"}};
    return Arrays{{"values", std::move(vals)}, {"sizes", std::move(sz)}};
  };
  return s;
}

// ── scale_selection ─────────────────────────────────────────────────────────────────────────────────────────────
// EACH EVENT'S OWN SIZE AND RHYTHM across a declared set of (spatial level, time level) pairs (the hand-wired `scaleselection compute` it replaces; spec docs/specs/2026-11-13-cortical-events.md, "Phase 2e"). From STORED
// rows only, no recording pass: what `run cortical_event_pairs` stored for every pair of the product `space_levels` x `time_levels` (the cells' amplitude on the pair's joint Selection, the cells' per-tile baseline, the
// labelling of the pair's events on the cells), read by the runner (Bound::scale_inputs). z = (amplitude - median) / SD per tile, the standardization the extents were found on (nxr::extent::standardize); events of
// different pairs are linked by the ladders' exact nesting in space and overlap in time (nxr::scale::selectScale); the connected components are the cross-pair events: each one's response at every pair (its footprint),
// best pair, the response there.
//
//   events   a spans Selection of the cross-pair events on the recording's time line (their span in time; one event per cross-pair event)
//   scale    measurements of the recording over it, one value an event: scale_space_level, scale_time_level, scale_peak, scale_tile_area (m2), scale_band_lo, scale_band_hi (Hz), scale_pairs, scale_edge (a bit mask),
//            and the footprint scale_response_s<space level>_t<time level>, a column a pair (NaN where the event has no member at that pair)
//   links    a statistic of the recording [pairs x events of the pair]: the cross-pair event of each pair's event (NaN beyond a pair's events)
struct ScaleState {
  Value recording;
  Json p;
  ScaleInputs in;
  Json pairs_json = Json::array();
};

StepSpec scale_selection_step() {
  StepSpec s;
  s.name = "scale_selection";
  s.kind = "setup";
  s.inputs = {"recording"};
  s.outputs = {"events", "scale", "links"};
  s.parameters = Json{{"space_levels", nullptr}, {"time_levels", nullptr}, {"source", "cortical_event_pairs"}, {"response", "peak"}, {"min_link", 1}, {"surface", nullptr}};
  s.matches = "Rheome detect.blobscale (the scale of the largest normalised response); compute nxr::scale::selectScale, nxr::extent::standardize";
  s.description = "each event's own size and rhythm across the product of two lists of levels, from the stored cells, baselines and labellings of the pairs: the cross-pair events, each one's footprint and best pair";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from stored cells");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& rec = ins.at("recording");
    need(rec.cls == "field" && rec.row && rec.time && rec.rate > 0, "scaleselection: the recording has no time line");
    need(p["space_levels"].is_array() && !p["space_levels"].empty(), "scaleselection: space_levels lists the surface ladder's levels (e.g. [4, 5])");
    need(p["time_levels"].is_array() && !p["time_levels"].empty(), "scaleselection: time_levels lists the recording's constant-Q levels (the pair set is the product of the two lists)");
    for (const char* key : {"space_levels", "time_levels"}) {
      std::set<int> seen;
      for (const auto& x : p[key]) {
        need(x.is_number_integer() && x.get<int>() >= 0, "scaleselection: levels are 0, 1, 2, \xe2\x80\xa6");
        need(seen.insert(x.get<int>()).second, "scaleselection: level " + std::to_string(x.get<int>()) + " is listed twice");
      }
    }
    try { (void)nxr::scale::parseResponse(p["response"].get<std::string>()); } catch (const std::invalid_argument& e) { refuse(std::string("scaleselection: ") + e.what()); }
    need(p["min_link"].is_number_integer() && p["min_link"].get<int>() >= 1, "scaleselection: min_link is at least 1");
    Value ev;
    ev.cls = "selection";
    ev.manifolds = {*rec.time};
    Value links;
    links.cls = "statistic";
    return Values{{"events", ev}, {"scale", measurement_value(ev, "scale_selection", {*rec.time})}, {"links", links}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    auto st = std::make_shared<ScaleState>();
    st->recording = ins.at("recording");
    st->p = p;
    std::vector<int> sl, tl;
    for (const auto& x : p["space_levels"]) sl.push_back(x.get<int>());
    for (const auto& x : p["time_levels"]) tl.push_back(x.get<int>());
    st->in = ctx.bound.scale_inputs(text(*st->recording.row, "id"), text(*st->recording.row, "name"), sl, tl, p["source"].get<std::string>(),
                                    p["surface"].is_string() ? std::optional<std::string>(p["surface"].get<std::string>()) : std::nullopt);
    return std::any(st);
  };
  s.record = [](const std::any& state) {
    const ScaleState& st = *std::any_cast<const std::shared_ptr<ScaleState>&>(state);
    Json pairs = Json::array();
    for (const auto& q : st.in.pairs)
      pairs.push_back(Json{{"pair", static_cast<std::int64_t>(pairs.size())}, {"space_level", q.space_level}, {"time_level", q.time_level}, {"window_samples", q.window}, {"band_hz", q.joint["band"]}, {"tiles", q.tiles},
                           {"time_tiles", q.frames}, {"events", q.events}, {"joint", Json{{"id", q.joint_id}, {"name", q.joint_name}}}, {"labelling", Json{{"id", q.labelling_id}, {"name", q.labelling_name}}}, {"baseline", q.baseline_name}});
    return Json{{"scale_pairs", pairs}, {"surface", st.in.surface}};
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const ScaleState& st = *std::any_cast<const std::shared_ptr<ScaleState>&>(state);
    const Json& p = st.p;
    const auto& pr = st.in.pairs;
    const std::int64_t P = static_cast<std::int64_t>(pr.size());
    const double dt = 1.0 / st.recording.rate, t0 = st.recording.origin;
    // each pair: the tiles' area, z = (amplitude - median) / SD per tile
    const SurfaceInput S = ctx.bound.surface(st.in.surface, true);
    std::vector<nxr::scale::Pair> pairs;
    std::vector<double> tile_area;
    Json pairs_json = Json::array();
    std::set<int> levels_seen;
    std::set<std::int64_t> windows_seen;
    for (const ScalePairInput& q : pr) {
      const SurfaceLevel SL = ctx.bound.surface_level(S, q.space_level);
      need(SL.nodes == q.tiles, "scaleselection: level " + std::to_string(q.space_level) + " of " + st.in.surface + " has " + std::to_string(SL.nodes) + " tiles, the joint Selection '" + q.joint_name + "' records " + std::to_string(q.tiles));
      double area = 0;
      for (const double a : SL.tile_area) area += a;
      tile_area.push_back(area / static_cast<double>(SL.nodes));
      Eigen::MatrixXd amp(q.tiles, q.frames);
      for (std::int64_t a = 0; a < q.tiles; ++a) for (std::int64_t f = 0; f < q.frames; ++f) amp(a, f) = q.amplitude[static_cast<std::size_t>(a * q.frames + f)];
      Eigen::VectorXd med(q.tiles), sd(q.tiles);
      for (std::int64_t a = 0; a < q.tiles; ++a) { med(a) = q.median[static_cast<std::size_t>(a)]; sd(a) = q.sd[static_cast<std::size_t>(a)]; }
      nxr::scale::Pair sp;
      sp.level = q.space_level; sp.window = q.window; sp.origin = 0; sp.events = static_cast<int>(q.events);
      sp.z = nxr::extent::standardize(amp, med, sd);
      sp.label.resize(q.tiles, q.frames);
      for (std::int64_t a = 0; a < q.tiles; ++a) for (std::int64_t f = 0; f < q.frames; ++f) sp.label(a, f) = q.labels[static_cast<std::size_t>(a * q.frames + f)];
      pairs.push_back(std::move(sp));
      levels_seen.insert(q.space_level);
      windows_seen.insert(q.window);
      pairs_json.push_back(Json{{"pair", static_cast<std::int64_t>(pairs_json.size())}, {"space_level", q.space_level}, {"time_level", q.time_level}, {"window_samples", q.window}, {"band_hz", q.joint["band"]},
                                {"tiles", q.tiles}, {"time_tiles", q.frames}, {"tile_area_m2", tile_area.back()}, {"events", q.events}, {"labelling", q.labelling_name}, {"baseline", q.baseline_name}});
    }
    {   // a pair's time tiles are the joint's: two pairs of one window and one level would be the same pair
      std::set<std::pair<int, std::int64_t>> uniq;
      for (const ScalePairInput& q : pr) need(uniq.insert({q.space_level, q.window}).second, "scaleselection: two time levels have time tiles of " + std::to_string(q.window) + " samples (the same pair twice)");
    }
    const nxr::scale::Response response = nxr::scale::parseResponse(p["response"].get<std::string>());
    nxr::scale::Selection sel;
    try { sel = nxr::scale::selectScale(pairs, response, p["min_link"].get<int>()); } catch (const std::invalid_argument& e) { refuse(std::string("scaleselection: ") + e.what()); }
    const std::int64_t E = sel.events;
    need(E > 0, "scaleselection: no event in any pair: nothing is stored");
    const int min_level = *levels_seen.begin(), max_level = *levels_seen.rbegin();
    const std::int64_t min_window = *windows_seen.begin(), max_window = *windows_seen.rbegin();
    const Json warnings = Json::array({
        "the response is a STANDARDIZED amplitude (z against each tile's own history at that pair), comparable across pairs as such; a pair with more cells has more chances for a large noise z, so the peak is selection-biased towards fine levels and short windows (the `mean` response is the less biased alternative)",
        "the size and rhythm are the best pair of the DECLARED set, quantized to its levels and bands, not a fitted scale; a selection on the set's edge (scale_edge) may lie beyond it",
        "a coarse level's footprint is wide: its tiles hold the pattern of a neighbourhood (phase 2a); an event is placed to within the level's scale"});
    // THE EVENTS' SPANS and their measurements [events x (8 + pairs)]
    constexpr int K = 8;
    const std::size_t Cn = static_cast<std::size_t>(K + P);
    Array ev, sc, lk;
    Json spans = Json::array();
    sc.rows = E; sc.cols = static_cast<std::int64_t>(Cn);
    sc.data.assign(static_cast<std::size_t>(E) * Cn, kNaN);
    for (std::int64_t k = 0; k < E; ++k) {
      const int b = sel.best(k);
      double* e = sc.data.data() + static_cast<std::size_t>(k) * Cn;
      int edge = 0;
      if (b >= 0) {
        const ScalePairInput& q = pr[static_cast<std::size_t>(b)];
        if (levels_seen.size() >= 2) { if (q.space_level == max_level) edge |= 1; if (q.space_level == min_level) edge |= 2; }
        if (windows_seen.size() >= 2) { if (q.window == min_window) edge |= 4; if (q.window == max_window) edge |= 8; }
        e[0] = q.space_level; e[1] = q.time_level; e[2] = sel.peak(k); e[3] = tile_area[static_cast<std::size_t>(b)]; e[4] = q.joint["band"][0].get<double>(); e[5] = q.joint["band"][1].get<double>();
      }
      e[6] = sel.pairs(k); e[7] = edge;
      for (std::int64_t a = 0; a < P; ++a) e[K + a] = sel.response(a, k);
      spans.push_back(Json{{"start", t0 + sel.start(k) * dt}, {"stop", t0 + sel.stop(k) * dt}});
    }
    Json ms = Json::array({Json{{"measure", "scale_space_level"}}, Json{{"measure", "scale_time_level"}}, Json{{"measure", "scale_peak"}}, Json{{"measure", "scale_tile_area"}, {"unit", "m\xc2\xb2"}},
                           Json{{"measure", "scale_band_lo"}, {"unit", "Hz"}}, Json{{"measure", "scale_band_hi"}, {"unit", "Hz"}}, Json{{"measure", "scale_pairs"}}, Json{{"measure", "scale_edge"}}});
    for (const ScalePairInput& q : pr) ms.push_back(Json{{"measure", "scale_response_s" + std::to_string(q.space_level) + "_t" + std::to_string(q.time_level)}});
    const Json params = Json{{"scale", Json{{"pairs", pairs_json}, {"response", p["response"]}, {"min_link", p["min_link"]}, {"source", p["source"]}, {"order", "space level ascending, then time level ascending"}, {"events", E}, {"warnings", warnings}}}};
    const std::string desc = "the cross-pair events of " + text(*st.recording.row, "name") + " (one event seen at several level pairs, matched by the ladders' nesting): each one's span in time; its size and rhythm are the measurements scale_*";
    ev.rows = E; ev.cols = 1;
    ev.meta["describe"] = Json{{"spans", spans}, {"params", params}, {"description", desc}};
    sc.meta["describe"] = Json{{"measures", ms}, {"n_members", E}, {"description", "per cross-pair event: its best pair's size and rhythm, the pairs that hold it, and its response at every pair"}};
    // THE LINKS: the cross-pair event of each pair's event
    std::int64_t maxe = 0;
    for (const ScalePairInput& q : pr) maxe = std::max(maxe, q.events);
    lk.rows = P; lk.cols = maxe;
    lk.data.assign(static_cast<std::size_t>(P * maxe), kNaN);
    for (std::int64_t a = 0; a < P; ++a)
      for (std::int64_t e = 0; e < pr[static_cast<std::size_t>(a)].events; ++e) lk.data[static_cast<std::size_t>(a * maxe + e)] = sel.cross[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)];
    Json prs = Json::array();
    for (const ScalePairInput& q : pr) prs.push_back(Json{{"name", "s" + std::to_string(q.space_level) + "_t" + std::to_string(q.time_level)}, {"labelling", q.labelling_name}});
    lk.meta["describe"] = Json{{"measure", "scale_event"}, {"method", "scale_selection"}, {"precision", "float64"}, {"n_members", E},
                               {"axes", Json::array({Json{{"name", "pair"}, {"size", P}, {"values", prs}}, Json{{"name", "event"}, {"size", maxe}}})},
                               {"options", Json{{"response", p["response"]}, {"min_link", p["min_link"]}, {"source", p["source"]}}}, {"producer", Json{{"pairs", pairs_json}, {"warnings", warnings}}},
                               {"description", "for each pair's event (a labelling's member), the cross-pair event it belongs to; NaN beyond the pair's events"}};
    return Arrays{{"events", std::move(ev)}, {"scale", std::move(sc)}, {"links", std::move(lk)}};
  };
  return s;
}

}  // namespace

void append_measure_steps(std::vector<StepSpec>& r) {
  r.push_back(window_measures_step());
  r.push_back(partition_measures_step());
  r.push_back(scale_selection_step());
}

}  // namespace nxr::pipelines::graph

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

}  // namespace

void append_measure_steps(std::vector<StepSpec>& r) {
  r.push_back(window_measures_step());
}

}  // namespace nxr::pipelines::graph

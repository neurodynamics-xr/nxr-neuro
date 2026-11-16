// nxr-pipelines — the registered steps of the SPATIAL EVENTS (spec docs/specs/2026-11-14-spatial-events.md): where a pattern happens as an event with an extent, as bursts are events with a duration.
//
//   at_level   chunk       one hat level of a multi-level field ([vertex × level] → [vertex]), a row selection
//   baseline   accumulate  the band power each vertex (and level) has over the good frames: the run outside the named bursts and the bad segments, a STATISTIC stored by name (piece 1)
//   (the threshold's distribution, `relative_histogram`, is registered with the cores' histogram in cores_steps.cpp)
//   patches    accumulate  the connected patches of a relative power map per frame (`nxr::cores::corePatches`, the vortex cores' finder), stored as POINT EVENTS with place, area, size and peak, and the HOME MAP
//                          (per vertex, the good frames it lies in a patch, and the good frames) (pieces 5 and 6)
//   scale      chunk       the characteristic level of each vertex and frame: the hat level whose scale-normalised power is the largest (Lindeberg; piece 4)
//   relative   chunk       a power field divided, row by row, by a stored baseline read by name (a diagonal scaling, `nxr::scaling::divideRows`; piece 2)
//
// Every step here is a node of the graph: a pure function of matrices (compute/) with typed ports, POINTWISE (nothing reduces across time or space on its own; a reduction is a measurement over a partition
// Selection). The runner binds the constants once and gives each step its blocks (a chunk's, or a block of frames').
#include <algorithm>
#include <cmath>
#include <memory>

#include "nxr/cores.h"
#include "nxr/histogram.h"
#include "nxr/io/errors.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/point_events.h"
#include "nxr/io/statistic.h"
#include "nxr/scaling.h"
#include "nxr/scalespace.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }
// the rows an ELEMENT (a vertex, or a vertex's level) holds: its components
std::int64_t per_element(const Value& x) { return x.value_type == "scalar" ? 1 : x.value_type == "quaternion" ? 4 : 3; }
// a field on … × time inside the chunk loop (a chunk's samples or the frame tiling's frames: these steps are pointwise in time)
void in_loop(const Value& x, const std::string& who) { need(x.cls == "field" && x.time && x.chunked, who + ": x must be a field on … × time, inside the chunk loop"); }
std::vector<int> levels_of(const Value& x, const std::string& who) {
  need(x.facts.contains("levels") && x.facts["levels"].is_array() && !x.facts["levels"].empty(), who + ": x is not a multi-level field (`apply_kernel` with `levels`); it has no level axis");
  std::vector<int> l;
  for (const auto& e : x.facts["levels"]) l.push_back(e.get<int>());
  return l;
}
// what the baseline and the fields divided by it must share: the band, the spatial band (levels, shape), the form, the glue and the frame tiling (NOT the bursts the frames are inside: the baseline's frames are outside them)
Json power_chain(const Value& x) {
  Json c = Json::object();
  c["band"] = x.facts.contains("band") ? x.facts["band"] : Json();
  c["form"] = x.facts.contains("form") ? x.facts["form"] : Json();
  c["spatial_band"] = x.facts.contains("spatial_band") ? Json{{"level", x.facts["spatial_band"].value("level", Json())}, {"levels", x.facts["spatial_band"].value("levels", Json())}, {"spatial_shape", x.facts["spatial_band"].value("spatial_shape", Json())}} : Json();
  c["glued"] = x.facts.value("glued", false);
  c["tiling"] = x.facts.contains("frames") ? x.facts["frames"]["tiling"] : Json();
  return c;
}
void power_input(const Value& x, const std::string& who) {
  in_loop(x, who);
  need(!x.complex && x.value_type == "scalar" && x.facts.value("made_by", std::string()) == "power", who + ": x must be a band's power per vertex (`power` of the analytic signal's linear map): x was made by " + x.facts.value("made_by", std::string("no non-linear step")));
}
void spans_on(const Value& ex, const Value& x, const std::string& who, const std::string& port) {
  need(ex.cls == "selection" && ex.row && nxr::io::text(*ex.row, "type") == "spans" && ex.manifolds[0] == *x.time, who + ": " + port + " must be a spans Selection on x's time line (" + *x.time + ")");
}
struct BaselineState { Json chain; double rate = 0, origin = 0; std::string frames; };
struct BaselineAcc { std::vector<double> sum; std::int64_t n = 0; };
struct RelativeState { Eigen::VectorXd d; std::string name; };
struct PatchState {
  double threshold = 2.0;
  double min_area = 0;
  std::int64_t block = 256, V = 0;
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  Eigen::VectorXd area;
  Eigen::MatrixXd positions;
  std::vector<int> levels;
  std::vector<double> size_cm;                       // each level's centre wavelength 2π/√λ_c, cm
  std::vector<std::pair<double, double>> bursts;
  bool events = true;                                // false: the summary only (a run outside the bursts makes too many patches to store as events)
  double fps = 0, origin = 0;
  Json record;
};
constexpr int kAreaBins = 32, kPerFrame = 17;          // the summary's area bins (log, 1e-7 to 1 m², upper edges) and its patches-per-frame counts (0 to 15, 16 or more)
struct PatchAcc {
  std::int64_t frames = 0, patches = 0;
  std::vector<double> area_hist = std::vector<double>(kAreaBins, 0.0), level_count, per_frame = std::vector<double>(kPerFrame, 0.0);
  std::vector<std::int64_t> frame, burst;
  std::vector<std::int32_t> centre_vertex;
  std::vector<double> centre, area, diameter, level, size, peak, in_patch;
};
struct ScaleState { Eigen::VectorXd norm2; std::vector<int> levels; Json record; };
}  // namespace

// what the stored threshold distribution (`relative_histogram`) and the patches cut from the same relative power must share: the band and its shape, the spatial band (reference level, spatial shape), the scale space it was read from,
// the form, the glue, the frame tiling and the baseline the power was divided by (NOT the bursts: the distribution is of the frames outside them)
Json relative_chain(const Value& x) {
  Json c = power_chain(x);
  c["band_shape"] = x.facts.contains("band_shape") ? x.facts["band_shape"] : Json();
  c["levels"] = x.facts.contains("from_levels") ? x.facts["from_levels"] : Json();
  c["relative_to"] = x.facts.value("relative_to", std::string());
  return c;
}

void add_spatial_events_steps(std::vector<StepSpec>& r) {
  // ───────────────────────── at_level ─────────────────────────
  {
    StepSpec s;
    s.name = "at_level"; s.kind = "chunk";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float32, OutputPrecision::follows_input, "copies the rows of one level, per sample: no arithmetic", {});
    s.inputs = {"x"}; s.outputs = {"y"};
    s.parameters = Json{{"level", nullptr}};
    s.matches = "a row selection of apply_kernel's [vertex × level] output (the hat level's rows of every vertex)";
    s.description = "ONE HAT LEVEL of a multi-level field (`apply_kernel` with `levels`: row = (element · levels + level) · components + component): the rows of that level, [elements × components], pointwise in time. The "
                    "level is the ladder's, one of the field's `levels`. What is left is a field of one level, as `apply_kernel` with `level` makes it";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      in_loop(x, "at_level");
      const std::vector<int> lv = levels_of(x, "at_level");
      need(p["level"].is_number_integer(), "at_level: level is the ladder's level to read, one of " + x.facts["levels"].dump());
      need(std::find(lv.begin(), lv.end(), p["level"].get<int>()) != lv.end(), "at_level: level " + p["level"].dump() + " is not one of the field's levels " + x.facts["levels"].dump());
      Value y = made(x);
      y.facts["from_levels"] = x.facts["levels"];   // the scale space the level was read from (the stored threshold and the baseline must be of it)
      y.facts.erase("levels");
      if (y.facts.contains("spatial_band")) { y.facts["spatial_band"]["level"] = p["level"]; y.facts["spatial_band"].erase("levels"); }
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
      const Value& x = ins.at("x");
      const std::vector<int> lv = levels_of(x, "at_level");
      const std::int64_t li = std::find(lv.begin(), lv.end(), p["level"].get<int>()) - lv.begin();
      return std::any(std::array<std::int64_t, 3>{li, static_cast<std::int64_t>(lv.size()), per_element(x)});
    };
    s.chunk = [](const std::any& st, const Blocks& ins, std::int64_t) {
      const auto [li, NL, per] = std::any_cast<const std::array<std::int64_t, 3>&>(st);
      const Block& b = ins.at("x");
      if (b.rows % (NL * per) != 0) refuse("at_level: x has " + std::to_string(b.rows) + " rows, not a whole number of " + std::to_string(NL) + " levels × " + std::to_string(per) + " components");
      const std::int64_t E = b.rows / (NL * per);
      Block y{E * per, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(E * per * b.width)), {}};
      if (b.complex()) y.im.resize(y.re.size());
      for (std::int64_t e = 0; e < E; ++e)
        for (std::int64_t c = 0; c < per; ++c) {
          const std::size_t from = static_cast<std::size_t>(((e * NL + li) * per + c) * b.width), to = static_cast<std::size_t>((e * per + c) * b.width);
          std::copy_n(b.re.begin() + from, b.width, y.re.begin() + to);
          if (b.complex()) std::copy_n(b.im.begin() + from, b.width, y.im.begin() + to);
        }
      return Blocks{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }
  // ───────────────────────── baseline ─────────────────────────
  {
    StepSpec s;
    s.name = "baseline"; s.kind = "accumulate";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "a per-row sum and a count of the good samples, in double: block order changes the result only by rounding", {});
    s.inputs = {"x", "exclude", "outside"}; s.optional = {"exclude", "outside"}; s.outputs = {"y"};
    s.parameters = Json{{"bad_segments", "exclude"}};
    s.matches = "nxr-neuro's own (spec 2026-11-14-spatial-events.md, piece 1): the Studio's `mean` measure (reduceTiles) over a time Selection, folded chunk by chunk; the lookup's `mean` of a statistic";
    s.description = "the BASELINE of a band's power: the mean of x (`power` of the band's analytic signal, per vertex and level) over the GOOD columns, a STATISTIC (measure power, method mean) stored by name, [vertex × level]. "
                    "The good columns are the time Selection the caller gives: the run LEAVING OUT `exclude` (the bad segments) and `outside` (the named bursts, or a stage of them: the run outside them), so the baseline is "
                    "what each vertex has when no burst is on. A mean over a partition's members, folded by additive sums (a measurement), never a reduction inside a pointwise step; the result does not depend on the "
                    "chunk or the block. Dividing by it removes the depth and sensitivity bias of MEG source power (`relative`). MEMORY: one double a row (vertices × levels)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value& x = ins.at("x");
      power_input(x, "baseline");
      need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "baseline: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
      if (ins.count("exclude")) spans_on(ins.at("exclude"), x, "baseline", "exclude");
      if (ins.count("outside")) spans_on(ins.at("outside"), x, "baseline", "outside");
      need(!ins.count("outside") || p["bad_segments"] == "exclude", "baseline: `outside` leaves a Selection out of the run: bad_segments must be exclude (the run's exclusion)");
      (void)c;
      Value y = made(x);
      y.row.reset();
      y.cls = "statistic";
      y.chunked = false;
      y.time.reset();
      y.rate = 0;
      y.facts = Json{{"measure", "power"}, {"method", "mean"}, {"chain", power_chain(x)}, {"levels", x.facts.contains("levels") ? x.facts["levels"] : Json()}, {"outside", ins.count("outside") ? Json(nxr::io::text(*ins.at("outside").row, "name")) : Json()}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json&, ExecContext&) {
      const Value& x = ins.at("x");
      BaselineState st;
      st.chain = power_chain(x); st.rate = x.rate; st.origin = x.origin;
      st.frames = x.facts.value("axis", std::string("samples"));
      return std::any(std::make_shared<BaselineState>(std::move(st)));
    };
    s.start = [](const std::any&) { return std::any(BaselineAcc{}); };
    s.fold = [](const std::any&, std::any& acc_, const Blocks& ins, const Place& place) {
      BaselineAcc& a = std::any_cast<BaselineAcc&>(acc_);
      const Block& b = ins.at("x");
      if (a.sum.empty()) a.sum.assign(static_cast<std::size_t>(b.rows), 0.0);
      if (static_cast<std::int64_t>(a.sum.size()) != b.rows) refuse("baseline: x has " + std::to_string(b.rows) + " rows, the first block had " + std::to_string(a.sum.size()));
      for (std::int64_t c = 0; c < place.n; ++c) {
        if (place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(c)]) continue;
        const std::size_t col = static_cast<std::size_t>(b.left + c);
        for (std::int64_t r = 0; r < b.rows; ++r) a.sum[static_cast<std::size_t>(r)] += b.re[static_cast<std::size_t>(r * b.width) + col];
        ++a.n;
      }
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const BaselineState& st = *std::any_cast<const std::shared_ptr<BaselineState>&>(st_);
      BaselineAcc& a = std::any_cast<BaselineAcc&>(acc_);
      if (a.n == 0) refuse("baseline: no good " + std::string(st.frames == "frames" ? "frame" : "sample") + " in the run (every column is excluded: the bursts and bad segments cover the span): there is no baseline to read");
      Array y{static_cast<std::int64_t>(a.sum.size()), 1, {}};
      y.data.resize(a.sum.size());
      for (std::size_t i = 0; i < a.sum.size(); ++i) y.data[i] = a.sum[i] / static_cast<double>(a.n);
      y.meta = Json{{"baseline", Json{{"columns", a.n}, {"axis", st.frames}, {"chain", st.chain}, {"rate", st.rate}, {"origin", st.origin}}}};
      return Arrays{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  // ───────────────────────── relative ─────────────────────────
  {
    StepSpec s;
    s.name = "relative"; s.kind = "chunk";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "divides each row by a stored constant, per sample; the quotient is formed in double and stored as float32", {});
    s.inputs = {"x", "baseline"}; s.outputs = {"y"};
    s.parameters = Json::object();
    s.matches = "nxr-neuro's own (spec 2026-11-14-spatial-events.md, piece 2); compute nxr::scaling::divideRows (a diagonal scaling, general algebra)";
    s.description = "x divided, row by row, by a stored per-row constant read BY NAME (`baseline`: a statistic of method mean, measure power): the band's power at each vertex in units of that vertex's own baseline "
                    "power. A diagonal scaling bound once in preparation, pointwise in time and in space (it reduces across nothing). The baseline must be of the same chain (band, spatial band and levels, form, glue, "
                    "frame tiling: a power divided by another chain's baseline is refused) and of x's rows; a baseline that is not above 0 at a row is refused. The result is unitless";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      const Value &x = ins.at("x"), &b = ins.at("baseline");
      power_input(x, "relative");
      need(b.cls == "statistic" && b.row && nxr::io::text(*b.row, "method") == "mean" && nxr::io::text(*b.row, "measure") == "power", "relative: baseline must be a stored baseline (`baseline`: a statistic of method mean, measure power), read by name");
      const Json op = Json::parse(nxr::io::opt_text(*b.row, "options_json").value_or("{}"), nullptr, false);
      need(op.is_object() && op.contains("chain"), "relative: " + nxr::io::text(*b.row, "name") + " records no chain: it is not a `baseline`'s");
      const Json mine = power_chain(x), theirs = op["chain"];
      for (const char* k : {"band", "form", "spatial_band", "glued", "tiling"})
        need(mine[k] == theirs.value(k, Json()), std::string("relative: the baseline ") + nxr::io::text(*b.row, "name") + " was made from another " + k + " (" + theirs.value(k, Json()).dump() + ") than the field's (" + mine[k].dump() + "): run `baseline` on the same chain");
      Value y = made(x);
      y.unit.reset();
      y.facts["made_by"] = "relative";
      y.facts["relative_to"] = nxr::io::text(*b.row, "name");
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json&, ExecContext& c) {
      const std::string name = nxr::io::text(*ins.at("baseline").row, "name");
      const nxr::io::StoredStatistic hs = nxr::io::read_statistic(c.sub, name);
      RelativeState st;
      st.name = name;
      st.d = Eigen::Map<const Eigen::VectorXd>(hs.values.data(), static_cast<Eigen::Index>(hs.values.size()));
      for (Eigen::Index i = 0; i < st.d.size(); ++i)
        if (!(std::isfinite(st.d(i)) && st.d(i) > 0)) refuse("relative: the baseline " + name + " is " + std::to_string(st.d(i)) + " at row " + std::to_string(i) + " (a vertex of no power, or a level of no signal): it is no divisor");
      return std::any(std::make_shared<RelativeState>(std::move(st)));
    };
    s.chunk = [](const std::any& st_, const Blocks& ins, std::int64_t) {
      const RelativeState& st = *std::any_cast<const std::shared_ptr<RelativeState>&>(st_);
      const Block& b = ins.at("x");
      if (b.rows != st.d.size()) refuse("relative: x has " + std::to_string(b.rows) + " rows, the baseline " + st.name + " " + std::to_string(st.d.size()));
      using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      const Eigen::MatrixXd X = Eigen::Map<const RM>(b.re.data(), b.rows, b.width).cast<double>();
      const RM Y = nxr::scaling::divideRows(X, st.d).cast<float>();
      Block y{b.rows, b.width, b.left, b.right, std::vector<float>(Y.data(), Y.data() + Y.size()), {}};
      return Blocks{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }
  // ───────────────────────── scale ─────────────────────────
  {
    StepSpec s;
    s.name = "scale"; s.kind = "chunk";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "the characteristic level of each vertex and sample, chosen over the level axis only; double arithmetic, float32 result", {});
    s.inputs = {"x"}; s.outputs = {"y"};
    s.parameters = Json{{"transform", "auto"}, {"modes", 0}, {"order", 0}, {"tolerance", 0.01}};
    s.matches = "Lindeberg (1998) scale selection with L2-normalised responses (spatial-events spec, decision 4); compute nxr::scalespace::selectLevel; the atoms are nxr `vortex_cores.hat_operator`'s columns (the bands of apply_kernel)";
    s.description = "the CHARACTERISTIC LEVEL of each vertex and frame (): x is the band power of a multi-level field (`apply_kernel` with `levels`, then `power`); at each vertex the level whose power over the "
                    "squared L2 norm of its atom is the largest, as the ladder's level (a float). The atom of a vertex at a level is the operator's column there (the level's hat applied to the native sources, read at that glued "
                    "vertex), its norm the Euclidean norm of that column over the native vertices, computed exactly in preparation (the band applied to the identity, a block of 256 columns at a time; MEMORY: the glued vertices × "
                    "256 doubles; TIME: one band application to as many columns as the surface has vertices, per level). So a blob of a level's own size gives its largest normalised power at that level, and white "
                    "noise the same at every level. Pointwise in time and in space (the maximum is over the level axis); the lowest level on a tie. `transform`, `modes`, `order`, `tolerance` are the bands' (apply_kernel's: the same "
                    "operators). The level in centimetres is read at the events (2π/√λ_c of the level's hat centre)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value& x = ins.at("x");
      power_input(x, "scale");
      levels_of(x, "scale");
      need(x.facts.value("glued", false), "scale: x must be on the glued surface (`apply_kernel` with `glued`): the atoms are the glued operator's");
      need(x.facts.contains("spatial_band"), "scale: x carries no spatial band");
      need(p["transform"] == "auto" || p["transform"] == "eigen" || p["transform"] == "chebyshev" || p["transform"] == "exact", "scale: transform " + p["transform"].dump() + ": auto, eigen, chebyshev or exact");
      need(p["modes"].is_number_integer() && p["modes"].get<int>() >= 0 && p["order"].is_number_integer() && p["order"].get<int>() >= 0 && p["tolerance"].is_number() && p["tolerance"].get<double>() > 0, "scale: modes and order are at least 0, tolerance above 0");
      const auto m = c.db.one("SELECT type FROM manifold WHERE id = ?", {nxr::io::V(x.manifolds[0])});
      need(m && nxr::io::text(*m, "type") == "surface", "scale: x's first factor is not a surface");
      Value y = made(x);
      y.facts.erase("levels");
      y.facts["made_by"] = "scale";
      y.facts["scale"] = Json{{"levels", x.facts["levels"]}, {"normalised", "L2 norm of the operator's column at the vertex"}, {"of", "band power"}};
      y.unit.reset();
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      const Value& x = ins.at("x");
      ScaleState st;
      st.levels = levels_of(x, "scale");
      const auto gm = c.sub.db().read("manifold", x.manifolds[0]);
      std::string native = nxr::io::text(*gm, "name");
      const std::string suffix = "_glued";
      if (native.size() > suffix.size() && native.compare(native.size() - suffix.size(), suffix.size(), suffix) == 0) native.resize(native.size() - suffix.size());
      const nxr::pipelines::SurfaceInput S = nxr::pipelines::read_surface(c.sub, native, true);
      const std::int64_t nV = S.n_vertices();
      if (!c.ds) refuse("scale: the atoms' norms are computed with the dataset (the bands may store the eigenmodes they compute)");
      std::vector<std::vector<double>> per_level;
      std::int64_t Vg = 0;
      for (const int l : st.levels) {
        const nxr::pipelines::SurfaceLevel L = nxr::pipelines::read_surface_level(c.sub, S, l);
        nxr::pipelines::SpatialBandOptions bo;
        bo.transform = p["transform"].get<std::string>(); bo.modes = p["modes"].get<int>(); bo.order = p["order"].get<int>(); bo.tolerance = p["tolerance"].get<double>();
        bo.spatial_shape = x.facts["spatial_band"].value("spatial_shape", std::string("mexhat"));
        bo.glued = true;
        const nxr::pipelines::SpatialBand band = nxr::pipelines::make_spatial_band(*c.ds, c.sub, S, L, bo, c.say);
        if (Vg == 0) Vg = band.glued_vertices;
        std::vector<double> n2(static_cast<std::size_t>(Vg), 0.0);
        const std::int64_t bs = 256;
        for (std::int64_t k0 = 0; k0 < nV; k0 += bs) {
          const std::int64_t w = std::min(bs, nV - k0);
          Eigen::MatrixXd I = Eigen::MatrixXd::Zero(nV, w);
          for (std::int64_t i = 0; i < w; ++i) I(k0 + i, i) = 1.0;
          const Eigen::MatrixXd A = band.apply_glued(I);                      // the atoms of the native vertices k0 … k0 + w (the columns), on every glued vertex
          for (std::int64_t v = 0; v < Vg; ++v) n2[static_cast<std::size_t>(v)] += A.row(v).squaredNorm();
        }
        per_level.push_back(std::move(n2));
        if (c.say) c.say(Json{{"stage", "scale"}, {"level", l}, {"atoms_normed", Vg}});
      }
      const std::int64_t NL = static_cast<std::int64_t>(st.levels.size());
      st.norm2.resize(Vg * NL);
      for (std::int64_t v = 0; v < Vg; ++v)
        for (std::int64_t li = 0; li < NL; ++li) {
          const double n = per_level[static_cast<std::size_t>(li)][static_cast<std::size_t>(v)];
          if (!(n > 0)) refuse("scale: the atom of glued vertex " + std::to_string(v) + " at level " + std::to_string(st.levels[static_cast<std::size_t>(li)]) + " has norm 0: the band reaches no source from it");
          st.norm2(v * NL + li) = n;
        }
      st.record = Json{{"levels", x.facts["levels"]}, {"atoms", "the operator's column at the vertex: the level's hat applied to the native sources, read at the glued vertex"}, {"norm", "Euclidean, over the native vertices"}, {"vertices", Vg}};
      return std::any(std::make_shared<ScaleState>(std::move(st)));
    };
    s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<ScaleState>&>(st)->record; };
    s.chunk = [](const std::any& st_, const Blocks& ins, std::int64_t) {
      const ScaleState& st = *std::any_cast<const std::shared_ptr<ScaleState>&>(st_);
      const Block& b = ins.at("x");
      if (b.rows != st.norm2.size()) refuse("scale: x has " + std::to_string(b.rows) + " rows, the atoms' norms " + std::to_string(st.norm2.size()));
      const int NL = static_cast<int>(st.levels.size());
      using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      const Eigen::MatrixXd P = Eigen::Map<const RM>(b.re.data(), b.rows, b.width).cast<double>();
      nxr::scalespace::LevelChoice C;
      try { C = nxr::scalespace::selectLevel(P, st.norm2, NL); } catch (const std::invalid_argument& e) { refuse(std::string("scale: ") + e.what()); }
      Block y{b.rows / NL, b.width, b.left, b.right, std::vector<float>(static_cast<std::size_t>(b.rows / NL * b.width)), {}};
      for (std::int64_t v = 0; v < b.rows / NL; ++v)
        for (std::int64_t t = 0; t < b.width; ++t) y.re[static_cast<std::size_t>(v * b.width + t)] = static_cast<float>(st.levels[static_cast<std::size_t>(C.index(v, t))]);
      return Blocks{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }
  // ───────────────────────── patches ─────────────────────────
  {
    StepSpec s;
    s.name = "patches"; s.kind = "accumulate";
    s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "finds the patches of each frame on its own, but appends them to one list in frame order and adds to counts, so the blocks must meet in time order", {});
    s.inputs = {"x", "scale", "bursts", "outside", "exclude", "threshold"}; s.optional = {"bursts", "outside", "exclude", "threshold"};
    s.outputs = {"spans", "frame", "burst", "centre_vertex", "centre", "area", "diameter_cm", "characteristic_level", "size_cm", "relative_peak", "home", "summary"};
    s.parameters = Json{{"threshold", "percentile"}, {"percentile", 99.0}, {"ratio", 2.0}, {"min_area", 0.0}, {"block_frames", 256}, {"bad_segments", "exclude"}, {"events", true}};
    s.matches = "compute nxr::cores::corePatches (the vortex cores' patch finder, on relative power instead of the curl); decisions 1-4 of docs/specs/2026-11-14-spatial-events.md";
    s.description = "the SPATIAL EVENTS of a relative band power map on the frame tiling: per frame, the connected vertices of the glued mesh whose relative power is at least the threshold, joined by the mesh's edges. "
                    "THE THRESHOLD is set by its false-alarm rate, measured as the time bursts' is from the envelope's own distribution: `threshold: percentile` (the default) is the `percentile` (99) of the "
                    "relative power OUTSIDE the bursts, derived here from the stored distribution `threshold` (`run spatial_threshold`, another run's statistic read BY NAME; refused when made from another band, levels, "
                    "reference level, shape, tiling or baseline), so about 1 % of the vertices of a frame with no event are above it; `threshold: ratio` is the named option (relative power at least `ratio` times the "
                    "vertex's baseline: a single frame's relative power is about exponential, so 2 is exceeded by about 13.5 % of the vertices on noise alone). The finder is `corePatches`, which never knows its input. Stored as POINT "
                    "EVENTS (point_events.h): a spans Selection, one member per patch on its frame, and per patch as measurements: frame, burst (the ordinal of the bursts Selection's span holding the frame), centre_vertex, "
                    "centre (x y z, the power-weighted centroid, m), area (m²), diameter_cm (the equal-area diameter 2√(area/π)), characteristic_level (the `scale` map's level at the centre vertex and frame), size_cm "
                    "(that level's centre wavelength 2π/√λ_c), relative_peak (the largest relative power in the patch). The derived threshold value is in the provenance. Patches smaller than `min_area` (m², 0: none) are dropped. "
                    "The HOME MAP `home` is a statistic [vertex × (frames in a patch, good frames)]: the counts are additive over frames, the fraction is derived at read; a vertex is in a patch when its relative power is at "
                    "least the threshold (min_area drops events, not vertices). Frames are independent, so the result does not depend on the chunk or the block. MEMORY: V × block_frames doubles a few times over";
    s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value &x = ins.at("x"), &sc = ins.at("scale");
      need(x.cls == "field" && x.time && x.chunked && x.facts.value("axis", std::string("samples")) == "frames", "patches: x must be a field on the frame tiling (`frames`)");
      need(!x.complex && x.value_type == "scalar" && x.facts.value("made_by", std::string()) == "relative" && !x.facts.contains("levels"), "patches: x must be the relative power of ONE level (`relative`, then `at_level`); x was made by " + x.facts.value("made_by", std::string("no non-linear step")));
      need(sc.cls == "field" && sc.time && sc.chunked && sc.facts.value("made_by", std::string()) == "scale", "patches: scale must be the characteristic-level map (`scale`)");
      need(sc.manifolds == x.manifolds && sc.facts.value("axis", std::string()) == "frames" && sc.facts["frames"]["tiling"] == x.facts["frames"]["tiling"], "patches: scale and x must be on the same surface and frame tiling");
      const auto m = c.db.one("SELECT type FROM manifold WHERE id = ?", {nxr::io::V(x.manifolds[0])});
      need(m && nxr::io::text(*m, "type") == "surface", "patches: x's first factor is not a surface (the patches are on the mesh)");
      need(p["threshold"] == "ratio" || p["threshold"] == "percentile", "patches: threshold " + p["threshold"].dump() + ": percentile (the default: the false-alarm rate, from `run spatial_threshold`) or ratio");
      need(p["ratio"].is_number() && p["ratio"].get<double>() > 0, "patches: ratio is a number above 0 (2: twice the baseline)");
      need(p["percentile"].is_number() && p["percentile"].get<double>() > 0 && p["percentile"].get<double>() < 100, "patches: percentile is in (0, 100) (99: one vertex in a hundred above it on frames with no event)");
      if (p["threshold"] == "percentile") {
        need(ins.count("threshold"), "patches: threshold percentile reads the stored distribution of the relative power outside the bursts (`run spatial_threshold`, named by `threshold_statistic`): none is stored under that name; run it first, or use threshold ratio");
        const Value& t = ins.at("threshold");
        need(t.cls == "statistic" && t.row && nxr::io::text(*t.row, "method") == "histogram" && nxr::io::text(*t.row, "measure") == "count", "patches: threshold must be the stored histogram (`relative_histogram`: a statistic of method histogram, measure count), read by name");
        const Json op = Json::parse(nxr::io::opt_text(*t.row, "options_json").value_or("{}"), nullptr, false);
        need(op.is_object() && op.contains("lo") && op.contains("hi") && op.contains("bins") && op.contains("chain") && op["chain"].contains("relative_to"), "patches: " + nxr::io::text(*t.row, "name") + " records no log-bin range or relative-power chain: it is not a `relative_histogram`'s");
        const Json mine = relative_chain(x), theirs = op["chain"];
        for (const char* k : {"band", "band_shape", "form", "spatial_band", "glued", "tiling", "levels", "relative_to"})
          need(mine[k] == theirs.value(k, Json()), std::string("patches: the threshold ") + nxr::io::text(*t.row, "name") + " was made from another " + k + " (" + theirs.value(k, Json()).dump() + ") than the field's (" + mine[k].dump() + "): run `relative_histogram` (`run spatial_threshold`) on the same chain");
      }
      need(p["min_area"].is_number() && p["min_area"].get<double>() >= 0, "patches: min_area is at least 0 (m²)");
      need(p["block_frames"].is_number_integer() && p["block_frames"].get<std::int64_t>() >= 1, "patches: block_frames is at least 1");
      need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "patches: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
      need(ins.count("bursts") != ins.count("outside"), "patches: name the bursts the frames are inside (`bursts`: the events) or the bursts the run is OUTSIDE (`outside`: the noise reference), one of the two");
      need(p["events"].is_boolean(), "patches: events is true (the patches are stored as events) or false (the summary only)");
      if (ins.count("bursts")) {
        const Value& b = ins.at("bursts");
        need(b.cls == "selection" && b.row && nxr::io::text(*b.row, "type") == "spans" && b.manifolds[0] == *x.time, "patches: bursts must be a spans Selection on x's time line (the bursts the frames are inside)");
        need(x.facts["frames"]["within"] == Json(nxr::io::text(*b.row, "name")), "patches: bursts (" + nxr::io::text(*b.row, "name") + ") must be the Selection the frames are inside (`frames` within " + x.facts["frames"]["within"].dump() + ")");
      } else {
        const Value& b = ins.at("outside");
        need(b.cls == "selection" && b.row && nxr::io::text(*b.row, "type") == "spans" && b.manifolds[0] == *x.time, "patches: outside must be a spans Selection on x's time line (the bursts the run leaves out)");
        need(x.facts["frames"].value("within", Json()).is_null(), "patches: outside: the frames must be the whole run's (`frames` with no `within`), the bursts are left out by `outside`");
        need(p["bad_segments"] == "exclude", "patches: outside leaves the bursts out of the run: bad_segments must be exclude");
      }
      if (ins.count("exclude")) spans_on(ins.at("exclude"), x, "patches", "exclude");
      const Json record = Json{{"threshold", p["threshold"]}, {"ratio", p["ratio"]}, {"percentile", p["percentile"]}, {"min_area", p["min_area"]}, {"chain", power_chain(x)}, {"relative_to", x.facts.value("relative_to", std::string())}};
      Values out;
      out["spans"] = point_events_value(x, Json{{"spatial_events", record}});
      for (const char* a : {"frame", "burst", "centre_vertex", "relative_peak", "characteristic_level"}) out[a] = point_attribute_value(x, a);
      out["burst"] = point_attribute_value(x, "burst_ordinal");
      out["centre"] = point_attribute_value(x, "centre", "m");
      out["area"] = point_attribute_value(x, "area", "m\xc2\xb2");
      out["diameter_cm"] = point_attribute_value(x, "diameter_cm", "cm");
      out["size_cm"] = point_attribute_value(x, "size_cm", "cm");
      Value h = made(x);
      h.cls = "statistic"; h.chunked = false; h.time.reset(); h.rate = 0;
      h.facts = Json{{"measure", "count"}, {"method", "occupancy"}, {"chain", power_chain(x)}};
      out["home"] = h;
      Value sm = made(x);
      sm.cls = "statistic"; sm.chunked = false; sm.time.reset(); sm.rate = 0; sm.manifolds.clear();
      sm.facts = Json{{"measure", "patch_summary"}, {"generic", true}, {"chain", power_chain(x)}};
      out["summary"] = sm;
      return out;
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      const Value &x = ins.at("x"), &sc = ins.at("scale");
      PatchState st;
      st.events = p["events"].get<bool>();
      st.threshold = p["ratio"].get<double>(); st.min_area = p["min_area"].get<double>(); st.block = p["block_frames"].get<std::int64_t>(); st.fps = x.rate; st.origin = x.origin;
      for (const auto& e : sc.facts["scale"]["levels"]) st.levels.push_back(e.get<int>());
      if (ins.count("bursts")) for (const auto& r : c.sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {nxr::io::V(nxr::io::text(*ins.at("bursts").row, "id"))})) st.bursts.push_back({std::stod(nxr::io::text(r, "start")), std::stod(nxr::io::text(r, "stop"))});
      const auto man = c.sub.db().read("manifold", x.manifolds[0]);
      const nxr::pipelines::SurfaceInput S = nxr::pipelines::read_surface(c.sub, nxr::io::text(*man, "name"), true);
      st.V = S.n_vertices();
      nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
      const std::vector<double> va = nxr::ladder::vertexAreas(mesh);
      st.area = Eigen::Map<const Eigen::VectorXd>(va.data(), static_cast<Eigen::Index>(va.size()));
      st.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(S.vertices.data(), S.n_vertices(), 3);
      std::vector<std::pair<std::int32_t, std::int32_t>> pr;
      for (std::int64_t f = 0; f < S.n_faces(); ++f)
        for (int k = 0; k < 3; ++k) {
          std::int32_t a = S.faces[static_cast<std::size_t>(3 * f + k)], b = S.faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)];
          if (a > b) std::swap(a, b);
          pr.emplace_back(a, b);
        }
      std::sort(pr.begin(), pr.end());
      pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
      st.pairs = std::move(pr);
      // each level's size: the centre wavelength 2π/√λ_c of its hat (the geometric centre of the level's Weyl band; level 0: hi/√2), on the NATIVE surface's ladder
      std::string native = nxr::io::text(*man, "name");
      const std::string suffix = "_glued";
      if (native.size() > suffix.size() && native.compare(native.size() - suffix.size(), suffix.size(), suffix) == 0) native.resize(native.size() - suffix.size());
      const nxr::pipelines::SurfaceInput N = nxr::pipelines::read_surface(c.sub, native, true);
      for (const int l : st.levels) {
        const nxr::pipelines::SurfaceLevel L = nxr::pipelines::read_surface_level(c.sub, N, l);
        const auto& m = L.bank.members[static_cast<std::size_t>(L.l)];
        const double lc = m.lo > 0 ? std::sqrt(m.lo * m.hi) : m.hi / std::sqrt(2.0);
        st.size_cm.push_back(100.0 * 2.0 * M_PI / std::sqrt(lc));
      }
      Json rule = Json{{"threshold", st.threshold}, {"threshold_rule", "ratio: relative power >= ratio x the vertex's baseline"}, {"ratio", st.threshold}};
      if (p["threshold"] == "percentile") {                                // THE THRESHOLD, derived from the stored distribution of the relative power outside the bursts (read by name)
        const Value& t = ins.at("threshold");
        const nxr::io::StoredStatistic hs = nxr::io::read_statistic(c.sub, nxr::io::text(*t.row, "name"));
        const std::int64_t bins = hs.options["bins"].get<std::int64_t>();
        if (static_cast<std::int64_t>(hs.values.size()) != bins) refuse("patches: " + nxr::io::text(*t.row, "name") + ": " + std::to_string(hs.values.size()) + " counts for " + std::to_string(bins) + " bins");
        std::vector<std::int64_t> counts(hs.values.size());
        for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = static_cast<std::int64_t>(std::llround(hs.values[i]));
        nxr::histogram::LogBins lb;
        lb.lo = hs.options["lo"].get<double>(); lb.hi = hs.options["hi"].get<double>(); lb.bins = static_cast<int>(bins);
        try { st.threshold = nxr::histogram::percentile(counts, lb, p["percentile"].get<double>()); } catch (const std::invalid_argument& e) { refuse(std::string("patches: the threshold: ") + e.what()); }
        rule = Json{{"threshold", st.threshold}, {"threshold_rule", "percentile: the upper edge of the first bin whose cumulative share of the relative power outside the bursts reaches the percentile (exact to one log bin, 0.34 %)"},
                    {"percentile", p["percentile"]}, {"threshold_statistic", Json{{"name", nxr::io::text(*t.row, "name")}, {"id", nxr::io::text(*t.row, "id")}}},
                    {"histogram", Json{{"lo", lb.lo}, {"hi", lb.hi}, {"bins", bins}, {"frames", hs.options.value("frames", Json())}, {"outside", hs.options["chain"].value("outside", Json())}}}};
      }
      st.record = Json{{"min_area", st.min_area},
                       {"surface", Json{{"name", nxr::io::text(*man, "name")}, {"vertices", st.V}, {"edges", static_cast<std::int64_t>(st.pairs.size())}}},
                       {"levels", sc.facts["scale"]["levels"]}, {"size_cm", st.size_cm}, {"size_rule", "2*pi/sqrt(lambda_c), lambda_c the hat's geometric centre of the level's Weyl band (level 0: hi/sqrt2), in cm"},
                       {"diameter_rule", "2*sqrt(area/pi), cm"}, {"reference", "nxr::cores::corePatches (nxr vortex._core_patches) on relative power"},
                       {"warnings", Json::array({"a patch is a connected region of band power above the threshold (a percentile of the relative power outside the bursts, or a multiple of the baseline) inside a burst: it describes where the band is strong, not that a local generator is there (volume conduction and the inverse spread power over the cortex)",
                                                 "the size is the hat level of the largest L2-normalised response at the patch's centre: octave-spaced, one level per patch, the instrument's blur inside it"})}};
      for (auto it = rule.begin(); it != rule.end(); ++it) st.record[it.key()] = it.value();
      return std::any(std::make_shared<PatchState>(std::move(st)));
    };
    s.record = [](const std::any& st_) { return std::any_cast<const std::shared_ptr<PatchState>&>(st_)->record; };
    s.start = [](const std::any& st_) {
      PatchAcc a;
      a.in_patch.assign(static_cast<std::size_t>(std::any_cast<const std::shared_ptr<PatchState>&>(st_)->V), 0.0);
      return std::any(std::move(a));
    };
    s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
      const PatchState& st = *std::any_cast<const std::shared_ptr<PatchState>&>(st_);
      PatchAcc& a = std::any_cast<PatchAcc&>(acc_);
      const Block &bx = ins.at("x"), &bs = ins.at("scale");
      if (bx.rows != st.V || bs.rows != st.V) refuse("patches: x has " + std::to_string(bx.rows) + " rows, the surface " + std::to_string(st.V) + " vertices");
      std::vector<std::int64_t> cols;
      for (std::int64_t c = 0; c < place.n; ++c) if (!place.good || place.good->empty() || (*place.good)[static_cast<std::size_t>(c)]) cols.push_back(c);
      for (std::size_t s0 = 0; s0 < cols.size(); s0 += static_cast<std::size_t>(st.block)) {
        const std::size_t k = std::min<std::size_t>(cols.size() - s0, static_cast<std::size_t>(st.block));
        Eigen::MatrixXd M(st.V, static_cast<Eigen::Index>(k));
        for (std::size_t j = 0; j < k; ++j) {
          const std::size_t col = static_cast<std::size_t>(bx.left + cols[s0 + j]);
          for (std::int64_t v = 0; v < st.V; ++v) M(v, static_cast<Eigen::Index>(j)) = bx.re[static_cast<std::size_t>(v) * static_cast<std::size_t>(bx.width) + col];
        }
        nxr::cores::Cores C;
        try { C = nxr::cores::corePatches(M, st.pairs, st.area, st.positions, st.threshold, 0); } catch (const std::invalid_argument& e) { refuse(std::string("patches: ") + e.what()); }
        for (std::int64_t v = 0; v < st.V; ++v) for (std::size_t j = 0; j < k; ++j) if (M(v, static_cast<Eigen::Index>(j)) >= st.threshold) a.in_patch[static_cast<std::size_t>(v)] += 1.0;
        std::vector<int> in_frame(k, 0);                                             // the patches kept in each frame of this block (the summary's per-frame count)
        if (a.level_count.empty()) a.level_count.assign(st.levels.size(), 0.0);
        for (std::int64_t i = 0; i < C.size(); ++i) {
          if (C.area[static_cast<std::size_t>(i)] < st.min_area) continue;
          const std::int64_t jj = C.frame[static_cast<std::size_t>(i)], frame = place.first + cols[s0 + static_cast<std::size_t>(jj)];
          const std::int32_t cv = C.centre_vertex[static_cast<std::size_t>(i)];
          const std::size_t col = static_cast<std::size_t>(bs.left + cols[s0 + static_cast<std::size_t>(jj)]);
          const int lev = static_cast<int>(std::lround(bs.re[static_cast<std::size_t>(cv) * static_cast<std::size_t>(bs.width) + col]));
          const auto it = std::find(st.levels.begin(), st.levels.end(), lev);
          if (it == st.levels.end()) refuse("patches: the scale map holds level " + std::to_string(lev) + ", not one of " + Json(st.levels).dump());
          // the summary: the area's log bin (1e-7 to 1 m², 32 bins, upper edges), the characteristic level, the frame's count
          { const double la = std::log10(std::max(C.area[static_cast<std::size_t>(i)], 1e-300)); const int bin = std::min(kAreaBins - 1, std::max(0, static_cast<int>(std::ceil((la + 7.0) / 7.0 * kAreaBins)) - 1));
            a.area_hist[static_cast<std::size_t>(bin)] += 1.0; a.level_count[static_cast<std::size_t>(it - st.levels.begin())] += 1.0; a.patches += 1; ++in_frame[static_cast<std::size_t>(jj)]; }
          if (!st.events) continue;
          const double t = st.origin + static_cast<double>(frame) / st.fps;
          std::int64_t burst = -1;
          for (std::size_t b = 0; b < st.bursts.size(); ++b) if (t >= st.bursts[b].first && t < st.bursts[b].second) { burst = static_cast<std::int64_t>(b); break; }
          a.frame.push_back(frame);
          a.burst.push_back(burst);
          a.centre_vertex.push_back(cv);
          for (int d = 0; d < 3; ++d) a.centre.push_back(C.centre(i, d));
          a.area.push_back(C.area[static_cast<std::size_t>(i)]);
          a.diameter.push_back(100.0 * 2.0 * std::sqrt(C.area[static_cast<std::size_t>(i)] / M_PI));
          a.level.push_back(lev);
          a.size.push_back(st.size_cm[static_cast<std::size_t>(it - st.levels.begin())]);
          a.peak.push_back(C.peak[static_cast<std::size_t>(i)]);
        }
        for (std::size_t j = 0; j < k; ++j) a.per_frame[static_cast<std::size_t>(std::min(in_frame[j], kPerFrame - 1))] += 1.0;
      }
      a.frames += static_cast<std::int64_t>(cols.size());
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const PatchState& st = *std::any_cast<const std::shared_ptr<PatchState>&>(st_);
      PatchAcc& a = std::any_cast<PatchAcc&>(acc_);
      Arrays out;
      if (st.events) {
        PointEvents e;
        e.fps = st.fps; e.origin = st.origin; e.frame = a.frame;
        e.record = st.record;
        e.record["frames_scanned"] = a.frames;
        e.record["events"] = static_cast<std::int64_t>(a.frame.size());
        auto col = [&](const char* m, std::int64_t comps, std::vector<double> v) { e.attributes.push_back(PointAttribute{m, comps, std::move(v)}); };
        std::vector<double> frame(a.frame.begin(), a.frame.end()), burst(a.burst.begin(), a.burst.end()), cv(a.centre_vertex.begin(), a.centre_vertex.end());
        col("frame", 1, frame); col("burst_ordinal", 1, burst); col("centre_vertex", 1, cv); col("centre", 3, a.centre); col("area", 1, a.area); col("diameter_cm", 1, a.diameter);
        col("characteristic_level", 1, a.level); col("size_cm", 1, a.size); col("relative_peak", 1, a.peak);
        Arrays all = point_event_arrays(e);
        out["spans"] = std::move(all.at("spans"));
        const char* names[] = {"frame", "burst", "centre_vertex", "centre", "area", "diameter_cm", "characteristic_level", "size_cm", "relative_peak"};
        const char* measures[] = {"frame", "burst_ordinal", "centre_vertex", "centre", "area", "diameter_cm", "characteristic_level", "size_cm", "relative_peak"};
        for (int i = 0; i < 9; ++i) out[names[i]] = std::move(all.at(measures[i]));

      }
      Array h{st.V, 2, {}};
      h.data.resize(static_cast<std::size_t>(st.V * 2));
      for (std::int64_t v = 0; v < st.V; ++v) { h.data[static_cast<std::size_t>(2 * v)] = a.in_patch[static_cast<std::size_t>(v)]; h.data[static_cast<std::size_t>(2 * v + 1)] = static_cast<double>(a.frames); }
      h.meta = Json{{"home_map", Json{{"frames", a.frames}, {"threshold", st.threshold}, {"events", static_cast<std::int64_t>(a.frame.size())}}}};
      out["home"] = std::move(h);
      {
        // THE SUMMARY (a statistic: what the events are, in a form that is the same for the bursts' frames and for the noise reference outside them): the area's log bins, the characteristic levels, the patches per frame, the totals
        Json names = Json::array();
        std::vector<double> v;
        for (int b = 0; b < kAreaBins; ++b) { names.push_back(Json{{"name", "area_bin_" + std::to_string(b)}, {"upper_edge_m2", std::pow(10.0, -7.0 + 7.0 * (b + 1) / kAreaBins)}}); v.push_back(a.area_hist[static_cast<std::size_t>(b)]); }
        for (std::size_t l = 0; l < st.levels.size(); ++l) { names.push_back(Json{{"name", "level_" + std::to_string(st.levels[l])}, {"size_cm", st.size_cm[l]}}); v.push_back(a.level_count.empty() ? 0.0 : a.level_count[l]); }
        for (int k2 = 0; k2 < kPerFrame; ++k2) { names.push_back(Json{{"name", "per_frame_" + std::to_string(k2) + (k2 == kPerFrame - 1 ? "_or_more" : "")}}); v.push_back(a.per_frame[static_cast<std::size_t>(k2)]); }
        names.push_back(Json{{"name", "frames"}}); v.push_back(static_cast<double>(a.frames));
        names.push_back(Json{{"name", "patches"}}); v.push_back(static_cast<double>(a.patches));
        const std::int64_t n = static_cast<std::int64_t>(v.size());
        Array sm{1, n, v};
        sm.meta = Json{{"generic_statistic", Json{{"measure", "patch_summary"}, {"method", "patch_statistics"}, {"shape", Json::array({n})}, {"axes", Json::array({Json{{"name", "summary"}, {"values", names}, {"size", n}}})},
                                                 {"n_members", a.frames},
                                                 {"description", "the patches of the band's relative power: counts by area (log bins, 1e-7 to 1 m2), by characteristic level and by patches per frame, the good frames scanned and the patches found"},
                                                 {"options", Json{{"threshold", st.threshold}, {"min_area", st.min_area}, {"events_stored", st.events}, {"frames", a.frames}, {"patches", a.patches}, {"record", st.record},
                                                                  {"layout", "area_bin_0..31 (log10 m2 from -7 to 0, the entries name their upper edge), level_<l> (the characteristic level of each patch), per_frame_0..16 (frames with that many patches; the last 16 or more), frames, patches"}}}}}};
        out["summary"] = std::move(sm);
      }
      return out;
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph

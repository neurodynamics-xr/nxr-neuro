// nxr-pipelines — THE FIVE SELECTION MAKERS AS NODES (select_nodes.h; spec docs/specs/2026-11-17-consolidation.md, section 1).
//
// ONE rule family a maker; the maths is compute's (nxr::selection: dyadic, transfer, time_grid, time_rule, constant_q_design, the level resolvers of grid.h, bands.h), the node only
// resolves the stated QUANTITIES to the grid's parameters, describes what it made, and refuses with the reason. A maker reads no data; the selection it makes is handed to the next
// node by its typed edge and stored once (get-or-create by the maker's name, resolved parameters and inputs' ids).
//
// THE DESCRIPTOR a made grid records (read by `reduce` and `roll_up`, measurement_nodes.cpp), in the describe's params:
//   params.grid     {kind: time, origin, width, hop, count, size, level, rate}      a time grid as a closed-form Rule on the line's samples
//                   {kind: dyadic, surface, level}                                  a level of a surface's grid
//                   {kind: bands, rule, unit, chart}                                a band set
//   params.product  {factors: [grid, ...], layout: product | joint, bits, radices} mixed radix, the FIRST factor fastest (Studio D100; D129 joint: first factor's radix 2^bits)
// plus `quantity` {stated, snapped}: what was asked and what the level actually is (spec 3.1, item 3).
//
// WHAT THEY CANNOT RESOLVE: a level is a power of two (tiles of 2^k samples, tiles halved by equal area): a stated quantity snaps to the nearest level the closed form allows, and the
// snapped value is recorded, never the stated one.
#include "nxr/pipelines/select_nodes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <tuple>

#include "nxr/graph/node.h"
#include "nxr/pipelines/measurement_nodes.h"
#include "nxr/quantity/quantity.h"
#include "nxr/select/grid.h"
#include "nxr/select/partition.h"
#include "nxr/selection/selection.h"

namespace nxr::pipelines::graph {

namespace ns = nxr::selection;
namespace ng = nxr::graph;

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
std::string g(double v) { char b[64]; std::snprintf(b, sizeof b, "%g", v); return b; }
bool is_power_of_two(std::int64_t n) { return n > 0 && (n & (n - 1)) == 0; }
int log2_exact(std::int64_t n) { int k = 0; while ((std::int64_t(1) << k) < n) ++k; return k; }

nxr::quantity::Quantity parse_q(const Json& j, const std::string& who, const std::string& what) {
  need(j.is_string(), who + ": " + what + " is a quantity with a unit (e.g. \"2 s\", \"8-12 Hz\"), not " + j.dump());
  try { return nxr::quantity::parse(j.get<std::string>()); } catch (const std::invalid_argument& e) { refuse(who + ": " + what + ": " + e.what()); }
}
// a band's [lo, hi] in Hz from a quantity (any frequency chart: a log10 chart is exponentiated) or from a pair
std::pair<double, double> hz_band(const Json& j, const std::string& who) {
  if (j.is_array() && j.size() == 2 && j[0].is_number() && j[1].is_number()) return {j[0].get<double>(), j[1].get<double>()};   // [lo, hi] in Hz (the written pair of a definition)
  const nxr::quantity::Quantity q = parse_q(j, who, "band");
  need(q.unit == "Hz" || q.chart == "log10" || q.chart == "linear", who + ": band " + j.dump() + " is not in a frequency chart (Hz, log10 Hz)");
  return q.chart == "log10" ? std::pair<double, double>{std::pow(10.0, q.lo), std::pow(10.0, q.hi)} : std::pair<double, double>{q.lo, q.hi};
}

// ── select.time_grid ─────────────────────────────────────────────────────────────────────────────────────────────
struct TimeGridState {
  std::string rule, recording, line_name;
  std::string rule_name;                       // the grid NAMED BY ITS RULE ("band 8-16 Hz", "frames 8-12 Hz 36 per cycle"): several grids coexist on a line, each found by this name
  double rate = 0, line_origin = 0;
  std::int64_t size = 0, origin = 0, width = 0, hop = 0, count = 0;
  double hop_exact = 0, width_exact = 0;       // in samples: equal to hop and width on a dyadic grid; a frame grid of frames per cycle is NOT whole samples (exactly, never snapped)
  int level = -1;
  Json stated = Json::object(), snapped = Json::object(), band = nullptr, levels = nullptr, design = nullptr;
};

// the grid a request makes on a line (shared by the plan, which names the output, and the run)
TimeGridState make_time_grid(const Values& ins, const Json& p, const Bound& bound) {
  need(ins.count("line") != 0, "select.time_grid: the input `line` (the time line, or a field on it) is missing");
  const Value& lv = ins.at("line");
  need(lv.time && lv.rate > 0, "select.time_grid: `line` has no time line with a rate");
  const auto line = bound.manifold(*lv.time);
  need(line.has_value(), "select.time_grid: the time line is not among the subject's rows");
  TimeGridState s;
  s.rule = p["rule"].get<std::string>();
  s.rate = lv.rate;
  s.line_origin = lv.origin;
  s.size = nxr::io::integer(*line, "n_vertices");
  s.line_name = nxr::io::text(*line, "name");
  if (lv.row) s.recording = nxr::io::text(*lv.row, "name");
  const std::string who = "select.time_grid " + s.rule;
  // THE ORIGIN of the grid (an option; the default is the line's first sample): the recording's start, a clock time, or an EVENT (the first member of a spans selection on the line)
  if (p["origin"].is_string() && p["origin"].get<std::string>() != "start" && p["origin"].get<std::string>() != "event") {
    const nxr::quantity::Quantity q = parse_q(p["origin"], who, "origin");
    need(q.unit == "s", who + ": origin " + p["origin"].dump() + " is a clock time in seconds (or `start`, or `event`)");
    s.origin = static_cast<std::int64_t>(std::llround((q.hi - s.line_origin) * s.rate));
  } else if (p["origin"] == "event") {
    need(ins.count("event") && ins.at("event").row, who + ": origin `event` names a spans selection on the line (input `event`): its first member's start is the grid's origin");
    const auto spans = bound.spans(nxr::io::text(*ins.at("event").row, "id"));
    need(!spans.empty(), who + ": the origin event has no members");
    s.origin = static_cast<std::int64_t>(std::llround((spans.front().first - s.line_origin) * s.rate));
  }
  need(s.origin >= 0 && s.origin < s.size, who + ": the origin lies outside the line (" + std::to_string(s.origin) + " of " + std::to_string(s.size) + " samples)");
  const std::int64_t room = s.size - s.origin;
  auto level_grid = [&](int k) {
    need(k >= 0 && k <= 62, who + ": level " + std::to_string(k) + " is outside 0..62");
    s.level = k;
    s.width = s.hop = std::int64_t(1) << k;
    s.hop_exact = s.width_exact = static_cast<double>(s.hop);
    s.count = (room + s.hop - 1) / s.hop;
  };
  auto band_level = [&](double lo, double hi) {
    const int maxk = static_cast<int>(std::floor(std::log2(static_cast<double>(std::max<std::int64_t>(1, s.size)))));
    try { return nxr::select::time_level_of_band(lo, hi, s.rate, p["cycles"].is_number() ? p["cycles"].get<double>() : nxr::select::kJointCycles, maxk); }
    catch (const std::invalid_argument& e) { refuse(who + ": " + e.what()); }
  };
  if (s.rule == "tiles") {
    const nxr::quantity::Quantity q = parse_q(p["duration"], who, "duration");
    need(q.unit == "s", who + ": duration " + p["duration"].dump() + " is in seconds");
    const auto r = nxr::select::level_of_duration(q.hi, s.rate);                    // the first level whose tile (2^k samples) is at least the duration
    level_grid(r.level);
    s.stated = Json{{"duration", p["duration"]}};
    s.snapped = Json{{"duration_s", r.snapped}, {"samples", s.width}};
  } else if (s.rule == "band" || s.rule == "frames") {
    const auto [lo, hi] = hz_band(p["band"], who);
    need(lo > 0 && hi >= lo, who + ": band " + p["band"].dump() + " needs 0 < lo <= hi (a grid follows from its centre)");
    s.band = Json::array({lo, hi});
    const int kb = band_level(lo, hi);
    const double centre = std::sqrt(lo * hi);
    if (s.rule == "band") {
      level_grid(kb);
      s.stated = Json{{"band", p["band"]}};
      s.snapped = Json{{"tile_s", static_cast<double>(s.width) / s.rate}, {"cycles_of_centre", static_cast<double>(s.width) / s.rate * centre}};
    } else {
      // FRAME DENSITY is an option of the node (frames per tile or per cycle; the pipeline's default is written in its YAML): never a constant here
      const bool per_tile = p["frames_per_tile"].is_number(), per_cycle = p["frames_per_cycle"].is_number();
      need(per_tile || per_cycle, who + ": give frames_per_tile (a power of two; it wins when both are given) or frames_per_cycle (no default here: the definition's)");
      s.stated = Json{{"band", p["band"]}, {"frames_per_tile", p["frames_per_tile"]}, {"frames_per_cycle", p["frames_per_cycle"]}};
      if (per_tile) {
        const std::int64_t fpt = p["frames_per_tile"].get<std::int64_t>();
        need(is_power_of_two(fpt), who + ": frames_per_tile " + std::to_string(fpt) + " is a power of two (2^s frames to the band's tile)");
        const int k = kb - log2_exact(fpt);
        need(k >= 0, who + ": " + std::to_string(fpt) + " frames to a tile of 2^" + std::to_string(kb) + " samples is finer than one sample");
        level_grid(k);
        s.snapped = Json{{"frame_s", static_cast<double>(s.width) / s.rate}, {"frames_per_tile", static_cast<double>(fpt)}, {"frames_per_cycle", 1.0 / (centre * (static_cast<double>(s.width) / s.rate))}};
      } else {
        // FRAMES PER CYCLE, EXACTLY (36 to a cycle, the 10 degree step of phase work, is not a power of two and is never snapped to one): the frame is 1 / (centre x frames_per_cycle) seconds,
        // by compute's time rule; the hop is generally not whole samples, a frame's start is origin + j hop (the frames node interpolates there)
        const double fpc = p["frames_per_cycle"].get<double>();
        need(fpc >= 1, who + ": frames_per_cycle is at least 1");
        nxr::select::TimeRule tr;
        tr.kind = nxr::select::TimeRule::Kind::frames;
        tr.band_hz = std::make_pair(lo, hi);
        tr.frames_per_cycle = fpc;
        const ns::Line ln{"s", room, s.line_origin + static_cast<double>(s.origin) / s.rate, 1.0 / s.rate};
        nxr::select::TimeGrid made;
        try { made = nxr::select::time_rule_resolved(ln, tr); } catch (const std::invalid_argument& e) { refuse(who + ": " + e.what()); }
        // a frame finer than the sampling (36 to a cycle at a high frequency) is allowed: its sample is made by band-limited interpolation (`frames`), so hop and width in whole samples are only the nearest (at least 1)
        s.hop_exact = s.width_exact = made.hop_s * s.rate;
        s.width = made.width_samples > 0 ? made.width_samples : std::max<std::int64_t>(1, static_cast<std::int64_t>(std::llround(s.width_exact)));
        s.hop = made.hop_samples > 0 ? made.hop_samples : std::max<std::int64_t>(1, static_cast<std::int64_t>(std::llround(s.hop_exact)));
        s.level = made.level;
        s.count = made.grid.tiling->count;
        s.snapped = Json{{"frame_s", made.hop_s}, {"frames_per_cycle", fpc}, {"frame_samples", s.hop_exact}};
      }
    }
  } else if (s.rule == "chunks") {
    level_grid(15);                                                        // the chunk grid (io's chunk service: 2^15 samples)
    s.snapped = Json{{"chunk_width", s.width}};
  } else if (s.rule == "windows") {
    const nxr::quantity::Quantity w = parse_q(p["window"], who, "window"), h = parse_q(p["step"], who, "step");
    need(w.unit == "s" && h.unit == "s", who + ": window and step are in seconds");
    const auto rw = nxr::select::level_of_duration(w.hi, s.rate), rh = nxr::select::level_of_duration(h.hi, s.rate);   // width and hop snapped to 2^k samples
    s.level = rw.level;
    s.width = std::int64_t(1) << rw.level;
    s.hop = std::int64_t(1) << rh.level;
    s.hop_exact = static_cast<double>(s.hop); s.width_exact = static_cast<double>(s.width);
    s.count = room < s.width ? 0 : (room - s.width) / s.hop + 1;
    s.stated = Json{{"window", p["window"]}, {"step", p["step"]}};
    s.snapped = Json{{"window_s", rw.snapped}, {"step_s", rh.snapped}};
  } else if (s.rule == "segments") {
    // members OVERLAP (a family, not a partition): width as Welch's presets resolve it, hop = width - floor(width x overlap) (compute's time_rule)
    const nxr::quantity::Quantity d = parse_q(p["segment"], who, "segment");
    need(d.unit == "s", who + ": segment is a duration in seconds");
    need(p["overlap"].is_number() && p["overlap"].get<double>() >= 0 && p["overlap"].get<double>() < 1, who + ": overlap is a fraction in [0, 1)");
    nxr::select::TimeRule tr;
    tr.kind = nxr::select::TimeRule::Kind::segments;
    tr.duration_s = d.hi;
    tr.overlap = p["overlap"].get<double>();
    tr.segment_rule = p["preset"].is_string() ? p["preset"].get<std::string>() : std::string("tile");   // tile | round | even (welch::segmentFor)
    const ns::Line ln{"s", s.size, s.line_origin, 1.0 / s.rate};
    nxr::select::TimeGrid made;
    try { made = nxr::select::time_rule_resolved(ln, tr); } catch (const std::invalid_argument& e) { refuse(who + ": " + e.what()); }
    need(made.grid.tiling.has_value(), who + ": the segments rule made no tiling");
    s.width = made.width_samples;
    s.hop = made.hop_samples;
    s.hop_exact = static_cast<double>(s.hop);
    s.count = made.grid.tiling->count;
    s.stated = Json{{"segment", p["segment"]}, {"overlap", p["overlap"]}, {"preset", tr.segment_rule}};
    s.snapped = Json{{"segment_s", static_cast<double>(s.width) / s.rate}, {"hop_s", static_cast<double>(s.hop) / s.rate}};
  } else if (s.rule == "constant_q") {
    // the ladder's levels: one per distinct window, each carrying the Studio's ConstantQLevelInfo attributes (rule 24: low_hz, high_hz, centre, window, ...)
    nxr::select::ConstantQ m;
    if (p["method"].is_string()) m.method = p["method"].get<std::string>();
    if (p["voices"].is_number_integer()) m.voices = p["voices"].get<int>();
    if (p["cycles"].is_number()) m.cycles = p["cycles"].get<double>();
    if (p.contains("range") && p["range"].is_string()) { const auto q = parse_q(p["range"], who, "range"); m.f_min = q.lo; m.f_max = q.hi; }   // "0.5-64 Hz": the ladder's range
    if (p["f_min"].is_string()) m.f_min = parse_q(p["f_min"], who, "f_min").hi;
    if (p["f_max"].is_string()) m.f_max = parse_q(p["f_max"], who, "f_max").hi;
    const ns::Line ln{"s", s.size, s.line_origin, 1.0 / s.rate};
    nxr::select::ConstantQDesign d;
    try { d = nxr::select::constant_q_design(ln, m); } catch (const std::invalid_argument& e) { refuse(who + ": " + e.what()); }
    Json levels = Json::array();
    for (const auto& l : d.levels)
      levels.push_back(Json{{"level", l.power}, {"window_samples", l.window_samples}, {"window_s", static_cast<double>(l.window_samples) / s.rate}, {"bands", l.bands}, {"low_hz", l.lo}, {"high_hz", l.hi},
                            {"centre", std::sqrt(std::max(l.lo, 1e-12) * l.hi)}, {"stride", l.stride}, {"rate_hz", l.rate_hz}, {"points", l.points}, {"samples_per_cycle_lo", l.samples_per_cycle_lo},
                            {"samples_per_cycle_hi", l.samples_per_cycle_hi}, {"nyquist_bound", l.nyquist_bound}});
    s.levels = levels;
    s.design = Json{{"method", d.method}, {"voices", d.voices}, {"cycles", d.cycles}, {"f_min", d.f_min}, {"f_max", d.f_max}, {"sfreq", d.sfreq}};
    s.stated = Json{{"f_min", p["f_min"]}, {"f_max", p["f_max"]}, {"voices", p["voices"]}};
    s.snapped = Json{{"f_min", d.f_min}, {"f_max", d.f_max}, {"levels", static_cast<std::int64_t>(d.levels.size())}};
    s.level = d.levels.empty() ? -1 : d.levels.front().power;
  } else refuse("select.time_grid: rule '" + s.rule + "': tiles, band, frames, constant_q, segments, windows or chunks");
  // THE RULE'S NAME: the stated quantities in a canonical form (a band as `lo-hi Hz`, a duration as seconds), so a consumer that states the same quantity finds the grid by name
  auto band_text = [&](const Json& b) { const auto [lo, hi] = hz_band(b, who); return g(lo) + "-" + g(hi) + " Hz"; };
  auto seconds_text = [&](const Json& d, const char* what) { const nxr::quantity::Quantity q = parse_q(d, who, what); return g(q.hi) + " s"; };
  if (s.rule == "tiles") s.rule_name = "tiles " + seconds_text(p["duration"], "duration");
  else if (s.rule == "band") s.rule_name = "band " + band_text(p["band"]);
  else if (s.rule == "frames")
    s.rule_name = "frames " + band_text(p["band"]) + (p["frames_per_tile"].is_number() ? " " + g(p["frames_per_tile"].get<double>()) + " per tile" : " " + g(p["frames_per_cycle"].get<double>()) + " per cycle");
  else if (s.rule == "constant_q") s.rule_name = "constant_q " + g(s.design["f_min"].get<double>()) + "-" + g(s.design["f_max"].get<double>()) + " Hz " + g(s.design["voices"].get<double>()) + " voices";
  else if (s.rule == "segments") s.rule_name = "segments " + seconds_text(p["segment"], "segment") + " overlap " + g(p["overlap"].get<double>());
  else if (s.rule == "windows") s.rule_name = "windows " + seconds_text(p["window"], "window") + " step " + seconds_text(p["step"], "step");
  else s.rule_name = s.rule;
  if (s.origin != 0) s.rule_name += " from sample " + std::to_string(s.origin);
  return s;
}

StepSpec time_grid_step() {
  StepSpec s;
  s.name = "select.time_grid";
  s.kind = "setup";
  s.inputs = {"line", "event"};
  s.optional = {"event"};
  s.outputs = {"grid"};
  s.parameters = Json{{"rule", nullptr}, {"duration", nullptr}, {"band", nullptr}, {"frames_per_tile", nullptr}, {"frames_per_cycle", nullptr}, {"origin", nullptr}, {"cycles", nullptr},
                      {"window", nullptr}, {"step", nullptr}, {"segment", nullptr}, {"overlap", nullptr}, {"preset", nullptr},
                      {"method", nullptr}, {"voices", nullptr}, {"range", nullptr}, {"f_min", nullptr}, {"f_max", nullptr}};
  s.matches = "the Cortical Flow Studio's time grid (jointTime.ts fitLevelOf, tower frames at the band's level), Rheome's natural_level and jointladder, scipy.signal.welch's segments; compute nxr::selection::time_grid, time_rule, constant_q_design";
  s.description =
      "The members of a time line's grid: member i is [origin + i hop, origin + i hop + width), counted in the line's samples. One rule family: `tiles` (a duration: the first level whose 2^k-sample tile holds it), "
      "`band` (a band in Hz: the tile holding 4 cycles of its centre, the Studio's fitLevelOf), `frames` (a band and frames_per_tile, a power of two, or frames_per_cycle: the frame grid of a band), `constant_q` "
      "(a frequency range and voices: the Studio's constant-Q ladder), `segments` (a duration and an overlap: Welch's segments), `windows` (a window and a step, each snapped to 2^k samples) and `chunks` (2^15 samples). "
      "`origin` is an option: the line's first sample (default), a clock time (\"12.5 s\") or `event` (the first member of the spans selection given as `event`). Every parameter is a quantity: a bare level "
      "index is never a parameter. The level is resolved in closed form and the SNAPPED quantity (what the level actually is) is recorded. A grid made by tiles, band or frames is one LEVEL of the line's dyadic time "
      "grid: nothing per level is stored.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "no time axis: runs once on numbers (the line's rate and length); tiles of 2^k samples are exact in binary");
  s.plan = [](const Values& ins, const Json& p, const PlanContext& ctx) {
    need(p["rule"].is_string(), "select.time_grid: rule is tiles, band, frames, constant_q, segments, windows or chunks (no default)");
    const TimeGridState t = make_time_grid(ins, p, ctx.bound);
    Value v;
    v.cls = "selection";
    v.manifolds = {*ins.at("line").time};
    v.facts = Json{{"node", "select.time_grid"}, {"port", "grid"}, {"rule", t.rule}, {"names", Json::object()}, {"family", t.rule == "segments" ? "irregular" : "grid"}};
    v.facts["names"]["grid_rule"] = t.rule_name;
    if (t.width > 0) v.facts["names"]["tile_s"] = g(t.width_exact / t.rate);
    v.facts["window_samples"] = t.width; v.facts["tiles"] = t.count; v.facts["width_s"] = t.width_exact / t.rate; v.facts["origin"] = t.line_origin + static_cast<double>(t.origin) / t.rate;
    if (!t.band.is_null()) v.facts["resolves_hz"] = Json{{"lo", t.band[0]}, {"hi", t.band[1]}};          // the level IS the band (D84): what detection and the events' units read
    v.facts["hop_s"] = t.hop_exact / t.rate;                      // what a node after it reads of the grid without a stored row (the frame rate, the first member's time)
    v.facts["origin_s"] = t.line_origin + static_cast<double>(t.origin) / t.rate;
    v.facts["level"] = t.level;
    v.facts["snapped"] = t.snapped;
    return Values{{"grid", v}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) { return std::any(make_time_grid(ins, p, ctx.bound)); };
  s.run = [](const std::any& st, const Arrays&, ExecContext&) {
    const auto& t = std::any_cast<const TimeGridState&>(st);
    const double origin_s = t.line_origin + static_cast<double>(t.origin) / t.rate;
    Json grid = Json{{"kind", "time"}, {"origin", t.origin}, {"width", t.width}, {"hop", t.hop}, {"hop_exact", t.hop_exact}, {"width_exact", t.width_exact}, {"count", t.count}, {"size", t.size}, {"level", t.level}, {"rate", t.rate}, {"rule", t.rule}};
    Json params = Json{{"grid", grid}, {"quantity", Json{{"stated", t.stated}, {"snapped", t.snapped}}}, {"rule", t.rule}, {"rule_name", t.rule_name}};
    if (!t.band.is_null()) params["band"] = t.band;
    if (!t.levels.is_null()) { params["levels"] = t.levels; params["design"] = t.design; }
    Json desc = Json{{"params", params},
                     {"family", t.rule == "segments" ? "irregular" : "grid"},
                     {"tiling", Json{{"origin", origin_s}, {"width", t.width_exact / t.rate}, {"hop", t.hop_exact / t.rate},
                                     {"end", origin_s + static_cast<double>(t.count) * t.hop_exact / t.rate}, {"count", t.count}}},
                     {"description", "the " + t.rule + " grid of " + t.line_name + ": " + std::to_string(t.count) + " members of " + std::to_string(t.width) + " samples, hop " + std::to_string(t.hop) + ", from sample " +
                                         std::to_string(t.origin) + (t.level >= 0 ? " (level " + std::to_string(t.level) + ")" : std::string())}};
    Array out;
    out.meta = Json{{"describe", desc}};
    Arrays o;
    o["grid"] = std::move(out);
    return o;
  };
  return s;
}

// ── select.bands (native) ────────────────────────────────────────────────────────────────────────────────────────
// A band set: named spans on the Frequency line (Hz) or the Eigenvalue line (1/m^2), the names its lookup. Rules frequency, eigenvalue, wavelength, parts, levels. A single band SELECTS; several
// are a band SET (one selection; a measurement's members include the bands, never a run per band). The set travels on its edge as its TYPE (members, in code order); a consumer picks a member by
// code or name and never re-makes the set.
struct BandsPrepared { std::string rule; std::vector<Json> members; Json piece = nullptr; bool grid = false; };

ng::PortType constant_port(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}

// a wavelength (mm) as the eigenvalue (1/m^2) it holds: lambda = (2 pi / w)^2, w in metres (the chart `wavelength`'s power transition)
double eigenvalue_of_wavelength_mm(double w_mm) { const double w = w_mm * 1e-3; return std::pow(2.0 * M_PI / w, 2.0); }

BandsPrepared bands_of(const ng::Types& in, const ng::Arrays* constants, const Json& p) {
  BandsPrepared b;
  b.rule = p.value("rule", std::string());
  const std::string who = "select.bands " + b.rule;
  const bool names_given = p["names"].is_array();
  const Json list = p["bands"];
  auto name_of = [&](std::size_t k, const std::string& fallback) { return names_given && k < p["names"].size() ? p["names"][k].get<std::string>() : fallback; };
  if (names_given && list.is_array()) need(p["names"].size() == list.size(), who + ": names has one name per band");
  if (b.rule == "frequency") {
    need(list.is_array() && !list.empty(), who + ": bands is a list of bands in a frequency chart (\"8-12 Hz\"); no default: the caller chooses, e.g. --bands '[\"8-12 Hz\"]'");
    for (std::size_t k = 0; k < list.size(); ++k) {
      const auto [lo, hi] = hz_band(list[k], who);
      need(hi > lo && lo >= 0, who + ": band " + list[k].dump() + " needs 0 <= lo < hi");
      b.members.push_back(Json{{"code", k}, {"name", name_of(k, g(lo) + "-" + g(hi) + " Hz")}, {"lo", lo}, {"hi", hi}, {"unit", "Hz"}, {"chart", "linear"}});
    }
  } else if (b.rule == "eigenvalue" || b.rule == "wavelength") {
    need(list.is_array() && !list.empty(), who + ": bands is a list of spans in the " + b.rule + " chart; no default: pass them with --bands '[...]'");
    for (std::size_t k = 0; k < list.size(); ++k) {
      const nxr::quantity::Quantity q = parse_q(list[k], who, "band");
      double lo, hi;
      if (b.rule == "wavelength") {
        need(q.chart == "wavelength" || q.unit == "mm", who + ": " + list[k].dump() + " is not a wavelength (mm)");
        lo = eigenvalue_of_wavelength_mm(q.hi);                         // the longer wavelength is the lower eigenvalue
        hi = eigenvalue_of_wavelength_mm(q.lo > 0 ? q.lo : q.hi);
      } else { need(q.chart == "eigenvalue" || q.unit == "1/m2", who + ": " + list[k].dump() + " is not an eigenvalue span (1/m2)"); lo = q.lo; hi = q.hi; }
      need(hi >= lo && lo >= 0, who + ": band " + list[k].dump() + " is empty");
      b.members.push_back(Json{{"code", k}, {"name", name_of(k, list[k].get<std::string>())}, {"lo", lo}, {"hi", hi}, {"unit", "1/m2"}, {"chart", "eigenvalue"}});
    }
  } else if (b.rule == "parts" || b.rule == "levels") {
    // a dyadic level's Weyl band [4 pi 2^(l-1) / A, 4 pi 2^l / A), level 0 from 0, on the mean connected piece's area A (the grid's record)
    need(in.count("grid"), who + ": the grid (a surface's dyadic grid: its record names the depth, area and pieces) is not given");
    b.grid = true;
    if (!constants) return b;                                          // the plan (types only): the grid's record is read at the run, its levels are the members then
    need(constants->count("grid") && !constants->at("grid").meta.value("placeholder", false), who + ": the grid was not handed over");
    const Json gp = constants->at("grid").meta.value("params", Json::object());
    need(gp.is_object() && gp.contains("depth") && gp.contains("area_m2") && gp.contains("pieces"), who + ": the grid records no depth, area or pieces");
    const double A = gp["area_m2"].get<double>() / std::max(1, gp["pieces"].get<int>());
    need(A > 0, who + ": the grid's pieces have no area");
    b.piece = Json{{"area_m2", A}, {"depth", gp["depth"]}};
    std::vector<int> levels;
    if (b.rule == "levels") for (int l = 0; l <= gp["depth"].get<int>(); ++l) levels.push_back(l);
    else {
      need(list.is_array() && !list.empty(), who + ": bands is a list of parts per hemisphere (\"parts 16\"); no default");
      for (const auto& e : list) {
        const nxr::quantity::Quantity q = parse_q(e, who, "scale");
        need(q.chart == "parts", who + ": " + e.dump() + " is not a number of parts per hemisphere (\"parts 16\")");
        levels.push_back(nxr::select::level_of_parts(q.hi, gp["depth"].get<int>()).level);
      }
    }
    for (std::size_t k = 0; k < levels.size(); ++k) {
      const int l = levels[k];
      const double hi = 4.0 * M_PI / A * std::ldexp(1.0, l), lo = l == 0 ? 0.0 : hi / 2;
      b.members.push_back(Json{{"code", b.rule == "levels" ? l : static_cast<int>(k)}, {"name", name_of(k, "level " + std::to_string(l))}, {"lo", lo}, {"hi", hi}, {"unit", "1/m2"}, {"chart", "eigenvalue"}, {"level_of_grid", l}});
    }
  } else refuse("select.bands: rule '" + b.rule + "': frequency, eigenvalue, wavelength, parts or levels");
  return b;
}

ng::NodeSpec select_bands() {
  ng::NodeSpec n;
  n.name = "select.bands";
  n.kind = ng::Kind::array;
  n.inputs = {{"line", "the dataset's Frequency or Eigenvalue line the spans lie on (its record; the one line, never made per recording)", true},
              {"grid", "rules parts and levels: the surface's dyadic grid (its record: depth, area, pieces); each level is its Weyl band", true}};
  n.outputs = {{"bands", "the named band set: spans on the line, their names, bounds, unit and chart the lookup", false}};
  n.parameters = {{"rule", Json(), "frequency | eigenvalue | wavelength | parts | levels", "-", {"frequency", "eigenvalue", "wavelength", "parts", "levels"}, false},
                  {"bands", Json(), "the bands as quantities in the rule's chart: \"8-12 Hz\" (frequency; a log10 chart too), \"1e3-4e3 1/m2\" (eigenvalue), \"20-40 mm\" (wavelength), \"parts 16\" (parts); no default", "-", {}, true},
                  {"names", Json(), "one name per band; none: the quantity as written", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a selection made from parameters: no data flows");
  n.matches = "nxr convert.frequency and the spectral bank (D158); the Studio's coordinate charts (space.ts: linear, log10; eigenvalue, wavelength); Weyl's law on the grid's piece area; compute nxr::selection bands";
  n.description = "a named band set as ONE selection of spans on the Frequency or Eigenvalue line (frequency, eigenvalue, wavelength, parts: listed; levels: a dyadic grid's Weyl bands). A single band selects; several "
                  "are a band SET, never a run or a field per band. Quantities only: a bare level index is never a parameter.";
  n.plan = [](const ng::Types& in, const Json& p) {
    need(p["rule"].is_string(), "select.bands: rule is frequency, eigenvalue, wavelength, parts or levels (no default)");
    const BandsPrepared b = bands_of(in, nullptr, p);                      // refuses a bad band before any data
    ng::Planned out;
    ng::PortType t = constant_port("selection");
    Json members = Json::array();
    for (const auto& m : b.members) members.push_back(m);
    t.facts["set"] = Json{{"type", "spans"}, {"unit", b.members.empty() ? "Hz" : b.members[0].value("unit", std::string("Hz"))}, {"members", members}, {"rule", b.rule}, {"family", b.rule == "levels" ? "grid" : "irregular"}};
    t.facts["names"] = Json{{"line", b.rule == "frequency" ? "Frequency" : "Eigenvalue"}};   // the shared line of the default subject the set lies on (the output's `on: default.{line}`)
    out.out["bands"] = t;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) { return std::any(bands_of(in, &constants, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const BandsPrepared& b = std::any_cast<const BandsPrepared&>(prepared);
    ng::Array a;
    a.rows = static_cast<std::int64_t>(b.members.size()); a.cols = 1;
    Json spans = Json::array(), entries = Json::array(), lookup = Json::array();
    for (const auto& m : b.members) {
      spans.push_back(Json{{"start", m["lo"]}, {"stop", m["hi"]}, {"code", m["code"]}});
      entries.push_back(m["name"]);
      lookup.push_back(m);
      a.data.push_back(m["code"].get<double>());
    }
    const std::string unit = b.members.empty() ? "Hz" : b.members[0]["unit"].get<std::string>();
    Json params = Json{{"grid", Json{{"kind", "bands"}, {"rule", b.rule}, {"unit", unit}, {"chart", b.members.empty() ? "linear" : b.members[0]["chart"]}}}};
    if (!b.piece.is_null()) params["spatial_bands"] = Json{{"levels", b.members.size()}, {"piece_area_m2", b.piece["area_m2"]}, {"unit", "1/m^2"}};
    a.meta["describe"] = Json{{"spans", spans}, {"entries", entries}, {"lookup", lookup}, {"unit", unit}, {"family", b.rule == "levels" ? "grid" : "irregular"}, {"params", params},
                              {"description", "the " + b.rule + " band set: " + std::to_string(b.members.size()) + " spans on the " + std::string(unit == "Hz" ? "Frequency" : "Eigenvalue") + " line"}};
    return ng::Arrays{{"bands", std::move(a)}};
  };
  return n;
}

// ── select.product (native) ──────────────────────────────────────────────────────────────────────────────────────
// A PRODUCT of grids on the product manifold (the Studio's D100 productSelection): mixed radix, the FIRST factor fastest (nxr::selection::product, product_code(s)); layout `joint` is D129: the
// first factor (the frames) gets a radix of 2^bits, a code is (tile << bits) | frame. Factors are grids (a time grid, a surface level, an elements grid, the run's span) or, as D100's atlas x
// windows example, an irregular set; roll-ups only along grid factors. COUPLING: with `couple`, a factor's level is the band's (a frequency band sets the time level, the Studio's fitLevelOf; an
// eigenvalue band sets the dyadic spatial level by Weyl's law), computed in closed form here and recorded in the factor, never stored as a selection of its own.
ng::Json factor_descriptor(const ng::Array& f, const std::string& who) {
  const Json prm = f.meta.contains("describe") && f.meta["describe"].is_object() && f.meta["describe"].contains("params") ? f.meta["describe"]["params"] : f.meta.value("params", Json::object());   // made in this run (describe) or stored (params)
  if (prm.is_object() && prm.contains("grid")) {
    Json g = prm["grid"];
    if (g.value("kind", std::string()) == "dyadic") {                         // a whole surface ladder: its surface (the name less " tiles"), piece area and pieces: a coupled level is resolved from them (Weyl's law, closed form)
      const std::string nm = f.meta.value("name", std::string());
      const auto at = nm.rfind(" tiles");
      if (at != std::string::npos && !g.contains("surface")) g["surface"] = nm.substr(0, at);
      if (prm.contains("area_m2") && !g.contains("area_m2")) g["area_m2"] = prm["area_m2"];
      if (prm.contains("length_unit")) g["length_unit"] = prm["length_unit"];
    } else if (g.value("kind", std::string()) == "time" && g.contains("samples") && !g.contains("size")) {   // a whole time ladder: its samples are its size
      g["size"] = g["samples"];
    }
    return g;
  }
  if (prm.is_object() && prm.contains("level") && prm.contains("tiles")) {
    std::string surface = prm.value("surface", f.meta.value("surface", std::string()));
    if (surface.empty()) {                                                  // `<surface> tiles L<l>`
      const std::string nm = f.meta.value("name", std::string());
      const auto at = nm.rfind(" tiles L");
      surface = at == std::string::npos ? std::string() : nm.substr(0, at);
    }
    need(!surface.empty(), who + ": the surface level " + f.meta.value("name", std::string()) + " names no surface");
    return Json{{"kind", "dyadic"}, {"surface", surface}, {"level", prm["level"]}, {"tiles", prm["tiles"]}, {"family", "grid"}};
  }
  if (f.meta.value("n_members", std::int64_t(0)) > 0 && prm.value("kind", std::string()) == "elements") return Json{{"kind", "elements"}, {"size", f.meta["n_members"]}};
  // a stored PARTITION (an irregular set: regions, lobes, an atlas): its members as they are (D100's atlas x windows), no level to roll up along
  if (f.meta.value("type", std::string()) == "set" && f.meta.value("n_members", std::int64_t(0)) > 0) return Json{{"kind", "set"}, {"size", f.meta["n_members"]}, {"family", "irregular"}};
  refuse(who + ": the factor " + f.meta.value("name", std::string("?")) + " records no grid (params.grid), surface level or elements grid");
}

struct ProductPrepared { std::vector<Json> factors; std::string layout; int bits = 0; std::vector<std::int64_t> radices; std::vector<std::int64_t> counts; Json coupled = Json::array(); };

ng::NodeSpec select_product() {
  ng::NodeSpec n;
  n.name = "select.product";
  n.kind = ng::Kind::array;
  n.inputs = {{"first", "the first factor (it varies fastest): a grid level, a band set, an elements grid or an irregular set", false},
              {"second", "the second factor", false},
              {"third", "a third factor (optional)", true},
              {"fourth", "a fourth factor (optional)", true},
              {"bands", "optional: the band set whose members COUPLE levels (`couple`)", true}};
  n.outputs = {{"product", "the product manifold the cells lie on", false}, {"cells", "the product selection: every combination of the factors' members, coded mixed radix, the FIRST factor fastest", false}};
  n.parameters = {{"layout", "product", "product (mixed radix) or joint (the Studio's D129: the first factor's radix 2^bits, code (tile << bits) | frame)", "-", {"product", "joint"}, false},
                  {"couple", Json(), "factors whose level the band sets: a list of {factor: index, band: name or code of the `bands` set}; a frequency band sets a time factor's level, an eigenvalue band a surface factor's", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "integer codes from the factors' counts");
  n.matches = "the Studio's D100 productSelection and D129 joint partition; compute nxr::selection::product, joint, product_code(s); fitLevelOf (jointTime.ts) and Weyl's law for the coupled levels";
  n.description = "the product of grids on the product manifold, mixed radix with the first factor fastest, or the joint layout; a band COUPLES a factor's level (closed form, recorded in the factor); roll-ups only along grid factors";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"first", "second"}) if (!in.count(k)) ng::refuse(std::string("select.product: ") + k + " is not given (a product has two factors or more)");
    need(p["layout"] == "product" || p["layout"] == "joint", "select.product: layout " + p["layout"].dump() + ": product or joint");
    ng::Planned out;
    out.out["product"] = constant_port("manifold");
    out.out["cells"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ProductPrepared pp;
    pp.layout = p["layout"].get<std::string>();
    for (const char* k : {"first", "second", "third", "fourth"}) {
      if (!c.count(k)) continue;
      need(!c.at(k).meta.value("placeholder", false), std::string("select.product: ") + k + " was not handed over");
      Json d = factor_descriptor(c.at(k), "select.product");
      d["name"] = c.at(k).meta.value("name", std::string());
      d["id"] = c.at(k).meta.value("id", std::string());
      pp.factors.push_back(std::move(d));
    }
    // COUPLED levels, in closed form. A factor's level is set by a BAND of the `bands` set (by name), or by a QUANTITY stated here: `band` (a frequency band, "8-16 Hz") for a time factor, `scale` (parts per hemisphere,
    // a wavelength, an eigenvalue or an area) for a surface factor. The resolvers are select's: time_level_of_band, space_level_of_band, level_of_quantity.
    if (p["couple"].is_array()) {
      const bool have_set = c.count("bands") && !c.at("bands").meta.value("placeholder", false);
      const Json set = have_set ? c.at("bands").meta.value("describe", Json::object()) : Json::object();
      for (const auto& cp : p["couple"]) {
        const std::size_t fi = cp.value("factor", std::size_t(0));
        need(fi < pp.factors.size(), "select.product: couple names factor " + std::to_string(fi) + " of " + std::to_string(pp.factors.size()));
        Json& f = pp.factors[fi];
        if (cp.contains("scale")) {                                                    // a surface factor's level from a stated scale
          need(f["kind"] == "dyadic", "select.product: `scale` couples a surface factor's level; factor " + std::to_string(fi) + " is of kind " + f["kind"].dump());
          need(f.contains("area_m2") && f.contains("pieces") && f.contains("depth"), "select.product: couple on a surface needs the whole grid (its record: area, pieces and depth), not one level");
          const int depth = f["depth"].get<int>();
          const double A = f["area_m2"].get<double>() / std::max<std::int64_t>(1, f["pieces"].get<std::int64_t>());   // a piece's area: the Weyl level 0
          nxr::selection::Selection grid;
          for (int l = 0; l <= depth; ++l) { nxr::selection::Selection m; m.tile_area = A / std::ldexp(1.0, l); grid.members.push_back(m); }
          Json sn = Json::object();
          const std::string unit = f.value("length_unit", std::string("m"));
          f["level"] = level_of_quantity(grid, cp["scale"], unit == "m" ? "m" : unit, "select.product", sn);
          f["snapped"] = Json{{"scale", cp["scale"]}, {"snapped", sn}};
          f["tiles"] = f["pieces"].get<std::int64_t>() << f["level"].get<int>();            // the tiles of a level: every piece's 2^level (`reduce` checks it against the surface's own level)
          pp.coupled.push_back(Json{{"factor", fi}, {"scale", cp["scale"]}, {"level", f["level"]}});
          continue;
        }
        need(cp.contains("band"), "select.product: a couple entry is {factor, band} or {factor, scale}");
        double lo = 0, hi = 0;
        std::string unit_of_band = "Hz";
        const Json* band = nullptr;
        for (const auto& m : set.value("lookup", Json::array())) if (cp["band"] == m["name"] || (cp["band"].is_number() && cp["band"] == m["code"])) band = &m;
        if (band) { lo = (*band)["lo"].get<double>(); hi = (*band)["hi"].get<double>(); unit_of_band = (*band)["unit"].get<std::string>(); }
        else {
          need(cp["band"].is_string() && !have_set, "select.product: the band " + cp["band"].dump() + " is not a member of the set");
          std::tie(lo, hi) = hz_band(cp["band"], "select.product");
        }
        if (f["kind"] == "time") {
          need(unit_of_band == "Hz", "select.product: a time factor's level is set by a frequency band; " + cp["band"].dump() + " is in " + unit_of_band);
          const int k = nxr::select::time_level_of_band(lo, hi, f["rate"].get<double>(), nxr::select::kJointCycles, static_cast<int>(std::floor(std::log2(static_cast<double>(f["size"].get<std::int64_t>())))));
          f["level"] = k;
          f["width"] = f["hop"] = std::int64_t(1) << k;
          f["count"] = (f["size"].get<std::int64_t>() - f.value("origin", std::int64_t(0)) + f["hop"].get<std::int64_t>() - 1) / f["hop"].get<std::int64_t>();
          f["snapped"] = Json{{"tile_s", static_cast<double>(f["width"].get<std::int64_t>()) / f["rate"].get<double>()}, {"band", cp["band"]}};
        } else if (f["kind"] == "dyadic") {
          need(band != nullptr && unit_of_band == "1/m2", "select.product: a surface factor's level is set by an eigenvalue band of the set, or by a `scale`; " + cp["band"].dump() + " is in " + unit_of_band);
          need(f.contains("piece_area_m2") && f.contains("depth"), "select.product: couple on a surface needs the whole grid (its record: piece area and depth), not one level");
          f["level"] = nxr::select::space_level_of_band(lo, hi, f["piece_area_m2"].get<double>(), f["depth"].get<int>());
          f["snapped"] = Json{{"band", cp["band"]}};
        } else refuse("select.product: a factor of kind " + f["kind"].dump() + " has no level to couple");
        pp.coupled.push_back(Json{{"factor", fi}, {"band", cp["band"]}, {"level", f["level"]}});
      }
    }
    for (const auto& f : pp.factors) {
      std::int64_t m = 0;
      if (f["kind"] == "time") m = f["count"].get<std::int64_t>();
      else if (f["kind"] == "dyadic") m = f["tiles"].get<std::int64_t>();
      else if (f["kind"] == "elements" || f["kind"] == "kernels" || f["kind"] == "set") m = f["size"].get<std::int64_t>();   // a kernel set: its members (a frequency or scale filterbank)
      else if (f["kind"] == "span" || f["kind"] == "whole") m = 1;
      need(m > 0, "select.product: the factor " + f.value("name", std::string("?")) + " has no members");
      pp.counts.push_back(m);
      pp.radices.push_back(m);
    }
    if (pp.layout == "joint") {
      pp.bits = log2_exact(pp.radices[0]);
      pp.radices[0] = std::int64_t(1) << pp.bits;
      std::int64_t total = 1;
      for (const auto r : pp.radices) total *= r;
      need(total < (std::int64_t(1) << 31), "select.product: the joint code space (" + std::to_string(total) + ") does not fit a 32-bit code");
    }
    return std::any(std::move(pp));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ProductPrepared& pp = std::any_cast<const ProductPrepared&>(prepared);
    std::int64_t members = 1;
    bool all_grids = true;
    std::string label;
    Json factors = Json::array();
    for (std::size_t i = 0; i < pp.factors.size(); ++i) {
      members *= pp.radices[i];
      all_grids = all_grids && pp.factors[i].value("family", std::string("grid")) == "grid";
      label += (label.empty() ? "" : " x ") + std::to_string(pp.counts[i]) + " " + pp.factors[i].value("name", std::string("members"));
      factors.push_back(pp.factors[i]);
    }
    Json product = Json{{"factors", factors}, {"layout", pp.layout}, {"bits", pp.bits}, {"radices", pp.radices}, {"counts", pp.counts}, {"code", pp.layout == "joint" ? "(tile << bits) | frame" : "c0 + n0 (c1 + n1 (..))"}};
    if (!pp.coupled.empty()) product["coupled"] = pp.coupled;
    ng::Array a;
    a.meta["describe"] = Json{{"n_members", members}, {"label", label}, {"family", all_grids ? "grid" : "irregular"}, {"params", Json{{"product", product}}},
                              {"description", "the product " + label + ", FIRST factor fastest (" + pp.layout + "); roll-ups only along grid factors"}};
    ng::Array space;
    space.meta["describe"] = Json{{"description", "the product manifold of " + label}};
    return ng::Arrays{{"product", std::move(space)}, {"cells", std::move(a)}};
  };
  return n;
}

}  // namespace

// The depth of a grid read from its level dictionary (members: level l at position l) and a piece's area: from the level 0 tile area when it is recorded, else from a coarser level's Weyl band
// (its lower edge is 4 pi 2^(l-1) / A, spec 3.3), else refused: the Weyl levels need it.
static double piece_area_of(const nxr::selection::Selection& grid) {
  if (!grid.members.empty() && grid.members.front().tile_area) return *grid.members.front().tile_area;
  for (std::size_t l = 1; l < grid.members.size(); ++l)
    if (grid.members[l].band && grid.members[l].band->first > 0) return 4.0 * std::acos(-1.0) * std::ldexp(1.0, static_cast<int>(l) - 1) / grid.members[l].band->first;
  return 0.0;
}

int level_of_quantity(const nxr::selection::Selection& grid, const Json& finest, const std::string& unit, const std::string& who, Json& snapped) {
  need(finest.is_string(), who + ": a scale is a quantity with a unit (a tile diameter \"20 mm\", an area \"400 mm^2\", a wavelength \"wavelength 40 mm\", \"parts 16\", an eigenvalue \"eigenvalue 1/m^2\", or \"index:5\" for diagnostics), not " + finest.dump());
  nxr::quantity::Quantity q = parse_q(finest, who, "scale");
  const int depth = static_cast<int>(grid.members.size()) - 1;                   // the level dictionary: levels 0 .. depth
  try {
    if (q.chart == "index") { snapped = Json{{"index", nxr::quantity::as_index(q)}}; return static_cast<int>(nxr::quantity::as_index(q)); }
    need(depth >= 0, who + ": the grid has no levels");
    if (q.chart == "parts") { const auto r = nxr::select::level_of_parts(q.hi, depth); snapped = Json{{"parts_per_hemisphere", r.snapped}}; return r.level; }
    const double A = piece_area_of(grid);
    if (q.chart == "area") {
      std::vector<double> areas;
      for (int l = 0; l <= depth; ++l) areas.push_back(grid.members[static_cast<std::size_t>(l)].tile_area ? *grid.members[static_cast<std::size_t>(l)].tile_area : (A > 0 ? A / std::ldexp(1.0, l) : std::nan("")));
      need(!areas.empty() && std::isfinite(areas.front()), who + ": the grid's level areas are not recorded");
      const double a = q.hi * nxr::quantity::unit_factor(q.unit, unit + "^2");
      const auto r = nxr::select::level_of_area(areas, a, unit + "^2");
      snapped = Json{{"area", r.snapped}, {"unit", unit + "^2"}};
      return r.level;
    }
    if (q.chart == "wavelength" || q.chart == "eigenvalue") {
      need(A > 0, who + ": a wavelength or an eigenvalue needs the piece area (the grid's level dictionary records neither)");
      const double v = q.hi * nxr::quantity::unit_factor(q.unit, q.chart == "wavelength" ? unit : "1/" + unit + "^2");
      const auto r = q.chart == "wavelength" ? nxr::select::level_of_wavelength(v, A, depth, unit) : nxr::select::level_of_eigenvalue(v, A, depth, unit);
      snapped = Json{{"wavelength", r.snapped}, {"unit", unit}};
      return r.level;
    }
    std::vector<double> diam;
    for (int l = 0; l <= depth; ++l) {
      need(grid.members[static_cast<std::size_t>(l)].diameter.has_value(), who + ": the grid's level " + std::to_string(l) + " records no measured tile diameter");
      diam.push_back(*grid.members[static_cast<std::size_t>(l)].diameter);
    }
    const double d = q.hi * nxr::quantity::unit_factor(q.unit, unit);          // the coarsest level whose MEASURED mean tile diameter is at most it
    const auto r = nxr::select::level_of_diameter(diam, d, unit);
    snapped = Json{{"diameter", r.snapped}, {"unit", r.unit.empty() ? unit : r.unit}};
    return r.level;
  } catch (const std::invalid_argument& e) { refuse(who + ": " + finest.get<std::string>() + ": " + e.what()); }
}

Json ladder_dictionary(const nxr::selection::Selection& grid) {
  Json d = Json::array();
  for (std::size_t l = 0; l < grid.members.size(); ++l) {
    Json e = Json{{"level", static_cast<int>(l)}};
    if (grid.members[l].diameter) e["diameter"] = *grid.members[l].diameter;
    if (grid.members[l].tile_area) e["tile_area"] = *grid.members[l].tile_area;
    if (grid.members[l].band) { e["lambda_min"] = grid.members[l].band->first; e["lambda_max"] = grid.members[l].band->second; }
    d.push_back(e);
  }
  return d;
}
nxr::selection::Selection grid_of_dictionary(const Json& dictionary) {
  nxr::selection::Selection grid;
  grid.type = nxr::selection::Type::set;
  if (!dictionary.is_array()) return grid;
  for (const auto& e : dictionary) {
    nxr::selection::Selection level;
    if (e.contains("diameter")) level.diameter = e["diameter"].get<double>();
    if (e.contains("tile_area")) level.tile_area = e["tile_area"].get<double>();
    if (e.contains("lambda_min")) level.band = std::make_pair(e["lambda_min"].get<double>(), e["lambda_max"].get<double>());
    grid.members.push_back(std::move(level));
  }
  return grid;
}
namespace {
nxr::selection::Selection grid_of_ladder_id(const Bound& bound, const std::string& ladder_id) {
  nxr::selection::Selection grid;                                           // the grid's levels with their measured diameters and bands, from the ladder's level records (no codes needed)
  grid.type = nxr::selection::Type::set;
  for (const auto& m : bound.level_members(ladder_id)) {
    const Json lj = Json::parse(nxr::io::opt_text(m, "params_json").value_or("{}"), nullptr, false);
    nxr::selection::Selection level;
    if (lj.is_object() && lj.contains("diameter")) level.diameter = lj["diameter"].get<double>();
    if (lj.is_object() && lj.contains("lambda_min")) level.band = std::make_pair(lj["lambda_min"].get<double>(), lj["lambda_max"].get<double>());
    grid.members.push_back(std::move(level));
  }
  return grid;
}
}  // namespace

nxr::selection::Selection ladder_grid(const Bound& bound, const Value& tiles, std::string* unit) {
  need(tiles.row.has_value(), "the tile ladder is not stored (`select_dyadic_surface` first)");
  if (unit) *unit = nxr::io::opt_text(*tiles.row, "unit").value_or(std::string()).empty() ? std::string("m") : nxr::io::text(*tiles.row, "unit");
  return grid_of_ladder_id(bound, nxr::io::text(*tiles.row, "id"));
}
int level_of_surface(const Bound& bound, const SurfaceInput& surface, const Json& quantity, const std::string& who, Json* snapped) {
  const SurfaceLevel L0 = bound.surface_level(surface, 0);                  // the surface's tile ladder (refused by name when there is none)
  const std::string unit = L0.ladder.params.is_object() ? L0.ladder.params.value("length_unit", std::string("m")) : std::string("m");
  const nxr::selection::Selection grid = grid_of_ladder_id(bound, L0.ladder.id);
  Json sn = Json::object();
  const int l = level_of_quantity(grid, quantity, unit, who, sn);
  need(l >= 0 && l < static_cast<int>(grid.members.size()), who + ": " + quantity.dump() + " is level " + std::to_string(l) + "; the tile ladder of " + surface.name + " has levels 0 .. " + std::to_string(static_cast<int>(grid.members.size()) - 1));
  if (snapped) *snapped = sn;
  return l;
}
int level_of_ladder(const Bound& bound, const Value& tiles, const Json& quantity, const std::string& who, Json* snapped) {
  std::string unit;
  const nxr::selection::Selection grid = ladder_grid(bound, tiles, &unit);
  Json sn = Json::object();
  const int l = level_of_quantity(grid, quantity, unit, who, sn);
  need(l >= 0 && l < static_cast<int>(grid.members.size()), who + ": " + quantity.dump() + " is level " + std::to_string(l) + "; the tile ladder has levels 0 .. " + std::to_string(static_cast<int>(grid.members.size()) - 1));
  if (snapped) *snapped = sn;
  return l;
}
int level_among(const Json& dictionary, const std::string& unit, const Json& quantity, const std::vector<int>& held, const std::string& who, Json& snapped) {
  need(dictionary.is_array() && !dictionary.empty(), who + ": the field records no level dictionary of its ladder (it is not a field of `apply_kernel` with a spatial scale)");
  const int l = level_of_quantity(grid_of_dictionary(dictionary), quantity, unit, who, snapped);
  if (std::find(held.begin(), held.end(), l) == held.end()) {
    std::string have;
    for (const int h : held) have += (have.empty() ? "" : ", ") + std::to_string(h);
    refuse(who + ": " + quantity.dump() + " is level " + std::to_string(l) + " of the ladder; the field holds levels " + have + " (the scales it was made with)");
  }
  return l;
}


// ── select.run_span and select.elements: the factors of a product that are not grids ─────────────────────────────────────────────────
// A product's factors are always Selections: the run's span, a channel set and a grid level are made by makers and named as factors; none is virtual. These two makers read
// no data. `select.run_span` is ONE span covering the whole line (the run's span as one member); `select.elements` is the elements of a space (the channels of a recording, the vertices of a
// surface) as `members` (each element its own member) or `one_member` (all elements in one member). Both describe the grid kind `reduce` folds (params.grid: span, elements, whole).
StepSpec run_span_step() {
  StepSpec s;
  s.name = "select.run_span";
  s.kind = "setup";
  s.inputs = {"line"};
  s.outputs = {"span"};
  s.parameters = Json::object();
  s.matches = "the Studio's span selection (one span on a line)";
  s.description = "The span of the whole line as ONE member (a spans selection of one span on the line): the factor of a product that stands for 'the run'. It reads no data.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "no time axis: runs once on numbers (the line's rate and length)");
  s.plan = [](const Values& ins, const Json&, const PlanContext&) {
    need(ins.count("line") != 0, "select.run_span: the input `line` (the time line, or a field on it) is missing");
    const Value& lv = ins.at("line");
    need(lv.time && lv.rate > 0, "select.run_span: `line` has no time line with a rate");
    Value v;
    v.cls = "selection";
    v.manifolds = {*lv.time};
    v.facts = Json{{"node", "select.run_span"}, {"port", "span"}, {"family", "grid"}, {"names", Json::object()}};
    return Values{{"span", v}};
  };
  s.prepare = [](const Values& ins, const Json&, ExecContext& ctx) {
    const Value& lv = ins.at("line");
    const auto line = ctx.bound.manifold(*lv.time);
    need(line.has_value(), "select.run_span: the time line is not among the subject's rows");
    return std::any(std::make_tuple(nxr::io::integer(*line, "n_vertices"), lv.rate, lv.origin));
  };
  s.run = [](const std::any& st, const Arrays&, ExecContext&) {
    const auto& [n, rate, origin] = std::any_cast<const std::tuple<std::int64_t, double, double>&>(st);
    Json span = Json{{"start", origin}, {"stop", origin + static_cast<double>(n) / rate}, {"code", 0}};
    Json desc = Json{{"spans", Json::array({span})}, {"n_members", 1}, {"family", "grid"}, {"params", Json{{"grid", Json{{"kind", "span"}, {"size", n}, {"rate", rate}}}}},
                     {"description", "the run's span as one member (" + std::to_string(n) + " samples)"}};
    Array out;
    out.meta = Json{{"describe", desc}};
    return Arrays{{"span", std::move(out)}};
  };
  return s;
}

StepSpec elements_step() {
  StepSpec s;
  s.name = "select.elements";
  s.kind = "setup";
  s.inputs = {"space", "channels_of"};
  s.optional = {"channels_of"};
  s.reads_arrays = true;                                                   // `channels_of`: a stored statistic's channel pick, read as its channels
  s.outputs = {"elements"};
  s.parameters = Json{{"as", "members"}};
  s.matches = "the Studio's set selection over a manifold's elements (a channel set, a cortex)";
  s.description = "The elements of a space (the channels of a recording, the vertices of a surface) as a set: `as: members` (each element its own member, the layout of a measurement per channel) or `as: one_member` "
                  "(all elements in one member: a mean over the channels). It reads no data; `space` is the manifold, or a field on it (its first factor that is not a time line). `channels_of` (optional): "
                  "a stored statistic whose channel axis is a pick (a cross-spectrum): only the picked channels are members (the others belong to none, member -1).";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "no time axis: labels only");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("space") != 0, "select.elements: the input `space` (a manifold, or a field on it) is missing");
    need(p["as"] == "members" || p["as"] == "one_member", "select.elements: as " + p["as"].dump() + ": members or one_member");
    const Value& sv = ins.at("space");
    std::string id;
    for (const auto& m : sv.manifolds) if (!sv.time || m != *sv.time) { id = m; break; }
    need(!id.empty(), "select.elements: `space` has no factor other than a time line");
    if (ins.count("channels_of")) need(ins.at("channels_of").cls == "statistic", "select.elements: channels_of is a stored statistic (its channel axis is a pick of the space's channels)");
    Value v;
    v.cls = "selection";
    v.manifolds = {id};
    v.facts = Json{{"node", "select.elements"}, {"port", "elements"}, {"family", "grid"}, {"names", Json::object()}};
    return Values{{"elements", v}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    const Value& sv = ins.at("space");
    std::string id;
    for (const auto& m : sv.manifolds) if (!sv.time || m != *sv.time) { id = m; break; }
    const auto m = ctx.bound.manifold(id);
    need(m.has_value(), "select.elements: the manifold is not among the subject's rows");
    return std::any(std::make_pair(nxr::io::integer(*m, "n_vertices"), p["as"].get<std::string>()));
  };
  s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const auto& [n, as] = std::any_cast<const std::pair<std::int64_t, std::string>&>(st);
    Array out;
    out.rows = n; out.cols = 1;
    if (ins.count("channels_of") && ins.at("channels_of").meta.contains("channels")) {   // a PICK of the elements (a cross-spectrum's channels): the others belong to no member
      const std::vector<std::int32_t> pick = ins.at("channels_of").meta["channels"].get<std::vector<std::int32_t>>();
      out.data.assign(static_cast<std::size_t>(n), -1.0);
      std::int64_t k = 0;
      for (const std::int32_t c : pick) { need(c >= 0 && c < n, "select.elements: the picked channel " + std::to_string(c) + " is not among the space's " + std::to_string(n) + " elements"); out.data[static_cast<std::size_t>(c)] = as == "members" ? static_cast<double>(k) : 0.0; ++k; }
      const std::int64_t members = as == "members" ? k : 1;
      Json desc = Json{{"n_members", members}, {"family", "grid"}, {"params", Json{{"grid", Json{{"kind", "set"}, {"size", n}}}}},
                       {"description", (as == "members" ? "each of the " + std::to_string(k) + " picked elements its own member" : "the " + std::to_string(k) + " picked elements as one member") + " (of " + std::to_string(n) + ", the pick of a stored statistic)"}};
      out.meta = Json{{"describe", desc}, {"n_members", members}};
      return Arrays{{"elements", std::move(out)}};
    }
    for (std::int64_t i = 0; i < n; ++i) out.data.push_back(as == "members" ? static_cast<double>(i) : 0.0);
    const std::int64_t members = as == "members" ? n : 1;
    Json desc = Json{{"n_members", members}, {"family", "grid"}, {"params", Json{{"grid", Json{{"kind", as == "members" ? "elements" : "whole"}, {"size", n}}}}},
                     {"description", as == "members" ? "each of the " + std::to_string(n) + " elements its own member" : "the " + std::to_string(n) + " elements as one member"}};
    out.meta = Json{{"describe", desc}};
    return Arrays{{"elements", std::move(out)}};
  };
  return s;
}

// ── select.kernels (native) ──────────────────────────────────────────────────────────────────────────────────────
// A KERNEL SET on a line (a frequency line, a mode axis): the selection system's kernel members (an anchor + a profile + a chart), stored as ONE selection whose members are kernels, so
// a measurement over them (alone, or over a product with another set: the cells of a joint filterbank) is found by its member. Every member is stated as a QUANTITY on the line's coordinate:
// a frequency in Hz on a frequency line; on a mode axis (its coordinate the wavenumber mu) a wavelength 2 pi / mu or the wavenumber itself. Centres listed, or a grid of `voices` per octave
// from `from` to `to` (the grid resolution); every profile parameter is the caller's (no default).
struct KernelsPrepared { std::vector<Json> members; std::string quantity; Json params; };
KernelsPrepared kernels_of(const Json& p) {
  const char* who = "select.kernels";
  for (const char* k : {"profile", "chart", "quantity"}) need(p.contains(k) && p[k].is_string(), std::string(who) + ": " + k + " has no default: name it");
  const std::string profile = p["profile"], chart = p["chart"], quantity = p["quantity"];
  const bool band = profile == "morlet_band" || profile == "meyer_band", centred = profile == "gaussian" || profile == "box";
  need(centred || band || profile == "cos2_partition" || profile == "mexican_hat",
       std::string(who) + ": profile is gaussian, box, cos2_partition, morlet_band, meyer_band or mexican_hat (nxr::signal::kernelGains)");
  // cos2_partition: an amplitude partition in the log2 chart whose members' edges are their neighbours' centres (no width; gain 1: the gains sum to one at every point)
  if (centred) need(p.contains("width") && p["width"].is_number(), std::string(who) + ": width has no default: name it");
  if (profile != "cos2_partition") need(p.contains("gain") && p["gain"].is_number(), std::string(who) + ": gain has no default: name it");
  if (profile == "meyer_band") need(p.contains("half_width") && p["half_width"].is_number(), std::string(who) + ": half_width (octaves) has no default: name it");
  if (centred) need(p.contains("width_rule") && p["width_rule"].is_string(), std::string(who) + ": width_rule has no default: name it (absolute | proportional)");
  const std::string width_rule = p.contains("width_rule") && p["width_rule"].is_string() ? p["width_rule"].get<std::string>() : "not read";
  need(!centred || width_rule == "absolute" || width_rule == "proportional", std::string(who) + ": width_rule is absolute (the width as given) or proportional (the width times the member's centre)");
  need(chart == "log2" || chart == "linear", std::string(who) + ": chart is log2 (octaves) or linear");
  need(quantity == "frequency" || quantity == "wavelength" || quantity == "wavenumber", std::string(who) + ": quantity is frequency (Hz), wavelength (2 pi / mu) or wavenumber (mu)");
  need(!band || quantity == "frequency", std::string(who) + ": a band profile (morlet_band, meyer_band) lies on a frequency line");
  std::vector<double> centres;
  std::vector<std::pair<double, double>> bands;
  if (band) {
    need(p.contains("bands") && p["bands"].is_array() && !p["bands"].empty(), std::string(who) + ": a band profile names its bands: [[lo, hi], ...] in Hz");
    need(!(p.contains("centres") && p["centres"].is_array() && !p["centres"].empty()) && !p.contains("grid_from"), std::string(who) + ": a band profile takes bands, not centres or a grid");
    for (const auto& b : p["bands"]) {
      need(b.is_array() && b.size() == 2 && b[0].is_number() && b[1].is_number() && b[0].get<double>() >= 0 && b[1].get<double>() > b[0].get<double>(),
           std::string(who) + ": a band is [lo, hi] with 0 <= lo < hi");
      bands.emplace_back(b[0].get<double>(), b[1].get<double>());
      centres.push_back(0.5 * (bands.back().first + bands.back().second));
    }
  } else if (p.contains("centres") && p["centres"].is_array() && !p["centres"].empty()) {
    for (const auto& c : p["centres"]) { need(c.is_number() && c.get<double>() > 0, std::string(who) + ": a centre is a positive number"); centres.push_back(c.get<double>()); }
  } else {
    need(p.contains("grid_from") && p.contains("grid_to") && p.contains("voices") && p["grid_from"].is_number() && p["grid_to"].is_number() && p["voices"].is_number_integer(),
         std::string(who) + ": name the centres, or a grid: grid_from, grid_to and voices (per octave)");
    const double lo = p["grid_from"], hi = p["grid_to"]; const int v = p["voices"];
    need(lo > 0 && hi > lo && v >= 1, std::string(who) + ": a grid is 0 < grid_from < grid_to, voices >= 1");
    for (int k = 0;; ++k) { const double c = lo * std::pow(2.0, static_cast<double>(k) / v); if (c > hi * (1 + 1e-12)) break; centres.push_back(c); }
  }
  std::vector<std::string> names;
  if (p.contains("names") && p["names"].is_array()) for (const auto& n : p["names"]) names.push_back(n.get<std::string>());
  need(names.empty() || names.size() == centres.size(), std::string(who) + ": one name a member");
  KernelsPrepared k;
  k.quantity = quantity;
  const std::string unit = quantity == "frequency" ? "Hz" : quantity == "wavelength" ? "length unit of the surface" : "1 / length unit";
  for (std::size_t i = 0; i < centres.size(); ++i) {
    const double at = quantity == "wavelength" ? 2.0 * M_PI / centres[i] : centres[i];          // the anchor on the line's own coordinate (a mode axis's: mu)
    char buf[64]; std::snprintf(buf, sizeof buf, "%.4g %s", centres[i], quantity == "frequency" ? "Hz" : quantity == "wavelength" ? "wavelength" : "mu");
    // the profile's numbers (nxr::selection::Profile::params): the width as fwhm, or as fwhm_per_centre for a proportional (constant-Q) width; a Meyer band's half-width
    Json params = Json::object();
    if (centred) params[width_rule == "proportional" ? "fwhm_per_centre" : "fwhm"] = p["width"];
    if (profile == "meyer_band") params["half_width"] = p["half_width"];
    const double start = band ? bands[i].first : at, stop = band ? bands[i].second : at;   // the anchor: the band [lo, hi], or the centre (a span with start = stop)
    k.members.push_back(Json{{"start", start}, {"stop", stop}, {"code", static_cast<std::int64_t>(i)}, {"family", profile}, {"chart", chart},
                             {"gain", profile == "cos2_partition" ? Json(1.0) : p["gain"]}, {"params", params}, {"name", names.empty() ? std::string(buf) : names[i]},
                             {"centre", centres[i]}, {"unit", unit}});
  }
  k.params = Json{{"kernels", Json{{"quantity", quantity}, {"unit", unit}, {"profile", profile}, {"chart", chart}, {"width", p.value("width", Json())}, {"width_rule", width_rule}, {"gain", p.value("gain", Json())}, {"half_width", p.value("half_width", Json())}, {"members", k.members}}},
                  // a factor of a product (the cells of a joint filterbank): its member count; a regular ladder (voices per octave) is a grid, listed centres an irregular set
                  {"grid", Json{{"kind", "kernels"}, {"size", static_cast<std::int64_t>(k.members.size())}, {"quantity", quantity},
                                {"family", p.contains("voices") && p["voices"].is_number_integer() && !(p.contains("centres") && p["centres"].is_array() && !p["centres"].empty()) ? "grid" : "irregular"}}}};
  return k;
}
ng::NodeSpec select_kernels() {
  ng::NodeSpec n;
  n.name = "select.kernels";
  n.kind = ng::Kind::array;
  n.inputs = {{"line", "the line the kernels lie on (a stored frequency line, or a mode axis): its record", true}};
  n.outputs = {{"kernels", "the kernel set: one member a kernel (anchor on the line's coordinate, profile, chart, gain), named", false}};
  n.parameters = {{"quantity", Json(), "what the centres are: frequency (Hz) | wavelength (2 pi / mu, on a mode axis) | wavenumber (mu) (no default)", "-", {"frequency", "wavelength", "wavenumber"}, false},
                  {"profile", Json(), "gaussian | box | cos2_partition (an amplitude partition in log2 whose gains sum to one; no width) | morlet_band | meyer_band (bands [lo, hi] Hz: the time bank's band shapes) | mexican_hat (the graph bank's hat at each centre) (no default; nxr::signal::kernelGains)", "-", {"gaussian", "box", "cos2_partition", "morlet_band", "meyer_band", "mexican_hat"}, false},
                  {"chart", Json(), "the chart the width is in: log2 (octaves) | linear (the quantity's unit) (no default)", "-", {"log2", "linear"}, false},
                  {"width", Json(), "each member's FWHM (gaussian) or full width (box), in the chart (no default; not read by cos2_partition)", "-", {}, true},
                  {"width_rule", Json(), "absolute (the width as given) | proportional (the width times the member's centre: constant Q on a linear chart); gaussian and box (no default)", "-", {"absolute", "proportional"}, true},
                  {"gain", Json(), "each member's peak amplitude g (no default; cos2_partition: 1)", "-", {}, true},
                  {"half_width", Json(), "a Meyer band's transition half-width, octaves (meyer_band; no default)", "-", {}, true},
                  {"bands", Json(), "the bands [[lo, hi], ...] in Hz of a band profile (morlet_band, meyer_band), instead of centres", "Hz", {}, true},
                  {"centres", Json(), "the members' centres, in the quantity's unit; or a grid (grid_from, grid_to, voices)", "-", {}, true},
                  {"grid_from", Json(), "a grid's first centre", "-", {}, true}, {"grid_to", Json(), "a grid's last centre (at most)", "-", {}, true},
                  {"voices", Json(), "a grid's members per octave (the grid resolution)", "-", {}, true},
                  {"names", Json(), "one name a member (none: the quantity as written)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a selection made from parameters: no data flows");
  n.matches = "the selection system's kernel members (anchor + profile + chart; compute nxr::signal::kernelGains); nxr's frequency and spatial filter banks (Gaussian on log2)";
  n.description = "a kernel set on a line (a frequency line, a mode axis): one selection whose members are kernels, every centre a quantity, every profile parameter the caller's";
  n.plan = [](const ng::Types&, const Json& p) {
    const KernelsPrepared k = kernels_of(p);
    ng::Planned out;
    ng::PortType t = constant_port("selection");
    Json names = Json::array();
    for (const auto& m : k.members) names.push_back(m["name"]);
    t.facts["set"] = Json{{"type", "kernels"}, {"members", names}, {"family", "irregular"}};
    out.out["kernels"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays&, const Json& p, ng::Shared&) { return std::any(kernels_of(p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const KernelsPrepared& k = std::any_cast<const KernelsPrepared&>(prepared);
    ng::Array a;
    a.rows = static_cast<std::int64_t>(k.members.size()); a.cols = 1;
    Json kernels = Json::array(), entries = Json::array();
    for (const auto& m : k.members) { kernels.push_back(m); entries.push_back(m["name"]); a.data.push_back(m["code"].get<double>()); }
    a.meta["describe"] = Json{{"kernels", kernels}, {"entries", entries}, {"family", "irregular"}, {"params", k.params}, {"n_members", static_cast<std::int64_t>(k.members.size())},
                              {"description", "a kernel set of " + std::to_string(k.members.size()) + " members (" + k.quantity + ")"}};
    return ng::Arrays{{"kernels", std::move(a)}};
  };
  return n;
}

void register_select_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {select_bands(), select_product(), select_kernels()})
    if (!r.find(n.name)) r.add(std::move(n));
}

void add_select_nodes(std::vector<StepSpec>& r) {
  // select.dyadic_surface: the tile grid and the group tile grid as the rules subject / group / transfer
  {
    StepSpec tile = take_step(r, "tile_grid");
    StepSpec transfer = take_step(r, "group_tile_grid");
    std::vector<std::pair<std::string, StepSpec>> parts;
    parts.emplace_back("subject", tile);
    parts.emplace_back("group", tile);
    parts.emplace_back("transfer", transfer);
    r.push_back(combine_by_param(
        "select.dyadic_surface", "rule", std::move(parts),
        "The dyadic grid of a surface (equal-area bisection; level l has 2^l tiles a connected piece; the hierarchy is the closed form code >> shift). Rule `subject`: this subject's own tiles; `group`: the default "
        "subject's grid as the group tree, with the dataset's atlas definitions; `transfer`: a subject's vertices on the group tree through the registered spheres, keeping the template's codes. `finest` is the "
        "finest tile as a QUANTITY (a tile diameter, an area, a wavelength or parts per hemisphere); the depth is resolved from the grid's measured level diameters and the snapped quantity recorded. The Levi-Civita "
        "connection between neighbouring tiles is a measurement over the product of a level with itself, not part of the grid.",
        [](const Json& p, const std::string& which) {
          Json q = p;
          if (which == "subject" || which == "group") q["tree"] = which;
          return q;
        }));
  }
  r.push_back(time_grid_step());
  r.push_back(run_span_step());
  r.push_back(elements_step());
  // select.events.*: detection makes irregular selections that record the grid they were found over
  {
    std::vector<std::pair<std::string, StepSpec>> parts;
    parts.emplace_back("detect", take_step(r, "detect_spans"));
    parts.emplace_back("keep_where", take_step(r, "fit_gate"));
    parts.emplace_back("stages", take_step(r, "burst_stages"));
    parts.emplace_back("edges", take_step(r, "event_edges"));
    parts.emplace_back("union", take_step(r, "event_union"));
    r.push_back(combine_by_param("select.events.spans", "rule", std::move(parts),
        "spans in time found by detection, an IRREGULAR selection (sparse, may overlap, no levels) that records the grid it was found over: rule `detect` (a measurement's members labelled by hysteresis "
        "crossings: was detect_spans), `keep_where` (the members whose rise-fall fit passes: was fit_gate; or, with `x` and `where`, whose measures lie in their ranges), `stages` (onset, plateau, offset of "
        "each member: was burst_stages), `edges` (each event's onset and offset: the tangent on its envelope, or the atom's half maximum), `union` (the global windows: the events of every row joined where "
        "they overlap in time and frequency). Filters on members are quantities (a duration range, a gap to merge)."));
  }
  {
    StepSpec s = take_step(r, "cell_extent");
    s.name = "select.events.regions";
    r.push_back(std::move(s));
  }
  {
    StepSpec s = take_step(r, "patches");
    s.name = "select.events.patches";                                       // accumulate (a pass over the frames); `select.events.regions` is the after-kind of the same family
    r.push_back(std::move(s));
  }
  {
    StepSpec c = take_step(r, "cores");
    c.name = "select.events.cores";
    r.push_back(std::move(c));
    StepSpec t = take_step(r, "tracks");
    t.name = "select.events.tracks";
    r.push_back(std::move(t));
  }
}

}  // namespace nxr::pipelines::graph

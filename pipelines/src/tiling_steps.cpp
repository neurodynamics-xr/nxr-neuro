// nxr-pipelines — the registered NODE of the frame tiling (`frame_tiling`). A pure function of numbers (nxr/tower.h: the frames are the cycles of ONE tower level, D135) with its ports,
// parameters and traits; it reads no file, subject or database. What it needs of the recording (the time line's rate, origin and length) is the description the runner binds. It runs once,
// before the loop (kind `setup`), and DESCRIBES the tiling it made: the definition places it (a `selection` output of type `tiling` on the recording's time line), io writes it.
//
//   frame_tiling   recording → tiling   the frames of the recording's time line: frame k is the tower tile `frame_code0 + k`, its time the tile's START
//
// What it cannot resolve: a frame is a cycle of one tower level, so its length is 86 400 · 2^L s: a band's 36 frames to a cycle is met by the coarsest level at or under it, never exactly.
#include <cmath>
#include <cstdio>
#include <sstream>

#include "nxr/pipelines/registry.h"
#include "nxr/tower.h"

namespace nxr::pipelines::graph {

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
std::string g(double v) { std::ostringstream s; s << v; return s.str(); }

struct TilingState {
  nxr::tower::FrameTiling t;
  int level = 0;
  double centre = 0, origin = 0, width = 0;
  bool from_band = false;
  Json band, start_s;
  std::string recording_name;
  std::int64_t samples = 0, frames_per_cycle = 36;
};

// the tiling a request makes on a recording's time line (shared by the plan, which names the output, and the run)
TilingState make(const Values& ins, const Json& p, const Bound& bound) {
  need(ins.count("recording") != 0, "frame_tiling: the input `recording` is missing");
  const Value& rv = ins.at("recording");
  TilingState s;
  const bool has_level = p["frame_level"].is_number_integer(), has_centre = p["centre_hz"].is_number(), has_band = p["band"].is_array();
  s.from_band = has_centre || has_band;
  need(!(has_level && s.from_band), "frames: --frame-level or the band (--centre / --band), not both");
  need(has_level || s.from_band, "frames: name the tower level of a frame (--frame-level, e.g. -25) or derive it from the band (--centre HZ or --band LO,HI, with --frames-per-cycle)");
  need(!(has_centre && has_band), "frames: --centre or --band, not both");
  s.frames_per_cycle = p["frames_per_cycle"].get<std::int64_t>();
  if (s.from_band) {
    if (has_band) {
      need(p["band"].size() == 2 && p["band"][0].is_number() && p["band"][1].is_number(), "frames: --band LO,HI is two numbers (Hz)");
      const double lo = p["band"][0].get<double>(), hi = p["band"][1].get<double>();
      need(hi > lo && lo >= 0, "frames: --band LO,HI needs 0 <= LO < HI (Hz)");
      s.centre = (lo + hi) / 2;
      s.band = p["band"];
    } else s.centre = p["centre_hz"].get<double>();
    need(s.centre > 0, "frames: the centre is above 0 Hz");
    need(s.frames_per_cycle >= 1, "frames: --frames-per-cycle is at least 1");
    s.level = nxr::tower::frameLevelFor(s.centre, static_cast<int>(s.frames_per_cycle));
  } else s.level = p["frame_level"].get<int>();
  need(s.level <= 0, "frames: --frame-level is 0 (the day) or finer (negative)");
  need(rv.time && rv.rate > 0, "frame_tiling: the recording has no time line with a rate");
  const auto line = bound.manifold(*rv.time);
  need(line.has_value(), "frame_tiling: the recording's time line is not among the subject's rows");
  s.samples = nxr::io::integer(*line, "n_vertices");
  s.start_s = p["start_s"];
  try {
    s.t = nxr::tower::makeFrameTiling(rv.rate, s.samples, s.level, p["start_s"].is_number() ? std::optional<double>(p["start_s"].get<double>()) : std::nullopt);
  } catch (const std::invalid_argument& e) { refuse(e.what()); }
  s.width = s.t.frame_s();
  s.origin = rv.origin + s.t.origin_s();
  if (rv.row) s.recording_name = nxr::io::text(*rv.row, "name");
  return s;
}

}  // namespace

void add_tiling_steps(std::vector<StepSpec>& r) {
  StepSpec s;
  s.name = "frame_tiling";
  s.kind = "setup";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64,
                         "no time axis: runs once on numbers (the line's rate and length); the tiling is exact in binary (86 400 * 2^L s)");
  s.inputs = {"recording"};
  s.outputs = {"tiling"};
  s.parameters = Json{{"frame_level", nullptr}, {"centre_hz", nullptr}, {"band", nullptr}, {"frames_per_cycle", 36}, {"start_s", nullptr}};
  s.matches = "nxr pipelines/atlas/build.py and tower.py (D135 TOWER): the frames are the cycles of one tower level";
  s.description =
      "The frame tiling of a recording's time line on the tower: the frames are the cycles of ONE tower level (86 400 * 2^L s; frame k is the tile frame_code0 + k, a level-m tile is code >> m), so every frame has a "
      "dyadic parent. Give the level (frame_level: 0 is the day, -18 is 0.33 s, -25 is 2.57 ms; a whole number <= 0) or derive it from a band (centre_hz, or band [lo, hi] whose middle is the centre) with "
      "frames_per_cycle (36: the 10 degree step): the coarsest level whose frame is at or under 1/(centre x frames_per_cycle) s. start_s anchors the recording's first sample to local midnight (seconds; default: "
      "the anchor is the recording). A frame's time is its tile's START.";
  s.plan = [](const Values& ins, const Json& p, const PlanContext& ctx) {
    const TilingState t = make(ins, p, ctx.bound);
    Value v;
    v.cls = "selection";
    v.manifolds = {*ins.at("recording").time};
    v.facts = Json{{"node", "frame_tiling"}, {"port", "tiling"}, {"names", Json::object()}};
    char b[64];
    std::snprintf(b, sizeof b, "%g", t.width);
    v.facts["names"]["frame_s"] = std::string(b);
    return Values{{"tiling", v}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) { return std::any(make(ins, p, ctx.bound)); };
  s.run = [](const std::any& st, const Arrays&, ExecContext&) {
    const auto& t = std::any_cast<const TilingState&>(st);
    const std::int64_t count = t.t.count;
    Json ft = Json{{"sfreq", t.t.sfreq}, {"samples", t.samples}, {"fps", t.t.fps()}, {"recording", t.recording_name}, {"frame_samples", t.t.tile_samples()},
                   {"count", "the tower tiles touched by at least one sample (the first and the last may be partial)"},
                   {"frame", "frame k is the tower tile frame_code0 + k; its time is the tile's START: t = origin + k * frame_s"}, {"reference", "nxr pipelines/atlas/tower.py (D135 TOWER)"}};
    if (t.from_band) { ft["centre_hz"] = t.centre; ft["frames_per_cycle"] = t.frames_per_cycle; }
    if (!t.band.is_null()) ft["band"] = t.band;
    Json params = Json{{"atlas", Json{{"frame_s", t.width}, {"frame_level", t.level}, {"frame_code0", t.t.code0}, {"anchor", t.t.day_anchored ? "day" : "recording"}, {"start_s", t.start_s}, {"made", true}}},
                       {"frame_tiling", ft}};
    Json desc = Json{{"params", params},
                     {"tiling", Json{{"origin", t.origin}, {"width", t.width}, {"hop", t.width}, {"end", t.origin + static_cast<double>(count) * t.width}, {"count", count}}},
                     {"description", "the measurement pyramid's time tiles: " + std::to_string(count) + " cycles of tower level " + std::to_string(t.level) + " (" + g(t.width) + " s), frame k is code " +
                                         std::to_string(t.t.code0) + " + k, a level-m tile code >> m"}};
    Array out;
    out.rows = 0; out.cols = 0;
    out.meta = Json{{"describe", desc}};
    Arrays o;
    o["tiling"] = std::move(out);
    return o;
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph

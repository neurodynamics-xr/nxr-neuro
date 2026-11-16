// nxr-pipelines — the registered step of the TRACKS of vortex cores (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 6; family C: a feature followed in time):
//
//   tracks   after   links the run's cores frame to frame (nxr `_track_stats`: greedy nearest neighbour, same sense, ≤ link_mm, consecutive frames only; compute nxr::tracks::Linker) and stores
//                    the TRACKS as point events (point_events.h, kind `tracks`): a spans Selection, one member per track, its span the track's first core's frame to its last core's frame, and the track's
//                    attributes as measurements over it; the labelling core → track is a measurement over the CORES' events (`track`: the track's member ordinal, -1 for a core in no stored track).
//
// It reads the cores a `cores` step of the same run accumulated (arrays, outside the chunk loop), never recording data: the linking is a pure function of the cores' frames, senses and centres, so the result does not
// depend on the chunk size (the cores do not). The centres are in the surface's unit (m); `link_mm` is given in mm, as nxr's CORE_LINK, and converted.
// The tracks are DESCRIPTIONS of where cores went. Their speed and lifetime equal the lag-free null's (the project's population work, 2026-10-30): they are never evidence that anything travels.
#include <cmath>
#include <memory>

#include "nxr/io/errors.h"
#include "nxr/pipelines/point_events.h"
#include "nxr/pipelines/registry.h"
#include "nxr/tracks.h"

namespace nxr::pipelines::graph {

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
struct TrackState { double link_m = 0.015; std::int64_t min_frames = 2; Json record; };
const char* const kWarnings[] = {
    "a track links cores of consecutive frames (the nearest of the same sense within link_mm): it describes where a core went, not that anything travels (its speed and lifetime equal the lag-free null's: the null is the evidence)",
    "a core with no partner in the next frame ends its track; two cores of one frame never share a track; a frame without a core, or a bad segment, closes every track"};
const std::vector<std::string> kAttributes = {"first_frame", "last_frame", "n_frames", "path_length", "displacement", "sense"};
}  // namespace

void add_tracks_steps(std::vector<StepSpec>& r) {
  StepSpec s;
  s.name = "tracks"; s.kind = "after";
  s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "links each frame's cores to the open tracks of the previous frame: the open tracks are carried forward in time, and a frame with no core closes them all", {});
  s.inputs = {"events", "frame", "sense", "centre"};
  s.outputs = {"spans", "first_frame", "last_frame", "n_frames", "path_length", "displacement", "sense", "track"};
  s.parameters = Json{{"link_mm", 15.0}, {"min_frames", 2}};
  s.matches = "nxr vortex._track_stats (the linking: greedy nearest neighbour, same sense, ≤ CORE_LINK = 15 mm, consecutive frames only); compute nxr::tracks (Linker, trackStats)";
  s.description = "the TRACKS of the run's cores (`cores`): each core of a frame joins, in the order the cores are given (sense +1 first), the nearest still-free track of the PREVIOUS frame of the same sense within `link_mm` "
                  "(15, nxr's) of its last centre, else starts a track; only consecutive frames link (a frame with no core, or a bad segment, closes every track). Stored as POINT EVENTS (kind `tracks`): a spans Selection, "
                  "one member per track of at least `min_frames` frames (2 by default: a lone core is no track; nxr's lifetime statistics use 3 or more, a filter at read), its span the track's first core's frame to its last core's, "
                  "in the order the tracks start; per track as measurements over it: first_frame, last_frame, n_frames (cores held), path_length (the sum of the steps' lengths, m), displacement (|last centre − first centre|, m) and "
                  "sense. The labelling `track`, a measurement over the CORES' events: the member ordinal of the core's track (-1: a core in no stored track). Linking reads the cores' list, not the chunks: the result does not "
                  "depend on the chunk size. DESCRIPTIONS, not evidence: a track's speed and lifetime equal the lag-free null's";
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& ev = ins.at("events");
    need(ev.cls == "spans" && ev.facts.value("events_kind", std::string()) == "cores", "tracks: events must be the spans of the `cores` step (kind cores)");
    for (const char* port : {"frame", "sense", "centre"}) {
      const Value& m = ins.at(port);
      need(m.cls == "measurement" && m.facts["over"].value("events_kind", std::string()) == "cores" && m.facts.value("measure", std::string()) == port,
           std::string("tracks: ") + port + " must be the cores' own `" + port + "` measurement (a `cores` output)");
    }
    need(p["link_mm"].is_number() && p["link_mm"].get<double>() > 0, "tracks: link_mm is above 0 (nxr: 15)");
    need(p["min_frames"].is_number_integer() && p["min_frames"].get<std::int64_t>() >= 1, "tracks: min_frames is a whole number, at least 1 (2: a track has a step)");
    Json record = Json{{"link_mm", p["link_mm"]}, {"min_frames", p["min_frames"]}, {"reference", "nxr vortex._track_stats"}, {"of", ev.facts.value("cores", Json())}};
    Values out;
    out["spans"] = point_events_value_from(ev, Json{{"tracks", record}}, "tracks");
    out["first_frame"] = point_attribute_value_from(ev, "first_frame", std::nullopt, "tracks");
    out["last_frame"] = point_attribute_value_from(ev, "last_frame", std::nullopt, "tracks");
    out["n_frames"] = point_attribute_value_from(ev, "n_frames", std::nullopt, "tracks");
    out["path_length"] = point_attribute_value_from(ev, "path_length", "m", "tracks");
    out["displacement"] = point_attribute_value_from(ev, "displacement", "m", "tracks");
    out["sense"] = point_attribute_value_from(ev, "sense", std::nullopt, "tracks");
    out["track"] = point_attribute_value_from(ev, "track", std::nullopt, "cores");           // over the CORES' events
    return out;
  };
  s.prepare = [](const Values&, const Json& p, ExecContext&) {
    TrackState st;
    st.link_m = p["link_mm"].get<double>() * 1e-3;                                   // the cores' centres are in the surface's unit, m
    st.min_frames = p["min_frames"].get<std::int64_t>();
    st.record = Json{{"link_mm", p["link_mm"]}, {"min_frames", st.min_frames}, {"reference", "nxr vortex._track_stats"}, {"warnings", Json::array({kWarnings[0], kWarnings[1]})}};
    return std::any(std::make_shared<TrackState>(std::move(st)));
  };
  s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<TrackState>&>(st)->record; };
  s.run = [](const std::any& st_, const Arrays& ins, ExecContext&) {
    const TrackState& st = *std::any_cast<const std::shared_ptr<TrackState>&>(st_);
    const Array &ev = ins.at("events"), &fr = ins.at("frame"), &sn = ins.at("sense"), &ce = ins.at("centre");
    const std::int64_t E = fr.cols;
    if (sn.cols != E || ce.cols != E || ce.comps != 3 || static_cast<std::int64_t>(ce.data.size()) != 3 * E) throw std::invalid_argument("tracks: the cores' attributes are of different lengths");
    std::vector<std::int64_t> frame(static_cast<std::size_t>(E));
    std::vector<std::int8_t> sense(static_cast<std::size_t>(E));
    Eigen::MatrixXd centre(E, 3);
    for (std::int64_t i = 0; i < E; ++i) {
      frame[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(std::llround(fr.data[static_cast<std::size_t>(i)]));
      sense[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(sn.data[static_cast<std::size_t>(i)] > 0 ? 1 : -1);
      for (int d = 0; d < 3; ++d) centre(i, d) = ce.data[static_cast<std::size_t>(3 * i + d)];
    }
    nxr::tracks::Linker lk(st.link_m);
    lk.feed(frame, sense, centre);
    lk.finish();
    const nxr::tracks::TrackStats S = nxr::tracks::trackStats(frame, sense, centre, lk.track(), lk.previous());
    std::vector<std::int32_t> ordinal(static_cast<std::size_t>(S.size()), -1);
    PointEvents t;
    t.fps = ev.meta["tiles_of"]["fps"].get<double>(); t.origin = ev.meta["tiles_of"]["origin"].get<double>();
    std::vector<double> first, last, n, path, disp, sen;
    std::int32_t kept = 0;
    for (std::int64_t k = 0; k < S.size(); ++k) {
      const std::size_t z = static_cast<std::size_t>(k);
      if (S.n_frames[z] < st.min_frames) continue;
      ordinal[z] = kept++;
      t.frame.push_back(S.first_frame[z]); t.last_frame.push_back(S.last_frame[z]);
      first.push_back(static_cast<double>(S.first_frame[z])); last.push_back(static_cast<double>(S.last_frame[z])); n.push_back(static_cast<double>(S.n_frames[z]));
      path.push_back(S.path[z]); disp.push_back(S.displacement[z]); sen.push_back(S.sense[z]);
    }
    t.record = Json{{"link_mm", st.link_m * 1e3}, {"min_frames", st.min_frames}, {"reference", "nxr vortex._track_stats"}, {"cores", E}, {"tracks_found", S.size()}, {"tracks", static_cast<std::int64_t>(kept)},
                    {"cores_in_tracks", [&] { std::int64_t c = 0; for (std::int64_t k = 0; k < S.size(); ++k) if (ordinal[static_cast<std::size_t>(k)] >= 0) c += S.n_frames[static_cast<std::size_t>(k)]; return c; }()},
                    {"warnings", Json::array({kWarnings[0], kWarnings[1]})}};
    t.attributes = {{"first_frame", 1, first}, {"last_frame", 1, last}, {"n_frames", 1, n}, {"path_length", 1, path}, {"displacement", 1, disp}, {"sense", 1, sen}};
    Arrays all = point_event_arrays(t), out;
    out["spans"] = std::move(all.at("spans"));
    for (const auto& m : kAttributes) out[m] = std::move(all.at(m));
    // the labelling of the CORES: the member ordinal of each core's track, a measurement over the cores' events (the cores' own `frame` array carries its counts and tiles)
    Array lab = fr;
    lab.comps = 1;
    lab.meta["measure"] = "track";
    for (std::int64_t i = 0; i < E; ++i) lab.data[static_cast<std::size_t>(i)] = ordinal[static_cast<std::size_t>(lk.track()[static_cast<std::size_t>(i)])];
    out["track"] = std::move(lab);
    return out;
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph

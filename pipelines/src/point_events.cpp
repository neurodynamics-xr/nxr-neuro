// nxr-pipelines — point events (point_events.h).
#include "nxr/pipelines/point_events.h"

#include <stdexcept>

namespace nxr::pipelines::graph {

Arrays point_event_arrays(const PointEvents& e) {
  const std::int64_t E = static_cast<std::int64_t>(e.frame.size());
  if (!(e.fps > 0)) throw std::invalid_argument("point events: the frame tiling's rate is not set");
  const double w = 1.0 / e.fps;
  Json tiles_of = Json{{"width_s", w}, {"origin", e.origin}, {"unit", "frame"}, {"fps", e.fps}};
  Array spans{E, 4, {}};
  spans.data.reserve(static_cast<std::size_t>(E * 4));
  for (std::int64_t k = 0; k < E; ++k) {
    const double j = static_cast<double>(e.frame[static_cast<std::size_t>(k)]);
    const double stop = e.last_frame.empty() ? j + 1.0 : static_cast<double>(e.last_frame[static_cast<std::size_t>(k)]) + 1.0;
    spans.data.insert(spans.data.end(), {j, stop, -1.0, 0.0});                   // the event's frame (a track: its first to its last); no channel (a global event); the one event type
  }
  spans.meta = Json{{"tiles_of", tiles_of}, {"events", E}, {"point_events", true}, {"record", e.record}};
  if (!e.last_frame.empty() && static_cast<std::int64_t>(e.last_frame.size()) != E) throw std::invalid_argument("point events: last_frame is for another number of events");
  Json counts = Json::array();
  for (std::int64_t k = 0; k < E; ++k) counts.push_back(Json::array({k, e.last_frame.empty() ? 1.0 : static_cast<double>(e.last_frame[static_cast<std::size_t>(k)] - e.frame[static_cast<std::size_t>(k)] + 1)}));   // the samples a member holds: its frames
  Arrays out;
  out["spans"] = std::move(spans);
  for (const PointAttribute& a : e.attributes) {
    if (a.comps < 1 || static_cast<std::int64_t>(a.values.size()) != E * a.comps)
      throw std::invalid_argument("point events: attribute " + a.measure + " has " + std::to_string(a.values.size()) + " values for " + std::to_string(E) + " events of " + std::to_string(a.comps) + " component(s)");
    Array m{1, E, a.values};
    m.comps = a.comps;
    m.meta = Json{{"measure", a.measure}, {"events", true}, {"tiles", E}, {"counts", counts}, {"whole_tiles", E}, {"tiles_measured", nullptr}, {"selection", "the events of this run"},
                  {"width_s", w}, {"origin", e.origin}, {"tile_level", tiles_of}, {"point_events", true}};
    out[a.measure] = std::move(m);
  }
  return out;
}

Value point_events_value(const Value& x, const Json& facts, const std::string& kind) {
  Value y = x;
  y.row.reset();
  y.cls = "spans";
  y.manifolds = {*x.time};
  y.chunked = false;
  y.complex = false;
  y.unit.reset();
  y.facts = facts;
  y.facts["point_events"] = true;
  if (!kind.empty()) y.facts["events_kind"] = kind;
  return y;
}

Value point_events_value_from(const Value& events, const Json& facts, const std::string& kind) {
  Value y = events;
  y.row.reset();
  y.cls = "spans";
  y.chunked = false;
  y.facts = facts;
  y.facts["point_events"] = true;
  y.facts["events_kind"] = kind;
  return y;
}

Value point_attribute_value(const Value& x, const std::string& measure, const std::optional<std::string>& unit, const std::string& kind) {
  Value y = x;
  y.row.reset();
  y.cls = "measurement";
  y.manifolds = {*x.time};
  y.chunked = false;
  y.complex = false;
  y.unit = unit;
  y.facts = Json{{"measure", measure}, {"over", Json{{"events", true}, {"selection", "the events of this run"}}}, {"point_events", true}};
  if (!kind.empty()) y.facts["over"]["events_kind"] = kind;
  return y;
}

Value point_attribute_value_from(const Value& events, const std::string& measure, const std::optional<std::string>& unit, const std::string& kind) {
  Value y = events;
  y.row.reset();
  y.cls = "measurement";
  y.chunked = false;
  y.unit = unit;
  y.facts = Json{{"measure", measure}, {"over", Json{{"events", true}, {"events_kind", kind}, {"selection", "the " + kind + " of this run"}}}, {"point_events", true}};
  return y;
}

}  // namespace nxr::pipelines::graph

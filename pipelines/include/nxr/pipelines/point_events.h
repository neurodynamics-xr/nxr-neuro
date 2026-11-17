// nxr-pipelines — POINT EVENTS: events that sit at ONE FRAME of the frame tiling, each with a place and a few numbers (a vortex core: its sense, centre, area, peak; a spatial event of relative band
// power: its centre, area, size, peak). The shared storage of the vortex-cores port (piece 5) and the spatial-events work:
//
//   the events    a SPANS Selection on the recording's time line, ONE MEMBER PER EVENT, its span the event's frame [t_j, t_j + 1/fps) (the frame tiling's tile, `run frametiling`); several members may
//                 share a frame (the model keys a span by its ordinal), in the order the step gives them (by frame)
//   attributes    each a MEASUREMENT over that Selection (selection_measurement rows, array-backed: the measurement's member axis is the events' order): the step names the measures (`frame`, `sense`,
//                 `centre_vertex`, `centre` (x y z), `area`, `peak`, …)
//
// The step makes the arrays (`point_event_arrays`) and its plan makes the Values (`point_events_value`, `point_attribute_value`); the RUNNER stores them as it stores any spans output and any measurement over
// the run's events (store_spans, store_measurement): this file only fixes their shape.
//   kind          a run may make SEVERAL sets of point events (the cores, then the tracks made of them): each carries a `kind` (facts.events_kind, and the same in its attributes' `over`), and a measurement is over
//                 the set of its kind. No kind: the run's one set, as before (the spatial events). Pure: numbers in, arrays out; no row is read or written here.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

/** one attribute of the events: a measure's name and its values [events × comps] (comps 1, or 3 for a vector measure such as `centre`: stored as `centre_x`, `centre_y`, `centre_z`) */
struct PointAttribute {
  std::string measure;
  std::int64_t comps = 1;
  std::vector<double> values;
};
/** the events of one run: the frame tiling they sit on (its rate and first time on the recording's clock), each event's frame, its attributes, and what the step records (threshold, options) */
struct PointEvents {
  double fps = 0, origin = 0;
  std::vector<std::int64_t> frame;
  std::vector<std::int64_t> last_frame;      // empty: each event is its one frame [frame, frame + 1). Else the event spans frames frame … last_frame (a TRACK: first to last core; its count is the frames it spans)
  std::vector<PointAttribute> attributes;
  Json record = Json::object();
};
/** the arrays of a run's events: "spans" (a row per event: [frame, frame + 1, -1, 0] in frame units, the tile width 1/fps and the origin in `tiles_of`, as the runner stores a spans output) and one array
 *  per attribute under its measure's name ([1 × events × comps]; its count row is 1: one frame). `std::invalid_argument` for an attribute whose values are not events × comps. */
Arrays point_event_arrays(const PointEvents& e);
/** the plan's value of the events' spans output, from the field the events were found on (a field on the frame tiling: its time line, rate and origin are the tiling's) */
Value point_events_value(const Value& x, const Json& facts, const std::string& kind = "");
/** the same from the events of ANOTHER set of the run (an `after` step: the tracks made of the cores), which has no field: `events` is the Value of a point-event spans output */
Value point_events_value_from(const Value& events, const Json& facts, const std::string& kind);
/** the plan's value of an attribute's measurement over the run's own events: `measure`, its unit (none: unitless or the field's own) */
Value point_attribute_value(const Value& x, const std::string& measure, const std::optional<std::string>& unit = std::nullopt, const std::string& kind = "");
/** the same over the events of an `after` step's own set (see point_events_value_from) */
Value point_attribute_value_from(const Value& events, const std::string& measure, const std::optional<std::string>& unit, const std::string& kind);

}  // namespace nxr::pipelines::graph

// nxr-pipelines — the registered NODES of the sensor surface (the definition `sensor_surface`): the channels given a geometry, a triangulated helmet whose vertices ARE
// the channels. Pure functions of arrays from compute/ with their ports and parameters; they read no file, subject or database (the channels and the pick are bound
// by the runner), and have no time axis, so they run once, before any chunk loop (kind `setup`).
//
//   meeg.good_channels_of_type  type, quality -> pick                        the good channels of one sensor type, MEG or EEG (nxr::meeg::goodChannelsOfType), ascending
//   meeg.channel_positions      channels, pick -> positions, channel_index   each picked channel as one point (nxr::meeg::channelPositions: Brainstorm's rule), in the pick's order
//   geometry.point_set_mesh     points -> faces                              the triangles of a point set on a cap, every point a vertex (nxr::geometry::pointset::pointSetMesh)
//   geometry.surface_from_parts vertices, faces -> surface                   the two arrays as a surface to store (the inverse of geometry.surface_parts)
//
// ARRAY CONVENTIONS (an `Array` is [rows x cols] doubles with a `meta`): positions and vertices [n x 3] in the channels' unit and coordinate system (meta.cs, meta.unit);
// faces [F x 3] 0-based indices stored as doubles; channel_index [n] the channel of the channels manifold each row is.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/geometry/point_set_mesh.h"
#include "nxr/meeg/channels.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

Value array_value(const std::string& node, const std::string& port) {
  Value v;
  v.cls = "array";
  v.facts = Json{{"node", node}, {"port", port}};
  return v;
}

StepSpec setup_step(const std::string& name, std::vector<std::string> ins, std::vector<std::string> outs, Json params, const std::string& description, const std::string& why, bool reads_arrays) {
  StepSpec s;
  s.name = name; s.kind = "setup";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, why);
  s.inputs = std::move(ins); s.outputs = std::move(outs);
  s.parameters = std::move(params);
  s.description = description;
  s.reads_arrays = reads_arrays;
  s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  return s;
}

std::vector<std::int32_t> int_list(const Json& j) {
  std::vector<std::int32_t> v;
  if (j.is_array()) for (const auto& x : j) v.push_back(x.get<std::int32_t>());
  return v;
}

}  // namespace

void add_sensor_surface_steps(std::vector<StepSpec>& r) {
  // ── meeg.channel_positions ──
  {
    StepSpec s = setup_step("meeg.channel_positions", {"channels", "pick"}, {"positions", "channel_index"}, Json{{"position_rule", "brainstorm"}, {"coil_points", 4}},
                            "Each picked channel as one point: channels (a points manifold, with its coils and their weights when stored) and pick (an indices selection of it: which channels, "
                            "in which order) -> positions [n x 3] in the channels' unit and coordinate system, and channel_index [n] (the channel each row is). position_rule brainstorm: the mean of "
                            "a channel's first coil_points coil points (all of them when it has fewer; Brainstorm's GetChannelPositions), its stored point when it has no coils; stored: the "
                            "stored point (the importer's: a channel's first coil). Without a pick, every channel in its stored order.",
                            "no time axis: runs once on the channels' geometry", true);
    s.matches = "nxr::meeg::channelPositions (Brainstorm's GetChannelPositions; nxr geom.topography.channel_positions; the Studio's channelPositions)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      need(ins.count("channels") != 0, "channel_positions: the input `channels` is missing");
      const std::string rule = p.value("position_rule", std::string("brainstorm"));
      need(rule == "brainstorm" || rule == "stored", "channel_positions: position_rule is brainstorm or stored, not '" + rule + "'");
      need(p["coil_points"].is_number_integer() && p["coil_points"].get<int>() >= 1, "channel_positions: coil_points is a whole number of at least 1");
      Values out;
      out["positions"] = array_value("meeg.channel_positions", "positions");
      out["channel_index"] = array_value("meeg.channel_positions", "channel_index");
      out["channel_index"].cls = "field";                                  // stored as a field of the surface made from these positions (as glue's map)
      out["channel_index"].value_type = "integer";
      return out;
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext& ctx) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array& C = ins.at("channels");
      const std::string name = C.meta.value("name", std::string());
      need(C.cols == 3 && C.rows > 0, "channel_positions: " + name + " holds no [n x 3] positions");
      const std::int64_t M = C.rows;
      std::vector<std::int32_t> pick;
      Json pick_record = Json{{"name", nullptr}, {"note", "every channel, in its stored order"}};
      if (ins.count("pick")) {
        const Array& S = ins.at("pick");
        for (const double x : S.data) pick.push_back(static_cast<std::int32_t>(std::llround(x)));
        pick_record = Json{{"name", S.meta.value("name", std::string())}, {"id", S.meta.value("id", std::string())}};
      } else {
        for (std::int64_t i = 0; i < M; ++i) pick.push_back(static_cast<std::int32_t>(i));
      }
      need(!pick.empty(), "channel_positions: the pick is empty");
      std::vector<char> seen(static_cast<std::size_t>(M), 0);
      for (const auto c : pick) {
        need(c >= 0 && c < M, "channel_positions: the pick names channel " + std::to_string(c) + ", outside " + name + "'s " + std::to_string(M) + " channels");
        need(!seen[static_cast<std::size_t>(c)], "channel_positions: the pick names channel " + std::to_string(c) + " twice");
        seen[static_cast<std::size_t>(c)] = 1;
      }
      const std::string rule = p.value("position_rule", std::string("brainstorm"));
      const int coil_points = p.value("coil_points", 4);
      const bool coils = rule == "brainstorm" && C.meta.contains("coils");
      std::vector<double> all;
      try {
        if (coils) {
          const Json& k = C.meta["coils"];
          all = nxr::meeg::channelPositions(C.data, k.at("positions").get<std::vector<double>>(), int_list(k.at("indptr")), int_list(k.at("indices")), coil_points);
        } else {
          all = nxr::meeg::channelPositions(C.data, {}, {}, {}, coil_points);
        }
      } catch (const std::invalid_argument& e) { refuse(std::string("channel_positions: ") + e.what()); }
      Array P, I;
      P.rows = static_cast<std::int64_t>(pick.size()); P.cols = 3;
      I.rows = P.rows; I.cols = 1;
      for (const auto c : pick) {
        for (int k = 0; k < 3; ++k) P.data.push_back(all[static_cast<std::size_t>(3 * c + k)]);
        I.data.push_back(c);
      }
      for (const double x : P.data) need(std::isfinite(x), "channel_positions: a picked channel of " + name + " has no position (NaN)");
      const std::string how = coils ? "the mean of each channel's first " + std::to_string(coil_points) + " coil points (Brainstorm's GetChannelPositions)"
                                    : rule == "brainstorm" ? std::string("each channel's stored point (no coils are stored)") : std::string("each channel's stored point (position_rule stored)");
      const Json record = Json{{"channels", name}, {"channels_id", C.meta.value("id", std::string())}, {"pick", pick_record}, {"position_rule", rule}, {"coil_points", coil_points}, {"positions", how}};
      P.meta = Json{{"precision", "float64"}, {"name", name}, {"cs", ctx.bound.coordinate_system_key(C.meta.value("id", std::string()))}, {"sensor_positions", record}};
      I.meta = Json{{"describe", Json{{"dtype", "i32"}, {"value_type", "integer"}, {"shape", Json::array({I.rows})},
                                      {"description", "per vertex: the channel of " + name + " it is (vertex i is channel channel_index[i]: a field on those channels is a field on the vertices)"}}}};
      Arrays o;
      o["positions"] = std::move(P); o["channel_index"] = std::move(I);
      return o;
    };
    r.push_back(std::move(s));
  }
  // ── meeg.good_channels_of_type ──
  {
    StepSpec s = setup_step("meeg.good_channels_of_type", {"type", "quality"}, {"pick"}, Json{{"sensor_type", "required"}},
                            "The good channels of one sensor type, the channels a sensor surface is made of: type and quality (labellings of the channels: a member per channel, each "
                            "member's name, e.g. MEG GRAD and good) -> pick [m] (an indices selection of the channels, ascending). sensor_type (required, no default) is MEG (MEG, MEG MAG, "
                            "MEG GRAD; never MEG REF) or EEG; the two are never mixed. Refused: fewer than four channels chosen, a quality other than good or bad.",
                            "no time axis: runs once on the channels' labellings", true);
    s.matches = "nxr::meeg::goodChannelsOfType";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      need(ins.count("type") != 0, "good_channels_of_type: the input `type` (the channels' type labelling) is missing");
      need(ins.count("quality") != 0, "good_channels_of_type: the input `quality` (the channels' good/bad labelling) is missing");
      const std::string type = p.value("sensor_type", std::string("required"));
      need(type == "MEG" || type == "EEG", "good_channels_of_type: sensor_type is MEG or EEG (required, no default), not '" + type + "'");
      Values out;
      out["pick"] = array_value("meeg.good_channels_of_type", "pick");
      return out;
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      auto names_per_channel = [](const Array& labelling, const std::string& what) {
        const Json names = labelling.meta.value("member_names", Json::array());
        need(names.is_array() && !names.empty(), "good_channels_of_type: the " + what + " labelling " + labelling.meta.value("name", std::string()) + " carries no member names");
        std::vector<std::string> per_channel;
        for (const double member : labelling.data) {
          const auto k = static_cast<std::int64_t>(std::llround(member));
          per_channel.push_back(k >= 0 && k < static_cast<std::int64_t>(names.size()) ? names[static_cast<std::size_t>(k)].get<std::string>() : std::string());
        }
        return per_channel;
      };
      const Array& types = ins.at("type");
      const Array& quality = ins.at("quality");
      const std::string sensor_type = p.value("sensor_type", std::string("required"));
      std::vector<std::int32_t> chosen;
      try {
        chosen = nxr::meeg::goodChannelsOfType(names_per_channel(types, "type"), names_per_channel(quality, "quality"), sensor_type);
      } catch (const std::invalid_argument& e) { refuse(std::string("good_channels_of_type: ") + e.what()); }
      Array pick;
      pick.rows = static_cast<std::int64_t>(chosen.size()); pick.cols = 1;
      for (const auto c : chosen) pick.data.push_back(c);
      pick.meta = Json{{"good_channels_of_type", Json{{"sensor_type", sensor_type}, {"type", types.meta.value("name", std::string())}, {"quality", quality.meta.value("name", std::string())}}}};
      Arrays o;
      o["pick"] = std::move(pick);
      return o;
    };
    r.push_back(std::move(s));
  }
  // ── geometry.point_set_mesh ──
  {
    StepSpec s = setup_step("geometry.point_set_mesh", {"points"}, {"faces"}, Json{{"cut", 2.0}, {"pole", Json::array({0, 0, 1})}, {"orientation", "outward"}, {"repair_passes", 20}},
                            "The triangles of a point set on a cap, every point a vertex: points [n x 3] -> faces [F x 3]. The points projected about the pole (an azimuthal equidistant map "
                            "from their centroid), the planar Delaunay, triangles with an edge of cut x the median edge or longer removed, pinched vertices and holes repaired, each triangle "
                            "turned outward (from the centroid) or left as_triangulated. Refused when a point is in no triangle. The mesh's topology (boundary loops, Euler characteristic) "
                            "and what the repair did are in meta.mesh.",
                            "no time axis: runs once on the points", false);
    s.matches = "nxr::geometry::pointset::pointSetMesh (nxr's sensor mesh: heatdist.sensor_mesh and SensorSurface's outward turn)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      need(ins.count("points") != 0, "point_set_mesh: the input `points` is missing");
      need(p["cut"].is_number() && p["cut"].get<double>() > 0, "point_set_mesh: cut is a positive multiple of the median edge");
      need(p["pole"].is_array() && p["pole"].size() == 3, "point_set_mesh: pole is a direction [x, y, z]");
      const std::string o = p.value("orientation", std::string("outward"));
      need(o == "outward" || o == "as_triangulated", "point_set_mesh: orientation is outward or as_triangulated, not '" + o + "'");
      need(p["repair_passes"].is_number_integer() && p["repair_passes"].get<int>() >= 1, "point_set_mesh: repair_passes is a whole number of at least 1");
      Values out;
      out["faces"] = array_value("geometry.point_set_mesh", "faces");
      return out;
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array& X = ins.at("points");
      need(X.cols == 3, "point_set_mesh: the points are not [n x 3]");
      nxr::geometry::pointset::PointSetMeshOptions o;
      o.cut = p["cut"].get<double>();
      for (int k = 0; k < 3; ++k) o.pole[static_cast<std::size_t>(k)] = p["pole"][static_cast<std::size_t>(k)].get<double>();
      o.orientation = p.value("orientation", std::string("outward")) == "as_triangulated" ? nxr::geometry::pointset::Orientation::as_triangulated : nxr::geometry::pointset::Orientation::outward;
      o.repair_passes = p["repair_passes"].get<int>();
      nxr::geometry::pointset::PointSetMesh m;
      try { m = nxr::geometry::pointset::pointSetMesh(X.data, o); } catch (const std::invalid_argument& e) { refuse(std::string("point_set_mesh: ") + e.what()); }
      if (!m.points_without_face.empty())
        refuse("point_set_mesh: " + std::to_string(m.points_without_face.size()) + " point(s) are in no triangle (the first: " + std::to_string(m.points_without_face.front()) +
               "): a larger cut, or another pole, keeps them");
      Array F;
      F.rows = static_cast<std::int64_t>(m.faces.size() / 3); F.cols = 3;
      for (const auto f : m.faces) F.data.push_back(f);
      F.meta = Json{{"mesh", Json{{"method", "pointSetMesh: azimuthal equidistant projection about the pole, planar Delaunay, cut, pinch and hole repair, orientation"},
                                  {"cut", o.cut}, {"pole", p["pole"]}, {"orientation", p.value("orientation", std::string("outward"))}, {"repair_passes", o.repair_passes},
                                  {"median_edge", m.median_edge}, {"delaunay_triangles", m.delaunay_triangles}, {"cut_triangles", m.cut_triangles}, {"pinches_closed", m.pinches_closed},
                                  {"fans_dropped", m.fans_dropped}, {"hole_triangles", m.hole_triangles}, {"edges", m.edges}, {"boundary_loops", m.boundary_loops},
                                  {"euler_characteristic", m.euler_characteristic}}}};
      Arrays out;
      out["faces"] = std::move(F);
      return out;
    };
    r.push_back(std::move(s));
  }
  // ── geometry.surface_from_parts ──
  {
    StepSpec s = setup_step("geometry.surface_from_parts", {"vertices", "faces"}, {"surface"}, Json::object(),
                            "Two arrays as a surface to store: vertices [V x 3] (meta.cs, the coordinate system) and faces [F x 3] (0-based indices) -> surface (the vertices, with the faces, the "
                            "coordinate system and what the producing nodes recorded: meta.sensor_positions of the vertices, meta.mesh of the faces). The inverse of geometry.surface_parts.",
                            "indexing only", false);
    s.matches = "the surface's stored vertices and faces";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      for (const char* k : {"vertices", "faces"}) need(ins.count(k) != 0, std::string("surface_from_parts: the input `") + k + "` is missing");
      Values out;
      out["surface"] = array_value("geometry.surface_from_parts", "surface");
      out["surface"].cls = "manifold";
      return out;
    };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) {
      const Array &V = ins.at("vertices"), &F = ins.at("faces");
      need(V.cols == 3 && V.rows >= 3, "surface_from_parts: the vertices are not [V x 3]");
      need(F.cols == 3 && F.rows >= 1, "surface_from_parts: the faces are not [F x 3]");
      std::vector<std::int32_t> faces;
      for (const double x : F.data) {
        const auto f = static_cast<std::int32_t>(std::llround(x));
        need(f >= 0 && f < V.rows, "surface_from_parts: a face names vertex " + std::to_string(f) + ", outside the " + std::to_string(V.rows) + " vertices");
        faces.push_back(f);
      }
      Json producer = Json::object();
      if (V.meta.is_object() && V.meta.contains("sensor_positions")) producer["sensor_surface"] = V.meta["sensor_positions"];
      if (F.meta.is_object() && F.meta.contains("mesh")) producer["mesh"] = F.meta["mesh"];
      std::string description = "a surface of " + std::to_string(V.rows) + " vertices and " + std::to_string(F.rows) + " triangles";
      if (producer.contains("sensor_surface"))
        description = "the sensors of " + producer["sensor_surface"].value("channels", std::string()) + " as a surface: vertex i is the channel `_map`[i] (" +
                      producer["sensor_surface"]["positions"].get<std::string>() + "), triangulated by the point-set mesher";
      Array S;
      S.rows = V.rows; S.cols = 3; S.data = V.data;
      S.meta = Json{{"describe", Json{{"faces", faces}, {"cs", V.meta.is_object() ? V.meta.value("cs", std::string()) : std::string()}, {"description", description}, {"producer", producer}}}};
      Arrays out;
      out["surface"] = std::move(S);
      return out;
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph

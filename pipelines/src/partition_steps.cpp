// nxr-pipelines — THE PARTITION NODES: partitions made from partitions, as pure nodes of nxr::graph::registry() (grouped names). A node reads the records of the
// selections it is handed (their member counts and parameters) and describes the partition it made; the definition says where it is stored.
//
//   positions.transition  array  positions (a manifold) + transition (an operator of kind transition)  ->  moved: the positions in the target system
//   joint.cells  array  space (a level of a surface's ladder) + time (a level of a line's constant-Q ladder)  ->  cells: the joint cells, code = (tile << bits) | frame
#include "nxr/pipelines/partition_steps.h"

#include <algorithm>
#include <string>

#include "nxr/graph/node.h"
#include "nxr/pipelines/flow_steps.h"
#include "nxr/pipelines/pca_steps.h"
#include "nxr/pipelines/pyramid_steps.h"
#include "nxr/selection/selection.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

ng::PortType constant_port(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}

// ── joint.cells ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE JOINT PARTITION OF SPACE x TIME (the hand-wired `joint` it replaces): the cells of a surface level's tiles and a constant-Q level's time tiles, closed form
// (no element stored): code = (tile << bits) | frame, the Studio atlas's layout.
ng::NodeSpec joint_cells() {
  ng::NodeSpec n;
  n.name = "joint.cells";
  n.kind = ng::Kind::array;
  n.inputs = {{"space", "a level of the surface's tile ladder (a set of its vertices: the tiles)", false},
              {"time", "a level of the time line's constant-Q ladder (a closed-form tiling: the time tiles; its record names its band)", false}};
  n.outputs = {{"space_time", "the product of the surface and the time line the cells lie on (its factors are the definition's)", false},
               {"cells", "the joint cells: (tile << bits) | frame, closed form", false}};
  n.parameters = {{"layout", Json::array({"tile", "frame"}), "the array layout of the measurements over the cells: [tile, frame], or [band, tile, frame] when each band's values are one array (the pyramid)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "integer codes from the two levels' counts");
  n.matches = "the Studio atlas's joint cells (tile << bits | frame; Studio D129 joint partition); compute nxr::selection::joint";
  n.description = "the joint cells of a surface level and a time level, closed form";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"space", "time"}) if (!in.count(k)) ng::refuse(std::string("joint.cells: ") + k + " is not given");
    ng::Planned out;
    out.out["space_time"] = constant_port("manifold");
    out.out["cells"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    for (const char* k : {"space", "time"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("joint.cells: ") + k + " was not handed over");
    ng::Arrays a{{"space", c.at("space")}, {"time", c.at("time")}, {"layout", ng::Array{}}};
    a["layout"].meta["layout"] = p.contains("layout") && p["layout"].is_array() ? p["layout"] : Json::array({"tile", "frame"});
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& c = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array& S = c.at("space");
    const ng::Array& T = c.at("time");
    const Json tp = T.meta.value("params", Json::object());
    // the time grid is a constant-Q level (its record names its band: the level IS the band) or a frame tiling of the tower (the pyramid's frames: no band); both are tilings of the time line
    if (!tp.is_object() || !tp.contains("tiling") || !(tp.contains("resolves_hz") || tp.contains("atlas"))) ng::refuse("joint.cells: " + T.meta.value("name", std::string()) + " is not a time grid (a constant-Q level or a frame tiling: no tiling in its record)");
    const std::int64_t frames = tp["tiling"]["count"].get<std::int64_t>();
    const std::int64_t tiles = S.meta.value("n_members", std::int64_t(0));
    if (tiles <= 0) ng::refuse("joint.cells: " + S.meta.value("name", std::string()) + " names no tile count");
    // the joint partition is compute's (Studio D129): the product of the grids, the FIRST factor (the frames) given a radix of 2^bits, so a code is (tile << bits) | frame
    nxr::selection::Selection jp;
    try {
      jp = nxr::selection::joint({nxr::selection::tiled(nxr::selection::Tiling{0.0, 1.0, 1.0, frames}, "s"), nxr::selection::elements_grid(tiles)});
    } catch (const std::invalid_argument& e) { ng::refuse(std::string("joint.cells: ") + e.what()); }
    int bits = 0;
    while ((std::int64_t(1) << bits) < jp.radices.at(0)) ++bits;
    if ((tiles << bits) >= (std::int64_t(1) << 31)) ng::refuse("joint.cells: " + std::to_string(tiles) + " tiles x 2^" + std::to_string(bits) + " time indices do not fit a 32-bit code");
    Json joint = Json{{"space", S.meta.value("id", std::string())}, {"time", T.meta.value("id", std::string())}, {"time_ladder", T.meta.value("member_of_id", std::string())},
                      {"bits", bits}, {"tiles", tiles}, {"frames", frames}, {"layout", c.at("layout").meta["layout"]}, {"code", "(tile << bits) | frame"},
                      {"window_samples", tp.value("window_samples", std::int64_t(0))}};
    if (tp.contains("resolves_hz")) joint["band"] = Json::array({tp["resolves_hz"]["lo"], tp["resolves_hz"]["hi"]});
    else joint["frame_level"] = tp["atlas"].value("frame_level", Json());
    Json params = Json{{"joint", joint}};
    const Json sp = S.meta.value("params", Json::object());                  // the leaves' record the roll-up reads: where their areas and arrays are (nxr's `atlas` block)
    if (sp.is_object() && sp.contains("areas")) params["atlas"] = Json{{"tree", sp.value("tree", std::string("subject"))}, {"areas", sp["areas"]}, {"node", sp.value("node", Json())}};
    ng::Array a;
    a.meta["describe"] = Json{{"n_members", tiles * frames}, {"label", std::to_string(tiles) + " tiles x " + std::to_string(frames) + " time tiles"}, {"params", params},
                              {"description", "the joint cells of " + S.meta.value("name", std::string()) + " and " + T.meta.value("name", std::string()) + ": code = (tile << " +
                                              std::to_string(bits) + ") | time tile (the Studio atlas's layout), closed form"}};
    return ng::Arrays{{"space_time", ng::Array{}}, {"cells", std::move(a)}};
  };
  return n;
}


// ── positions.transition ────────────────────────────────────────────────────────────────────────────────────────
// A MANIFOLD'S POSITIONS IN ANOTHER COORDINATE SYSTEM OF ITS SPACE (type system T5): an operator of kind transition, an affine [3 x 4] (or [4 x 4]) applied to [x y z 1]. The checker
// names this node when two manifolds' systems differ and never inserts it. Its output is the same manifold term with the target system; stored, a surface in that system.
ng::NodeSpec positions_transition() {
  ng::NodeSpec n;
  n.name = "positions.transition";
  n.kind = ng::Kind::array;
  n.inputs = {{"positions", "a manifold with positions (a surface: vertices n x 3, its faces with them)", false},
              {"transition", "an operator of kind transition: an affine [3 x 4] or [4 x 4] from the positions' system to another of the same space", false}};
  n.outputs = {{"moved", "the positions in the transition's target system (a surface: its faces kept)", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "positions in double");
  n.matches = "Brainstorm's cs_convert; the Studio's coordinate transitions (one operator a pair of systems)";
  n.description = "a manifold's positions brought into another coordinate system of the same space by its transition operator";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"positions", "transition"}) if (!in.count(k)) ng::refuse(std::string("positions.transition: ") + k + " is not given");
    ng::Planned out;
    ng::PortType m = in.at("positions");
    m.facts["cls"] = "manifold";
    out.out["moved"] = m;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json&, ng::Shared&) {
    for (const char* k : {"positions", "transition"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("positions.transition: ") + k + " was not handed over");
    ng::Arrays a{{"positions", c.at("positions")}, {"transition", c.at("transition")}};
    if (in.count("transition") && in.at("transition").transition) a["transition"].meta["to_cs"] = in.at("transition").transition->second.name;
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& c = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array& X = c.at("positions");
    const ng::Array& T = c.at("transition");
    if (X.cols != 3) ng::refuse("positions.transition: the positions are not n x 3");
    if (T.cols != 4 || (T.rows != 3 && T.rows != 4)) ng::refuse("positions.transition: the transition is not an affine [3 x 4] or [4 x 4]");
    ng::Array Y;
    Y.rows = X.rows; Y.cols = 3;
    Y.data.resize(X.data.size());
    for (std::int64_t i = 0; i < X.rows; ++i)
      for (int r = 0; r < 3; ++r) {
        double v = T.data[static_cast<std::size_t>(r * 4 + 3)];
        for (int k = 0; k < 3; ++k) v += T.data[static_cast<std::size_t>(r * 4 + k)] * X.data[static_cast<std::size_t>(i * 3 + k)];
        Y.data[static_cast<std::size_t>(i * 3 + r)] = v;
      }
    Y.meta["describe"] = Json{{"faces", X.meta.value("faces", Json::array())}, {"cs", T.meta.value("to_cs", std::string())},
                              {"description", X.meta.value("name", std::string()) + " in " + T.meta.value("to_cs", std::string()) + " by " + T.meta.value("name", std::string())}};
    return ng::Arrays{{"moved", std::move(Y)}};
  };
  return n;
}

}  // namespace

void register_partition_nodes() {
  register_pyramid_nodes();
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {joint_cells(), positions_transition()})
    if (!r.find(n.name)) r.add(std::move(n));
  register_pca_nodes();
  register_flow_nodes();
}

}  // namespace nxr::pipelines::graph

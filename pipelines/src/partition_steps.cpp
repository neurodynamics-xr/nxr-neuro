// nxr-pipelines — THE PARTITION NODES: partitions made from partitions, as pure nodes of nxr::graph::registry() (grouped names). A node reads the records of the
// selections it is handed (their member counts and parameters) and describes the partition it made; the definition says where it is stored.
//
//   joint.cells  array  space (a level of a surface's ladder) + time (a level of a line's constant-Q ladder)  ->  cells: the joint cells, code = (tile << bits) | frame
#include "nxr/pipelines/partition_steps.h"

#include <algorithm>
#include <string>

#include "nxr/graph/node.h"

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

// the bits a frame index takes in a joint code: the fewest that hold `frames` (at least 1)
int joint_bits(std::int64_t frames) {
  int b = 1;
  while ((std::int64_t(1) << b) < std::max<std::int64_t>(2, frames)) ++b;
  return b;
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
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "integer codes from the two levels' counts");
  n.matches = "the Studio atlas's joint cells (tile << bits | frame)";
  n.description = "the joint cells of a surface level and a time level, closed form";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"space", "time"}) if (!in.count(k)) ng::refuse(std::string("joint.cells: ") + k + " is not given");
    ng::Planned out;
    out.out["space_time"] = constant_port("manifold");
    out.out["cells"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json&, ng::Shared&) {
    for (const char* k : {"space", "time"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("joint.cells: ") + k + " was not handed over");
    return std::any(ng::Arrays{{"space", c.at("space")}, {"time", c.at("time")}});
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& c = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array& S = c.at("space");
    const ng::Array& T = c.at("time");
    const Json tp = T.meta.value("params", Json::object());
    if (!tp.is_object() || !tp.contains("tiling") || !tp.contains("resolves_hz")) ng::refuse("joint.cells: " + T.meta.value("name", std::string()) + " is not a constant-Q level (no tiling or band in its record)");
    const std::int64_t frames = tp["tiling"]["count"].get<std::int64_t>();
    const std::int64_t tiles = S.meta.value("n_members", std::int64_t(0));
    if (tiles <= 0) ng::refuse("joint.cells: " + S.meta.value("name", std::string()) + " names no tile count");
    const int bits = joint_bits(frames);
    if ((tiles << bits) >= (std::int64_t(1) << 31)) ng::refuse("joint.cells: " + std::to_string(tiles) + " tiles x 2^" + std::to_string(bits) + " time indices do not fit a 32-bit code");
    const Json joint = Json{{"space", S.meta.value("id", std::string())}, {"time", T.meta.value("id", std::string())}, {"time_ladder", T.meta.value("member_of_id", std::string())},
                            {"band", Json::array({tp["resolves_hz"]["lo"], tp["resolves_hz"]["hi"]})}, {"bits", bits}, {"tiles", tiles}, {"frames", frames},
                            {"layout", Json::array({"tile", "frame"})}, {"code", "(tile << bits) | frame"}, {"window_samples", tp.value("window_samples", std::int64_t(0))}};
    ng::Array a;
    a.meta["describe"] = Json{{"n_members", tiles * frames}, {"label", std::to_string(tiles) + " tiles x " + std::to_string(frames) + " time tiles"}, {"params", Json{{"joint", joint}}},
                              {"description", "the joint cells of " + S.meta.value("name", std::string()) + " and " + T.meta.value("name", std::string()) + ": code = (tile << " +
                                              std::to_string(bits) + ") | time tile (the Studio atlas's layout), closed form"}};
    return ng::Arrays{{"space_time", ng::Array{}}, {"cells", std::move(a)}};
  };
  return n;
}

}  // namespace

void register_partition_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {joint_cells()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph

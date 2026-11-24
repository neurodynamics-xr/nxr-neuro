// nxr-pipelines — THE PARTITION NODES: partitions made from partitions, as pure nodes of nxr::graph::registry() (grouped names). A node reads the records of the
// selections it is handed (their member counts and parameters) and describes the partition it made; the definition says where it is stored.
//
//   positions.transition  array  positions (a manifold) + transition (an operator of kind transition)  ->  moved: the positions in the target system
//   (joint.cells is retired: the maker `select.product`, select_nodes.cpp)
#include "nxr/pipelines/partition_steps.h"

#include <algorithm>
#include <string>

#include "nxr/graph/node.h"
#include "nxr/pipelines/flow_steps.h"
#include "nxr/pipelines/form_steps.h"
#include "nxr/pipelines/sensor_flow_steps.h"
#include "nxr/pipelines/spectral_steps.h"
#include "nxr/pipelines/identification_steps.h"
#include "nxr/pipelines/pca_steps.h"
#include "nxr/selection/selection.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {


// ── positions.transition ────────────────────────────────────────────────────────────────────────────────────────
// A MANIFOLD'S POSITIONS IN ANOTHER COORDINATE SYSTEM OF ITS SPACE (type system T5): an operator of kind transition, an affine [3 x 4] (or [4 x 4]) applied to [x y z 1]. The checker
// names this node when two manifolds' systems differ and never inserts it. Its output is the same manifold term with the target system; stored, a surface in that system.
ng::NodeSpec positions_transition() {
  ng::NodeSpec n;
  n.name = "meeg.positions_transition";
  n.kind = ng::Kind::array;
  n.inputs = {{"positions", "a manifold with positions (a surface: vertices n x 3, its faces with them)", false},
              {"transition", "an operator of kind transition: an affine [3 x 4] or [4 x 4] from the positions' system to another of the same space", false}};
  n.outputs = {{"moved", "the positions in the transition's target system (a surface: its faces kept)", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "positions in double");
  n.matches = "Brainstorm's cs_convert; the Studio's coordinate transitions (one operator a pair of systems)";
  n.description = "a manifold's positions brought into another coordinate system of the same space by its transition operator";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"positions", "transition"}) if (!in.count(k)) ng::refuse(std::string("meeg.positions_transition: ") + k + " is not given");
    ng::Planned out;
    ng::PortType m = in.at("positions");
    m.facts["cls"] = "manifold";
    out.out["moved"] = m;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json&, ng::Shared&) {
    for (const char* k : {"positions", "transition"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("meeg.positions_transition: ") + k + " was not handed over");
    ng::Arrays a{{"positions", c.at("positions")}, {"transition", c.at("transition")}};
    if (in.count("transition") && in.at("transition").transition) a["transition"].meta["to_cs"] = in.at("transition").transition->second.name;
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& c = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array& X = c.at("positions");
    const ng::Array& T = c.at("transition");
    if (X.cols != 3) ng::refuse("meeg.positions_transition: the positions are not n x 3");
    if (T.cols != 4 || (T.rows != 3 && T.rows != 4)) ng::refuse("meeg.positions_transition: the transition is not an affine [3 x 4] or [4 x 4]");
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
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {positions_transition()})
    if (!r.find(n.name)) r.add(std::move(n));
  register_pca_nodes();
  register_identification_nodes();
  register_flow_nodes();
  register_form_nodes();
  register_sensor_flow_nodes();
  register_spectral_nodes();
}

}  // namespace nxr::pipelines::graph

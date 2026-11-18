// nxr-graph — `compute`: A NODE OR A GRAPH ON ARRAYS HANDED TO IT, through the same checker as `run` (spec docs/specs/2026-11-17-type-system.md, section 10; step T4).
//
// `run` finds its inputs' types in stored rows; `compute` makes them from the arrays and what the caller says about them (`ArrayMeta`). The check is the same function
// (`check`, graph.h): only the origin of the input types differs.
//
//   - A field array [rows x cols x comps] is typed `field<#1 x #2; V>`, #1 and #2 ANONYMOUS manifolds (equal only to themselves). With a sample rate its last factor is a time
//     line (a streamed input); the value type is the caller's, else from the components (1 real, 2 complex, 3 vector3, 6 complex vector3).
//   - LABELS name the anonymous manifolds: two arrays labelled `on = {"sensors", "time"}` lie on the SAME sensors and time line; an operator labelled `from = "sensors"`,
//     `to = "cortex"` reads them. A label met with two different sizes is refused. Unlabelled arrays never unify with each other, except through a node's variables.
//   - A partition is an integer member index per element (-1: none) with the label of the factor it partitions: `selection<set on #k; partition>`.
//   - A complex tangent array (spin != 0) needs its gauge (G5): it is refused without one. A vector3 value is gauge-free.
//   - A coordinate system (`coordinate_systems`: label -> space/name/unit) is part of the manifold's type; no transition is ever inserted (T5).
//   - A file's SIDECAR holds each array's printed type (`sidecar`), so `compute` on files reads its types back exactly (`type_from_sidecar`).
#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/graph/execution.h"

namespace nxr::graph {

/** what the caller says about an array that the array does not carry */
struct ArrayMeta {
  std::string cls = "field";                    // field | operator | partition | constant (an array no class is claimed for: checked by the nodes' plans alone)
  std::vector<std::string> on;                  // the labels of a field's factors in axis order ({"sensors", "time"}); of a partition, the one factor it partitions
  std::string from, to;                         // an operator's source (its columns) and target (its rows)
  std::optional<double> sfreq;                  // Hz: the array is streamed (its columns are samples)
  std::string unit;                             // SI
  std::string value;                            // real | complex | vector3 | quaternion | hermitian3 | "complex vector3"; empty: from the components
  std::string quantity;                         // what the values are (the closed vocabulary); empty: `recording` for a streamed field
  int spin = 0;                                 // G5: 0 scalar, 1 tangent vector, n an n-RoSy
  std::string gauge;                            // G5: the gauge of a tangent array
  std::map<std::string, CoordinateSystem> coordinate_systems;   // label -> the system its positions are in
};

struct ComputeInput {
  Array array;
  ArrayMeta meta;
  std::optional<PortType> type;                 // a type read back from a sidecar: used as it is, the metadata ignored
};

/** makes the types of the arrays of ONE call: the anonymous manifolds are shared by label across the arrays of the call */
class ArrayTyper {
 public:
  PortType type_of(const std::string& name, const Array& a, const ArrayMeta& m);
 private:
  int next_ = 1;
  std::map<std::string, ManifoldRef> labelled_;
  ManifoldRef factor(const std::string& name, const std::string& label, std::int64_t n, ManifoldKind kind, Role role, std::optional<double> spacing,
                     const ArrayMeta& m);
};

/** the graph of one use of `node`: its input ports (`ports`; empty: all) are the graph's inputs, its output ports the graph's outputs (the port names are the value names); `with` sets its parameters */
Graph graph_of_node(const NodeSpec& node, const std::vector<std::string>& ports = {}, const Json& with = Json::object());

struct ComputeResult {
  Arrays arrays;                                // the graph's outputs
  Types types;                                  // every output's type, checked before any data
  Json sidecar;                                 // {"nxr_types": 1, "run": id, "types": {output: printed type}}
  Json record;                                  // every node's resolved parameters and what its preparation resolved
};

/** the types of the arrays, then `check`: refuses with the use, the port, the two types and what to do, before any array is read */
Checked check_arrays(const Graph& g, const std::map<std::string, ComputeInput>& inputs, const Json& params = Json::object(),
                     const Registry& registry = nxr::graph::registry(), const DispatchContext& ctx = {});
/** the same check, then `apply`: whole arrays as one block */
ComputeResult compute(const Graph& g, const std::map<std::string, ComputeInput>& inputs, const Json& params = Json::object(),
                      const Registry& registry = nxr::graph::registry(), const DispatchContext& ctx = {}, const Blocking& blocking = {});
/** one registered node by name, on arrays by port */
ComputeResult compute_node(const std::string& node, const std::map<std::string, ComputeInput>& inputs, const Json& params = Json::object(),
                           const Registry& registry = nxr::graph::registry(), const DispatchContext& ctx = {}, const Blocking& blocking = {});

/** the types of a result as a sidecar, and one type read back from it; the anonymous manifolds of a sidecar are renamed by its run id, so two files' `#1` are never one manifold */
Json sidecar_of(const Types& types, const std::string& run_id);
PortType type_from_sidecar(const Json& sidecar, const std::string& name);

}  // namespace nxr::graph

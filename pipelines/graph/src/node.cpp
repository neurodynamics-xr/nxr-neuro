// nxr-graph — the nodes' declarations: names for the traits, the trait check, and the registry of nodes (node.h).
#include <map>
#include "nxr/graph/node.h"
#include "nxr/graph/frame_nodes.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace nxr::graph {

void refuse(const std::string& message) { throw std::invalid_argument(message); }

const char* to_string(ValueType v) {
  switch (v) {
    case ValueType::real: return "real";
    case ValueType::complex: return "complex";
    case ValueType::vector3: return "vector3";
    case ValueType::quaternion: return "quaternion";
    case ValueType::hermitian3: return "hermitian3";
    case ValueType::monogenic: return "monogenic";
    case ValueType::integer: return "integer";
  }
  return "?";
}
const char* to_string(PortClass c) {
  switch (c) {
    case PortClass::field: return "field";
    case PortClass::operator_: return "operator";
    case PortClass::selection: return "selection";
    case PortClass::partition: return "partition";
    case PortClass::constant: return "constant";
    case PortClass::measurement: return "measurement";
    case PortClass::statistic: return "statistic";
    case PortClass::manifold: return "manifold";
  }
  return "?";
}
const char* to_string(Axis a) {
  switch (a) {
    case Axis::none: return "none";
    case Axis::samples: return "samples";
    case Axis::frames: return "frames";
    case Axis::members: return "members";
  }
  return "?";
}
const char* to_string(TimeBehaviour t) {
  switch (t) {
    case TimeBehaviour::undeclared: return "undeclared";
    case TimeBehaviour::pointwise: return "pointwise";
    case TimeBehaviour::windowed: return "windowed";
    case TimeBehaviour::stateful: return "stateful";
  }
  return "?";
}
const char* to_string(Precision p) {
  switch (p) {
    case Precision::undeclared: return "undeclared";
    case Precision::float32: return "float32";
    case Precision::float64: return "float64";
  }
  return "?";
}
const char* to_string(OutputPrecision p) {
  switch (p) {
    case OutputPrecision::undeclared: return "undeclared";
    case OutputPrecision::follows_input: return "follows_input";
    case OutputPrecision::float64: return "float64";
  }
  return "?";
}
const char* to_string(Kind k) {
  switch (k) {
    case Kind::array: return "array";
    case Kind::chunk: return "chunk";
    case Kind::accumulate: return "accumulate";
  }
  return "?";
}

void check_traits(const NodeSpec& node) {
  if (!node.traits.declared())
    refuse(node.name + ": the node declares no (or incomplete) traits: time behaviour, numeric precision and, for a windowed node, its window rule (ARCHITECTURE.md, node traits)");
}

void Registry::add(NodeSpec node) {
  if (node.name.empty()) refuse("a node needs a name");
  if (nodes_.count(node.name)) refuse("node " + node.name + " is registered twice");
  if (!node.signature) {
    if (const Signature* s = builtin_signature(node.name)) {
      // the signature's ports are the node's: the same input names, the same outputs in the same order (the definitions' `out` lists follow it)
      std::set<std::string> in;
      for (const auto& p : node.inputs) in.insert(p.name);
      bool fits = s->inputs.size() == in.size() && s->outputs.size() == node.outputs.size();
      for (const auto& [port, _] : s->inputs) fits = fits && in.count(port);
      for (std::size_t i = 0; fits && i < s->outputs.size(); ++i) fits = s->outputs[i].first == node.outputs[i].name;
      if (fits) node.signature = *s; else unmatched_.push_back(node.name);
    }
  }
  const std::string name = node.name;
  nodes_.emplace(name, std::move(node));
}
const NodeSpec* Registry::find(const std::string& name) const {
  const auto it = nodes_.find(name);
  return it == nodes_.end() ? nullptr : &it->second;
}
std::vector<std::string> Registry::names() const {
  std::vector<std::string> v;
  for (const auto& [n, _] : nodes_) v.push_back(n);
  return v;
}
namespace {
std::size_t edit_distance(const std::string& a, const std::string& b) {
  std::vector<std::size_t> prev(b.size() + 1), cur(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) prev[j] = j;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    cur[0] = i;
    for (std::size_t j = 1; j <= b.size(); ++j) cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0u : 1u)});
    std::swap(prev, cur);
  }
  return prev[b.size()];
}
std::string group_of(const std::string& n) { return n.substr(0, n.find('.')); }
}  // namespace
std::vector<std::string> Registry::near(const std::string& name, std::size_t limit) const {
  std::vector<std::pair<std::pair<int, std::size_t>, std::string>> scored;
  for (const auto& [n, _] : nodes_) {
    const bool same_group = group_of(n) == group_of(name);
    const std::size_t d = edit_distance(n, name);
    if (same_group || d <= 3) scored.push_back({{same_group ? 0 : 1, d}, n});
  }
  std::sort(scored.begin(), scored.end());
  std::vector<std::string> out;
  for (std::size_t i = 0; i < scored.size() && i < limit; ++i) out.push_back(scored[i].second);
  return out;
}
const NodeSpec& Registry::at(const std::string& name) const {
  if (const NodeSpec* n = find(name)) return *n;
  // renamed to the toolbox words: refused with the new name, never answered as an alias (the test names_across_layers reads this table)
  static const std::map<std::string, std::string> retired = {
      {"frame.embed", "geometry.frame_embed"}, {"frame.project", "geometry.frame_project"}, {"frame.rotate", "geometry.frame_rotate"}};
  if (const auto it = retired.find(name); it != retired.end()) refuse("the node '" + name + "' was renamed: use " + it->second);
  std::string msg = "unknown node '" + name + "'";
  const auto near_names = near(name);
  if (!near_names.empty()) {
    msg += "; near names:";
    for (const auto& n : near_names) msg += " " + n;
  } else {
    msg += "; no node of that group or a near name is registered";
  }
  refuse(msg);
}
Registry& registry() {
  static Registry r = [] { Registry x; register_frame_nodes(x); return x; }();
  return r;
}

}  // namespace nxr::graph

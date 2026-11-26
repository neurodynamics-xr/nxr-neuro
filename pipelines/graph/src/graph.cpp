// nxr-graph — THE CHECK of a graph on types (graph.h), and the graph's JSON form.
#include "nxr/graph/graph.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>

namespace nxr::graph {

const char* to_string(Section s) {
  switch (s) {
    case Section::before: return "before";
    case Section::chunks: return "chunks";
    case Section::after: return "after";
  }
  return "?";
}

namespace {

std::vector<std::string> strings_of(const Json& j, const std::string& where) {
  std::vector<std::string> v;
  if (j.is_string()) { v.push_back(j.get<std::string>()); return v; }
  if (!j.is_array()) refuse(where + ": a name or a list of names");
  for (const auto& e : j) {
    if (!e.is_string()) refuse(where + ": names are strings");
    v.push_back(e.get<std::string>());
  }
  return v;
}

std::vector<Use> uses_of(const Json& doc, const char* section) {
  std::vector<Use> v;
  if (!doc.contains(section)) return v;
  if (!doc[section].is_array()) refuse(std::string(section) + ": a list of uses");
  int i = 0;
  for (const auto& u : doc[section]) {
    const std::string where = std::string(section) + "[" + std::to_string(i++) + "]";
    if (!u.is_object() || !u.contains("node") || !u["node"].is_string()) refuse(where + ": a use names its `node`");
    for (const auto& [k, _] : u.items())
      if (k != "node" && k != "in" && k != "with" && k != "out") refuse(where + ": unknown key '" + k + "' (a use has node, in, with, out)");
    Use use;
    use.node = u["node"].get<std::string>();
    if (u.contains("in")) {
      if (!u["in"].is_object()) refuse(where + ": `in` maps ports to value names");
      for (const auto& [port, name] : u["in"].items()) {
        if (!name.is_string()) refuse(where + ": `in." + port + "` is a value name");
        use.in[port] = name.get<std::string>();
      }
    }
    if (u.contains("with")) {
      if (!u["with"].is_object()) refuse(where + ": `with` maps parameters to values");
      use.with = u["with"];
    }
    if (u.contains("out")) use.out = strings_of(u["out"], where + ": out");
    v.push_back(std::move(use));
  }
  return v;
}

bool compatible(const Json& def, const Json& v) {
  if (def.is_null()) return true;
  if (def.is_number() && v.is_number()) return true;
  return def.type() == v.type();
}
const char* kind_of(const Json& v) {
  if (v.is_null()) return "null";
  if (v.is_boolean()) return "a boolean";
  if (v.is_number()) return "a number";
  if (v.is_string()) return "a string";
  if (v.is_array()) return "a list";
  return "a table";
}

Json default_of(const Json& entry) { return entry.is_object() && entry.contains("default") ? entry["default"] : entry; }

// the graph's parameters resolved: the defaults, then `set`
Json resolve_graph_parameters(const Graph& g, const Json& set) {
  Json out = Json::object();
  std::set<std::string> optional;                                  // `{default: null, optional: true}`: null is "not given", a value the nodes read
  for (const auto& [name, entry] : g.parameters.items()) {
    out[name] = default_of(entry);
    if (entry.is_object() && entry.contains("default") && entry.value("optional", false)) optional.insert(name);
  }
  if (!set.is_object()) refuse("parameters: a table of name to value");
  for (const auto& [name, v] : set.items()) {
    if (!out.contains(name)) {
      std::string valid;
      for (const auto& [n, _] : out.items()) valid += " " + n;
      refuse("unknown parameter '" + name + "'; the graph's parameters:" + (valid.empty() ? " (none)" : valid));
    }
    if (!compatible(out[name], v)) refuse("parameter '" + name + "' is " + kind_of(v) + "; its default is " + kind_of(out[name]));
    out[name] = v;
  }
  for (const auto& [name, v] : out.items())
    if (v.is_null() && !optional.count(name)) refuse("parameter '" + name + "' has no default and was not set");
  return out;
}

}  // namespace

Graph graph_from_json(const Json& doc) {
  if (!doc.is_object()) refuse("a graph is a table");
  for (const auto& [k, _] : doc.items())
    if (k != "name" && k != "version" && k != "inputs" && k != "outputs" && k != "parameters" && k != "before" && k != "chunks" && k != "after")
      refuse("graph: unknown key '" + k + "'");
  Graph g;
  if (doc.contains("name")) g.name = doc["name"].get<std::string>();
  if (doc.contains("version")) g.version = doc["version"].get<int>();
  if (doc.contains("inputs")) g.inputs = strings_of(doc["inputs"], "inputs");
  if (doc.contains("outputs")) g.outputs = strings_of(doc["outputs"], "outputs");
  if (doc.contains("parameters")) {
    if (!doc["parameters"].is_object()) refuse("parameters: a table of name to default");
    g.parameters = doc["parameters"];
  }
  g.before = uses_of(doc, "before");
  g.chunks = uses_of(doc, "chunks");
  g.after = uses_of(doc, "after");
  g.text = doc.dump();
  return g;
}

Checked check(const Graph& g, const Types& inputs, const Json& params, const Registry& reg, const DispatchContext& ctx) {
  Checked c;
  c.name = g.name;
  c.inputs = g.inputs;
  c.outputs = g.outputs;

  // 1. STRUCTURE: parameters resolved; every input typed; every name defined before it is used
  c.parameters = resolve_graph_parameters(g, params);
  std::set<std::string> seen;
  for (const auto& n : g.inputs) {
    if (!seen.insert(n).second) refuse("input '" + n + "' is named twice");
    const auto t = inputs.find(n);
    if (t == inputs.end()) refuse("input '" + n + "' has no type: the caller types every input of the graph");
    c.values[n] = t->second;
  }
  for (const auto& [n, _] : inputs)
    if (!seen.count(n)) refuse("a type was given for '" + n + "', which is not an input of the graph");

  struct Item { const Use* use; Section section; std::size_t position; };
  std::vector<Item> items;
  auto add_section = [&](const std::vector<Use>& uses, Section s) {
    for (std::size_t i = 0; i < uses.size(); ++i) items.push_back({&uses[i], s, i});
  };
  add_section(g.before, Section::before);
  add_section(g.chunks, Section::chunks);
  add_section(g.after, Section::after);

  std::map<std::string, std::string> made_by;                  // value -> the use that made it, for the shared checker's messages
  for (const Item& it : items) {
    const Use& u = *it.use;
    CheckedNode n;
    const std::string where = std::string(to_string(it.section)) + "[" + std::to_string(it.position) + "] " + u.node;
    n.label = where;
    auto fail = [&](const std::string& why) { refuse(where + ": " + why); };

    // 2. THE NODE: found in the registry (an unknown name is refused with the near names)
    try { n.spec = std::make_shared<const NodeSpec>(reg.at(u.node)); } catch (const std::invalid_argument& e) { fail(e.what()); }
    const NodeSpec& spec = *n.spec;
    n.section = it.section;

    // a node is complete for its kind
    if (!spec.plan && !spec.signature) fail("the node has no `plan` (its arrows cannot be checked)");
    if (spec.kind == Kind::array && !spec.run) fail("an array node needs `run`");
    if (spec.kind == Kind::chunk && spec.traits.time != TimeBehaviour::stateful && !spec.chunk) fail("a chunk node needs `chunk` (or `start` and `step`, when stateful)");
    if ((spec.kind == Kind::accumulate || (spec.kind == Kind::chunk && spec.traits.time == TimeBehaviour::stateful)) && (!spec.start || !spec.step)) fail("a stateful or accumulate node needs `start` and `step`");
    if (spec.kind == Kind::accumulate && !spec.finish) fail("an accumulate node needs `finish`");

    // 3. TRAITS
    try { check_traits(spec); } catch (const std::invalid_argument& e) { fail(e.what()); }
    if (ctx.block_parallel && spec.traits.time == TimeBehaviour::stateful)
      fail("a stateful node (" + spec.traits.note + ") cannot run on blocks in parallel or out of time order");

    // 4. KINDS BY SECTION
    if (it.section == Section::chunks && spec.kind == Kind::array) fail("an array node belongs in `before` or `after`, not in `chunks`");
    if (it.section != Section::chunks && spec.kind != Kind::array)
      fail(std::string("a ") + to_string(spec.kind) + " node belongs in `chunks`, not in `" + to_string(it.section) + "`");

    // ports: known, present, defined
    std::set<std::string> ports, many;
    for (const auto& p : spec.inputs) { ports.insert(p.name); if (p.many) many.insert(p.name); }
    for (const auto& [port, value] : u.in) {
      if (!ports.count(port) && !(many.count(port_of_key(port)) && port_of_key(port) != port)) {
        std::string valid;
        for (const auto& p : spec.inputs) valid += " " + p.name;
        fail("no input port '" + port + "'; its ports:" + valid);
      }
      if (!c.values.count(value)) fail("port '" + port + "' reads '" + value + "', which is neither an input of the graph nor made by an earlier use");
      n.in[port] = value;
      n.in_types[port] = c.values.at(value);
    }
    for (const auto& p : spec.inputs)
      if (!p.optional && !n.in.count(p.name) && !(p.many && n.in.count(p.name + ".0"))) fail("input port '" + p.name + "' is not wired" + (p.many ? std::string(" (a list: ") + p.name + ".0, " + p.name + ".1, ...)" : std::string()));

    // the streamed values: a `before` node takes none; an `after` node takes no block
    for (const auto& [port, t] : n.in_types) {
      if (it.section == Section::before && t.streamed()) fail("port '" + port + "' takes the streamed value '" + n.in.at(port) + "': a `before` node runs once, on arrays");
      if (it.section == Section::after && t.streamed()) fail("port '" + port + "' takes the block '" + n.in.at(port) + "': an `after` node runs once, on what the blocks reduced");
      if (it.section != Section::chunks && t.per_block()) fail("port '" + port + "' takes '" + n.in.at(port) + "', a value made once per block: only a node inside the loop reads it (an accumulate node reduces it)");
    }

    // parameters: the node's defaults, then `with` ($name from the graph's parameters)
    n.params = Json::object();
    for (const auto& p : spec.parameters) n.params[p.name] = p.def;
    if (!u.with.is_object()) fail("`with` maps parameters to values");
    for (const auto& [name, v0] : u.with.items()) {
      const ParamSpec* ps = nullptr;
      for (const auto& p : spec.parameters) if (p.name == name) ps = &p;
      if (!ps) {
        std::string valid;
        for (const auto& p : spec.parameters) valid += " " + p.name;
        fail("no parameter '" + name + "'; its parameters:" + (valid.empty() ? " (none)" : valid));
      }
      Json v = v0;
      if (v.is_string() && !v.get<std::string>().empty() && v.get<std::string>()[0] == '$') {
        const std::string ref = v.get<std::string>().substr(1);
        if (!c.parameters.contains(ref)) fail("parameter '" + name + "' reads $" + ref + ", which is not a parameter of the graph");
        v = c.parameters[ref];
      }
      if (!compatible(ps->def, v)) fail("parameter '" + name + "' is " + kind_of(v) + "; its default is " + kind_of(ps->def));
      if (!ps->choices.empty() && v.is_string() && std::find(ps->choices.begin(), ps->choices.end(), v.get<std::string>()) == ps->choices.end()) {
        std::string valid;
        for (const auto& ch : ps->choices) valid += " " + ch;
        fail("parameter '" + name + "' is '" + v.get<std::string>() + "'; the choices:" + valid);
      }
      n.params[name] = v;
    }
    for (const auto& [name, v] : n.params.items()) {
      bool optional = false;
      for (const auto& p : spec.parameters) if (p.name == name) optional = p.optional;
      if (v.is_null() && !optional) fail("parameter '" + name + "' has no default and is not set");
    }

    // D106: a linear map never follows a non-linear node
    bool any_nonlinear = false;
    for (const auto& [port, t] : n.in_types) {
      if (spec.linear_map && t.nonlinear) fail("port '" + port + "' was made by a non-linear node and this is a linear map (D106: a linear map reads the chain before any non-linear transform)");
      any_nonlinear = any_nonlinear || t.nonlinear;
    }

    // 5. THE ARROWS: the node's plan on its inputs' types
    try { if (spec.plan) n.planned = spec.plan(n.in_types, n.params); } catch (const std::invalid_argument& e) { fail(e.what()); }
    //    and the SHARED CHECKER beside it: the node's signature, on typed inputs (types.h `typed`; a use with an untyped input is left to its plan)
    bool typed_inputs = !n.in_types.empty();
    for (const auto& [_, t] : n.in_types) typed_inputs = typed_inputs && t.typed;
    if (spec.signature && typed_inputs) {
      TypeUse tu;
      tu.label = where;
      tu.chunk_node = spec.kind == Kind::chunk;
      tu.in = n.in_types;
      tu.in_names = n.in;
      tu.made_by = made_by;
      {                                                                 // the names given, in the order of the outputs the use makes (a node that makes a part of its outputs by a parameter: outputs_for)
        const std::vector<std::string> only = spec.outputs_for ? spec.outputs_for(n.params) : std::vector<std::string>();
        std::size_t k = 0;
        for (std::size_t i = 0; i < spec.signature->outputs.size() && k < u.out.size(); ++i) {
          const std::string& port = spec.signature->outputs[i].first;
          if (spec.outputs_for && std::find(only.begin(), only.end(), port) == only.end()) continue;
          tu.out_names[port] = u.out[k++];
        }
      }
      Types made_types;
      try { made_types = type_use(u.node, *spec.signature, tu, n.params); } catch (const TypeRefusal& e) { refuse(e.what()); }
      for (auto& [port, mt] : made_types) {
        auto planned = n.planned.out.find(port);
        if (planned == n.planned.out.end()) { n.planned.out[port] = mt; continue; }
        PortType& dst = planned->second;                 // the plan's type keeps what C1 reads; the typed members are the checker's
        dst.typed = true; dst.components = mt.components; dst.domain = mt.domain; dst.target = mt.target; dst.quantity = mt.quantity; dst.res = mt.res; dst.from_value = mt.from_value;
        if (dst.cls == PortClass::constant && mt.cls != PortClass::constant) dst.cls = mt.cls;   // the checker's class wins: a Value maps measurements, statistics and spans to the stored-constant class
        dst.spin = mt.spin; dst.gauge = mt.gauge; dst.sel = mt.sel; dst.of = mt.of; dst.transition = mt.transition; dst.complex_parts = mt.complex_parts;
      }
    }
    std::vector<std::string> used;                       // the ports this use makes: all, or the ones its parameters' part makes
    {
      const std::vector<std::string> only = spec.outputs_for ? spec.outputs_for(n.params) : std::vector<std::string>();
      for (const auto& o : spec.outputs) if (!spec.outputs_for || std::find(only.begin(), only.end(), o.name) != only.end()) used.push_back(o.name);
    }
    if (u.out.size() != used.size()) fail("the node makes " + std::to_string(used.size()) + " output(s); `out` names " + std::to_string(u.out.size()));
    for (std::size_t i = 0; i < used.size(); ++i) {
      const std::string& port = used[i];
      const auto t = n.planned.out.find(port);
      if (t == n.planned.out.end()) fail("its plan gives no type for output '" + port + "'");
      PortType ot = t->second;
      ot.nonlinear = ot.nonlinear || any_nonlinear;
      if (spec.kind == Kind::chunk && !ot.streamed() && !ot.per_block()) fail("output '" + port + "' of a chunk node must be streamed (axis samples or frames) or made once per block (on a line the node makes)");
      if (spec.kind != Kind::chunk && ot.per_block()) fail("output '" + port + "' of " + std::string(to_string(spec.kind)) + " node is an array; only a chunk node makes a value once per block");
      if (spec.kind != Kind::chunk && ot.streamed()) fail("output '" + port + "' of " + std::string(to_string(spec.kind)) + " node is an array, not a stream");
      const std::string& name = u.out[i];
      if (c.values.count(name)) fail("`out` names '" + name + "', which already exists");
      n.out[port] = name;
      c.values[name] = ot;
      made_by[name] = where;
    }
    if (spec.kind != Kind::chunk && spec.kind != Kind::accumulate && n.planned.halo != 0) fail("only a chunk node or an accumulate node has a halo");   // an accumulate node's is to the right only (a segment family's n - 1)
    if (n.planned.halo < 0) fail("a negative halo");
    if (n.planned.align < 0) fail("a negative span grid");
    c.align = std::max(c.align, n.planned.align);
    c.nodes.push_back(std::move(n));
  }

  // outputs: values with a type
  std::set<std::string> outs;
  for (const auto& o : g.outputs) {
    if (!c.values.count(o)) refuse("output '" + o + "' is not a value of the graph");
    if (!outs.insert(o).second) refuse("output '" + o + "' is named twice");
    if (c.values.at(o).per_block()) refuse("output '" + o + "' is made once per block: an accumulate node reduces it across blocks into an output");
  }

  // the loop: the chunks section
  for (std::size_t i = 0; i < c.nodes.size(); ++i) if (c.nodes[i].section == Section::chunks) c.loop.push_back(i);
  for (std::size_t li = 0; li < c.loop.size(); ++li) {
    const CheckedNode& n = c.nodes[c.loop[li]];
    if (n.uses_step()) c.has_state = true;
    if (n.spec->frame_place) {
      if (c.retime) refuse(n.label + ": a graph has one retiming node (the other is " + c.nodes[c.loop[*c.retime]].label + ")");
      c.retime = li;
      bool frames = false;
      for (const auto& [_, t] : n.planned.out) frames = frames || t.axis == Axis::frames;
      if (!frames) refuse(n.label + ": a retiming node's outputs carry axis `frames`");
    }
  }

  // 6. THE HALO, composed backwards over the chunk nodes
  std::map<std::string, std::vector<std::size_t>> readers;   // value -> the loop positions of the chunk nodes reading it
  for (std::size_t li = 0; li < c.loop.size(); ++li)
    for (const auto& [_, name] : c.nodes[c.loop[li]].in) if (c.nodes[c.loop[li]].spec->kind == Kind::chunk || c.nodes[c.loop[li]].planned.halo > 0) readers[name].push_back(li);
  for (std::size_t li = c.loop.size(); li-- > 0;) {
    CheckedNode& n = c.nodes[c.loop[li]];
    std::int64_t keep = 0;
    for (const auto& [_, name] : n.out)
      for (const std::size_t r : readers[name]) keep = std::max(keep, c.nodes[c.loop[r]].planned.halo + c.nodes[c.loop[r]].keep);
    n.keep = keep;
  }
  for (const std::size_t i : c.loop) {
    const CheckedNode& n = c.nodes[i];
    bool reads_input = false;
    for (const auto& [_, name] : n.in) reads_input = reads_input || (std::find(g.inputs.begin(), g.inputs.end(), name) != g.inputs.end() && c.values.at(name).streamed());
    if (reads_input) c.halo = std::max(c.halo, n.planned.halo + n.keep);
  }
  // where the pointwise tail starts: after the last chunk node that needs a halo
  for (std::size_t li = 0; li < c.loop.size(); ++li) if (c.nodes[c.loop[li]].planned.halo > 0) c.tail = li + 1;   // a chunk node's, or an accumulate node's right halo: they read the block with its halo
  // and after the last node that reads or makes a value of the block (a change of basis): those run on the whole block, never on sub-blocks of samples
  for (std::size_t li = 0; li < c.loop.size(); ++li) {
    const CheckedNode& n = c.nodes[c.loop[li]];
    bool per_block = false;
    for (const auto& [_, name] : n.in) per_block = per_block || c.values.at(name).per_block();
    for (const auto& [_, name] : n.out) per_block = per_block || c.values.at(name).per_block();
    if (per_block) c.tail = std::max(c.tail, li + 1);
  }

  // 7. RETIMING: the nodes after it run on frames, are pointwise, and read no stream of samples
  if (c.retime) {
    for (std::size_t li = *c.retime + 1; li < c.loop.size(); ++li)
      for (const auto& [_, name] : c.nodes[c.loop[li]].out)
        if (c.values.at(name).per_block()) refuse(c.nodes[c.loop[li]].label + ": makes a value once per block after the retiming node; the nodes after it are pointwise on its frames");
    if (c.tail > *c.retime + 1) refuse(c.nodes[c.loop[c.tail - 1]].label + ": a node after the retiming node (" + c.nodes[c.loop[*c.retime]].label + ") needs a halo; the nodes after it are pointwise on its frames");
    for (std::size_t li = *c.retime + 1; li < c.loop.size(); ++li)
      for (const auto& [port, name] : c.nodes[c.loop[li]].in)
        if (c.values.at(name).axis == Axis::samples) refuse(c.nodes[c.loop[li]].label + ": port '" + port + "' reads the stream of samples '" + name + "' after the retiming node; it runs on frames");
  }

  // a value is released after its last reader in the loop; a graph output is kept
  for (std::size_t li = 0; li < c.loop.size(); ++li)
    for (const auto& [_, name] : c.nodes[c.loop[li]].in) c.last_use[name] = li;
  for (const auto& o : g.outputs) c.last_use[o] = std::numeric_limits<std::size_t>::max();
  return c;
}

std::string Checked::text() const {
  std::ostringstream o;
  o << "graph " << (name.empty() ? "(unnamed)" : name) << ": inputs";
  for (const auto& i : inputs) o << " " << i << "(" << to_string(values.at(i).axis) << ")";
  o << "\n";
  for (const auto& n : nodes) {
    o << "  " << n.label << "  [" << to_string(n.spec->kind) << ", " << to_string(n.spec->traits.time) << "]";
    o << "  in";
    for (const auto& [p, v] : n.in) o << " " << p << "=" << v;
    o << "  out";
    for (const auto& [p, v] : n.out) o << " " << p << "=" << v << "(" << to_string(values.at(v).value) << "," << to_string(values.at(v).axis) << ")";
    if (!n.params.empty()) o << "  with " << n.params.dump();
    if (n.section == Section::chunks) o << "  halo " << n.planned.halo << " keep " << n.keep;
    o << "\n";
  }
  o << "  loop halo " << halo << " samples each side\n  outputs";
  for (const auto& v : outputs) o << " " << v;
  o << "\n";
  return o.str();
}

}  // namespace nxr::graph

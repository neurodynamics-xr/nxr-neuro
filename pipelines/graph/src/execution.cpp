// nxr-graph — THE EXECUTOR (execution.h): `before` once, each block through the chunk and accumulate nodes with the state carried across blocks, `after` once.
// The step logic is the runner's (pipelines/src/runner.cpp, `execute`), made generic over a checked graph and free of io; the runner is split onto it in spec step C5.
#include "nxr/graph/execution.h"

#include <algorithm>
#include <set>

namespace nxr::graph {

namespace {

// ── Array <-> Block ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
std::int64_t components_of(ValueType v) {
  switch (v) {
    case ValueType::vector3: return 3;
    case ValueType::quaternion: return 4;
    case ValueType::hermitian3: return 6;
    default: return 1;
  }
}
Block to_block(const Array& a, const PortType& t, const std::string& name) {
  Block b;
  const std::int64_t C = a.comps;
  if (static_cast<std::int64_t>(a.data.size()) != a.rows * a.cols * C) refuse("input '" + name + "': " + std::to_string(a.data.size()) + " values for [" + std::to_string(a.rows) + " x " + std::to_string(a.cols) + " x " + std::to_string(C) + "]");
  if (t.value == ValueType::complex) {
    if (C != 2) refuse("input '" + name + "' is complex: handed as [rows x samples x 2] (re, im)");
    b.rows = a.rows; b.width = a.cols;
    b.re.resize(static_cast<std::size_t>(a.rows * a.cols)); b.im.resize(b.re.size());
    for (std::size_t i = 0; i < b.re.size(); ++i) { b.re[i] = static_cast<float>(a.data[2 * i]); b.im[i] = static_cast<float>(a.data[2 * i + 1]); }
    return b;
  }
  if (C != components_of(t.value)) refuse("input '" + name + "' is " + to_string(t.value) + ": handed with " + std::to_string(components_of(t.value)) + " component(s) per element, not " + std::to_string(C));
  b.rows = a.rows * C; b.width = a.cols;
  b.re.resize(static_cast<std::size_t>(b.rows * b.width));
  for (std::int64_t r = 0; r < a.rows; ++r)
    for (std::int64_t k = 0; k < C; ++k)
      for (std::int64_t s = 0; s < a.cols; ++s) b.re[static_cast<std::size_t>((r * C + k) * a.cols + s)] = static_cast<float>(a.data[static_cast<std::size_t>((r * a.cols + s) * C + k)]);
  return b;
}
Array to_array(const Block& b, const PortType& t) {
  Array a;
  if (b.complex()) {
    a.rows = b.rows; a.cols = b.width; a.comps = 2;
    a.data.resize(static_cast<std::size_t>(b.rows * b.width * 2));
    for (std::size_t i = 0; i < b.re.size(); ++i) { a.data[2 * i] = b.re[i]; a.data[2 * i + 1] = b.im[i]; }
    return a;
  }
  const std::int64_t C = components_of(t.value);
  if (b.rows % C != 0) refuse("a block of " + std::to_string(b.rows) + " rows is not whole " + to_string(t.value) + " elements");
  a.rows = b.rows / C; a.cols = b.width; a.comps = C;
  a.data.resize(static_cast<std::size_t>(b.rows * b.width));
  for (std::int64_t r = 0; r < a.rows; ++r)
    for (std::int64_t k = 0; k < C; ++k)
      for (std::int64_t s = 0; s < a.cols; ++s) a.data[static_cast<std::size_t>((r * a.cols + s) * C + k)] = b.re[static_cast<std::size_t>((r * C + k) * a.cols + s)];
  return a;
}

// a block cut to the columns [o, o + w) of its OWN columns, no halo
Block cut_block(const Block& b, std::int64_t o, std::int64_t w) {
  Block c{b.rows, w, 0, 0, std::vector<float>(static_cast<std::size_t>(b.rows * w)), {}};
  if (b.complex()) c.im.resize(c.re.size());
  for (std::int64_t r = 0; r < b.rows; ++r) {
    const std::size_t from = static_cast<std::size_t>(r * b.width + b.left + o), to = static_cast<std::size_t>(r * w);
    std::copy_n(b.re.begin() + from, w, c.re.begin() + to);
    if (b.complex()) std::copy_n(b.im.begin() + from, w, c.im.begin() + to);
  }
  return c;
}
// the own columns of each piece, joined in order
Block join(const std::vector<Block>& pieces) {
  Block out;
  if (pieces.empty()) return out;
  std::int64_t w = 0;
  for (const auto& p : pieces) w += p.own();
  out.rows = pieces.front().rows; out.width = w;
  out.re.resize(static_cast<std::size_t>(out.rows * w));
  if (pieces.front().complex()) out.im.resize(out.re.size());
  std::int64_t at = 0;
  for (const auto& p : pieces) {
    const std::int64_t n = p.own();
    for (std::int64_t r = 0; r < out.rows; ++r) {
      std::copy_n(p.re.begin() + (r * p.width + p.left), n, out.re.begin() + (r * w + at));
      if (p.complex()) std::copy_n(p.im.begin() + (r * p.width + p.left), n, out.im.begin() + (r * w + at));
    }
    at += n;
  }
  return out;
}

}  // namespace

Execution::Execution(const Checked& checked, const Arrays& constants, const DispatchContext& ctx, const Blocking& blocking)
    : checked_(checked), ctx_(ctx), blocking_(blocking), prepared_(checked.nodes.size()), states_(checked.nodes.size()) {
  for (const auto& name : checked.inputs) {
    const bool streamed = checked.values.at(name).streamed();
    if (streamed && constants.count(name)) refuse("input '" + name + "' is streamed: it comes with each block, not as a constant");
    if (!streamed && !constants.count(name)) refuse("input '" + name + "' is not streamed: the constants must hold it");
  }
  for (const auto& [name, a] : constants)
    if (!checked.values.count(name)) refuse("a constant was given for '" + name + "', which is not an input of the graph");
  data_ = constants;
  for (std::size_t i = 0; i < checked.nodes.size(); ++i) {
    const CheckedNode& n = checked.nodes[i];
    if (n.section == Section::before) { run_arrays(i); continue; }
    if (n.section == Section::after) continue;                       // prepared when its inputs exist (finish)
    Arrays consts;
    for (const auto& [port, name] : n.in) if (!n.in_types.at(port).streamed() && n.in_types.at(port).axis != Axis::block) consts[port] = data_.at(name);   // a per-block value comes with each block
    NodeCall p = NodeCall::prepare(n.in_types, consts, n.params, shared_);
    dispatch(*n.spec, p, ctx_);
    prepared_[i] = std::move(p.made);
    if (n.uses_step()) { NodeCall s = NodeCall::start(prepared_[i]); dispatch(*n.spec, s, ctx_); states_[i] = std::move(s.made); }
  }
}

// an array node (`before` or `after`): prepared on its inputs, run, its outputs into the data
void Execution::run_arrays(std::size_t i) {
  const CheckedNode& n = checked_.nodes[i];
  Arrays in;
  for (const auto& [port, name] : n.in) {
    const auto a = data_.find(name);
    if (a == data_.end()) refuse(n.label + ": port '" + port + "' reads '" + name + "', which no node made");
    in[port] = a->second;
  }
  NodeCall p = NodeCall::prepare(n.in_types, in, n.params, shared_);
  dispatch(*n.spec, p, ctx_);
  prepared_[i] = std::move(p.made);
  NodeCall r = NodeCall::run(prepared_[i], in);
  dispatch(*n.spec, r, ctx_);
  for (auto& [port, a] : r.made_arrays) {
    const auto o = n.out.find(port);
    if (o == n.out.end()) refuse(n.label + ": made '" + port + "', which is not one of its outputs");
    data_[o->second] = std::move(a);
  }
  for (const auto& [port, name] : n.out) if (!data_.count(name)) refuse(n.label + ": made no '" + port + "'");
}

Blocks Execution::block(const Blocks& in, const Place& at) {
  const Checked& c = checked_;
  if (finished_) refuse("a block was handed after finish");
  for (const auto& name : c.inputs) {
    if (!c.values.at(name).streamed()) continue;
    const auto b = in.find(name);
    if (b == in.end()) refuse("block: the streamed input '" + name + "' is missing");
    if (b->second.own() != at.n) refuse("block: input '" + name + "' has " + std::to_string(b->second.own()) + " own columns in a block of " + std::to_string(at.n));
    if (b->second.left < 0 || b->second.right < 0) refuse("block: input '" + name + "' has a negative halo");
  }
  for (const auto& [name, b] : in) if (!c.values.count(name) || !c.values.at(name).streamed() || std::find(c.inputs.begin(), c.inputs.end(), name) == c.inputs.end()) refuse("block: '" + name + "' is not a streamed input of the graph");
  if (c.has_state) {
    if (next_ && at.first != *next_) refuse("block: the graph carries state (a stateful or accumulate node): blocks must meet in time order; expected the block at " + std::to_string(*next_) + ", got " + std::to_string(at.first));
    next_ = at.first + at.n;
  }

  Blocks set = in;
  std::map<std::string, std::vector<Block>> pieces;       // the streamed outputs' own columns, in order
  std::set<std::string> harvested;
  auto harvest = [&](const Blocks& s, bool sub) {
    for (const auto& name : c.outputs) {
      if (!c.values.at(name).streamed() || !s.count(name) || (sub && harvested.count(name))) continue;
      pieces[name].push_back(s.at(name));
      if (!sub) harvested.insert(name);
    }
  };
  // a node run on a set of blocks over `place`: its outputs join the set; a value is released after its last reader
  auto run_step = [&](std::size_t li, Blocks& s, const Place& place) {
    const std::size_t idx = c.loop[li];
    const CheckedNode& n = c.nodes[idx];
    Blocks ins;
    for (const auto& [port, name] : n.in) if (s.count(name)) ins[port] = s.at(name);
    for (const auto& [port, b] : ins)
      if (c.values.at(n.in.at(port)).streamed() && b.own() != place.n) refuse(n.label + ": " + port + " has " + std::to_string(b.own()) + " own columns in a block of " + std::to_string(place.n) + ": a node's halo bookkeeping is off");
    Blocks made;
    if (n.uses_step()) {
      NodeCall call = NodeCall::step(prepared_[idx], std::move(states_[idx]), ins, n.keep, place);
      dispatch(*n.spec, call, ctx_);
      states_[idx] = std::move(call.made_state);
      made = std::move(call.made_blocks);
    } else {
      NodeCall call = NodeCall::chunk(prepared_[idx], ins, n.keep, place);
      dispatch(*n.spec, call, ctx_);
      made = std::move(call.made_blocks);
    }
    for (const auto& [port, name] : n.out) {
      auto b = made.find(port);
      if (b == made.end()) {
        if (n.spec->kind == Kind::accumulate) continue;
        refuse(n.label + ": made no block for output '" + port + "'");
      }
      s[name] = std::move(b->second);
    }
    for (const auto& [port, name] : n.in) {
      const auto u = c.last_use.find(name);
      if (u != c.last_use.end() && u->second == li) s.erase(name);
    }
  };
  // the sub-block of the values on `axis`, own columns [o, o + w)
  auto cut = [&](const Blocks& s, std::int64_t o, std::int64_t w, Axis axis) {
    Blocks sub;
    for (const auto& [name, b] : s) if (c.values.at(name).axis == axis) sub[name] = cut_block(b, o, w);
    return sub;
  };

  const std::size_t whole = c.retime ? *c.retime + 1 : c.tail;
  for (std::size_t li = 0; li < whole; ++li) run_step(li, set, at);
  harvest(set, false);
  if (c.retime) {
    // the steps after the retiming node run on BLOCKS OF FRAMES of this block's frames, in frame order
    NodeCall fp_call = NodeCall::frame_place(prepared_[c.loop[*c.retime]], at);
    dispatch(*c.nodes[c.loop[*c.retime]].spec, fp_call, ctx_);
    const FramePlace& fp = fp_call.made_frames;
    const std::int64_t B = blocking_.frames > 0 ? blocking_.frames : fp.n;
    for (std::int64_t o = 0; *c.retime + 1 < c.loop.size() && o < fp.n; o += B) {
      const std::int64_t w = std::min(B, fp.n - o);
      Blocks sub = cut(set, o, w, Axis::frames);
      std::vector<std::uint8_t> g;
      if (!fp.good.empty()) g.assign(fp.good.begin() + o, fp.good.begin() + o + w);
      const Place place{fp.first + o, w, &g};
      for (std::size_t li = *c.retime + 1; li < c.loop.size(); ++li) run_step(li, sub, place);
      harvest(sub, true);
    }
  } else {
    // THE POINTWISE TAIL on sub-blocks of the own columns, in time order: the same values, a block's worth of memory
    const std::int64_t B = blocking_.samples > 0 ? blocking_.samples : at.n;
    for (std::int64_t o = 0; whole < c.loop.size() && o < at.n; o += B) {
      const std::int64_t w = std::min(B, at.n - o);
      Blocks sub = cut(set, o, w, Axis::samples);
      std::vector<std::uint8_t> g;
      if (at.good && !at.good->empty()) g.assign(at.good->begin() + o, at.good->begin() + o + w);
      const Place place{at.first + o, w, &g};
      for (std::size_t li = whole; li < c.loop.size(); ++li) run_step(li, sub, place);
      harvest(sub, true);
    }
  }
  Blocks out;
  for (auto& [name, ps] : pieces) out[name] = join(ps);
  return out;
}

Arrays Execution::finish() {
  if (finished_) refuse("finish was called twice");
  finished_ = true;
  for (const std::size_t idx : checked_.loop) {
    const CheckedNode& n = checked_.nodes[idx];
    if (n.spec->kind != Kind::accumulate) continue;
    NodeCall call = NodeCall::finish(prepared_[idx], std::move(states_[idx]));
    dispatch(*n.spec, call, ctx_);
    for (auto& [port, a] : call.made_arrays) {
      const auto o = n.out.find(port);
      if (o == n.out.end()) refuse(n.label + ": finished '" + port + "', which is not one of its outputs");
      data_[o->second] = std::move(a);
    }
    // an accumulate node may leave an output unmade when it has nothing to report (a run that found no event): what reads it, or the graph's outputs, refuse then
  }
  for (std::size_t i = 0; i < checked_.nodes.size(); ++i) if (checked_.nodes[i].section == Section::after) run_arrays(i);
  Arrays out;
  for (const auto& name : checked_.outputs) {
    if (checked_.values.at(name).streamed()) continue;
    const auto a = data_.find(name);
    if (a == data_.end()) refuse("output '" + name + "' was not made");
    out[name] = a->second;
  }
  return out;
}

Json Execution::record() const {
  Json r = Json::array();
  for (std::size_t i = 0; i < checked_.nodes.size(); ++i) {
    const CheckedNode& n = checked_.nodes[i];
    NodeCall call = NodeCall::record(prepared_[i]);
    dispatch(*n.spec, call, ctx_);
    r.push_back(Json{{"use", n.label}, {"node", n.spec->name}, {"parameters", n.params}, {"resolved", call.made_json}});
  }
  return r;
}

Arrays apply(const Checked& checked, const Arrays& inputs, const DispatchContext& ctx, const Blocking& blocking) {
  Arrays constants;
  Blocks blocks;
  std::int64_t n = -1;
  for (const auto& name : checked.inputs) {
    const auto a = inputs.find(name);
    if (a == inputs.end()) refuse("apply: input '" + name + "' was not handed");
    const PortType& t = checked.values.at(name);
    if (!t.streamed()) { constants[name] = a->second; continue; }
    Block b = to_block(a->second, t, name);
    if (n >= 0 && b.width != n) refuse("apply: the streamed inputs differ in length (" + std::to_string(n) + " and " + std::to_string(b.width) + ")");
    n = b.width;
    blocks[name] = std::move(b);
  }
  for (const auto& [name, _] : inputs) if (!checked.values.count(name) || std::find(checked.inputs.begin(), checked.inputs.end(), name) == checked.inputs.end()) refuse("apply: '" + name + "' is not an input of the graph");
  Execution ex(checked, constants, ctx, blocking);
  Blocks outb = ex.block(blocks, Place{0, n < 0 ? 0 : n, nullptr});
  Arrays out = ex.finish();
  for (auto& [name, b] : outb) out[name] = to_array(b, checked.values.at(name));
  return out;
}

}  // namespace nxr::graph

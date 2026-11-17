// nxr-neuro compute: a registered node applied to arrays in .npy files, through the same checker as `run` (docs/specs/2026-11-17-compute-and-run.md 6.1; type system T4).
//   nxr-neuro compute NODE IN... OUT... [--<optional port> FILE] [--<parameter> VALUE] [--sfreq HZ] [--on PORT=LABEL,LABEL] [--from PORT=LABEL] [--to PORT=LABEL]
//                      [--value PORT=TYPE] [--class PORT=field|operator|partition|constant] [--cs LABEL=SPACE/NAME/UNIT] [--gauge PORT=ID] [--dry-run] [--replace]
// The required ports are positional, inputs then outputs, in the node's declared order. The types are made from the arrays and these labels (anonymous manifolds; arrays that share a label
// lie on one manifold); `--dry-run` prints the checked types and runs nothing. Every output gets a sidecar `<output>.json` with its printed type.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#include "nxr/graph/compute.h"
#include "nxr/io/errors.h"
#include "nxr/io/npy.h"

namespace {
namespace ng = nxr::graph;
using Json = ng::Json;

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> o; std::string cur;
  for (char ch : s) { if (ch == c) { o.push_back(cur); cur.clear(); } else cur += ch; }
  o.push_back(cur);
  return o;
}
std::pair<std::string, std::string> key_value(const std::string& s, const char* flag) {
  const auto eq = s.find('=');
  if (eq == std::string::npos) throw std::invalid_argument(std::string("nxr-neuro compute: ") + flag + " takes PORT=VALUE, not '" + s + "'");
  return {s.substr(0, eq), s.substr(eq + 1)};
}

ng::Array read_npy(const std::string& path) {
  const std::string file = path.substr(0, path.find('#'));
  if (std::filesystem::path(file).extension() != ".npy") throw std::invalid_argument("nxr-neuro compute: '" + file + "': only .npy files are read so far (.mat, GIFTI, NIfTI and recordings come with their readers)");
  const auto f = nxr::io::NpyFile::open(file);
  ng::Array a;
  const auto& s = f.shape();
  if (s.empty() || s.size() > 3) throw std::invalid_argument("nxr-neuro compute: '" + file + "' must be an array of 1 to 3 axes [elements x samples x components]");
  a.rows = s[0]; a.cols = s.size() > 1 ? s[1] : 1; a.comps = s.size() > 2 ? s[2] : 1;
  a.data = f.all<double>();
  return a;
}
void write_npy(const std::string& path, const ng::Array& a) {
  std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(a.rows) + ", " + std::to_string(a.cols) + (a.comps > 1 ? ", " + std::to_string(a.comps) : std::string()) + "), }";
  while ((10 + header.size() + 1) % 64) header += ' ';
  header += '\n';
  std::ofstream o(path, std::ios::binary);
  o.write("\x93NUMPY\x01\x00", 8);
  const std::uint16_t n = static_cast<std::uint16_t>(header.size());
  o.write(reinterpret_cast<const char*>(&n), 2);   // little-endian hosts
  o << header;
  o.write(reinterpret_cast<const char*>(a.data.data()), static_cast<std::streamsize>(a.data.size() * sizeof(double)));
}
}  // namespace

int compute_main(const std::vector<std::string>& args) {
  try {
    if (args.empty() || args[0] == "-h" || args[0] == "--help") {
      std::cout << "usage: nxr-neuro compute NODE IN... OUT... [--<port> FILE] [--<parameter> VALUE] [--sfreq HZ] [--on PORT=L1,L2] [--from PORT=L] [--to PORT=L]\n"
                   "                         [--value PORT=TYPE] [--class PORT=CLASS] [--cs LABEL=SPACE/NAME/UNIT] [--gauge PORT=ID] [--dry-run] [--replace]\n"
                   "A registered node on .npy arrays, checked on their types before any data is read (nxr-neuro list nodes). `compute NODE --help` lists its ports and parameters.\n";
      return 0;
    }
    const ng::Registry& reg = ng::registry();
    const ng::NodeSpec& node = reg.at(args[0]);   // an unknown name is refused with the near names
    std::vector<std::string> positional;
    std::map<std::string, std::string> files, params;
    std::map<std::string, ng::ArrayMeta> metas;
    std::optional<double> sfreq;
    std::map<std::string, ng::CoordinateSystem> systems;
    bool dry = false, replace = false, help = false;
    auto is_port = [&](const std::string& n) { for (const auto& p : node.inputs) if (p.name == n) return true; return false; };
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& a = args[i];
      if (a.rfind("--", 0) != 0) { positional.push_back(a); continue; }
      const std::string name = a.substr(2);
      auto value = [&]() -> std::string { if (i + 1 >= args.size()) throw std::invalid_argument("nxr-neuro compute: " + a + " needs a value"); return args[++i]; };
      if (name == "dry-run") dry = true;
      else if (name == "replace") replace = true;
      else if (name == "help") help = true;
      else if (name == "sfreq") sfreq = std::stod(value());
      else if (name == "on") { auto [p, v] = key_value(value(), "--on"); metas[p].on = split(v, ','); }
      else if (name == "from") { auto [p, v] = key_value(value(), "--from"); metas[p].from = v; }
      else if (name == "to") { auto [p, v] = key_value(value(), "--to"); metas[p].to = v; }
      else if (name == "value") { auto [p, v] = key_value(value(), "--value"); metas[p].value = v; }
      else if (name == "class") { auto [p, v] = key_value(value(), "--class"); metas[p].cls = v; }
      else if (name == "gauge") { auto [p, v] = key_value(value(), "--gauge"); metas[p].gauge = v; metas[p].spin = metas[p].spin ? metas[p].spin : 1; }
      else if (name == "cs") {
        auto [l, v] = key_value(value(), "--cs");
        const auto parts = split(v, '/');
        if (parts.size() != 3) throw std::invalid_argument("nxr-neuro compute: --cs takes LABEL=SPACE/NAME/UNIT");
        systems[l] = ng::CoordinateSystem{parts[0], parts[1], parts[2]};
      }
      else if (is_port(name)) files[name] = value();
      else {
        std::string key = name;
        for (auto& c : key) if (c == '-') c = '_';
        params[key] = value();
      }
    }
    if (help) {
      std::cout << node.name << ": " << node.description << "\n  inputs:";
      for (const auto& p : node.inputs) std::cout << " " << p.name << (p.optional ? "?" : "");
      std::cout << "\n  outputs:";
      for (const auto& p : node.outputs) std::cout << " " << p.name;
      std::cout << "\n  parameters:\n";
      for (const auto& p : node.parameters) std::cout << "    --" << p.name << "  " << p.meaning << " (default " << p.def.dump() << (p.unit.empty() ? "" : " " + p.unit) << ")\n";
      return 0;
    }
    // positional: the required inputs, then the outputs, in the declared order
    std::size_t k = 0;
    for (const auto& p : node.inputs) if (!p.optional) { if (k >= positional.size()) throw std::invalid_argument("nxr-neuro compute " + node.name + ": the input '" + p.name + "' is missing"); files[p.name] = positional[k++]; }
    std::map<std::string, std::string> outs;
    for (const auto& p : node.outputs) { if (k < positional.size()) outs[p.name] = positional[k++]; }
    if (k < positional.size()) throw std::invalid_argument("nxr-neuro compute " + node.name + ": too many arguments ('" + positional[k] + "')");
    if (!dry && outs.size() != node.outputs.size()) throw std::invalid_argument("nxr-neuro compute " + node.name + ": name a file for each output (" + std::to_string(node.outputs.size()) + ")");
    Json with = Json::object();
    for (const auto& [name, v] : params) { try { with[name] = Json::parse(v); } catch (...) { with[name] = v; } }
    std::map<std::string, ng::ComputeInput> inputs;
    for (const auto& [port, path] : files) {
      ng::ComputeInput in;
      in.array = read_npy(path);
      in.meta = metas[port];
      in.meta.coordinate_systems = systems;
      if (!node.inputs.empty() && sfreq && in.meta.cls == "field") in.meta.sfreq = sfreq;
      inputs[port] = std::move(in);
    }
    std::vector<std::string> ports;
    for (const auto& [p, _] : inputs) ports.push_back(p);
    const ng::Graph g = ng::graph_of_node(node, ports, with);
    const ng::Checked checked = ng::check_arrays(g, inputs, Json::object(), reg);
    if (dry) { std::cout << checked.text(); return 0; }
    for (const auto& [port, path] : outs)
      if (!replace && std::filesystem::exists(path)) throw std::invalid_argument("nxr-neuro compute: '" + path + "' exists (--replace)");
    const ng::ComputeResult r = ng::compute(g, inputs, Json::object(), reg);
    for (const auto& [port, path] : outs) {
      write_npy(path, r.arrays.at(port));
      Json side = r.sidecar;
      side["output"] = port;
      side["node"] = node.name;
      side["parameters"] = with;
      std::ofstream(path + ".json") << side.dump(2) << "\n";
    }
    return 0;
  } catch (const std::invalid_argument& e) {
    std::cerr << "nxr-neuro compute: error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "nxr-neuro compute: error: " << e.what() << "\n";
    return 1;
  }
}

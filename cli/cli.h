// nxr-neuro's command line, shared by its commands: argparse-like arguments, JSON-lines output, the producer stamp.
#pragma once
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "py_json.h"

#ifndef NXR_NEURO_VERSION
#define NXR_NEURO_VERSION "0.0.0"
#endif

namespace cli {

using nxr::io::Json;

inline void emit(const Json& obj) { std::cout << pyjson::dumps(obj) << std::endl; }

inline std::string stamp() { return std::string("nxr-neuro ") + NXR_NEURO_VERSION + " / C++17"; }

struct Usage : std::runtime_error { using std::runtime_error::runtime_error; };

// argparse-like: options with a value, options taking several (nargs="+"), boolean flags, positionals
struct Args {
  std::map<std::string, std::string> opt;
  std::map<std::string, std::vector<std::string>> multi;
  std::set<std::string> flag;
  std::vector<std::string> pos;
  std::string get(const std::string& k) const { return opt.at(k); }
  std::optional<std::string> maybe(const std::string& k) const { auto it = opt.find(k); return it == opt.end() ? std::nullopt : std::optional<std::string>(it->second); }
};

inline Args parse(const std::string& prog, const std::vector<std::string>& argv, const std::set<std::string>& valued, const std::set<std::string>& flags,
                  const std::set<std::string>& required, std::size_t npos = 0, const std::set<std::string>& multi = {}, const std::string& pos_name = "FILE") {
  Args a;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    const std::string& t = argv[i];
    if (t.rfind("--", 0) == 0) {
      if (valued.count(t)) {
        if (i + 1 >= argv.size()) throw Usage(prog + ": error: argument " + t + ": expected one argument");
        a.opt[t] = argv[++i];
      } else if (multi.count(t)) {
        if (i + 1 >= argv.size() || argv[i + 1].rfind("--", 0) == 0) throw Usage(prog + ": error: argument " + t + ": expected at least one argument");
        while (i + 1 < argv.size() && argv[i + 1].rfind("--", 0) != 0) a.multi[t].push_back(argv[++i]);
      } else if (flags.count(t)) {
        a.flag.insert(t);
      } else {
        throw Usage(prog + ": error: unrecognized arguments: " + t);
      }
    } else {
      a.pos.push_back(t);
    }
  }
  for (const auto& r : required) if (!a.opt.count(r) && !a.multi.count(r)) throw Usage(prog + ": error: the following arguments are required: " + r);
  if (a.pos.size() != npos) throw Usage(prog + ": error: " + (a.pos.size() < npos ? "the following arguments are required: " + pos_name : "unrecognized arguments: " + a.pos[npos]));
  return a;
}

}  // namespace cli

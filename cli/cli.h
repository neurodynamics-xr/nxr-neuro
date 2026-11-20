// nxr-neuro's command line, shared by its commands: argparse-like arguments, JSON-lines output, the producer stamp.
#pragma once
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/io/catalogue.h"
#include "nxr/io/json_text.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/writer.h"
#include "nxr/io/py_json.h"

#ifndef NXR_NEURO_VERSION
#define NXR_NEURO_VERSION "0.0.0"
#endif

namespace cli {

using nxr::io::Json;

// every line a command prints goes through emit: JSON lines on stdout (today's format, unchanged) or, in a terminal, the readable form (cli/output.cpp)
void emit(const Json& obj);

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

// the option-table check (developers and tests): `nxr-neuro __options <old words>` runs a command up to its first parse and reports the options the parser knows,
// so the help tables (cli/commands.cpp) are compared with the parsers and cannot drift
struct ProbeDone : std::exception {};
struct Probe {
  bool on = false;
  std::set<std::string> valued, flags, multi, required;
  std::size_t npos = 0;
};
inline Probe& probe() { static Probe p; return p; }

inline Args parse(const std::string& prog, const std::vector<std::string>& argv, const std::set<std::string>& valued, const std::set<std::string>& flags,
                  const std::set<std::string>& required, std::size_t npos = 0, const std::set<std::string>& multi_in = {}, const std::string& pos_name = "FILE") {
  // Every command that names a subject of a dataset (`--dataset` and `--subject`, both required) also takes `--templates DIR ...`: where to look for the released templates a
  // subject package is registered to (the template search path; the environment variable NXR_TEMPLATE_PATH and the user setting follow it). A command that declares
  // `--templates` itself keeps its own meaning of it.
  std::set<std::string> multi = multi_in;
  if (required.count("--dataset") && required.count("--subject") && !valued.count("--templates") && !multi.count("--templates")) multi.insert("--templates");
  if (probe().on) {
    Probe& p = probe();
    p.valued = valued; p.flags = flags; p.multi = multi; p.required = required; p.npos = npos;
    throw ProbeDone();
  }
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

namespace cli {

// The template search path a command names (`--templates`, repeatable); the environment and the user setting are added by io.
inline nxr::io::TemplateSearch template_search_of(const Args& a) {
  nxr::io::TemplateSearch s;
  if (const auto it = a.multi.find("--templates"); it != a.multi.end()) for (const auto& p : it->second) s.command_paths.push_back(p);
  return s;
}

// THE DATASET A COMPUTATION RUNS ON. A dataset is a catalogue of subject packages (`create dataset`): the subject named by `--subject` is found through the catalogue
// and its package is opened (io's `open_subject_package`), so everything the command writes goes into that package. A one-database dataset (`dataset.sqlite`, store
// formats 50-52) still opens as it always did; `split dataset` makes it a catalogue.
inline nxr::io::DatasetWriter open_dataset(const Args& a, const nxr::io::OpenOptions& o = {}) {
  const std::string folder = a.get("--dataset");
  if (nxr::io::is_catalogue_dataset(folder)) {
    const auto subject = a.maybe("--subject");
    if (!subject) throw Usage("error: " + folder + " is a catalogue of subject packages: name the subject to work on with --subject");
    return nxr::io::open_subject_package(folder, *subject, o, template_search_of(a));
  }
  static bool told = false;
  if (!told && std::filesystem::is_regular_file(std::filesystem::path(folder) / "dataset.sqlite")) {
    told = true;
    std::cerr << "note: " << folder << " is a one-database dataset; `nxr-neuro split dataset` makes it a catalogue of subject packages (the default form).\n";
  }
  return nxr::io::open_writer(folder, o);
}

// The commands that make or change a one-database dataset itself refuse a catalogue, naming what to use.
inline nxr::io::DatasetWriter open_one_database(const Args& a, const std::string& what, const nxr::io::OpenOptions& o = {}) {
  if (nxr::io::is_catalogue_dataset(a.get("--dataset")))
    throw Usage("error: " + what + " works on a one-database dataset; " + a.get("--dataset") + " is a catalogue of subject packages (`list subjects`, `show subject` and the `compute` verbs open its packages)");
  return nxr::io::open_writer(a.get("--dataset"), o);
}

// ---- a command: parse, ONE library call (nxr/io/task/commands.h, nxr/pipelines/face.h), print ----

// a command's progress lines: a note on stderr (`note: TEXT`), every other line as the command's JSON line
inline void progress_line(const Json& j) {
  if (j.is_object() && j.contains("stage") && j["stage"] == "note") std::cerr << "note: " << j["note"].get<std::string>() << "\n";
  else emit(j);
}
inline nxr::io::task::Progress progress() { return progress_line; }

// the library call of a command: a request it refuses as given is the command line's usage error (exit 2), worded as before
template <class F>
auto call(const std::string& prog, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const nxr::io::task::Refused& e) {
    throw Usage((e.bare ? std::string() : prog + ": ") + "error: " + e.what());
  }
}

// the result line: `"stage": "result"` first, then the command's result object
inline void emit_result(const Json& result) {
  Json out = Json{{"stage", "result"}};
  for (auto it = result.begin(); it != result.end(); ++it) out[it.key()] = it.value();
  emit(out);
}

// a repeatable option's values (none: empty)
inline std::vector<std::string> values_of(const Args& a, const std::string& k) {
  const auto it = a.multi.find(k);
  return it == a.multi.end() ? std::vector<std::string>{} : it->second;
}
inline std::optional<std::vector<std::string>> maybe_values(const Args& a, const std::string& k) {
  const auto it = a.multi.find(k);
  return it == a.multi.end() ? std::nullopt : std::optional<std::vector<std::string>>(it->second);
}
// a comma-separated list, empty items dropped
inline std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::string x;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == ',') { if (!x.empty()) out.push_back(x); x.clear(); }
    else x += s[i];
  }
  return out;
}

}  // namespace cli

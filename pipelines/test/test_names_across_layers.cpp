// ONE NAME PER THING ACROSS THE LAYERS (maintainers' rule, 2026-11-17): the C++ function (nxr::<toolbox>::<word>), the registered node (`<toolbox>.<word>`) and every binding
// entry (the MEX commands and the MATLAB package, the Python module's native functions, the Node addon's exports, the wasm class's functions) use the SAME word.
//
// pipelines/test/names.tsv lists, one row per entry: kind, entry, the C++ function it calls (nxr::<toolbox>::<word>, or - for a composite or a method of the Manifold), status.
//   status `ok`        the entry's last word is the function's, its toolbox prefix is the function's toolbox, and the function is declared in that toolbox's headers (checked here);
//   status `composite` a node that runs several functions: its prefix is a toolbox, its word names its job (no single function to match);
//   status `pending <batch>: <why>` a known mismatch, with the batch that closes it (the behaviour batch: the function is not yet named as decided, e.g. Welch -> psd);
//   status `legacy`    a binding command of the Manifold context (create, solve, heat, ...), not a compute function by itself;
//   kind `function` a compute function that is not (yet) a node or a binding: entry `<toolbox>.<word>`, function nxr::<toolbox>::<word> (status `ok`: declared in the toolbox's headers, same word);
//   kind `retired`     an old name and the new one: the layer refuses the old one with the new in its message (the node registries are called here; the MEX table is read from its source).
// The table is checked both ways: every node of the registries and every binding string found in the sources is in it, and every row is found.
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/graph/node.h"
#include "nxr/pipelines/registry.h"

namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;

namespace {
struct Row { std::string kind, entry, function, status; };

std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
std::vector<Row> read_table(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    Row r;
    std::stringstream ss(line);
    std::getline(ss, r.kind, '\t'); std::getline(ss, r.entry, '\t'); std::getline(ss, r.function, '\t'); std::getline(ss, r.status, '\t');
    rows.push_back(r);
  }
  return rows;
}
const std::set<std::string>& toolboxes() {
  static const std::set<std::string> s = {"core", "quantity", "optimisation", "stats", "geometry", "operators", "selection", "measurement", "select", "signal", "synthesis", "meeg"};
  return s;
}
std::string last_word(const std::string& s) {
  const auto d = s.find_last_of(".:");
  return d == std::string::npos ? s : s.substr(d + 1);
}
// the toolbox of nxr::<toolbox>::...::<word>
std::string toolbox_of(const std::string& fn) {
  static const std::regex rx("^nxr::([a-z]+)::");
  std::smatch m;
  return std::regex_search(fn, m, rx) ? m[1].str() : "";
}
// the text of every header of a toolbox (compute and meeg)
std::string headers_of(const std::string& root, const std::string& tb) {
  std::string all;
  for (const std::string& dir : {root + "/compute/include/nxr/" + tb, root + "/meeg/include/nxr/" + tb}) {
    if (!fs::exists(dir)) continue;
    for (const auto& e : fs::recursive_directory_iterator(dir)) if (e.path().extension() == ".h") all += slurp(e.path().string());
  }
  return all;
}
bool declared(const std::string& root, const std::string& tb, const std::string& word) {
  static std::map<std::string, std::string> cache;
  if (!cache.count(tb)) cache[tb] = headers_of(root, tb);
  return std::regex_search(cache[tb], std::regex("\\b" + word + "\\s*\\("));
}
// every match of a regex's first group in a source text
std::set<std::string> found(const std::string& text, const std::string& rx) {
  std::set<std::string> out;
  const std::regex re(rx);
  for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) out.insert((*it)[1].str());
  return out;
}
bool refused(const std::function<void()>& f, const std::string& part) {
  try { f(); } catch (const std::exception& e) { return std::string(e.what()).find(part) != std::string::npos; }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string root = NXR_REPO_ROOT;
  const std::string table = argc > 1 ? argv[1] : root + "/pipelines/test/names.tsv";   // a table given on the command line is the fixture of the twin test, which must fail
  const std::vector<Row> rows = read_table(table);
  CHECK(rows.size() > 50);

  // 1. nodes: the registries and the table list the same names
  std::set<std::string> registered;
  for (const auto& s : g::registry()) registered.insert(s.name);
  for (const auto& n : nxr::graph::registry().names()) registered.insert(n);
  std::set<std::string> listed;
  int pending = 0, composite = 0, ok = 0, legacy = 0;
  for (const Row& r : rows) {
    if (r.kind == "node") listed.insert(r.entry);
  }
  for (const auto& n : registered) if (!listed.count(n)) { std::printf("  node not in names.tsv: %s\n", n.c_str()); CHECK(false); }
  for (const auto& n : listed) if (!registered.count(n)) { std::printf("  names.tsv lists a node that is not registered: %s\n", n.c_str()); CHECK(false); }

  // 2. every row: the same word, the same toolbox, the function declared
  for (const Row& r : rows) {
    if (r.kind == "retired") continue;
    if (r.status.rfind("pending", 0) == 0) { ++pending; continue; }
    if (r.status == "legacy") { ++legacy; continue; }
    const bool is_node = r.kind == "node";
    if (is_node) {
      const auto dot = r.entry.find('.');
      const std::string prefix = dot == std::string::npos ? "" : r.entry.substr(0, dot);
      if (!toolboxes().count(prefix)) { std::printf("  node %s: the prefix '%s' is not a toolbox\n", r.entry.c_str(), prefix.c_str()); CHECK(false); }
    }
    if (r.status == "composite") { ++composite; continue; }
    if (r.status != "ok") { std::printf("  %s %s: unknown status '%s'\n", r.kind.c_str(), r.entry.c_str(), r.status.c_str()); CHECK(false); continue; }
    ++ok;
    const std::string word = last_word(r.function), tb = toolbox_of(r.function);
    // a binding's word is the entry's last word after a dot, or after the toolbox_ of a Python native (`signal_psd`)
    std::string entry_word = last_word(r.entry);
    if (r.kind == "python") entry_word = r.entry.substr(r.entry.find('_') + 1);
    if (entry_word != word) { std::printf("  %s %s calls %s: the words differ ('%s' and '%s')\n", r.kind.c_str(), r.entry.c_str(), r.function.c_str(), entry_word.c_str(), word.c_str()); CHECK(false); }
    if (!toolboxes().count(tb)) { std::printf("  %s %s: %s is in no toolbox\n", r.kind.c_str(), r.entry.c_str(), r.function.c_str()); CHECK(false); continue; }
    if (!declared(root, tb, word)) { std::printf("  %s %s: %s is not declared in the headers of %s\n", r.kind.c_str(), r.entry.c_str(), word.c_str(), tb.c_str()); CHECK(false); }
    if (is_node) {
      const std::string prefix = r.entry.substr(0, r.entry.find('.'));
      if (prefix != tb) { std::printf("  node %s: its toolbox is %s\n", r.entry.c_str(), tb.c_str()); CHECK(false); }
    }
  }

  // 3. the bindings: every string in the sources is in the table, and every row is in the sources
  std::set<std::string> in_sources;
  const std::string mex = slurp(root + "/bindings/mex/src/first_slice.cpp");
  std::string mex_live = mex.substr(0, mex.find("const char* const kRetiredCommands"));
  mex_live += mex.substr(mex.find("bool dispatchOps"));
  for (const auto& s : found(mex_live, "\"(compute\\.[a-z_]+\\.[a-zA-Z_]+)\"")) in_sources.insert("mex " + s);
  for (const auto& s : found(slurp(root + "/bindings/mex/src/nxr_compute_mex.cpp"), "cmd == \"([\\w.]+)\"")) in_sources.insert("mex " + s);
  for (const auto& s : found(slurp(root + "/bindings/node/src/addon.cpp"), "exports\\.Set\\(\"([\\w.]+)\"")) in_sources.insert("addon " + s);
  for (const auto& s : found(slurp(root + "/bindings/wasm/src/nxr_compute_wasm.cpp"), "(?:\\.|::)function\\(\"([\\w.]+)\"")) in_sources.insert("wasm " + s);
  for (const char* file : {"compute.cpp", "access.cpp"}) {   // access.cpp: the events and the wavelet and filterbank functions
    const std::string py = slurp(root + "/bindings/python/src/" + file);
    for (const auto& s : found(py, "m\\.def\\(\"(\\w+)\"")) in_sources.insert("python " + s);
    for (const auto& s : found(py, "def_both\\(m, \"(\\w+)\"")) in_sources.insert("python " + s);
  }
  for (const auto& e : fs::recursive_directory_iterator(root + "/bindings/mex/matlab/+nxr/+compute")) {
    if (e.path().extension() != ".m") continue;
    in_sources.insert("matlab compute." + e.path().parent_path().filename().string().substr(1) + "." + e.path().stem().string());
  }
  std::set<std::string> in_table;
  for (const Row& r : rows) if (r.kind != "node" && r.kind != "retired" && r.kind != "function") in_table.insert(r.kind + " " + r.entry);
  for (const auto& s : in_sources) if (!in_table.count(s)) { std::printf("  a binding entry is not in names.tsv: %s\n", s.c_str()); CHECK(false); }
  for (const auto& s : in_table) if (!in_sources.count(s)) { std::printf("  names.tsv lists a binding entry the sources do not have: %s\n", s.c_str()); CHECK(false); }

  // 4. an old name is refused with the new one (no alias)
  int retired = 0;
  for (const Row& r : rows) {
    if (r.kind != "retired") continue;
    ++retired;
    const std::string kind_of = r.function;   // node | graphnode | mex
    if (kind_of == "node") {
      const bool refused_ok = refused([&] { g::step(r.entry); }, r.status);
      if (!refused_ok) std::printf("  not refused with the new name: %s -> %s\n", r.entry.c_str(), r.status.c_str());
      CHECK(refused_ok);
    }
    else if (kind_of == "graphnode") CHECK(refused([&] { nxr::graph::registry().at(r.entry); }, r.status));
    else if (kind_of == "mex") {
      const std::regex line("\\{\"" + r.entry + "\", \"" + r.status + "\"\\}");
      CHECK(std::regex_search(mex, line));
    } else { std::printf("  retired %s: unknown layer '%s'\n", r.entry.c_str(), kind_of.c_str()); CHECK(false); }
  }
  std::printf("  rows: %zu (ok %d, composite %d, pending %d, legacy %d, retired %d); registered nodes %zu\n", rows.size(), ok, composite, pending, legacy, retired, registered.size());
  return nxrtest::finish("test_names_across_layers");
}

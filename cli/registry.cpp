// The command table read, the command line built from it (CLI11), and what is made from the table: help, aliases, man pages, completion.
#include "registry.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

#include "CLI11/CLI11.hpp"
#include "cli.h"
#include "nxr/io/errors.h"
#include "output.h"
#ifdef NXR_NEURO_WITH_PIPELINES
int compute_main(const std::vector<std::string>& args);   // cli/compute_command.cpp
#include "nxr/pipelines/definition.h"
#endif

int run_legacy(const std::string& cmd, const std::vector<std::string>& rest);
void install_chunk_loop_defaults(const std::vector<std::string>& args);
std::vector<const char*> command_table_text();      // cli/command_table.cpp: the pieces of the table
int schema_version_for_version_flag();

namespace cli {

namespace {

std::vector<std::string> words(const std::string& s) {
  std::istringstream in(s);
  std::vector<std::string> out;
  for (std::string t; in >> t;) out.push_back(t);
  return out;
}
std::string trim(const std::string& s) {
  const auto a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
}

struct Table {
  std::vector<CommandDoc> commands;
  std::map<std::string, OptionDoc> options;
  std::vector<std::pair<std::string, std::string>> verbs;
};

#ifdef NXR_NEURO_WITH_PIPELINES
// A PIPELINE DEFINITION as a command (docs/specs/2026-11-17-compute-and-run.md, 6.2): `run <name>`, its parameters the options `--<parameter> VALUE` (dashes for underscores),
// each helped with its declared meaning, default and unit (the definition's long form); a parameter written in the short form is helped as undocumented.
std::string dashed(std::string s) {
  std::replace(s.begin(), s.end(), '_', '-');
  return s;
}
std::string default_text(const nxr::io::Json& v) {
  if (v.is_string()) return v.get<std::string>();
  return v.dump();
}
void add_definition(Table& t, const std::string& noun, const std::string& definition) {
  const nxr::pipelines::graph::Definition d = nxr::pipelines::graph::load(definition);
  CommandDoc c;
  c.verb = "run";
  c.noun = noun;
  c.old = {"run"};
  c.alias = false;
  c.unprobed = true;
  c.span = true;
  c.definition = definition;
  c.group = "Pipeline definitions";
  std::string first = d.description.substr(0, d.description.find_first_of(".\n"));
  if (!first.empty()) first[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(first[0])));
  c.summary = first.empty() ? "Run the pipeline definition " + d.name + "." : first + ".";
  c.positional = "";
  const std::string path = c.path();
  auto generic = [&](const std::string& name) {   // the options every definition has: described once, under `--name@run`
    const auto own = t.options.find(name + "@run");
    const auto any = t.options.find(name);
    OptionDoc o = own != t.options.end() ? own->second : any->second;
    t.options[name + "@" + path] = o;
  };
  std::set<std::string> params;
  for (const auto& [k, _] : d.parameters.items()) params.insert(dashed(k));
  for (const char* name : {"--dataset", "--subject", "--replace", "--dry-run", "--block-samples", "--block-frames", "--chunks", "--start", "--stop"}) {
    if (params.count(name + 2)) continue;   // a parameter of that name takes the option
    OptionUse u;
    u.name = name;
    u.required = std::string(name) == "--dataset" || std::string(name) == "--subject";
    generic(name);
    c.options.push_back(u);
  }
  if (!d.presets.empty()) {                       // `--set NAME` chooses a preset: every one listed with the values it sets
    std::string list;
    for (const auto& [pn, pv] : d.presets.items()) {
      std::string vals;
      for (const auto& [k, x] : pv.items()) vals += (vals.empty() ? "" : ", ") + k + "=" + default_text(x);
      list += std::string(list.empty() ? "" : "; ") + pn + " (" + vals + ")";
    }
    OptionDoc so;
    so.name = "--set";
    so.value = "PRESET|NAME=VALUE";
    so.meaning = "A preset, one of: " + list + ". A parameter given as --NAME VALUE or --set NAME=VALUE wins over the preset.";
    so.def = "none (the parameters' defaults)";
    so.unit = "-";
    so.show_default = true;
    t.options["--set@" + path] = so;
    OptionUse u;
    u.name = "--set";
    c.options.push_back(u);
  }
  for (const auto& [k, v] : d.parameters.items()) {
    const nxr::io::Json doc = d.parameter_docs.contains(k) ? d.parameter_docs[k] : nxr::io::Json::object();
    OptionDoc o;
    o.name = "--" + dashed(k);
    o.value = "VALUE";
    o.meaning = doc.contains("meaning") ? doc["meaning"].get<std::string>() : "(undocumented: the definition declares no meaning for this parameter)";
    if (doc.contains("choices") && doc["choices"].is_array() && !doc["choices"].empty()) {
      o.meaning += " (one of:";
      for (const auto& ch : doc["choices"]) o.meaning += " " + default_text(ch);
      o.meaning += ")";
    }
    o.def = default_text(v);
    o.unit = doc.contains("unit") ? doc["unit"].get<std::string>() : "-";
    o.show_default = true;
    t.options[o.name + "@" + path] = o;
    OptionUse u;
    u.name = o.name;
    u.legacy = k;
    u.param = true;
    c.options.push_back(u);
  }
  t.commands.push_back(c);
}
#endif

Table read_table() {
  Table t;
  CommandDoc* cur = nullptr;
  for (const char* piece : command_table_text()) {
    std::istringstream in(piece);
    for (std::string line; std::getline(in, line);) {
      if (trim(line).empty() || line[0] == '#') continue;
      if (line.rfind("verb ", 0) == 0) {   // verb <name><TAB>sentence
        const auto tab = line.find('\t');
        t.verbs.push_back({trim(line.substr(5, tab - 5)), trim(line.substr(tab + 1))});
      } else if (line.rfind("--", 0) == 0) {   // an option: --name<TAB>VALUE<TAB>meaning<TAB>default<TAB>unit
        std::vector<std::string> f;
        std::size_t a = 0;
        for (;;) {
          const auto b = line.find('\t', a);
          f.push_back(line.substr(a, b == std::string::npos ? b : b - a));
          if (b == std::string::npos) break;
          a = b + 1;
        }
        f.resize(5);
        t.options[f[0]] = OptionDoc{f[0], f[1], f[2], f[3], f[4]};
      } else if (line[0] == '@') {
        const auto w = words(line.substr(1));
        t.commands.emplace_back();
        cur = &t.commands.back();
        cur->verb = w.at(0);
        cur->noun = w.size() > 1 ? w[1] : "";
      } else if (cur) {
        const auto colon = line.find(':');
        const std::string key = line.substr(0, colon), val = trim(line.substr(colon + 1));
        if (key == "old") cur->old = words(val);
        else if (key == "group") cur->group = val;
        else if (key == "summary") cur->summary = val;
        else if (key == "example") cur->example = val;
        else if (key == "pos") cur->positional = val;
        else if (key == "span") cur->span = val == "yes";
        else if (key == "seconds") cur->seconds = val != "no";
        else if (key == "alias") { cur->alias = val != "no" && val != "refused"; cur->refused = val == "refused"; }
        else if (key == "unprobed") cur->unprobed = val == "yes";
        else if (key == "opts") {
          for (std::string w : words(val)) {
            OptionUse u;
            if (w.back() == '*') { u.required = true; w.pop_back(); }
            if (w.back() == '+') { u.multi = true; w.pop_back(); }
            const auto eq = w.find('=');
            if (eq != std::string::npos) { u.legacy = w.substr(eq + 1); w = w.substr(0, eq); }
            u.name = w;
            cur->options.push_back(u);
          }
        }
      }
    }
  }
#ifdef NXR_NEURO_WITH_PIPELINES
  for (const std::string& name : nxr::pipelines::graph::available()) add_definition(t, name, name);
#endif
  return t;
}

const Table& table() {
  static const Table t = read_table();
  return t;
}

}  // namespace

const std::vector<CommandDoc>& commands() { return table().commands; }
const std::vector<std::pair<std::string, std::string>>& verbs() { return table().verbs; }
namespace {
std::map<std::string, OptionDoc>& late_options() {   // the options of a definition file, described when `run FILE.yaml` reads it
  static std::map<std::string, OptionDoc> m;
  return m;
}
}  // namespace
const OptionDoc& option_doc(const std::string& name, const std::string& path) {
  if (!path.empty()) {
    const auto own = table().options.find(name + "@" + path);
    if (own != table().options.end()) return own->second;
    const auto late = late_options().find(name + "@" + path);
    if (late != late_options().end()) return late->second;
  }
  const auto it = table().options.find(name);
  if (it == table().options.end()) throw std::logic_error("the command table has no description of " + name);
  return it->second;
}
std::vector<std::string> pipeline_names() {
  std::vector<std::string> v;
  for (const auto& c : commands()) if (c.verb == "run" && !c.noun.empty()) v.push_back(c.noun);
  std::sort(v.begin(), v.end());
  return v;
}
bool is_flag(const std::string& name) { return option_doc(name).value.empty(); }

double parse_time(const std::string& option, const std::string& text) {
  const auto bad = [&] { return Usage("error: argument " + option + ": '" + text + "' is not a time: give seconds (90) or m:ss (1:30) or h:mm:ss"); };
  std::vector<std::string> parts;
  std::istringstream in(text);
  for (std::string p; std::getline(in, p, ':');) parts.push_back(p);
  if (parts.empty() || parts.size() > 3) throw bad();
  double total = 0;
  try {
    for (std::size_t i = 0; i < parts.size(); ++i) {
      std::size_t used = 0;
      const double v = std::stod(parts[i], &used);
      if (used != parts[i].size() || v < 0) throw bad();
      total = total * 60 + v;
    }
  } catch (const std::logic_error&) {
    throw bad();
  }
  return total;
}

namespace {

std::string seconds_text(double s) {
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", s);
  return b;
}

std::string option_help(const OptionDoc& d) {   // meaning, then the default and unit
  std::string t = d.meaning;
  std::string extra;
  if (!d.def.empty() && (d.show_default || (d.def != "none" && d.def != "off" && d.def != "required"))) extra += "default: " + d.def;
  if (!d.unit.empty() && d.unit != "-") extra += std::string(extra.empty() ? "" : "; ") + "unit: " + d.unit;
  if (!extra.empty()) t += " (" + extra + ")";
  return t;
}

bool span_hidden(const CommandDoc& c, const std::string& name) { return c.span && (name == "--chunks" || name == "--start" || name == "--stop"); }

// the full example line of a command: its own, or made from its required options
std::string example_of(const CommandDoc& c) {
  if (!c.example.empty()) return c.example;
  std::string t = "nxr-neuro " + c.path();
  if (!c.positional.empty()) t += " " + c.positional;
  for (const auto& o : c.options)
    if (o.required && !span_hidden(c, o.name)) t += " " + o.name + (is_flag(o.name) ? "" : " " + option_doc(o.name, c.path()).value);
  if (c.span) t += c.seconds ? " --from 0:10 --to 1:10" : " --all";
  return t;
}

struct Bound {
  std::string name, legacy;
  bool flag = false, multi = false, param = false;
  std::string sval;
  bool bval = false;
  std::vector<std::string> mval;
  CLI::Option* opt = nullptr;
};

struct Leaf {
  const CommandDoc* doc = nullptr;
  CLI::App* app = nullptr;
  std::deque<Bound> bound;
  std::vector<std::string> pos;
  std::string from, to;
  bool all = false, json = false, human = false, quiet = false, no_color = false;
};

void add_global(CLI::App* app, Leaf& l) {
  app->add_flag("--json", l.json, "print today's JSON lines on stdout (the default when stdout is not a terminal)");
  app->add_flag("-q,--quiet", l.quiet, "print only the result line");
  app->add_flag("--no-color", l.no_color, "no colour (also: the NO_COLOR environment variable)");
  app->add_flag("--human", l.human)->group("");   // the readable form even when stdout is not a terminal (tests)
}

std::string leaf_footer(const CommandDoc& c) {
  std::string t = "Example:\n  " + example_of(c) + "\n";
  if (c.span)
    t += "\nA time range is required: --from SECONDS --to SECONDS (seconds or m:ss), or --all for the whole recording. The computation reads one chunk at a time\n"
         "whatever the range; it records the range it actually used (after snapping to its own boundaries) with the result.\n";
  t += "\nOutput: a readable summary in a terminal, today's JSON lines with --json or when stdout is not a terminal. Progress goes to stderr, results to stdout.\n";
  if (c.refused) t += "Replaces the old spelling `nxr-neuro " + [&] { std::string s; for (auto& w : c.old) s += (s.empty() ? "" : " ") + w; return s; }() + "`, which is no longer accepted.\n";
  else if (!c.definition.empty()) t += "A pipeline definition: each parameter is an option, `--<parameter> VALUE` (dashes for underscores; VALUE is read as YAML: 8, [15, 29], dSPM); `nxr-neuro list pipelines` lists them.\n";
  else if (!c.old.empty() && c.old[0].rfind("__", 0) != 0) t += "Old spelling (still works, prints a notice): nxr-neuro " + [&] { std::string s; for (auto& w : c.old) s += (s.empty() ? "" : " ") + w; return s; }() + "\n";
  return t;
}

void build_leaf(CLI::App* verb_app, const CommandDoc& c, Leaf& l) {
  CLI::App* a = verb_app->add_subcommand(c.noun.empty() ? c.verb : c.noun, c.summary);
  l.app = a;
  l.doc = &c;
  a->footer(leaf_footer(c));
  if (!c.group.empty()) a->group(c.group);
  if (!c.positional.empty()) a->add_option(c.positional, l.pos)->required()->expected(1, 1)->type_name(c.positional);
  for (const auto& u : c.options) {
    const OptionDoc& d = option_doc(u.name, c.path());
    l.bound.emplace_back();
    Bound& b = l.bound.back();
    b.name = u.name;
    b.legacy = u.legacy.empty() ? u.name : u.legacy;
    b.flag = d.value.empty();
    b.multi = u.multi;
    b.param = u.param;
    if (b.flag) b.opt = a->add_flag(u.name, b.bval, option_help(d));
    else if (u.multi) b.opt = a->add_option(u.name, b.mval, option_help(d))->type_name(d.value)->expected(1, 1000000);
    else b.opt = a->add_option(u.name, b.sval, option_help(d))->type_name(d.value);
    if (u.required) b.opt->required();
    if (span_hidden(c, u.name)) b.opt->group("");
  }
  if (c.span) {
    if (c.seconds) {
      a->add_option("--from", l.from, "start of the time range: seconds, m:ss or h:mm:ss (unit: s)")->type_name("TIME");
      a->add_option("--to", l.to, "end of the time range: seconds, m:ss or h:mm:ss (unit: s)")->type_name("TIME");
    }
    a->add_flag("--all", l.all, "the whole recording (still read one chunk at a time)");
  }
  add_global(a, l);
}

// the old command's argv from what was parsed; throws Usage when the time range is missing
std::vector<std::string> rewrite(Leaf& l, bool& dry) {
  const CommandDoc& c = *l.doc;
  std::vector<std::string> out;
  for (std::size_t i = 1; i < c.old.size(); ++i) out.push_back(c.old[i]);
  if (!c.definition.empty()) out.push_back(c.definition);   // a definition's command: the definition is run's positional argument
  for (const auto& p : l.pos) out.push_back(p);
  bool chunks = false, start = false, stop = false;
  dry = false;
  for (const Bound& b : l.bound) {
    if (b.flag) {
      if (b.opt->count() > 0) {
        out.push_back(b.legacy);
        if (b.name == "--dry-run") dry = true;
      }
    } else if (b.multi) {
      if (b.opt->count() > 0) { out.push_back(b.legacy); for (const auto& v : b.mval) out.push_back(v); }
    } else if (b.opt->count() > 0 && b.param) {   // a parameter of a definition: the runner takes it as `--set NAME=VALUE`
      out.push_back("--set");
      out.push_back(b.legacy + "=" + b.sval);
    } else if (b.opt->count() > 0) {
      out.push_back(b.legacy);
      out.push_back(b.sval);
      if (b.name == "--chunks") chunks = true;
      if (b.name == "--start") start = true;
      if (b.name == "--stop") stop = true;
    }
  }
  if (c.span) {
    const bool from = !l.from.empty(), to = !l.to.empty();
    const std::string prog = "nxr-neuro " + c.path();
    const int given = (from || to ? 1 : 0) + (l.all ? 1 : 0) + (chunks ? 1 : 0) + (start || stop ? 1 : 0);
    if (given == 0 && !dry)
      throw Usage(prog + ": error: a time range is required: give " + (c.seconds ? "--from SECONDS --to SECONDS (seconds or m:ss), or " : "") + "--all for the whole recording");
    if (given > 1) throw Usage(prog + ": error: give one time range: " + (c.seconds ? "--from and --to, or " : "") + "--all");
    if (from != to) throw Usage(prog + ": error: --from and --to go together: give both");
    if (from) { out.push_back("--start"); out.push_back(seconds_text(parse_time("--from", l.from))); out.push_back("--stop"); out.push_back(seconds_text(parse_time("--to", l.to))); }
    if (l.all) { out.push_back("--chunks"); out.push_back("all"); }
  }
  return out;
}

std::string old_words(const CommandDoc& c) {
  std::string s;
  for (const auto& w : c.old) s += (s.empty() ? "" : " ") + w;
  return s;
}

int dispatch(const CommandDoc& c, const std::vector<std::string>& rest) {
  install_chunk_loop_defaults(rest);
  return run_legacy(c.old.at(0), rest);
}

int guarded(const std::function<int()>& f) {
  try {
    return f();
  } catch (const Usage& e) {
    std::cerr << e.what() << '\n';
    return 2;
  } catch (const nxr::io::Error& e) {
    std::string m = e.what();
    if (e.python_name() == "KeyError") {   // Python prints a KeyError's message as its repr
      const char q = m.find('\'') != std::string::npos && m.find('"') == std::string::npos ? '"' : '\'';
      std::string r(1, q);
      for (char ch : m) { if (ch == '\\') r += "\\\\"; else if (ch == q) { r += '\\'; r += ch; } else r += ch; }
      m = r + q;
    }
    report_error(e.python_name() + ": " + m, e.code());
    return 1;
  } catch (const std::exception& e) {
    report_error(std::string("RuntimeError: ") + e.what(), "runtime_error");
    return 1;
  }
}

std::string root_description() {
  std::string t = "nxr-neuro: measure the spatiotemporal patterns of cortical activity from MEG/EEG, and keep the results in an nxr dataset.\n\nVerbs:\n";
  for (const auto& v : verbs()) { std::string n = v.first; n.resize(std::max<std::size_t>(n.size(), 10), ' '); t += "  " + n + v.second + "\n"; }
  return t;
}

}  // namespace

int run_command_line(int argc, char** argv) {
  // a hidden developer check: each command's options against what its parser accepts
  if (argc >= 2 && std::string(argv[1]) == "__check-options") {
    int bad = 0;
    for (const CommandDoc& c : commands()) {
      if (c.unprobed || c.old.empty()) continue;
      probe() = Probe();
      probe().on = true;
      try { run_legacy(c.old[0], std::vector<std::string>(c.old.begin() + 1, c.old.end())); } catch (const ProbeDone&) {} catch (const std::exception& e) { std::cout << c.path() << ": probe failed: " << e.what() << "\n"; ++bad; continue; }
      probe().on = false;
      Probe p = probe();
      std::set<std::string> parser, req, mul, flg, table_names, table_req, table_mul, table_flg;
      parser.insert(p.valued.begin(), p.valued.end()); parser.insert(p.multi.begin(), p.multi.end()); parser.insert(p.flags.begin(), p.flags.end());
      for (const auto& u : c.options) { const std::string n = u.legacy.empty() ? u.name : u.legacy; table_names.insert(n); if (u.required) table_req.insert(n); if (u.multi) table_mul.insert(n); if (is_flag(u.name)) table_flg.insert(n); }
      auto diff = [&](const char* what, const std::set<std::string>& a, const std::set<std::string>& b) {
        std::string m;
        for (const auto& x : a) if (!b.count(x)) m += " " + x;
        if (!m.empty()) { std::cout << c.path() << ": " << what << m << "\n"; ++bad; }
      };
      diff("the parser has, the table lacks:", parser, table_names);
      diff("the table has, the parser lacks:", table_names, parser);
      diff("required by the parser only:", p.required, table_req);
      diff("required by the table only:", table_req, p.required);
      diff("multi in the parser only:", p.multi, table_mul);
      diff("multi in the table only:", table_mul, p.multi);
      diff("flag in the parser only:", p.flags, table_flg);
      diff("flag in the table only:", table_flg, p.flags);
      if ((p.npos > 0) != !c.positional.empty()) { std::cout << c.path() << ": positional argument differs\n"; ++bad; }
      for (const auto& u : c.options) if (!table().options.count(u.name)) { std::cout << c.path() << ": " << u.name << " has no description\n"; ++bad; }
    }
    std::cout << (bad ? "the command table does not match the parsers\n" : "the command table matches the parsers\n");
    return bad ? 1 : 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__refused") {   // for the tests: `old words|new words` of every refused spelling
    for (const auto& c : commands()) if (c.refused) std::cout << old_words(c) << "|" << c.path() << "\n";
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__check-pipelines") {   // the pipelines' names and the shipped definitions' parameters (spec 2026-11-17-compute-and-run.md, 6.2)
    int bad = 0;
    std::set<std::string> seen;
    for (const auto& c : commands()) {
      if (c.verb != "run" || c.noun.empty()) continue;
      if (!seen.insert(c.noun).second) { std::cout << c.noun << ": a pipeline name is in two tables (the hand-wired pipelines and the shipped definitions)\n"; ++bad; }
      if (c.noun.find('.') != std::string::npos) { std::cout << c.noun << ": a pipeline's name has no dot (a dot is a node's: <group>.<name>)\n"; ++bad; }
      if (c.definition.empty()) continue;
      for (const auto& u : c.options) {
        if (!u.param) continue;
        const OptionDoc& d = option_doc(u.name, c.path());
        if (d.meaning.rfind("(undocumented", 0) == 0) { std::cout << c.noun << ": " << u.name << " declares no meaning\n"; ++bad; }
        if (d.unit.empty()) { std::cout << c.noun << ": " << u.name << " declares no unit\n"; ++bad; }
      }
    }
    std::cout << (bad ? "the pipelines' names or parameters are not as declared\n" : "every pipeline has its own name and every shipped definition declares its parameters\n");
    return bad ? 1 : 0;
  }
  if (argc >= 3 && std::string(argv[1]) == "__options") {   // what an old command's parser accepts (the generator of the table uses it)
    probe() = Probe();
    probe().on = true;
    try { run_legacy(argv[2], std::vector<std::string>(argv + 3, argv + argc)); } catch (const ProbeDone&) {}
    const Probe& p = probe();
    auto j = [](const std::set<std::string>& s) { std::string t; for (const auto& x : s) t += (t.empty() ? "" : " ") + x; return t; };
    std::cout << "valued: " << j(p.valued) << "\nmulti: " << j(p.multi) << "\nflags: " << j(p.flags) << "\nrequired: " << j(p.required) << "\nnpos: " << p.npos << "\n";
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__check-help") {   // every option of every command carries its own description (except the few that mean the same everywhere)
    static const std::set<std::string> shared = {"--dataset", "--subject", "--replace", "--json", "--quiet", "--no-color", "--from", "--to", "--all"};
    int bad = 0;
    for (const CommandDoc& c : commands()) {
      if (!c.definition.empty()) continue;   // a definition's parameters are checked by __check-pipelines
      for (const auto& u : c.options) {
        if (span_hidden(c, u.name) || shared.count(u.name)) continue;
        const auto it = table().options.find(u.name + "@" + c.path());
        if (it == table().options.end()) { std::cout << c.path() << ": " << u.name << " has no description of its own\n"; ++bad; continue; }
        const OptionDoc& d = it->second;
        if (trim(d.meaning).size() < 10) { std::cout << c.path() << ": " << u.name << " has an empty description\n"; ++bad; }
        if (d.def.empty() || d.unit.empty()) { std::cout << c.path() << ": " << u.name << " has no default or unit column\n"; ++bad; }
        const std::string h = option_help(d);
        if (!d.def.empty() && d.def != "none" && d.def != "off" && d.def != "required" && h.find("default: " + d.def) == std::string::npos) { std::cout << c.path() << ": " << u.name << " help lacks its default\n"; ++bad; }
        if (!d.unit.empty() && d.unit != "-" && h.find("unit: " + d.unit) == std::string::npos) { std::cout << c.path() << ": " << u.name << " help lacks its unit\n"; ++bad; }
        if (d.value.empty() != is_flag(u.name)) { std::cout << c.path() << ": " << u.name << " flag and value disagree\n"; ++bad; }
      }
    }
    std::cout << (bad ? "some options lack their own description\n" : "every option has its own description\n");
    return bad ? 1 : 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__commands") {   // for the tests: `new words | old words | span or nospan | seconds or all-only`
    for (const auto& c : commands()) {
      std::cout << c.path() << "|" << (c.alias && !c.old.empty() && c.old[0].rfind("__", 0) != 0 ? old_words(c) : std::string()) << "|" << (c.span ? "span" : "nospan") << "|" << (c.seconds ? "seconds" : "all-only") << "\n";
    }
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__man") {
    const std::filesystem::path dir = argc >= 3 ? argv[2] : ".";
    std::filesystem::create_directories(dir);
    auto write = [&](const std::string& v, const std::string& n) {
      std::string name = "nxr-neuro" + (v.empty() ? "" : "-" + v) + (n.empty() ? "" : "-" + n);
      std::ofstream(dir / (name + ".1")) << man_page(v, n);
    };
    write("", "");
    std::set<std::string> done;
    for (const auto& c : commands()) { if (done.insert(c.verb).second) write(c.verb, ""); write(c.verb, c.noun); }
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "__completions") {   // the three scripts into a folder (the build installs them)
    const std::filesystem::path dir = argc >= 3 ? argv[2] : ".";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "nxr-neuro.bash") << completion_script("bash");
    std::ofstream(dir / "_nxr-neuro") << completion_script("zsh");
    std::ofstream(dir / "nxr-neuro.fish") << completion_script("fish");
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "completion") {
    if (argc < 3 || (std::string(argv[2]) != "bash" && std::string(argv[2]) != "zsh" && std::string(argv[2]) != "fish")) {
      std::cerr << "usage: nxr-neuro completion bash|zsh|fish\n  e.g. nxr-neuro completion bash > /etc/bash_completion.d/nxr-neuro\n";
      return 2;
    }
    std::cout << completion_script(argv[2]);
    return 0;
  }

  // THE OLD SPELLINGS ARE NOT KEPT (docs/specs/2026-11-17-compute-and-run.md, 6.3): a pipeline is run with `run <pipeline>`; the old spelling is refused, naming the new one. `compute` is reserved
  // for nodes (a later step).
  if (argc >= 2 && std::string(argv[1]) == "compute") {
    const std::string next = argc >= 3 ? argv[2] : "";
    const std::vector<std::string> names = pipeline_names();
    const bool pipeline = !next.empty() && std::find(names.begin(), names.end(), next) != names.end();
    if (pipeline) { std::cerr << "nxr-neuro: error: 'compute " << next << "' is now 'run " << next << "' (the same options): pipelines are run with `run`; `compute` is for nodes\n"; return 2; }
#ifdef NXR_NEURO_WITH_PIPELINES
    return compute_main(std::vector<std::string>(argv + 2, argv + argc));
#else
    std::cerr << "nxr-neuro compute needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON\n";
    return 2;
#endif
  }
  if (argc >= 2) {
    const std::string first = argv[1];
    for (const auto& c : commands()) {
      if (!c.refused || c.old.empty() || c.old[0] != first) continue;
      bool match = true;
      for (std::size_t i = 1; i < c.old.size() && match; ++i) match = argc > static_cast<int>(1 + i) && std::string(argv[1 + i]) == c.old[i];
      if (!match) continue;
      std::cerr << "nxr-neuro: error: '" << old_words(c) << "' is now '" << c.path() << "' (the same options); the old spelling is not kept\n";
      return 2;
    }
  }
  // `run FILE.yaml`: the user's own definition, a command made when it is read (a name that CLI11 cannot hold is stood in for by `file`)
  std::deque<CommandDoc> extra;
  std::vector<std::string> args(argv, argv + argc);
  if (argc >= 3 && std::string(argv[1]) == "run") {
    const std::string target = argv[2];
    const bool file = target.size() > 5 && (target.compare(target.size() - 5, 5, ".yaml") == 0 || target.compare(target.size() - 4, 4, ".yml") == 0 || target.find('/') != std::string::npos);
    if (file) {
#ifdef NXR_NEURO_WITH_PIPELINES
      try {
        Table one;
        for (const auto& [k, v] : table().options) one.options[k] = v;
        add_definition(one, "file", target);
        extra.push_back(one.commands.back());
        for (const auto& [k, v] : one.options) if (k.size() > 9 && k.compare(k.size() - 9, 9, "@run file") == 0) late_options()[k] = v;   // the file's options, described beside the table's
        args[2] = "file";
      } catch (const std::exception& e) {
        return guarded([&]() -> int { throw; });
      }
#endif
    }
  }
  // the old spellings: kept working for a deprecation period, with a one-line notice on stderr
  if (argc >= 2) {
    const std::string first = argv[1];
    const CommandDoc* hit = nullptr;
    for (const auto& c : commands()) {
      if (!c.alias || c.old.empty() || c.old[0].rfind("__", 0) == 0 || c.old[0] != first) continue;
      bool match = true;
      for (std::size_t i = 1; i < c.old.size() && match; ++i) match = argc > static_cast<int>(1 + i) && std::string(argv[1 + i]) == c.old[i];
      // a word that is also a new verb: only the old spelling's own arguments tell them apart (`run --list`, `read FOLDER`)
      // a word that is also a verb of the new grammar: the old spelling only when no new command has the next word (`read FOLDER`, not `read crossspectrum`)
      if (match && c.old.size() == 1)
        for (const auto& o : commands()) if (o.verb == first && !o.noun.empty() && argc > 2 && std::string(argv[2]) == o.noun) match = false;
      if (match && (!hit || c.old.size() > hit->old.size())) hit = &c;
    }
    if (hit) {
      std::vector<std::string> rest(hit->old.begin() + 1, hit->old.end());   // the action word the old main expects (`compute`, `create`, ...)
      for (int i = 1 + static_cast<int>(hit->old.size()); i < argc; ++i) rest.push_back(argv[i]);
      const bool wants_help = std::find_if(rest.begin() + (hit->old.size() - 1), rest.end(), [](const std::string& s) { return s == "-h" || s == "--help"; }) != rest.end();
      if (!wants_help) {
        configure_output(false, false, false, false, "");
        std::cerr << "nxr-neuro: note: '" << old_words(*hit) << "' is the old spelling; use '" << hit->path() << "' (the old one still works for now)\n";
        output().command = hit->path();
        return guarded([&] { return dispatch(*hit, rest); });
      }
      std::vector<std::string> nargv = {argv[0], hit->verb};
      if (!hit->noun.empty()) nargv.push_back(hit->noun);
      nargv.push_back("--help");
      std::vector<char*> p;
      for (auto& s : nargv) p.push_back(s.data());
      return run_command_line(static_cast<int>(p.size()), p.data());
    }
  }

  CLI::App app{root_description(), "nxr-neuro"};
  app.set_version_flag("--version", std::string("nxr-neuro ") + NXR_NEURO_VERSION + " (database schema " + std::to_string(schema_version_for_version_flag()) + ")");
  app.require_subcommand(1);
  app.footer("Run `nxr-neuro <verb> --help` for a verb's commands, `nxr-neuro <verb> <what> --help` for one command's options.\n"
             "Shell completion: `nxr-neuro completion bash|zsh|fish`.");
  std::map<std::string, CLI::App*> verb_apps;
  std::deque<Leaf> leaves;
  std::vector<const CommandDoc*> all;
  for (const auto& c : commands()) all.push_back(&c);
  for (const auto& c : extra) all.push_back(&c);
  std::set<std::string> direct;   // a verb that is itself the command (`read FOLDER`)
  for (const CommandDoc* c : all) if (c->noun.empty()) direct.insert(c->verb);
  for (const auto& v : verbs()) {
    if (direct.count(v.first)) continue;
    CLI::App* va = app.add_subcommand(v.first, v.second);
    va->require_subcommand(1);
    verb_apps[v.first] = va;
  }
  for (const CommandDoc* c : all) {
    CLI::App* va = c->noun.empty() ? &app : verb_apps.at(c->verb);
    leaves.emplace_back();
    build_leaf(va, *c, leaves.back());
  }
  for (const auto& v : verbs()) {
    if (direct.count(v.first)) continue;
    std::string ex;
    for (const CommandDoc* c : all) if (c->verb == v.first && ex.empty()) ex = example_of(*c);
    verb_apps[v.first]->footer("Example:\n  " + ex + "\n");
  }
  // `--set NAME=VALUE` was a definition's way; a parameter is now its own option
  if (args.size() >= 3 && args[1] == "run") {
    for (const CommandDoc* c : all)
      if (c->verb == "run" && c->noun == args[2] && !c->definition.empty())
        for (std::size_t i = 3; i < args.size(); ++i)
          if (args[i] == "--set" || args[i].rfind("--set=", 0) == 0) {
            std::cerr << "nxr-neuro run " << args[2] << ": error: `--set NAME=VALUE` is now `--NAME VALUE`: a parameter is an option of its own (`nxr-neuro run " << args[2] << " --help` lists them; dashes stand for underscores)\n";
            return 2;
          }
  }
  std::vector<const char*> av;
  for (const auto& a : args) av.push_back(a.c_str());
  try {
    app.parse(static_cast<int>(av.size()), av.data());
  } catch (const CLI::ParseError& e) {
    return app.exit(e) == 0 ? 0 : 2;   // a usage error is 2, whatever CLI11 numbers it
  }
  for (Leaf& l : leaves) {
    if (!l.app->parsed()) continue;
    return guarded([&] {
      configure_output(l.json, l.human, l.quiet, l.no_color, l.doc->path());
      bool dry = false;
      const std::vector<std::string> rest = rewrite(l, dry);
      return dispatch(*l.doc, rest);
    });
  }
  std::cerr << app.help();
  return 2;
}

// ── completion ──
std::string completion_script(const std::string& shell) {
  std::map<std::string, std::vector<const CommandDoc*>> by_verb;
  for (const auto& c : commands()) if (!c.noun.empty()) by_verb[c.verb].push_back(&c);
  auto opts_of = [](const CommandDoc& c) {
    std::vector<std::string> o;
    for (const auto& u : c.options) if (!span_hidden(c, u.name)) o.push_back(u.name);
    if (c.span) { if (c.seconds) { o.push_back("--from"); o.push_back("--to"); } o.push_back("--all"); }
    for (const char* g : {"--json", "--quiet", "--no-color", "--help"}) o.push_back(g);
    return o;
  };
  auto join = [](const std::vector<std::string>& v) { std::string s; for (const auto& x : v) s += (s.empty() ? "" : " ") + x; return s; };
  std::ostringstream o;
  if (shell == "bash") {
    o << "# bash completion for nxr-neuro (generated by `nxr-neuro completion bash`)\n_nxr_neuro() {\n  local cur=\"${COMP_WORDS[COMP_CWORD]}\" verb=\"${COMP_WORDS[1]}\" noun=\"${COMP_WORDS[2]}\"\n  if [ \"$COMP_CWORD\" -eq 1 ]; then\n    COMPREPLY=($(compgen -W \"";
    std::vector<std::string> vs;
    for (const auto& v : verbs()) vs.push_back(v.first);
    o << join(vs) << " completion --help --version\" -- \"$cur\")); return\n  fi\n  case \"$verb\" in\n";
    for (const auto& [v, cs] : by_verb) {
      std::vector<std::string> ns;
      for (auto* c : cs) ns.push_back(c->noun);
      o << "    " << v << ")\n      if [ \"$COMP_CWORD\" -eq 2 ]; then COMPREPLY=($(compgen -W \"" << join(ns) << "\" -- \"$cur\")); return; fi\n      case \"$noun\" in\n";
      for (auto* c : cs) o << "        " << c->noun << ") COMPREPLY=($(compgen -W \"" << join(opts_of(*c)) << "\" -- \"$cur\")) ;;\n";
      o << "      esac ;;\n";
    }
    o << "  esac\n}\ncomplete -F _nxr_neuro nxr-neuro\n";
  } else if (shell == "fish") {
    o << "# fish completion for nxr-neuro (generated by `nxr-neuro completion fish`)\ncomplete -c nxr-neuro -f\n";
    for (const auto& v : verbs()) o << "complete -c nxr-neuro -n '__fish_use_subcommand' -a " << v.first << " -d '" << v.second << "'\n";
    for (const auto& [v, cs] : by_verb) {
      for (auto* c : cs) {
        std::string q = c->summary;
        std::replace(q.begin(), q.end(), '\'', ' ');
        o << "complete -c nxr-neuro -n '__fish_seen_subcommand_from " << v << "; and not __fish_seen_subcommand_from";
        for (auto* d : cs) o << " " << d->noun;
        o << "' -a " << c->noun << " -d '" << q << "'\n";
        for (const auto& name : opts_of(*c)) {
          if (name == "--help") continue;
          o << "complete -c nxr-neuro -n '__fish_seen_subcommand_from " << c->noun << "' -l " << name.substr(2);
          if (table().options.count(name)) { std::string h = option_help(option_doc(name, c->path())); std::replace(h.begin(), h.end(), '\'', ' '); o << (is_flag(name) ? "" : " -r") << " -d '" << h << "'"; }
          o << "\n";
        }
      }
    }
  } else {
    o << "#compdef nxr-neuro\n# zsh completion for nxr-neuro (generated by `nxr-neuro completion zsh`)\n_nxr_neuro() {\n  local -a verbs nouns opts\n  if (( CURRENT == 2 )); then\n    verbs=(";
    for (const auto& v : verbs()) o << "'" << v.first << ":" << v.second << "' ";
    o << ")\n    _describe 'verb' verbs; return\n  fi\n  case \"$words[2]\" in\n";
    for (const auto& [v, cs] : by_verb) {
      o << "    " << v << ")\n      if (( CURRENT == 3 )); then nouns=(";
      for (auto* c : cs) { std::string s = c->summary; std::replace(s.begin(), s.end(), '\'', ' '); std::replace(s.begin(), s.end(), ':', ' '); o << "'" << c->noun << ":" << s << "' "; }
      o << "); _describe 'command' nouns; return; fi\n      case \"$words[3]\" in\n";
      for (auto* c : cs) o << "        " << c->noun << ") opts=(" << join(opts_of(*c)) << "); compadd -a opts ;;\n";
      o << "      esac ;;\n";
    }
    o << "  esac\n}\n_nxr_neuro \"$@\"\n";
  }
  return o.str();
}

// ── man pages ──
namespace {
std::string roff(std::string s) {
  std::string out;
  for (char c : s) { if (c == '\\') out += "\\\\"; else if (c == '-') out += "\\-"; else out += c; }
  return out;
}
}  // namespace

std::string man_page(const std::string& verb, const std::string& noun) {
  std::ostringstream o;
  const std::string name = "nxr-neuro" + (verb.empty() ? "" : " " + verb) + (noun.empty() ? "" : " " + noun);
  std::string dashed = name;
  std::replace(dashed.begin(), dashed.end(), ' ', '-');
  o << ".TH " << roff(dashed) << " 1 \"\" \"nxr-neuro " << NXR_NEURO_VERSION << "\" \"nxr-neuro\"\n.SH NAME\n" << roff(name) << " \\- ";
  if (verb.empty()) {
    o << "measure the spatiotemporal patterns of cortical activity\n.SH SYNOPSIS\n.B nxr-neuro\n<verb> [<what>] [options]\n.SH VERBS\n";
    for (const auto& v : verbs()) o << ".TP\n.B " << v.first << "\n" << roff(v.second) << "\n";
  } else if (noun.empty()) {
    for (const auto& v : verbs()) if (v.first == verb) o << roff(v.second) << "\n";
    o << ".SH SYNOPSIS\n.B nxr-neuro " << verb << "\n<what> [options]\n.SH COMMANDS\n";
    for (const auto& c : commands()) if (c.verb == verb) o << ".TP\n.B " << c.noun << "\n" << roff(c.summary) << (c.group.empty() ? "" : " (" + roff(c.group) + ")") << "\n";
  } else {
    for (const auto& c : commands()) {
      if (c.verb != verb || c.noun != noun) continue;
      o << roff(c.summary) << "\n.SH SYNOPSIS\n.B " << roff(name) << "\n" << (c.positional.empty() ? "" : roff(c.positional) + " ") << "[options]\n.SH OPTIONS\n";
      auto opt = [&](const std::string& n, const std::string& v, const std::string& h) { o << ".TP\n.B " << roff(n) << (v.empty() ? "" : " \\fI" + roff(v) + "\\fR") << "\n" << roff(h) << "\n"; };
      for (const auto& u : c.options) { if (span_hidden(c, u.name)) continue; const OptionDoc& d = option_doc(u.name, c.path()); opt(u.name, d.value, option_help(d) + (u.required ? " Required." : "")); }
      if (c.span) {
        if (c.seconds) { opt("--from", "TIME", "Start of the time range: seconds, m:ss or h:mm:ss."); opt("--to", "TIME", "End of the time range."); }
        opt("--all", "", "The whole recording. One of the time range options is required.");
      }
      opt("--json", "", "Print JSON lines on stdout (the default when stdout is not a terminal).");
      opt("--quiet", "", "Print only the result line.");
      opt("--no-color", "", "No colour.");
      o << ".SH EXAMPLES\n.nf\n" << roff(example_of(c)) << "\n.fi\n";
    }
  }
  return o.str();
}

}  // namespace cli

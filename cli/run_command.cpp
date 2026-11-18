// nxr-neuro run <pipeline definition> — a pipeline definition (nxr's format, YAML) run on a subject: the runner (pipelines/include/nxr/pipelines/runner.h; the spec
// docs/specs/2026-11-12-chain-graph.md §4). The command line (cli/registry.cpp) is what the user types: `run <name | file.yaml> --dataset D --subject S --<parameter> VALUE
// (--from S --to S | --all)`, each definition's parameters its own options with their declared meaning, default and unit (docs/specs/2026-11-17-compute-and-run.md, 6.2);
// it hands this parser the old form below. The hand-wired pipelines (`run psd`, ...) have their own parsers.
//
//   DEFINITION [--set NAME=VALUE ...] --dataset D/N --subject S [--chunks K|A-B|all | --start S --stop S] [--replace] [--dry-run]
//   --list       (reached as `nxr-neuro list pipelines`)
//
//   DEFINITION   one of nxr-neuro's own (by name) or a .yaml path
//   --set        a parameter overridden, NAME=VALUE with VALUE read as YAML (8, [15, 29], dSPM); repeatable (what `--<parameter> VALUE` is turned into).
//                `--set NAME` alone chooses one of the definition's presets (named parameter sets): defaults < the preset < the caller's own values
//   --chunks / --start --stop   the span read (io's chunk service: one chunk by default)
//   --projectors active|none   the recording's active projectors applied to every block loaded (active, the default) or the samples as recorded (none)
//   --replace    another job's outputs of the same names are moved aside (never deleted) and replaced; without it they are refused. A rerun with the same
//                definition, parameters, inputs and span is a cache hit (nothing computed).
//   --dry-run    the plan: inputs found, every arrow checked, every output's place known; nothing read beyond the rows, nothing written
//   --list       nxr-neuro's definitions and the registered steps
#include <algorithm>
#include <iostream>

#include "cli.h"
#include "registry.h"
#include "nxr/io/database.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/writer.h"
#include "nxr/io/errors.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::io;
namespace g = nxr::pipelines::graph;

namespace {
[[noreturn]] void usage(const std::string& prog, const std::string& m) { throw cli::Usage(prog + ": error: " + m); }
}  // namespace

int run_main(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro run";
  const bool list = std::find(rest.begin(), rest.end(), "--list") != rest.end();
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--chunks", "--start", "--stop", "--block-samples", "--block-frames", "--projectors"}, {"--dry-run", "--list", "--replace"}, {}, list ? 0 : 1, {"--set"}, "DEFINITION");
  if (a.flag.count("--list")) {
    Json defs = Json::array(), all = Json::array();
    for (const auto& n : g::available()) defs.push_back(n);
    for (const auto& n : cli::pipeline_names()) all.push_back(n);   // every pipeline `run` takes: the hand-wired ones and the definitions
    cli::emit(Json{{"stage", "result"}, {"pipelines", all}, {"definitions", defs}, {"steps", g::catalogue()}});
    return 0;
  }
  if (a.pos.empty()) usage(prog, "name a DEFINITION (one of nxr-neuro's: `list pipelines`; or a .yaml path)");
  if (!a.maybe("--dataset") || !a.maybe("--subject")) usage(prog, "--dataset and --subject are required");
  Json set = Json::object();
  std::string preset;                                    // `--set NAME` (no `=`): one of the definition's named parameter sets; the caller's own `--set P=V` and `--P V` win over it
  if (a.multi.count("--set"))
    for (const auto& s : a.multi.at("--set")) {
      const auto eq = s.find('=');
      if (eq == 0) usage(prog, "--set " + s + ": NAME=VALUE, or a preset's NAME");
      if (eq == std::string::npos) {
        if (!preset.empty()) usage(prog, "--set " + s + ": one preset a run (already --set " + preset + ")");
        preset = s;
        continue;
      }
      set[s.substr(0, eq)] = parse_yaml(s.substr(eq + 1), "--set " + s.substr(0, eq));
    }
  const g::Definition d = g::load(a.pos[0]);
  if (a.flag.count("--dry-run")) {
    // the subject is opened as `run` opens it (a one-database dataset or a catalogue of subject packages: the subject's own database); opening writes nothing and nothing is computed or stored
    DatasetWriter ds = open_dataset(a);
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"));
    const Database& db = sub.db();
    for (const auto& P : g::plan_runs(d, db, sub.id(), set, preset)) {
      std::cout << P.text(db);
      Json out = Json{{"stage", "plan"}, {"pipeline", d.name + "@" + std::to_string(d.version)}, {"subject", a.get("--subject")}, {"parameters", P.parameters},
                      {"inputs", nxr::io::finder::describe(P.found)}};
      if (const auto anchor = d.anchor()) out["anchor"] = text(*P.found.at(*anchor).row, "name");
      cli::emit(out);
    }
    return 0;
  }
  g::RunOptions o;
  o.chunks = a.maybe("--chunks");
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  o.replace = a.flag.count("--replace") > 0;
  o.preset = preset;
  if (const auto v = a.maybe("--block-samples")) o.block_samples = std::stoll(*v);
  if (const auto v = a.maybe("--block-frames")) o.block_frames = std::stoll(*v);
  if (const auto v = a.maybe("--projectors")) {
    if (*v != "active" && *v != "none") usage(prog, "--projectors is active (the recording's active projectors applied on load, the default) or none (the samples as recorded)");
    o.projectors = nxr::io::projectors_from(*v);
  }
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = cli::stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  for (const auto& r : g::run(ds, sub, d, set, o, false, [](const Json& j) { cli::emit(j); })) {
    Json out = Json{{"stage", "result"}};
    for (auto it = r.begin(); it != r.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  }
  return 0;
}

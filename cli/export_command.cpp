// nxr-neuro export <what>: stored results written as the field's standard files (docs/specs/2026-11-17-interoperability.md, sections 3.3 and 5).
//
//   nxr-neuro export gifti --package PATH (--field NAME | --labelling NAME) --out STEM [--kind func|shape] [--with-surface] [--hemispheres] [--surface-unit m|mm] [--from S --to S | --all]
//
// The command parses and prints; io's exporters (nxr/exporters/) do the work. A series needs its time range (--from and --to in seconds, or --all): there is no default.
// Progress and results are JSON lines.
#include <cstdlib>

#include "cli.h"
#include "nxr/exporters/gifti.h"

using namespace nxr::io;
using namespace cli;

namespace {
// seconds as `12.5`, `1:30` or `1:02:03`
double seconds_of(const std::string& prog, const std::string& name, const std::string& text) {
  double total = 0;
  std::size_t start = 0;
  try {
    for (;;) {
      const std::size_t colon = text.find(':', start);
      std::size_t used = 0;
      const std::string part = text.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
      const double v = std::stod(part, &used);
      if (used != part.size() || v < 0) throw std::invalid_argument(part);
      total = total * 60.0 + v;
      if (colon == std::string::npos) break;
      start = colon + 1;
    }
  } catch (const std::exception&) {
    throw Usage(prog + ": error: argument " + name + ": '" + text + "' is not a time in seconds (12.5, 1:30 or 1:02:03)");
  }
  return total;
}

std::string command_line(const std::string& what, const std::vector<std::string>& rest) {
  std::string c = "nxr-neuro export " + what;
  for (const auto& t : rest) c += " " + t;
  return c;
}

int gifti_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export gifti";
  const Args a = parse(prog, rest, {"--package", "--field", "--labelling", "--out", "--kind", "--surface-unit", "--from", "--to"}, {"--with-surface", "--hemispheres", "--all"}, {"--package", "--out"});
  nxr::exporters::GiftiExport o;
  o.field = a.maybe("--field").value_or("");
  o.labelling = a.maybe("--labelling").value_or("");
  if (o.field.empty() == o.labelling.empty()) throw Usage(prog + ": error: give exactly one of --field and --labelling");
  o.out = a.get("--out");
  o.kind = a.maybe("--kind").value_or("func");
  o.surface = a.flag.count("--with-surface") > 0;
  o.hemispheres = a.flag.count("--hemispheres") > 0;
  o.surface_unit = a.maybe("--surface-unit");
  const bool from = a.maybe("--from").has_value(), to = a.maybe("--to").has_value(), all = a.flag.count("--all") > 0;
  if (from != to) throw Usage(prog + ": error: --from and --to go together: give both");
  if (all && from) throw Usage(prog + ": error: give one time range: --from and --to, or --all");
  if (from) o.range = nxr::exporters::SeriesRange{false, seconds_of(prog, "--from", a.get("--from")), seconds_of(prog, "--to", a.get("--to"))};
  if (all) o.range = nxr::exporters::SeriesRange{true, 0, 0};
  o.ctx.version = NXR_NEURO_VERSION;
  o.ctx.command = command_line("gifti", rest);
  Json opts = Json::object();
  for (const auto& kv : a.opt) opts[kv.first.substr(2)] = kv.second;
  for (const auto& f : a.flag) opts[f.substr(2)] = true;
  o.ctx.options = opts;
  o.ctx.progress = [](const Json& j) { emit(j); };
  Json out = nxr::exporters::export_gifti(a.get("--package"), o);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}
}  // namespace

int export_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro export: error: the following arguments are required: what");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  if (what == "gifti") return gifti_command(rest);
  throw Usage("nxr-neuro export: error: argument what: invalid choice: '" + what + "' (choose from 'gifti')");
}

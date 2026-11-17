// nxr-neuro export <what>: stored results written as the field's standard files (docs/specs/2026-11-17-interoperability.md, sections 3.3 and 5).
//
//   nxr-neuro export bids-derivatives --package PATH --out DERIVATIVES_FOLDER --results REGEX [--space NAME] [--session S] [--task T] [--run N] [--from S --to S | --all] [--with-surface] [--stc] [--measurements P:F:m1,m2]
//   nxr-neuro export fif --package PATH --field NAME --out STEM [--from S --to S | --all]      a source estimate (-lh.stc, -rh.stc)
//   nxr-neuro export fif --package PATH --covariance NAME --out STEM                           a covariance statistic (-cov.fif)
//   nxr-neuro export tsv --package PATH --out STEM (--statistic NAME | --partition NAME --field NAME --measures a,b,...)
//   nxr-neuro export gifti --package PATH (--field NAME | --labelling NAME) --out STEM [--kind func|shape] [--with-surface] [--hemispheres] [--surface-unit m|mm] [--from S --to S | --all]
//
// The command parses and prints; io's exporters (nxr/exporters/) do the work. A series needs its time range (--from and --to in seconds, or --all): there is no default.
// Progress and results are JSON lines.
#include <cstdlib>

#include "cli.h"
#include "nxr/exporters/bids.h"
#include "nxr/exporters/fif.h"
#include "nxr/exporters/gifti.h"
#include "nxr/exporters/tsv.h"

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
int bids_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export bids-derivatives";
  const Args a = parse(prog, rest, {"--package", "--out", "--results", "--space", "--session", "--task", "--run", "--from", "--to", "--measurements"}, {"--with-surface", "--stc", "--all"},
                       {"--package", "--out", "--results"});
  nxr::exporters::BidsDerivativesExport o;
  o.root = a.get("--out");
  o.results = a.get("--results");
  o.space = a.maybe("--space");
  o.session = a.maybe("--session");
  o.task = a.maybe("--task");
  o.run = a.maybe("--run");
  o.with_surface = a.flag.count("--with-surface") > 0;
  o.stc = a.flag.count("--stc") > 0;
  o.measurements = a.maybe("--measurements").value_or("");
  const bool from = a.maybe("--from").has_value(), to = a.maybe("--to").has_value(), all = a.flag.count("--all") > 0;
  if (from != to) throw Usage(prog + ": error: --from and --to go together: give both");
  if (all && from) throw Usage(prog + ": error: give one time range: --from and --to, or --all");
  if (from) o.range = nxr::exporters::SeriesRange{false, seconds_of(prog, "--from", a.get("--from")), seconds_of(prog, "--to", a.get("--to"))};
  if (all) o.range = nxr::exporters::SeriesRange{true, 0, 0};
  o.ctx.version = NXR_NEURO_VERSION;
  o.ctx.command = command_line("bids-derivatives", rest);
  Json opts = Json::object();
  for (const auto& kv : a.opt) opts[kv.first.substr(2)] = kv.second;
  for (const auto& f : a.flag) opts[f.substr(2)] = true;
  o.ctx.options = opts;
  o.ctx.progress = [](const Json& j) { emit(j); };
  Json out = nxr::exporters::export_bids_derivatives(a.get("--package"), o);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

int fif_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export fif";
  const Args a = parse(prog, rest, {"--package", "--field", "--covariance", "--forward", "--inverse", "--out", "--from", "--to"}, {"--all"}, {"--package", "--out"});
  if (a.maybe("--forward") || a.maybe("--inverse"))
    throw Usage(prog + ": error: a forward solution and an inverse operator are not written: an MNE forward needs each channel's FIF info (kind, coil type, position), which a package does not keep, and an MNE inverse operator is the eigen-decomposition of the whitened forward, which a stored kernel does not hold");
  if (a.maybe("--covariance") && a.maybe("--field")) throw Usage(prog + ": error: give a field (--field) or a covariance (--covariance), not both");
  if (!a.maybe("--covariance") && !a.maybe("--field")) throw Usage(prog + ": error: the following arguments are required: --field or --covariance");
  if (a.maybe("--covariance")) {
    if (a.maybe("--from") || a.maybe("--to") || a.flag.count("--all")) throw Usage(prog + ": error: a covariance has no time range");
    nxr::exporters::CovarianceExport c;
    c.statistic = a.get("--covariance");
    c.out = a.get("--out");
    c.ctx.version = NXR_NEURO_VERSION;
    c.ctx.command = command_line("fif", rest);
    Json copts = Json::object();
    for (const auto& kv : a.opt) copts[kv.first.substr(2)] = kv.second;
    c.ctx.options = copts;
    c.ctx.progress = [](const Json& j) { emit(j); };
    Json cout_ = nxr::exporters::export_covariance(a.get("--package"), c);
    Json cline = Json{{"stage", "result"}};
    for (auto it = cout_.begin(); it != cout_.end(); ++it) cline[it.key()] = it.value();
    emit(cline);
    return 0;
  }
  if (!a.maybe("--field")) throw Usage(prog + ": error: --field is required");
  nxr::exporters::StcExport o;
  o.field = a.get("--field");
  o.out = a.get("--out");
  const bool from = a.maybe("--from").has_value(), to = a.maybe("--to").has_value(), all = a.flag.count("--all") > 0;
  if (from != to) throw Usage(prog + ": error: --from and --to go together: give both");
  if (all && from) throw Usage(prog + ": error: give one time range: --from and --to, or --all");
  if (from) o.range = nxr::exporters::SeriesRange{false, seconds_of(prog, "--from", a.get("--from")), seconds_of(prog, "--to", a.get("--to"))};
  if (all) o.range = nxr::exporters::SeriesRange{true, 0, 0};
  o.ctx.version = NXR_NEURO_VERSION;
  o.ctx.command = command_line("fif", rest);
  Json opts = Json::object();
  for (const auto& kv : a.opt) opts[kv.first.substr(2)] = kv.second;
  for (const auto& f : a.flag) opts[f.substr(2)] = true;
  o.ctx.options = opts;
  o.ctx.progress = [](const Json& j) { emit(j); };
  Json out = nxr::exporters::export_stc(a.get("--package"), o);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

int tsv_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export tsv";
  const Args a = parse(prog, rest, {"--package", "--out", "--statistic", "--partition", "--field", "--measures"}, {}, {"--package", "--out"});
  nxr::exporters::TsvExport o;
  o.statistic = a.maybe("--statistic").value_or("");
  o.partition = a.maybe("--partition").value_or("");
  o.field = a.maybe("--field").value_or("");
  if (const auto m = a.maybe("--measures")) {
    std::string cur;
    for (const char c : *m + ",") { if (c == ',') { if (!cur.empty()) o.measures.push_back(cur); cur.clear(); } else cur += c; }
  }
  if (o.statistic.empty() == (o.partition.empty() && o.field.empty() && o.measures.empty()))
    throw Usage(prog + ": error: give either --statistic, or --partition with --field and --measures");
  o.out = a.get("--out");
  o.ctx.version = NXR_NEURO_VERSION;
  o.ctx.command = command_line("tsv", rest);
  Json opts = Json::object();
  for (const auto& kv : a.opt) opts[kv.first.substr(2)] = kv.second;
  o.ctx.options = opts;
  o.ctx.progress = [](const Json& j) { emit(j); };
  Json out = nxr::exporters::export_tsv(a.get("--package"), o);
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
  if (what == "tsv") return tsv_command(rest);
  if (what == "fif") return fif_command(rest);
  if (what == "bids-derivatives") return bids_command(rest);
  throw Usage("nxr-neuro export: error: argument what: invalid choice: '" + what + "' (choose from 'gifti', 'tsv', 'fif', 'bids-derivatives')");
}

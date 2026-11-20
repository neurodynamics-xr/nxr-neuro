// nxr-neuro export <what>: stored results written as the field's standard files (docs/specs/2026-11-17-interoperability.md, sections 3.3 and 5).
//
//   nxr-neuro export bids-derivatives --package PATH --out DERIVATIVES_FOLDER --results REGEX [--space NAME] [--session S] [--task T] [--run N] [--from S --to S | --all] [--with-surface] [--stc] [--measurements P:F:m1,m2]
//   nxr-neuro export fif --package PATH --field NAME --out STEM [--from S --to S | --all]      a source estimate (-lh.stc, -rh.stc)
//   nxr-neuro export fif --package PATH --covariance NAME --out STEM                           a covariance statistic (-cov.fif)
//   nxr-neuro export tsv --package PATH --out STEM (--statistic NAME | --partition NAME --field NAME --measures a,b,...)
//   nxr-neuro export gifti --package PATH (--field NAME | --labelling NAME) --out STEM [--kind func|shape] [--with-surface] [--hemispheres] [--surface-unit m|mm] [--from S --to S | --all]
//
// The command parses, calls its library function (nxr/io/task/commands.h) once and prints; io's exporters (nxr/exporters/) do the work. A series needs its time range (--from and --to in seconds, or --all): there is no default.
// Progress and results are JSON lines.
#include <cstdlib>

#include "cli.h"

using namespace nxr::io;
using namespace cli;
namespace task = nxr::io::task;

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

// the time range as given: --from and --to in seconds (or m:ss), or --all
task::ExportRange range_of(const std::string& prog, const Args& a) {
  task::ExportRange r;
  if (const auto v = a.maybe("--from")) r.from = seconds_of(prog, "--from", *v);
  if (const auto v = a.maybe("--to")) r.to = seconds_of(prog, "--to", *v);
  r.all = a.flag.count("--all") > 0;
  return r;
}

// what the export records of this command: the version, the command as typed, its options (flags too, unless `flags` is false)
task::ExportProvenance provenance_of(const std::string& what, const std::vector<std::string>& rest, const Args& a, bool flags = true) {
  task::ExportProvenance p;
  p.version = NXR_NEURO_VERSION;
  p.command = command_line(what, rest);
  for (const auto& kv : a.opt) p.options[kv.first.substr(2)] = kv.second;
  if (flags) for (const auto& f : a.flag) p.options[f.substr(2)] = true;
  return p;
}

int gifti_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export gifti";
  const Args a = parse(prog, rest, {"--package", "--field", "--labelling", "--out", "--kind", "--surface-unit", "--from", "--to"}, {"--with-surface", "--hemispheres", "--all"}, {"--package", "--out"});
  task::ExportGiftiRequest rq;
  rq.package = a.get("--package");
  rq.out = a.get("--out");
  rq.field = a.maybe("--field");
  rq.labelling = a.maybe("--labelling");
  rq.kind = a.maybe("--kind");
  rq.surface_unit = a.maybe("--surface-unit");
  rq.with_surface = a.flag.count("--with-surface") > 0;
  rq.hemispheres = a.flag.count("--hemispheres") > 0;
  rq.range = range_of(prog, a);
  rq.provenance = provenance_of("gifti", rest, a);
  emit_result(call(prog, [&] { return task::export_gifti(rq, progress()); }).result);
  return 0;
}

int bids_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export bids-derivatives";
  const Args a = parse(prog, rest, {"--package", "--out", "--results", "--space", "--session", "--task", "--run", "--from", "--to", "--measurements"}, {"--with-surface", "--stc", "--all"},
                       {"--package", "--out", "--results"});
  task::ExportBidsDerivativesRequest rq;
  rq.package = a.get("--package");
  rq.out = a.get("--out");
  rq.results = a.get("--results");
  rq.space = a.maybe("--space");
  rq.session = a.maybe("--session");
  rq.task = a.maybe("--task");
  rq.run = a.maybe("--run");
  rq.measurements = a.maybe("--measurements");
  rq.with_surface = a.flag.count("--with-surface") > 0;
  rq.stc = a.flag.count("--stc") > 0;
  rq.range = range_of(prog, a);
  rq.provenance = provenance_of("bids-derivatives", rest, a);
  emit_result(call(prog, [&] { return task::export_bids_derivatives(rq, progress()); }).result);
  return 0;
}

int fif_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export fif";
  const Args a = parse(prog, rest, {"--package", "--field", "--covariance", "--forward", "--inverse", "--out", "--from", "--to"}, {"--all"}, {"--package", "--out"});
  task::ExportFifRequest rq;
  rq.package = a.get("--package");
  rq.out = a.get("--out");
  rq.field = a.maybe("--field");
  rq.covariance = a.maybe("--covariance");
  rq.forward = a.maybe("--forward");
  rq.inverse = a.maybe("--inverse");
  rq.range = range_of(prog, a);
  rq.provenance = provenance_of("fif", rest, a, !rq.covariance);   // a covariance's record lists the valued options only, as it always did
  emit_result(call(prog, [&] { return task::export_fif(rq, progress()); }).result);
  return 0;
}

int tsv_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro export tsv";
  const Args a = parse(prog, rest, {"--package", "--out", "--statistic", "--partition", "--field", "--measures"}, {}, {"--package", "--out"});
  task::ExportTsvRequest rq;
  rq.package = a.get("--package");
  rq.out = a.get("--out");
  rq.statistic = a.maybe("--statistic");
  rq.partition = a.maybe("--partition");
  rq.field = a.maybe("--field");
  if (const auto m = a.maybe("--measures")) rq.measures = split_commas(*m);
  rq.provenance = provenance_of("tsv", rest, a);
  emit_result(call(prog, [&] { return task::export_tsv(rq, progress()); }).result);
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

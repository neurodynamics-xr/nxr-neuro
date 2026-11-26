// The export commands' library functions (nxr/io/task/commands.h, "export"): stored results written as the field's standard files by io's exporters
// (nxr/exporters/), moved out of cli/ so the command line and the faces call one function. Behaviour, refusals and results are the command line's, unchanged.
#include "nxr/exporters/bids.h"
#include "nxr/exporters/fif.h"
#include "nxr/exporters/gifti.h"
#include "nxr/exporters/tsv.h"
#include "nxr/io/task/commands.h"

namespace nxr::io::task {

namespace {

nxr::exporters::Context context_of(const ExportProvenance& p, const Progress& progress) {
  nxr::exporters::Context c;
  c.version = p.version;
  c.command = p.command;
  c.options = p.options;
  c.progress = progress;
  return c;
}

// a series' time range: from and to (both), or all, or none (a map); A SERIES NEEDS ONE: there is no default
std::optional<nxr::exporters::SeriesRange> range_of(const ExportRange& r) {
  const bool from = r.from.has_value(), to = r.to.has_value();
  if (from != to) throw Refused("--from and --to go together: give both");
  if (r.all && from) throw Refused("give one time range: --from and --to, or --all");
  if (from) return nxr::exporters::SeriesRange{false, *r.from, *r.to};
  if (r.all) return nxr::exporters::SeriesRange{true, 0, 0};
  return std::nullopt;
}

CommandResult result_of(Json body) {
  CommandResult r;
  r.result = std::move(body);
  return r;
}

}  // namespace

CommandResult export_gifti(const ExportGiftiRequest& rq, const Progress& progress) {
  nxr::exporters::GiftiExport o;
  o.field = rq.field.value_or("");
  o.labelling = rq.labelling.value_or("");
  if (o.field.empty() == o.labelling.empty()) throw Refused("give exactly one of --field and --labelling");
  o.out = rq.out;
  o.kind = rq.kind.value_or("func");
  o.surface = rq.with_surface;
  o.hemispheres = rq.hemispheres;
  o.surface_unit = rq.surface_unit;
  o.range = range_of(rq.range);
  o.ctx = context_of(rq.provenance, progress);
  return result_of(nxr::exporters::export_gifti(rq.package, o));
}

CommandResult export_bids_derivatives(const ExportBidsDerivativesRequest& rq, const Progress& progress) {
  nxr::exporters::BidsDerivativesExport o;
  o.root = rq.out;
  o.results = rq.results;
  o.space = rq.space;
  o.session = rq.session;
  o.task = rq.task;
  o.run = rq.run;
  o.with_surface = rq.with_surface;
  o.stc = rq.stc;
  o.measurements = rq.measurements.value_or("");
  o.range = range_of(rq.range);
  o.ctx = context_of(rq.provenance, progress);
  return result_of(nxr::exporters::export_bids_derivatives(rq.package, o));
}

CommandResult export_fif(const ExportFifRequest& rq, const Progress& progress) {
  if (rq.forward || rq.inverse)
    throw Refused("a forward solution and an inverse operator are not written: an MNE forward needs each channel's FIF info (kind, coil type, position), which a package does not keep, and an MNE inverse operator is the eigen-decomposition of the whitened forward, which a stored kernel does not hold");
  if (rq.covariance && rq.field) throw Refused("give a field (--field) or a covariance (--covariance), not both");
  if (!rq.covariance && !rq.field) throw Refused("the following arguments are required: --field or --covariance");
  if (rq.covariance) {
    if (rq.range.from || rq.range.to || rq.range.all) throw Refused("a covariance has no time range");
    nxr::exporters::CovarianceExport c;
    c.statistic = *rq.covariance;
    c.out = rq.out;
    c.ctx = context_of(rq.provenance, progress);
    return result_of(nxr::exporters::export_covariance(rq.package, c));
  }
  nxr::exporters::StcExport o;
  o.field = *rq.field;
  o.out = rq.out;
  o.range = range_of(rq.range);
  o.ctx = context_of(rq.provenance, progress);
  return result_of(nxr::exporters::export_stc(rq.package, o));
}

CommandResult export_tsv(const ExportTsvRequest& rq, const Progress& progress) {
  nxr::exporters::TsvExport o;
  o.statistic = rq.statistic.value_or("");
  o.partition = rq.partition.value_or("");
  o.field = rq.field.value_or("");
  if (rq.measures) o.measures = *rq.measures;
  if (o.statistic.empty() == (o.partition.empty() && o.field.empty() && o.measures.empty()))
    throw Refused("give either --statistic, or --partition with --field and --measures");
  o.out = rq.out;
  o.ctx = context_of(rq.provenance, progress);
  return result_of(nxr::exporters::export_tsv(rq.package, o));
}

}  // namespace nxr::io::task

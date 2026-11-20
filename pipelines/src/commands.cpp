// nxr-pipelines — the commands that need compute (face.h, "THE COMMANDS THAT NEED COMPUTE"): what each command of the command line does, moved out of cli/ so the
// command line and the faces call one function. Behaviour, refusals and results are the command line's, unchanged.
#include <cmath>
#include <complex>

#include "nxr/io/catalogue.h"
#include "nxr/io/errors.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/dataset.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/registration.h"
#include "nxr/signal/crossspectral.h"

using namespace nxr::io;
namespace fs = std::filesystem;
using nxr::io::task::Refused;

namespace nxr::pipelines::face {

namespace {

TemplateSearch search_of(const std::vector<std::string>& templates) {
  TemplateSearch s;
  for (const auto& p : templates) s.command_paths.push_back(p);
  return s;
}

// the subject's dataset as every command opens it (a catalogue's subject package, or a one-database dataset; nxr/io/task/commands.h)
DatasetWriter open_subject_dataset(const std::string& folder, const std::string& subject, const OpenOptions& o, const std::vector<std::string>& templates, const Progress& progress) {
  if (is_catalogue_dataset(folder)) return open_subject_package(folder, subject, o, search_of(templates));
  if (progress && fs::is_regular_file(fs::path(folder) / "dataset.sqlite"))
    progress(Json{{"stage", "note"}, {"note", folder + " is a one-database dataset; `nxr-neuro split dataset` makes it a catalogue of subject packages (the default form)."}});
  return open_writer(folder, o);
}

CommandResult list_nodes(const ListNodesRequest& rq, const std::string& what, const Progress& progress) {
  OpenOptions o;
  o.lock = false;
  DatasetWriter ds = open_subject_dataset(rq.dataset, rq.subject, o, rq.templates, progress);
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject);
  CommandResult r;
  r.result = Json{{"subject", sub.name()}, {what, list_subject_nodes(ds, sub, what)}};
  return r;
}

}  // namespace

CommandResult create_dataset(const nxr::io::task::CreateDatasetRequest& rq, const Progress& progress) {
  if (rq.dataset || !rq.datastore || !rq.templates) return nxr::io::task::create_dataset(rq, progress);
  // THE DEFAULT SUBJECT, with the dataset: both or neither (the library function removes the dataset it made when a step fails)
  DatasetWithTemplateOptions w;
  w.dataset = nxr::io::task::one_database_options(rq, progress);
  w.subject.templates = *rq.templates;
  if (rq.default_) w.subject.default_resolution = *rq.default_;
  if (rq.atlas_finest) w.atlas_finest = *rq.atlas_finest;     // a quantity (parts per hemisphere, a tile diameter, a tile area): parsed where the grid is made
  if (rq.lc_scales) w.lc_scales = *rq.lc_scales;              // the tile scales of the Levi-Civita measurement ("parts 16", "parts 64")
  if (rq.modes) {                                              // the number of Hodge–Dirac modes to solve (as `run eigenmodes --modes`)
    if (*rq.modes < 1) throw Refused("--modes is a whole number of at least 1");
    w.modes = *rq.modes;
  }
  CommandResult r;
  r.result = create_dataset_with_template(*rq.datastore, *rq.name, w, rq.created_by, progress);
  return r;
}

CommandResult create_registration(const CreateRegistrationRequest& rq, const Progress& progress) {
  RegistrationCreateOptions o;
  o.package = rq.package;
  o.name = rq.name;
  o.from = rq.from;
  o.to = rq.to;
  o.from_sphere = rq.from_sphere;
  o.to_sphere = rq.to_sphere;
  if (rq.method) o.method = *rq.method;
  if (rq.neighbours) o.neighbours = *rq.neighbours;
  o.both_directions = !rq.one_direction;
  for (const auto& p : rq.templates) o.templates.push_back(p);
  if (rq.home_dataset) o.home_dataset = *rq.home_dataset;
  o.created_by = rq.created_by;
  o.source = rq.source;
  o.licence = rq.licence;
  o.attribution = rq.attribution;
  if (rq.redistributable) {
    if (*rq.redistributable != "yes" && *rq.redistributable != "no") throw Refused("argument --redistributable: invalid choice: '" + *rq.redistributable + "' (choose from 'yes', 'no')");
    o.redistributable = *rq.redistributable == "yes";
  }
  const auto res = registration_create(o, progress);
  CommandResult r;
  r.result = res.info;
  r.result["package"] = res.package.string();
  r.result["content_hash"] = SubjectPackage::open(res.package).record().content_hash;
  return r;
}

CommandResult list_recordings(const ListNodesRequest& rq, const Progress& progress) { return list_nodes(rq, "recordings", progress); }
CommandResult list_kernels(const ListNodesRequest& rq, const Progress& progress) { return list_nodes(rq, "kernels", progress); }
CommandResult list_results(const ListNodesRequest& rq, const Progress& progress) { return list_nodes(rq, "results", progress); }

namespace {
CommandResult read_crossspectrum_checked(const ReadCrossspectrumRequest& rq, const Progress& progress);
}  // namespace

CommandResult read_crossspectrum(const ReadCrossspectrumRequest& rq, const Progress& progress) {
  const std::string& reading = rq.reading;
  if (reading != "coherency" && reading != "imag" && reading != "phase" && reading != "delay" && reading != "psi") throw Refused("--reading is coherency, imag, phase, delay or psi");
  if (rq.weights != "coherence" && rq.weights != "none") throw Refused("--weights is coherence or none");
  if (rq.pair && rq.seed) throw Refused("--pair and --seed are alternatives");
  if (!rq.pair && !rq.seed) throw Refused("name the channels to read: --pair A,B (row A, column B; recording channel indices) or --seed A (A against every channel)");
  try {
    return read_crossspectrum_checked(rq, progress);
  } catch (const Refused&) {
    throw;
  } catch (const std::invalid_argument& e) {   // a refusal of the request by the readers it calls: the command's usage error, as before
    throw Refused(e.what());
  }
}

namespace {
CommandResult read_crossspectrum_checked(const ReadCrossspectrumRequest& rq, const Progress& progress) {
  const std::string& reading = rq.reading;
  DatasetWriter ds = open_subject_dataset(rq.dataset, rq.subject, OpenOptions{}, rq.templates, progress);
  OpenSubjectOptions oo;
  oo.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject, oo);
  const auto S = crossspectrum_bins_read(sub, rq.name);
  auto position = [&](std::int32_t ch) {
    for (std::size_t i = 0; i < S.channels.size(); ++i) if (S.channels[i] == ch) return static_cast<Eigen::Index>(i);
    std::string have;
    for (std::size_t i = 0; i < S.channels.size() && i < 8; ++i) have += (i ? "," : "") + std::to_string(S.channels[i]);
    throw Refused("channel " + std::to_string(ch) + " is not one of the statistic's " + std::to_string(S.channels.size()) + " channels (recording channel indices, from " + have +
                  (S.channels.size() > 8 ? ", ..." : "") + ")");
  };
  std::vector<Eigen::Index> rows, cols;                         // the pairs (rows[i], cols[i]) wanted
  if (rq.pair) {
    rows.push_back(position((*rq.pair)[0])); cols.push_back(position((*rq.pair)[1]));
  } else {
    const Eigen::Index row = position(*rq.seed);
    for (std::size_t j = 0; j < S.channels.size(); ++j) { rows.push_back(row); cols.push_back(static_cast<Eigen::Index>(j)); }
  }
  const Eigen::VectorXd f = Eigen::Map<const Eigen::VectorXd>(S.frequencies.data(), static_cast<Eigen::Index>(S.frequencies.size()));
  Json out = Json{{"name", rq.name}, {"reading", reading}, {"frequencies_hz", S.frequencies}, {"sign", "positive delay and psi: the row channel leads the column channel"}};
  Json chans = Json::array(), vals = Json::array();
  for (std::size_t i = 0; i < rows.size(); ++i) chans.push_back(Json::array({S.channels[static_cast<std::size_t>(rows[i])], S.channels[static_cast<std::size_t>(cols[i])]}));
  out["pairs"] = chans;
  if (reading == "delay") {
    nxr::signal::PhaseSlopeOptions po;
    po.weights = rq.weights == "none" ? nxr::signal::PhaseSlopeOptions::Weights::None : nxr::signal::PhaseSlopeOptions::Weights::Coherence;
    const nxr::signal::PhaseSlope ps = nxr::signal::phaseSlope(S.C, f, po);
    Json res = Json::array();
    for (std::size_t i = 0; i < rows.size(); ++i) { vals.push_back(ps.delayMs(rows[i], cols[i])); res.push_back(ps.residual(rows[i], cols[i])); }
    out["delay_ms"] = vals; out["residual_rad"] = res; out["weights"] = rq.weights;
  } else if (reading == "psi") {
    const Eigen::MatrixXd psi = nxr::signal::phase_slope_index(S.C);
    for (std::size_t i = 0; i < rows.size(); ++i) vals.push_back(psi(rows[i], cols[i]));
    out["psi"] = vals;
  } else {
    const auto H = nxr::signal::coherency(S.C);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      Json re = Json::array(), im = Json::array(), ab = Json::array(), ph = Json::array(), un = Json::array();
      double last = 0;
      for (std::size_t k = 0; k < H.size(); ++k) {
        const std::complex<double> h = H[k](rows[i], cols[i]);
        re.push_back(h.real()); im.push_back(h.imag()); ab.push_back(std::abs(h));
        const double p = std::arg(h);
        ph.push_back(p);
        last = k == 0 ? p : last + std::remainder(p - last, 2 * M_PI);
        un.push_back(last);
      }
      if (reading == "imag") vals.push_back(im);
      else if (reading == "phase") vals.push_back(Json{{"phase_rad", ph}, {"unwrapped_rad", un}});
      else vals.push_back(Json{{"re", re}, {"im", im}, {"abs", ab}});
    }
    out["values"] = vals;
  }
  CommandResult r;
  r.result = std::move(out);
  return r;
}
}  // namespace

CommandResult query_measurements(const QueryMeasurementsRequest& rq) {
  // the face's query: the selection, the measure, the field, the level asked as ONE quantity (a band, a spatial scale or a tile: the quantity names its own chart), where it runs
  // (io: the SQL query; compute: in memory from the finest ingredients) and the subjects it merges
  MeasurementQuery q;
  q.selection = rq.selection;
  q.measure = rq.measure;
  q.of_field = rq.field;
  int quantities = 0;
  for (const auto* v : {&rq.band, &rq.spatial_scale, &rq.tile})
    if (*v) { q.to = **v; ++quantities; }
  if (quantities > 1) throw Refused("give one of --band, --spatial-scale, --tile: the level asked is one quantity");
  q.factor = rq.factor;
  if (rq.subjects) q.subjects = *rq.subjects;
  if (rq.where) { if (*rq.where != "io" && *rq.where != "compute") throw Refused("--where is io or compute"); q.where = *rq.where; }
  // the value filters, the order and the limit are io's query (SQL): they run where io
  q.min_value = rq.min_value;
  q.max_value = rq.max_value;
  if (rq.limit) q.limit = *rq.limit;
  if (rq.order) q.order = *rq.order;
  if ((q.min_value || q.max_value || q.limit > 0 || !q.order.empty()) && !rq.where) q.where = "io";
  CommandResult r;
  try {
    r.result = Json{{"selection", q.selection}, {"measure", q.measure}, {"result", measurement_query(rq.dataset, rq.subject, q)}};
  } catch (const Refused&) {
    throw;
  } catch (const std::invalid_argument& e) {   // the query's own refusals of the request (a level, a factor, a measure): the command's usage errors, as before
    throw Refused(e.what());
  }
  return r;
}

}  // namespace nxr::pipelines::face

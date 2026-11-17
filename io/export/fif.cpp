// Export of MNE objects: see nxr/exporters/fif.h.
#include "nxr/exporters/fif.h"

#include <cmath>
#include <map>
#include <memory>

#include "cortex.h"
#include "nxr/formats/fif_objects.h"
#include "nxr/formats/stc.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/errors.h"

namespace nxr::exporters {

namespace code = nxr::io::code;
namespace stc = nxr::formats::stc;
using namespace nxr::exporters::detail;
using nxr::io::ArrayReader;
using nxr::io::Error;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
}  // namespace

Json export_stc(const std::filesystem::path& package, const StcExport& o) {
  if (o.field.empty()) refuse("name the field to export");
  if (o.out.empty()) refuse("an export names the path stem it writes under");
  const nxr::io::SubjectPackage pkg = nxr::io::SubjectPackage::open(package);
  const task::Subject subject = task::Subject::of(pkg);
  const task::Node input = subject.require({.cls = task::Cls::field, .name = o.field}, "the field '" + o.field + "'");
  if (!input.path || !input.manifold_id) refuse("the field '" + o.field + "' stores no values");
  const std::string vt = input.value_type.value_or("scalar");
  if (vt != "scalar" && vt != "integer") refuse("export fif writes scalar fields; '" + o.field + "' is " + vt + " (name one of its components as a field)");
  const auto factors = subject.factors(*input.manifold_id);
  if (factors.empty() || factors.size() > 2) refuse("the field '" + o.field + "' lies on a product of " + std::to_string(factors.size()) + " manifolds; a source estimate is a map or a series on one cortex");
  const bool series = factors.size() == 2;
  if (series && !o.range) refuse("'" + o.field + "' is a series over time: state the time range to export (from and to in seconds, or all)");
  if (!series && o.range) refuse("'" + o.field + "' is a map: it has no time range");
  const task::Node cortex_node = subject.require_id(task::Cls::manifold, factors[0]);
  const Surface whole = load_surface(pkg, subject, cortex_node);
  if (!whole.hemi.empty()) refuse("'" + cortex_node.name + "' is one hemisphere: a source estimate holds both (the glued cortex with its two hemispheres as components)");
  const std::vector<Surface> hemis = hemispheres_of(pkg, subject, cortex_node);
  if (hemis.size() != 2 || hemis[0].hemi != "L" || hemis[1].hemi != "R") refuse("'" + cortex_node.name + "' needs exactly a left and a right hemisphere to be written as -lh.stc and -rh.stc");
  std::vector<std::int64_t> first_row;
  for (const Surface& h : hemis) first_row.push_back(h.node.fact("from_index").is_number() ? h.node.fact("from_index").get<std::int64_t>() : 0);

  const ArrayReader a = ArrayReader::open(pkg.store(), *input.path);
  if (a.shape().empty() || a.shape()[0] != whole.n || (!series && a.shape().size() > 1 && a.shape()[1] != 1) || (series && a.shape().size() != 2))
    refuse("the field '" + o.field + "' does not have a value per vertex of '" + cortex_node.name + "' (" + std::to_string(whole.n) + " vertices)");

  std::vector<std::vector<std::uint32_t>> vertices(2);
  for (int h = 0; h < 2; ++h) for (std::int64_t i = 0; i < hemis[static_cast<std::size_t>(h)].n; ++i) vertices[static_cast<std::size_t>(h)].push_back(static_cast<std::uint32_t>(i));
  const std::vector<std::string> files = {o.out + "-lh.stc", o.out + "-rh.stc"};
  std::vector<std::unique_ptr<nxr::formats::FileSink>> sinks;
  for (const auto& f : files) sinks.push_back(std::make_unique<nxr::formats::FileSink>(f));

  Json span;
  Json details = Json::object();
  std::int64_t frames = 0;
  auto put = [&](const std::vector<float>& frame, std::vector<std::unique_ptr<stc::Writer>>& writers) {
    for (std::size_t h = 0; h < 2; ++h) writers[h]->frame(frame.data() + first_row[h]);
  };
  std::vector<std::unique_ptr<stc::Writer>> writers;
  if (!series) {
    for (std::size_t h = 0; h < 2; ++h) writers.push_back(std::make_unique<stc::Writer>(*sinks[h], 0.0, 1.0, vertices[h], 1));
    const std::vector<double> v = widen(a.read(), a.dtype());
    put(std::vector<float>(v.begin(), v.end()), writers);
    frames = 1;
    details["time"] = Json{{"tmin_s", 0.0}, {"tstep_s", 1.0}, {"note", "a map: one time point, written with start 0 s and step 1 s"}};
  } else {
    const nxr::io::finder::Value v = subject.describe(input);
    if (!v.time || v.rate <= 0) refuse("the field '" + o.field + "' has no time line with a rate: it is not a series");
    const Json chunk = input.fact("chunk_shape_json").is_string() ? Json::parse(input.fact("chunk_shape_json").get<std::string>(), nullptr, false) : input.fact("chunk_shape_json");
    const std::int64_t width = nxr::io::chunk_width(chunk);
    nxr::io::SpanRequest rq;
    if (o.range->all) rq.chunks = "all";
    else {
      if (!(o.range->to > o.range->from)) refuse("the time range must end after it starts (from " + std::to_string(o.range->from) + " s, to " + std::to_string(o.range->to) + " s)");
      rq.start = o.range->from;
      rq.stop = o.range->to;
    }
    const nxr::io::Span sp = nxr::io::resolve_span(rq, a.shape()[1], v.rate, v.origin, width);
    const std::int64_t n_times = sp.s1 - sp.s0;
    if (n_times <= 0) refuse("the time range holds no sample of '" + o.field + "'");
    span = nxr::io::span_record(sp, v.rate, v.origin);
    const double tmin = v.origin + static_cast<double>(sp.s0) / v.rate;
    details["time"] = Json{{"rate_hz", v.rate}, {"origin_s", v.origin}, {"tmin_s", tmin}, {"tstep_s", 1.0 / v.rate}};
    for (std::size_t h = 0; h < 2; ++h) writers.push_back(std::make_unique<stc::Writer>(*sinks[h], tmin, 1.0 / v.rate, vertices[h], static_cast<std::uint32_t>(n_times)));
    const nxr::io::ChunkSource src(pkg.store(), *input.path, width, v.rate, v.origin);
    nxr::io::chunk_loop(src, sp, [&](const nxr::io::Chunk& c) {
      std::vector<float> frame(static_cast<std::size_t>(whole.n));
      for (std::int64_t j = 0; j < c.n; ++j) {
        for (std::int64_t r = 0; r < whole.n; ++r) frame[static_cast<std::size_t>(r)] = static_cast<float>(c.data[static_cast<std::size_t>(r * c.n + j)]);
        put(frame, writers);
        ++frames;
      }
      if (o.ctx.progress) o.ctx.progress(Json{{"stage", "frames"}, {"chunk", c.k}, {"frames", frames}});
    });
  }
  for (auto& w : writers) w->finish();
  for (auto& s : sinks) s->close();
  if (a.dtype() != nxr::io::DType::Float32) details["converted"] = Json::array({std::string(nxr::io::dtype_to_string(a.dtype())) + " -> float32 (an .stc file holds float32 only)"});
  details["frames"] = frames;
  details["vertices"] = Json{{"left", hemis[0].n}, {"right", hemis[1].n}, {"numbering", "every vertex of each hemisphere's surface, from 0"}};
  details["unit"] = input.unit ? Json(*input.unit) : Json();
  details["surface"] = cortex_node.name;
  const std::vector<task::Node> inputs = {input, cortex_node};
  const Json prov = provenance(pkg, subject, o.ctx, inputs, span, details);
  const std::string sidecar = write_sidecar(files.front(), prov, files);
  return Json{{"files", files}, {"sidecar", sidecar}, {"frames", frames}, {"details", details}, {"terms", terms_of(subject, inputs).to_json()}};
}

Json export_covariance(const std::filesystem::path& package, const CovarianceExport& o) {
  namespace fif = nxr::formats::fif;
  if (o.statistic.empty()) refuse("name the covariance statistic to export");
  if (o.out.empty()) refuse("an export names the path stem it writes under");
  const nxr::io::SubjectPackage pkg = nxr::io::SubjectPackage::open(package);
  const task::Subject subject = task::Subject::of(pkg);
  const task::Node st = subject.require({.cls = task::Cls::statistic, .name = o.statistic}, "the statistic '" + o.statistic + "'");
  const Json measure = st.fact("measure");
  if (!measure.is_string() || measure.get<std::string>() != "covariance") refuse("'" + o.statistic + "' is a '" + (measure.is_string() ? measure.get<std::string>() : std::string("?")) + "' statistic: export fif --covariance writes a covariance");
  if (!st.path) refuse("the statistic '" + o.statistic + "' stores no values");
  if (st.value_type.value_or("scalar") != "scalar") refuse("'" + o.statistic + "' is a " + *st.value_type + " statistic: only a real covariance is written");
  const ArrayReader a = ArrayReader::open(pkg.store(), *st.path);
  if (a.shape().size() != 2 || a.shape()[0] != a.shape()[1]) refuse("'" + o.statistic + "' is not a square [channels x channels] matrix");
  const std::int64_t M = a.shape()[0];
  Json axes = st.fact("axes_json");
  if (axes.is_string()) axes = Json::parse(axes.get<std::string>(), nullptr, false);
  if (!axes.is_array() || axes.empty() || !axes[0].contains("selection_id") || !axes[0]["selection_id"].is_string()) refuse("'" + o.statistic + "': its first axis does not name the channels it covers (a Selection)");
  const task::Node sel = subject.require_id(task::Cls::selection, axes[0]["selection_id"].get<std::string>());
  if (!sel.manifold_id) refuse("the channel Selection '" + sel.name + "' lies on no manifold");
  // the channel names: the names labelling of the channels' manifold (one level per channel, code = the channel's index)
  std::map<std::int64_t, std::string> names;
  for (const task::Node& cand : subject.find_all({.cls = task::Cls::selection, .manifold_id = *sel.manifold_id}))
    if (cand.dictionary_id && cand.name.size() > 6 && cand.name.compare(cand.name.size() - 6, 6, "_names") == 0) {
      for (const auto& e : subject.dictionary_entries(*cand.dictionary_id)) names[e.code] = e.name;
      break;
    }
  if (names.empty()) refuse("the channels of '" + o.statistic + "' have no names labelling on their manifold: a covariance file names its channels");
  std::vector<std::int64_t> index(static_cast<std::size_t>(M));
  if (sel.array_path) {
    const ArrayReader ia = ArrayReader::open(pkg.store(), *sel.array_path);
    const std::vector<double> v = widen(ia.read(), ia.dtype());
    if (static_cast<std::int64_t>(v.size()) != M) refuse("the channel Selection '" + sel.name + "' holds " + std::to_string(v.size()) + " channels, the covariance has " + std::to_string(M));
    for (std::int64_t i = 0; i < M; ++i) index[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(v[static_cast<std::size_t>(i)]);
  } else {
    for (std::int64_t i = 0; i < M; ++i) index[static_cast<std::size_t>(i)] = i;
  }
  fif::Covariance c;
  c.kind = fif::mne::kNoiseCov;
  c.dim = static_cast<std::int32_t>(M);
  for (const std::int64_t i : index) {
    const auto it = names.find(i);
    if (it == names.end()) refuse("channel " + std::to_string(i) + " of '" + o.statistic + "' has no name");
    c.names.push_back(it->second);
  }
  c.data = widen(a.read(), a.dtype());
  const Json dof = st.fact("dof");
  if (dof.is_number() && dof.get<double>() >= 1) c.nfree = static_cast<std::int32_t>(std::llround(dof.get<double>()));
  const std::string file = o.out + "-cov.fif";
  {
    nxr::formats::FileSink sink(file);
    fif::write_covariance_file(sink, c, fif::make_id(0, 0));
    sink.close();
  }
  Json details = Json::object();
  details["channels"] = M;
  details["nfree"] = c.nfree > 0 ? Json(c.nfree) : Json();
  details["unit"] = st.unit ? Json(*st.unit) : Json();
  details["kind"] = "noise covariance";
  details["method"] = st.method ? Json(*st.method) : Json();
  if (a.dtype() != nxr::io::DType::Float64) details["converted"] = Json::array({std::string(nxr::io::dtype_to_string(a.dtype())) + " -> float64 (a covariance is written in double)"});
  const std::vector<task::Node> inputs = {st, sel};
  const Json prov = provenance(pkg, subject, o.ctx, inputs, st.fact("span_json"), details);
  const std::string sidecar = write_sidecar(file, prov, {file});
  return Json{{"files", Json::array({file})}, {"sidecar", sidecar}, {"channels", M}, {"details", details}, {"terms", terms_of(subject, inputs).to_json()}};
}

}  // namespace nxr::exporters

// Export of MNE objects: see nxr/exporters/fif.h.
#include "nxr/exporters/fif.h"

#include <cmath>
#include <memory>

#include "cortex.h"
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

}  // namespace nxr::exporters

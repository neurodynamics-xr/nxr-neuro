// Export of cortical fields and labellings as GIFTI: see nxr/exporters/gifti.h.
#include "nxr/exporters/gifti.h"

#include "cortex.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "nxr/formats/gifti.h"
#include "nxr/formats/tsv.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/errors.h"

namespace nxr::exporters {

namespace gifti = nxr::formats::gifti;
namespace code = nxr::io::code;
using nxr::io::ArrayReader;
using nxr::io::Error;

using namespace nxr::exporters::detail;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

std::string json_text(const Json& j) { return j.dump(-1, ' ', false); }

double metres_per(const std::string& unit) {
  if (unit == "m") return 1.0;
  if (unit == "cm") return 1e-2;
  if (unit == "mm") return 1e-3;
  refuse("the surface's coordinate unit is '" + unit + "'; a conversion to m or mm needs it stated as m, cm or mm");
}

void add_meta(std::vector<std::pair<std::string, std::string>>& m, const std::string& k, const std::string& v) { m.emplace_back(k, v); }

std::vector<std::pair<std::string, std::string>> file_meta(const GiftiExport& o, const Json& producer, const Terms& terms, const Json& span, const std::string& structure) {
  std::vector<std::pair<std::string, std::string>> m;
  add_meta(m, "nxr-neuro:version", o.ctx.version);
  add_meta(m, "nxr-neuro:command", o.ctx.command);
  add_meta(m, "nxr-neuro:producer", json_text(producer));
  add_meta(m, "nxr-neuro:origin", json_text(terms.to_json()));
  add_meta(m, "nxr-neuro:licence", terms.licence);
  add_meta(m, "nxr-neuro:attribution", terms.attribution);
  if (!span.is_null()) add_meta(m, "nxr-neuro:span", json_text(span));
  if (!structure.empty()) add_meta(m, "AnatomicalStructurePrimary", structure);
  return m;
}

// the surface as a .surf.gii
void write_surface(const std::string& path, const GiftiExport& o, const Surface& s, double scale, const Json& producer, const Terms& terms) {
  gifti::Image img;
  img.metadata = file_meta(o, producer, terms, Json(), s.structure);
  add_meta(img.metadata, "nxr-neuro:coordinate_unit", o.surface_unit ? *o.surface_unit : s.coordinate_unit);
  std::vector<float> xyz(s.xyz.size());
  for (std::size_t i = 0; i < xyz.size(); ++i) xyz[i] = static_cast<float>(s.xyz[i] * scale);
  gifti::DataArray p = gifti::make_float32("NIFTI_INTENT_POINTSET", {s.n, 3}, std::move(xyz));
  p.encoding = gifti::Encoding::gzip_base64;
  add_meta(p.metadata, "GeometricType", "Anatomical");
  if (!s.structure.empty()) add_meta(p.metadata, "AnatomicalStructurePrimary", s.structure);
  add_meta(p.metadata, "Name", s.node.name);
  gifti::DataArray t = gifti::make_int32("NIFTI_INTENT_TRIANGLE", {static_cast<std::int64_t>(s.faces.size() / 3), 3}, s.faces);
  t.encoding = gifti::Encoding::gzip_base64;
  img.arrays.push_back(std::move(p));
  img.arrays.push_back(std::move(t));
  gifti::write_file(path, img);
}

}  // namespace

Json export_gifti(const std::filesystem::path& package, const GiftiExport& o) {
  if (o.field.empty() == o.labelling.empty()) refuse("name exactly one of a field and a labelling to export");
  if (o.out.empty()) refuse("an export names the path stem it writes under");
  if (o.kind != "func" && o.kind != "shape") refuse("kind is 'func' or 'shape', not '" + o.kind + "'");
  if (o.surface_unit && *o.surface_unit != "m" && *o.surface_unit != "mm") refuse("the surface unit is 'm' or 'mm', not '" + *o.surface_unit + "'");
  const nxr::io::SubjectPackage pkg = nxr::io::SubjectPackage::open(package);
  const task::Subject subject = task::Subject::of(pkg);
  const bool labelling = !o.labelling.empty();

  // ---- what is exported, and the surface it lies on
  task::Node input;
  std::string surface_id;
  bool series = false;
  if (labelling) {
    input = subject.require({.cls = task::Cls::selection, .name = o.labelling}, "the labelling '" + o.labelling + "'");
    if (!input.type || (*input.type != "partition" && *input.type != "set")) refuse("'" + o.labelling + "' is a selection of type " + (input.type ? *input.type : std::string("none")) + ", not a labelling (a partition or a set)");
    if (!input.array_path) refuse("the labelling '" + o.labelling + "' stores no codes");
    if (!input.manifold_id) refuse("the labelling '" + o.labelling + "' lies on no manifold");
    surface_id = *input.manifold_id;
    if (o.range) refuse("a labelling is not a series: no time range");
  } else {
    input = subject.require({.cls = task::Cls::field, .name = o.field}, "the field '" + o.field + "'");
    if (!input.path || !input.manifold_id) refuse("the field '" + o.field + "' stores no values");
    const std::string vt = input.value_type.value_or("scalar");
    if (vt != "scalar" && vt != "integer") refuse("export gifti writes scalar fields; '" + o.field + "' is " + vt + " (name one of its components as a field, or use another export)");
    const auto factors = subject.factors(*input.manifold_id);
    if (factors.size() == 1) surface_id = factors[0];
    else if (factors.size() == 2) {
      surface_id = factors[0];
      series = true;
    } else refuse("the field '" + o.field + "' lies on a product of " + std::to_string(factors.size()) + " manifolds; GIFTI holds a map or a series on one surface");
  }
  const task::Node surface_node = subject.require_id(task::Cls::manifold, surface_id);
  const Surface base = load_surface(pkg, subject, surface_node);

  if (series && !o.range) refuse("'" + o.field + "' is a series over time: state the time range to export (from and to in seconds, or all)");
  if (!series && o.range && !labelling) refuse("'" + o.field + "' is a map: it has no time range");
  if (series && o.kind == "shape") refuse("a series is exported as .func.gii: shape files hold static data");

  std::vector<Surface> parts;                      // what is written: the surface as one, or its hemispheres
  if (o.hemispheres) {
    if (!base.hemi.empty()) refuse("'" + surface_node.name + "' is one hemisphere already: --hemispheres splits a glued cortex");
    parts = hemispheres_of(pkg, subject, surface_node);
  } else {
    parts.push_back(base);
  }
  std::vector<std::int64_t> first_row(parts.size(), 0);   // each part's first row of the whole surface's data
  if (o.hemispheres)
    for (std::size_t k = 0; k < parts.size(); ++k) {
      const Json f = parts[k].node.fact("from_index");
      first_row[k] = f.is_number() ? f.get<std::int64_t>() : 0;
    }
  auto stem = [&](const Surface& s) { return o.out + (o.hemispheres ? "_hemi-" + s.hemi : std::string()); };

  // ---- provenance inputs and terms
  std::vector<task::Node> inputs = {input, surface_node};
  const Terms terms = terms_of(subject, inputs);
  Json details = Json::object();
  details["surface"] = surface_node.name;
  details["hemispheres"] = o.hemispheres;
  Json span;                                       // null for a map
  std::vector<std::string> files;
  std::int64_t frames = 0;

  double scale = 1.0;
  if (o.surface_unit) {
    if (base.coordinate_unit.empty()) refuse("the surface '" + surface_node.name + "' states no coordinate unit, so it cannot be converted to " + *o.surface_unit);
    scale = metres_per(base.coordinate_unit) / metres_per(*o.surface_unit);
    details["surface_unit"] = Json{{"stored", base.coordinate_unit}, {"written", *o.surface_unit}, {"scale", scale}};
  }

  const Json producer_json = labelling ? input.params : input.producer;

  if (labelling) {
    // ---- a labelling: int32 codes and the label table
    const ArrayReader a = ArrayReader::open(pkg.store(), *input.array_path);
    if (a.shape().size() != 1 || a.shape()[0] != base.n) refuse("the labelling '" + o.labelling + "' holds " + (a.shape().empty() ? std::string("a scalar") : std::to_string(a.shape()[0])) + " codes for a surface of " + std::to_string(base.n) + " vertices");
    std::vector<std::int32_t> codes = read_int32(a);
    std::vector<task::DictionaryLevel> entries;
    if (input.dictionary_id) entries = subject.dictionary_entries(*input.dictionary_id);
    std::map<std::int32_t, gifti::Label> table;
    for (const auto& e : entries) {
      gifti::Label l;
      l.key = static_cast<std::int32_t>(e.code);
      l.name = e.name;
      if (e.color) {
        l.rgba = {static_cast<float>((*e.color)[0]) / 255.0f, static_cast<float>((*e.color)[1]) / 255.0f, static_cast<float>((*e.color)[2]) / 255.0f, 1.0f};
        l.has_colour = true;
      }
      table[l.key] = l;
    }
    const bool negatives = std::any_of(codes.begin(), codes.end(), [](std::int32_t c) { return c < 0; });
    if (negatives) {
      if (table.count(0)) refuse("the labelling '" + o.labelling + "' has elements with no label (code below 0) and its dictionary already uses code 0: no key is free for them");
      for (auto& c : codes) if (c < 0) c = 0;
      gifti::Label l;
      l.key = 0;
      l.name = "???";
      table[0] = l;
      details["unlabelled_as_key_0"] = true;
    }
    Json added = Json::array();
    for (const auto c : codes)
      if (!table.count(c)) {
        gifti::Label l;
        l.key = c;
        l.name = "code " + std::to_string(c);
        table[c] = l;
        added.push_back(c);
      }
    if (!added.empty()) details["codes_not_in_the_dictionary"] = added;
    for (std::size_t k = 0; k < parts.size(); ++k) {
      gifti::Image img;
      img.metadata = file_meta(o, producer_json, terms, span, parts[k].structure);
      add_meta(img.metadata, "nxr-neuro:labelling", o.labelling);
      for (const auto& kv : table) img.labels.push_back(kv.second);
      std::vector<std::int32_t> part(codes.begin() + first_row[k], codes.begin() + first_row[k] + parts[k].n);
      gifti::DataArray d = gifti::make_int32("NIFTI_INTENT_LABEL", {parts[k].n}, std::move(part));
      d.encoding = gifti::Encoding::gzip_base64;
      add_meta(d.metadata, "Name", o.labelling);
      add_meta(d.metadata, "nxr-neuro:field", o.labelling);
      img.arrays.push_back(std::move(d));
      const std::string path = stem(parts[k]) + ".label.gii";
      gifti::write_file(path, img);
      files.push_back(path);
    }
  } else {
    // ---- a field: one array per frame of the range (a map: one)
    const ArrayReader a = ArrayReader::open(pkg.store(), *input.path);
    const std::string intent = series ? "NIFTI_INTENT_TIME_SERIES" : o.kind == "shape" ? "NIFTI_INTENT_SHAPE" : "NIFTI_INTENT_NONE";
    const std::string ext = o.kind == "shape" ? ".shape.gii" : ".func.gii";
    std::vector<gifti::Image> images(parts.size());
    double series_rate = 0;
    auto add_frame = [&](const std::vector<float>& values, double t, std::int64_t frame) {
      for (std::size_t k = 0; k < parts.size(); ++k) {
        std::vector<float> v(values.begin() + first_row[k], values.begin() + first_row[k] + parts[k].n);
        gifti::DataArray d = gifti::make_float32(intent, {parts[k].n}, std::move(v));
        d.encoding = gifti::Encoding::gzip_base64;
        add_meta(d.metadata, "Name", o.field);
        add_meta(d.metadata, "nxr-neuro:field", o.field);
        if (input.unit) add_meta(d.metadata, "nxr-neuro:unit", *input.unit);
        if (series) {
          add_meta(d.metadata, "nxr-neuro:frame", std::to_string(frame));
          add_meta(d.metadata, "nxr-neuro:time", nxr::formats::tsv::format_number(t));
        }
        images[k].arrays.push_back(std::move(d));
      }
    };
    if (!series) {
      if (!(a.shape().size() == 1 || (a.shape().size() == 2 && a.shape()[1] == 1)) || a.shape()[0] != base.n)
        refuse("the map '" + o.field + "' has the shape [" + std::to_string(a.shape().empty() ? 0 : a.shape()[0]) + (a.shape().size() > 1 ? ", " + std::to_string(a.shape()[1]) : std::string()) + "] for a surface of " + std::to_string(base.n) + " vertices");
      const std::vector<double> v = widen(a.read(), a.dtype());
      std::vector<float> f(v.begin(), v.end());
      add_frame(f, 0.0, 0);
      frames = 1;
    } else {
      if (a.shape().size() != 2 || a.shape()[0] != base.n) refuse("the series '" + o.field + "' is not [vertices x samples] for a surface of " + std::to_string(base.n) + " vertices");
      const nxr::io::finder::Value v = subject.describe(input);
      series_rate = v.rate;
      if (!v.time || v.rate <= 0) refuse("the field '" + o.field + "' has no time line with a rate: it is not a series");
      const std::int64_t width = nxr::io::chunk_width(input.fact("chunk_shape_json").is_string() ? Json::parse(input.fact("chunk_shape_json").get<std::string>(), nullptr, false) : input.fact("chunk_shape_json"));
      nxr::io::SpanRequest rq;
      if (o.range->all) rq.chunks = "all";
      else {
        if (!(o.range->to > o.range->from)) refuse("the time range must end after it starts (from " + std::to_string(o.range->from) + " s, to " + std::to_string(o.range->to) + " s)");
        rq.start = o.range->from;
        rq.stop = o.range->to;
      }
      const nxr::io::Span sp = nxr::io::resolve_span(rq, a.shape()[1], v.rate, v.origin, width);
      span = nxr::io::span_record(sp, v.rate, v.origin);
      details["time"] = Json{{"rate_hz", v.rate}, {"origin_s", v.origin}};
      const nxr::io::ChunkSource src(pkg.store(), *input.path, width, v.rate, v.origin);
      nxr::io::chunk_loop(src, sp, [&](const nxr::io::Chunk& c) {
        std::vector<float> frame(static_cast<std::size_t>(base.n));
        for (std::int64_t j = 0; j < c.n; ++j) {
          for (std::int64_t r = 0; r < base.n; ++r) frame[static_cast<std::size_t>(r)] = static_cast<float>(c.data[static_cast<std::size_t>(r * c.n + j)]);
          add_frame(frame, v.origin + static_cast<double>(c.first + j) / v.rate, c.first + j);
          ++frames;
        }
        if (o.ctx.progress) o.ctx.progress(Json{{"stage", "frames"}, {"chunk", c.k}, {"frames", frames}});
      });
      if (frames == 0) refuse("the time range holds no sample of '" + o.field + "'");
    }
    if (a.dtype() == nxr::io::DType::Float64) details["converted"] = Json::array({"float64 -> float32 (GIFTI holds float32 only)"});
    else if (a.dtype() != nxr::io::DType::Float32) details["converted"] = Json::array({std::string(nxr::io::dtype_to_string(a.dtype())) + " -> float32 (GIFTI holds float32 only)"});
    for (std::size_t k = 0; k < parts.size(); ++k) {
      images[k].metadata = file_meta(o, producer_json, terms, span, parts[k].structure);
      if (series) {
        add_meta(images[k].metadata, "TimeStep", nxr::formats::tsv::format_number(1.0 / series_rate));
        add_meta(images[k].metadata, "nxr-neuro:frames", std::to_string(frames));
      }
      const std::string path = stem(parts[k]) + ext;
      gifti::write_file(path, images[k]);
      files.push_back(path);
    }
  }
  if (o.surface)
    for (const Surface& s : parts) {
      const std::string path = stem(s) + ".surf.gii";
      write_surface(path, o, s, scale, surface_node.producer, terms);
      files.push_back(path);
    }

  details["frames"] = frames;
  const Json prov = provenance(pkg, subject, o.ctx, inputs, span, details);
  const std::string sidecar = write_sidecar(files.front(), prov, files);
  return Json{{"files", files}, {"sidecar", sidecar}, {"frames", frames}, {"details", details}, {"terms", terms.to_json()}};
}

}  // namespace nxr::exporters

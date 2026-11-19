// nxr_neuro._core, the pipeline level: `run`, the lists and `read` (docs/specs/2026-11-17-compute-and-run.md, section 7). A binding only converts: the arguments become the library's
// request (pipelines/face.h: the same calls as `nxr-neuro run`, `list` and `read` make), the answers go back as JSON text and as bytes with their dtype and shape; the
// package (nxr_neuro/_pipeline.py, list.py) turns them into dictionaries and NumPy arrays.
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <string>

#include "arrays.h"
#include "nxr/pipelines/face.h"

namespace nxrpy {

namespace {

namespace face = nxr::pipelines::face;
using Json = face::Json;

}  // namespace

void bind_pipeline(nb::module_& m) {
  m.def("_pipelines", [] { return face::pipelines().dump(); }, "The pipelines `run` takes, as JSON text.");
  m.def("_datasets", [](const std::string& datastore) { return face::datasets(datastore).dump(); }, nb::arg("datastore"));
  m.def(
      "_subjects",
      [](const std::string& dataset, const std::vector<std::string>& names, const std::optional<std::string>& where) { return face::subjects(dataset, names, where).dump(); },
      nb::arg("dataset"), nb::arg("names"), nb::arg("where").none());
  m.def("_results", [](const std::string& dataset, const std::string& subject, const std::string& kind) { return face::results(dataset, subject, kind).dump(); }, nb::arg("dataset"),
        nb::arg("subject"), nb::arg("kind"));
  m.def(
      "_run",
      [](const std::string& dataset, const std::string& subject, const std::string& pipeline, const std::string& parameters, std::optional<double> start, std::optional<double> stop,
         bool all, bool replace, nb::object progress) {
        face::RunRequest rq;
        rq.pipeline = pipeline;
        rq.parameters = Json::parse(parameters);
        rq.from = start;
        rq.to = stop;
        rq.all = all;
        rq.replace = replace;
        face::Progress say;
        if (!progress.is_none()) say = [&progress](const Json& j) {   // called from the library with the GIL released: take it for the Python callback
          nb::gil_scoped_acquire gil;
          progress(j.dump());
        };
        std::string out;
        {
          nb::gil_scoped_release release;
          out = face::run(dataset, subject, rq, say).dump();
        }
        return out;
      },
      nb::arg("dataset"), nb::arg("subject"), nb::arg("pipeline"), nb::arg("parameters"), nb::arg("start").none(), nb::arg("stop").none(), nb::arg("all"), nb::arg("replace"),
      nb::arg("progress").none());
  m.def(
      "_read",
      [](const std::string& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure, const std::optional<std::string>& of_field,
         std::optional<std::int64_t> start, std::optional<std::int64_t> stop, const std::vector<std::int32_t>& rows, const std::string& projectors, const std::string& reading,
         const std::optional<std::string>& family) {
        face::ReadRequest rq;
        rq.measure = measure;
        rq.of_field = of_field;
        rq.start = start;
        rq.stop = stop;
        rq.rows = rows;
        rq.projectors = projectors;
        rq.reading = reading;
        rq.family = family;
        face::Stored s;
        {
          nb::gil_scoped_release release;
          s = face::read(dataset, subject, name, rq);
        }
        nb::list arrays;
        for (const auto& a : s.arrays) {
          nb::list shape;
          for (const auto d : a.shape) shape.append(d);
          arrays.append(nb::make_tuple(a.name, a.dtype, nb::tuple(shape), nb::bytes(reinterpret_cast<const char*>(a.bytes.data()), a.bytes.size())));
        }
        return nb::make_tuple(s.meta.dump(), arrays);
      },
      nb::arg("dataset"), nb::arg("subject"), nb::arg("name"), nb::arg("measure").none(), nb::arg("of_field").none(), nb::arg("start").none(), nb::arg("stop").none(),
      nb::arg("rows"), nb::arg("projectors"), nb::arg("reading"), nb::arg("family").none());
  m.def(
      "_synthetic_dataset",
      [](const std::string& datastore, const std::string& name, const std::string& subject, const std::string& recording, int channels, double sfreq, double seconds) {
        return face::synthetic_dataset(datastore, name, subject, recording, channels, sfreq, seconds).string();
      },
      nb::arg("datastore"), nb::arg("name") = "demo", nb::arg("subject") = "S1", nb::arg("recording") = "rest", nb::arg("channels") = 3, nb::arg("sfreq") = 100.0,
      nb::arg("seconds") = 40.0, "A small synthetic dataset for tests and demos (a recording of known signals with one bad span); returns its folder.");
}

}  // namespace nxrpy

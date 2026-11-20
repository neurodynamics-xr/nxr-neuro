// nxr_neuro._core, the pipeline level: `run` (one subject, or a group) and what it takes (docs/specs/2026-11-17-compute-and-run.md, section 7). A binding only converts: the
// arguments become the library's request (pipelines/face.h: the same calls as `nxr-neuro run` makes), the answers go back as JSON text; the package (nxr_neuro/_commands.py)
// turns them into dictionaries. Every other command is commands.cpp.
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
  m.def(
      "_run",
      [](const std::string& dataset, const std::string& subject, const std::string& pipeline, const std::string& parameters, std::optional<double> from, std::optional<double> to,
         bool all, bool replace, bool dry_run, const std::string& preset, const std::vector<std::string>& subjects, const std::string& into, nb::object progress) {
        const Json params = Json::parse(parameters);
        face::Progress say;
        if (!progress.is_none()) say = [&progress](const Json& j) {   // called from the library with the GIL released: take it for the Python callback
          nb::gil_scoped_acquire gil;
          progress(j.dump());
        };
        std::string out;
        nb::gil_scoped_release release;
        if (!subjects.empty() || !into.empty()) {   // a group run: the members and the group package are the caller's
          face::GroupRunRequest rq;
          rq.pipeline = pipeline;
          rq.parameters = params;
          rq.subjects = subjects;
          rq.into = into;
          rq.replace = replace;
          rq.preset = preset;
          rq.dry_run = dry_run;
          out = face::run_group(dataset, rq, say).dump();
        } else {
          face::RunRequest rq;
          rq.pipeline = pipeline;
          rq.parameters = params;
          rq.from = from;
          rq.to = to;
          rq.all = all;
          rq.replace = replace;
          rq.preset = preset;
          rq.dry_run = dry_run;
          out = face::run(dataset, subject, rq, say).dump();
        }
        return out;
      },
      nb::arg("dataset"), nb::arg("subject"), nb::arg("pipeline"), nb::arg("parameters"), nb::arg("from").none(), nb::arg("to").none(), nb::arg("all"), nb::arg("replace"),
      nb::arg("dry_run"), nb::arg("preset"), nb::arg("subjects"), nb::arg("into"), nb::arg("progress").none());
  m.def(
      "_synthetic_dataset",
      [](const std::string& datastore, const std::string& name, const std::string& subject, const std::string& recording, int channels, double sfreq, double seconds) {
        return face::synthetic_dataset(datastore, name, subject, recording, channels, sfreq, seconds).string();
      },
      nb::arg("datastore"), nb::arg("name") = "demo", nb::arg("subject") = "S1", nb::arg("recording") = "rest", nb::arg("channels") = 3, nb::arg("sfreq") = 100.0,
      nb::arg("seconds") = 40.0, "A small synthetic dataset for tests and demos (a recording of known signals with one bad span); returns its folder.");
}

}  // namespace nxrpy

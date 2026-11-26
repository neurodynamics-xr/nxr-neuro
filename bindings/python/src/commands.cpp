// nxr_neuro._core, THE COMMANDS (docs/specs/2026-11-19-io-service.md): `nxr-neuro <verb> <noun> --option VALUE` as `_command(verb, noun, options)`, the options as JSON
// text keyed by the command's option names (bindings/common/commands.h converts them to the library's request and calls the command's function once). The result goes back
// as JSON text and the command's arrays as (name, dtype, shape, bytes); the package (nxr_neuro/_commands.py) makes them a dictionary and NumPy arrays. And the writer lock a
// handle (`nxr_neuro.open(dataset, mode="write")`) holds until it is closed.
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "commands.h"
#include "handle.h"
#include "nxr/pipelines/face.h"

namespace nb = nanobind;

namespace nxrpy {

namespace {

using nxr::io::Json;

}  // namespace

void bind_commands(nb::module_& m) {
  m.def(
      "_command",
      [](const std::string& verb, const std::string& noun, const std::string& options, nb::object progress, nb::object in_memory, HandlePtr handle) {
        const Json opts = Json::parse(options);
        // the arrays in memory: [(option, name, dtype, shape, bytes)] (no temporary file; the name is what the producer records)
        nxr::bindings::Arrays held;
        if (!in_memory.is_none())
          for (nb::handle item : in_memory) {
            nb::tuple t = nb::cast<nb::tuple>(item);
            std::vector<std::int64_t> shape;
            for (nb::handle d : nb::cast<nb::tuple>(t[3])) shape.push_back(nb::cast<std::int64_t>(d));
            nb::bytes data = nb::cast<nb::bytes>(t[4]);
            const auto* p = reinterpret_cast<const std::uint8_t*>(data.c_str());
            held[nb::cast<std::string>(t[0])] = nxr::io::task::ArrayInput::memory(nb::cast<std::string>(t[1]), nb::cast<std::string>(t[2]), std::move(shape), std::vector<std::uint8_t>(p, p + data.size()));
          }
        nxr::io::task::Progress say;
        if (!progress.is_none()) say = [&progress](const Json& j) {   // called from the library with the GIL released: take it for the Python callback
          nb::gil_scoped_acquire gil;
          progress(j.dump());
        };
        nxr::io::task::CommandResult r;
        {
          nb::gil_scoped_release release;
          r = nxr::bindings::command(verb, noun, opts, say, nxr::pipelines::face::stamp(), held, handle ? handle->handle.get() : nullptr);
        }
        nb::list arrays;
        for (const auto& a : r.arrays) {
          nb::list shape;
          for (const auto d : a.shape) shape.append(d);
          arrays.append(nb::make_tuple(a.name, a.dtype, nb::tuple(shape), nb::bytes(reinterpret_cast<const char*>(a.bytes.data()), a.bytes.size())));
        }
        return nb::make_tuple(r.result.dump(), arrays);
      },
      nb::arg("verb"), nb::arg("noun"), nb::arg("options"), nb::arg("progress").none(), nb::arg("in_memory").none() = nb::none(), nb::arg("handle").none() = nb::none(),
      "The command `nxr-neuro VERB NOUN` with its options (JSON text) and its arrays in memory: (the result as JSON text, [(name, dtype, shape, bytes)]).");

  nb::class_<PyHandle>(m, "_Handle")
      .def(
          "__init__",
          [](PyHandle* self, const std::string& dataset, bool write) {
            using nxr::io::task::Handle;
            new (self) PyHandle{std::make_unique<Handle>(dataset, write ? Handle::Mode::write : Handle::Mode::read, nxr::pipelines::face::stamp())};
          },
          nb::arg("dataset"), nb::arg("write"),
          "A dataset opened once: its read-only databases kept open between calls; write=True takes the writer lock now and holds it until close().")
      .def_prop_ro("closed", [](const PyHandle& h) { return h.handle->closed(); })
      .def_prop_ro("file", [](const PyHandle& h) { return h.handle->lock_file(); })
      .def("close", [](PyHandle& h) { h.handle->close(); });
}

}  // namespace nxrpy

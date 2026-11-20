// nxr_neuro._core, THE COMMANDS (docs/specs/2026-11-19-io-service.md): `nxr-neuro <verb> <noun> --option VALUE` as `_command(verb, noun, options)`, the options as JSON
// text keyed by the command's option names (bindings/common/commands.h converts them to the library's request and calls the command's function once). The result goes back
// as JSON text and the command's arrays as (name, dtype, shape, bytes); the package (nxr_neuro/_commands.py) makes them a dictionary and NumPy arrays. And the writer lock a
// handle (`nxr_neuro.open(dataset, mode="write")`) holds until it is closed.
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <memory>
#include <string>

#include "commands.h"
#include "nxr/io/writer_lock.h"
#include "nxr/pipelines/face.h"

namespace nb = nanobind;

namespace nxrpy {

namespace {

using nxr::io::Json;

/** the dataset's writer lock, held by a handle until release() (or its collection; io's atexit releases what is left) */
struct HeldLock {
  nxr::io::WriterLock lock;
  std::string file() const { return lock ? lock.file().string() : std::string(); }
  void release() { if (lock) lock.release(); }
};

}  // namespace

void bind_commands(nb::module_& m) {
  m.def(
      "_command",
      [](const std::string& verb, const std::string& noun, const std::string& options, nb::object progress) {
        const Json opts = Json::parse(options);
        nxr::io::task::Progress say;
        if (!progress.is_none()) say = [&progress](const Json& j) {   // called from the library with the GIL released: take it for the Python callback
          nb::gil_scoped_acquire gil;
          progress(j.dump());
        };
        nxr::io::task::CommandResult r;
        {
          nb::gil_scoped_release release;
          r = nxr::bindings::command(verb, noun, opts, say, nxr::pipelines::face::stamp());
        }
        nb::list arrays;
        for (const auto& a : r.arrays) {
          nb::list shape;
          for (const auto d : a.shape) shape.append(d);
          arrays.append(nb::make_tuple(a.name, a.dtype, nb::tuple(shape), nb::bytes(reinterpret_cast<const char*>(a.bytes.data()), a.bytes.size())));
        }
        return nb::make_tuple(r.result.dump(), arrays);
      },
      nb::arg("verb"), nb::arg("noun"), nb::arg("options"), nb::arg("progress").none(),
      "The command `nxr-neuro VERB NOUN` with its options (JSON text): (the result as JSON text, [(name, dtype, shape, bytes)]).");

  nb::class_<HeldLock>(m, "_HeldLock")
      .def_prop_ro("file", &HeldLock::file)
      .def("release", &HeldLock::release);
  m.def(
      "_writer_lock",
      [](const std::string& dataset) {
        auto h = std::make_unique<HeldLock>();
        h->lock = nxr::io::acquire_writer_lock(dataset, nxr::pipelines::face::stamp());
        return h.release();
      },
      nb::arg("dataset"), nb::rv_policy::take_ownership, "Take the dataset folder's writer lock (refused while another live writer holds it).");
  m.attr("_writer_lock_env") = nxr::io::kWriterLockEnv;
}

}  // namespace nxrpy

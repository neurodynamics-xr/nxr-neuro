// nxr_neuro._core: the nanobind module behind the Python package. Bindings only bind: arrays in, the library's call, arrays out.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "nxr/core/errors.h"
#include "nxr/io/errors.h"

namespace nb = nanobind;

namespace nxrpy {
void bind_compute(nb::module_& m);
void bind_formats(nb::module_& m);
void bind_pipeline(nb::module_& m);
}  // namespace nxrpy

namespace {

// The library's errors become `nxr_neuro.Error` and its subclass for the code (nxr_neuro/_errors.py), with the code as `.code`.
void raise(const std::string& code, const std::string& message) {
  nb::object make = nb::module_::import_("nxr_neuro._errors").attr("make");
  nb::object exc = make(code, message);
  PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "The compiled core of nxr_neuro (private: use the package's functions).";
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void*) {
        try {
          std::rethrow_exception(p);
        } catch (const nxr::io::Error& e) {
          raise(e.code(), e.what());
        } catch (const nxr::core::Error& e) {
          raise(std::string(nxr::core::errorCodeName(e.code())), e.what());
        }
        // anything else (std::invalid_argument -> ValueError, std::runtime_error -> RuntimeError) is translated by nanobind
      },
      nullptr);
  nxrpy::bind_compute(m);
  nxrpy::bind_formats(m);
  nxrpy::bind_pipeline(m);
}

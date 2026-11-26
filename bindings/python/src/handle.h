// The dataset handle behind `nxr_neuro.open` (nxr/io/task/handle.h): the library's Handle, owned by the Python object that `_command` and `_run` are given.
#pragma once
#include <memory>
#include <string>

#include "nxr/io/task/handle.h"

namespace nxrpy {

struct PyHandle {
  std::unique_ptr<nxr::io::task::Handle> handle;
};
using HandlePtr = PyHandle*;

}  // namespace nxrpy

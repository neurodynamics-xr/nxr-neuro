// nxr-pipelines (internal) — a kernel's rows for a Hodge-Dirac package's surface (source_flow_steps.cpp defines it; signal.mode_analysis and the source-flow nodes use it).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nxr/graph/node.h"

namespace nxr::pipelines::graph {

/** Which vertex of a kernel's surface each vertex of the package's surface reads (THE GLUE IS A SELECTION): the package's input surface itself; a component of the kernel's
 *  surface (a hemisphere: its vertex range, from the package's `input_from_index`); or the glued surface (the package's `keep`: its kept vertices' input vertices, -1 on the
 *  bridge, whose rows are zero). Refuses a kernel that is not free-orientation, not on the package's input surface or the surface it is a component of, or whose columns are
 *  not its channels. `description` says which. */
struct KernelSelection {
  std::vector<std::int64_t> source_vertex;
  std::string description;
  std::int64_t bridge_vertices = 0;
};
KernelSelection kernel_rows_for_package(const nxr::graph::Array& kernel, const nxr::graph::Array& package, const char* who);

}  // namespace nxr::pipelines::graph

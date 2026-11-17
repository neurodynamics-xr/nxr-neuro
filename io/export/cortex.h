// The exporters' shared reading of a cortical surface and its hemispheres (internal to io/export/): a surface manifold's positions and triangles as stored, and the left and
// right hemispheres a glued cortex keeps as components (their `label`, their first vertex `from_index`).
#pragma once
#include <string>
#include <vector>

#include "nxr/exporters/common.h"
#include "nxr/io/array_reader.h"

namespace nxr::exporters::detail {

// a surface as stored: its positions and triangles, in the manifold's own unit
struct Surface {
  task::Node node;
  std::int64_t n = 0;
  std::vector<double> xyz;                 // [n x 3]
  std::vector<std::int32_t> faces;         // [m x 3]
  std::string coordinate_unit;             // "m", "mm", "": as the manifold states it
  std::string structure;                   // "CortexLeft" · "CortexRight" · "" when it is not a hemisphere
  std::string hemi;                        // "L" · "R" · ""
};

// the surface of a manifold node; a hemisphere component of a glued cortex reads its slice of the union's arrays (`from_index`)
Surface load_surface(const nxr::io::SubjectPackage& pkg, const task::Subject& subject, const task::Node& manifold);
// the hemispheres of a glued cortex: its components by ordinal, each labelled left or right
std::vector<Surface> hemispheres_of(const nxr::io::SubjectPackage& pkg, const task::Subject& subject, const task::Node& whole);
// an integer array read as int32 (any value that an int32 cannot hold is refused)
std::vector<std::int32_t> read_int32(const nxr::io::ArrayReader& a);

}  // namespace nxr::exporters::detail

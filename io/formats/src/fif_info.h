// The measurement-info and annotation readers shared by the FIF sources (not installed).
#pragma once
#include <optional>

#include "nxr/formats/fif.h"

namespace nxr::formats::fif {

// The measurement info of the (single) measurement block `meas` of `f`.
Info read_info(const File& f, const Block& meas);
// The first annotation block anywhere in the tree, if any.
std::optional<AnnotationSet> read_annotations(const File& f, const Block& root);

}  // namespace nxr::formats::fif

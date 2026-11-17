// nxr-pipelines — THE BRIDGE between compute's selection and io's: the pipelines are the one place both are visible. Compute makes a selection in memory
// (`nxr::selection::Selection`, compute/include/nxr/selection/selection.h); io stores and loads it as its own description (`nxr::io::selection::Description`,
// io/include/nxr/io/selection_description.h), a field-for-field mirror. These two functions copy one into the other, member by member, and decide nothing.
#pragma once
#include "nxr/io/selection_description.h"
#include "nxr/selection/selection.h"

namespace nxr::pipelines {

/** a selection compute made, as io's description (to store) */
nxr::io::selection::Description describe(const nxr::selection::Selection& made);
/** a selection io loaded, as compute's selection (to use) */
nxr::selection::Selection made(const nxr::io::selection::Description& loaded);

}  // namespace nxr::pipelines

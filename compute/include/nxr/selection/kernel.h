// nxr::selection (compute) — A KERNEL MEMBER (a member with a `Profile`, selection.h: an anchor + a profile + a chart) checked for its structure. Its gains on a
// line are evaluated by ONE function, nxr::signal::kernelGains (nxr/signal/kernel_gains.h), beside the filterbanks whose shapes it uses: the toolbox order lets
// signal use selection, never the reverse. Each family's own parameters are checked there.
#pragma once
#include <optional>
#include <string>

#include "nxr/selection/selection.h"

namespace nxr::selection {

/** why a profile is refused for its structure (no family, no or an unknown chart, a gain that is not finite), or nothing */
std::optional<std::string> profile_refusal(const Profile& profile);

}  // namespace nxr::selection

// nxr::selection: a kernel member's profile checked for its structure (include/nxr/selection/kernel.h). Its gains on a line are nxr::signal::kernelGains.
#include <cmath>
#include "nxr/selection/kernel.h"

namespace nxr::selection {

std::optional<std::string> profile_refusal(const Profile& p) {
  if (p.family.empty()) return std::string("a kernel's profile states its family");
  if (p.chart.empty()) return std::string("a kernel's profile states its chart (linear | log2 | log10)");
  if (p.chart != "linear" && p.chart != "log2" && p.chart != "log10") return "unknown chart '" + p.chart + "' (linear | log2 | log10)";
  if (!std::isfinite(p.gain)) return std::string("a kernel's profile states a finite gain");
  return std::nullopt;
}

}  // namespace nxr::selection

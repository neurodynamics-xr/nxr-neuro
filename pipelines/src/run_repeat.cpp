// nxr-pipelines — the stage names of a run's stored crossings (run_repeat.h).
#include "nxr/pipelines/run_repeat.h"

#include <stdexcept>

namespace nxr::pipelines {

std::vector<std::string> crossing_stage_names(const std::vector<std::string>& computed_by) {
  std::vector<std::string> out;
  if (computed_by.empty()) return out;
  const std::string tag = " \xc2\xb7 stages ";
  const std::size_t at = computed_by[0].find(tag);
  if (at == std::string::npos) return out;
  const std::string rest = computed_by[0].substr(at + tag.size()) + ",";
  std::string cur;
  for (const char c : rest) { if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
  return out;
}
int crossing_stage_index(const std::string& which, const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t i = 0; i < names.size(); ++i) if (names[i] == name) return static_cast<int>(i);
  std::string have;
  for (const auto& n : names) have += (have.empty() ? "" : ", ") + n;
  throw std::invalid_argument(std::string("run_repeat: ") + which + (names.empty() ? " does not name its stages (crossings stored before stage names: run vortex_cores again with replace, or give stage as an index)" : " has no stage '" + name + "' (it has: " + have + ")"));
}

}  // namespace nxr::pipelines

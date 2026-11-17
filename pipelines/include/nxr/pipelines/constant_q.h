// nxr-pipelines — THE CONSTANT-Q TILING of a time Line, stored: the bridge between compute's design and io's store (the Studio's `backend/src/db/constant-q.ts` is the
// reference of the design; the design is compute's: `nxr::selection::constant_q`, compute/include/nxr/selection/partition.h). This function READS the line (io),
// asks compute to DESIGN the ladder (a set with levels, each a tiling and the band it resolves) and hands the result to io, which only stores it
// (`nxr::io::selection::save`: a `ladder` Selection on the time Line, its members the levels, each a `spans` member whose membership is the closed form). Idempotent:
// a line that has a ladder returns it, its parameters are not compared.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/writer.h"

namespace nxr::pipelines {

/** the design's options (compute's `ConstantQ`): left empty, the method's defaults apply (hertz: f_max 64 Hz capped by Nyquist, f_min 0.5 Hz; cycles: 8 cycles, Nyquist, 0) */
struct ConstantQOptions {
  std::optional<std::string> method;      // hertz | cycles
  std::optional<int> voices;              // bands per octave (default 6)
  std::optional<double> cycles;           // cycles of a band's low edge in its window
  std::optional<double> f_min, f_max;     // Hz
};

/** a stored level, for reports */
struct ConstantQLevel { int level = 0; std::int64_t window_samples = 0; double window_s = 0; std::int64_t tiles = 0; double resolves_lo = 0, resolves_hi = 0; };

/** what constant_q_write made, or found */
struct ConstantQWritten {
  std::string set_id, name;
  std::vector<std::string> member_ids;
  bool existed = false;
  std::vector<ConstantQLevel> levels;     // from the stored members (empty when it existed)
};

/** DESIGN (compute) and STORE (io) the constant-Q ladder of the time Line `manifold_id` (a closed-form line), named `<line>_constantQ`. Inside the caller's composition. */
ConstantQWritten constant_q_write(nxr::io::SubjectWriter& sub, const std::string& manifold_id, const ConstantQOptions& options = {}, const std::optional<std::string>& by = std::nullopt);

}  // namespace nxr::pipelines

// nxr::select::constantq (compute) — THE BAND-DESIGN ARITHMETIC OF A CONSTANT-Q LADDER: the rules that turn a sample rate and a frequency into a window, a
// decimation and the anchors of a method (the Cortical Flow Studio's `backend/src/db/constant-q.ts`: `anchorsFor`, `strideFor`, and the window `pow2ceil(cycles ·
// sfreq / lo)`). Design is compute's: io stores the designed ladder (it keeps only the tile grid's closed form, io/include/nxr/io/constant_q.h) and the pipelines
// call these for the bands they run. Pure functions of numbers; std::invalid_argument for an argument outside its domain.
#pragma once
#include <cstdint>
#include <optional>
#include <utility>

namespace nxr::select::constantq {

enum class Method { Hertz, Cycles };
constexpr double kCapHz = 64, kFloorHz = 0.5;
constexpr int kMultirateMin = 24, kMultirateMax = 48;

/** the anchors a method implies at this rate: {cycles, fMax} (hertz: the cycles of a power-of-two window at `target` samples a cycle, f_max a power of two in Hz
 *  capped at 64 Hz; cycles: `target` cycles, f_max Nyquist) */
std::pair<double, double> anchorsFor(Method method, double sfreq, double target = 8);
/** the decimation for a band, a power of two leaving at least `min` samples a cycle at `hz` (never below 1) */
std::int64_t strideFor(double sfreq, double hz, int min = kMultirateMin);
/** a band's window in samples, as the ladder sizes it: `cycles` cycles of its LOW edge rounded UP to a power of two (the Studio's `pow2ceil(cycles·sfreq/lo)`);
 *  `cycles` defaults to the hertz anchor's (`anchorsFor(Hertz, sfreq)`: 6.83 at 1200 Hz) */
std::int64_t windowFor(double sfreq, double lo, std::optional<double> cycles = std::nullopt);

}  // namespace nxr::select::constantq

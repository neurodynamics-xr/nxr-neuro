#pragma once
// nxr::frametiling — sampling a band-limited signal on the TOWER's frames (the frames themselves are `nxr::tower::FrameTiling`, tower.h;: the old per-band tiling, frames_per_cycle × the band's centre,
// is replaced by tower frames: it broke the dyadic nesting of time). Pure functions of arrays: no file, no chunk, no recording.
//
//   sincInterpolate  a block of an ANALYTIC (band-limited) signal sampled at the frames' (non-integer) sample positions: a Kaiser-windowed sinc, `half_width` samples either side (the step's halo).
//                  nxr resamples the whole run by FFT (exact for a band-limited signal); this is the same band-limited interpolation as a local filter, the one a chunk can run. Beyond the block it
//                  reads 0, as a band's taps do beyond the recording. A frame's time is its tile's START (tower.h `FrameTiling::position`).
#include <cstdint>
#include <vector>

namespace nxr::frametiling {

/** x [rows × width] (row-major, column 0 = sample 0 of the block) sampled at `positions` (samples from column 0, not whole): [rows × positions], 0 read outside the block */
std::vector<float> sincInterpolate(const float* x, std::int64_t rows, std::int64_t width, const std::vector<double>& positions, int half_width, double kaiser_beta = 8.0);

}  // namespace nxr::frametiling

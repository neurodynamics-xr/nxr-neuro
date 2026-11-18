#pragma once
// nxr::signal — sampling a band-limited signal at a frame grid's (non-integer) positions: a block of an analytic signal read at the frames of a time rule (`nxr::selection::time_rule`, kind frames).
// Pure functions of arrays: no file, no chunk, no recording.
//
//   sincInterpolate  a block of an ANALYTIC (band-limited) signal sampled at the frames' (non-integer) sample positions: a Kaiser-windowed sinc, `half_width` samples either side (the step's halo).
//                  nxr resamples the whole run by FFT (exact for a band-limited signal); this is the same band-limited interpolation as a local filter, the one a chunk can run. Beyond the block it
//                  reads 0, as a band's taps do beyond the recording. A frame's time is its tile's START (the frame grid's origin plus its number times the hop).
#include <cstdint>
#include <vector>

namespace nxr::signal {

/** x [rows × width] (row-major, column 0 = sample 0 of the block) sampled at `positions` (samples from column 0, not whole): [rows × positions], 0 read outside the block */
std::vector<float> sincInterpolate(const float* x, std::int64_t rows, std::int64_t width, const std::vector<double>& positions, int half_width, double kaiser_beta = 8.0);

}  // namespace nxr::signal

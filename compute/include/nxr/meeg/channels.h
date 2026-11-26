// nxr::meeg — where a channel IS, as one point: the position a topography, a sensor surface or a neighbourhood places a channel at.
//   channelPositions(points, coils, indptr, indices, coilPoints)   Brainstorm's rule (GetChannelPositions; nxr geom.topography.channel_positions; the Studio's
//     channelPositions): a channel with coils is the mean of its first `coilPoints` coil points (all of them when it has fewer), one without coils its own stored
//     point. For a CTF axial gradiometer (eight integration points: four on the pickup coil, four on the reference coil) the four are the pickup coil's: the
//     place the field is measured, not the first integration point alone.
//   goodChannelsOfType(channelTypes, channelQuality, sensorType)   which channels a sensor surface is made of: the good channels of one sensor type, MEG
//     (magnetometers and gradiometers: MEG, MEG MAG, MEG GRAD; never MEG REF) or EEG, never the two together.
// PURE: arrays in, arrays out. Positions in the coils' unit and frame.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::meeg {

/** points [M x 3] row-major (each channel's stored point); coils [Q x 3] row-major (every coil / integration point); indptr [M + 1] and indices [nnz]: the coils
 *  of channel m are coils[indices[indptr[m] .. indptr[m + 1])] in their stored order (the rows of the coil-weights operator, CSR). Empty coils, indptr and
 *  indices: every channel keeps its point. coilPoints >= 1 (Brainstorm: 4). Returns [M x 3]. Refuses (std::invalid_argument) arrays that do not agree. */
std::vector<double> channelPositions(const std::vector<double>& points, const std::vector<double>& coils, const std::vector<std::int32_t>& indptr,
                                     const std::vector<std::int32_t>& indices, int coilPoints = 4);

/** the GOOD channels of one sensor type, as channel indices in ascending order. channelTypes[i] and channelQuality[i] are channel i's type name (Brainstorm's:
 *  "MEG GRAD", "EEG", ...) and quality name ("good" or "bad"). sensorType "MEG" takes MEG, MEG MAG and MEG GRAD (never MEG REF); "EEG" takes EEG. No default: the
 *  caller names the type. Refuses (std::invalid_argument) arrays of different lengths, another sensorType, a quality other than good or bad, and fewer than four
 *  channels chosen (a surface needs at least a few triangles). */
std::vector<std::int32_t> goodChannelsOfType(const std::vector<std::string>& channelTypes, const std::vector<std::string>& channelQuality, const std::string& sensorType);

}  // namespace nxr::meeg

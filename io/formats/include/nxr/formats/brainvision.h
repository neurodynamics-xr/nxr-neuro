// BrainVision (.vhdr header, .vmrk markers, .eeg binary data): header, markers and block reads.
//
// - Binary data only (DataFormat=BINARY): INT_16, INT_32 and IEEE_FLOAT_32, MULTIPLEXED (all channels of a sample together: one read per block) or
//   VECTORIZED (a channel's samples together: one read per channel). ASCII data and other binary formats are refused with the reason.
// - A channel's resolution times the stored value is in the channel's unit (µV when none); values are converted to SI.
// - [Coordinates] (radius, theta, phi in degrees) become Cartesian locations, x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta),
//   in the file's radius unit (unstated: `position_unit` is "").
// - Markers become annotations labelled "<type>/<description>" (the commas the file escapes as \1 restored); onset = (position - 1) / rate,
//   duration = size / rate (size in data points). `New Segment` markers are annotations too, and `info["segment_starts"]` lists their sample
//   positions: the data stay contiguous in the file, so a segment boundary is not a gap of samples (`gaps` is empty).
#pragma once
#include <memory>
#include <string>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"

namespace nxr::formats::brainvision {

// `header` is the .vhdr, `markers` the .vmrk (may be null: no markers), `data` the .eeg.
std::unique_ptr<RecordingReader> open(std::shared_ptr<const ByteSource> header, std::shared_ptr<const ByteSource> markers, std::shared_ptr<const ByteSource> data);

// By path to the .vhdr: the DataFile and MarkerFile it names are looked up beside it. A missing marker file is no markers; a missing data file is an io error.
std::unique_ptr<RecordingReader> open_file(const std::string& vhdr_path);

}  // namespace nxr::formats::brainvision

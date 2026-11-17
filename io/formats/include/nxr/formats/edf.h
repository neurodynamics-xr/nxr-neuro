// EDF, EDF+, BDF and BDF+ (continuous and discontinuous): header, annotations and block reads.
//
//   auto r = nxr::formats::edf::open_file(path);      // parses the header and the annotation directory (small), never the signals
//   r->read(s0, n, c0, c1, out);                      // one read per data record it touches
//
// - Values are calibrated from each signal's physical and digital ranges and converted to SI (`uV` to volts); the file's own unit stays in
//   `Channel::file_unit`. 16-bit (EDF) and 24-bit (BDF) integers read as 32-bit floats (24-bit integers are exact in them).
// - `EDF Annotations` / `BDF Annotations` signals (time-stamped annotation lists) become `header().annotations` and are not channels.
// - EDF+D / BDF+D (discontinuous): the records' own start times place the samples on one continuous axis; samples between records are
//   `header().gaps` (spans), read as zeros and never interpolated.
// - Signals at several sampling rates are refused unless the caller names the rate to read (`OpenOptions::sampling_rate`); the other rates
//   are not channels of this reader. The refusal lists the rates.
// - The patient and recording identification fields are in `header().info` as the file has them ("patient", "recording").
#pragma once
#include <memory>
#include <string>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"

namespace nxr::formats::edf {

struct OpenOptions {
  double sampling_rate = 0.0;   // Hz of the signals to read; 0 = the file's only rate (a file with several is refused)
};

std::unique_ptr<RecordingReader> open(std::shared_ptr<const ByteSource> src, const OpenOptions& options = {});
std::unique_ptr<RecordingReader> open_file(const std::string& path, const OpenOptions& options = {});

}  // namespace nxr::formats::edf

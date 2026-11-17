// EEGLAB datasets (`.set`, with its `.fdt` when the data are kept apart), read through matio.
//
// - The header is `EEG` (a struct, or the file's top-level variables when the struct was saved with its fields spread out): channels (labels, type,
//   X Y Z), sampling rate, sample count, events (latency, duration, type) as annotations, and every other scalar and string field in `info`;
//   the ICA matrices are listed in `info` by size ("icaweights": "4x4"), not applied.
// - Latency is 1-based: onset = (latency - 1) / rate; duration is in samples. `boundary` events are annotations like the others (the data are
//   contiguous in the file, so `gaps` stays empty).
// - Units: EEGLAB stores no unit; by convention the values are microvolts, read as volts (`info["unit"]` says so).
// - Data: a `.fdt` (float32, channels x samples, sample after sample) is read by seek. Data inside the `.set` are read in blocks (a linear read
//   of the MATLAB variable) when it is uncompressed (MATLAB -v6); an HDF5 (-v7.3) file goes through matio's HDF5 reader the same way (no test of it
//   yet: there is no synthetic v7.3 generator without MATLAB). A zlib-compressed variable (-v7) cannot be read in blocks (reading the end inflates
//   everything before it, as matio measured here), so it is refused with the reason: re-save the dataset with its data in a `.fdt` file.
// - Precision follows the file: float32 data read as float32; data stored in double read as float64 (use the double overload of `read`).
// - Epoched datasets (trials > 1) are refused.
#pragma once
#include <memory>
#include <string>

#include "nxr/formats/recording.h"

namespace nxr::formats::eeglab {

// `set_path` is the .set; its data file is looked up beside it.
std::unique_ptr<RecordingReader> open_file(const std::string& set_path);

}  // namespace nxr::formats::eeglab

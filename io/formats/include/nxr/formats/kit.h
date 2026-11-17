// KIT / Yokogawa MEG recordings (`.con` continuous and `.sqd`): the directory-based header, the channel table, the digitization stored in the file and
// the sample data, read by block.
//
// Ported from MNE-Python's KIT reader (mne/io/kit/{kit,constants,coreg}.py), BSD-3-Clause, Copyright the MNE-Python contributors; see
// THIRD_PARTY_NOTICES.md. MNE-Python is the reference the goldens come from (test/golden/gen_kit.py, which writes minimal valid files from the format
// description, as MNE cannot write KIT files).
//
//   auto r = kit::open_file("sub-01_task-rest.con");    // the header and the directory, never the signals
//   r->header();                                         // channels, rate, samples, bad channels
//   r->info();                                           // every field of the file's directory and system block
//   r->read(s0, n, c0, c1, out);                         // samples [s0, s0 + n) of channels [c0, c1), in SI units (T for MEG, V otherwise), by block
//
// What is read, and how:
// - A directory at the start of the file (offset, size, maximum count and count of each block) names the system, channel, calibration, amplifier,
//   acquisition, raw data, coregistration and digitization blocks. Little-endian throughout.
// - Samples are int16 (or int32, as `adc_allocated` says), sample after sample with the channels interleaved: a block read seeks to the first sample
//   and reads `n x nchannels` values.
// - Values: MEG channels `stored x adc_range / 2^adc_stored / amp_gain x gain` (T), EEG and ECG channels with their own gain, others `stored x
//   adc_range / 2^adc_stored` (V), as MNE reads them. No stimulus channel is synthesised from the trigger channels (the trigger channels are "misc"
//   channels in the file); the importer chooses how to make events from them.
// - Evoked and epoched files, and files older than format V2R003, are refused with the reason.
// - The digitization held in the file (head shape points, fiducials, HPI coil positions in the digitizer frame, the marker coil positions in the
//   device frame) is parsed as stored, in metres; the transforms are the importer's to make.
#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"

namespace nxr::formats::kit {

namespace channel_type {   // KIT channel types
constexpr std::int32_t kNull = 0, kMagnetometer = 1, kAxialGradiometer = 2, kPlanarGradiometer = 3, kSecondOrderAxialGradiometer = 4,
                       kMagnetometerReference = 0x101, kAxialGradiometerReference = 0x102, kPlanarGradiometerReference = 0x103,
                       kSecondOrderAxialGradiometerReference = 0x104, kTrigger = -1, kEeg = -2, kEcg = -3, kEtc = -4;
}  // namespace channel_type

struct Directory { std::uint32_t offset = 0; std::int32_t size = 0, max_count = 0, count = 0; };

struct KitChannel {
  std::int32_t type = 0;
  std::array<double, 5> loc{};            // MEG channels: x, y, z in metres, then theta and phi in degrees (the sensor's axis)
  bool has_loc = false;
  std::string name;                       // the file's own name ("" when it has none)
  std::int32_t number = 0;                // misc channels: the channel number
  std::optional<double> exg_gain;         // EEG and ECG channels
  double offset = 0.0, gain = 0.0;        // the calibration block: volts offset and tesla per volt
};

struct Digitization {
  std::vector<std::pair<std::string, std::array<double, 3>>> named;   // the named points (lower-cased, as MNE looks them up): fiducials, HPI coils, electrodes
  std::vector<std::array<double, 3>> head_shape;                      // the unnamed points
  std::vector<std::array<double, 3>> elp;                             // nasion, left and right preauricular points, then the HPI coils (digitizer frame)
  std::vector<std::array<double, 3>> marker_coils;                    // the HPI coils in the device frame, one per HPI coil digitized
  std::vector<bool> marker_done;                                      // whether each HPI coil was measured in the device frame
};

struct Info {
  std::int32_t version = 0, revision = 0;
  std::int32_t system_id = 0;
  std::string system_name, model_name, comment;
  std::int32_t nchan = 0;
  std::int32_t create_time = 0, last_modified_time = 0;
  std::int32_t dewar_style = 0, fll_type = 0, trigger_type = 0, adboard_type = 0;
  double adc_range = 0.0;
  std::int32_t adc_polarity = 0, adc_allocated = 0, adc_stored = 0;
  std::int32_t amp_data = 0;
  double amp_gain = 0.0;
  double highpass = 0.0, lowpass = 0.0, notch = 0.0;
  std::int32_t acq_type = 0;                       // 1 continuous
  double sfreq = 0.0;
  std::int32_t n_samples = 0;
  std::vector<Directory> directories;
  std::vector<KitChannel> channels;
  std::vector<double> conv_factor;                  // per channel: stored value -> SI
  std::optional<Digitization> digitization;
};

class Reader : public RecordingReader {
 public:
  const RecordingHeader& header() const override { return header_; }
  const Info& info() const { return info_; }
  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override;

 private:
  friend std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource>);
  RecordingHeader header_;
  Info info_;
  std::shared_ptr<const ByteSource> src_;
  std::uint64_t data_offset_ = 0;
  std::size_t bytes_per_sample_ = 2;
};

std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource> file);
std::unique_ptr<Reader> open_file(const std::string& path);

}  // namespace nxr::formats::kit

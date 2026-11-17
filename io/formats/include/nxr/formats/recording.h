// The shape every recording format shares: a header the reader parses whole, and block reads by sample range.
//
//   auto r = nxr::formats::edf::open(source);          // parses the header and the directory only
//   r->header().channels, r->header().sampling_rate, ...
//   r->read(s0, n, c0, c1, out);                        // samples [s0, s0+n) of channels [c0, c1), in SI units
//
// A reader never loads signal data except what a call asks for: memory is bounded by the call, never by the recording, so the io chunk
// service can stream a recording of any length. Output is row after row: out[(c - c0) * n + i] is sample s0 + i of channel c.
// Precision follows the file: the formats here hold integers or 32-bit floats, read as 32-bit floats (24-bit integers are exact in them).
// Values are in SI units (a file's microvolts come out as volts); the file's own unit and its scale are kept in the channel for the
// producer record.
//
// A header holds every field the file has, identifying ones included (`info`): what an importer keeps of them is the importer's decision.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/errors.h"

namespace nxr::formats {

enum class Precision { float32, float64 };

struct Channel {
  std::string name;
  std::string kind;               // "eeg", "meg", "eog", "ecg", "emg", "stim", "misc", ... as the format says, lower case; "" when it does not
  std::string unit;               // the SI unit the read values are in ("V", "T", "" for none)
  std::string file_unit;          // the unit as the file writes it ("uV", "µV", "mV", ...)
  double scale_to_si = 1.0;       // file_unit -> unit (1e-6 for uV)
  double gain = 1.0;              // stored value -> file_unit: file_unit value = stored * gain + offset
  double offset = 0.0;
  double sampling_rate = 0.0;     // this channel's own rate (formats with one rate repeat it)
  std::string transducer;
  std::string prefilter;
  std::string reference;
  std::optional<std::array<double, 3>> location;      // as the file gives them, in the file's unit (see RecordingHeader::position_unit)
  std::optional<std::array<double, 3>> orientation;
  std::string coil_type;
  // the coordinate frame `location` and `orientation` are in, as the reader names it: "device" (the sensor array's own frame), "head" (Neuromag head),
  // "ctf-head", "sphere" (BrainVision: the unit sphere) or "eeglab"; "" when the file does not say
  std::string position_frame;
  bool bad = false;
};

struct Annotation {
  double onset = 0.0;       // seconds from the first sample
  double duration = 0.0;    // seconds (0 for an instant)
  std::string label;
  std::string channels;     // the channel(s) it names, as the file says; "" for all
};

struct Span {
  std::uint64_t first = 0;     // first sample
  std::uint64_t count = 0;     // number of samples
};

struct RecordingHeader {
  std::string format;                       // "edf", "edf+", "bdf", "bdf+", "brainvision", "eeglab", ...
  std::vector<Channel> channels;
  double sampling_rate = 0.0;               // Hz
  std::uint64_t n_samples = 0;              // per channel
  Precision precision = Precision::float32;
  std::string start_time;                   // the first sample's wall-clock time as the file gives it (ISO 8601 when it has a date), or ""
  std::vector<Annotation> annotations;
  std::vector<std::string> bad_channels;
  std::vector<Span> gaps;                   // samples the file does not hold (EDF+D, BrainVision new segments): read as zeros, never interpolated
  std::string position_unit;                // the unit of channels' location ("m", "mm", "cm", "" when unstated)
  std::map<std::string, std::string> info;  // every other header field, by the file's own name
};

class RecordingReader {
 public:
  virtual ~RecordingReader() = default;
  virtual const RecordingHeader& header() const = 0;
  // Samples [s0, s0 + n) of channels [c0, c1) into out (size (c1 - c0) * n), row after row. Throws code::out_of_range outside the recording.
  virtual void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const = 0;
  // The same in 64-bit floats: for a recording whose precision is float64 (an overriding reader gives the values as the file holds them; the float
  // overload then refuses rather than narrow). For a float32 recording it widens the float read (exactly).
  virtual void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, double* out) const {
    if (header().precision != Precision::float32) throw refused("this recording holds 64-bit values and its reader gives no double read");
    std::vector<float> tmp(static_cast<std::size_t>((c1 - c0) * n));
    read(s0, n, c0, c1, tmp.data());
    for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];
  }

 protected:
  void check_range(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1) const {
    const RecordingHeader& h = header();
    if (c0 > c1 || c1 > h.channels.size())
      throw range_error("channels [" + std::to_string(c0) + ", " + std::to_string(c1) + ") of " + std::to_string(h.channels.size()));
    if (s0 > h.n_samples || n > h.n_samples - s0)
      throw range_error("samples [" + std::to_string(s0) + ", " + std::to_string(s0) + " + " + std::to_string(n) + ") of " + std::to_string(h.n_samples));
  }
};

// Reads [s0, s0 + n) in blocks of at most `block` samples, handing each to `fn(first_sample, count, rows)` (rows as `read` gives them, for
// `count` samples). The way a streaming caller and the tests use a reader: the result must not depend on `block`.
inline void for_each_block(const RecordingReader& r, std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, std::uint64_t block,
                           const std::function<void(std::uint64_t, std::uint64_t, const float*)>& fn) {
  if (block == 0) throw range_error("a block has at least one sample");
  std::vector<float> buf;
  for (std::uint64_t done = 0; done < n;) {
    const std::uint64_t m = std::min<std::uint64_t>(block, n - done);
    buf.resize((c1 - c0) * m);
    r.read(s0 + done, m, c0, c1, buf.data());
    fn(s0 + done, m, buf.data());
    done += m;
  }
}

}  // namespace nxr::formats

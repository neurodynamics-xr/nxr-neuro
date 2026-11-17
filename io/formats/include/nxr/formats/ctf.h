// CTF MEG data sets (`<name>.ds` folders): the `.res4` header, the `.meg4` data (and its continuation files `<name>.1_meg4`, ...), the head coils
// (`.hc`), markers (`MarkerFile.mrk`), bad channels (`BadChannels`), bad segments (`bad.segments`) and the electrode and digitizer files (`.eeg`, `.pos`).
//
// Ported from MNE-Python's CTF reader (mne/io/ctf/{res4,ctf,info,hc,trans,eeg,markers}.py), BSD-3-Clause, Copyright the MNE-Python contributors; see
// THIRD_PARTY_NOTICES.md. MNE-Python is the reference the goldens come from (test/golden/gen_ctf.py, which writes minimal valid data sets from the
// format description, as MNE cannot write CTF).
//
//   ctf::DirectoryFolder ds("/data/sub-01_task-rest_meg.ds");
//   auto r = ctf::open(ds);               // the header and the sample directory, never the signals
//   r->header();                          // channels, rate, samples, annotations (markers, bad segments), bad channels
//   r->res4();                            // every field of the .res4 header, coils and compensation included
//   r->read(s0, n, c0, c1, out);          // samples [s0, s0 + n) of channels [c0, c1), in SI units (T for MEG, V otherwise), by block
//
// What is read, and how:
// - `.meg4`: a 8-byte header ("MEG41CP"), then trials of `nsamp` samples, each trial channel after channel as big-endian int32 (so a channel's samples
//   of one trial are contiguous: a block read seeks per channel and per trial, never reads a whole trial). The trials of the file (and of its
//   continuation files, in order) are one recording, as MNE reads them.
// - Values: sample = stored x (1 / (proper_gain x qgain)), as MNE reads them, in T for MEG and reference channels, V for the others. A MEG channel
//   whose coil has no position is a "misc" channel in volts (MNE's rule).
// - The system clock channel (SCLK01-177): the recording stops at the first zero of it (`OpenOptions::system_clock = truncate`, MNE's default), or
//   it is ignored.
// - Epoched and averaged data sets (pre-trigger points, or more than one average) are refused with the reason: they are not a continuous recording.
// - Every header field is parsed, identifying ones (`nf_subject_id`, `nf_operator`, dates) included: what an importer keeps is its decision.
// - Positions are the file's, in metres (CTF writes centimetres; the reader converts), in the CTF device and head frames; `transforms()` gives the
//   transforms MNE derives from the head coils.
#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"

namespace nxr::formats::ctf {

// ---- the folder -----------------------------------------------------------------------------------------------------------------------------

// The files of a data set by name (`<name>.res4` and so on, `MarkerFile.mrk`): a folder on disk or in memory.
class Folder {
 public:
  virtual ~Folder() = default;
  virtual std::vector<std::string> list() const = 0;                              // file names, sorted
  virtual std::shared_ptr<const ByteSource> open(const std::string& file) const = 0;   // null when there is no such file
  virtual std::string name() const = 0;                                           // the data set's name: the folder's name without ".ds"
};

class DirectoryFolder final : public Folder {
 public:
  explicit DirectoryFolder(std::string path);   // refuses (io) a path that is not a folder ending in ".ds"
  std::vector<std::string> list() const override;
  std::shared_ptr<const ByteSource> open(const std::string& file) const override;
  std::string name() const override { return name_; }
 private:
  std::string path_, name_;
};

class MemoryFolder final : public Folder {
 public:
  explicit MemoryFolder(std::string name) : name_(std::move(name)) {}
  void add(const std::string& file, std::string content) { files_.emplace_back(file, std::move(content)); }
  std::vector<std::string> list() const override;
  std::shared_ptr<const ByteSource> open(const std::string& file) const override;
  std::string name() const override { return name_; }
 private:
  std::string name_;
  std::vector<std::pair<std::string, std::string>> files_;
};

// ---- the .res4 header -------------------------------------------------------------------------------------------------------------------------

struct Filter {
  double freq = 0.0;
  std::int32_t klass = 0, type = 0;   // type: 1 low pass, 2 high pass
  std::int16_t npar = 0;
  std::vector<double> pars;
};

struct Coil {                          // one sensor coil: metres, unit normal, turns, area in m^2
  std::array<double, 3> pos{}, norm{};
  std::int16_t turns = 0;
  double area = 0.0;
};

struct ChannelRecord {
  std::string name;
  std::int16_t sensor_type_index = 0;  // 0 reference magnetometer, 1 reference gradiometer, 5 MEG, 9 EEG, 11 stim, others misc
  std::int16_t original_run_no = 0;
  std::int32_t coil_type = 0;
  double proper_gain = 0.0, qgain = 0.0, io_gain = 0.0, io_offset = 0.0;
  std::int16_t num_coils = 0, grad_order_no = 0;   // the software gradiometer order applied to the data
  std::array<Coil, 8> coil{}, head_coil{};         // in the CTF device frame, and in the CTF head frame
};

struct CompensationRecord {            // one row of a compensation (balancing) matrix
  std::string sensor_name;
  std::int32_t coeff_type = 0;         // the grade (G1BR 0x47314252, G2BR, G3BR, ...)
  std::int16_t ncoeff = 0;
  std::vector<std::string> sensors;
  std::vector<double> coeffs;
};

struct Res4 {
  std::string head, appname, origin, desc;
  std::int16_t nave = 0;
  std::string data_time, data_date;    // as the file has them (MNE swaps them when entered the other way round; so does the reader)
  std::int32_t nsamp = 0;              // samples per trial
  std::int16_t nchan = 0;
  double sfreq = 0.0, epoch_time = 0.0;
  std::int16_t no_trials = 0;
  std::int32_t pre_trig_pts = 0;
  std::int16_t no_trials_done = 0, no_trials_bst_message_windowlay = 0;
  std::int32_t save_trials = 0;
  unsigned char primary_trigger = 0;
  std::array<unsigned char, 8> secondary_trigger{};
  unsigned char trigger_polarity_mask = 0;
  std::int16_t trigger_mode = 0;
  std::int32_t accept_reject = 0;
  std::int16_t run_time_bst_message_windowlay = 0;
  std::int32_t zero_head = 0, artifact_mode = 0;
  std::string nf_run_name, nf_run_title, nf_instruments, nf_collect_descriptor, nf_subject_id, nf_operator;
  std::vector<unsigned char> nf_sensor_file_name;   // 60 bytes
  std::int32_t rdlen = 0;
  std::string run_desc;
  std::vector<Filter> filters;
  std::vector<ChannelRecord> channels;
  std::vector<CompensationRecord> compensation;
};

// ---- the other files ----------------------------------------------------------------------------------------------------------------------

struct CoilPoint {                     // a line group of the `.hc` file
  bool valid = false;                  // "measured"
  std::int32_t coord_frame = -1;       // 1001 CTF device (relative to dewar), 1004 CTF head (relative to head), -1 unknown
  std::int32_t kind = -1;              // 1 left ear (LPA), 2 right ear (RPA), 3 nasion, 4 spare, -1 unknown
  std::array<double, 3> r{};           // metres
  std::string descriptor;
};

struct MarkerEntry { std::int32_t trial = 0; double sync = 0.0; };   // sync: seconds from the trial's start
struct Marker { std::string label; std::vector<MarkerEntry> samples; };

struct BadSegment { std::string label; double start = 0.0, end = 0.0; };   // seconds, as the file has them

struct ElectrodePoint {                // a line of the `.eeg` file (CTF head frame, metres)
  std::int32_t id = 0;
  std::string label;
  std::array<double, 3> r{};
};

struct PolhemusPoint {                 // a line of the `.pos` file (CTF head frame, metres)
  std::string id;                      // as the file has it (a number or a name)
  std::string label;                   // the optional extra column, else ""
  std::array<double, 3> r{};
};

using Matrix4 = std::array<double, 16>;   // row-major

struct Transforms {
  std::optional<Matrix4> ctf_head_to_head;       // CTF head -> Neuromag-like head (from the head-frame nasion and ears)
  std::optional<Matrix4> ctf_dev_to_ctf_head;    // CTF device -> CTF head (least-squares rigid fit of the coil positions)
  std::optional<Matrix4> ctf_dev_to_head;        // CTF device -> head
  std::optional<Matrix4> dev_head;               // Neuromag-like device -> head
  Matrix4 ctf_dev_to_dev{};                      // CTF device -> Neuromag-like device (fixed: 45 degrees about z and 190 mm up)
};

struct OpenOptions {
  enum class SystemClock { truncate, ignore };
  SystemClock system_clock = SystemClock::truncate;
};

class Reader : public RecordingReader {
 public:
  const RecordingHeader& header() const override { return header_; }
  const Res4& res4() const { return res4_; }
  const std::vector<CoilPoint>& coils() const { return coils_; }               // every point of `.hc`, measured or not
  const std::vector<Marker>& markers() const { return markers_; }
  const std::vector<BadSegment>& bad_segments() const { return bad_segments_; }
  const std::vector<ElectrodePoint>& electrodes() const { return electrodes_; }
  const std::vector<PolhemusPoint>& polhemus() const { return polhemus_; }
  const Transforms& transforms() const { return transforms_; }
  std::int32_t compensation_grade() const { return grade_; }                   // the grade on the MEG channels
  const std::vector<std::string>& files() const { return files_; }              // every file name in the folder
  std::int64_t start_epoch_seconds() const { return start_seconds_; }           // data date and time read as UTC (MNE's rule), seconds since 1970
  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override;

 private:
  friend std::unique_ptr<Reader> open(const Folder&, const OpenOptions&);
  struct Part { std::shared_ptr<const ByteSource> src; std::uint64_t n_samp = 0, n_samp_tot = 0, offset = 0; };
  RecordingHeader header_;
  Res4 res4_;
  std::vector<CoilPoint> coils_;
  std::vector<Marker> markers_;
  std::vector<BadSegment> bad_segments_;
  std::vector<ElectrodePoint> electrodes_;
  std::vector<PolhemusPoint> polhemus_;
  Transforms transforms_;
  std::int32_t grade_ = 0;
  std::vector<std::string> files_;
  std::int64_t start_seconds_ = 0;
  std::vector<Part> parts_;
  std::vector<double> cal_;
};

std::unique_ptr<Reader> open(const Folder& folder, const OpenOptions& options = {});
std::unique_ptr<Reader> open_directory(const std::string& path, const OpenOptions& options = {});

// MNE's name cleaning: the part of a name before its first dash ("MLC11-4408" -> "MLC11").
std::string clean_name(const std::string& name);
// The 4 x 4 matrices: apply to a point, invert, compose (a * b applies b first).
std::array<double, 3> apply(const Matrix4& m, const std::array<double, 3>& p);
Matrix4 invert(const Matrix4& m);
Matrix4 multiply(const Matrix4& a, const Matrix4& b);
// The Neuromag-like head frame from the nasion and the two ears: the transform from the frame the points are in (MNE's get_ras_to_neuromag_trans).
Matrix4 neuromag_head_from_landmarks(const std::array<double, 3>& nasion, const std::array<double, 3>& lpa, const std::array<double, 3>& rpa);
// The rigid transform (rotation and translation) that best takes `from` onto `to` in the least-squares sense (unit quaternions, Horn 1987).
Matrix4 fit_rigid(const std::vector<std::array<double, 3>>& from, const std::vector<std::array<double, 3>>& to);

}  // namespace nxr::formats::ctf

// FIF (Neuromag / MNE): the tag layer and the raw MEG/EEG recording reader.
//
// Ported from MNE-Python's FIF reading (mne/_fiff/{open,tree,tag,meas_info,proj,ctf_comp,matrix}.py, mne/io/fiff/raw.py, mne/annotations.py),
// BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md. MNE-Python is the reference the goldens come from
// (test/golden/gen_fif.py).
//
//   auto r = nxr::formats::fif::open_file(path);      // the tag directory and the header, never the signals; follows a split file's parts
//   r->header();                                       // the common recording header (channels, rate, gaps, annotations, bad channels)
//   r->info();                                         // every field of the measurement info, as the file has it
//   r->read(s0, n, c0, c1, out);                       // samples [s0, s0 + n) of channels [c0, c1), in SI units, by block
//
// What is read, and how:
// - The tag layer: a FIF file is a sequence of tags (kind, type, size, next; big-endian) and nested blocks. `File` reads the file id, the tag
//   directory (the directory the file points to, else by walking the tags) and the block tree. Only tag headers and the small tags a caller asks
//   for are read; the data buffers of a raw file are never read until `read()` asks.
// - Raw files only: a file whose measurement block holds epochs, an evoked response, processed data, or no raw-data block (a covariance, a
//   forward solution, ...) is refused with the reason. MaxShield (IAS) raw data are refused unless `OpenOptions::allow_maxshield`.
// - Values: sample = stored value x (cal x range) of the channel, as MNE reads them; in the channel's unit (T, T/m, V, ... in `Channel::unit`).
//   Neither MNE nor this reader applies the channel's `unit_mul`; it is kept in `info().channels`. Active projectors are NOT applied (MNE's
//   `get_data` applies them); `info().projectors` lists them with their `active` flag, and the importer decides.
// - Data types: int16 (also the "DAU pack" form), int32, float32, float64, complex64, complex128. int16, int32 and float32 buffers read as
//   32-bit floats, float64 as 64-bit floats (`header().precision`); complex buffers are read with `read_complex` (the real reads refuse them).
//   A file may mix types between buffers.
// - Skips: `FIFF_DATA_SKIP` before the first buffer shifts the first sample (as MNE does); between buffers it is a gap: `header().gaps` spans,
//   read as zeros (as MNE fills them).
// - Split files: the part named by the file's next-file reference (`FIFF_REF_FILE_NAME`, else `FIFF_REF_FILE_NUM`) is read as the continuation,
//   and so on; the recording is one sample axis. `open_file` finds the parts next to the first file; `open` takes a resolver.
// - Annotations come from the file's annotation block, onsets in seconds from the first sample (`first_samp / sfreq` taken off, as MNE shows
//   them). Skips are in `gaps`, not duplicated as annotations.
// - Every header field is parsed (`Info`); identifying ones (`subject_info`, `experimenter`, ...) are in it too: what an importer keeps is its decision.
#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"

namespace nxr::formats::fif {

// ---- The tag layer ------------------------------------------------------------------------------------------------------------------------

namespace dtype {   // FIFFT_*: the type of a tag's data
constexpr std::uint32_t kVoid = 0, kByte = 1, kShort = 2, kInt = 3, kFloat = 4, kDouble = 5, kJulian = 6, kUShort = 7, kUInt = 8, kString = 10,
                        kDauPack16 = 16, kComplexFloat = 20, kComplexDouble = 21, kOldPack = 23, kChInfoStruct = 30, kIdStruct = 31,
                        kDirEntryStruct = 32, kDigPointStruct = 33, kCoordTransStruct = 35, kDigStringStruct = 36;
constexpr std::uint32_t kMatrix = 0x40000000u, kSparseCcs = 0x00100000u, kSparseRcs = 0x00200000u;   // matrix coding bits (high half of the type)
}  // namespace dtype

namespace tag {   // FIFF_* kinds used by the reader
constexpr std::int32_t kFileId = 100, kDirPointer = 101, kBlockId = 103, kBlockStart = 104, kBlockEnd = 105, kFreeList = 106, kNop = 108,
                       kParentBlockId = 110, kCreator = 113, kRefRole = 115, kRefFileNum = 117, kRefFileName = 118, kUtcOffset = 159,
                       kNchan = 200, kSfreq = 201, kChInfo = 203, kMeasDate = 204, kComment = 206, kDescription = 206, kNave = 207, kFirstSample = 208,
                       kLastSample = 209, kExperimenter = 212, kDigPoint = 213, kHpiSlopes = 215, kHpiNcoil = 216, kLowpass = 219,
                       kCoordTrans = 222, kHighpass = 223, kHpiCorrCoeff = 226, kName = 233, kDigString = 234, kLineFreq = 235,
                       kHpiCoilFreq = 236, kHpiCoilMoments = 240, kHpiFitGoodness = 241, kHpiFitAccept = 242, kHpiFitGoodLimit = 243,
                       kHpiFitDistLimit = 244, kHpiCoilNo = 245, kHpiCoilsUsed = 246, kHpiDigitizationOrder = 247, kChScanNo = 250,
                       kChLogicalNo = 251, kChKind = 252, kChRange = 253, kChCal = 254, kChLoc = 255, kChUnit = 256, kChUnitMul = 257,
                       kChDacqName = 258, kGantryAngle = 282, kDataBuffer = 300, kDataSkip = 301, kEpoch = 302, kChCoilType = 350,
                       kChCoordFrame = 351, kProjId = 500, kProjName = 501, kEventChannels = 600, kEventList = 601, kEventChannel = 602,
                       kEventBits = 603, kSubjId = 400, kSubjFirstName = 401, kSubjMiddleName = 402, kSubjLastName = 403, kSubjBirthDay = 404,
                       kSubjSex = 405, kSubjHand = 406, kSubjWeight = 407, kSubjHeight = 408, kSubjHisId = 410, kDacqPars = 150,
                       kDacqStim = 151, kDeviceType = 152, kDeviceModel = 153, kDeviceSerial = 154, kDeviceSite = 155, kHeLevelRaw = 156,
                       kHeliumLevel = 157, kOrigFileGuid = 158, kProjItemKind = 3411, kProjItemNvec = 3414, kProjItemVectors = 3415,
                       kProjItemChNameList = 3417, kMneRowNames = 3502, kMneColNames = 3503, kMneNrow = 3504, kMneNcol = 3505,
                       kMneCoordFrame = 3506, kMneChNameList = 3507, kMneProjItemActive = 3560, kMneCustomRef = 3567,
                       kMneBaselineMin = 3568, kMneBaselineMax = 3569, kMneCtfCompKind = 3580, kMneCtfCompData = 3581,
                       kMneCtfCompCalibrated = 3582, kMneKitSystemId = 3612, kMneIcaPcaExplainedVar = 3605, kXplotterLayout = 3501,
                       kMneEpochsDropLog = 3801;
}  // namespace tag

namespace block {   // FIFFB_*
constexpr std::int32_t kMeas = 100, kMeasInfo = 101, kRawData = 102, kProcessedData = 103, kEvoked = 104, kAspect = 105, kSubject = 106,
                       kIsotrak = 107, kHpiMeas = 108, kHpiResult = 109, kHpiCoil = 110, kContinuousData = 112, kChInfo = 113, kEvents = 115,
                       kDacqPars = 117, kRef = 118, kIasRawData = 119, kHpiSubsystem = 121, kDevice = 124, kHelium = 125, kProj = 313,
                       kProjItem = 314, kMneNamedMatrix = 357, kMneBadChannels = 359, kMneCtfComp = 370, kMneCtfCompData = 371,
                       kMneEpochs = 373, kMneAnnotations = 3810, kProcessingHistory = 900;
}  // namespace block

// A tag as the directory lists it. `pos` is the offset of the tag's 16-byte header; its data follow.
struct TagHeader {
  std::int32_t kind = 0;
  std::uint32_t type = 0;
  std::int32_t size = 0;      // bytes of data
  std::int32_t next = 0;      // 0: the next tag follows; > 0: its offset; -1: the last
  std::uint64_t pos = 0;
  std::uint64_t data_pos() const { return pos + 16; }
};

struct Id {   // FIFFT_ID_STRUCT
  std::int32_t version = 0;
  std::array<std::int32_t, 2> machid{0, 0};
  std::int32_t secs = 0;
  std::int32_t usecs = 0;
};

// The content of a tag, in the plain form its type gives. Structured types (channel info, digitisation points, coordinate transforms, ids,
// directory entries) are left as their raw big-endian `bytes` for the decoders below.
struct Value {
  std::vector<std::int64_t> ints;                  // byte, short, ushort, int, uint, julian, DAU pack (one entry per element)
  std::vector<double> reals;                       // float, double; the elements of a real matrix (row-major, as MNE reshapes them)
  std::vector<std::complex<double>> complexes;     // complex float / double; the elements of a complex matrix
  std::string text;                                // string: ISO 8859-1 read as UTF-8 (trailing NULs dropped)
  std::vector<std::int32_t> dims;                  // a matrix's dimensions (row count first)
  std::vector<unsigned char> bytes;                // a structured type's data, untouched
};

// A block of the file's tree: the tags directly in it, then its nested blocks.
struct Block {
  std::int32_t kind = 0;                 // FIFFB_*; 0 for the root
  std::vector<TagHeader> tags;
  std::vector<Block> children;
  std::optional<Id> id, parent_id;
};

// An open FIF file: its tag directory and block tree.
class File {
 public:
  explicit File(std::shared_ptr<const ByteSource> source);   // refuses a file that does not start with a file id and a directory pointer
  const Id& file_id() const { return file_id_; }
  const std::vector<TagHeader>& directory() const { return directory_; }
  const Block& tree() const { return root_; }
  const ByteSource& source() const { return *source_; }
  std::vector<unsigned char> bytes(const TagHeader& t) const;   // a tag's data, as stored
  Value value(const TagHeader& t) const;                         // the data decoded by type; throws a refusal for types it does not decode

 private:
  std::shared_ptr<const ByteSource> source_;
  Id file_id_;
  std::vector<TagHeader> directory_;
  Block root_;
};

// dir_tree_find: the blocks of a kind at or below `root`, in file order.
std::vector<const Block*> find_blocks(const Block& root, std::int32_t kind);
const TagHeader* find_tag(const Block& b, std::int32_t kind);   // the first tag of a kind directly in the block

// ---- The measurement info (every field) ---------------------------------------------------------------------------------------------------

struct ChannelInfo {   // FIFFT_CH_INFO_STRUCT, with the extended information (long names) applied
  std::int32_t scanno = 0, logno = 0, kind = 0;
  double range = 1.0, cal = 1.0;       // the file's 32-bit floats; sample = stored x cal x range
  std::int32_t coil_type = 0;          // for MEG channels the high 16 bits are the CTF compensation grade in effect
  std::array<double, 12> loc{};        // position (3), then the coil's orientation (9) or an EEG reference position (3)
  std::int32_t unit = 0, unit_mul = 0;
  std::string name;
  std::int32_t coord_frame = 0;        // FIFFV_COORD_*: from the channel's own tag, else from its kind (MEG device; EEG head)
};

struct DigPoint {
  std::int32_t kind = 0;               // FIFFV_POINT_CARDINAL 1, HPI 2, EEG 3, EXTRA 4, HEAD 5
  std::int32_t ident = 0;              // cardinal: LPA 1, NASION 2, RPA 3, INION 4
  std::array<double, 3> r{};           // metres (the file's 32-bit floats)
  std::int32_t coord_frame = 0;        // the isotrak block's frame (head by default)
};

struct CoordTrans {   // FIFFT_COORD_TRANS_STRUCT
  std::int32_t from = 0, to = 0;       // FIFFV_COORD_*
  std::array<double, 16> trans{};      // 4 x 4, row-major, translation in metres (rotation and move as the file stores them)
};

struct Projector {    // one SSP item
  std::int32_t kind = 0;
  bool active = false;
  std::string desc;
  std::int32_t nvec = 0;
  std::vector<std::string> channels;   // the channels the vectors span
  std::vector<double> vectors;         // nvec x channels.size(), row-major
  std::optional<double> explained_var;
};

struct Compensation {   // one CTF software compensation matrix
  std::int32_t ctfkind = 0;
  std::int32_t grade = 0;              // 1, 2, 3 for the CTF kinds, else the kind itself
  bool calibrated = false;             // the file's `save_calibrated` flag
  std::int32_t nrow = 0, ncol = 0;
  std::vector<std::string> row_names, col_names;
  std::vector<double> data;            // nrow x ncol, row-major, as stored
};

struct SubjectInfo {
  std::optional<std::int32_t> id, sex, hand;
  std::optional<std::string> his_id, last_name, first_name, middle_name;
  std::optional<std::int32_t> birthday_julian;   // the file's day number
  std::optional<std::string> birthday;           // ISO date computed from it
  std::optional<double> weight, height;
};

struct DeviceInfo { std::optional<std::string> type, model, serial, site; };

struct HeliumInfo {
  std::optional<double> he_level_raw, helium_level;
  std::optional<std::int32_t> gantry_angle;
  std::optional<std::string> orig_file_guid;
  std::optional<std::array<std::int32_t, 2>> meas_date;   // seconds, microseconds
};

struct HpiCoil {
  std::optional<std::int32_t> number;
  std::vector<double> epoch, slopes, corr_coeff;
  std::optional<double> coil_freq;
};

struct HpiMeasurement {
  std::optional<Id> block_id, parent_id;
  std::optional<std::string> creator;
  std::optional<double> sfreq;
  std::optional<std::int32_t> nchan, nave, ncoil, first_samp, last_samp;
  std::vector<HpiCoil> coils;
};

struct HpiResult {
  std::vector<DigPoint> dig_points;
  std::vector<std::int64_t> order, used;
  std::vector<double> moments, goodness;
  std::optional<double> good_limit, dist_limit;
  std::optional<std::int32_t> accept;
  std::optional<CoordTrans> coord_trans;
};

struct HpiSubsystem {
  std::optional<std::int32_t> ncoil;
  std::optional<std::string> event_channel;
  std::vector<std::vector<std::int64_t>> coil_event_bits;
};

struct EventList {
  std::vector<std::int64_t> channels;
  std::vector<std::int64_t> list;   // as stored (sample, from, to triplets for the acquisition's event lists)
};

// What the file has that has no dedicated field (Maxfilter processing history, cross-talk and fine-calibration data, vendor blocks): every tag
// of the blocks, with its content decoded as `Value`.
struct GenericTag { std::int32_t kind = 0; std::uint32_t type = 0; Value value; };
struct GenericBlock {
  std::int32_t kind = 0;
  std::vector<GenericTag> tags;
  std::vector<GenericBlock> children;
};

struct AnnotationRecord {
  double onset = 0.0;                  // seconds from the annotation's own origin (`orig_time`), as the file has it
  double duration = 0.0;
  std::string description;
  std::vector<std::string> channels;   // the channels it names (empty: all)
  std::string extras_json;             // the file's per-annotation extras, as JSON text ("" when none)
};

struct AnnotationSet {
  std::optional<double> orig_time;     // seconds since the epoch; nothing when the annotations are relative to the first sample
  std::vector<AnnotationRecord> items;
};

struct Info {
  Id file_id;
  std::optional<Id> meas_id;                           // the measurement id MNE picks (measurement info id, parent id, ...)
  std::int32_t nchan = 0;
  double sfreq = 0.0;
  std::optional<double> highpass, lowpass;             // nothing when absent (MNE then reads 0 Hz and sfreq / 2)
  std::optional<std::array<std::int32_t, 2>> meas_date;   // seconds since the epoch, microseconds; nothing when absent or "no date"
  std::optional<std::string> utc_offset, experimenter, description, proj_name, xplotter_layout;
  std::optional<std::int32_t> proj_id, gantry_angle, kit_system_id;
  std::optional<double> line_freq;
  std::int32_t custom_ref_applied = 0;
  std::vector<ChannelInfo> channels;
  std::vector<std::string> bads;                       // as the file lists them (names already mapped to the long names)
  std::optional<CoordTrans> dev_head_t, ctf_head_t, dev_ctf_t;
  bool has_dig = false;
  std::vector<DigPoint> dig;
  std::vector<std::string> dig_channel_names;
  std::optional<std::string> acq_pars, acq_stim;
  std::vector<Projector> projectors;
  std::vector<Compensation> compensations;
  std::optional<SubjectInfo> subject_info;
  std::optional<DeviceInfo> device_info;
  std::optional<HeliumInfo> helium_info;
  std::vector<HpiResult> hpi_results;
  std::vector<HpiMeasurement> hpi_measurements;
  std::optional<HpiSubsystem> hpi_subsystem;
  std::vector<EventList> events;
  std::vector<GenericTag> other_tags;                  // measurement-info tags no field above takes
  std::vector<GenericBlock> other_blocks;              // measurement-info blocks no field above takes (processing history, SSS data, ...)
  std::optional<AnnotationSet> annotations;
  std::int32_t first_samp = 0;                         // the first part's first sample (a sample count since the acquisition's start)
  std::int32_t compensation_grade = 0;                 // the grade in effect on the MEG channels (-1 when they disagree)
};

// ---- The raw reader -----------------------------------------------------------------------------------------------------------------------

struct OpenOptions {
  bool allow_maxshield = false;   // read internal-active-shielding (IAS) raw data, which MNE refuses by default
  std::string name;               // the first file's own name, used to build a continuation's name when the file gives only a part number
};

// Finds the byte source of the next part of a split file by the name the file gives (a file name, no directory); returns nothing when it is not there.
using PartResolver = std::function<std::shared_ptr<const ByteSource>(const std::string& name)>;

// One file of a recording (the first, or a continuation).
struct Part {
  std::string name;                // the file's name as the reference gave it ("" for the first)
  std::int32_t first_samp = 0;     // the part's own first sample
  std::uint64_t n_samples = 0;
  std::uint64_t offset = 0;        // the part's first sample on the recording's axis
};

class RawReader : public RecordingReader {
 public:
  const RecordingHeader& header() const override { return header_; }
  const Info& info() const { return info_; }
  const std::vector<Part>& parts() const { return parts_; }
  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override;
  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, double* out) const override;
  // Complex files (complex64 / complex128 buffers): the same read in complex doubles. Real files refuse it.
  void read_complex(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, std::complex<double>* out) const;
  bool is_complex() const { return complex_; }

  struct Segment {   // one data buffer (or one skipped stretch) on the recording's axis
    std::uint64_t first = 0, count = 0;
    std::size_t file = 0;            // index of the part's file
    std::uint32_t type = 0;
    std::uint64_t data_pos = 0;      // offset of the buffer's data; unused for a gap
    bool gap = false;
  };

 private:
  friend std::unique_ptr<RawReader> open(std::shared_ptr<const ByteSource>, const PartResolver&, const OpenOptions&);
  template <class T> void read_real(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, T* out) const;
  RecordingHeader header_;
  Info info_;
  std::vector<Part> parts_;
  std::vector<std::shared_ptr<const ByteSource>> sources_;
  std::vector<Segment> segments_;
  std::vector<double> cals_;         // cal x range per channel
  bool complex_ = false;
};

// `next` finds the later parts of a split file; without one, a file that names a next part is refused.
std::unique_ptr<RawReader> open(std::shared_ptr<const ByteSource> first, const PartResolver& next = nullptr, const OpenOptions& options = {});
// A path; the later parts are looked for in the same folder.
std::unique_ptr<RawReader> open_file(const std::string& path, const OpenOptions& options = {});

// Names (for the importers and the tests): the channel kind as the common header names it ("meg", "eeg", ...), the unit as an SI string, the coil
// type as MNE names it ("" when unknown), the date of a julian day number.
std::string channel_kind_name(std::int32_t fiff_kind);
std::string unit_name(std::int32_t fiff_unit);
std::string coil_type_name(std::int32_t coil_type);
std::string julian_to_iso_date(std::int32_t julian_day);

}  // namespace nxr::formats::fif

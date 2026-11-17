// FIF raw recordings: the header, the data buffers as segments of one sample axis (split files joined, skips as gaps), and block reads.
// Ported from MNE-Python's mne/io/fiff/raw.py (_read_raw_file, _read_segment_file, _get_next_fname) and mne/io/base.py (_check_raw_compatibility)
// (BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

#include "fif_info.h"
#include "fif_internal.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/fif.h"

namespace nxr::formats::fif {

std::string channel_kind_name(std::int32_t k) {
  switch (k) {
    case 1: return "meg";
    case 301: return "meg_ref";
    case 2: return "eeg";
    case 3: return "stim";
    case 102: return "bio";
    case 201: return "mcg";
    case 202: return "eog";
    case 302: return "emg";
    case 402: return "ecg";
    case 502: return "misc";
    case 602: return "resp";
    case 802: return "seeg";
    case 803: return "dbs";
    case 900: return "syst";
    case 902: return "ecog";
    case 910: return "ias";
    case 920: return "exci";
    case 1000: return "dipole_wave";
    case 1001: return "goodness_fit";
    case 1100: return "fnirs";
    case 1200: return "temperature";
    case 1300: return "gsr";
    case 1400: return "eyetrack";
    default: return "";
  }
}

std::string unit_name(std::int32_t u) {
  switch (u) {
    case -1: return "";
    case 0: return "";
    case 1: return "m";
    case 2: return "kg";
    case 3: return "s";
    case 4: return "A";
    case 5: return "K";
    case 6: return "mol";
    case 7: return "rad";
    case 8: return "sr";
    case 9: return "cd";
    case 10: return "mol/m^3";
    case 101: return "Hz";
    case 102: return "N";
    case 103: return "Pa";
    case 104: return "J";
    case 105: return "W";
    case 106: return "C";
    case 107: return "V";
    case 108: return "F";
    case 109: return "ohm";
    case 110: return "S";
    case 111: return "Wb";
    case 112: return "T";
    case 113: return "H";
    case 114: return "degC";
    case 115: return "lm";
    case 116: return "lx";
    case 117: return "V/m^2";
    case 118: return "s^2";
    case 201: return "T/m";
    case 202: return "Am";
    case 203: return "Am/m^2";
    case 204: return "Am/m^3";
    case 210: return "px";
    default: return "";
  }
}

std::string coil_type_name(std::int32_t c) {
  struct E { std::int32_t id; const char* name; };
  static const E table[] = {
      {0, "NONE"}, {1, "EEG"}, {2, "NM_122"}, {3, "NM_24"}, {4, "NM_MCG_AXIAL"}, {5, "EEG_BIPOLAR"}, {6, "EEG_CSD"}, {200, "DIPOLE"},
      {300, "FNIRS_HBO"}, {301, "FNIRS_HBR"}, {302, "FNIRS_RAW"}, {303, "FNIRS_OD"}, {304, "FNIRS_FD_AC_AMPLITUDE"}, {305, "FNIRS_FD_PHASE"},
      {306, "FNIRS_TD_GATED_AMPLITUDE"}, {307, "FNIRS_TD_MOMENTS_INTENSITY"}, {308, "FNIRS_TD_MOMENTS_MEAN"}, {309, "FNIRS_TD_MOMENTS_VARIANCE"},
      {400, "EYETRACK_POS"}, {401, "EYETRACK_PUPIL"}, {1000, "MCG_42"}, {2000, "POINT_MAGNETOMETER"}, {2001, "AXIAL_GRAD_5CM"},
      {3011, "VV_PLANAR_W"}, {3012, "VV_PLANAR_T1"}, {3013, "VV_PLANAR_T2"}, {3014, "VV_PLANAR_T3"}, {3015, "VV_PLANAR_T4"},
      {3021, "VV_MAG_W"}, {3022, "VV_MAG_T1"}, {3023, "VV_MAG_T2"}, {3024, "VV_MAG_T3"}, {3025, "VV_MAG_T4"},
      {4001, "MAGNES_MAG"}, {4002, "MAGNES_GRAD"}, {4003, "MAGNES_REF_MAG"}, {4004, "MAGNES_REF_GRAD"}, {4005, "MAGNES_OFFDIAG_REF_GRAD"},
      {5001, "CTF_GRAD"}, {5002, "CTF_REF_MAG"}, {5003, "CTF_REF_GRAD"}, {5004, "CTF_OFFDIAG_REF_GRAD"}, {6001, "KIT_GRAD"}, {6002, "KIT_REF_MAG"},
      {7001, "BABY_GRAD"}, {7002, "BABY_MAG"}, {7003, "BABY_REF_MAG"}, {7004, "BABY_REF_MAG2"}, {7501, "ARTEMIS123_GRAD"},
      {7502, "ARTEMIS123_REF_MAG"}, {7503, "ARTEMIS123_REF_GRAD"}, {8001, "QUSPIN_ZFOPM_MAG"}, {8002, "QUSPIN_ZFOPM_MAG2"},
      {8101, "FIELDLINE_OPM_MAG_GEN1"}, {8201, "KERNEL_OPM_MAG_GEN1"}, {9001, "KRISS_GRAD"}, {9101, "COMPUMEDICS_ADULT_GRAD"},
      {9102, "COMPUMEDICS_PEDIATRIC_GRAD"}};
  for (const E& e : table)
    if (e.id == c) return e.name;
  return "";
}

namespace {

constexpr std::uint64_t kSlab = 4096;   // samples read at once from a buffer: bounds the memory of a read whatever its length

std::size_t sample_bytes(std::uint32_t type) {
  switch (type) {
    case dtype::kDauPack16: case dtype::kShort: return 2;
    case dtype::kFloat: case dtype::kInt: return 4;
    case dtype::kDouble: case dtype::kComplexFloat: return 8;
    case dtype::kComplexDouble: return 16;
    default: return 0;
  }
}
bool is_complex_type(std::uint32_t t) { return t == dtype::kComplexFloat || t == dtype::kComplexDouble; }

std::string base_name(const std::string& path) {
  const std::size_t s = path.find_last_of("/\\");
  return s == std::string::npos ? path : path.substr(s + 1);
}
std::string dir_name(const std::string& path) {
  const std::size_t s = path.find_last_of("/\\");
  return s == std::string::npos ? std::string() : path.substr(0, s + 1);
}

// The next part of a split file (mne/_get_next_fname): the name the file gives; else its part number applied to this file's name.
std::string next_part_name(const File& f, const std::string& this_name) {
  std::string next;
  for (const Block* ref : find_blocks(f.tree(), block::kRef)) {
    next.clear();
    for (const TagHeader& t : ref->tags) {
      if (t.kind == tag::kRefRole) {
        const Value v = f.value(t);
        if (v.ints.empty() || v.ints[0] != 2) { next.clear(); break; }   // not the next-file role
      }
      if (t.kind != tag::kRefFileName && t.kind != tag::kRefFileNum) continue;
      if (t.kind == tag::kRefFileName) {
        next = base_name(f.value(t).text);
      } else {
        if (!next.empty()) continue;
        const Value v = f.value(t);
        if (v.ints.empty()) continue;
        const std::int64_t num = v.ints[0];
        const std::string base = this_name;
        if (base.empty()) throw refused("FIF: the file names its next part by number only, and the first file's name was not given (OpenOptions::name)");
        const std::size_t idx = base.find('.');
        std::size_t idx2 = base.rfind('-');
        if (idx == std::string::npos) throw format_error("FIF: a split file's name has no extension: " + base);
        bool digits = false;
        if (idx2 != std::string::npos && idx2 < idx) {
          const std::string num_str = base.substr(idx2 + 1, idx - idx2 - 1);
          digits = !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), [](unsigned char c) { return std::isdigit(c); });
        }
        if (!digits) idx2 = std::string::npos;
        if (idx2 == std::string::npos && num == 1) { next = base.substr(0, idx) + "-1." + base.substr(idx + 1); continue; }
        if (idx2 == std::string::npos) idx2 = idx;
        next = base.substr(0, idx2) + "-" + std::to_string(num) + "." + base.substr(idx + 1);
      }
    }
    if (!next.empty()) break;
  }
  return next;
}

struct PartData {
  std::shared_ptr<const ByteSource> src;
  std::unique_ptr<File> file;
  Info info;
};

std::string block_kinds(const Block& b) {
  std::set<std::int32_t> kinds;
  struct W { static void go(const Block& x, std::set<std::int32_t>& s) { for (const Block& c : x.children) { s.insert(c.kind); go(c, s); } } };
  W::go(b, kinds);
  std::string out;
  for (std::int32_t k : kinds) out += (out.empty() ? "" : ", ") + std::to_string(k);
  return out.empty() ? "none" : out;
}

const Block& measurement_block(const File& f) {
  const auto meas = find_blocks(f.tree(), block::kMeas);
  if (meas.empty())
    throw refused("FIF: this file has no measurement block, so it is not a raw recording (it may be a covariance, forward solution, source space, "
                  "transform or other object; blocks present: " + block_kinds(f.tree()) + ")");
  if (meas.size() > 1) throw refused("FIF: this file holds " + std::to_string(meas.size()) + " measurement blocks; only one is read");
  return *meas[0];
}

const Block& raw_block(const Block& meas, const OpenOptions& opt) {
  if (!find_blocks(meas, block::kMneEpochs).empty())
    throw refused("FIF: this file holds epochs (an epoched recording); only continuous raw recordings are read for now");
  auto raw = find_blocks(meas, block::kRawData);
  if (raw.empty()) raw = find_blocks(meas, block::kContinuousData);
  if (raw.empty()) {
    raw = find_blocks(meas, block::kIasRawData);
    if (!raw.empty() && !opt.allow_maxshield)
      throw refused("FIF: this recording was made with internal active shielding (MaxShield); read it only after Maxwell filtering, or set allow_maxshield");
  }
  if (raw.empty()) {
    if (!find_blocks(meas, block::kEvoked).empty() || !find_blocks(meas, block::kProcessedData).empty())
      throw refused("FIF: this file holds an evoked (averaged) response, not a raw recording; only continuous raw recordings are read for now");
    throw refused("FIF: this file has no raw-data block (blocks inside the measurement: " + block_kinds(meas) + "); only continuous raw recordings are read");
  }
  if (raw.size() > 1) throw refused("FIF: " + std::to_string(raw.size()) + " raw-data blocks in one file");
  return *raw[0];
}

std::string iso_utc(std::int64_t secs, std::int64_t usecs) {
  std::int64_t days = secs / 86400, rem = secs % 86400;
  if (rem < 0) { rem += 86400; --days; }
  // civil-from-days
  std::int64_t z = days + 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const std::int64_t doe = z - era * 146097;
  const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  std::int64_t y = yoe + era * 400;
  const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const std::int64_t mp = (5 * doy + 2) / 153;
  const std::int64_t d = doy - (153 * mp + 2) / 5 + 1;
  const std::int64_t m = mp < 10 ? mp + 3 : mp - 9;
  if (m <= 2) ++y;
  char buf[64];
  std::snprintf(buf, sizeof buf, "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld", static_cast<long long>(y), static_cast<long long>(m), static_cast<long long>(d),
                static_cast<long long>(rem / 3600), static_cast<long long>(rem % 3600 / 60), static_cast<long long>(rem % 60));
  std::string s = buf;
  if (usecs != 0) {
    std::snprintf(buf, sizeof buf, ".%06lld", static_cast<long long>(usecs));
    s += buf;
  }
  return s + "Z";
}

std::string num(double v) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

}  // namespace

std::unique_ptr<RawReader> open(std::shared_ptr<const ByteSource> first, const PartResolver& next, const OpenOptions& options) {
  auto reader = std::unique_ptr<RawReader>(new RawReader());
  RawReader& R = *reader;

  std::vector<PartData> parts;
  std::string this_name = options.name;
  std::shared_ptr<const ByteSource> src = std::move(first);
  std::set<std::string> seen;
  std::uint64_t axis = 0;
  std::vector<Part> part_table;

  for (std::size_t part_index = 0;; ++part_index) {
    PartData pd;
    pd.src = src;
    pd.file = std::make_unique<File>(src);
    const File& f = *pd.file;
    const Block& meas = measurement_block(f);
    const Block& raw = raw_block(meas, options);
    pd.info = read_info(f, meas);
    const Info& info = pd.info;

    if (part_index > 0) {   // mne/io/base.py _check_raw_compatibility
      const Info& a = parts[0].info;
      auto mismatch = [&](const std::string& what) {
        return format_error("FIF: part " + std::to_string(part_index + 1) + " of the split file does not match the first (" + what + ")");
      };
      if (info.nchan != a.nchan) throw mismatch("channel count");
      if (info.sfreq != a.sfreq) throw mismatch("sampling frequency");
      for (std::size_t c = 0; c < a.channels.size(); ++c) {
        if (a.channels[c].name != info.channels[c].name) {
          std::set<std::string> s1, s2;
          for (auto& x : a.channels) s1.insert(x.name);
          for (auto& x : info.channels) s2.insert(x.name);
          if (s1 != s2) throw mismatch("channel names");
          break;
        }
      }
      for (std::size_t c = 0; c < a.channels.size(); ++c)
        if (a.channels[c].cal * a.channels[c].range != info.channels[c].cal * info.channels[c].range) throw mismatch("channel calibration");
      if (std::set<std::string>(a.bads.begin(), a.bads.end()) != std::set<std::string>(info.bads.begin(), info.bads.end())) throw mismatch("bad channels");
      if (a.projectors.size() != info.projectors.size()) throw mismatch("projectors");
    }

    // the raw-data block's tags (mne/io/fiff/raw.py): an optional first sample, an optional skip, then buffers and skips
    const std::vector<TagHeader>& tags = raw.tags;
    std::size_t k = 0;
    std::int32_t first_samp = 0, first_skip = 0;
    auto need = [&](std::size_t i) {
      if (i >= tags.size()) throw format_error("FIF: could not read the data, perhaps this is a corrupt file");
    };
    need(k);
    if (tags[k].kind == tag::kFirstSample) {
      first_samp = static_cast<std::int32_t>(f.value(tags[k]).ints.at(0));
      ++k;
      need(k);
    }
    if (tags[k].kind == tag::kDataSkip) {
      first_skip = static_cast<std::int32_t>(f.value(tags[k]).ints.at(0));
      ++k;
      need(k);
    }
    const std::size_t nchan = static_cast<std::size_t>(info.nchan);
    std::int64_t nskip = 0;
    std::int64_t pos_in_part = first_samp;   // MNE's running first_samp, advanced by every buffer
    const std::uint64_t part_start_axis = axis;
    std::int32_t part_first_samp = first_samp;
    bool shifted = false;
    for (; k < tags.size(); ++k) {
      const TagHeader& t = tags[k];
      if (t.kind == tag::kDataBuffer) {
        const std::size_t es = sample_bytes(t.type);
        if (es == 0) throw refused("FIF: cannot handle data buffers of type " + std::to_string(t.type));
        if (t.size < 0 || static_cast<std::uint64_t>(t.size) % (es * nchan) != 0)
          throw format_error("FIF: the data buffer at " + std::to_string(t.pos) + " holds " + std::to_string(t.size) + " bytes, not a whole number of samples of " + std::to_string(nchan) + " channels");
        if (t.data_pos() + static_cast<std::uint64_t>(t.size) > src->size())
          throw format_error("FIF: the data buffer at " + std::to_string(t.pos) + " runs past the end of the file (truncated file)");
        const std::uint64_t nsamp = static_cast<std::uint64_t>(t.size) / (es * nchan);
        if (first_skip > 0 && !shifted) {   // a skip before the first buffer moves the first sample (MNE)
          pos_in_part += static_cast<std::int64_t>(nsamp) * first_skip;
          part_first_samp = static_cast<std::int32_t>(pos_in_part);
          first_skip = 0;
          shifted = true;
        }
        if (nskip > 0) {
          RawReader::Segment g;
          g.first = axis;
          g.count = nsamp * static_cast<std::uint64_t>(nskip);
          g.file = part_index;
          g.gap = true;
          R.segments_.push_back(g);
          axis += g.count;
          pos_in_part += static_cast<std::int64_t>(g.count);
          nskip = 0;
        }
        RawReader::Segment s;
        s.first = axis;
        s.count = nsamp;
        s.file = part_index;
        s.type = t.type;
        s.data_pos = t.data_pos();
        R.segments_.push_back(s);
        axis += nsamp;
        pos_in_part += static_cast<std::int64_t>(nsamp);
        if (is_complex_type(t.type)) R.complex_ = true;
      } else if (t.kind == tag::kDataSkip) {
        nskip = f.value(t).ints.at(0);
      }
    }
    if (axis == part_start_axis && part_index == 0) throw format_error("FIF: the raw-data block holds no data buffers");
    Part p;
    p.name = this_name;
    p.first_samp = part_first_samp;
    p.n_samples = axis - part_start_axis;
    p.offset = part_start_axis;
    part_table.push_back(p);

    const std::string next_name = next_part_name(f, this_name);
    parts.push_back(std::move(pd));
    if (next_name.empty()) break;
    if (!seen.insert(next_name).second || part_index > 100000) throw format_error("FIF: the split file's parts refer to each other in a loop (" + next_name + ")");
    std::shared_ptr<const ByteSource> nsrc = next ? next(next_name) : nullptr;
    if (!nsrc) throw io_error("FIF: this recording continues in the file '" + next_name + "', which was not found" + (next ? "" : " (no way to look for it was given)"));
    src = std::move(nsrc);
    this_name = next_name;
  }

  // ---- assemble ----
  R.info_ = parts[0].info;
  Info& info = R.info_;
  info.first_samp = part_table[0].first_samp;
  info.annotations = read_annotations(*parts[0].file, parts[0].file->tree());
  for (auto& p : parts) R.sources_.push_back(p.src);
  R.parts_ = part_table;
  const std::size_t nch = info.channels.size();
  R.cals_.resize(nch);
  for (std::size_t c = 0; c < nch; ++c) R.cals_[c] = static_cast<double>(info.channels[c].cal) * static_cast<double>(info.channels[c].range);

  RecordingHeader& h = R.header_;
  h.format = "fif";
  h.sampling_rate = info.sfreq;
  h.n_samples = axis;
  h.precision = Precision::float32;
  for (const auto& s : R.segments_)
    if (!s.gap && (s.type == dtype::kDouble || s.type == dtype::kComplexDouble)) h.precision = Precision::float64;
  std::set<std::string> names;
  for (const auto& c : info.channels) names.insert(c.name);
  for (const std::string& b : info.bads)
    if (names.count(b)) h.bad_channels.push_back(b);
  const std::set<std::string> bad_set(h.bad_channels.begin(), h.bad_channels.end());
  for (const ChannelInfo& c : info.channels) {
    Channel ch;
    ch.name = c.name;
    ch.kind = channel_kind_name(c.kind);
    ch.unit = unit_name(c.unit);
    ch.file_unit = ch.unit;
    ch.scale_to_si = 1.0;
    ch.gain = static_cast<double>(c.cal) * static_cast<double>(c.range);
    ch.offset = 0.0;
    ch.sampling_rate = info.sfreq;
    std::string pre;
    if (info.highpass || info.lowpass) {
      pre = "HP:" + num(info.highpass.value_or(0.0)) + "Hz LP:" + num(info.lowpass.value_or(info.sfreq / 2.0)) + "Hz";
      ch.prefilter = pre;
    }
    ch.coil_type = coil_type_name(c.coil_type & 0xFFFF);
    const bool has_pos = std::isfinite(c.loc[0]) && std::isfinite(c.loc[1]) && std::isfinite(c.loc[2]) && (c.loc[0] != 0 || c.loc[1] != 0 || c.loc[2] != 0);
    if (has_pos) ch.location = std::array<double, 3>{c.loc[0], c.loc[1], c.loc[2]};
    if (has_pos) ch.position_frame = c.coord_frame == 1 ? "device" : (c.coord_frame == 4 ? "head" : "");
    if ((c.kind == 1 || c.kind == 301) && has_pos) ch.orientation = std::array<double, 3>{c.loc[9], c.loc[10], c.loc[11]};   // the coil's normal (z axis of its frame)
    ch.bad = bad_set.count(c.name) > 0;
    h.channels.push_back(std::move(ch));
  }
  h.position_unit = "m";
  // gaps
  for (const auto& s : R.segments_)
    if (s.gap) h.gaps.push_back(Span{s.first, s.count});
  // start time: the measurement date plus the first sample's offset
  if (info.meas_date) {
    const std::int64_t off_us = static_cast<std::int64_t>(std::llround(static_cast<double>(info.first_samp) / info.sfreq * 1e6));
    std::int64_t us = static_cast<std::int64_t>((*info.meas_date)[1]) + off_us;
    std::int64_t secs = (*info.meas_date)[0] + us / 1000000;
    us %= 1000000;
    if (us < 0) { us += 1000000; --secs; }
    h.start_time = iso_utc(secs, us);
  }
  // annotations: onsets from the first sample
  if (info.annotations) {
    const AnnotationSet& as = *info.annotations;
    const double first_time = static_cast<double>(info.first_samp) / info.sfreq;
    double shift = 0.0;
    if (as.orig_time && info.meas_date)
      shift = *as.orig_time - (static_cast<double>((*info.meas_date)[0]) + static_cast<double>((*info.meas_date)[1]) * 1e-6);
    for (const AnnotationRecord& a : as.items) {
      Annotation o;
      o.onset = a.onset + shift - first_time;
      o.duration = a.duration;
      o.label = a.description;
      for (std::size_t i = 0; i < a.channels.size(); ++i) o.channels += (i ? "," : "") + a.channels[i];
      h.annotations.push_back(std::move(o));
    }
  }
  // the scalar fields, by the file's own names
  auto& m = h.info;
  m["first_samp"] = std::to_string(info.first_samp);
  m["n_parts"] = std::to_string(parts.size());
  m["custom_ref_applied"] = std::to_string(info.custom_ref_applied);
  m["compensation_grade"] = std::to_string(info.compensation_grade);
  m["highpass"] = num(info.highpass.value_or(0.0));
  m["lowpass"] = num(info.lowpass.value_or(info.sfreq / 2.0));
  if (info.meas_date) m["meas_date"] = iso_utc((*info.meas_date)[0], (*info.meas_date)[1]);
  if (info.line_freq) m["line_freq"] = num(*info.line_freq);
  if (info.gantry_angle) m["gantry_angle"] = std::to_string(*info.gantry_angle);
  if (info.utc_offset) m["utc_offset"] = *info.utc_offset;
  if (info.experimenter) m["experimenter"] = *info.experimenter;
  if (info.description) m["description"] = *info.description;
  if (info.proj_name) m["proj_name"] = *info.proj_name;
  if (info.proj_id) m["proj_id"] = std::to_string(*info.proj_id);
  if (info.subject_info) {
    const SubjectInfo& s = *info.subject_info;
    if (s.id) m["subject_info.id"] = std::to_string(*s.id);
    if (s.his_id) m["subject_info.his_id"] = *s.his_id;
    if (s.last_name) m["subject_info.last_name"] = *s.last_name;
    if (s.first_name) m["subject_info.first_name"] = *s.first_name;
    if (s.middle_name) m["subject_info.middle_name"] = *s.middle_name;
    if (s.birthday) m["subject_info.birthday"] = *s.birthday;
    if (s.sex) m["subject_info.sex"] = std::to_string(*s.sex);
    if (s.hand) m["subject_info.hand"] = std::to_string(*s.hand);
    if (s.weight) m["subject_info.weight"] = num(*s.weight);
    if (s.height) m["subject_info.height"] = num(*s.height);
  }
  return reader;
}

std::unique_ptr<RawReader> open_file(const std::string& path, const OpenOptions& options) {
  OpenOptions o = options;
  if (o.name.empty()) o.name = base_name(path);
  const std::string dir = dir_name(path);
  auto resolver = [dir](const std::string& name) -> std::shared_ptr<const ByteSource> {
    const std::string p = dir + name;
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return nullptr;
    std::fclose(f);
    return std::make_shared<FileSource>(p);
  };
  return open(std::make_shared<FileSource>(path), resolver, o);
}

// ---- reading ---------------------------------------------------------------------------------------------------------------------------------

namespace {

template <class Out>
inline Out decode_real(std::uint32_t type, const unsigned char* p) {
  switch (type) {
    case dtype::kDauPack16: case dtype::kShort: return static_cast<Out>(load_i16(p, false));
    case dtype::kInt: return static_cast<Out>(load_i32(p, false));
    case dtype::kFloat: return static_cast<Out>(load_f32(p, false));
    default: return static_cast<Out>(load_f64(p, false));
  }
}

}  // namespace

template <class T>
void RawReader::read_real(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, T* out) const {
  check_range(s0, n, c0, c1);
  if (complex_) throw refused("FIF: this recording holds complex values; use read_complex");
  const std::size_t nch = info_.channels.size(), width = c1 - c0;
  std::vector<unsigned char> buf;
  // the first segment that reaches s0
  auto it = std::upper_bound(segments_.begin(), segments_.end(), s0, [](std::uint64_t v, const Segment& s) { return v < s.first + s.count; });
  std::uint64_t done = 0;
  for (; done < n && it != segments_.end(); ++it) {
    const std::uint64_t at = s0 + done;
    const std::uint64_t local0 = at - it->first;
    const std::uint64_t take = std::min<std::uint64_t>(it->count - local0, n - done);
    if (it->gap) {
      for (std::size_t c = 0; c < width; ++c) std::fill(out + c * n + done, out + c * n + done + take, T(0));
      done += take;
      continue;
    }
    const std::size_t es = sample_bytes(it->type);
    const ByteSource& src = *sources_[it->file];
    for (std::uint64_t got = 0; got < take;) {
      const std::uint64_t m = std::min<std::uint64_t>(kSlab, take - got);
      buf.resize(static_cast<std::size_t>(m * nch * es));
      src.read(it->data_pos + (local0 + got) * nch * es, buf.size(), buf.data());
      for (std::uint64_t i = 0; i < m; ++i) {
        const unsigned char* row = buf.data() + i * nch * es;
        for (std::size_t c = 0; c < width; ++c) {
          const double v = decode_real<double>(it->type, row + (c0 + c) * es) * cals_[c0 + c];
          out[c * n + done + got + i] = static_cast<T>(v);
        }
      }
      got += m;
    }
    done += take;
  }
}

void RawReader::read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const {
  if (header_.precision == Precision::float64) {
    check_range(s0, n, c0, c1);
    throw refused("FIF: this recording holds 64-bit values and the 32-bit read would narrow them; read it in doubles");
  }
  read_real<float>(s0, n, c0, c1, out);
}

void RawReader::read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, double* out) const {
  if (header_.precision == Precision::float64) {
    read_real<double>(s0, n, c0, c1, out);
    return;
  }
  std::vector<float> tmp(static_cast<std::size_t>((c1 - c0) * n));   // check_range is inside the float read
  read_real<float>(s0, n, c0, c1, tmp.data());
  for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i];
}

void RawReader::read_complex(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, std::complex<double>* out) const {
  check_range(s0, n, c0, c1);
  if (!complex_) throw refused("FIF: this recording holds real values; use read");
  const std::size_t nch = info_.channels.size(), width = c1 - c0;
  std::vector<unsigned char> buf;
  auto it = std::upper_bound(segments_.begin(), segments_.end(), s0, [](std::uint64_t v, const Segment& s) { return v < s.first + s.count; });
  std::uint64_t done = 0;
  for (; done < n && it != segments_.end(); ++it) {
    const std::uint64_t local0 = s0 + done - it->first;
    const std::uint64_t take = std::min<std::uint64_t>(it->count - local0, n - done);
    if (it->gap) {
      for (std::size_t c = 0; c < width; ++c) std::fill(out + c * n + done, out + c * n + done + take, std::complex<double>(0, 0));
      done += take;
      continue;
    }
    const std::size_t es = sample_bytes(it->type);
    const ByteSource& src = *sources_[it->file];
    for (std::uint64_t got = 0; got < take;) {
      const std::uint64_t m = std::min<std::uint64_t>(kSlab, take - got);
      buf.resize(static_cast<std::size_t>(m * nch * es));
      src.read(it->data_pos + (local0 + got) * nch * es, buf.size(), buf.data());
      for (std::uint64_t i = 0; i < m; ++i) {
        const unsigned char* row = buf.data() + i * nch * es;
        for (std::size_t c = 0; c < width; ++c) {
          const unsigned char* q = row + (c0 + c) * es;
          std::complex<double> v;
          if (it->type == dtype::kComplexFloat) v = {load_f32(q, false), load_f32(q + 4, false)};
          else if (it->type == dtype::kComplexDouble) v = {load_f64(q, false), load_f64(q + 8, false)};
          else v = {decode_real<double>(it->type, q), 0.0};
          out[c * n + done + got + i] = v * cals_[c0 + c];
        }
      }
      got += m;
    }
    done += take;
  }
}

template void RawReader::read_real<float>(std::uint64_t, std::uint64_t, std::size_t, std::size_t, float*) const;
template void RawReader::read_real<double>(std::uint64_t, std::uint64_t, std::size_t, std::size_t, double*) const;

}  // namespace nxr::formats::fif

// KIT / Yokogawa recordings. Ported from MNE-Python's mne/io/kit/{kit,constants}.py (BSD-3-Clause, Copyright the MNE-Python contributors; see
// THIRD_PARTY_NOTICES.md).
#include "nxr/formats/kit.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::kit {

namespace {

constexpr int kDirIndexSystem = 1, kDirIndexChannels = 4, kDirIndexCalibration = 5, kDirIndexAmpFilter = 7, kDirIndexAcqCond = 8, kDirIndexRawData = 9,
              kDirIndexCoreg = 12, kDirIndexDigPoints = 26;
constexpr int kGains[8] = {1, 2, 5, 10, 20, 50, 100, 200};
const std::map<int, std::array<int, 3>> kFllSettings = {{0, {1, 1, 1}}, {10, {1, 1, 1}}, {20, {1, 1, 1}}, {50, {2, 1, 1}}, {60, {2, 1, 1}},
                                                         {100, {3, 3, 3}}, {101, {1, 3, 2}}, {120, {3, 3, 3}}, {200, {4, 4, 3}}, {300, {2, 2, 2}}};
const std::map<int, std::vector<double>> kHpfs = {{1, {0, 1, 3, 3}}, {2, {0, 0.03, 0.1, 0.3, 1, 3, 10, 30}}, {3, {0, 0.03, 0.1, 0.3, 1, 3, 10, 30}}, {4, {0, 1, 3, 10, 30, 100, 200, 500}}};
const std::map<int, std::vector<double>> kLpfs = {{1, {10, 20, 50, 100, 200, 500, 1000, 2000}}, {2, {10, 20, 50, 100, 200, 500, 1000, 2000}},
                                                  {3, {10, 20, 50, 100, 200, 500, 1000, 10000}}, {4, {10, 30, 100, 300, 1000, 2000, 5000, 10000}}};
const std::map<int, std::vector<double>> kBefs = {{1, {0, 50, 60, 60}}, {2, {0, 0, 0}}, {3, {0, 60, 50, 50}}};

bool is_meg(std::int32_t t) { return (t >= 1 && t <= 4) || (t >= 0x101 && t <= 0x104); }
bool is_reference(std::int32_t t) { return t >= 0x101 && t <= 0x104; }

// the channel-name width of each type that has a name
int name_width(std::int32_t t) {
  switch (t) {
    case channel_type::kMagnetometer: case channel_type::kAxialGradiometer: return 6;
    case channel_type::kTrigger: return 32;
    case channel_type::kEeg: return 8;
    case channel_type::kEcg: case channel_type::kEtc: return 32;
    default: return 0;
  }
}

struct Cur {
  const ByteSource& s;
  std::uint64_t p = 0;
  void need(std::size_t n) const { if (p + n > s.size()) throw format_error("KIT: the file is truncated (a read of " + std::to_string(n) + " bytes at " + std::to_string(p) + " of " + std::to_string(s.size()) + ")"); }
  std::int32_t i32() { need(4); unsigned char b[4]; s.read(p, 4, b); p += 4; return load_i32(b, true); }
  std::uint32_t u32() { need(4); unsigned char b[4]; s.read(p, 4, b); p += 4; return load_u32(b, true); }
  double f64() { need(8); unsigned char b[8]; s.read(p, 8, b); p += 8; return load_f64(b, true); }
  std::string name(std::size_t n) {
    need(n);
    std::vector<unsigned char> b(n);
    s.read(p, n, b.data());
    p += n;
    std::size_t e = 0;
    while (e < n && b[e] != 0) ++e;
    return std::string(reinterpret_cast<const char*>(b.data()), e);
  }
  void skip(std::uint64_t n) { p += n; }
};

std::string stamp_to_iso(std::int64_t secs) {
  const std::int64_t days = secs >= 0 ? secs / 86400 : -((-secs + 86399) / 86400);
  const std::int64_t rem = secs - days * 86400;
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
  char buf[40];
  std::snprintf(buf, sizeof buf, "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld", static_cast<long long>(y), static_cast<long long>(m), static_cast<long long>(d),
                static_cast<long long>(rem / 3600), static_cast<long long>(rem % 3600 / 60), static_cast<long long>(rem % 60));
  return buf;
}

std::string lower(std::string s) { for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }

}  // namespace

std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource> file) {
  auto rd = std::unique_ptr<Reader>(new Reader());
  Reader& r = *rd;
  r.src_ = file;
  Info& I = r.info_;
  Cur c{*file};
  // directories
  {
    Directory d0;
    d0.offset = c.u32(); d0.size = c.i32(); d0.max_count = c.i32(); d0.count = c.i32();
    if (d0.count < 10 || d0.count > 1000) throw format_error("KIT: a directory of " + std::to_string(d0.count) + " entries is not a KIT file's");
    I.directories.push_back(d0);
    for (std::int32_t i = 1; i < d0.count; ++i) {
      Directory d;
      d.offset = c.u32(); d.size = c.i32(); d.max_count = c.i32(); d.count = c.i32();
      I.directories.push_back(d);
    }
  }
  auto dir = [&](int i) -> const Directory& {
    if (i >= static_cast<int>(I.directories.size())) throw format_error("KIT: the directory has no entry " + std::to_string(i));
    return I.directories[static_cast<std::size_t>(i)];
  };
  // system block
  c.p = dir(kDirIndexSystem).offset;
  I.version = c.i32();
  I.revision = c.i32();
  if (I.version < 2 || (I.version == 2 && I.revision < 3))
    throw refused("KIT: SQD file format V" + std::to_string(I.version) + "R" + (I.revision < 100 ? (I.revision < 10 ? "00" : "0") : "") + std::to_string(I.revision) + " is not officially supported (read and re-save the data with the KIT MEG Laboratory application)");
  I.system_id = c.i32();
  I.system_name = c.name(128);
  I.model_name = c.name(128);
  I.nchan = c.i32();
  if (I.nchan <= 0 || I.nchan > 100000) throw format_error("KIT: " + std::to_string(I.nchan) + " channels");
  I.comment = c.name(256);
  I.create_time = c.i32();
  I.last_modified_time = c.i32();
  c.skip(12);
  I.dewar_style = c.i32();
  c.skip(12);
  I.fll_type = c.i32();
  c.skip(12);
  I.trigger_type = c.i32();
  c.skip(12);
  I.adboard_type = c.i32();
  c.skip(116);
  if (I.version < 2 || (I.version == 2 && I.revision <= 3)) I.adc_range = static_cast<double>(c.i32());
  else I.adc_range = c.f64();
  I.adc_polarity = c.i32();
  I.adc_allocated = c.i32();
  I.adc_stored = c.i32();
  if (I.adc_allocated % 8 != 0 || (I.adc_allocated != 16 && I.adc_allocated != 32)) throw refused("KIT: " + std::to_string(I.adc_allocated) + " bits allocated per sample (only 16 and 32 are read)");
  r.bytes_per_sample_ = static_cast<std::size_t>(I.adc_allocated / 8);
  if (!kFllSettings.count(I.fll_type)) throw refused("KIT: unknown filter settings (FLL type " + std::to_string(I.fll_type) + ") for system '" + I.system_name + "'");

  // channels
  const Directory& cd = dir(kDirIndexChannels);
  std::vector<double> exg_gains;
  for (std::int32_t i = 0; i < I.nchan; ++i) {
    c.p = static_cast<std::uint64_t>(cd.offset) + static_cast<std::uint64_t>(cd.size) * static_cast<std::uint64_t>(i);
    KitChannel ch;
    ch.type = c.i32();
    if (I.system_id == 52 && i < 160 && ch.type == channel_type::kNull) ch.type = channel_type::kMagnetometerReference;   // system 52 mislabelled its reference channels
    if (is_meg(ch.type)) {
      for (double& x : ch.loc) x = c.f64();
      ch.has_loc = true;
      if (name_width(ch.type)) { c.skip(16); ch.name = c.name(static_cast<std::size_t>(name_width(ch.type))); }
    } else if (ch.type == channel_type::kTrigger || ch.type == channel_type::kEeg || ch.type == channel_type::kEcg || ch.type == channel_type::kEtc) {
      ch.number = c.i32();
      c.skip(4);
      ch.name = c.name(static_cast<std::size_t>(name_width(ch.type)));
      if (ch.type == channel_type::kEeg || ch.type == channel_type::kEcg) {
        c.skip(ch.type == channel_type::kEeg ? 6 : 8);
        ch.exg_gain = c.f64();
        exg_gains.push_back(*ch.exg_gain);
      }
    } else if (ch.type != channel_type::kNull) {
      throw format_error("KIT: unknown channel type " + std::to_string(ch.type));
    }
    I.channels.push_back(std::move(ch));
  }
  // calibration: (offset [V], gain [T/V]) per channel
  c.p = dir(kDirIndexCalibration).offset;
  for (KitChannel& ch : I.channels) { ch.offset = c.f64(); ch.gain = c.f64(); }
  for (const KitChannel& ch : I.channels) if (ch.offset != 0.0) throw refused("KIT: a channel with a nonzero calibration offset is not read");
  // amplifier gain and filters
  c.p = dir(kDirIndexAmpFilter).offset;
  I.amp_data = c.i32();
  const std::uint32_t amp = static_cast<std::uint32_t>(I.amp_data);
  int hpf, lpf, bef;
  if (I.fll_type >= 100) {
    const int g1 = static_cast<int>((amp & 0x00007000u) >> 12), g2 = static_cast<int>((amp & 0x70000000u) >> 28), g3 = static_cast<int>((amp & 0x07000000u) >> 24);
    I.amp_gain = static_cast<double>(kGains[g1]) * kGains[g2] * kGains[g3];
    hpf = static_cast<int>((amp & 0x00000700u) >> 8);
    lpf = static_cast<int>((amp & 0x00070000u) >> 16);
    bef = static_cast<int>(amp & 0x00000003u);
  } else {
    const int in = static_cast<int>((amp & 0x1800u) >> 11), out = static_cast<int>(amp & 0x0007u);
    I.amp_gain = static_cast<double>(kGains[in]) * kGains[out];
    hpf = static_cast<int>((amp & 0x007u) >> 4);
    lpf = static_cast<int>((amp & 0x0700u) >> 8);
    bef = static_cast<int>((amp & 0xC000u) >> 14);
  }
  const auto opts = kFllSettings.at(I.fll_type);
  auto pick = [](const std::map<int, std::vector<double>>& t, int k, int i, const char* what) {
    const auto& v = t.at(k);
    if (i < 0 || i >= static_cast<int>(v.size())) throw format_error(std::string("KIT: the amplifier word selects a ") + what + " setting that the system's table does not have");
    return v[static_cast<std::size_t>(i)];
  };
  I.highpass = pick(kHpfs, opts[0], hpf, "high-pass");
  I.lowpass = pick(kLpfs, opts[1], lpf, "low-pass");
  I.notch = pick(kBefs, opts[2], bef, "notch");
  // acquisition conditions
  c.p = dir(kDirIndexAcqCond).offset;
  I.acq_type = c.i32();
  I.sfreq = c.f64();
  if (I.acq_type == 1) {
    c.skip(4);
    I.n_samples = c.i32();
  } else if (I.acq_type == 2 || I.acq_type == 3) {
    throw refused(std::string("KIT: an ") + (I.acq_type == 2 ? "evoked" : "epoched") + " file is not read; only continuous recordings are");
  } else {
    throw format_error("KIT: invalid acquisition type " + std::to_string(I.acq_type) + " (neither continuous nor epoched data)");
  }
  if (I.sfreq <= 0.0 || I.n_samples < 0) throw format_error("KIT: a sampling rate of " + std::to_string(I.sfreq) + " and " + std::to_string(I.n_samples) + " samples");
  // digitization and coregistration
  if (static_cast<int>(I.directories.size()) > kDirIndexDigPoints) {
    const Directory& dd = dir(kDirIndexDigPoints);
    const Directory& cr = dir(kDirIndexCoreg);
    if (dd.count > 0 && cr.count > 0) {
      Digitization dg;
      std::map<std::string, std::array<double, 3>> by_name;
      c.p = dd.offset;
      for (std::int32_t i = 0; i < dd.count; ++i) {
        std::string nm = lower(c.name(8));
        while (!nm.empty() && (nm.back() == ' ')) nm.pop_back();
        std::size_t lead = 0;
        while (lead < nm.size() && nm[lead] == ' ') ++lead;
        nm = nm.substr(lead);
        std::array<double, 3> rr{c.f64(), c.f64(), c.f64()};
        if (!nm.empty()) { if (by_name.count(nm)) throw format_error("KIT: the digitization names '" + nm + "' twice"); by_name[nm] = rr; dg.named.emplace_back(nm, rr); }
        else dg.head_shape.push_back(rr);
      }
      for (const char* key : {"fidnz", "fidt9", "fidt10", "hpi_1", "hpi_2", "hpi_3", "hpi_4", "hpi_5"}) {
        const auto it = by_name.find(key);
        if (it != by_name.end() && std::isfinite(it->second[0]) && std::isfinite(it->second[1]) && std::isfinite(it->second[2])) dg.elp.push_back(it->second);
      }
      const std::size_t n_mrk = dg.elp.size() >= 3 ? dg.elp.size() - 3 : 0;
      c.p = cr.offset;
      dg.marker_coils.assign(n_mrk, {0, 0, 0});
      dg.marker_done.assign(n_mrk, true);
      for (std::int32_t i = 0; i < cr.count; ++i) {
        const std::int32_t done = c.i32();
        c.skip(16 * 8 + 16 * 8);
        const std::int32_t marker_count = c.i32();
        if (!done) continue;
        if (marker_count < static_cast<std::int32_t>(n_mrk)) throw format_error("KIT: fewer marker coils in the coregistration than HPI coils digitized");
        for (std::size_t m = 0; m < n_mrk; ++m) {
          c.i32(); c.i32(); c.i32();
          dg.marker_done[m] = c.i32() != 0;
          c.skip(3 * 8);
          dg.marker_coils[m] = {c.f64(), c.f64(), c.f64()};
        }
        c.skip(256);
      }
      I.digitization = std::move(dg);
    }
  }
  // conversion factors
  const double ad_to_volt = I.adc_range / std::pow(2.0, I.adc_stored);
  const double ad_to_tesla_base = ad_to_volt / I.amp_gain;
  std::size_t exg_i = 0;
  for (const KitChannel& ch : I.channels) {
    double f = is_meg(ch.type) ? ad_to_tesla_base * ch.gain : ad_to_volt;
    if (ch.type == channel_type::kEeg || ch.type == channel_type::kEcg) {
      double g = exg_gains[exg_i++] / std::pow(2.0, I.adc_stored - 14);
      f = g == 0.0 ? ad_to_volt : g;
    }
    I.conv_factor.push_back(f);
  }
  c.p = dir(kDirIndexRawData).offset;
  r.data_offset_ = dir(kDirIndexRawData).offset;
  const std::uint64_t need = r.data_offset_ + static_cast<std::uint64_t>(I.n_samples) * static_cast<std::uint64_t>(I.nchan) * r.bytes_per_sample_;
  if (need > file->size()) throw format_error("KIT: the file is truncated: " + std::to_string(I.n_samples) + " samples of " + std::to_string(I.nchan) + " channels need " + std::to_string(need) + " bytes, the file has " + std::to_string(file->size()));

  // the common header
  RecordingHeader& h = r.header_;
  h.format = "kit";
  h.sampling_rate = I.sfreq;
  h.n_samples = static_cast<std::uint64_t>(I.n_samples);
  h.start_time = stamp_to_iso(I.create_time);
  h.position_unit = "m";
  std::map<std::string, int> counts;
  int idx = 0;
  for (const KitChannel& kc : I.channels) {
    ++idx;
    Channel ch;
    ch.sampling_rate = I.sfreq;
    ch.gain = I.conv_factor[static_cast<std::size_t>(idx - 1)];
    char num[16];
    if (is_meg(kc.type)) {
      std::snprintf(num, sizeof num, "%03d", idx);
      ch.name = kc.name.empty() ? std::string("MEG ") + num : kc.name;
      ch.kind = is_reference(kc.type) ? "meg_ref" : "meg";
      ch.unit = ch.file_unit = "T";
      ch.location = std::array<double, 3>{kc.loc[0], kc.loc[1], kc.loc[2]};
      const double th = kc.loc[3] * 3.14159265358979323846 / 180.0, ph = kc.loc[4] * 3.14159265358979323846 / 180.0;
      std::array<double, 3> z{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
      const double n = std::sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
      for (double& x : z) x /= n;
      ch.orientation = z;
      if (kc.type == channel_type::kMagnetometerReference) ch.coil_type = "KIT_REF_MAG";
      else if (kc.type == channel_type::kAxialGradiometer) ch.coil_type = "KIT_GRAD";
    } else {
      const char* label = kc.type == channel_type::kTrigger ? "TRIGGER" : kc.type == channel_type::kEeg ? "EEG" : kc.type == channel_type::kEcg ? "ECG" : "MISC";
      const int k = ++counts[label];
      std::snprintf(num, sizeof num, "%03d", k);
      ch.name = (kc.name.empty() || kc.name == "EEG") ? std::string(label) + " " + num : kc.name;
      ch.kind = kc.type == channel_type::kEeg ? "eeg" : kc.type == channel_type::kEcg ? "ecg" : "misc";
      ch.unit = ch.file_unit = "V";
      if (kc.type == channel_type::kEeg) ch.coil_type = "EEG";
    }
    h.channels.push_back(std::move(ch));
  }
  auto put = [&](const char* k, const std::string& v) { h.info[k] = v; };
  put("version", "V" + std::to_string(I.version) + "R" + (I.revision < 100 ? (I.revision < 10 ? "00" : "0") : "") + std::to_string(I.revision));
  put("system_id", std::to_string(I.system_id)); put("system_name", I.system_name); put("model_name", I.model_name); put("comment", I.comment);
  put("dewar_style", std::to_string(I.dewar_style)); put("fll_type", std::to_string(I.fll_type)); put("trigger_type", std::to_string(I.trigger_type));
  put("adboard_type", std::to_string(I.adboard_type)); put("adc_range", std::to_string(I.adc_range)); put("adc_stored", std::to_string(I.adc_stored));
  put("adc_allocated", std::to_string(I.adc_allocated)); put("amp_gain", std::to_string(I.amp_gain)); put("highpass", std::to_string(I.highpass));
  put("lowpass", std::to_string(I.lowpass)); put("notch", std::to_string(I.notch));
  return rd;
}

std::unique_ptr<Reader> open_file(const std::string& path) { return open(std::make_shared<FileSource>(path)); }

void Reader::read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const {
  check_range(s0, n, c0, c1);
  const std::size_t nchan = static_cast<std::size_t>(info_.nchan);
  std::vector<unsigned char> buf(static_cast<std::size_t>(n) * nchan * bytes_per_sample_);
  if (!buf.empty()) src_->read(data_offset_ + s0 * nchan * bytes_per_sample_, buf.size(), buf.data());
  for (std::size_t c = c0; c < c1; ++c) {
    const double f = info_.conv_factor[c];
    float* dst = out + (c - c0) * n;
    for (std::uint64_t i = 0; i < n; ++i) {
      const unsigned char* p = &buf[(static_cast<std::size_t>(i) * nchan + c) * bytes_per_sample_];
      const double v = bytes_per_sample_ == 2 ? static_cast<double>(load_i16(p, true)) : static_cast<double>(load_i32(p, true));
      dst[i] = static_cast<float>(v * f);
    }
  }
}

}  // namespace nxr::formats::kit
